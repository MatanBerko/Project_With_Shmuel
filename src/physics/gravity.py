"""
External gravity (Guo+20) and gas self-gravity, in two modes.

g_ext ported verbatim from (all three agree exactly):
  fig2_histograms/computing_data.py:70-73 g_ext_cgs()
  fig3_slices/compute_data.py:72-76 g_ext_cgs()
  fig4_vertical_profiles/compute_data.py:86-89 g_ext_cgs()

    g_ext(z) = 2*pi*G * [Sigma_star*(1 - exp(-|z|/z_h)) + 2*rho_dm*|z|]

Gas self-gravity is NEW (not in any reference script). Two modes
(SELF_GRAVITY_MODE in src.conventions):

  "per_column" (Step 1 original): g_gas(z) = 2*pi*G*Sigma_gas(|z'|<|z|)
    computed independently for EACH sightline from its own density
    profile. A single compact dense cloud sources its own large local
    g_gas -- found in Phase B to inflate mass-weighted alpha by x1.9-2.6
    near the midplane, since mass-weighting already emphasizes exactly
    those clumps.

  "footprint_mean" (Step 1b, NEW DEFAULT): rho_bar(z) = the mean, over the
    +-500 pc square footprint, of the SAME density used as the pressure
    weight (per variant, per SELF_GRAVITY_DENSITY), NaN cells excluded
    from the mean (their per-z fraction is reported, not silently
    dropped). Sigma_bar(|z'|<|z|) is then the enclosed column of that
    single, horizontally-uniform profile (trapezoid, real z grid), and
    g_gas(z) = 2*pi*G*Sigma_bar(|z'|<|z|) is applied identically to every
    column. This matches the physical picture of Guo+20's external field:
    a smooth, horizontally uniform disk, not a per-sightline slab.

Both modes reuse the same 2*pi*G prefactor as g_ext's stellar term, since
Sigma_star*(1-exp(-|z|/z_h)) is itself "stellar surface density enclosed
within |z|" for an exponential disk -- structurally identical to
Sigma_gas(|z'|<|z|) or Sigma_bar(|z'|<|z|).
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

from src.conventions import (
    G_PC_MSUN_KMS,
    KM2S2_PER_PC_TO_CGS,
    M_H,
    M_SUN_G,
    MU,
    PC_CM,
    RHO_DM_MSUN_PC3,
    SELF_GRAVITY_MODE_FOOTPRINT_MEAN,
    SELF_GRAVITY_MODE_OFF,
    SELF_GRAVITY_MODE_PER_COLUMN,
    SIGMA_STAR_MSUN_PC2,
    Z_H_PC,
)

# cm/s^2 per (Msun/pc^2): common prefactor for g_ext's stellar term and both g_gas modes.
GRAV_PREFACTOR_CGS = 2.0 * np.pi * G_PC_MSUN_KMS * KM2S2_PER_PC_TO_CGS

# g/cm^3 per (cm^-3): converts n_H to mass density.
RHO_PER_N = MU * M_H

# Msun/pc^3 per (cm^-3): converts n_H directly to mass density in Msun/pc^3,
# for use in the self-gravity column integral (kept in pc/Msun units
# throughout, matching Sigma_star's units, to reuse GRAV_PREFACTOR_CGS).
RHO_MSUN_PC3_PER_N = MU * M_H * (PC_CM ** 3) / M_SUN_G


def g_ext_cgs(z_pc: np.ndarray) -> np.ndarray:
    """External (stellar disk + dark matter) gravity, cm/s^2."""
    z_abs = np.abs(z_pc)
    s = SIGMA_STAR_MSUN_PC2 * (1.0 - np.exp(-z_abs / Z_H_PC)) + 2.0 * RHO_DM_MSUN_PC3 * z_abs
    return GRAV_PREFACTOR_CGS * s


def _enclosed_column_from_profile(z_pc: np.ndarray, rho_msun_pc3: np.ndarray) -> np.ndarray:
    """Sigma(|z'| < |z|) [Msun/pc^2] for every z in z_pc, from an ALREADY
    mass-density profile in Msun/pc^3 (shape (Nz,) or (Nz, ...), aligned
    to z_pc along axis 0). Shared machinery for both self-gravity modes.
    """
    order = np.argsort(z_pc)
    z_sorted = z_pc[order]
    rho_sorted = rho_msun_pc3[order]
    cum_from_bottom = cumulative_trapezoid(rho_sorted, x=z_sorted, axis=0, initial=0.0)

    idx_pos = np.clip(np.searchsorted(z_sorted, np.abs(z_pc)), 0, len(z_sorted) - 1)
    idx_neg = np.clip(np.searchsorted(z_sorted, -np.abs(z_pc)), 0, len(z_sorted) - 1)

    return cum_from_bottom[idx_pos] - cum_from_bottom[idx_neg]


def sigma_gas_enclosed(z_pc: np.ndarray, n_cm3: np.ndarray) -> np.ndarray:
    """Sigma_gas(|z'| < |z|) [Msun/pc^2] for every z in z_pc (any sign order).

    z_pc is 1D (Nz,), both signs present. n_cm3 is (Nz,) for a single
    sightline or (Nz, ...) for a full cube (per-column self-gravity is
    inherently a per-column quantity -- each XY column has its own
    density profile, so this vectorizes over any trailing XY dimensions).
    Integrated against the real z_pc values via trapezoid (never a
    hardcoded voxel size). Returns an array shaped like n_cm3: the gas
    column strictly between -|z| and +|z|, so it is 0 at z=0 and equals
    the full column Sigma_gas at the domain edges.
    """
    return _enclosed_column_from_profile(z_pc, n_cm3 * RHO_MSUN_PC3_PER_N)


def g_gas_per_column_cgs(z_pc: np.ndarray, n_cm3_for_gravity: np.ndarray) -> np.ndarray:
    """Per-column gas self-gravity g_gas(z) = 2*pi*G*Sigma_gas(|z'|<|z|),
    cm/s^2, independently for every sightline (Step 1 original behavior,
    unchanged). Shaped like n_cm3_for_gravity.
    """
    return GRAV_PREFACTOR_CGS * sigma_gas_enclosed(z_pc, n_cm3_for_gravity)


# Backwards-compatible alias (Step 1 name).
g_gas_cgs = g_gas_per_column_cgs


def footprint_mean_density(n_cm3_cube: np.ndarray, footprint_mask: np.ndarray):
    """Horizontal mean of a density cube over the footprint, per z-plane.

    n_cm3_cube: (Nz, Ny, Nx). footprint_mask: (Ny, Nx) boolean.
    NaN cells within the footprint are excluded from the mean (not
    treated as zero). Returns (rho_bar (Nz,), nan_fraction (Nz,)) where
    nan_fraction is the fraction of in-footprint cells that were NaN at
    each z (0 if the density has no NaNs there).
    """
    footprint_cells = n_cm3_cube[:, footprint_mask]  # (Nz, Nfootprint)
    finite = np.isfinite(footprint_cells)
    n_footprint = footprint_cells.shape[1]
    nan_fraction = 1.0 - finite.sum(axis=1) / n_footprint
    with np.errstate(invalid="ignore"):
        rho_bar = np.nanmean(footprint_cells, axis=1)
    return rho_bar, nan_fraction


def g_gas_footprint_mean_cgs(z_pc: np.ndarray, n_cm3_cube: np.ndarray, footprint_mask: np.ndarray):
    """Footprint-averaged gas self-gravity (Step 1b default): a single
    g_gas(z) [cm/s^2], shape (Nz,), applied identically to every column.

    Returns (g_gas_1d (Nz,), nan_fraction (Nz,)).
    """
    rho_bar_n, nan_fraction = footprint_mean_density(n_cm3_cube, footprint_mask)
    rho_bar_msun_pc3 = rho_bar_n * RHO_MSUN_PC3_PER_N
    sigma_bar_enclosed = _enclosed_column_from_profile(z_pc, rho_bar_msun_pc3)
    return GRAV_PREFACTOR_CGS * sigma_bar_enclosed, nan_fraction


def g_total_cgs(z_pc: np.ndarray, n_cm3_for_gravity, mode: str, footprint_mask: np.ndarray | None = None):
    """g_ext, optionally plus gas self-gravity in the given mode.

    mode: one of SELF_GRAVITY_MODE_OFF / _PER_COLUMN / _FOOTPRINT_MEAN
    (src.conventions). "footprint_mean" requires footprint_mask.

    Returns (g_total_cgs, nan_fraction_or_None):
      - "off": shape (Nz,), nan_fraction None.
      - "per_column": shaped like n_cm3_for_gravity (e.g. (Nz,Ny,Nx)),
        nan_fraction None (per-column mode does not track this).
      - "footprint_mean": shape (Nz,) (identical for every column),
        nan_fraction shape (Nz,).
    """
    g_ext = g_ext_cgs(z_pc)

    if mode == SELF_GRAVITY_MODE_OFF:
        return g_ext, None

    if mode == SELF_GRAVITY_MODE_PER_COLUMN:
        if n_cm3_for_gravity is None:
            raise ValueError(f"mode={mode!r} requires n_cm3_for_gravity.")
        g_gas = g_gas_per_column_cgs(z_pc, n_cm3_for_gravity)
        g_ext_b = g_ext.reshape((g_ext.shape[0],) + (1,) * (g_gas.ndim - 1))
        return g_ext_b + g_gas, None

    if mode == SELF_GRAVITY_MODE_FOOTPRINT_MEAN:
        if n_cm3_for_gravity is None or footprint_mask is None:
            raise ValueError(f"mode={mode!r} requires n_cm3_for_gravity and footprint_mask.")
        g_gas_1d, nan_fraction = g_gas_footprint_mean_cgs(z_pc, n_cm3_for_gravity, footprint_mask)
        return g_ext + g_gas_1d, nan_fraction

    raise ValueError(f"Unknown self-gravity mode: {mode!r}")
