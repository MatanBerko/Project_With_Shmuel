"""
External gravity (Guo+20) and NEW gas self-gravity.

g_ext ported verbatim from (all three agree exactly):
  fig2_histograms/computing_data.py:70-73 g_ext_cgs()
  fig3_slices/compute_data.py:72-76 g_ext_cgs()
  fig4_vertical_profiles/compute_data.py:86-89 g_ext_cgs()

    g_ext(z) = 2*pi*G * [Sigma_star*(1 - exp(-|z|/z_h)) + 2*rho_dm*|z|]

g_gas is NEW (not in any reference script): the gas disk's own self-gravity,
g_gas(z) = 2*pi*G * Sigma_gas(|z'| < |z|), where Sigma_gas(|z'| < |z|) is
the real column of gas mass between -|z| and +|z| along that sightline
(trapezoidal, real z spacing). Because the Guo+20 stellar term
Sigma_star*(1-exp(-|z|/z_h)) is itself "stellar surface density enclosed
within |z|" for an exponential disk, g_gas uses the exact same
2*pi*G prefactor, just applied to the gas column instead of an analytic
stellar profile.
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
    SIGMA_STAR_MSUN_PC2,
    Z_H_PC,
)

# cm/s^2 per (Msun/pc^2): common prefactor for both g_ext's stellar term and g_gas.
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


def sigma_gas_enclosed(z_pc: np.ndarray, n_cm3: np.ndarray) -> np.ndarray:
    """Sigma_gas(|z'| < |z|) [Msun/pc^2] for every z in z_pc (any sign order).

    z_pc is 1D (Nz,), both signs present. n_cm3 is (Nz,) for a single
    sightline or (Nz, ...) for a full cube (self-gravity is inherently a
    per-column quantity -- each XY column has its own density profile, so
    this vectorizes over any trailing XY dimensions). Integrated against
    the real z_pc values via trapezoid (never a hardcoded voxel size).
    Returns an array shaped like n_cm3: the gas column strictly between
    -|z| and +|z|, so it is 0 at z=0 and equals the full column Sigma_gas
    at the domain edges.
    """
    order = np.argsort(z_pc)
    z_sorted = z_pc[order]
    rho_sorted = n_cm3[order] * RHO_MSUN_PC3_PER_N
    cum_from_bottom = cumulative_trapezoid(rho_sorted, x=z_sorted, axis=0, initial=0.0)

    idx_pos = np.clip(np.searchsorted(z_sorted, np.abs(z_pc)), 0, len(z_sorted) - 1)
    idx_neg = np.clip(np.searchsorted(z_sorted, -np.abs(z_pc)), 0, len(z_sorted) - 1)

    return cum_from_bottom[idx_pos] - cum_from_bottom[idx_neg]


def g_gas_cgs(z_pc: np.ndarray, n_cm3_for_gravity: np.ndarray) -> np.ndarray:
    """Gas self-gravity g_gas(z) = 2*pi*G*Sigma_gas(|z'|<|z|), cm/s^2."""
    return GRAV_PREFACTOR_CGS * sigma_gas_enclosed(z_pc, n_cm3_for_gravity)


def g_total_cgs(z_pc: np.ndarray, n_cm3_for_gravity: np.ndarray | None,
                  include_self_gravity: bool) -> np.ndarray:
    """g_ext, optionally plus g_gas. Returns shape (Nz,) if self-gravity is
    off, or shaped like n_cm3_for_gravity (e.g. (Nz, Ny, Nx)) if on, since
    self-gravity is inherently per-column.
    """
    g = g_ext_cgs(z_pc)
    if include_self_gravity:
        if n_cm3_for_gravity is None:
            raise ValueError("include_self_gravity=True requires n_cm3_for_gravity.")
        g_gas = g_gas_cgs(z_pc, n_cm3_for_gravity)
        g = g.reshape((g.shape[0],) + (1,) * (g_gas.ndim - 1)) + g_gas
    return g
