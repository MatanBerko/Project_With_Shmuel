"""
The balanced midplane (Step 1i): where the area-averaged weight above equals
the weight below.

Why: P_tot is integrated top-down for the upper half-column and bottom-up for
the lower one, with g changing sign at the plane where they meet. With that
plane at the Sun (z = 0, MIDPLANE_MODE = "sun") and ~12% more gas below the Sun
than above, the area-averaged P_tot jumps at z = 0 (7245 vs 8571 K cm^-3). The
physical midplane is where the two weights balance:

    int_{z_mid}^{top} rho_bar g dz'  =  int_{bottom}^{z_mid} rho_bar |g| dz'

with g CENTRED ON z_mid itself:
  * g_ext (Guo+20 stars + DM) at |z - z_mid|  (gravity.g_ext_cgs(z - z_mid));
  * the footprint-mean gas self-gravity from the column of rho_bar within
    |z' - z_mid| < |z - z_mid| (left out for self-gravity "off").
rho_bar is the footprint-mean density -- every cell, HIM included, the same
density as the hydrostatic weight -- on the SOURCE cube's float64 z grid.

Sub-grid exactness: z_mid is not on the 2 pc grid. The density is treated as
piecewise linear between grid planes (exactly what the trapezoid integrals
assume), so
  * the enclosed column at an off-grid end point is the cumulative integral
    to the grid plane below plus the exact trapezoid to that point;
  * the branch values at z_mid are integrals to an inserted node at z_mid with
    linearly interpolated rho_bar (g = 0 there, since g is centred on it).
The two area-averaged branches therefore meet exactly at z_mid (to the
root-finder's tolerance).

g is horizontally uniform in both fiducial self-gravity settings, so the same
z_mid and the same g apply to every column; per-column P_tot then uses
hydrostatic.p_tot_kb_full_column(..., z_split=z_mid). Individual columns are
NOT balanced, and are not forced to be.

"sun" mode, and "balanced" mode whenever z_mid comes out exactly 0 (a profile
symmetric about the Sun), go through the original Sun-centred functions
unchanged, so the old results stay bit-for-bit reproducible.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

from src.conventions import (
    MIDPLANE_BRACKET_PC,
    MIDPLANE_MODE_BALANCED,
    MIDPLANE_MODE_SUN,
    SELF_GRAVITY_MODE_FOOTPRINT_MEAN,
    SELF_GRAVITY_MODE_OFF,
)
from src.physics import gravity, hydrostatic

ZMID_XTOL_PC = 1e-9


def _cumulative_at(z, cum, rho, x):
    """Integral of the piecewise-linear rho from z[0] to each x (clipped to the grid)."""
    x = np.clip(np.asarray(x, dtype=float), z[0], z[-1])
    j = np.clip(np.searchsorted(z, x, side="right") - 1, 0, z.size - 2)
    rho_x = rho[j] + (rho[j + 1] - rho[j]) * (x - z[j]) / (z[j + 1] - z[j])
    return cum[j] + 0.5 * (rho[j] + rho_x) * (x - z[j])


def enclosed_column_about(z_pc, rho_msun_pc3, z_c):
    """Sigma(|z' - z_c| < |z - z_c|) [Msun/pc^2] at every grid z, exact for a
    piecewise-linear density. z_pc ascending, rho 1-D."""
    z = np.asarray(z_pc, dtype=float)
    rho = np.asarray(rho_msun_pc3, dtype=float)
    cum = cumulative_trapezoid(rho, x=z, initial=0.0)
    s = np.abs(z - z_c)
    return _cumulative_at(z, cum, rho, z_c + s) - _cumulative_at(z, cum, rho, z_c - s)


def centred_g_cgs(z_pc, n_bar_cm3, z_c, self_gravity=True):
    """|g| [cm/s^2] at every grid z for gravity centred on z_c.

    g_ext at |z - z_c|, plus (self_gravity) 2 pi G times the column of the
    footprint-mean density within |z' - z_c| < |z - z_c|."""
    z = np.asarray(z_pc, dtype=float)
    g = gravity.g_ext_cgs(z - z_c)
    if self_gravity:
        rho = np.asarray(n_bar_cm3, dtype=float) * gravity.RHO_MSUN_PC3_PER_N
        g = g + gravity.GRAV_PREFACTOR_CGS * enclosed_column_about(z, rho, z_c)
    return g


def branch_pressures(z_pc, rho_g_cm3, g_cgs, z_split, g_at_split=0.0):
    """(P top-down, P bottom-up) [K cm^-3] AT z_split, for a 1-D profile.

    z_split is inserted as a node (rho linearly interpolated, g = g_at_split,
    which is 0 for gravity centred on z_split) unless it is already a grid
    plane. Each branch is hydrostatic.integrate_ptot_halfcolumn from its own
    domain edge (P = 0) to z_split."""
    z = np.asarray(z_pc, dtype=float)
    rho = np.asarray(rho_g_cm3, dtype=float)
    g = np.asarray(g_cgs, dtype=float)
    on_grid = np.isclose(z, z_split, rtol=0.0, atol=1e-12)
    if on_grid.any():
        k = int(np.argmax(on_grid))
    else:
        k = int(np.searchsorted(z, z_split))
        z = np.insert(z, k, z_split)
        rho = np.insert(rho, k, np.interp(z_split, z_pc, rho_g_cm3))
        g = np.insert(g, k, g_at_split)
    top = hydrostatic.integrate_ptot_halfcolumn(z[k:][::-1], rho[k:][::-1], g[k:][::-1])[-1]
    bot = hydrostatic.integrate_ptot_halfcolumn(z[:k + 1], rho[:k + 1], g[:k + 1])[-1]
    return float(top), float(bot)


def weight_imbalance(z_c, z_pc, n_bar_cm3, self_gravity=True):
    """Top-down minus bottom-up area-averaged weight [K cm^-3] for gravity
    centred on z_c. Zero at the balanced midplane."""
    rho = gravity.RHO_PER_N * np.asarray(n_bar_cm3, dtype=float)
    g = centred_g_cgs(z_pc, n_bar_cm3, z_c, self_gravity)
    top, bot = branch_pressures(z_pc, rho, g, z_c)
    return top - bot


def find_balanced_midplane(z_pc, n_bar_cm3, self_gravity=True,
                           bracket_pc=MIDPLANE_BRACKET_PC, xtol_pc=ZMID_XTOL_PC):
    """z_mid [pc]: brentq root of weight_imbalance in [-bracket, +bracket].

    Returns exactly 0.0 when the weights already balance at the Sun (a
    symmetric profile), so that case reproduces "sun" mode bit for bit."""
    f0 = weight_imbalance(0.0, z_pc, n_bar_cm3, self_gravity)
    if f0 == 0.0:
        return 0.0
    return float(brentq(weight_imbalance, -bracket_pc, bracket_pc,
                        args=(z_pc, n_bar_cm3, self_gravity), xtol=xtol_pc))


def column_balance_height(z_pc, n_bar_cm3):
    """z [pc] with equal gas column above and below (piecewise-linear exact)."""
    z = np.asarray(z_pc, dtype=float)
    rho = np.asarray(n_bar_cm3, dtype=float) * gravity.RHO_MSUN_PC3_PER_N
    cum = cumulative_trapezoid(rho, x=z, initial=0.0)
    half = 0.5 * cum[-1]
    return float(brentq(lambda x: _cumulative_at(z, cum, rho, x) - half, z[0], z[-1],
                        xtol=1e-12))


def midplane_gravity(z_pc, n_bar_cm3, self_gravity_mode, midplane_mode,
                     n_cube_for_sun=None, footprint_mask=None):
    """(z_mid, g (Nz,), nan_fraction or None) for one self-gravity setting.

    "sun", or "balanced" with z_mid == 0: z_mid = 0 and g from the original
    gravity.g_total_cgs (with the cube for "footprint_mean", exactly as the
    build always called it). Otherwise z_mid from find_balanced_midplane and g
    centred on it. Only the horizontally uniform modes ("off",
    "footprint_mean") are supported in "balanced" mode.
    """
    if self_gravity_mode not in (SELF_GRAVITY_MODE_OFF, SELF_GRAVITY_MODE_FOOTPRINT_MEAN):
        raise NotImplementedError(
            f"balanced midplane needs a horizontally uniform g; got {self_gravity_mode!r}")
    sg = self_gravity_mode == SELF_GRAVITY_MODE_FOOTPRINT_MEAN
    if midplane_mode == MIDPLANE_MODE_SUN:
        z_mid = 0.0
    elif midplane_mode == MIDPLANE_MODE_BALANCED:
        z_mid = find_balanced_midplane(z_pc, n_bar_cm3, self_gravity=sg)
    else:
        raise ValueError(f"Unknown midplane mode: {midplane_mode!r}")
    if z_mid == 0.0:
        cube = n_cube_for_sun if sg else None
        if sg and cube is None:
            cube = np.asarray(n_bar_cm3, dtype=float)[:, None, None]
            footprint_mask = np.ones((1, 1), dtype=bool)
        g, nan_fraction = gravity.g_total_cgs(z_pc, cube, mode=self_gravity_mode,
                                              footprint_mask=footprint_mask)
        return 0.0, np.asarray(g, dtype=float), nan_fraction
    return z_mid, centred_g_cgs(z_pc, n_bar_cm3, z_mid, self_gravity=sg), None


def slab_center(z_center_pc, z_mid_pc, midplane_mode):
    """The midplane slab follows z_mid in "balanced" mode; other slabs do not."""
    if midplane_mode == MIDPLANE_MODE_BALANCED and z_center_pc == 0.0:
        return float(z_mid_pc)
    return float(z_center_pc)
