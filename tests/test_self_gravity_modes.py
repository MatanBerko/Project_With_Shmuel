"""
Step 1b: SELF_GRAVITY_MODE tests.

1. Horizontally uniform cube: per_column and footprint_mean must agree to
   numerical precision (when density doesn't vary over XY, per-column
   self-gravity IS the footprint mean, trivially).
2. Single dense clump: footprint_mean g_gas, at |z| above the clump, must
   equal the analytic 2*pi*G*(clump mass / footprint area) -- the whole
   point of footprint-averaging is that a compact clump's mass gets
   diluted over the full footprint area, unlike per_column.
"""

import numpy as np

from src.conventions import RHO_DM_MSUN_PC3, SIGMA_STAR_MSUN_PC2, Z_H_PC
from src.physics.gravity import (
    GRAV_PREFACTOR_CGS,
    RHO_MSUN_PC3_PER_N,
    g_gas_footprint_mean_cgs,
    g_gas_per_column_cgs,
    g_total_cgs,
)
from src.conventions import SELF_GRAVITY_MODE_FOOTPRINT_MEAN, SELF_GRAVITY_MODE_PER_COLUMN


def test_uniform_cube_per_column_and_footprint_mean_agree():
    z_pc = np.arange(-300.0, 300.0 + 1.0, 2.0)
    ny, nx = 11, 13
    n0 = 0.4
    h = 120.0
    n_profile = n0 * np.exp(-np.abs(z_pc) / h)  # (Nz,)
    n_cube = np.broadcast_to(n_profile[:, None, None], (len(z_pc), ny, nx)).copy()
    footprint = np.ones((ny, nx), dtype=bool)

    g_col = g_gas_per_column_cgs(z_pc, n_cube)  # (Nz, ny, nx)
    g_mean, nan_frac = g_gas_footprint_mean_cgs(z_pc, n_cube, footprint)  # (Nz,)

    assert np.allclose(nan_frac, 0.0)
    # every column of g_col must match g_mean exactly (uniform density -> no dilution)
    for iy in range(ny):
        for ix in range(nx):
            assert np.allclose(g_col[:, iy, ix], g_mean, rtol=1e-10, atol=0.0)

    g_tot_col, _ = g_total_cgs(z_pc, n_cube, mode=SELF_GRAVITY_MODE_PER_COLUMN)
    g_tot_mean, _ = g_total_cgs(z_pc, n_cube, mode=SELF_GRAVITY_MODE_FOOTPRINT_MEAN, footprint_mask=footprint)
    assert np.allclose(g_tot_col, np.broadcast_to(g_tot_mean[:, None, None], g_tot_col.shape), rtol=1e-10)


def test_footprint_mean_single_clump_matches_analytic_dilution():
    dz = 2.0
    z_pc = np.arange(-100.0, 100.0 + dz / 2, dz)
    dx, dy = 2.0, 2.0
    x_pc = np.arange(-50.0, 50.0 + dx / 2, dx)
    y_pc = np.arange(-50.0, 50.0 + dy / 2, dy)
    nz, ny, nx = len(z_pc), len(y_pc), len(x_pc)

    n_cube = np.zeros((nz, ny, nx))

    # Clump: a contiguous z-plateau (exact-0 immediately outside, so
    # trapz integrates it as exactly N_CLUMP_Z * dz -- see module docstring
    # derivation) over a small block of XY cells. Offset from z=0 (not
    # straddling the midplane) so there is a genuine "not yet enclosed"
    # regime at |z| below the clump's near edge -- g_gas(z) encloses
    # [-|z|, +|z|] symmetrically, so a clump straddling z=0 would be
    # partially enclosed at every nonzero |z|.
    iz_center = nz // 2 + 15
    N_CLUMP_Z = 5
    iz_lo = iz_center - N_CLUMP_Z // 2
    iz_hi = iz_lo + N_CLUMP_Z  # exclusive
    assert iz_lo > 0 and iz_hi < nz - 1  # guarantee exact-0 neighbors both sides

    iy_lo, iy_hi = 4, 7  # 3 cells
    ix_lo, ix_hi = 5, 9  # 4 cells
    K_CLUMP = (iy_hi - iy_lo) * (ix_hi - ix_lo)
    K_TOTAL = ny * nx

    N_CLUMP_CM3 = 50.0
    n_cube[iz_lo:iz_hi, iy_lo:iy_hi, ix_lo:ix_hi] = N_CLUMP_CM3

    footprint = np.ones((ny, nx), dtype=bool)

    rho_bar, nan_frac = None, None
    from src.physics.gravity import footprint_mean_density
    rho_bar, nan_frac = footprint_mean_density(n_cube, footprint)
    assert np.allclose(nan_frac, 0.0)
    # rho_bar at the clump's z-planes should be the diluted mean
    expected_rho_bar_at_clump = N_CLUMP_CM3 * K_CLUMP / K_TOTAL
    assert np.allclose(rho_bar[iz_lo:iz_hi], expected_rho_bar_at_clump)
    assert np.allclose(rho_bar[:iz_lo], 0.0)
    assert np.allclose(rho_bar[iz_hi:], 0.0)

    g_mean, _ = g_gas_footprint_mean_cgs(z_pc, n_cube, footprint)

    # Analytic: Sigma_bar_total = (clump mass) / (footprint area), derived
    # exactly (not approximately) from the trapezoidal-plateau identity:
    # trapz of N equal points flanked by exact 0 = N * height * dz.
    clump_mass_msun = N_CLUMP_CM3 * RHO_MSUN_PC3_PER_N * K_CLUMP * dx * dy * N_CLUMP_Z * dz
    footprint_area_pc2 = K_TOTAL * dx * dy
    expected_sigma_bar_total = clump_mass_msun / footprint_area_pc2
    expected_g_above_clump = GRAV_PREFACTOR_CGS * expected_sigma_bar_total

    # at any |z| beyond the clump's z-extent, the full clump is enclosed
    z_above_clump = z_pc[iz_hi] + 10.0
    i_test = np.argmin(np.abs(z_pc - z_above_clump))
    assert abs(z_pc[i_test]) > abs(z_pc[iz_hi - 1])

    assert np.isclose(g_mean[i_test], expected_g_above_clump, rtol=1e-8)
    # and symmetric on the negative side
    i_test_neg = np.argmin(np.abs(z_pc + z_above_clump))
    assert np.isclose(g_mean[i_test_neg], expected_g_above_clump, rtol=1e-8)

    # sanity: g_gas = 0 well below the clump (nothing enclosed yet)
    i_below = np.argmin(np.abs(z_pc - (z_pc[iz_lo] - 10.0)))
    assert abs(g_mean[i_below]) < 1e-6 * expected_g_above_clump
