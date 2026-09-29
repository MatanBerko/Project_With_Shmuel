"""
Uniform slab: n_H = n0 constant over z in [-Zmax, Zmax]. Sigma_gas and
P_tot (external gravity only) compared against closed-form analytic
expressions, on both a 1 pc and a 2 pc grid, to make sure no hardcoded
voxel size sneaks into either integral (the exact bug this whole rebuild
exists to prevent).
"""

import numpy as np

from src.conventions import G_PC_MSUN_KMS, K_B, KM2S2_PER_PC_TO_CGS, M_H, MU, \
    RHO_DM_MSUN_PC3, SIGMA_STAR_MSUN_PC2, Z_H_PC
from src.physics.derived import sigma_gas_map
from src.physics.gravity import g_ext_cgs
from src.physics.hydrostatic import p_tot_kb_full_column

N0_CM3 = 0.5
ZMAX_PC = 400.0


def _make_grid(dz):
    return np.arange(-ZMAX_PC, ZMAX_PC + dz / 2, dz)


def _closed_form_sigma_gas_msun_pc2(n0, zmax):
    PC_CM = 3.085677581491367e18
    M_SUN_G = 1.989e33
    N_H_column_pc_cm3 = n0 * (2 * zmax)
    N_H_cm2 = N_H_column_pc_cm3 * PC_CM
    Sigma_g_cm2 = MU * M_H * N_H_cm2
    return Sigma_g_cm2 * (PC_CM ** 2) / M_SUN_G


def _closed_form_ptot_kb(z, n0, zmax):
    """P_tot(z)/k_B for a uniform slab under Guo+20 external gravity only.

    Derived analytically from g_ext(z') = 2piG[Sigma_star(1-e^-z'/zh) + 2 rho_dm z'],
    integrated from z to zmax (z >= 0) with dz' converted pc -> cm:
      integral = 2piG * [ Sigma_star*((zmax-z) - zh*(e^-z/zh - e^-zmax/zh))
                           + rho_dm*(zmax**2 - z**2) ] * PC_CM
    """
    PC_CM = 3.085677581491367e18
    prefactor = 2.0 * np.pi * G_PC_MSUN_KMS * KM2S2_PER_PC_TO_CGS
    stellar = SIGMA_STAR_MSUN_PC2 * (
        (zmax - z) - Z_H_PC * (np.exp(-z / Z_H_PC) - np.exp(-zmax / Z_H_PC))
    )
    dm = RHO_DM_MSUN_PC3 * (zmax ** 2 - z ** 2)
    g_integral_cgs = prefactor * (stellar + dm) * PC_CM
    rho0 = MU * M_H * n0
    return rho0 * g_integral_cgs / K_B


def test_sigma_gas_uniform_slab_1pc_and_2pc():
    for dz in (1.0, 2.0):
        z_pc = _make_grid(dz)
        n_cm3 = np.full_like(z_pc, N0_CM3)
        sigma = sigma_gas_map(z_pc, n_cm3)
        expected = _closed_form_sigma_gas_msun_pc2(N0_CM3, z_pc.max())
        assert np.isclose(sigma, expected, rtol=1e-8), (dz, sigma, expected)


def test_ptot_uniform_slab_no_self_gravity_1pc_and_2pc():
    for dz in (1.0, 2.0):
        z_pc = _make_grid(dz)
        n_cm3 = np.full_like(z_pc, N0_CM3)
        rho_g_cm3 = MU * M_H * n_cm3
        g_cgs = g_ext_cgs(z_pc)

        ptot = p_tot_kb_full_column(z_pc, rho_g_cm3, g_cgs)

        for z_test in (0.0, 100.0, 300.0, z_pc.max()):
            idx = np.argmin(np.abs(z_pc - z_test))
            expected = _closed_form_ptot_kb(z_pc[idx], N0_CM3, z_pc.max())
            rel_err = abs(ptot[idx] - expected) / max(abs(expected), 1e-30)
            assert rel_err < 1e-3, (dz, z_test, ptot[idx], expected, rel_err)

        # boundary: P=0 at the top and bottom of the domain
        assert abs(ptot[np.argmax(z_pc)]) < 1e-6 * max(abs(ptot).max(), 1.0)
        assert abs(ptot[np.argmin(z_pc)]) < 1e-6 * max(abs(ptot).max(), 1.0)
