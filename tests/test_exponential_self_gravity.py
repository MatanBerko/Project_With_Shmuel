"""
Exponential density layer n(z) = n0*exp(-|z|/h), WITH self-gravity. P_tot
from src.physics is compared against an independently-coded, high-resolution
(0.2 pc) reference integrator that does not call into src.physics at all,
so this checks correctness of the self-gravity nesting, not just internal
self-consistency.
"""

import numpy as np

from src.conventions import G_PC_MSUN_KMS, K_B, KM2S2_PER_PC_TO_CGS, M_H, MU, \
    PC_CM, M_SUN_G, RHO_DM_MSUN_PC3, SIGMA_STAR_MSUN_PC2, Z_H_PC
from src.physics.gravity import g_gas_cgs, g_total_cgs
from src.physics.hydrostatic import p_tot_kb_full_column

N0_CM3 = 1.0
H_PC = 150.0
ZMAX_PC = 400.0

_PREFACTOR = 2.0 * np.pi * G_PC_MSUN_KMS * KM2S2_PER_PC_TO_CGS
_RHO_MSUN_PC3_PER_N = MU * M_H * (PC_CM ** 3) / M_SUN_G


def _independent_reference(dz_fine=0.2):
    """From-scratch reference: fine-grid manual cumulative trapezoid for
    both the self-gravity column and P_tot, entirely independent of
    src.physics.hydrostatic / src.physics.gravity.
    """
    z = np.arange(-ZMAX_PC, ZMAX_PC + dz_fine / 2, dz_fine)
    n = N0_CM3 * np.exp(-np.abs(z) / H_PC)
    rho_msun_pc3 = n * _RHO_MSUN_PC3_PER_N
    rho_g_cm3 = MU * M_H * n

    # cumulative mass column from the bottom boundary, manual trapezoid
    cum_from_bottom = np.zeros_like(z)
    for j in range(1, len(z)):
        cum_from_bottom[j] = cum_from_bottom[j - 1] + 0.5 * (
            rho_msun_pc3[j] + rho_msun_pc3[j - 1]
        ) * (z[j] - z[j - 1])

    sigma_enclosed = np.array([
        cum_from_bottom[np.argmin(np.abs(z - abs(zv)))]
        - cum_from_bottom[np.argmin(np.abs(z + abs(zv)))]
        for zv in z
    ])

    g_ext = _PREFACTOR * (
        SIGMA_STAR_MSUN_PC2 * (1.0 - np.exp(-np.abs(z) / Z_H_PC))
        + 2.0 * RHO_DM_MSUN_PC3 * np.abs(z)
    )
    g_gas = _PREFACTOR * sigma_enclosed
    g_tot = g_ext + g_gas

    integ = rho_g_cm3 * g_tot
    n_z = len(z)
    j_top = n_z - 1
    P_cgs = np.zeros_like(z)
    for j in range(j_top - 1, -1, -1):
        P_cgs[j] = P_cgs[j + 1] + 0.5 * (integ[j] + integ[j + 1]) * (z[j + 1] - z[j]) * PC_CM
    return z, P_cgs / K_B


def test_ptot_with_self_gravity_exponential_layer_vs_independent_integral():
    z_ref, ptot_ref = _independent_reference(dz_fine=0.2)

    dz_pkg = 2.0
    z_pkg = np.arange(-ZMAX_PC, ZMAX_PC + dz_pkg / 2, dz_pkg)
    n_pkg = N0_CM3 * np.exp(-np.abs(z_pkg) / H_PC)
    rho_pkg = MU * M_H * n_pkg
    g_pkg = g_total_cgs(z_pkg, n_pkg, include_self_gravity=True)
    ptot_pkg = p_tot_kb_full_column(z_pkg, rho_pkg, g_pkg)

    for z_test in (0.0, 100.0, 300.0):
        i_ref = np.argmin(np.abs(z_ref - z_test))
        i_pkg = np.argmin(np.abs(z_pkg - z_test))
        expected = ptot_ref[i_ref]
        got = ptot_pkg[i_pkg]
        rel_err = abs(got - expected) / max(abs(expected), 1e-30)
        assert rel_err < 0.02, (z_test, got, expected, rel_err)


def test_self_gravity_increases_ptot_over_ext_only():
    """Sanity: adding self-gravity should never decrease P_tot for a
    positive-density profile (extra inward pull only adds weight)."""
    z = np.arange(-ZMAX_PC, ZMAX_PC + 1.0, 2.0)
    n = N0_CM3 * np.exp(-np.abs(z) / H_PC)
    rho = MU * M_H * n

    g_ext_only = g_total_cgs(z, n, include_self_gravity=False)
    g_with_self = g_total_cgs(z, n, include_self_gravity=True)

    p_ext_only = p_tot_kb_full_column(z, rho, g_ext_only)
    p_with_self = p_tot_kb_full_column(z, rho, g_with_self)

    mid = len(z) // 2
    assert p_with_self[mid] >= p_ext_only[mid] - 1e-6 * max(abs(p_ext_only[mid]), 1.0)
