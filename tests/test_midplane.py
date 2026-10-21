"""
Step 1i: the balanced midplane (src.physics.midplane) and the split-aware
hydrostatic integral.

1. A density symmetric about the Sun gives z_mid = 0 EXACTLY, and "balanced"
   then reproduces "sun" bit for bit (same g, same P_tot).
2. More mass below the Sun gives z_mid < 0, and the area-averaged P_tot is
   continuous there: both branches agree at z_mid.
3. An off-grid z_mid: the branches still agree to < 1e-6, the enclosed
   column is exact for a piecewise-linear density, and grid planes on each
   side take the right branch.
4. "sun" mode is the pre-Step-1i code path: z_split = 0 is the default, and
   midplane_gravity("sun") is gravity.g_total_cgs exactly. The full-pipeline
   counterpart (every cached cube and every RAW numbers_table row reproduced
   exactly) is scripts/validation/midplane_sun_reproduction.py, whose report
   is checked here when present.
"""

from pathlib import Path

import numpy as np
import pytest

from src.conventions import MIDPLANE_MODE, MIDPLANE_MODE_BALANCED
from src.physics import gravity, hydrostatic, midplane

Z = np.arange(-400.0, 400.0 + 1.0, 2.0)          # symmetric, 2 pc, like the cube
RHO_PER_N = gravity.RHO_PER_N


def _gauss(z0=0.0, h=60.0, n0=1.0):
    return n0 * np.exp(-0.5 * ((Z - z0) / h) ** 2) + 0.02


def test_default_mode_is_balanced():
    assert MIDPLANE_MODE == MIDPLANE_MODE_BALANCED


@pytest.mark.parametrize("sg_mode", ["footprint_mean", "off"])
def test_symmetric_profile_gives_zmid_zero_and_reproduces_sun(sg_mode):
    n = _gauss()
    zb, gb, _ = midplane.midplane_gravity(Z, n, sg_mode, "balanced")
    zs, gs, _ = midplane.midplane_gravity(Z, n, sg_mode, "sun")
    assert zb == 0.0 and zs == 0.0
    np.testing.assert_array_equal(gb, gs)
    rho = RHO_PER_N * n
    np.testing.assert_array_equal(hydrostatic.p_tot_kb_full_column(Z, rho, gb, z_split=zb),
                                  hydrostatic.p_tot_kb_full_column(Z, rho, gs))


@pytest.mark.parametrize("sg", [True, False])
def test_more_mass_below_gives_negative_zmid_and_continuous_ptot(sg):
    n = _gauss(z0=-20.0)
    zm = midplane.find_balanced_midplane(Z, n, self_gravity=sg)
    assert -20.0 < zm < 0.0
    g = midplane.centred_g_cgs(Z, n, zm, self_gravity=sg)
    top, bot = midplane.branch_pressures(Z, RHO_PER_N * n, g, zm)
    assert abs(top / bot - 1) < 1e-6
    # the Sun-centred split is NOT continuous for this profile
    gs, _ = gravity.g_total_cgs(Z, n[:, None, None] if sg else None,
                                "footprint_mean" if sg else "off",
                                footprint_mask=np.ones((1, 1), bool) if sg else None)
    t0, b0 = midplane.branch_pressures(Z, RHO_PER_N * n, np.asarray(gs, float).ravel(), 0.0)
    assert abs(t0 / b0 - 1) > 1e-2


def test_offgrid_zmid_branches_column_and_plane_assignment():
    n = _gauss(z0=-7.3, h=45.0) + _gauss(z0=90.0, h=25.0, n0=0.3)
    zm = midplane.find_balanced_midplane(Z, n, self_gravity=True)
    assert np.min(np.abs(Z - zm)) > 1e-3                     # really between planes
    rho = RHO_PER_N * n
    g = midplane.centred_g_cgs(Z, n, zm, self_gravity=True)
    top, bot = midplane.branch_pressures(Z, rho, g, zm)
    assert abs(top / bot - 1) < 1e-6

    # enclosed column exact for piecewise-linear rho: compare with a 100x
    # refined grid of the SAME piecewise-linear profile
    rho_msun = n * gravity.RHO_MSUN_PC3_PER_N
    sig = midplane.enclosed_column_about(Z, rho_msun, zm)
    zf = np.linspace(Z[0], Z[-1], (Z.size - 1) * 100 + 1)
    rf = np.interp(zf, Z, rho_msun)
    for k in (10, 150, 200, 260, 390):
        s = abs(Z[k] - zm)
        m = (zf >= max(zm - s, Z[0])) & (zf <= min(zm + s, Z[-1]))
        lo, hi = max(zm - s, Z[0]), min(zm + s, Z[-1])
        zz = np.concatenate([[lo], zf[m], [hi]])
        ref = np.trapezoid(np.interp(zz, Z, rho_msun), zz)
        assert sig[k] == pytest.approx(ref, rel=1e-9, abs=1e-12)

    # planes above zm take the top-down branch, planes below the bottom-up one
    P = hydrostatic.p_tot_kb_full_column(Z, rho, g, z_split=zm)
    k_up = int(np.searchsorted(Z, zm))                    # first plane above zm
    top_k = hydrostatic.integrate_ptot_halfcolumn(Z[k_up:][::-1], rho[k_up:][::-1], g[k_up:][::-1])[-1]
    bot_k = hydrostatic.integrate_ptot_halfcolumn(Z[:k_up], rho[:k_up], g[:k_up])[-1]
    assert P[k_up] == top_k and P[k_up - 1] == bot_k


def test_closed_form_holds_about_the_balanced_centre():
    """0.5 [P_gas top + P_gas bottom] = pi G Sigma^2 / 2 about ANY centre."""
    n = _gauss(z0=-15.0)
    zm = midplane.find_balanced_midplane(Z, n, self_gravity=True)
    rho_msun = n * gravity.RHO_MSUN_PC3_PER_N
    g_gas = gravity.GRAV_PREFACTOR_CGS * midplane.enclosed_column_about(Z, rho_msun, zm)
    top, bot = midplane.branch_pressures(Z, RHO_PER_N * n, g_gas, zm)
    sigma = float(np.trapezoid(rho_msun, Z))
    from src.conventions import K_B, M_SUN_G, PC_CM
    closed = 0.25 * gravity.GRAV_PREFACTOR_CGS * sigma * sigma * M_SUN_G / PC_CM ** 2 / K_B
    assert 0.5 * (top + bot) == pytest.approx(closed, rel=1e-3)


def test_sun_mode_is_the_old_code_path():
    n = _gauss(z0=-20.0)
    z0, g, _ = midplane.midplane_gravity(Z, n, "footprint_mean", "sun")
    g_old, _ = gravity.g_total_cgs(Z, n[:, None, None], "footprint_mean",
                                   footprint_mask=np.ones((1, 1), bool))
    assert z0 == 0.0
    np.testing.assert_array_equal(g, g_old)
    rho = RHO_PER_N * n
    np.testing.assert_array_equal(hydrostatic.p_tot_kb_full_column(Z, rho, g),
                                  hydrostatic.p_tot_kb_full_column(Z, rho, g, z_split=0.0))


REPRO_TXT = Path(__file__).resolve().parents[1] / "results" / "midplane_sun_reproduction.txt"


@pytest.mark.skipif(not REPRO_TXT.exists(), reason="validation report not generated")
def test_sun_mode_reproduced_the_pipeline_exactly():
    text = REPRO_TXT.read_text()
    assert "bitwise False" not in text and text.count("bitwise True") >= 12
    assert "rows differing in any field (text or value, exact): 0" in text
    assert "differing (bitwise): 0" in text
