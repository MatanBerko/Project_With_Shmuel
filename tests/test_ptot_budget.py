"""
Fig. 3 weight budget: the pieces of scripts/compute/compute_ptot_budget.py on
toy profiles.

(i)   Symmetric profile: the mean of the two gas-self-gravity midplane
      branches is pi G Sigma^2 / 2, and the balanced-midplane root-finder
      returns z_mid = 0.
(ii)  More mass below the plane: z_mid < 0, and at z_mid the top-down and
      bottom-up weights agree.
(iii) The stars + DM + gas components sum to the total, in g and in P.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.physics import gravity, hydrostatic

_SPEC = importlib.util.spec_from_file_location(
    "compute_ptot_budget",
    Path(__file__).resolve().parents[1] / "scripts" / "compute" / "compute_ptot_budget.py")
cpb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cpb)

Z = np.arange(-600.0, 600.0 + 0.5, 1.0)          # symmetric, fine grid


def _gauss(z0=0.0, h=80.0, n0=1.0):
    return n0 * np.exp(-0.5 * ((Z - z0) / h) ** 2) + 0.01


def test_symmetric_closed_form_and_zmid_zero():
    n = _gauss()
    rho = gravity.RHO_PER_N * n
    sig_gas = cpb.sigma_components(Z, n)["gas"]
    p_top, p_bot = cpb.pressure_branches(Z, rho, cpb.g_from_sigma(sig_gas))
    closed = cpb.closed_form_sg_kb(float(sig_gas[-1]))
    assert 0.5 * (p_top + p_bot) == pytest.approx(closed, rel=1e-3)
    assert abs(cpb.find_zmid(Z, n, self_gravity=True)) < 0.5
    assert abs(cpb.find_zmid(Z, n, self_gravity=False)) < 0.5


def test_closed_form_holds_for_an_asymmetric_profile_too():
    """The identity needs a symmetric DOMAIN, not a symmetric profile."""
    n = _gauss(z0=-40.0, h=60.0) + _gauss(z0=120.0, h=30.0, n0=0.4)
    rho = gravity.RHO_PER_N * n
    sig_gas = cpb.sigma_components(Z, n)["gas"]
    p_top, p_bot = cpb.pressure_branches(Z, rho, cpb.g_from_sigma(sig_gas))
    assert abs(p_top / p_bot - 1) > 0.05                       # really asymmetric
    assert 0.5 * (p_top + p_bot) == pytest.approx(cpb.closed_form_sg_kb(float(sig_gas[-1])),
                                                  rel=1e-3)


@pytest.mark.parametrize("self_gravity", [True, False])
def test_more_mass_below_puts_zmid_below_and_balances(self_gravity):
    n = _gauss(z0=-25.0)
    zm = cpb.find_zmid(Z, n, self_gravity=self_gravity)
    assert zm < 0
    sig = cpb.sigma_components(Z, n, z_ref_pc=zm)
    tot = sig["star"] + sig["dm"] + (sig["gas"] if self_gravity else 0.0)
    top, bot = cpb.pressure_branches(Z, gravity.RHO_PER_N * n, cpb.g_from_sigma(tot),
                                     z_split=zm)
    assert abs(top / bot - 1) < 1e-6


def test_components_sum_to_total():
    n = _gauss(z0=-10.0) + _gauss(z0=200.0, h=40.0, n0=0.2)
    rho = gravity.RHO_PER_N * n
    sig = cpb.sigma_components(Z, n)
    g = {k: cpb.g_from_sigma(v) for k, v in sig.items()}
    g_ref, _ = gravity.g_total_cgs(Z, n[:, None, None], "footprint_mean",
                                   footprint_mask=np.ones((1, 1), dtype=bool))
    np.testing.assert_allclose(g["star"] + g["dm"], gravity.g_ext_cgs(Z), rtol=1e-12, atol=0)
    np.testing.assert_allclose(g["star"] + g["dm"] + g["gas"], g_ref, rtol=1e-12, atol=0)
    P = {k: hydrostatic.p_tot_kb_full_column(Z, rho, gk) for k, gk in g.items()}
    P_tot = hydrostatic.p_tot_kb_full_column(Z, rho, g_ref)
    np.testing.assert_allclose(P["star"] + P["dm"] + P["gas"], P_tot, rtol=1e-12, atol=1e-9)
    br = {k: cpb.pressure_branches(Z, rho, gk) for k, gk in g.items()}
    br_tot = cpb.pressure_branches(Z, rho, g_ref)
    for j in (0, 1):
        assert sum(br[k][j] for k in g) == pytest.approx(br_tot[j], rel=1e-12)
