"""
Fig. 7: the velocity statistics of scripts/compute/compute_velocity.py on toy cubes.

1. A SINGLE UNIFORM PHASE reproduces the closed forms exactly:
   sigma_eff = sqrt(k_B W / rho), sigma_th = sqrt(k_B P / rho), <M^2> = W/P - 1.
2. TWO PHASES AT DIFFERENT DENSITIES under one shared W: each phase's
   sigma_eff uses its OWN <rho>_k (so the denser phase needs less velocity),
   and the mass-weighted <M^2> of the union equals the mass-weighted mean of
   the per-phase <M^2> -- the identity behind assertion (i).
3. A CELL WITH P_th > W contributes a NEGATIVE M^2: it is kept, not clipped,
   and it is counted in the alpha_cell < 1 mass fraction.
"""

import importlib.util
from pathlib import Path

import numpy as np

from src.conventions import K_B, M_H, MU
from src.physics.him import PHASE_CNM, PHASE_HIM, PHASE_WNM

_SPEC = importlib.util.spec_from_file_location(
    "compute_velocity",
    Path(__file__).resolve().parents[1] / "scripts" / "compute" / "compute_velocity.py")
cv = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cv)

SHAPE = (3, 12, 12)


def _stats(n_H, p_th, phase, W, n_min=1):
    comb = cv.combine(cv.phase_sums(n_H, p_th, phase, W))
    n_total = comb["ALL"]["n"]
    return comb, {name: cv.velocity_stats(comb[name], W, n_total, n_min=n_min)[0]
                  for name in cv.STAT_PHASES}


def test_single_uniform_phase_closed_forms():
    n0, p0, W = 0.7, 3500.0, 9100.0
    n_H = np.full(SHAPE, n0)
    p_th = np.full(SHAPE, p0)
    phase = np.full(SHAPE, PHASE_WNM)
    comb, st = _stats(n_H, p_th, phase, W)
    rho = MU * M_H * n0
    w = st["WNM"]
    np.testing.assert_allclose(w["sigma_eff"], np.sqrt(K_B * W / rho) / 1e5, rtol=1e-12)
    np.testing.assert_allclose(w["sigma_th"], np.sqrt(K_B * p0 / rho) / 1e5, rtol=1e-12)
    np.testing.assert_allclose(w["M2"], W / p0 - 1.0, rtol=1e-12)
    np.testing.assert_allclose(w["sigma_nt"] ** 2, w["sigma_eff"] ** 2 - w["sigma_th"] ** 2,
                               rtol=1e-12)
    assert w["vol_frac"] == 1.0 and w["mass_frac_alpha_lt1"] == 0.0
    # NEUTRAL is the same cells; ALL gives the same sigma_eff
    for k in ("sigma_eff", "sigma_th", "M2"):
        np.testing.assert_allclose(st["NEUTRAL"][k], w[k], rtol=1e-12)
    s_all, _ = cv.sigma_eff_all(comb["ALL"], W)
    np.testing.assert_allclose(s_all, w["sigma_eff"], rtol=1e-12)
    # the absent phase is NaN under the cell threshold
    assert np.isnan(st["CNM"]["sigma_eff"])


def test_two_phases_shared_W():
    rng = np.random.default_rng(3)
    W = 8000.0
    phase = np.where(rng.random(SHAPE) < 0.3, PHASE_CNM, PHASE_WNM)
    cold = phase == PHASE_CNM
    n_H = np.where(cold, rng.uniform(20.0, 60.0, SHAPE), rng.uniform(0.2, 0.6, SHAPE))
    p_th = np.where(cold, rng.uniform(1500.0, 4000.0, SHAPE), rng.uniform(2000.0, 6000.0, SHAPE))
    _, st = _stats(n_H, p_th, phase, W)

    for name, m in (("CNM", cold), ("WNM", ~cold)):
        rho_k = MU * M_H * n_H[m].mean()
        np.testing.assert_allclose(st[name]["sigma_eff"], np.sqrt(K_B * W / rho_k) / 1e5,
                                   rtol=1e-13)
        mw = n_H[m]
        np.testing.assert_allclose(st[name]["M2"], np.sum(mw * (W / p_th[m] - 1)) / mw.sum(),
                                   rtol=1e-13)
        np.testing.assert_allclose(
            st[name]["sigma_th"], np.sqrt(K_B * p_th[m].sum() / (MU * M_H * mw.sum())) / 1e5,
            rtol=1e-13)
    assert st["CNM"]["sigma_eff"] < st["WNM"]["sigma_eff"]
    # mass-weighted recombination, and the <M^2> + 1 = mass-weighted <alpha_cell> identity
    m_c, m_w = n_H[cold].sum(), n_H[~cold].sum()
    np.testing.assert_allclose(
        st["NEUTRAL"]["M2"], (m_c * st["CNM"]["M2"] + m_w * st["WNM"]["M2"]) / (m_c + m_w),
        rtol=1e-13)
    alpha_mw = W * np.sum(n_H / p_th) / n_H.sum()
    np.testing.assert_allclose(st["NEUTRAL"]["M2"] + 1.0, alpha_mw, rtol=1e-13)


def test_overpressured_cell_negative_M2():
    W = 4000.0
    n_H = np.full(SHAPE, 1.0)
    p_th = np.full(SHAPE, 2000.0)              # alpha_cell = 2 -> M^2 = +1
    p_th[0, 0, :4] = 8000.0                    # alpha_cell = 0.5 -> M^2 = -0.5
    n_H[0, 0, :4] = 3.0
    phase = np.full(SHAPE, PHASE_WNM)
    phase[2] = PHASE_HIM                       # HIM cells are not part of NEUTRAL
    _, st = _stats(n_H, p_th, phase, W)
    neu = phase != PHASE_HIM
    mw = n_H[neu]
    m2_cell = W / p_th[neu] - 1.0
    assert (m2_cell < 0).sum() == 4
    expect = np.sum(mw * m2_cell) / mw.sum()
    np.testing.assert_allclose(st["NEUTRAL"]["M2"], expect, rtol=1e-12)
    clipped = np.sum(mw * np.maximum(m2_cell, 0)) / mw.sum()
    assert st["NEUTRAL"]["M2"] < clipped       # the negative contribution is kept
    np.testing.assert_allclose(st["NEUTRAL"]["mass_frac_alpha_lt1"], 12.0 / mw.sum(),
                               rtol=1e-12)
    # an all-overpressured phase: <M^2> < 0 and sqrt(<M^2>) is NaN
    _, st2 = _stats(np.ones(SHAPE), np.full(SHAPE, 8000.0), np.full(SHAPE, PHASE_CNM), W)
    np.testing.assert_allclose(st2["CNM"]["M2"], -0.5, rtol=1e-12)
    assert np.isnan(st2["CNM"]["sqrt_M2"])
    assert st2["CNM"]["sigma_nt"] == 0.0 and st2["CNM"]["sigma_nt2_unclipped"] < 0


def test_mass_weighted_percentiles_known_answer():
    # values 1..5 with mass weights 1, 1, 1, 1, 16 (total 20): the cumulative
    # weight reaches 15% (= 3) at the third value and 85% (= 17) only at the fifth
    n_H = np.array([1.0, 1.0, 1.0, 1.0, 16.0])
    W = 1000.0
    p_th = W / (np.arange(1.0, 6.0) + 1.0)          # M2_cell = 1, 2, 3, 4, 5
    out = cv.cell_percentiles(n_H, p_th, W, n_min=1)
    assert out["M2_cell_p_lo"] == 3.0 and out["M2_cell_p_hi"] == 5.0
    # sigma_eff,cell decreases with rho: the heavy cell is the LOW end
    s = np.sqrt(K_B * W / (MU * M_H * n_H)) / 1e5
    assert out["sigma_eff_cell_p_lo"] == s[4] and out["sigma_eff_cell_p_hi"] == s[0]
    # NaN under the cell threshold
    assert np.isnan(cv.cell_percentiles(n_H, p_th, W)["M2_cell_p_lo"])


def test_equal_weights_reproduce_unweighted_percentiles():
    # uniform density -> equal mass weights -> the plain nearest-rank percentiles
    rng = np.random.default_rng(5)
    p_th = rng.uniform(1000.0, 9000.0, 1001)
    W = 6000.0
    out = cv.cell_percentiles(np.full(p_th.size, 0.5), p_th, W, n_min=1)
    m2 = W / p_th - 1.0
    cs = np.sqrt(K_B * p_th / (MU * M_H * 0.5)) / 1e5
    for q, v in (("M2_cell", m2), ("cs_cell", cs)):
        lo, hi = np.percentile(v, (15.0, 85.0), method="inverted_cdf")
        assert out[f"{q}_p_lo"] == lo and out[f"{q}_p_hi"] == hi
