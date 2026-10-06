"""
Step 1e: the three alpha estimators, and the alpha < 1 convention.

    alpha_median_of_ratios = weighted median of per-cell P_tot,i/P_th,i
    alpha_mean_of_ratios   = weighted mean   of per-cell P_tot,i/P_th,i
    alpha_ratio_of_means   = sum(w P_tot) / sum(w P_th)          (NEW)

The third is the one that is easy to get wrong, because it looks like it
should be derivable from the alpha cube. It is not: it is the
(w_i * p_th,i)-weighted mean of alpha_i, which is a different weighting
from w_i. The first test here is that identity, and the hand-checked
three-cell case below shows how far apart the three can sit on data that
fits on one line (2.5 / 2.2778 / 2.0).
"""

import numpy as np
import pytest

from src.physics.derived import (
    alpha_below_one_fraction,
    mach_number,
    sigma_eff_kmps,
    sigma_nt_kmps,
    sound_speed_kmps,
)
from src.physics.stats import (
    mass_weighted_ratio_of_means,
    ratio_of_means,
    volume_weighted_ratio_of_means,
    weighted_mean,
    weighted_stats,
)


# ---------------------------------------------------------------------------
# 1. the defining identity
# ---------------------------------------------------------------------------
def test_ratio_of_means_is_the_p_th_weighted_mean_of_the_per_cell_alphas():
    """sum(w P_tot)/sum(w P_th) == the (w*P_th)-weighted mean of alpha_i.

    Checked on random data across several decades, and -- the part that
    matters -- shown NOT to equal the w-weighted mean, so the identity is
    a real constraint rather than an algebraic restatement of
    weighted_mean().
    """
    rng = np.random.default_rng(1105)
    for _ in range(25):
        n = rng.integers(5, 400)
        p_th = 10.0 ** rng.uniform(1.0, 4.5, n)
        alpha_i = 10.0 ** rng.uniform(-1.0, 1.5, n)
        p_tot = alpha_i * p_th
        w = 10.0 ** rng.uniform(-2, 2, n)

        rom = ratio_of_means(p_tot, p_th, w)
        assert rom == pytest.approx(weighted_mean(alpha_i, w * p_th), rel=1e-12)
        # and it is genuinely a different number from the w-weighted mean
        assert rom != pytest.approx(weighted_mean(alpha_i, w), rel=1e-6)

    # the two weightings of ratio_of_means, through their wrappers
    mask = np.ones(n, dtype=bool)
    assert volume_weighted_ratio_of_means(p_tot, p_th, mask) == pytest.approx(
        ratio_of_means(p_tot, p_th, np.ones(n)), rel=1e-14)
    assert mass_weighted_ratio_of_means(p_tot, p_th, mask, w) == pytest.approx(
        ratio_of_means(p_tot, p_th, w), rel=1e-14)


def test_ratio_of_means_cell_selection_and_degenerate_cases():
    p_tot = np.array([3000.0, np.nan, 5000.0, 6000.0, 7000.0])
    p_th = np.array([1000.0, 2000.0, np.nan, 0.0, 2000.0])
    w = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    # only cell 0 survives: 1 is NaN p_tot, 2 is NaN p_th, 3 has p_th = 0,
    # 4 has zero weight
    assert ratio_of_means(p_tot, p_th, w) == pytest.approx(3.0)

    assert np.isnan(ratio_of_means(np.array([1.0]), np.array([1.0]), np.array([0.0])))
    assert np.isnan(ratio_of_means(np.array([]), np.array([]), np.array([])))
    # masked-out everything
    assert np.isnan(volume_weighted_ratio_of_means(
        np.array([1.0, 2.0]), np.array([1.0, 1.0]), np.array([False, False])))


# ---------------------------------------------------------------------------
# 2. constant P_tot: median_of_ratios collapses onto P_tot / median(P_th)
# ---------------------------------------------------------------------------
def test_constant_ptot_slab_median_of_ratios_equals_ptot_over_median_pth():
    """With P_tot the same in every cell, alpha_i = P_tot / P_th,i is a
    strictly decreasing function of P_th,i. A rank-based median commutes
    with any monotonic transform, so the median of the ratios must be
    exactly P_tot divided by the median P_th -- no tolerance needed.

    Uses an ODD cell count: for the nearest-rank weighted estimator with
    equal weights and an even count, the chosen rank is not symmetric
    under reversal, so the identity holds only up to one order statistic.
    The odd case is the one with an exact answer, and the even case is
    checked separately to be within one order statistic rather than
    silently skipped.
    """
    P_TOT = 6000.0
    rng = np.random.default_rng(7)

    for n_cells in (1, 3, 5, 51, 501):
        p_th = 10.0 ** rng.uniform(1.5, 4.0, n_cells)
        p_tot = np.full(n_cells, P_TOT)
        alpha_i = p_tot / p_th
        w = np.ones(n_cells)

        got = weighted_stats(alpha_i, w).median
        expected = P_TOT / np.median(p_th)
        assert got == pytest.approx(expected, rel=1e-13), (n_cells, got, expected)

    # even count: still within one order statistic of P_tot/median(P_th)
    p_th = np.sort(10.0 ** rng.uniform(1.5, 4.0, 100))
    alpha_i = P_TOT / p_th
    got = weighted_stats(alpha_i, np.ones(100)).median
    bracket = sorted((P_TOT / p_th[49], P_TOT / p_th[50]))
    assert bracket[0] <= got <= bracket[1]

    # and the other two estimators are NOT equal to it, so this test is
    # about the median specifically
    assert weighted_mean(alpha_i, np.ones(100)) != pytest.approx(got, rel=1e-6)
    assert ratio_of_means(P_TOT * np.ones(100), p_th, np.ones(100)) != pytest.approx(
        got, rel=1e-6)


def test_constant_ptot_slab_ratio_of_means_is_ptot_over_mean_pth():
    """The companion identity for the third estimator: with P_tot
    constant, sum(w P_tot)/sum(w P_th) = P_tot / weighted-mean(P_th)."""
    P_TOT = 6000.0
    rng = np.random.default_rng(8)
    p_th = 10.0 ** rng.uniform(1.5, 4.0, 300)
    w = 10.0 ** rng.uniform(-1, 1, 300)
    got = ratio_of_means(np.full(300, P_TOT), p_th, w)
    assert got == pytest.approx(P_TOT / weighted_mean(p_th, w), rel=1e-13)


# ---------------------------------------------------------------------------
# 3. the hand-checked three-cell case
# ---------------------------------------------------------------------------
def test_three_cell_hand_checked_case():
    """P_tot = (3000, 4000, 5000), P_th = (1000, 3000, 2000), w = 1:

        alpha_i                = (3, 4/3, 5/2)
        median_of_ratios       = 5/2        = 2.5
        mean_of_ratios         = 41/18      = 2.27778
        ratio_of_means         = 12000/6000 = 2.0

    All three on the same three cells. The spread (2.5 vs 2.0, 25%) is
    the reason the paper reports all three rather than one.
    """
    p_tot = np.array([3000.0, 4000.0, 5000.0])
    p_th = np.array([1000.0, 3000.0, 2000.0])
    w = np.ones(3)
    alpha_i = p_tot / p_th

    np.testing.assert_allclose(alpha_i, [3.0, 4.0 / 3.0, 2.5], rtol=1e-15)
    st = weighted_stats(alpha_i, w)
    assert st.median == pytest.approx(2.5, abs=1e-12)
    assert st.arithmetic_mean == pytest.approx(2.2777777777777777, abs=1e-12)
    assert ratio_of_means(p_tot, p_th, w) == pytest.approx(2.0, abs=1e-12)

    # strictly ordered: ratio_of_means < mean_of_ratios < median_of_ratios
    # here, because the high-alpha cell is the low-P_th one and
    # ratio_of_means down-weights exactly that cell.
    assert ratio_of_means(p_tot, p_th, w) < st.arithmetic_mean < st.median


# ---------------------------------------------------------------------------
# 4. alpha < 1 is NaN, and the fraction is reported
# ---------------------------------------------------------------------------
def test_mach_and_sigma_nt_are_nan_below_alpha_one_but_sigma_eff_is_not():
    alpha = np.array([0.0, 0.5, 0.999, 1.0, 2.0, 4.0, np.nan])
    T = np.full(alpha.shape, 100.0)

    mach = mach_number(alpha)
    s_nt = sigma_nt_kmps(alpha, T)
    s_eff = sigma_eff_kmps(alpha, T)

    assert np.all(np.isnan(mach[:3])), mach
    assert np.all(np.isnan(s_nt[:3])), s_nt
    assert np.isnan(mach[-1]) and np.isnan(s_nt[-1])

    # exactly at alpha = 1 both are defined and zero -- there is genuinely
    # no non-thermal support there, as opposed to it being unmeasurable
    assert mach[3] == 0.0 and s_nt[3] == 0.0

    np.testing.assert_allclose(mach[4:6], np.sqrt(3.0 * (alpha[4:6] - 1.0)), rtol=1e-14)
    # sigma_eff is defined for every alpha >= 0: it measures total support
    assert np.all(np.isfinite(s_eff[:6]))
    np.testing.assert_allclose(s_eff[:6], np.sqrt(alpha[:6]) * sound_speed_kmps(T[:6]),
                                 rtol=1e-14)
    # and sigma_nt = Mach * c_s wherever defined
    ok = np.isfinite(s_nt)
    np.testing.assert_allclose(s_nt[ok], mach[ok] * sound_speed_kmps(T[ok]), rtol=1e-12)


def test_alpha_below_one_fraction_is_weighted_and_matches_the_nans():
    alpha = np.array([0.5, 0.5, 2.0, 2.0])
    assert alpha_below_one_fraction(alpha, np.ones(4)) == pytest.approx(0.5)
    # mass-weighted: the two sub-unity cells carry 1 of 10 units of weight
    assert alpha_below_one_fraction(alpha, np.array([0.5, 0.5, 4.5, 4.5])) == pytest.approx(0.1)
    # non-finite alpha and zero-weight cells are excluded from both halves
    assert alpha_below_one_fraction(np.array([0.5, np.nan, 2.0]),
                                      np.array([1.0, 1.0, 1.0])) == pytest.approx(0.5)
    assert alpha_below_one_fraction(alpha, np.array([1.0, 1.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert np.isnan(alpha_below_one_fraction(alpha, np.zeros(4)))

    # the fraction is exactly the fraction of NaN Mach values, which is
    # the whole point of reporting it
    mach = mach_number(alpha)
    assert np.mean(np.isnan(mach)) == pytest.approx(
        alpha_below_one_fraction(alpha, np.ones(4)))


# ---------------------------------------------------------------------------
# 5. the pipeline helpers that assemble all of the above
# ---------------------------------------------------------------------------
def test_pipeline_alpha_estimates_and_derived_agree_with_the_primitives():
    """pipeline.compute_all.alpha_estimates / derived_from_alpha are the
    single place the finalize stage builds these numbers, for both the
    slabs and the profile bins. Check they are just the primitives, so a
    slab number and a profile number cannot drift apart."""
    import pipeline.compute_all as m

    rng = np.random.default_rng(99)
    p_th = 10.0 ** rng.uniform(2, 4, 200)
    alpha_i = 10.0 ** rng.uniform(-0.6, 1.2, 200)
    p_tot = alpha_i * p_th
    n = 10.0 ** rng.uniform(-2, 1, 200)
    neutral = rng.random(200) > 0.2

    est = m.alpha_estimates(alpha_i, p_tot, p_th, neutral, n)

    w_vol = np.where(neutral, 1.0, 0.0)
    w_mw = np.where(neutral, n, 0.0)
    for wt, w in (("vol", w_vol), ("mw", w_mw)):
        s = weighted_stats(alpha_i, w)
        assert est[wt]["alpha_median_of_ratios"] == pytest.approx(s.median, rel=1e-14)
        assert est[wt]["alpha_mean_of_ratios"] == pytest.approx(s.arithmetic_mean, rel=1e-14)
        assert est[wt]["alpha_ratio_of_means"] == pytest.approx(
            ratio_of_means(p_tot, p_th, w), rel=1e-14)
        assert est[wt]["alpha_p15_of_ratios"] == pytest.approx(s.p_lo, rel=1e-14)
        assert est[wt]["alpha_p85_of_ratios"] == pytest.approx(s.p_hi, rel=1e-14)
        assert est[wt]["frac_alpha_lt1"] == pytest.approx(
            alpha_below_one_fraction(alpha_i, w), rel=1e-14)

        # every estimator saw the same cells: none of them is computed on
        # the full array by accident
        assert est[wt]["alpha_median_of_ratios"] != pytest.approx(
            weighted_stats(alpha_i, np.ones(200)).median, rel=1e-9) or neutral.all()

        d = m.derived_from_alpha(est[wt], 120.0)
        for tag, alpha_name in m.ALPHA_ESTIMATORS:
            A = np.array([est[wt][alpha_name]])
            assert np.allclose(d[f"Mach_from_{tag}"], mach_number(A)[0], equal_nan=True)
            assert np.allclose(d[f"sigma_nt_from_{tag}"],
                                 sigma_nt_kmps(A, np.array([120.0]))[0], equal_nan=True)
        assert d["c_s"] == pytest.approx(sound_speed_kmps(np.array([120.0]))[0], rel=1e-14)
        assert d["sigma_eff"] == pytest.approx(
            sigma_eff_kmps(np.array([est[wt]["alpha_mean_of_ratios"]]),
                            np.array([120.0]))[0], rel=1e-14)

    # the three estimators are distinct on this data -- otherwise the test
    # above would pass even if two of them were the same call
    v = est["vol"]
    assert len({round(v["alpha_median_of_ratios"], 9),
                round(v["alpha_mean_of_ratios"], 9),
                round(v["alpha_ratio_of_means"], 9)}) == 3
