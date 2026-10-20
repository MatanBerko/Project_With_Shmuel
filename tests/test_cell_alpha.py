"""
Step 2e: the per-cell alpha_cell = <P_tot>_area / p_th against alpha_area.

Three properties on toy data, each exact:

1. THE HARMONIC-MEAN IDENTITY. With the numerator one number per group,
   the weighted mean of alpha_cell is <P_tot>_area * <1/p_th>_w =
   <P_tot>_area / P_th,harm. cell_alpha_stats() computes it both ways
   (from the harmonic mean, and by reducing the formed alpha_cell array
   with weighted_stats) and both must match each other and a by-hand sum.

2. THE MEDIAN IDENTITY. alpha_cell is monotone decreasing in p_th, so a
   rank-based median commutes with it: for an odd cell count,
   median(alpha_cell) == <P_tot>_area / median(p_th) bit for bit, and the
   15th percentile of alpha_cell is <P_tot>_area over the 85th of p_th.
   For an even count the nearest-rank estimator lands one order statistic
   off, which median_identity_residual() measures; that is checked too,
   so the in-script check is pinned to the right rank.

3. UNIFORM P_th. With every cell at the same p_th the harmonic and
   arithmetic means coincide, so <alpha_cell> = median = alpha_area, with
   alpha_area taken through the script's own area_averages_from_sums().

Plus the negative control: check_cell_alpha() must raise when
<alpha_cell> < alpha_area, or it checks nothing.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.physics.stats import weighted_percentile

_SPEC = importlib.util.spec_from_file_location(
    "compute_area_alpha",
    Path(__file__).resolve().parents[1] / "scripts" / "compute" / "compute_area_alpha.py")
caa = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(caa)

P_AREA = 7800.3


def _toy_cells(n, seed=5):
    """Log-normal p_th and n_H, as lumpy as the real neutral medium."""
    rng = np.random.default_rng(seed)
    p_th = 3000.0 * np.exp(rng.normal(0.0, 0.9, n))
    n_h = np.exp(rng.normal(0.0, 1.5, n))
    return p_th, n_h


# ---------------------------------------------------------------------------
# 1. harmonic-mean identity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("weighting", ["vol", "mw"])
def test_mean_alpha_cell_is_p_area_over_harmonic_mean(weighting):
    p_th, n_h = _toy_cells(1001)
    w = np.ones_like(p_th) if weighting == "vol" else n_h
    st = caa.cell_alpha_stats(p_th, w, P_AREA)

    by_hand_harm = np.sum(w) / np.sum(w / p_th)
    assert st["Pth_harm"] == pytest.approx(by_hand_harm, rel=1e-14)
    assert st["alpha_cell_mean"] == pytest.approx(P_AREA / by_hand_harm, rel=1e-14)
    assert st["alpha_cell_mean_direct"] == pytest.approx(st["alpha_cell_mean"], rel=1e-13)
    # ... and the data is lumpy enough that harmonic and arithmetic differ,
    # so the identity is not passing on a degenerate case
    arith = np.sum(w * p_th) / np.sum(w)
    assert st["Pth_harm"] < 0.8 * arith


# ---------------------------------------------------------------------------
# 2. median identity
# ---------------------------------------------------------------------------
def test_median_alpha_cell_is_p_area_over_median_pth_for_odd_n():
    p_th, _ = _toy_cells(101)   # odd, and 0.15 * 101 is not an integer
    w = np.ones_like(p_th)
    st = caa.cell_alpha_stats(p_th, w, P_AREA)

    assert st["alpha_cell_median"] == P_AREA / weighted_percentile(p_th, w, 50.0)
    assert st["Pth_median"] == weighted_percentile(p_th, w, 50.0)
    # the percentile pair swaps ends through 1/p_th
    assert st["alpha_cell_p_lo"] == P_AREA / weighted_percentile(p_th, w, 85.0)
    assert st["alpha_cell_p_hi"] == P_AREA / weighted_percentile(p_th, w, 15.0)

    rel_med, rel_map, n = caa.median_identity_residual(
        p_th, P_AREA, st["alpha_cell_median"])
    assert n == 101 and rel_med == 0.0 and rel_map == 0.0


def test_even_n_median_lands_on_the_mirrored_order_statistic():
    p_th, _ = _toy_cells(100)
    st = caa.cell_alpha_stats(p_th, np.ones_like(p_th), P_AREA)
    rel_med, rel_map, _ = caa.median_identity_residual(
        p_th, P_AREA, st["alpha_cell_median"])
    srt = np.sort(p_th)
    assert st["alpha_cell_median"] == P_AREA / srt[50]   # one above p_th's median (srt[49])
    assert rel_map == 0.0
    assert rel_med > 0.0   # distinct continuous values: the gap is real, and reported


# ---------------------------------------------------------------------------
# 3. uniform P_th
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("weighting", ["vol", "mw"])
def test_uniform_pth_gives_mean_equal_median_equal_alpha_area(weighting):
    n = 64
    p_th = np.full(n, 2500.0)
    _, n_h = _toy_cells(n)
    ptot = np.linspace(5000.0, 9000.0, n)   # any P_tot; only its mean enters
    neutral = np.ones(n, dtype=bool)

    acc = caa._empty_sums((1,))
    caa._accumulate_plane_full(acc, [0], ptot, p_th, n_h, neutral)
    area = caa.area_averages_from_sums(acc)
    p_area = float(area["Ptot_area"][0])
    alpha_area = float(area[f"alpha_area_{weighting}"][0])

    w = np.ones(n) if weighting == "vol" else n_h
    st = caa.cell_alpha_stats(p_th, w, p_area)
    assert st["alpha_cell_mean"] == pytest.approx(alpha_area, rel=1e-14)
    assert st["alpha_cell_median"] == pytest.approx(alpha_area, rel=1e-14)
    assert st["alpha_cell_p_lo"] == st["alpha_cell_p_hi"] == st["alpha_cell_median"]
    assert st["Pth_harm"] == pytest.approx(2500.0, rel=1e-14)


# ---------------------------------------------------------------------------
# the in-script check must be able to fail
# ---------------------------------------------------------------------------
def test_check_cell_alpha_raises_when_mean_is_below_alpha_area():
    good = {k: np.array([2.0]) for k in caa.CELL_ALPHA_KEYS}
    stats = {("mean", "vol"): good}
    res = {"mean": np.array([[0.0, 0.0]])}
    caa.check_cell_alpha(stats, {("mean", "vol"): np.array([1.9])}, res)
    with pytest.raises(RuntimeError, match="alpha_area FAILED"):
        caa.check_cell_alpha(stats, {("mean", "vol"): np.array([2.1])}, res)
    with pytest.raises(RuntimeError, match="median"):
        caa.check_cell_alpha(stats, {("mean", "vol"): np.array([1.9])},
                             {"mean": np.array([[1e-3, 1e-3]])})
