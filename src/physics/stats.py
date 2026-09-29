"""
Weighted statistics: median, arithmetic mean, 16th/84th percentiles, both
volume-weighted (w=1 per finite voxel) and mass-weighted (w=n_model).

The nearest-rank cumulative-weight formula is ported verbatim from:
  fig2_histograms/designing_histograms.py:146-156 weighted_median()
  fig4_vertical_profiles/compute_data.py:144-154 weighted_percentile()
(these two agree exactly -- same cumulative-weight-search algorithm,
default side="left").

NOTE on an old-script inconsistency found while porting (NOT reproduced
here -- see the final report): fig4_vertical_profiles/compute_data.py's
own `stats_vol()` (:156-164) used plain np.percentile/np.median (LINEAR
interpolation between order statistics) for its "volume-weighted" branch,
while its `stats_mw()` (:166-177) used the nearest-rank weighted_percentile
above for the mass-weighted branch -- two different percentile estimators
within the same script depending on weighting. The task's SETTLED
"Statistics" convention asks for ONE consistent recipe applied to both
weightings, so this module always uses the nearest-rank weighted formula,
with the volume-weighted case simply passing w=1 for every finite voxel
(mathematically the nearest-rank estimator with uniform weights, not
np.percentile's linear-interpolation estimator -- values can differ at the
sub-percent level for the same data).

Percentile levels: the SETTLED convention specifies 16th/84th (1-sigma),
which supersedes fig4_vertical_profiles/compute_data.py:49-50's PCT_LO=15,
PCT_HI=85 -- also flagged in the final report, not a silent change.

Every "mean" returned here is the plain arithmetic (weighted) mean -- never
a mean of log10(quantity). Label accordingly wherever displayed.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class WeightedStats:
    n: int
    median: float
    arithmetic_mean: float
    p16: float
    p84: float


def weighted_percentile(values: np.ndarray, weights: np.ndarray, pct: float) -> float:
    """Nearest-rank weighted percentile (pct in [0, 100])."""
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not m.any():
        return float("nan")
    v, w = v[m], w[m]
    idx = np.argsort(v)
    v, w = v[idx], w[idx]
    cw = np.cumsum(w)
    i = np.searchsorted(cw, pct / 100.0 * cw[-1], side="left")
    i = min(max(int(i), 0), v.size - 1)
    return float(v[i])


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Arithmetic weighted mean: sum(w*v)/sum(w)."""
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not m.any():
        return float("nan")
    return float(np.average(v[m], weights=w[m]))


def weighted_stats(values: np.ndarray, weights: np.ndarray) -> WeightedStats:
    """median, arithmetic mean, 16th/84th percentile, all weighted.

    Sorts once and reuses it for all three percentiles (median/p16/p84)
    instead of calling weighted_percentile() three times -- same nearest-
    rank formula and results as calling weighted_percentile() directly,
    just without re-sorting the same data three times over (this matters:
    this function runs inside a bin loop over ~150 vertical-profile bins x
    2 self-gravity settings x 3 quantities, on up to ~1.25M elements each).
    """
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    n = int(m.sum())
    if n == 0:
        return WeightedStats(n=0, median=float("nan"), arithmetic_mean=float("nan"),
                               p16=float("nan"), p84=float("nan"))
    v, w = v[m], w[m]
    idx = np.argsort(v)
    v_sorted, w_sorted = v[idx], w[idx]
    cw = np.cumsum(w_sorted)
    total = cw[-1]

    def _pct(pct):
        i = np.searchsorted(cw, pct / 100.0 * total, side="left")
        i = min(max(int(i), 0), v_sorted.size - 1)
        return float(v_sorted[i])

    return WeightedStats(
        n=n,
        median=_pct(50.0),
        arithmetic_mean=float(np.average(v, weights=w)),
        p16=_pct(16.0),
        p84=_pct(84.0),
    )


def volume_weighted_stats(values: np.ndarray, mask: np.ndarray) -> WeightedStats:
    """Volume-weighted (uniform weight=1 per finite, masked-in voxel)."""
    v = np.asarray(values, dtype=float).ravel()
    m = np.asarray(mask, dtype=bool).ravel()
    w = np.where(m, 1.0, 0.0)
    return weighted_stats(v, w)


def mass_weighted_stats(values: np.ndarray, mask: np.ndarray, n_model: np.ndarray) -> WeightedStats:
    """Mass-weighted (weight = n_model, restricted to mask)."""
    v = np.asarray(values, dtype=float).ravel()
    m = np.asarray(mask, dtype=bool).ravel()
    n_model = np.asarray(n_model, dtype=float).ravel()
    w = np.where(m & np.isfinite(n_model), n_model, 0.0)
    return weighted_stats(v, w)
