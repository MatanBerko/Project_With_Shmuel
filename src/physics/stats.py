"""
Weighted statistics: median, arithmetic mean, and a low/high percentile
pair, both volume-weighted (w=1 per finite voxel) and mass-weighted
(w=n_model).

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

Percentile levels (PERCENTILE_SCHEME in src.conventions): Shelest et al.
2026 report 15th/85th, which is now the DEFAULT ("15_85"). "16_84" (the
nominal 1-sigma pair used through Phase B) stays available behind the
switch. Note that 15/85 is also what this project's own
fig4_vertical_profiles/compute_data.py:49-50 used (PCT_LO=15, PCT_HI=85)
before Phase B moved to 16/84 -- so the default now agrees with both
Shelest+26 and the original reference script.

Every WeightedStats carries the levels it was computed at (pct_lo/pct_hi)
alongside the values (p_lo/p_hi), so a stored or printed number can never
be mistaken for the other convention.

Every "mean" returned here is the plain arithmetic (weighted) mean -- never
a mean of log10(quantity). Label accordingly wherever displayed.
"""

from dataclasses import dataclass

import numpy as np

from src.conventions import PERCENTILE_LEVELS_BY_SCHEME, PERCENTILE_SCHEME_DEFAULT


def percentile_levels(scheme: str = PERCENTILE_SCHEME_DEFAULT) -> tuple[float, float]:
    """(low, high) percentile levels for a PERCENTILE_SCHEME name."""
    try:
        return PERCENTILE_LEVELS_BY_SCHEME[scheme]
    except KeyError:
        raise ValueError(
            f"Unknown percentile scheme: {scheme!r}. Expected one of "
            f"{tuple(PERCENTILE_LEVELS_BY_SCHEME)}."
        ) from None


@dataclass
class WeightedStats:
    n: int
    median: float
    arithmetic_mean: float
    p_lo: float
    p_hi: float
    pct_lo: float
    pct_hi: float

    @property
    def lo_label(self) -> str:
        """e.g. "p15" -- the stat name to store/print this value under."""
        return f"p{self.pct_lo:g}"

    @property
    def hi_label(self) -> str:
        return f"p{self.pct_hi:g}"


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


def weighted_stats(values: np.ndarray, weights: np.ndarray,
                     scheme: str = PERCENTILE_SCHEME_DEFAULT) -> WeightedStats:
    """median, arithmetic mean, and the scheme's low/high percentile pair,
    all weighted.

    Sorts once and reuses it for all three percentiles (median/lo/hi)
    instead of calling weighted_percentile() three times -- same nearest-
    rank formula and results as calling weighted_percentile() directly,
    just without re-sorting the same data three times over (this matters:
    this function runs inside a loop over ~400 vertical-profile planes x
    2 self-gravity settings x 3 quantities, on up to ~250k elements each).
    """
    pct_lo, pct_hi = percentile_levels(scheme)
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    n = int(m.sum())
    if n == 0:
        return WeightedStats(n=0, median=float("nan"), arithmetic_mean=float("nan"),
                               p_lo=float("nan"), p_hi=float("nan"),
                               pct_lo=pct_lo, pct_hi=pct_hi)
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
        p_lo=_pct(pct_lo),
        p_hi=_pct(pct_hi),
        pct_lo=pct_lo,
        pct_hi=pct_hi,
    )


def volume_weighted_stats(values: np.ndarray, mask: np.ndarray,
                            scheme: str = PERCENTILE_SCHEME_DEFAULT) -> WeightedStats:
    """Volume-weighted (uniform weight=1 per finite, masked-in voxel)."""
    v = np.asarray(values, dtype=float).ravel()
    m = np.asarray(mask, dtype=bool).ravel()
    w = np.where(m, 1.0, 0.0)
    return weighted_stats(v, w, scheme=scheme)


def mass_weighted_stats(values: np.ndarray, mask: np.ndarray, n_model: np.ndarray,
                          scheme: str = PERCENTILE_SCHEME_DEFAULT) -> WeightedStats:
    """Mass-weighted (weight = n_model, restricted to mask)."""
    v = np.asarray(values, dtype=float).ravel()
    m = np.asarray(mask, dtype=bool).ravel()
    n_model = np.asarray(n_model, dtype=float).ravel()
    w = np.where(m & np.isfinite(n_model), n_model, 0.0)
    return weighted_stats(v, w, scheme=scheme)
