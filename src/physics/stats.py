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

Three alpha estimators (Step 1e)
--------------------------------
The paper reports alpha three ways side by side, because they answer
different questions and differ substantially in a medium this
inhomogeneous:

    alpha_median_of_ratios = weighted median of the per-cell alpha_i
    alpha_mean_of_ratios   = weighted mean   of the per-cell alpha_i
    alpha_ratio_of_means   = sum(w_i P_tot,i) / sum(w_i p_th_phys,i)

The first two come from weighted_stats() applied to a precomputed alpha
cube; the third is ratio_of_means() below and canNOT be recovered from
alpha_i alone with w_i -- it is the w_i*p_th-weighted mean of alpha_i, a
different weighting (see that function's docstring). All three are
computed over exactly the same cell selection.

Which one to prefer is a physics question, not a statistics one, so all
three are reported rather than one being chosen here:
  * ratio_of_means is the only one that answers "what is the total
    non-thermal support of this volume" -- it is the ratio of the
    volume's total P_tot to its total P_th, so it conserves pressure.
    It is dominated by the highest-pressure cells.
  * median_of_ratios describes a typical cell and is insensitive to the
    tails.
  * mean_of_ratios sits between them and is pulled up by the low-p_th
    (high-alpha) tail, which in this cube is large.
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


def ratio_of_means(p_tot: np.ndarray, p_th: np.ndarray, weights: np.ndarray) -> float:
    """alpha_ratio_of_means = sum(w_i * P_tot,i) / sum(w_i * p_th,i).

    The third alpha estimator. Unlike the other two it is NOT a statistic
    of the per-cell alpha_i at all -- it is the ratio of two weighted
    sums, i.e. the ratio of the selection's total P_tot to its total
    P_th. Equivalently (and this is the identity the tests pin down) it
    is the (w_i * p_th,i)-weighted mean of alpha_i:

        sum(w p_tot) / sum(w p_th) = sum(w p_th * alpha) / sum(w p_th)

    so it weights each cell by its own thermal pressure on top of w_i.
    That is why it cannot be reproduced by calling weighted_mean() on an
    alpha cube with w_i, and why it is the estimator that conserves
    pressure over the volume.

    Cell selection: a cell contributes only if p_tot, p_th and w are all
    finite, w > 0 and p_th > 0. Cells are selected on the SAME criteria
    the other two estimators use (finite value, positive weight), with
    the extra p_th > 0 guard that the alpha cube already has baked in via
    src.physics.derived.alpha's P_FLOOR -- so all three estimators see
    the same population.

    Returns NaN if nothing is selected. Both pressures must be in the
    same units (this project: P/k_B in K cm^-3).
    """
    pt = np.asarray(p_tot, dtype=float).ravel()
    pth = np.asarray(p_th, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    m = (np.isfinite(pt) & np.isfinite(pth) & np.isfinite(w)
         & (w > 0) & (pth > 0))
    if not m.any():
        return float("nan")
    denom = float(np.sum(w[m] * pth[m]))
    if denom <= 0:
        return float("nan")
    return float(np.sum(w[m] * pt[m]) / denom)


def volume_weighted_ratio_of_means(p_tot: np.ndarray, p_th: np.ndarray,
                                     mask: np.ndarray) -> float:
    """ratio_of_means with w_i = 1 on the masked-in cells."""
    m = np.asarray(mask, dtype=bool).ravel()
    return ratio_of_means(p_tot, p_th, np.where(m, 1.0, 0.0))


def mass_weighted_ratio_of_means(p_tot: np.ndarray, p_th: np.ndarray,
                                   mask: np.ndarray, n_model: np.ndarray) -> float:
    """ratio_of_means with w_i = n_model on the masked-in cells."""
    m = np.asarray(mask, dtype=bool).ravel()
    n = np.asarray(n_model, dtype=float).ravel()
    return ratio_of_means(p_tot, p_th, np.where(m & np.isfinite(n), n, 0.0))


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
