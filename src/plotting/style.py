"""
Shared style for the alpha paper's figures: one place for rcParams, the
variant colours, the estimator line styles, the PROVISIONAL label, the
save helper and a robust log y-range.

Every figure in scripts/design/ imports from here and nothing else that
is style-related, so "change it once, every figure follows" actually
holds. Nothing in this module reads data or computes physics.

Relationship to src/conventions.py
----------------------------------
conventions.apply_style() is the OLD, general-purpose style used by the
pre-paper exploratory scripts (pipeline/design_*.py). It is left alone.
apply_paper_style() here supersedes it for the paper figures: smaller
type, tighter layouts, and the explicit no-titles rule below. The ISM
phase colours still come from src.conventions so there is exactly one
definition of them; the VARIANT colours are new and live here.

Style rules these helpers enforce or assume
-------------------------------------------
  * No main titles and no per-axes titles. What a panel shows is said by
    an in-panel annotation (panel_label / annotate) and the axis labels.
  * Panel letters "(a)", "(b)", ... go INSIDE the panel, top-left by
    default, in a light box so they stay readable over data.
  * Compact layouts: shared axes, tick labels only on the outer edges,
    minimal whitespace. Use constrained_layout and let it do the work.
  * Legends are small, frameless, and placed ONCE per figure wherever the
    panels share the same encoding.
  * Every figure carries a small grey "PROVISIONAL (f98)" label, driven
    by the single SHOW_PROVISIONAL switch below.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src.conventions import CNM, HIM, UNM, WNM

# ---------------------------------------------------------------------------
# PROVISIONAL label -- one switch for every figure
# ---------------------------------------------------------------------------
# Every number in this project comes from the f98 cube, which is known to
# be mirrored in XY and to carry a provisional n_H / A' factor (see
# results/README.md). Until the re-oriented Porter cube lands, every
# figure says so. Set to False only when the final cube is in place.
SHOW_PROVISIONAL = True
PROVISIONAL_TEXT = "PROVISIONAL (f98)"
PROVISIONAL_COLOR = "0.55"
PROVISIONAL_SIZE = 6.0

FIGURES_DIR = Path("figures")

# ---------------------------------------------------------------------------
# Variant colours -- consistent across every figure in the paper
# ---------------------------------------------------------------------------
# Okabe-Ito, picked for colourblind safety AND for distinct greyscale
# lightness, so the figures survive both a colourblind reader and a
# black-and-white print.
VARIANT_COLORS = {
    "RAW": "#0072B2",    # blue
    "HIM_A": "#D55E00",  # vermillion
    "HIM_B": "#009E73",  # bluish green
}
VARIANT_LABELS = {
    "RAW": "RAW",
    "HIM_A": "HIM$_\\mathrm{A}$",
    "HIM_B": "HIM$_\\mathrm{B}$",
}

# Phase colours come from src.conventions -- one definition only.
PHASE_COLORS = {"CNM": CNM, "UNM": UNM, "WNM": WNM, "HIM": HIM}

# Flagged (HIM) vs non-flagged (neutral) cells, for the diagnostics figure.
FLAG_COLORS = {"flagged": HIM, "neutral": "#1a1a2e"}
FLAG_LABELS = {"flagged": "HIM-flagged", "neutral": "non-flagged"}

# Volume- vs mass-weighted, where both appear in one panel.
WEIGHT_COLORS = {"vol": "#1a1a2e", "mw": "#c0392b"}
WEIGHT_LABELS = {"vol": "volume-weighted", "mw": "mass-weighted"}

# ---------------------------------------------------------------------------
# Estimator line styles -- the three alpha estimators, same encoding
# everywhere so a reader learns it once
# ---------------------------------------------------------------------------
# median_of_ratios is the only one that gets a shaded band, because it is
# the only one with a matching pair of percentiles of the same quantity
# (p15/p85 OF THE RATIOS). A band around the mean or the ratio of means
# would be a band around a different estimator than the one drawn.
ESTIMATOR_STYLES = {
    "alpha_median_of_ratios": dict(linestyle="-", linewidth=1.5, zorder=3),
    "alpha_mean_of_ratios": dict(linestyle="--", linewidth=1.2, zorder=2),
    "alpha_ratio_of_means": dict(linestyle=":", linewidth=1.6, zorder=2),
}
# Note: matplotlib's text/mathtext does NOT interpret "\%" (that is a TeX
# escape and usetex is off here) -- it draws the backslash. Percent signs
# are written plainly throughout.
ESTIMATOR_LABELS = {
    "alpha_median_of_ratios": "median of ratios",
    "alpha_mean_of_ratios": "mean of ratios",
    "alpha_ratio_of_means": "ratio of means",
}
ESTIMATOR_ORDER = ("alpha_median_of_ratios", "alpha_mean_of_ratios",
                   "alpha_ratio_of_means")
BAND_ALPHA = 0.18

# Where a panel shows ONE variant and colour is free to encode the
# estimator instead (fig_self_gravity, where linestyle is already spent
# on self-gravity off/on). Deliberately NOT the variant palette, so a
# reader never has to wonder whether a colour means a variant or an
# estimator within one figure.
ESTIMATOR_COLORS = {
    "alpha_median_of_ratios": "#1a1a2e",   # near-black
    "alpha_mean_of_ratios": "#8c6d1f",     # ochre
    "alpha_ratio_of_means": "#c0392b",     # red
}

# Reference guides (alpha = 1, the 300/6000 K phase cuts).
GUIDE_KW = dict(color="0.45", linewidth=0.7, linestyle="-", zorder=1)
GUIDE_LABEL_SIZE = 6.0


def apply_paper_style():
    """Publication rcParams for the paper figures. Idempotent."""
    mpl.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,       # titles are not used; set for safety
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "figure.dpi": 150,            # on-screen; save_figure overrides to 300
        "savefig.dpi": 300,
        "savefig.bbox": "standard",   # constrained_layout already handles this
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "lines.linewidth": 1.2,
        "lines.solid_capstyle": "round",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "legend.handletextpad": 0.6,
        "legend.labelspacing": 0.35,
        "legend.borderaxespad": 0.3,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "figure.constrained_layout.hspace": 0.02,
        "figure.constrained_layout.wspace": 0.02,
    })


# ---------------------------------------------------------------------------
# Annotations: panel letters and in-panel text instead of titles
# ---------------------------------------------------------------------------
def panel_label(ax, letter, loc="upper left", pad=0.035, size=8.5):
    """Put "(a)" inside the panel. Titles are not used in this paper."""
    x, ha = (pad, "left") if "left" in loc else (1.0 - pad, "right")
    y, va = (1.0 - pad, "top") if "upper" in loc else (pad, "bottom")
    return ax.text(x, y, f"({letter})", transform=ax.transAxes,
                   ha=ha, va=va, fontsize=size, fontweight="bold",
                   bbox=dict(boxstyle="square,pad=0.18", facecolor="white",
                             edgecolor="none", alpha=0.75), zorder=10)


def annotate(ax, text, loc="upper right", pad=0.04, size=7.5, color="0.15",
               boxed=True, xy=None):
    """In-panel annotation -- the replacement for a per-axes title.

    `loc` places it in a corner; `xy` (axes fraction) overrides it for the
    cases where every corner is occupied by data, which happens often
    enough on a dense profile panel to be worth supporting directly
    rather than reaching past this module for ax.text.
    """
    if xy is not None:
        x, y = xy
        ha = "left" if x < 0.5 else "right"
        va = "top" if y > 0.5 else "bottom"
    else:
        x, ha = (pad, "left") if "left" in loc else (1.0 - pad, "right")
        y, va = (1.0 - pad, "top") if "upper" in loc else (pad, "bottom")
    bbox = (dict(boxstyle="square,pad=0.2", facecolor="white",
                 edgecolor="none", alpha=0.75) if boxed else None)
    return ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
                   fontsize=size, color=color, bbox=bbox, zorder=10)


def provisional_label(fig, loc="lower right"):
    """The single grey PROVISIONAL marker, placed in a figure corner.

    Figure-level (not axes-level) so it cannot land on top of data, and
    driven by SHOW_PROVISIONAL so one edit removes it everywhere.
    """
    if not SHOW_PROVISIONAL:
        return None
    x, ha = (0.995, "right") if "right" in loc else (0.005, "left")
    y, va = (0.004, "bottom") if "lower" in loc else (0.996, "top")
    return fig.text(x, y, PROVISIONAL_TEXT, ha=ha, va=va,
                    fontsize=PROVISIONAL_SIZE, color=PROVISIONAL_COLOR,
                    zorder=20)


def guide_line(ax, y, label=None, label_x=0.015, label_loc="above", **kw):
    """A thin horizontal reference line (alpha = 1, 300 K, 6000 K ...),
    optionally labelled in-panel just above or below itself."""
    kwargs = dict(GUIDE_KW)
    kwargs.update(kw)
    line = ax.axhline(y, **kwargs)
    if label is not None:
        va = "bottom" if label_loc == "above" else "top"
        offset = 1.0 if label_loc == "above" else -1.0
        ax.annotate(label, xy=(label_x, y), xycoords=("axes fraction", "data"),
                    xytext=(0, offset * 1.5), textcoords="offset points",
                    ha="left", va=va, fontsize=GUIDE_LABEL_SIZE,
                    color=kwargs["color"], zorder=9)
    return line


# ---------------------------------------------------------------------------
# Robust log y-range
# ---------------------------------------------------------------------------
def robust_log_ylim(arrays, z=None, z_range=None, lo_pct=1.0, hi_pct=99.0,
                      pad_dex=0.08, min_dex=0.4):
    """A log y-range from the 1st-99th percentile of the data ACTUALLY
    PLOTTED, within the z range actually shown.

    Percentiles rather than min/max because a single pathological bin --
    one near-empty 4 pc bin at the top of the box, say -- otherwise
    stretches the axis until the real structure is a flat line. The
    percentiles are taken over the pooled values of every array passed,
    so a shared axis across panels is driven by all of them at once.

    arrays   : iterable of 1-D arrays (the y data of every curve drawn,
               bands included -- pass the p15 and p85 arrays too, or the
               band will overflow the axis).
    z, z_range : if given, values are restricted to z within z_range
               (inclusive) before the percentiles are taken. z must be
               the common x array of every passed array.
    pad_dex  : extra padding on each side, in decades.
    min_dex  : floor on the total span, so a nearly-constant curve does
               not get a microscopic axis.

    Returns (lo, hi), both strictly positive. Returns (None, None) if
    there is nothing positive and finite to scale to, which the caller
    should pass straight to set_ylim (a no-op).
    """
    pooled = []
    for a in arrays:
        a = np.asarray(a, dtype=float).ravel()
        if z is not None and z_range is not None:
            zz = np.asarray(z, dtype=float).ravel()
            if zz.shape != a.shape:
                raise ValueError(
                    f"z has shape {zz.shape} but a passed array has {a.shape}; "
                    "robust_log_ylim restricts by z and needs them aligned.")
            a = a[(zz >= z_range[0]) & (zz <= z_range[1])]
        a = a[np.isfinite(a) & (a > 0)]
        if a.size:
            pooled.append(a)
    if not pooled:
        return None, None

    v = np.concatenate(pooled)
    lo = np.percentile(v, lo_pct)
    hi = np.percentile(v, hi_pct)
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0 or hi <= 0:
        return None, None

    log_lo, log_hi = np.log10(lo), np.log10(hi)
    if log_hi - log_lo < min_dex:
        mid = 0.5 * (log_lo + log_hi)
        log_lo, log_hi = mid - min_dex / 2, mid + min_dex / 2
    return 10.0 ** (log_lo - pad_dex), 10.0 ** (log_hi + pad_dex)


def robust_linear_ylim(arrays, z=None, z_range=None, lo_pct=1.0, hi_pct=99.0,
                         pad_frac=0.06, include_zero=False):
    """Linear-axis counterpart of robust_log_ylim (for fractions)."""
    pooled = []
    for a in arrays:
        a = np.asarray(a, dtype=float).ravel()
        if z is not None and z_range is not None:
            zz = np.asarray(z, dtype=float).ravel()
            a = a[(zz >= z_range[0]) & (zz <= z_range[1])]
        a = a[np.isfinite(a)]
        if a.size:
            pooled.append(a)
    if not pooled:
        return None, None
    v = np.concatenate(pooled)
    lo = float(np.percentile(v, lo_pct))
    hi = float(np.percentile(v, hi_pct))
    if include_zero:
        lo = min(lo, 0.0)
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0) * 0.1
    return lo - pad_frac * span, hi + pad_frac * span


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def save_figure(fig, name, directory=FIGURES_DIR, dpi=300, close=True):
    """Write figures/<name>.png (300 dpi) and figures/<name>.pdf.

    PNG for review and for the report, PDF because that is what goes into
    the manuscript. Both from the same figure object, so they cannot
    drift apart.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, kw in (("png", dict(dpi=dpi)), ("pdf", {})):
        path = directory / f"{name}.{ext}"
        fig.savefig(path, **kw)
        paths.append(path)
        print(f"  saved {path}")
    if close:
        plt.close(fig)
    return paths
