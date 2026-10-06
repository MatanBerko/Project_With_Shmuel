"""
fig_alpha_profiles -- alpha vs height, three estimators, three variants.

Reads ONLY cache/core/summary.npz (written by pipeline/compute_all.py's
finalize stage). No physics here and no src.physics import.

Layout: 2 rows (volume-weighted, mass-weighted) x 3 columns (RAW, HIM_A,
HIM_B), sharing the x axis (signed z) and ONE log y axis across all six
panels, so a reader can compare panels by eye without re-reading the
ticks. Each panel draws the three alpha estimators in that variant's
colour, separated by linestyle, with the 15-85 band on the median; a thin
guide marks alpha = 1, below which there is no non-thermal support to
measure and Mach/sigma_nt are undefined.

Why colour = variant and linestyle = estimator (and not the other way
round): the variant colours are fixed across every figure in the paper,
so they have to survive here too. The estimator encoding is a linestyle
and is explained once, in a single figure-level legend.

Self-gravity is the fiducial "mean" (footprint-averaged) everywhere;
fig_self_gravity.py is the one that varies it.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.plotting.style import (  # noqa: E402
    BAND_ALPHA,
    ESTIMATOR_LABELS,
    ESTIMATOR_ORDER,
    ESTIMATOR_STYLES,
    VARIANT_COLORS,
    VARIANT_LABELS,
    annotate,
    apply_paper_style,
    guide_line,
    panel_label,
    provisional_label,
    robust_log_ylim,
    save_figure,
)

CACHE = Path("cache/core/summary.npz")
NAME = "fig_alpha_profiles"

VARIANTS = ("RAW", "HIM_A", "HIM_B")
WEIGHTINGS = (("vol", "volume-weighted"), ("mw", "mass-weighted"))
SELF_GRAVITY = "mean"
PROFILE = "signedz"
Z_RANGE = (-400.0, 400.0)
Z_TICKS = (-400, -200, 0, 200, 400)
PANEL_LETTERS = (("a", "b", "c"), ("d", "e", "f"))


def key(variant, field):
    return f"{variant}__sg{SELF_GRAVITY}__profile_{PROFILE}__{field}"


def main():
    apply_paper_style()
    d = np.load(CACHE, allow_pickle=False)
    z = d[key("RAW", "z_pc")]

    # ---- one robust log range for ALL six panels, from everything drawn
    # in them (the band edges included -- leaving p15/p85 out would let
    # the shading spill past the axis).
    pooled = []
    for v in VARIANTS:
        for wt, _ in WEIGHTINGS:
            for est in ESTIMATOR_ORDER:
                pooled.append(d[key(v, f"{est}_{wt}")])
            pooled.append(d[key(v, f"alpha_p15_of_ratios_{wt}")])
            pooled.append(d[key(v, f"alpha_p85_of_ratios_{wt}")])
    y_lo, y_hi = robust_log_ylim(pooled, z=z, z_range=Z_RANGE)
    # alpha = 1 is drawn in every panel, so it must be inside the axis.
    y_lo, y_hi = min(y_lo, 0.9), max(y_hi, 1.1)

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.3), sharex=True, sharey=True)

    for i, (wt, wt_label) in enumerate(WEIGHTINGS):
        for j, variant in enumerate(VARIANTS):
            ax = axes[i, j]
            color = VARIANT_COLORS[variant]

            ax.fill_between(z, d[key(variant, f"alpha_p15_of_ratios_{wt}")],
                            d[key(variant, f"alpha_p85_of_ratios_{wt}")],
                            color=color, alpha=BAND_ALPHA, linewidth=0, zorder=1)
            for est in ESTIMATOR_ORDER:
                ax.plot(z, d[key(variant, f"{est}_{wt}")], color=color,
                        **ESTIMATOR_STYLES[est])

            guide_line(ax, 1.0)
            ax.set_yscale("log")
            ax.set_xlim(*Z_RANGE)
            ax.set_xticks(Z_TICKS)
            ax.set_ylim(y_lo, y_hi)

            panel_label(ax, PANEL_LETTERS[i][j])
            annotate(ax, VARIANT_LABELS[variant], loc="upper right",
                     color=color, size=8.0)
            if i == len(WEIGHTINGS) - 1:
                ax.set_xlabel("$z$  [pc]")
            if j == 0:
                ax.set_ylabel(f"$\\alpha = P_\\mathrm{{tot}}/P_\\mathrm{{th}}$\n{wt_label}")

    # alpha = 1 gets its label once, in the bottom-left panel only, so the
    # guide is explained without repeating it six times.
    annotate(axes[-1, 0], "$\\alpha = 1$", xy=(0.035, 0.055), size=6.5,
             color="0.35")

    # ---- ONE legend for the whole figure: the encoding is identical in
    # every panel, so repeating it per panel would be pure clutter.
    handles = [Line2D([], [], color="0.25", **ESTIMATOR_STYLES[est])
               for est in ESTIMATOR_ORDER]
    labels = [ESTIMATOR_LABELS[est] for est in ESTIMATOR_ORDER]
    handles.append(Patch(facecolor="0.25", alpha=BAND_ALPHA, linewidth=0))
    labels.append("15–85% of per-cell ratios")
    fig.legend(handles, labels, loc="outside upper center", ncols=4)

    annotate(axes[0, -1], f"self-gravity: {SELF_GRAVITY}", xy=(0.97, 0.055),
             size=6.5, color="0.35")

    # Lower LEFT here: with three columns the right-hand panel's x-label
    # reaches the figure's bottom-right corner, where the default sits.
    provisional_label(fig, loc="lower left")
    save_figure(fig, NAME)

    # ---- numbers for the report
    print(f"  shared y-range: {y_lo:.4g} to {y_hi:.4g} (log)")
    iz0 = int(np.argmin(np.abs(z)))
    for v in VARIANTS:
        for wt, _ in WEIGHTINGS:
            vals = [d[key(v, f'{e}_{wt}')][iz0] for e in ESTIMATOR_ORDER]
            print(f"  {v:<6} {wt:<4} at z~0: median {vals[0]:.3f}, "
                  f"mean {vals[1]:.3f}, ratio-of-means {vals[2]:.3f}")
    for v in VARIANTS:
        m = d[key(v, "alpha_median_of_ratios_vol")]
        below = np.isfinite(m) & (m < 1.0)
        print(f"  {v:<6} vol median_of_ratios below 1 in "
              f"{below.sum()}/{len(m)} bins")


if __name__ == "__main__":
    main()
