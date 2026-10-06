"""
fig_cell_selection -- does excluding the HIM-flagged cells change alpha?

Reads ONLY cache/core/summary.npz. No physics here.

Every statistic in this project has always been taken over non-flagged
cells. The Step 2a diagnostics showed the flag selects low-density WARM
gas (n_H ~ 0.03 cm^-3, T ~ 9000 K) rather than hot gas, and takes ~47% of
the volume at the midplane rising to ~97% at |z| = 400 pc -- so the
exclusion removes most of the box on a criterion that is not selecting
what its name says. Step 1f therefore computes RAW both ways; this figure
is the comparison.

Two panels (volume- and mass-weighted), RAW, self-gravity = mean.
Colour encodes the estimator and linestyle the cell selection, so each
estimator's pair sits in one colour and the shift is read directly. The
band is drawn for the all_cells median only: two nested bands per panel
would hide the six lines the figure exists to compare.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.plotting.style import (  # noqa: E402
    BAND_ALPHA,
    ESTIMATOR_COLORS,
    ESTIMATOR_LABELS,
    ESTIMATOR_ORDER,
    annotate,
    apply_paper_style,
    guide_line,
    panel_label,
    provisional_label,
    robust_log_ylim,
    save_figure,
)

CACHE = Path("cache/core/summary.npz")
NAME = "fig_cell_selection"

VARIANT = "RAW"
SELF_GRAVITY = "mean"
PROFILE = "signedz"
WEIGHTINGS = (("vol", "volume-weighted"), ("mw", "mass-weighted"))
# linestyle encodes the cell selection
SEL_STYLES = {"all_cells": dict(linestyle="-", linewidth=1.5),
              "exclude_him_flag": dict(linestyle="--", linewidth=1.2)}
SEL_LABELS = {"all_cells": "all cells",
              "exclude_him_flag": "HIM-flagged excluded (default)"}
SEL_ORDER = ("all_cells", "exclude_him_flag")
Z_RANGE = (-400.0, 400.0)
Z_TICKS = (-400, -200, 0, 200, 400)


def key(selection, field):
    return (f"{VARIANT}__sg{SELF_GRAVITY}__sel{selection}"
            f"__profile_{PROFILE}__{field}")


def main():
    apply_paper_style()
    d = np.load(CACHE, allow_pickle=False)
    z = d[key("all_cells", "z_pc")]

    # one shared log range over everything drawn, band edges included
    pooled = []
    for selection in SEL_ORDER:
        for wt, _ in WEIGHTINGS:
            for est in ESTIMATOR_ORDER:
                pooled.append(d[key(selection, f"{est}_{wt}")])
    for wt, _ in WEIGHTINGS:
        pooled.append(d[key("all_cells", f"alpha_p15_of_ratios_{wt}")])
        pooled.append(d[key("all_cells", f"alpha_p85_of_ratios_{wt}")])
    y_lo, y_hi = robust_log_ylim(pooled, z=z, z_range=Z_RANGE)
    y_lo, y_hi = min(y_lo, 0.9), max(y_hi, 1.1)

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.2), sharex=True, sharey=True)

    for i, (wt, wt_label) in enumerate(WEIGHTINGS):
        ax = axes[i]
        ax.fill_between(z, d[key("all_cells", f"alpha_p15_of_ratios_{wt}")],
                        d[key("all_cells", f"alpha_p85_of_ratios_{wt}")],
                        color=ESTIMATOR_COLORS["alpha_median_of_ratios"],
                        alpha=BAND_ALPHA, linewidth=0, zorder=1)
        for selection in SEL_ORDER:
            for est in ESTIMATOR_ORDER:
                ax.plot(z, d[key(selection, f"{est}_{wt}")],
                        color=ESTIMATOR_COLORS[est], **SEL_STYLES[selection])

        guide_line(ax, 1.0)
        ax.set_yscale("log")
        # sharey=True, so the limit is set once for both panels -- a
        # per-panel set_ylim here would just be overwritten by the next
        # one, which is how the earlier "headroom on panel (a)" silently
        # did nothing. The legend lives outside the axes instead.
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(*Z_RANGE)
        ax.set_xticks(Z_TICKS)
        ax.set_ylabel(f"$\\alpha = P_\\mathrm{{tot}}/P_\\mathrm{{th}}$\n{wt_label}")
        panel_label(ax, "ab"[i])

    axes[-1].set_xlabel("$z$  [pc]")
    annotate(axes[-1], "$\\alpha = 1$", xy=(0.035, 0.05), size=6.5, color="0.35")
    annotate(axes[-1], f"{VARIANT}, self-gravity: {SELF_GRAVITY}\n"
                       f"band: 15–85% of ratios, all cells",
             xy=(0.97, 0.95), size=6.5, color="0.35")

    # ONE figure-level legend, above the panels. The band in panel (a) is
    # wide enough to fill every corner, so an in-axes legend sits on the
    # data however it is placed. Colour = estimator (coloured handles),
    # linestyle = cell selection (grey handles), so neither encoding has
    # to be inferred from the other.
    est_handles = [Line2D([], [], color=ESTIMATOR_COLORS[e], linestyle="-",
                          linewidth=1.5) for e in ESTIMATOR_ORDER]
    sel_handles = [Line2D([], [], color="0.35", **SEL_STYLES[s]) for s in SEL_ORDER]
    fig.legend(est_handles + sel_handles,
               [ESTIMATOR_LABELS[e] for e in ESTIMATOR_ORDER]
               + [f"cells: {SEL_LABELS[s]}" for s in SEL_ORDER],
               loc="outside upper center", ncols=3)

    provisional_label(fig)
    save_figure(fig, NAME)

    # ---- numbers for the report
    print(f"  shared y-range (both panels): {y_lo:.4g} to {y_hi:.4g} (log)")
    iz0 = int(np.argmin(np.abs(z)))
    for wt, _ in WEIGHTINGS:
        for est in ESTIMATOR_ORDER:
            a = d[key("all_cells", f"{est}_{wt}")]
            e = d[key("exclude_him_flag", f"{est}_{wt}")]
            print(f"  {wt:<4} {est:<24} z~0: all {a[iz0]:.3f} vs excl {e[iz0]:.3f} "
                  f"(x{a[iz0] / e[iz0]:.3f}); box-median ratio "
                  f"x{np.nanmedian(a / e):.3f}")
    for wt, _ in WEIGHTINGS:
        a = d[key("all_cells", f"frac_alpha_lt1_{wt}")]
        e = d[key("exclude_him_flag", f"frac_alpha_lt1_{wt}")]
        print(f"  {wt:<4} frac(alpha<1): all {a[iz0]:.4f} vs excl {e[iz0]:.4f} at z~0; "
              f"profile medians {np.nanmedian(a):.4f} vs {np.nanmedian(e):.4f}")


if __name__ == "__main__":
    main()
