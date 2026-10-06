"""
fig_clean_columns -- what survives the "HIM-free columns" cut, and what
alpha is in what survives.

Reads ONLY cache/core/clean_columns.npz (written by
scripts/compute/compute_clean_columns.py). No physics here, no 3D cube,
and no orientation logic: the re-oriented map and its true-Galactic axes
are computed and verified in the compute script and come out of the cache
ready to draw.

  (a) the fraction of (x, y) columns that stay HIM-free up to Z_clean,
      strictly and under two looser "HIM volume fraction below 5% / 10%"
      versions. This is the survival curve for the PI's proposal.
  (b) the z_clean map -- how high each sightline stays clean -- shown in
      TRUE Galactic orientation, with the Sun at the origin. The cube is
      mirrored in XY, so this is the cube map re-indexed through the
      mapping measured by check_orientation.py.
  (c) alpha in the surviving columns against alpha over all columns
      (HIM cells excluded) over the same |z| range, mass-weighted.

This is the one figure in the set that resolves the XY plane, which is
only legitimate because the map is drawn in the measured true orientation
rather than the cube's own.
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
    annotate,
    apply_paper_style,
    guide_line,
    panel_label,
    provisional_label,
    robust_log_ylim,
    save_figure,
)

CACHE = Path("cache/core/clean_columns.npz")
NAME = "fig_clean_columns"

ESTIMATORS = ("alpha_median_of_ratios", "alpha_ratio_of_means")
WEIGHTING = "mw"
POP_STYLES = {"clean": dict(linestyle="-", marker="o", markersize=3.5, linewidth=1.5),
              "all": dict(linestyle="--", marker="s", markersize=3.2, linewidth=1.2)}
POP_LABELS = {"clean": "clean columns only", "all": "all columns (HIM cells excluded)"}
STRICT_COLOR = "#1a1a2e"
LOOSE_COLORS = ("#c0392b", "#8c6d1f")
MAP_CMAP = "viridis"


def main():
    apply_paper_style()
    d = np.load(CACHE, allow_pickle=False)
    Z_grid = d["Z_clean_grid_pc"]
    Z_alpha = d["Z_clean_alpha_pc"]
    pct_lo, pct_hi = d["percentile_levels"]
    thresholds = d["loose_thresholds"]
    n_cols_total = int(d["n_columns_total"][0])

    # (b) is a square map with aspect="equal", so the right column is made
    # wider than the left: with equal widths the map fills its slot
    # horizontally and leaves a band of dead space below it.
    fig = plt.figure(figsize=(7.4, 4.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.32])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[1, 0])

    # ---------------------------------------------------------------- (a)
    def no_zeros(y):
        """Exact zeros masked out. A log axis cannot draw them, and
        leaving them in produces a vertical line plunging to the axis
        floor that reads as "very small" rather than "none at all" --
        which is the opposite of the point. The curves simply stop, and
        the annotation says where they reach zero."""
        y = np.asarray(y, dtype=float)
        return np.where(y > 0, y, np.nan)

    ax_a.plot(Z_grid, no_zeros(d["frac_clean_strict"]), color=STRICT_COLOR,
              linestyle="-", marker="o", markersize=3.5, linewidth=1.5,
              label="no HIM at all")
    for thr, col in zip(thresholds, LOOSE_COLORS):
        ax_a.plot(Z_grid, no_zeros(d[f"frac_clean_lt{int(thr * 100)}pct"]), color=col,
                  linestyle="--", marker="^", markersize=3.2, linewidth=1.2,
                  label=f"HIM fraction below {thr:.0%}")
    ax_a.set_yscale("log")
    # Floor below 1/N so that EVERY nonzero point is on the panel: the
    # "below 5%" curve's last point is a single column out of 251,001
    # (4e-6), which a tighter floor pushed off the axis while still
    # drawing the line towards it -- visually identical to reaching zero,
    # which it does not. The 1-column guide marks the resolution limit.
    one_column = 1.0 / n_cols_total
    ax_a.set_ylim(one_column / 2.5, 2.0)
    guide_line(ax_a, one_column, label="1 column", label_loc="above", label_x=0.80)
    ax_a.set_xlim(-10, 410)
    ax_a.set_xticks([0, 100, 200, 300, 400])
    ax_a.set_xlabel("$Z_\\mathrm{clean}$  [pc]")
    ax_a.set_ylabel("fraction of columns\nclean to $Z_\\mathrm{clean}$")
    ax_a.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02))
    panel_label(ax_a, "a", loc="upper left")
    # A log axis cannot show an exact zero, and the strict curve really
    # does hit zero -- say so, so the vertical drop is not misread as
    # "small but nonzero".
    annotate(ax_a, f"{n_cols_total:,} columns\nno column is clean\nabove "
                   f"$Z_\\mathrm{{clean}} = 274$ pc",
             xy=(0.97, 0.93), size=6.5, color="0.35")

    # ---------------------------------------------------------------- (b)
    z_true = d["z_clean_true"]
    x_true = d["x_true_pc"]
    y_true = d["y_true_pc"]
    extent = (x_true.min(), x_true.max(), y_true.min(), y_true.max())
    im = ax_b.imshow(z_true, origin="lower", extent=extent, cmap=MAP_CMAP,
                     aspect="equal", interpolation="nearest", vmin=0.0,
                     vmax=float(np.nanmax(z_true)))
    cb = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.02)
    cb.set_label("$z_\\mathrm{clean}$  [pc]", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    # the Sun sits at the origin of this heliocentric frame
    ax_b.plot(0, 0, marker="$\\odot$", color="w", markersize=7,
              markeredgewidth=0.0, linestyle="none", zorder=5)
    ax_b.set_xlabel("$x_\\mathrm{true}$  [pc]   ($l = 0$)")
    ax_b.set_ylabel("$y_\\mathrm{true}$  [pc]   ($l = 90^\\circ$)")
    ax_b.set_xticks([-500, -250, 0, 250, 500])
    ax_b.set_yticks([-500, -250, 0, 250, 500])
    panel_label(ax_b, "b")
    annotate(ax_b, f"true Galactic orientation\nmapping {str(d['orientation_label'][0])}",
             loc="lower left", size=6.5)

    # ---------------------------------------------------------------- (c)
    pooled = []
    for pop in ("clean", "all"):
        for est in ESTIMATORS:
            y = d[f"{est}__{pop}__{WEIGHTING}"]
            ax_c.plot(Z_alpha, y, color=ESTIMATOR_COLORS[est], **POP_STYLES[pop])
            pooled.append(y)
    lo = d[f"alpha_p15_of_ratios__clean__{WEIGHTING}"]
    hi = d[f"alpha_p85_of_ratios__clean__{WEIGHTING}"]
    ax_c.fill_between(Z_alpha, lo, hi,
                      color=ESTIMATOR_COLORS["alpha_median_of_ratios"],
                      alpha=BAND_ALPHA, linewidth=0, zorder=1)
    pooled += [lo, hi]

    y_lo, y_hi = robust_log_ylim(pooled)
    y_lo, y_hi = min(y_lo, 0.95), max(y_hi, 1.05)
    ax_c.set_yscale("log")
    ax_c.set_ylim(y_lo, y_hi)
    ax_c.set_xlim(25, 225)
    ax_c.set_xticks([50, 100, 150, 200])
    # Guide only, unlabelled: every curve here sits well above 1, and the
    # label landed on the four-row legend in the same corner.
    guide_line(ax_c, 1.0)
    ax_c.set_xlabel("$Z_\\mathrm{clean}$  [pc]   (cells with $|z| \\leq Z_\\mathrm{clean}$)")
    ax_c.set_ylabel("$\\alpha$, mass-weighted")
    panel_label(ax_c, "c")
    annotate(ax_c, f"band: {pct_lo:g}–{pct_hi:g}% of ratios,\nclean columns",
             loc="upper right", size=6.5, color="0.35")

    handles = [Line2D([], [], color=ESTIMATOR_COLORS[e], linestyle="-", linewidth=1.5)
               for e in ESTIMATORS]
    handles += [Line2D([], [], color="0.35", **POP_STYLES[p]) for p in ("clean", "all")]
    labels = [ESTIMATOR_LABELS[e] for e in ESTIMATORS] + \
             [POP_LABELS[p] for p in ("clean", "all")]
    ax_c.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncols=1)

    provisional_label(fig)
    save_figure(fig, NAME)

    # ---- numbers for the report
    print(f"  (a) y-range {ax_a.get_ylim()}, (c) y-range "
          f"{ax_c.get_ylim()[0]:.4g} to {ax_c.get_ylim()[1]:.4g}")
    print(f"  (b) z_clean map: {z_true.shape}, range {z_true.min():.0f}-{z_true.max():.0f} pc, "
          f"median {np.median(z_true):.0f} pc, mapping {str(d['orientation_label'][0])}")
    for i, Z in enumerate(Z_grid):
        print(f"      Z={Z:>5.0f}: strict {d['frac_clean_strict'][i]:.5f}, "
              + ", ".join(f"<{t:.0%} {d[f'frac_clean_lt{int(t * 100)}pct'][i]:.5f}"
                          for t in thresholds))
    for i, Z in enumerate(Z_alpha):
        for est in ESTIMATORS:
            c = d[f"{est}__clean__{WEIGHTING}"][i]
            a = d[f"{est}__all__{WEIGHTING}"][i]
            print(f"      Z={Z:>5.0f} {est:<24} clean {c:.4f} vs all {a:.4f} "
                  f"(x{c / a:.3f})")


if __name__ == "__main__":
    main()
