"""
fig_patch_alpha -- Step 2f: how the area-averaged pressures and alpha
scatter between random square patches of the footprint.

Reads ONLY cache/core/patch_alpha.npz (written by
scripts/compute/compute_patch_alpha.py). No physics here.

Three stacked panels on a shared signed-z axis, self-gravity on
(fiducial), volume-weighted, one figure per patch size:
  (a) <P_tot>_i        plain mean of the per-column P_tot over the patch
  (b) <1/P_th>^-1_i    over the patch's non-HIM cells
  (c) <alpha>_i        = <P_tot>_i <1/P_th>_i
Each panel: median over patches (solid), mean over patches (dashed),
15-85 percentile strip over patches, and the whole-footprint value (the
Step 2e number) as a thin black line. (c) adds the alpha = 1 guide (simulation
comparisons such as TIGRESS R8 belong only in Fig. 8).

The patches overlap and over-sample the footprint's centre (see
results/patch_alpha.txt, section 2), so the strip is a spread between
overlapping areas, not between independent samples.

Outputs figures/fig_patch_alpha.{png,pdf} (L = 500 pc, fiducial) and
figures/fig_patch_alpha_250.{png,pdf} (L = 250 pc).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.conventions import COLOR_ALPHA  # noqa: E402
from src.plotting.style import (  # noqa: E402
    PRESSURE_COLORS,
    annotate,
    apply_paper_style,
    guide_line,
    panel_label,
    provisional_label,
    robust_log_ylim,
    save_figure,
)

CACHE = Path("cache/core/patch_alpha.npz")
FIGURES = {500: "fig_patch_alpha", 250: "fig_patch_alpha_250"}
Z_RANGE = (-400.0, 400.0)
Z_TICKS = (-400, -200, 0, 200, 400)
SG = "mean"

MEDIAN_LS = "-"
MEAN_LS = (0, (4.5, 2.0))
BAND_ALPHA_FILL = 0.18
FULL_COLOR = "black"
FULL_LW = 0.8
LOG_HEADROOM_LEGEND = 2.4   # room above the data for the two legend rows
LOG_HEADROOM = 1.3
LIN_HEADROOM = 1.22
LEGEND_GREY = "0.35"
ZMID_KW = dict(color="0.45", lw=0.8, ls=":", zorder=7)


def _mark_zmid(ax, d, ymax=1.0):
    """Step 1i: the balanced midplane as a thin dotted line, labelled at the bottom."""
    if "z_mid_pc_mean" not in d.files:
        return
    z_mid = float(d["z_mid_pc_mean"][0])
    ax.axvline(z_mid, ymax=ymax, **ZMID_KW)
    ax.text(z_mid - 4, 0.03, r"$z_\mathrm{mid}$", transform=ax.get_xaxis_transform(),
            ha="right", va="bottom", fontsize=6.0, color="0.35", zorder=8)


def _strip(ax, z, d, stem, color):
    """Median, mean and 15-85 strip of one per-patch quantity."""
    ax.fill_between(z, d[stem + "__p_lo"], d[stem + "__p_hi"], color=color,
                    alpha=BAND_ALPHA_FILL, linewidth=0, zorder=1)
    ax.plot(z, d[stem + "__mean"], color=color, linestyle=MEAN_LS, linewidth=1.4, zorder=4)
    ax.plot(z, d[stem + "__median"], color=color, linestyle=MEDIAN_LS, linewidth=2.0, zorder=5)


def _log_axis(ax, arrays, z, headroom=LOG_HEADROOM):
    ax.set_yscale("log")
    lo, hi = robust_log_ylim(arrays, z=z, z_range=Z_RANGE)
    ax.set_ylim(lo, hi * headroom)
    ticks = [t for t in (200, 500, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4) if lo <= t <= hi * headroom]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:,.0f}".replace(",", " ") for t in ticks])
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def _style_legend(ax):
    """One legend for the line styles, shared by all three panels.

    Neutral grey handles: colour already says which quantity (one per
    panel), so the legend only has to say which statistic each style is.
    """
    g = LEGEND_GREY
    handles = [
        Line2D([], [], color=g, linestyle=MEDIAN_LS, linewidth=2.0),
        Patch(facecolor=g, alpha=0.30, linewidth=0),
        Line2D([], [], color=g, linestyle=MEAN_LS, linewidth=1.4),
        Line2D([], [], color=FULL_COLOR, linewidth=FULL_LW),
    ]
    labels = ["median over patches", "15-85 pct over patches",
              "mean over patches", "full 1 kpc footprint"]
    # ncols=2 fills column-major: median/strip | mean/full
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.40, 1.0),
              ncols=2, columnspacing=1.6, handlelength=2.2)


def make_figure(d, L):
    tag = f"L{L}"
    z = d["z_pc"]
    m = int(d[f"{tag}__n_cells_side"][0])
    n_patch = int(d["n_patch"][0])

    fig, (ax_a, ax_b, ax_c) = plt.subplots(3, 1, figsize=(7.0, 8.2), sharex=True)

    # (a) <P_tot>_i
    c = PRESSURE_COLORS["Ptot"]
    stem = f"{tag}__{SG}__Ptot"
    _strip(ax_a, z, d, stem, c)
    ax_a.plot(z, d[f"full__{SG}__Ptot"], color=FULL_COLOR, linewidth=FULL_LW, zorder=6)
    _mark_zmid(ax_a, d, ymax=0.88)   # stops below the style legend
    _log_axis(ax_a, [d[stem + "__p_lo"], d[stem + "__p_hi"]], z,
              headroom=LOG_HEADROOM_LEGEND)
    ax_a.set_ylabel(r"$\langle P_{\rm tot}\rangle_i/k_{\rm B}$  [K cm$^{-3}$]")
    _style_legend(ax_a)
    panel_label(ax_a, "a", loc="lower left")
    annotate(ax_a, f"{n_patch} random {L} × {L} pc patches (overlapping)",
             loc="lower right", size=6.5)

    # (b) <1/P_th>^-1_i
    c = PRESSURE_COLORS["Pth"]
    stem = f"{tag}__{SG}__Pth_inv_inv_vol"
    _strip(ax_b, z, d, stem, c)
    _mark_zmid(ax_b, d)
    ax_b.plot(z, d[f"full__{SG}__Pth_inv_inv_vol"], color=FULL_COLOR, linewidth=FULL_LW,
              zorder=6)
    _log_axis(ax_b, [d[stem + "__p_lo"], d[stem + "__p_hi"]], z)
    ax_b.set_ylabel(r"$\langle 1/P_{\rm th}\rangle_i^{-1}/k_{\rm B}$  [K cm$^{-3}$]")
    panel_label(ax_b, "b", loc="lower left")

    # (c) <alpha>_i
    c = COLOR_ALPHA
    stem = f"{tag}__{SG}__alpha_vol"
    _strip(ax_c, z, d, stem, c)
    ax_c.plot(z, d[f"full__{SG}__alpha_vol"], color=FULL_COLOR, linewidth=FULL_LW, zorder=6)
    guide_line(ax_c, 1.0, label=r"$\alpha=1$", label_x=0.42, label_loc="below")
    top = float(np.nanmax(d[stem + "__p_hi"]))
    ax_c.set_ylim(0.0, top * LIN_HEADROOM)
    ax_c.set_ylabel(r"$\langle\alpha\rangle_i = \langle P_{\rm tot}\rangle_i"
                    r"\,\langle 1/P_{\rm th}\rangle_i$")
    panel_label(ax_c, "c", loc="lower left")
    annotate(ax_c, "self-gravity on (fiducial), volume-weighted", loc="lower right", size=6.5)

    ax_c.set_xlim(*Z_RANGE)
    ax_c.set_xticks(Z_TICKS)
    ax_c.set_xlabel(r"$z$  [pc]")
    provisional_label(fig, loc="upper right")
    save_figure(fig, FIGURES[L])
    return m


def main():
    apply_paper_style()
    d = np.load(CACHE, allow_pickle=False)
    for L in FIGURES:
        make_figure(d, L)


if __name__ == "__main__":
    main()
