"""
fig_him_coeff -- how much of the answer rests on the HIM-flag coefficient.

The flag is p_nT = n_H T < C P_min(I_UV) with C = 0.5, and C is a choice
rather than a measurement. This figure shows what moves when C moves over
{0.1, 0.25, 0.5, 1.0}:

  (a) the flagged volume fraction per 4 pc bin -- how much of the box each
      choice of C removes from the statistics;
  (b) the mass-weighted median-of-ratios alpha profile for each C;
  (c) the same for the ratio-of-means alpha profile.

Panel (a) also carries the number the choice of C is supposed to be
protecting: the box-level fraction of the flagged cells that are really
CNM or UNM, and the exact C at which the first such cell is caught.

DESIGN script: reads ONLY cache/core/him_coeff_sensitivity.npz (written
by scripts/compute/compute_him_coeff_sensitivity.py). No physics, no cube.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.plotting.style import (  # noqa: E402
    ESTIMATOR_LABELS,
    annotate,
    apply_paper_style,
    guide_line,
    panel_label,
    provisional_label,
    robust_log_ylim,
    save_figure,
)

CACHE = Path("cache/core/him_coeff_sensitivity.npz")
NAME = "fig_him_coeff"

WEIGHTING = "mw"
# Light to dark with the fiducial the darkest line on the panel: the eye
# should land on C = 0.5 first and read the others as excursions from it.
C_COLORS = {0.1: "#7fb2d6", 0.25: "#2f7fb8", 0.5: "#12263f", 1.0: "#c0392b"}
FIDUCIAL_LW = 2.0
OTHER_LW = 1.1


def c_label(C, fiducial):
    txt = f"$C = {C:g}$"
    return txt + " (fiducial)" if np.isclose(C, fiducial) else txt


def draw_family(ax, z, values, C_grid, fiducial, legend_kw=None):
    """One line per C, fiducial emphasised. Returns nothing; the panels
    differ only in what they put on the y-axis."""
    for i, C in enumerate(C_grid):
        is_fid = np.isclose(C, fiducial)
        ax.plot(z, values[i], color=C_COLORS[float(C)],
                linewidth=FIDUCIAL_LW if is_fid else OTHER_LW,
                zorder=5 if is_fid else 3, label=c_label(C, fiducial))
    if legend_kw is not None:
        ax.legend(**legend_kw)


def main():
    apply_paper_style()
    d = np.load(CACHE, allow_pickle=True)

    z = d["z_pc"]
    C_grid = d["C_grid"]
    fiducial = float(d["C_fiducial"][0])
    bin_pc = float(d["bin_width_pc"][0])

    fig, (ax_a, ax_b, ax_c) = plt.subplots(3, 1, figsize=(7.0, 7.8), sharex=True)
    # Room at the bottom for the caption note and at the top for the
    # legends, which sit above the data rather than on it.
    fig.get_layout_engine().set(rect=(0.0, 0.030, 1.0, 0.962))

    # ---- (a) how much of the box each C removes ---------------------------
    draw_family(ax_a, z, d["flag_frac_vol"], C_grid, fiducial,
                legend_kw=dict(loc="upper center", bbox_to_anchor=(0.5, 0.995),
                               ncols=4))
    ax_a.set_ylabel("flagged fraction of volume")
    ax_a.set_ylim(0.0, 1.22)          # headroom for the legend row
    ax_a.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    panel_label(ax_a, "a")

    # The claim C = 0.5 is making, checked rather than repeated. It goes
    # in the lower right, the one corner of this panel no curve reaches.
    contam_vol = d["box_contam_dPdn__vol"]
    contam_mw = d["box_contam_dPdn__mw"]
    clean = [f"{C:g}" for i, C in enumerate(C_grid)
             if contam_vol[i] == 0.0 and contam_mw[i] == 0.0]
    dirty = [f"{100 * contam_vol[i]:.2f}% by volume ({100 * contam_mw[i]:.0f}% "
             f"by mass) at $C = {C:g}$"
             for i, C in enumerate(C_grid)
             if not (contam_vol[i] == 0.0 and contam_mw[i] == 0.0)]
    rows = ["CNM+UNM contamination of the flagged set (box, dP/dn scheme):",
            "none at $C$ = " + ", ".join(clean) + ";  " + ";  ".join(dirty),
            f"the first CNM/UNM cell is caught at "
            f"$C = {float(d['C_crit_dPdn'][0]):.3f}$"]
    annotate(ax_a, "\n".join(rows), loc="lower right", pad=0.02, size=6.2)

    # ---- (b), (c) the two alpha estimators --------------------------------
    for ax, letter, key in ((ax_b, "b", "alpha_median_of_ratios"),
                            (ax_c, "c", "alpha_ratio_of_means")):
        values = d[f"{key}__{WEIGHTING}"]
        draw_family(ax, z, values, C_grid, fiducial,
                    legend_kw=dict(loc="upper center",
                                   bbox_to_anchor=(0.5, 1.0), ncols=4))
        ax.set_yscale("log")
        lo, hi = robust_log_ylim([values], lo_pct=0.5, hi_pct=99.5)
        ax.set_ylim(lo, hi * 1.7)     # the legend sits in the top of the panel
        ax.set_ylabel("$\\alpha = P_\\mathrm{tot}/P_\\mathrm{th}$\n"
                      f"{ESTIMATOR_LABELS[key]}, mass-weighted")
        guide_line(ax, 1.0, label="$\\alpha = 1$")
        panel_label(ax, letter)

    ax_c.set_xlabel("$z$  [pc]")
    ax_c.set_xlim(z.min(), z.max())

    fig.text(0.5, 0.004,
             f"RAW, self-gravity = footprint mean, {bin_pc:g} pc bins; "
             "flagged cells excluded from the statistics but keeping their "
             "observed mass in the weight.\n"
             "$P_\\mathrm{tot}$ does not depend on $C$ under this treatment, so only which "
             "cells enter a statistic changes.",
             ha="center", va="bottom", fontsize=6.4, color="0.25")
    provisional_label(fig, loc="upper left")

    save_figure(fig, NAME)


if __name__ == "__main__":
    main()
