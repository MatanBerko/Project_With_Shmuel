"""
fig_phase_alpha -- Fig. 6: the area-averaged alpha split by phase.

Reads ONLY cache/core/phase_alpha.npz (scripts/compute/compute_phase_alpha.py).
RAW, self-gravity mean (fiducial), volume-weighted, whole footprint.

Three stacked panels on a shared signed-z axis (as fig_area_alpha):
  (a) <1/P_th>_k^-1 for CNM, UNM, WNM (log y), with <P_tot>_area (red,
      long-dashed) and the all-phase <1/P_th>^-1 (thin black dash-dot).
  (b) <alpha>_k = <P_tot>_area <1/P_th>_k per phase (solid), with the 15-85
      percentile range of alpha_cell as a light strip for CNM and WNM (with
      a UNM strip too the three overlap into an unreadable orange);
      alpha = 1 guide. NaN bins (fewer than N_MIN_CELLS cells of the phase)
      are not drawn.
  (c) volume fraction of CNM, UNM, WNM and HIM as a STACKED area (the four
      sum to 1 in every bin, and stacking shows that directly; separate
      lines leave CNM, a few per cent at most, as an unreadable trace on
      the axis).

Colours: the project phase colours (src.conventions). UNM's yellow is too
light for a line on white, so UNM LINES use UNM_LINE (the same hue at 62%
brightness, #8d8252) and the original yellow is kept for its strip and its
area in (c). HIM is the light tint Fig. 2 uses (35% HIM + 65% white). WNM's
red is close to the <P_tot> red, so <P_tot> is drawn thicker, long-dashed and
labelled in-panel.

The step at z = 0 is the existing P_tot boundary convention (Step 1i, not
fixed here).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.conventions import CNM, COLOR_PTOT, HIM, UNM, WNM  # noqa: E402
from src.plotting.style import (  # noqa: E402
    annotate,
    apply_paper_style,
    guide_line,
    panel_label,
    provisional_label,
    robust_log_ylim,
    save_figure,
)

CACHE = Path("cache/core/phase_alpha.npz")
NAME = "fig_phase_alpha"
Z_RANGE = (-400.0, 400.0)
Z_TICKS = (-400, -200, 0, 200, 400)

HIM_TINT = tuple(0.35 * c + 0.65 for c in to_rgb(HIM))
UNM_LINE = tuple(0.62 * c for c in to_rgb(UNM))
LINE_COLORS = {"CNM": CNM, "UNM": UNM_LINE, "WNM": WNM}
FILL_COLORS = {"CNM": CNM, "UNM": UNM, "WNM": WNM, "HIM": HIM_TINT}
NEUTRAL = ("CNM", "UNM", "WNM")
STRIP_ALPHA = {"CNM": 0.20, "WNM": 0.18}
# Three overlapping strips blend into an unreadable orange where UNM's yellow
# sits on WNM's red; CNM and WNM (the two stable phases) keep strips, UNM is
# the line alone.
STRIP_PHASES = ("CNM", "WNM")
PTOT_KW = dict(color=COLOR_PTOT, lw=2.0, ls=(0, (6.0, 2.0)), zorder=5)
ALL_KW = dict(color="black", lw=0.8, ls=(0, (5.0, 1.5, 1.2, 1.5)), zorder=4)
LINE_LW = 1.4


def main():
    apply_paper_style()
    d = np.load(CACHE, allow_pickle=False)
    z = d["z_pc"]
    pct_lo, pct_hi = d["percentile_levels"]

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        3, 1, figsize=(7.0, 7.4), sharex=True,
        gridspec_kw=dict(height_ratios=(1.0, 1.0, 0.62)))

    # ------------------------------------------------------------------ (a)
    for k in NEUTRAL:
        ax_a.plot(z, d[f"{k}__Pth_inv_inv"], color=LINE_COLORS[k], lw=LINE_LW, zorder=3)
    ax_a.plot(z, d["Ptot_area"], **PTOT_KW)
    ax_a.plot(z, d["Pth_inv_inv_all"], **ALL_KW)
    ax_a.set_yscale("log")
    # full data range, not the robust one: the robust lower limit clips the
    # CNM curve, which is exactly the one this panel is about
    vals = np.concatenate([d[f"{k}__Pth_inv_inv"] for k in NEUTRAL] + [d["Ptot_area"]])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    ax_a.set_ylim(vals.min() / 1.25, vals.max() * 1.9)
    ax_a.set_ylabel(r"$\langle 1/P_{\rm th}\rangle_k^{-1}/k_{\rm B}$  [K cm$^{-3}$]")
    panel_label(ax_a, "a", loc="lower left")
    ax_a.legend([Line2D([], [], **PTOT_KW), Line2D([], [], **ALL_KW)],
                [r"$\langle P_{\rm tot}\rangle_{\rm area}$",
                 r"$\langle 1/P_{\rm th}\rangle^{-1}$, all neutral phases"],
                loc="upper right", ncols=2, frameon=False, handlelength=2.6)

    # ------------------------------------------------------------------ (b)
    for k in STRIP_PHASES:
        lo_k, hi_k = d[f"{k}__alpha_cell_p_lo"], d[f"{k}__alpha_cell_p_hi"]
        ok = np.isfinite(lo_k) & np.isfinite(hi_k)
        ax_b.fill_between(z, lo_k, hi_k, where=ok, color=FILL_COLORS[k],
                          alpha=STRIP_ALPHA[k], lw=0, zorder=1)
    guide_line(ax_b, 1.0, label=r"$\alpha=1$", label_x=0.42, label_loc="below")
    for k in NEUTRAL:
        ax_b.plot(z, d[f"{k}__alpha"], color=LINE_COLORS[k], lw=LINE_LW, zorder=4)
    tops = np.concatenate([d[f"{k}__alpha_cell_p_hi"] for k in STRIP_PHASES]
                          + [d[f"{k}__alpha"] for k in NEUTRAL])
    ax_b.set_ylim(0.0, float(np.nanpercentile(tops, 99.5)) * 1.12)
    ax_b.set_ylabel(r"$\langle\alpha\rangle_k = \langle P_{\rm tot}\rangle_{\rm area}"
                    r"\,\langle 1/P_{\rm th}\rangle_k$")
    panel_label(ax_b, "b", loc="upper left")
    annotate(ax_b, f"strips: {pct_lo:g}-{pct_hi:g}% of " + r"$\alpha_{\rm cell}$"
             " within CNM and WNM", loc="upper right", size=6.5)

    # ------------------------------------------------------------------ (c)
    order = ("CNM", "UNM", "WNM", "HIM")
    ax_c.stackplot(z, *[d[f"{k}__vol_frac"] for k in order],
                   colors=[FILL_COLORS[k] for k in order], lw=0)
    ax_c.set_ylim(0, 1)
    ax_c.set_yticks((0, 0.5, 1))
    ax_c.set_ylabel("volume fraction")
    panel_label(ax_c, "c", loc="upper left")

    # one legend for the phases, above (a)
    handles = [Line2D([], [], color=LINE_COLORS[k], lw=LINE_LW) for k in NEUTRAL]
    handles.append(Patch(facecolor=HIM_TINT, edgecolor="0.6", lw=0.4))
    fig.legend(handles, ["CNM", "UNM", "WNM", "HIM (c only)"], loc="outside upper center",
               ncols=4, frameon=False, handlelength=2.0, columnspacing=1.8)

    ax_c.set_xlim(*Z_RANGE)
    ax_c.set_xticks(Z_TICKS)
    ax_c.set_xlabel(r"$z$  [pc]")
    annotate(ax_a, "self-gravity on (fiducial), volume-weighted", loc="upper left", size=6.5)
    provisional_label(fig, loc="upper right")
    save_figure(fig, NAME)


if __name__ == "__main__":
    main()
