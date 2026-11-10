"""
fig_area_alpha -- the area-averaged gas weight W and thermal pressure, and
the two alphas built on <W>_area: the PI's per-cell alpha_cell (Step 2e) as
the main curve, and the ratio of area averages alpha_area (Step 2d) for
comparison.

W(z) is the hydrostatic weight of the gas above (below) z, the integral of
rho g; it equals the total midplane pressure P_tot only under vertical
dynamical equilibrium, so it is labelled W, not P_tot. The cache keys
(*__Ptot_area, ...) keep their historical names.

Reads ONLY cache/core/area_alpha.npz (written by
scripts/compute/compute_area_alpha.py). No physics here. Volume weighting
only; simulation comparisons (TIGRESS R8, GHOSDT) live in the simulation
figure, not here.

Two stacked panels on a shared signed-z axis:
  (a) <W>_area -- the plain horizontal mean of the per-column W over every
      cell -- for both self-gravity settings, and two volume-weighted means
      of the non-HIM thermal pressure: the arithmetic <P_th> and the
      HARMONIC mean <1/P_th>^-1, which alpha_cell's mean is built on.
  (b) alpha_cell = <W>_area / p_th per non-HIM cell, volume-weighted:
      median with its 15-85 percentile band (main curve), and the mean
      <alpha_cell> = <W>_area <1/P_th> dashed. alpha_area =
      <W>_area / <P_th> (ratio of means) is the thin grey line.
      Harmonic <= arithmetic, so the dashed line can never dip below it.

Encoding: COLOUR says which QUANTITY. In (a) LINESTYLE says which MEAN
(solid = arithmetic, dash-dot = harmonic); in (b) it says which STATISTIC
(solid = median, dashed = mean).

The patch scale test (Step 2d) is no longer drawn here; patches will be
their own analysis.
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

CACHE = Path("cache/core/area_alpha.npz")
NAME = "fig_area_alpha"
Z_RANGE = (-400.0, 400.0)
Z_TICKS = (-400, -200, 0, 200, 400)

SG_FIDUCIAL = "mean"
SG_OTHER = "off"

BAND_ALPHA_FILL = 0.18

# colour = quantity, linestyle = weighting (see the module docstring)
VOL_LS = "-"
HARM_LS = (0, (5.0, 1.6, 1.2, 1.6))
MEAN_LS = (0, (4.5, 2.0))
AREA_COLOR = "0.30"
A_HEADROOM = 1.8
B_HEADROOM = 1.30
SG_OFF_LS = (0, (1.2, 1.6))
ZMID_KW = dict(color="0.45", lw=0.8, ls=":", zorder=2)


def main():
    apply_paper_style()
    d = np.load(CACHE, allow_pickle=False)
    z = d["z_pc"]
    pct_lo, pct_hi = d["percentile_levels"]

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(7.0, 6.1), sharex=True)
    # rect is (left, bottom, WIDTH, HEIGHT), NOT (x0, y0, x1, y1): the
    # strip reserved at the bottom for the two-line definition note has to
    # come out of the height too, or the top panel runs off the canvas.
    fig.get_layout_engine().set(rect=(0.0, 0.058, 1.0, 0.942))

    # ------------------------------------------------------------------ (a)
    ptot_fid = d[f"{SG_FIDUCIAL}__Ptot_area"]      # <W>_area (historical key name)
    ptot_off = d[f"{SG_OTHER}__Ptot_area"]
    pth_vol = d[f"{SG_FIDUCIAL}__Pth_neutral_vol"]
    pth_harm = d[f"{SG_FIDUCIAL}__Pth_harm_vol"]
    c_ptot = PRESSURE_COLORS["Ptot"]
    c_pth = PRESSURE_COLORS["Pth"]

    ax_a.plot(z, ptot_fid, color=c_ptot, linestyle=VOL_LS, linewidth=2.0, zorder=5)
    ax_a.plot(z, ptot_off, color=c_ptot, linestyle=SG_OFF_LS, linewidth=1.1, zorder=4)
    ax_a.plot(z, pth_vol, color=c_pth, linestyle=VOL_LS, linewidth=1.5, zorder=5)
    ax_a.plot(z, pth_harm, color=c_pth, linestyle=HARM_LS, linewidth=1.5, zorder=5)
    # Step 1i: the balanced midplane (the fiducial self-gravity setting's)
    if "mean__z_mid_pc" in d.files:
        z_mid = float(d["mean__z_mid_pc"][0])
        ax_a.axvline(z_mid, ymax=0.86, **ZMID_KW)   # stops below the legend rows
        ax_a.text(z_mid - 4, 0.03, r"$z_\mathrm{mid}$", transform=ax_a.get_xaxis_transform(),
                  ha="right", va="bottom", fontsize=6.5, color="0.35")

    ax_a.set_yscale("log")
    ax_a.set_ylabel(r"$\langle W\rangle/k_\mathrm{B}$, $\langle P_\mathrm{th}\rangle/k_\mathrm{B}$  [K cm$^{-3}$]")
    lo, hi = robust_log_ylim([ptot_fid, ptot_off, pth_vol, pth_harm],
                             z=z, z_range=Z_RANGE)
    ax_a.set_ylim(lo, hi * A_HEADROOM)   # headroom for the two legend rows
    # The panel spans ~1.5 decades, where the default log locator offers
    # exactly one labelled tick (10^4). Label the half-decades by hand.
    ticks = [t for t in (200, 500, 1e3, 2e3, 5e3, 1e4, 2e4) if lo <= t <= hi * A_HEADROOM]
    ax_a.set_yticks(ticks)
    ax_a.set_yticklabels([f"{t:,.0f}".replace(",", " ") for t in ticks])
    ax_a.yaxis.set_minor_formatter(mticker.NullFormatter())
    panel_label(ax_a, "a", loc="lower left")

    # ncols=2 fills column-major: W pair | P_th pair
    handles_a = [
        Line2D([], [], color=c_ptot, linestyle=VOL_LS, linewidth=2.0),
        Line2D([], [], color=c_ptot, linestyle=SG_OFF_LS, linewidth=1.1),
        Line2D([], [], color=c_pth, linestyle=VOL_LS, linewidth=1.5),
        Line2D([], [], color=c_pth, linestyle=HARM_LS, linewidth=1.5),
    ]
    labels_a = [
        r"$\langle W\rangle_\mathrm{area}$, self-gravity on",
        r"$\langle W\rangle_\mathrm{area}$, self-gravity off",
        r"$\langle P_\mathrm{th}\rangle$",
        r"$\langle 1/P_{\rm th}\rangle^{-1}$",
    ]
    # anchored left of centre so the right column clears the PROVISIONAL tag
    ax_a.legend(handles_a, labels_a, loc="upper center",
                bbox_to_anchor=(0.46, 1.00), ncols=2, columnspacing=1.4)

    # ------------------------------------------------------------------ (b)
    a_med = d[f"{SG_FIDUCIAL}__alpha_cell_median_vol"]
    a_lo = d[f"{SG_FIDUCIAL}__alpha_cell_p_lo_vol"]
    a_hi = d[f"{SG_FIDUCIAL}__alpha_cell_p_hi_vol"]
    a_mean = d[f"{SG_FIDUCIAL}__alpha_cell_mean_vol"]
    a_area = d[f"{SG_FIDUCIAL}__alpha_area_vol"]

    ax_b.fill_between(z, a_lo, a_hi, color=COLOR_ALPHA,
                      alpha=BAND_ALPHA_FILL, linewidth=0, zorder=1)
    guide_line(ax_b, 1.0, label=r"$\alpha=1$", label_x=0.42, label_loc="below")
    ax_b.plot(z, a_area, color=AREA_COLOR, linestyle=VOL_LS, linewidth=0.9, zorder=3)
    ax_b.plot(z, a_mean, color=COLOR_ALPHA, linestyle=MEAN_LS, linewidth=1.5, zorder=4)
    ax_b.plot(z, a_med, color=COLOR_ALPHA, linestyle=VOL_LS, linewidth=2.0, zorder=5)

    ax_b.set_ylabel(r"$\alpha$")
    ax_b.set_xlabel(r"$z$  [pc]")
    # Linear: the band's upper edge sets the top.
    top = float(np.nanmax(np.concatenate([a_hi, a_mean])))
    ax_b.set_ylim(0.0, top * B_HEADROOM)   # headroom for the legend rows
    panel_label(ax_b, "b", loc="lower left")

    handles_b = [
        Line2D([], [], color=COLOR_ALPHA, linestyle=VOL_LS, linewidth=2.0),
        Patch(facecolor=COLOR_ALPHA, alpha=BAND_ALPHA_FILL, linewidth=0),
        Line2D([], [], color=COLOR_ALPHA, linestyle=MEAN_LS, linewidth=1.5),
        Line2D([], [], color=AREA_COLOR, linestyle=VOL_LS, linewidth=0.9),
    ]
    labels_b = [
        r"median $\alpha_\mathrm{cell}$",
        f"{pct_lo:g}-{pct_hi:g} pct of " + r"$\alpha_\mathrm{cell}$",
        r"$\langle\alpha_{\rm cell}\rangle = \langle W\rangle_{\rm area}"
        r"\,\langle 1/P_{\rm th}\rangle$",
        r"$\alpha_\mathrm{area} = \langle W\rangle_\mathrm{area}"
        r"/\langle P_\mathrm{th}\rangle$",
    ]
    ax_b.legend(handles_b, labels_b, loc="upper center",
                bbox_to_anchor=(0.5, 1.00), ncols=2, columnspacing=1.4)

    annotate(ax_b, "self-gravity on (fiducial), volume-weighted",
             loc="lower right", size=6.5)

    # ---------------------------------------------------------------- axes
    ax_b.set_xlim(*Z_RANGE)
    ax_b.set_xticks(Z_TICKS)

    fig.text(
        0.5, 0.032,
        r"$\langle W\rangle_\mathrm{area}$: equal-weight mean over ALL cells "
        r"in the plane, HIM-flagged included.    "
        r"$P_\mathrm{th}$ statistics: non-HIM cells only.",
        ha="center", va="bottom", fontsize=6.5, color="0.25")
    fig.text(
        0.5, 0.006,
        r"$\alpha_\mathrm{cell} = \langle W\rangle_\mathrm{area}/p_\mathrm{th}$ "
        "per non-HIM cell, with one area-averaged numerator per 4 pc bin.",
        ha="center", va="bottom", fontsize=6.5, color="0.25")
    provisional_label(fig, loc="upper right")

    save_figure(fig, NAME)


if __name__ == "__main__":
    main()
