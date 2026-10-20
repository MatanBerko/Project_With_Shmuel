"""
fig_area_alpha -- the area-averaged pressures and the two alphas built on
<P_tot>_area: the PI's per-cell alpha_cell (Step 2e) as the main curve,
and the ratio of area averages alpha_area (Step 2d) for comparison.

Reads ONLY cache/core/area_alpha.npz (written by
scripts/compute/compute_area_alpha.py, which also carries the TIGRESS R8
reference values across from results/reference/ok22_tigress_R8.csv). No
physics here.

Two stacked panels on a shared signed-z axis:
  (a) <P_tot>_area -- the plain horizontal mean of the per-column P_tot
      over every cell -- for both self-gravity settings, and three means
      of the non-HIM thermal pressure: arithmetic volume- and
      mass-weighted, and the volume-weighted HARMONIC mean P_th,harm,
      which is the one alpha_cell's mean is built on.
  (b) alpha_cell = <P_tot>_area / p_th per non-HIM cell, volume-weighted:
      median with its 15-85 percentile band (main curve), and the mean
      <alpha_cell> = <P_tot>_area / P_th,harm dashed. alpha_area =
      <P_tot>_area / <P_th> (ratio of means) is the thin grey line.
      Harmonic <= arithmetic, so the dashed line can never dip below it.

Encoding: COLOUR says which QUANTITY. In (a) LINESTYLE says which MEAN
(solid = volume-weighted arithmetic, dashed = mass-weighted arithmetic,
dash-dot = volume-weighted harmonic); in (b) it says which STATISTIC
(solid = median, dashed = mean). This figure cannot use
style.WEIGHT_COLORS, the project's usual volume/mass pair, because its
mass-weighted colour is byte-identical to COLOR_PTOT -- and panel (a)
draws P_tot and a mass-weighted P_th in the same axes.

The patch scale test (Step 2d) is no longer drawn here; patches will be
their own analysis.

R8's midplane values are open markers at z = 0. R8 has about twice the
local gas surface density, so the absolute pressures in (a) are not
expected to match; alpha in (b) is a ratio and can be compared.

The step at z = 0 in every P_tot-derived curve is the existing P_tot
boundary convention, not an artefact of this figure: the two half-columns
are each integrated from their own domain edge and the positive side wins
the shared midplane point (see src/physics/hydrostatic.py).
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
    REFERENCE_COLOR,
    REFERENCE_LABEL,
    REFERENCE_MARKER,
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
MW_LS = (0, (4.5, 2.0))
HARM_LS = (0, (5.0, 1.6, 1.2, 1.6))
MEAN_LS = (0, (4.5, 2.0))
AREA_COLOR = "0.30"
A_HEADROOM = 1.8
B_HEADROOM = 1.30
SG_OFF_LS = (0, (1.2, 1.6))


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
    ptot_fid = d[f"{SG_FIDUCIAL}__Ptot_area"]
    ptot_off = d[f"{SG_OTHER}__Ptot_area"]
    pth_vol = d[f"{SG_FIDUCIAL}__Pth_neutral_vol"]
    pth_mw = d[f"{SG_FIDUCIAL}__Pth_neutral_mw"]
    pth_harm = d[f"{SG_FIDUCIAL}__Pth_harm_vol"]
    c_ptot = PRESSURE_COLORS["Ptot"]
    c_pth = PRESSURE_COLORS["Pth"]

    ax_a.plot(z, ptot_fid, color=c_ptot, linestyle=VOL_LS, linewidth=2.0, zorder=5)
    ax_a.plot(z, ptot_off, color=c_ptot, linestyle=SG_OFF_LS, linewidth=1.1, zorder=4)
    ax_a.plot(z, pth_vol, color=c_pth, linestyle=VOL_LS, linewidth=1.5, zorder=5)
    ax_a.plot(z, pth_mw, color=c_pth, linestyle=MW_LS, linewidth=1.5, zorder=5)
    ax_a.plot(z, pth_harm, color=c_pth, linestyle=HARM_LS, linewidth=1.5, zorder=5)
    for q in ("P_tot_2p", "P_th_2p"):
        ax_a.plot([0.0], [float(d[f"ref_{q}"][0])], color=REFERENCE_COLOR,
                  **REFERENCE_MARKER)

    ax_a.set_yscale("log")
    ax_a.set_ylabel(r"$\langle P\rangle/k_\mathrm{B}$  [K cm$^{-3}$]")
    lo, hi = robust_log_ylim([ptot_fid, ptot_off, pth_vol, pth_mw, pth_harm],
                             z=z, z_range=Z_RANGE)
    hi = max(hi, float(d["ref_P_tot_2p"][0]))
    ax_a.set_ylim(lo, hi * A_HEADROOM)   # headroom for the two legend rows
    # The panel spans ~1.5 decades, where the default log locator offers
    # exactly one labelled tick (10^4). Label the half-decades by hand.
    ticks = [t for t in (200, 500, 1e3, 2e3, 5e3, 1e4, 2e4) if lo <= t <= hi * A_HEADROOM]
    ax_a.set_yticks(ticks)
    ax_a.set_yticklabels([f"{t:,.0f}".replace(",", " ") for t in ticks])
    ax_a.yaxis.set_minor_formatter(mticker.NullFormatter())
    panel_label(ax_a, "a", loc="lower left")

    # ncols=3 fills column-major: P_tot pair | arithmetic P_th pair | harm + R8
    handles_a = [
        Line2D([], [], color=c_ptot, linestyle=VOL_LS, linewidth=2.0),
        Line2D([], [], color=c_ptot, linestyle=SG_OFF_LS, linewidth=1.1),
        Line2D([], [], color=c_pth, linestyle=VOL_LS, linewidth=1.5),
        Line2D([], [], color=c_pth, linestyle=MW_LS, linewidth=1.5),
        Line2D([], [], color=c_pth, linestyle=HARM_LS, linewidth=1.5),
        Line2D([], [], color=REFERENCE_COLOR, **REFERENCE_MARKER),
    ]
    labels_a = [
        r"$\langle P_\mathrm{tot}\rangle_\mathrm{area}$, self-gravity on",
        r"$\langle P_\mathrm{tot}\rangle_\mathrm{area}$, self-gravity off",
        r"$\langle P_\mathrm{th}\rangle$, vol-weighted",
        r"$\langle P_\mathrm{th}\rangle$, mass-weighted",
        r"$\langle 1/P_{\rm th}\rangle^{-1}$, vol-weighted",
        REFERENCE_LABEL,
    ]
    # anchored left of centre so the right column clears the PROVISIONAL tag
    ax_a.legend(handles_a, labels_a, loc="upper center",
                bbox_to_anchor=(0.46, 1.00), ncols=3, columnspacing=1.0)

    # ------------------------------------------------------------------ (b)
    a_med = d[f"{SG_FIDUCIAL}__alpha_cell_median_vol"]
    a_lo = d[f"{SG_FIDUCIAL}__alpha_cell_p_lo_vol"]
    a_hi = d[f"{SG_FIDUCIAL}__alpha_cell_p_hi_vol"]
    a_mean = d[f"{SG_FIDUCIAL}__alpha_cell_mean_vol"]
    a_area = d[f"{SG_FIDUCIAL}__alpha_area_vol"]
    alpha_ref = float(d["ref_alpha_2p"][0])

    ax_b.fill_between(z, a_lo, a_hi, color=COLOR_ALPHA,
                      alpha=BAND_ALPHA_FILL, linewidth=0, zorder=1)
    guide_line(ax_b, 1.0, label=r"$\alpha=1$", label_x=0.42, label_loc="below")
    ax_b.plot(z, a_area, color=AREA_COLOR, linestyle=VOL_LS, linewidth=0.9, zorder=3)
    ax_b.plot(z, a_mean, color=COLOR_ALPHA, linestyle=MEAN_LS, linewidth=1.5, zorder=4)
    ax_b.plot(z, a_med, color=COLOR_ALPHA, linestyle=VOL_LS, linewidth=2.0, zorder=5)
    ax_b.plot([0.0], [alpha_ref], color=REFERENCE_COLOR, **REFERENCE_MARKER)

    ax_b.set_ylabel(r"$\alpha$")
    ax_b.set_xlabel(r"$z$  [pc]")
    # Linear: the band's upper edge sets the top.
    top = float(np.nanmax(np.concatenate([a_hi, a_mean, [alpha_ref]])))
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
        r"$\langle\alpha_{\rm cell}\rangle = \langle P_{\rm tot}\rangle_{\rm area}"
        r"\,\langle 1/P_{\rm th}\rangle$",
        r"$\alpha_\mathrm{area} = \langle P_\mathrm{tot}\rangle_\mathrm{area}"
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
        r"$\langle P_\mathrm{tot}\rangle_\mathrm{area}$: equal-weight mean over ALL cells "
        r"in the plane, HIM-flagged included.    "
        r"$P_\mathrm{th}$ statistics: non-HIM cells only.",
        ha="center", va="bottom", fontsize=6.5, color="0.25")
    fig.text(
        0.5, 0.006,
        r"$\alpha_\mathrm{cell} = \langle P_\mathrm{tot}\rangle_\mathrm{area}/p_\mathrm{th}$ "
        "per non-HIM cell, with one area-averaged numerator per 4 pc bin.",
        ha="center", va="bottom", fontsize=6.5, color="0.25")
    provisional_label(fig, loc="upper right")

    save_figure(fig, NAME)


if __name__ == "__main__":
    main()
