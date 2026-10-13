"""
fig_area_alpha -- the area-averaged pressures and alpha, the PI's
physical definition, with the sub-area scale test as a band.

Reads ONLY cache/core/area_alpha.npz (written by
scripts/compute/compute_area_alpha.py, which also carries the TIGRESS R8
reference values across from results/reference/ok22_tigress_R8.csv). No
physics here.

Two stacked panels on a shared signed-z axis:
  (a) <P_tot>_area -- the plain horizontal mean of the per-column P_tot
      over every cell -- and <P_th>_neutral, the mean thermal pressure of
      the non-HIM cells, volume- and mass-weighted. P_tot is shown for
      both self-gravity settings because it is the only curve in the
      panel that depends on the setting at all; <P_th>_neutral does not.
  (b) alpha_area = <P_tot>_area / <P_th>_neutral for both weightings,
      with the spread across 250 pc sub-patches as a light band, and the
      project's existing CELL-based mass-weighted ratio of means as a
      thin dotted line so the two definitions can be read against each
      other directly.

Encoding: COLOUR says which QUANTITY, LINESTYLE says which WEIGHTING
(solid = volume-weighted, dashed = mass-weighted). This figure cannot use
style.WEIGHT_COLORS, the project's usual volume/mass pair, because its
mass-weighted colour is byte-identical to COLOR_PTOT -- and panel (a)
draws P_tot and a mass-weighted P_th in the same axes, so those two
curves would come out the same red. One rule, learned once, covers both
panels.

The point of the figure is that (b)'s thick lines are flatter and lower
than the dotted one: averaging over the area first and taking the ratio
second is a different question from reducing per-cell ratios, and in this
cube the two answers differ by a factor of ~1.5-2.5.

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

# The patch size whose spread is drawn as the band. 250 pc is the middle
# of the three tested scales -- small enough that the spread is a real
# measurement rather than 4 numbers, large enough that most patches still
# hold neutral gas at every height.
BAND_N = 4
BAND_ALPHA_FILL = 0.15

# colour = quantity, linestyle = weighting (see the module docstring)
VOL_LS = "-"
MW_LS = (0, (4.5, 2.0))
SG_OFF_LS = (0, (1.2, 1.6))


def main():
    apply_paper_style()
    d = np.load(CACHE, allow_pickle=False)
    z = d["z_pc"]
    pct_lo, pct_hi = d["percentile_levels"]
    side = float(d["patch_sides_pc"][list(d["patch_ns"]).index(BAND_N)])

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
    c_ptot = PRESSURE_COLORS["Ptot"]
    c_pth = PRESSURE_COLORS["Pth"]

    ax_a.plot(z, ptot_fid, color=c_ptot, linestyle=VOL_LS, linewidth=2.0, zorder=5)
    ax_a.plot(z, ptot_off, color=c_ptot, linestyle=SG_OFF_LS, linewidth=1.1, zorder=4)
    ax_a.plot(z, pth_vol, color=c_pth, linestyle=VOL_LS, linewidth=1.5, zorder=5)
    ax_a.plot(z, pth_mw, color=c_pth, linestyle=MW_LS, linewidth=1.5, zorder=5)
    for q in ("P_tot_2p", "P_th_2p"):
        ax_a.plot([0.0], [float(d[f"ref_{q}"][0])], color=REFERENCE_COLOR,
                  **REFERENCE_MARKER)

    ax_a.set_yscale("log")
    ax_a.set_ylabel(r"$\langle P\rangle/k_\mathrm{B}$  [K cm$^{-3}$]")
    lo, hi = robust_log_ylim([ptot_fid, ptot_off, pth_vol, pth_mw],
                             z=z, z_range=Z_RANGE)
    hi = max(hi, float(d["ref_P_tot_2p"][0]))
    ax_a.set_ylim(lo, hi * 1.8)   # headroom for the two legend rows
    # The panel spans ~1.5 decades, where the default log locator offers
    # exactly one labelled tick (10^4). Label the half-decades by hand.
    ticks = [t for t in (1e3, 2e3, 5e3, 1e4, 2e4) if lo <= t <= hi * 1.8]
    ax_a.set_yticks(ticks)
    ax_a.set_yticklabels([f"{t:,.0f}".replace(",", " ") for t in ticks])
    ax_a.yaxis.set_minor_formatter(mticker.NullFormatter())
    panel_label(ax_a, "a", loc="lower left")

    handles_a = [
        Line2D([], [], color=c_ptot, linestyle=VOL_LS, linewidth=2.0),
        Line2D([], [], color=c_ptot, linestyle=SG_OFF_LS, linewidth=1.1),
        Line2D([], [], color=c_pth, linestyle=VOL_LS, linewidth=1.5),
        Line2D([], [], color=c_pth, linestyle=MW_LS, linewidth=1.5),
        Line2D([], [], color=REFERENCE_COLOR, **REFERENCE_MARKER),
    ]
    labels_a = [
        r"$\langle P_\mathrm{tot}\rangle_\mathrm{area}$, self-gravity on",
        r"$\langle P_\mathrm{tot}\rangle_\mathrm{area}$, self-gravity off",
        r"$\langle P_\mathrm{th}\rangle_\mathrm{neutral}$, vol-weighted",
        r"$\langle P_\mathrm{th}\rangle_\mathrm{neutral}$, mass-weighted",
        REFERENCE_LABEL,
    ]
    ax_a.legend(handles_a, labels_a, loc="upper center",
                bbox_to_anchor=(0.5, 1.00), ncols=3, columnspacing=1.4)

    # ------------------------------------------------------------------ (b)
    a_vol = d[f"{SG_FIDUCIAL}__alpha_area_vol"]
    a_mw = d[f"{SG_FIDUCIAL}__alpha_area_mw"]
    band_lo = d[f"{SG_FIDUCIAL}__patch{BAND_N}__alpha_area_vol_p_lo"]
    band_hi = d[f"{SG_FIDUCIAL}__patch{BAND_N}__alpha_area_vol_p_hi"]
    cell_rom = d[f"{SG_FIDUCIAL}__cellbased_alpha_ratio_of_means_mw"]

    ax_b.fill_between(z, band_lo, band_hi, color=COLOR_ALPHA,
                      alpha=BAND_ALPHA_FILL, linewidth=0, zorder=1)
    guide_line(ax_b, 1.0, label=r"$\alpha=1$", label_x=0.42, label_loc="below")
    ax_b.plot(z, cell_rom, color="0.35", linestyle=":", linewidth=1.1, zorder=3)
    ax_b.plot(z, a_vol, color=COLOR_ALPHA, linestyle=VOL_LS, linewidth=2.0, zorder=5)
    ax_b.plot(z, a_mw, color=COLOR_ALPHA, linestyle=MW_LS, linewidth=2.0, zorder=5)
    ax_b.plot([0.0], [float(d["ref_alpha_2p"][0])], color=REFERENCE_COLOR,
              **REFERENCE_MARKER)

    ax_b.set_ylabel(r"$\alpha_\mathrm{area}$")
    ax_b.set_xlabel(r"$z$  [pc]")
    # Linear, not log: how FLAT the thick lines are is the whole point,
    # and a log axis would flatten them further. The top is set by the
    # cell-based line, which runs well above everything else.
    top = float(np.nanmax(np.concatenate(
        [cell_rom, band_hi, [float(d["ref_alpha_2p"][0])]])))
    ax_b.set_ylim(0.0, top * 1.30)   # headroom for the two legend rows
    panel_label(ax_b, "b", loc="lower left")

    handles_b = [
        Line2D([], [], color=COLOR_ALPHA, linestyle=VOL_LS, linewidth=2.0),
        Line2D([], [], color=COLOR_ALPHA, linestyle=MW_LS, linewidth=2.0),
        Patch(facecolor=COLOR_ALPHA, alpha=BAND_ALPHA_FILL, linewidth=0),
        Line2D([], [], color="0.35", linestyle=":", linewidth=1.1),
    ]
    labels_b = [
        r"$\alpha_\mathrm{area}$, vol-weighted $\langle P_\mathrm{th}\rangle$",
        r"$\alpha_\mathrm{area}$, mass-weighted $\langle P_\mathrm{th}\rangle$",
        f"{side:.0f} pc patches, {pct_lo:g}-{pct_hi:g} pct (vol)",
        "cell-based mass-weighted ratio of means",
    ]
    ax_b.legend(handles_b, labels_b, loc="upper center",
                bbox_to_anchor=(0.5, 1.00), ncols=2, columnspacing=1.4)

    annotate(ax_b, "self-gravity on (fiducial)", loc="lower right", size=6.5)

    # ---------------------------------------------------------------- axes
    ax_b.set_xlim(*Z_RANGE)
    ax_b.set_xticks(Z_TICKS)

    fig.text(
        0.5, 0.032,
        r"$\langle P_\mathrm{tot}\rangle_\mathrm{area}$: equal-weight mean over ALL cells "
        r"in the plane, HIM-flagged included.    "
        r"$\langle P_\mathrm{th}\rangle_\mathrm{neutral}$: non-HIM cells only.",
        ha="center", va="bottom", fontsize=6.5, color="0.25")
    fig.text(
        0.5, 0.006,
        "The area average is taken first and the ratio second.    "
        "Colour = quantity, linestyle = weighting.",
        ha="center", va="bottom", fontsize=6.5, color="0.25")
    provisional_label(fig, loc="upper right")

    save_figure(fig, NAME)


if __name__ == "__main__":
    main()
