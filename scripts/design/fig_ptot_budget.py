"""
fig_ptot_budget -- Fig. 3, "Building P_tot": the 1-D weight budget.

Reads ONLY cache/ptot_budget/ptot_budget.npz (scripts/compute/compute_ptot_budget.py).
RAW, all footprint cells (HIM included), self-gravity footprint_mean, the
current Sun-centred midplane convention.

Three stacked panels on a shared signed-z axis:
  (a) <n_H>_area(z), log y, with the gas column above and below z = 0.
  (b) the enclosed column Sigma_enc(<|z|) of stars, dark matter and gas, and
      their total -- the gravity each term supplies, g = 2 pi G Sigma_enc.
  (c) <P_tot>_area(z) built up as a stack: stars, then DM, then gas
      self-gravity; its top edge is the cached <P_tot>_area (red). The z = 0
      jump (top-down 0+ vs bottom-up 0-) is the current midplane convention
      and is left visible.
Optionally (SHOW_ZMID_PREVIEW) a thin dotted line marks the balanced
midplane from the compute script's 1-D DIAGNOSTIC in (a) and (c).
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
    COLOR_DM,
    COLOR_GAS,
    COLOR_STARS,
    PRESSURE_COLORS,
    annotate,
    apply_paper_style,
    panel_label,
    provisional_label,
    save_figure,
)

# ═══════════════════════════════════════════════════════════════════════
# ██  CONTROL BLOCK
# ═══════════════════════════════════════════════════════════════════════
CACHE = Path("cache/ptot_budget/ptot_budget.npz")
NAME = "fig_ptot_budget"
FIG_W, FIG_H = 5.0, 5.2
HEIGHT_RATIOS = (1.0, 1.0, 1.4)
Z_RANGE = (-400.0, 400.0)
Z_TICKS = (-400, -200, 0, 200, 400)
SHOW_ZMID_PREVIEW = True
ZMID_KW = dict(color="0.6", lw=0.8, ls=":", zorder=2)
ZMID_LABEL = r"balanced $z_\mathrm{mid}$ (preview)"
COLORS = {"star": COLOR_STARS, "dm": COLOR_DM, "gas": COLOR_GAS}
LABELS = {"star": "stars", "dm": "dark matter", "gas": "gas self-gravity"}
TOTAL_COLOR = "0.1"
PTOT_COLOR = PRESSURE_COLORS["Ptot"]
TEXT_FS = 6.5


def _round100(v):
    return f"{int(round(v / 100.0) * 100):,}".replace(",", r"\,")


def main():
    apply_paper_style()
    d = np.load(CACHE, allow_pickle=False)
    z = d["z_pc"]
    box = (z >= Z_RANGE[0]) & (z <= Z_RANGE[1])
    zmid = float(d["z_mid_diag_on"][0])

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        3, 1, figsize=(FIG_W, FIG_H), sharex=True,
        gridspec_kw=dict(height_ratios=HEIGHT_RATIOS))

    # ------------------------------------------------------------------ (a)
    ax_a.plot(z, d["nH_area"], color=COLOR_GAS, lw=1.3, zorder=3)
    ax_a.set_yscale("log")
    nh = d["nH_area"][box]
    ax_a.set_ylim(nh.min() / 1.3, nh.max() * 3.0)
    ax_a.set_ylabel(r"$\langle n_{\rm H}\rangle$ [cm$^{-3}$]")
    panel_label(ax_a, "a", loc="upper left")
    annotate(ax_a, r"$\Sigma_{\rm above}$ = " f"{float(d['sigma_above'][0]):.2f}, "
             r"$\Sigma_{\rm below}$ = " f"{float(d['sigma_below'][0]):.2f} "
             r"M$_\odot$ pc$^{-2}$", xy=(0.97, 0.84), size=TEXT_FS)   # below PROVISIONAL

    # ------------------------------------------------------------------ (b)
    for k in ("star", "dm", "gas"):
        ax_b.plot(z, d[f"sigma_enc_{k}"], color=COLORS[k], lw=1.3, zorder=3)
    ax_b.plot(z, d["sigma_enc_total"], color=TOTAL_COLOR, lw=2.0, zorder=4)
    # headroom: the total reaches its maximum at the panel edges, under the
    # panel letter and the legend row
    ax_b.set_ylim(0, float(d["sigma_enc_total"][box].max()) * 1.38)
    ax_b.set_ylabel(r"$\Sigma_{\rm enc}(<|z|)$ [M$_\odot$ pc$^{-2}$]")
    panel_label(ax_b, "b", loc="upper left")
    handles_b = [Line2D([], [], color=COLORS[k], lw=1.3) for k in ("star", "dm", "gas")]
    handles_b.append(Line2D([], [], color=TOTAL_COLOR, lw=2.0))
    ax_b.legend(handles_b, ["stars", "dark matter", "gas", "total"], loc="upper center",
                ncols=4, frameon=False, handlelength=1.6, columnspacing=1.2,
                fontsize=TEXT_FS, borderaxespad=0.2)

    # ------------------------------------------------------------------ (c)
    p1 = d["P_star"]
    p2 = p1 + d["P_dm"]
    p3 = p2 + d["P_gas"]
    ax_c.fill_between(z, 0, p1, color=COLOR_STARS, lw=0, zorder=1)
    ax_c.fill_between(z, p1, p2, color=COLOR_DM, lw=0, zorder=1)
    ax_c.fill_between(z, p2, p3, color=COLOR_GAS, lw=0, zorder=1)
    ax_c.plot(z, d["ptot_area_cached_on"], color=PTOT_COLOR, lw=1.4, zorder=4)
    ax_c.set_ylim(0, float(p3[box].max()) * 1.28)
    ax_c.set_ylabel(r"$\langle P_{\rm tot}\rangle/k_{\rm B}$ [K cm$^{-3}$]")
    panel_label(ax_c, "c", loc="upper left")
    p0p, p0m = float(d["P0_total"][0]), float(d["P0_total"][1])
    ax_c.annotate(rf"$\approx {_round100(p0p)}$ ($0^+$), ${_round100(p0m)}$ ($0^-$)",
                  (0.0, p0m), xytext=(6, 4), textcoords="offset points",
                  ha="left", va="bottom", fontsize=TEXT_FS, color=PTOT_COLOR, zorder=6)
    sh = {k: float(d[f"P0_{k}"][0]) / p0p for k in ("star", "dm", "gas")}
    annotate(ax_c, f"at $0^+$: stars {sh['star']:.0%}, DM {sh['dm']:.0%}, "
             f"gas {sh['gas']:.0%}", loc="upper right", size=TEXT_FS)
    handles_c = [Patch(facecolor=COLORS[k], lw=0) for k in ("star", "dm", "gas")]
    handles_c.append(Line2D([], [], color=PTOT_COLOR, lw=1.4))
    ax_c.legend(handles_c, ["stars", "dark matter", "gas self-gravity",
                            r"$\langle P_{\rm tot}\rangle_{\rm area}$"],
                loc="lower left", bbox_to_anchor=(0.0, 0.62), ncols=2, frameon=False,
                handlelength=1.4, columnspacing=1.0, fontsize=TEXT_FS, borderaxespad=0.3)

    # ---------------------------------------------------- z_mid preview
    if SHOW_ZMID_PREVIEW:
        for ax in (ax_a, ax_c):
            ax.axvline(zmid, **ZMID_KW)
        ax_c.text(zmid - 4, 0.03, ZMID_LABEL, transform=ax_c.get_xaxis_transform(),
                  ha="right", va="bottom", fontsize=TEXT_FS - 0.5, color="0.25")

    ax_c.set_xlim(*Z_RANGE)
    ax_c.set_xticks(Z_TICKS)
    ax_c.set_xlabel(r"$z$  [pc]")
    provisional_label(fig, loc="upper right")
    save_figure(fig, NAME)


if __name__ == "__main__":
    main()
