"""
fig_alpha_ztop -- DIAGNOSTIC: the alpha(z) profile (Fig. 4) for four
integration heights z_top.

Reads ONLY cache/alpha_ztop/alpha_ztop.npz (scripts/compute/compute_alpha_ztop.py).
Three panels sharing z (Fig. 4's orientation, 4 pc bins, |z| <= 400 pc):
  (a) <W>(z) for z_top = 400, 500, 600, 750 pc (shades of one colour,
      darkest = 750, the fiducial) and <1/P_th>^-1 (black), log y.
  (b) <alpha_cell>(z), same shades; the 15-85% band for z_top = 750 only.
  (c) the shape <alpha_cell>(z) / <alpha_cell>_mid, each z_top's own midplane value
      (the mean over |z - z_mid| <= 30 pc; cache key slab_alpha_cell_mean).
Dotted line: z_mid of the fiducial (z_top = 750); the others lie within 6 pc.
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

from src.conventions import COLOR_ALPHA  # noqa: E402
from src.plotting.style import (  # noqa: E402
    apply_paper_style,
    guide_line,
    panel_label,
    provisional_label,
    save_figure,
)

# ═══════════════════════════════════════════════════════════════════════
# ██  CONTROL BLOCK
# ═══════════════════════════════════════════════════════════════════════
CACHE = Path("cache/alpha_ztop/alpha_ztop.npz")
NAME = "fig_alpha_ztop"
FIG_W, FIG_H = 5.0, 6.4
Z_RANGE = (-400.0, 400.0)
Z_TICKS = (-400, -200, 0, 200, 400)
FIDUCIAL = 750.0
BASE_COLOR = COLOR_ALPHA
SHADE_MIX = {400.0: 0.72, 500.0: 0.50, 600.0: 0.27, 750.0: 0.0}   # fraction of white mixed in
LW = {400.0: 1.1, 500.0: 1.1, 600.0: 1.1, 750.0: 1.7}
PTH_KW = dict(color="black", lw=1.2)
BAND_ALPHA = 0.15
ZMID_KW = dict(color="0.6", lw=0.8, ls=":", zorder=2)
TEXT_FS = 6.5


def shade(zt):
    c = np.array(to_rgb(BASE_COLOR))
    return tuple(c + (1.0 - c) * SHADE_MIX[zt])


def main():
    apply_paper_style()
    d = np.load(CACHE, allow_pickle=False)
    z = d["bin_z_pc"]
    zts = [float(v) for v in d["z_tops_pc"]]
    zf = f"zt{int(FIDUCIAL)}__"
    z_mid = float(d[zf + "z_mid"][0])

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        3, 1, figsize=(FIG_W, FIG_H), sharex=True, gridspec_kw=dict(height_ratios=(1.0, 1.0, 0.8)))

    # ------------------------------------------------------------------ (a)
    for zt in zts:
        ax_a.plot(z, d[f"zt{int(zt)}__W_bin"], color=shade(zt), lw=LW[zt], zorder=3 + zt / 1000)
    ax_a.plot(z, d["Pth_harm_bin"], **PTH_KW, zorder=5)
    ax_a.set_yscale("log")
    ax_a.set_ylabel(r"$(W, P_\mathrm{th})/k_\mathrm{B}$ [K cm$^{-3}$]")
    panel_label(ax_a, "a", loc="upper left")
    ax_a.legend([Line2D([], [], color=shade(zt), lw=LW[zt]) for zt in zts] + [Line2D([], [], **PTH_KW)],
                [rf"$\langle W\rangle$, $z_\mathrm{{top}}$ = {zt:.0f}" + (" (fid.)" if zt == FIDUCIAL else "")
                 for zt in zts] + [r"$\langle 1/P_\mathrm{th}\rangle^{-1}$"],
                loc="lower center", ncols=2, frameon=False, fontsize=TEXT_FS, handlelength=1.6,
                columnspacing=1.0, borderaxespad=0.2, labelspacing=0.25)
    lo = np.nanmin(np.concatenate([d[f"zt{int(zts[0])}__W_bin"][np.abs(z - z_mid) <= 360],
                                   d["Pth_harm_bin"]]))
    hi = np.nanmax(d[zf + "W_bin"])
    ax_a.set_ylim(lo / 6.0, hi * 1.6)

    # ------------------------------------------------------------------ (b)
    ax_b.fill_between(z, d[zf + "alpha_cell_p_lo"], d[zf + "alpha_cell_p_hi"], color=shade(FIDUCIAL),
                      alpha=BAND_ALPHA, lw=0, zorder=1)
    guide_line(ax_b, 1.0, label=r"$\alpha = 1$", label_x=0.42, label_loc="below")
    for zt in zts:
        ax_b.plot(z, d[f"zt{int(zt)}__alpha_cell_mean"], color=shade(zt), lw=LW[zt], zorder=3 + zt / 1000)
    ax_b.set_ylim(0.0, float(np.ceil(np.nanpercentile(d[zf + "alpha_cell_p_hi"], 99.0))))
    ax_b.set_ylabel(r"$\langle\alpha_\mathrm{cell}\rangle$")
    panel_label(ax_b, "b", loc="upper left")
    ax_b.legend([Patch(facecolor=shade(FIDUCIAL), alpha=BAND_ALPHA, lw=0)],
                [r"15–85%, $z_\mathrm{top}$ = 750"], loc="upper right", frameon=False,
                fontsize=TEXT_FS, handlelength=1.6, borderaxespad=0.2)

    # ------------------------------------------------------------------ (c)
    guide_line(ax_c, 1.0)
    for zt in zts:
        k = f"zt{int(zt)}__"
        ax_c.plot(z, d[k + "alpha_cell_mean"] / float(d[k + "slab_alpha_cell_mean"][0]),
                  color=shade(zt), lw=LW[zt], zorder=3 + zt / 1000)
    ax_c.set_ylim(0.0, 1.25)
    ax_c.set_ylabel(r"$\langle\alpha_\mathrm{cell}\rangle / \langle\alpha_\mathrm{cell}\rangle_\mathrm{mid}$")
    panel_label(ax_c, "c", loc="upper left")

    for ax, ymin in ((ax_a, 0.25), (ax_b, 0.0), (ax_c, 0.0)):   # (a): starts above the legend
        ax.axvline(z_mid, ymin=ymin, **ZMID_KW)
    ax_c.text(z_mid - 6, 0.04, r"$z_\mathrm{mid}$", transform=ax_c.get_xaxis_transform(),
              ha="right", va="bottom", fontsize=TEXT_FS, color="0.35")
    ax_c.set_xlim(*Z_RANGE)
    ax_c.set_xticks(Z_TICKS)
    ax_c.set_xlabel(r"$z$ [pc]")
    fig.align_ylabels((ax_a, ax_b, ax_c))
    provisional_label(fig, loc="upper right")
    save_figure(fig, NAME)


if __name__ == "__main__":
    main()
