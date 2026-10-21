"""
fig_high_z_tail -- DIAGNOSTIC figure (not a paper figure): the high-|z| gas
tail behind the midplane <P_tot>_area.

Reads ONLY cache/high_z_tail/high_z_tail.npz (scripts/validation/high_z_tail.py).
Three stacked panels on a shared signed z from -1000 to +1000 pc:
  (a) <n_H>_area: the cube profile (solid), the direct Edenhofer+23 query
      beyond the cube edge (C1, points) and the exponential fits to the
      400-750 pc tail (C2, dashed, each side).
  (b) median posterior SNR of the map per 20 pc |z| bin, mirrored to both
      signs (15-85% band), and the C1 query's median SNR per z level.
  (c) the cumulative share of the midplane P_tot carried by the gas beyond z:
      P_sum(z)/P(0+) for z > 0 and P_sum(z)/P(0-) for z < 0, for the Fig. 3
      profile and the C2-extended one.
Dotted verticals mark the stats box (|z| = 400) and the cube edge (|z| = 750).
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
    COLOR_GAS,
    apply_paper_style,
    guide_line,
    panel_label,
    provisional_label,
    save_figure,
)

# ═══════════════════════════════════════════════════════════════════════
# ██  CONTROL BLOCK
# ═══════════════════════════════════════════════════════════════════════
CACHE = Path("cache/high_z_tail/high_z_tail.npz")
NAME = "fig_high_z_tail"
FIG_W, FIG_H = 5.0, 5.4
Z_RANGE = (-1000.0, 1000.0)
Z_TICKS = (-1000, -750, -400, 0, 400, 750, 1000)
BOX_PC, EDGE_PC = 400.0, 750.0
C1_COLOR = "#b5651d"
C2_KW = dict(color=COLOR_GAS, lw=1.1, ls=(0, (4.0, 2.0)))
CUBE_KW = dict(color=COLOR_GAS, lw=1.4)
MARK_KW = dict(color="0.55", lw=0.6, ls=":", zorder=0)
TEXT_FS = 6.5


def mark_limits(ax):
    for zz in (-EDGE_PC, -BOX_PC, BOX_PC, EDGE_PC):
        ax.axvline(zz, **MARK_KW)


def main():
    apply_paper_style()
    d = np.load(CACHE, allow_pickle=False)
    z, nH = d["z_pc"], d["nH_area"]

    fig, (ax_a, ax_b, ax_c) = plt.subplots(3, 1, figsize=(FIG_W, FIG_H), sharex=True)

    # ------------------------------------------------------------------ (a)
    ax_a.plot(z, nH, **CUBE_KW, zorder=3)
    for side, sgn in (("pos", 1.0), ("neg", -1.0)):
        ax_a.plot(sgn * d[f"c1_{side}_z"], d[f"c1_{side}_nH"], ls="none", marker="o", ms=1.8,
                  color=C1_COLOR, zorder=4)
        n0, h = d[f"c2_n0_h_{side}"]
        zz = np.linspace(BOX_PC, -Z_RANGE[0], 200)
        ax_a.plot(sgn * zz, n0 * np.exp(-zz / h), **C2_KW, zorder=2)
    ax_a.set_yscale("log")
    ax_a.set_ylim(8e-3, 3.0)
    ax_a.set_ylabel(r"$\langle n_{\rm H}\rangle$ [cm$^{-3}$]")
    mark_limits(ax_a)
    panel_label(ax_a, "a", loc="upper left")
    ax_a.legend([Line2D([], [], **CUBE_KW),
                 Line2D([], [], ls="none", marker="o", ms=2.5, color=C1_COLOR),
                 Line2D([], [], **C2_KW)],
                ["cube", "map query (C1)", "exp. fit 400-750 (C2)"],
                # lower middle: empty under the profile's peak, clear of PROVISIONAL
                loc="lower center", frameon=False, fontsize=TEXT_FS, handlelength=1.8)

    # ------------------------------------------------------------------ (b)
    zc = d["A_abs_z_pc"]
    med, lo, hi = d["A__all__med"], d["A__all__p15"], d["A__all__p85"]
    for sgn in (1.0, -1.0):
        ax_b.fill_between(sgn * zc, lo, hi, color=COLOR_GAS, alpha=0.18, lw=0)
        ax_b.plot(sgn * zc, med, **CUBE_KW)
    for side, sgn in (("pos", 1.0), ("neg", -1.0)):
        levels = np.unique(d[f"c1_{side}_snr_z"])
        meds = [np.nanmedian(d[f"c1_{side}_snr"][d[f"c1_{side}_snr_z"] == lv]) for lv in levels]
        ax_b.plot(sgn * levels, meds, ls="none", marker="o", ms=1.8, color=C1_COLOR, zorder=4)
    guide_line(ax_b, 1.0, label="SNR = 1", label_x=0.44, label_loc="above")
    guide_line(ax_b, 2.0, label="SNR = 2", label_x=0.44, label_loc="above")
    ax_b.set_ylim(0, 3.0)
    ax_b.set_ylabel("posterior SNR")
    mark_limits(ax_b)
    panel_label(ax_b, "b", loc="upper left")
    ax_b.legend([Line2D([], [], **CUBE_KW),
                 Line2D([], [], ls="none", marker="o", ms=2.5, color=C1_COLOR)],
                ["cube cells: median, 15-85%", "map query (C1): median"],
                loc="upper right", frameon=False, fontsize=TEXT_FS, handlelength=1.8)

    # ------------------------------------------------------------------ (c)
    p0p, p0m = d["P0_fig3"]
    P = d["fig3_P"]
    pos, neg = z > 0, z < 0
    ax_c.plot(z[pos], P[pos] / p0p, **CUBE_KW)
    ax_c.plot(z[neg], P[neg] / p0m, **CUBE_KW)
    z2, P2 = d["c2_z"], d["c2_P"]
    q0p, q0m = d["P0_c2"]
    m2 = np.abs(z2) <= -Z_RANGE[0]
    ax_c.plot(z2[m2 & (z2 > 0)], P2[m2 & (z2 > 0)] / q0p, **C2_KW)
    ax_c.plot(z2[m2 & (z2 < 0)], P2[m2 & (z2 < 0)] / q0m, **C2_KW)
    ax_c.set_ylim(0, 1.05)
    ax_c.set_ylabel(r"share of $P(0^\pm)$" "\n" r"from gas beyond $z$")
    mark_limits(ax_c)
    panel_label(ax_c, "c", loc="upper left")
    ax_c.legend([Line2D([], [], **CUBE_KW), Line2D([], [], **C2_KW)],
                ["Fig. 3 profile (cube)", "C2-extended to 1500 pc"],
                loc="upper right", frameon=False, fontsize=TEXT_FS, handlelength=1.8)

    ax_c.set_xlim(*Z_RANGE)
    ax_c.set_xticks(Z_TICKS)
    ax_c.set_xlabel(r"$z$  [pc]")
    provisional_label(fig, loc="upper right")
    save_figure(fig, NAME)


if __name__ == "__main__":
    main()
