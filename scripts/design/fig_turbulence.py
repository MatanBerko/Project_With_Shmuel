"""
fig_turbulence -- Fig. 6: turbulence two ways.

Reads ONLY cache/velocity/velocity.npz (scripts/compute/compute_velocity.py)
and cache/turbulence_pdf/turbulence_pdf.npz
(scripts/compute/compute_turbulence_pdf.py). RAW, self-gravity mean
(fiducial), whole footprint, 4 pc bins of signed z. The PDF method uses the
native 2 pc cells (sigma_s).

2 x 2 panels, one phase per column, in its phase colour:
  columns  WNM (left, full z range) | CNM (right, z_mid +- CNM_HALF_RANGE_PC)
  rows     top = 1D velocities [km/s] | bottom = M^2 (1D); linear axes, own
           y range per panel, x shared within each column
In every panel:
  solid          from the weight (alpha): sigma_nt (top), <alpha_cell - 1> (bottom)
  very light fill  its mass-weighted 15-85% range of the per-cell quantity
  dashed         from the density-PDF width, b = 0.4: sigma_turb (top), M2_pdf (bottom)
  hatch          the PDF method over b = 1 .. 1/3
  dotted (top)   the sound speed c_s = sqrt(sum P_th / sum rho)

Phase colours, the z_mid line and the visibility rule (contiguous run from
z_mid with >= N_MIN_CELLS cells and >= 1% of the bin volume) are
fig_velocity's, imported from it.
"""

import importlib.util
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
    apply_paper_style,
    guide_line,
    panel_label,
    provisional_label,
    robust_linear_ylim,
    save_figure,
)

_SPEC = importlib.util.spec_from_file_location(
    "fig_velocity", Path(__file__).resolve().parent / "fig_velocity.py")
fv = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fv)

# ═══════════════════════════════════════════════════════════════════════
# ██  CONTROL BLOCK
# ═══════════════════════════════════════════════════════════════════════
VEL_CACHE = fv.CACHE
PDF_CACHE = Path("cache/turbulence_pdf/turbulence_pdf.npz")
NAME = "fig_turbulence"
FIG_W, FIG_H = 6.6, 4.6
COLUMNS = ("WNM", "CNM")
WIDTH_RATIOS = (2.5, 1.0)
WNM_Z_RANGE, WNM_Z_TICKS = fv.Z_RANGE, fv.Z_TICKS
CNM_HALF_RANGE_PC = 100.0             # CNM column: z_mid +- this
CNM_TICK_STEP_PC = 50.0
PHASE_COLORS = fv.PHASE_COLORS
LINE_LW = fv.LINE_LW
ZMID_KW, ZMID_LABEL = fv.ZMID_KW, fv.ZMID_LABEL
TEXT_FS = fv.TEXT_FS
FILL_ALPHA = 0.12                     # alpha method's 15-85% range: very light, no edge
PDF_LS = (0, (4.0, 2.0))              # dashed: the density-PDF method
CS_LS = (0, (1.0, 1.4))               # dotted: c_s
CS_LW = 0.9
B_LINE, B_LO, B_HI = "b0p4", "b1", "b1_3"   # line and hatched band (b = 1 gives the lowest M^2)
HATCH = "////"
HATCH_ALPHA = 0.55
HATCH_LW = 0.5
TOP_HEADROOM = 1.12                   # top row: room for the phase name above the data
LEGEND_COLOR = "0.3"
METHOD_LABELS = ("from the weight (alpha)", r"from the density-PDF width ($b = 0.4$)",
                 r"sound speed $c_\mathrm{s}$", "15–85%, mass-weighted", r"$b = 1/3$ – $1$")
# per row: (alpha mean, alpha band source/key, PDF key, y label, with c_s)
ROWS = (
    (("vel", "sigma_nt"), ("pdf", "sigma_nt_cell"), "sigma_turb_pdf",
     r"$\sigma_\mathrm{nt}$, $c_\mathrm{s}$ [km s$^{-1}$]", True),
    (("vel", "M2"), ("vel", "M2_cell"), "M2_pdf", r"$\mathcal{M}^2$ (1D)", False),
)


def main():
    apply_paper_style()
    plt.rcParams["hatch.linewidth"] = HATCH_LW
    d = {"vel": np.load(VEL_CACHE, allow_pickle=False),
         "pdf": np.load(PDF_CACHE, allow_pickle=False)}
    z = d["vel"]["z_pc"]
    if not np.array_equal(z, d["pdf"]["z_pc"]):
        raise RuntimeError("the two caches have different bins")
    z_mid = float(d["vel"]["z_mid_pc"][0])
    keep = {k: fv.drawn_run(d["vel"], z, z_mid, k) for k in COLUMNS}
    for k in keep:
        if not np.array_equal(keep[k], fv.drawn_run(
                {"n_min_cells": d["pdf"]["n_min_cells"], f"{k}__n_cells": d["pdf"][f"{k}__n"],
                 f"{k}__vol_frac": d["pdf"][f"{k}__vol_frac"]}, z, z_mid, k)):
            raise RuntimeError(f"{k}: drawn runs differ between the caches")
    m = lambda src, k, key: np.where(keep[k], d[src][f"{k}__{key}"], np.nan)  # noqa: E731

    cnm_range = (z_mid - CNM_HALF_RANGE_PC, z_mid + CNM_HALF_RANGE_PC)
    zk = z[keep["CNM"]]
    if zk.min() < cnm_range[0] or zk.max() > cnm_range[1]:
        print(f"  WARNING: the CNM run z = {zk.min():+.0f}..{zk.max():+.0f} exceeds the CNM x range")
    xr = {"WNM": WNM_Z_RANGE, "CNM": cnm_range}

    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_H), sharex="col",
                             gridspec_kw=dict(width_ratios=WIDTH_RATIOS))
    for c, k in enumerate(COLUMNS):
        col = PHASE_COLORS[k]
        in_x = (z >= xr[k][0]) & (z <= xr[k][1])
        for r, ((src, key), (bsrc, band), pkey, ylabel, with_cs) in enumerate(ROWS):
            ax = axes[r, c]
            lo, hi = m(bsrc, k, f"{band}_p_lo"), m(bsrc, k, f"{band}_p_hi")
            ax.fill_between(z, lo, hi, where=np.isfinite(lo) & np.isfinite(hi),
                            color=col, alpha=FILL_ALPHA, lw=0, zorder=1)
            plo, phi = m("pdf", k, f"{pkey}_{B_LO}"), m("pdf", k, f"{pkey}_{B_HI}")
            ax.fill_between(z, plo, phi, where=np.isfinite(plo) & np.isfinite(phi),
                            facecolor="none", edgecolor=col, hatch=HATCH, alpha=HATCH_ALPHA,
                            lw=0, zorder=2)
            a_line, p_line = m(src, k, key), m("pdf", k, f"{pkey}_{B_LINE}")
            ax.plot(z, a_line, color=col, lw=LINE_LW, zorder=4)
            ax.plot(z, p_line, color=col, lw=LINE_LW, ls=PDF_LS, zorder=4)
            vals = [lo, hi, plo, phi, a_line, p_line]
            if with_cs:
                cs = m("vel", k, "sigma_th")
                ax.plot(z, cs, color=col, lw=CS_LW, ls=CS_LS, zorder=3)
                vals.append(cs)
            ylo, yhi = robust_linear_ylim([np.concatenate([v[in_x] for v in vals])],
                                          lo_pct=0.0, hi_pct=100.0, include_zero=True)
            if with_cs:                  # velocities are >= 0: start the axis at 0
                ylo, yhi = 0.0, yhi * TOP_HEADROOM
            ax.set_ylim(ylo, yhi)
            if ylo < 0:
                guide_line(ax, 0.0)
            ax.axvline(z_mid, **ZMID_KW)
            if c == 0:
                ax.set_ylabel(ylabel)
            print(f"  {k} row {r}: y {ylo:.3f} .. {yhi:.3f}")
        axes[0, c].text(0.97, 0.96, k, transform=axes[0, c].transAxes, ha="right", va="top",
                        fontsize=TEXT_FS + 1.5, color=col, fontweight="bold")
        axes[0, c].text(z_mid + 0.0075 * (xr[k][1] - xr[k][0]), 0.97, ZMID_LABEL,
                        transform=axes[0, c].get_xaxis_transform(), ha="left", va="top",
                        fontsize=TEXT_FS, color="0.35", zorder=6)
        axes[1, c].set_xlim(*xr[k])
        axes[1, c].set_xlabel(r"$z$ [pc]")
    axes[1, 0].set_xticks(WNM_Z_TICKS)
    step = CNM_TICK_STEP_PC
    axes[1, 1].set_xticks(np.arange(np.ceil(cnm_range[0] / step) * step, cnm_range[1] + 1, step))
    for ax, letter in zip((axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]), "abcd"):
        panel_label(ax, letter, loc="upper left")

    handles = [Line2D([], [], color=LEGEND_COLOR, lw=LINE_LW),
               Line2D([], [], color=LEGEND_COLOR, lw=LINE_LW, ls=PDF_LS),
               Line2D([], [], color=LEGEND_COLOR, lw=CS_LW, ls=CS_LS),
               Patch(facecolor=LEGEND_COLOR, alpha=0.25, lw=0),
               Patch(facecolor="none", edgecolor=LEGEND_COLOR, hatch=HATCH, lw=0)]
    fig.legend(handles, METHOD_LABELS, loc="outside upper center", ncols=3, frameon=False,
               fontsize=TEXT_FS, handlelength=2.2, columnspacing=1.4)
    provisional_label(fig, loc="upper right")   # clear of the x labels
    for k in COLUMNS:
        for (src, key), _, pkey, _, _ in ROWS:
            a, p = m(src, k, key), m("pdf", k, f"{pkey}_{B_LINE}")
            ok = np.isfinite(a) & np.isfinite(p)
            print(f"  {k} {pkey}: PDF > alpha in {int((ok & (p > a)).sum())}/{int(ok.sum())} bins")
    save_figure(fig, NAME)


if __name__ == "__main__":
    main()
