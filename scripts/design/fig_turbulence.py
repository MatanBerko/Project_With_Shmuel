"""
fig_turbulence -- Fig. 6: turbulence two ways.

Reads ONLY cache/velocity/velocity.npz (scripts/compute/compute_velocity.py)
and cache/turbulence_pdf/turbulence_pdf.npz
(scripts/compute/compute_turbulence_pdf.py). RAW, self-gravity mean
(fiducial), whole footprint, 4 pc bins of signed z.

COLOUR = phase (CNM, WNM in phase colours; all neutral thin black);
LINE STYLE = method:
  solid   from the weight (alpha): alpha - 1 assigned entirely to motions
  dashed  from the density-PDF width: M3D^2 = (exp(sigma_s^2) - 1)/b^2, b = 0.4
  dotted  the sound speed c_s = sqrt(sum P_th / sum rho)   (panel a)
Bands, for CNM and WNM only (as fig_velocity): filled = the alpha method's
mass-weighted 15-85% range of the per-cell quantity; hatched = the PDF
method over b = 1 .. 1/3. All neutral gas is drawn for the alpha method
only: its density-PDF width measures the CNM/WNM phase contrast, not
turbulence, and is shown only in the appendix figure fig_density_pdf_width.

Two stacked panels on a shared signed-z axis:
  (a) 1D velocities, linear y: sigma_nt (alpha), sigma_turb_pdf, c_s
  (b) M^2 (1D): <alpha_cell - 1> and M2_pdf = M3D^2 / 3, linear y

Layout, axis orientation, phase colours, z_mid line and the visibility rule
(contiguous run from z_mid with >= N_MIN_CELLS cells and >= 1% of the bin
volume) are fig_velocity's, imported from it.
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
FIG_W, FIG_H = fv.FIG_W, 5.0
Z_RANGE, Z_TICKS = fv.Z_RANGE, fv.Z_TICKS
PHASE_COLORS = fv.PHASE_COLORS
BAND_PHASES = fv.BAND_PHASES
BAND_ALPHA = fv.BAND_ALPHA
BAND_LEGEND_COLOR = fv.BAND_LEGEND_COLOR
LINE_LW = fv.LINE_LW
NEUTRAL_KW = fv.NEUTRAL_KW
ZMID_KW, ZMID_LABEL = fv.ZMID_KW, fv.ZMID_LABEL
TEXT_FS = fv.TEXT_FS
VEL_YLIM = (0.0, 14.0)                # panel (a), km/s
PDF_LS = (0, (4.0, 2.0))              # dashed: the density-PDF method
CS_LS = (0, (1.0, 1.4))               # dotted: c_s
CS_LW = 0.9
B_LINE, B_LO, B_HI = "b0p4", "b1", "b1_3"   # line and hatched band (b = 1 gives the lowest M^2)
HATCH = {"CNM": "////", "WNM": "\\\\\\\\"}
HATCH_ALPHA = 0.55
HATCH_LW = 0.5
METHOD_LABELS = ("from the weight (alpha)", r"from the density-PDF width ($b = 0.4$)",
                 r"sound speed $c_\mathrm{s}$", "15–85%, mass-weighted", r"$b = 1/3$ – $1$")
# per panel: (alpha mean, alpha band source/key, PDF key, y label)
PANELS = (
    (("vel", "sigma_nt"), ("pdf", "sigma_nt_cell"), "sigma_turb_pdf",
     r"$\sigma_\mathrm{nt}$, $c_\mathrm{s}$ [km s$^{-1}$]"),
    (("vel", "M2"), ("vel", "M2_cell"), "M2_pdf", r"$\mathcal{M}^2$ (1D)"),
)
# Both legend blocks sit in a strip above panel (a), outside the axes, so they
# cover no data (no region inside either panel is free over its full width).
PHASE_LEGEND_LOC = "outside upper left"
METHOD_LEGEND_LOC = "outside upper right"


def main():
    apply_paper_style()
    plt.rcParams["hatch.linewidth"] = HATCH_LW
    d = {"vel": np.load(VEL_CACHE, allow_pickle=False),
         "pdf": np.load(PDF_CACHE, allow_pickle=False)}
    z = d["vel"]["z_pc"]
    if not np.array_equal(z, d["pdf"]["z_pc"]):
        raise RuntimeError("the two caches have different bins")
    z_mid = float(d["vel"]["z_mid_pc"][0])
    keep = {k: fv.drawn_run(d["vel"], z, z_mid, k) for k in ("CNM", "WNM", "NEUTRAL")}
    for k in keep:
        if not np.array_equal(keep[k], fv.drawn_run(
                {"n_min_cells": d["pdf"]["n_min_cells"], f"{k}__n_cells": d["pdf"][f"{k}__n"],
                 f"{k}__vol_frac": d["pdf"][f"{k}__vol_frac"]}, z, z_mid, k)):
            raise RuntimeError(f"{k}: drawn runs differ between the caches")
    m = lambda src, k, key: np.where(keep[k], d[src][f"{k}__{key}"], np.nan)  # noqa: E731

    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, FIG_H), sharex=True)
    drawn = []
    for ax, ((src, key), (bsrc, band), pkey, ylabel) in zip(axes, PANELS):
        vals = []
        for k in BAND_PHASES:
            lo, hi = m(bsrc, k, f"{band}_p_lo"), m(bsrc, k, f"{band}_p_hi")
            ax.fill_between(z, lo, hi, where=np.isfinite(lo) & np.isfinite(hi),
                            color=PHASE_COLORS[k], alpha=BAND_ALPHA[k], lw=0, zorder=1)
            plo, phi = m("pdf", k, f"{pkey}_{B_LO}"), m("pdf", k, f"{pkey}_{B_HI}")
            ax.fill_between(z, plo, phi, where=np.isfinite(plo) & np.isfinite(phi),
                            facecolor="none", edgecolor=PHASE_COLORS[k], hatch=HATCH[k],
                            alpha=HATCH_ALPHA, lw=0, zorder=2)
            vals += [lo, hi, plo, phi]
        for k in BAND_PHASES:
            ax.plot(z, m(src, k, key), color=PHASE_COLORS[k], lw=LINE_LW, zorder=4)
            ax.plot(z, m("pdf", k, f"{pkey}_{B_LINE}"), color=PHASE_COLORS[k], lw=LINE_LW,
                    ls=PDF_LS, zorder=4)
            vals += [m(src, k, key), m("pdf", k, f"{pkey}_{B_LINE}")]
        ax.plot(z, m(src, "NEUTRAL", key), zorder=5, **NEUTRAL_KW)
        vals.append(m(src, "NEUTRAL", key))
        drawn.append(vals)
        ax.set_ylabel(ylabel)
        ax.axvline(z_mid, **ZMID_KW)

    ax_a, ax_b = axes
    for k in BAND_PHASES:                       # (a): c_s per phase, dotted
        cs = m("vel", k, "sigma_th")
        ax_a.plot(z, cs, color=PHASE_COLORS[k], lw=CS_LW, ls=CS_LS, zorder=3)
        drawn[0].append(cs)
    ax_a.set_ylim(*VEL_YLIM)
    v_max = float(np.nanmax(np.concatenate(drawn[0])))
    if v_max > VEL_YLIM[1]:
        print(f"  WARNING: (a) data reach {v_max:.2f} km/s, above the axis top {VEL_YLIM[1]}")
    lo_b, hi_b = robust_linear_ylim([np.concatenate(drawn[1])], lo_pct=0.0, hi_pct=100.0,
                                    include_zero=True)
    ax_b.set_ylim(lo_b, hi_b)
    guide_line(ax_b, 0.0)

    ax_a.text(z_mid + 6, 0.97, ZMID_LABEL, transform=ax_a.get_xaxis_transform(),
              ha="left", va="top", fontsize=TEXT_FS, color="0.35", zorder=6)
    for ax, letter in zip(axes, "ab"):
        panel_label(ax, letter, loc="upper left")

    leg_kw = dict(frameon=False, fontsize=TEXT_FS, handlelength=2.2)
    ph_handles = [Patch(facecolor=PHASE_COLORS["CNM"], lw=0),
                  Patch(facecolor=PHASE_COLORS["WNM"], lw=0),
                  Line2D([], [], **NEUTRAL_KW)]
    fig.legend(ph_handles, ["CNM", "WNM", "all neutral (alpha only)"], loc=PHASE_LEGEND_LOC,
               **leg_kw)
    me_handles = [Line2D([], [], color="0.25", lw=LINE_LW),
                  Line2D([], [], color="0.25", lw=LINE_LW, ls=PDF_LS),
                  Line2D([], [], color="0.25", lw=CS_LW, ls=CS_LS),
                  Patch(facecolor=BAND_LEGEND_COLOR, alpha=0.4, lw=0),
                  Patch(facecolor="none", edgecolor="0.35", hatch="////", lw=0)]
    fig.legend(me_handles, METHOD_LABELS, loc=METHOD_LEGEND_LOC, ncols=2, **leg_kw)

    ax_b.set_xlim(*Z_RANGE)
    ax_b.set_xticks(Z_TICKS)
    ax_b.set_xlabel(r"$z$ [pc]")
    provisional_label(fig, loc="lower right")
    print(f"  (a) y {VEL_YLIM[0]:.1f} .. {VEL_YLIM[1]:.1f} km/s (data max {v_max:.2f}); "
          f"(b) y {lo_b:.3f} .. {hi_b:.3f}")
    for (src, key), _, pkey, _ in PANELS:
        for k in BAND_PHASES:
            a, p = m(src, k, key), m("pdf", k, f"{pkey}_{B_LINE}")
            ok = np.isfinite(a) & np.isfinite(p)
            print(f"  {k} {pkey}: PDF > alpha in {int((ok & (p > a)).sum())}/{int(ok.sum())} bins")
    save_figure(fig, NAME)


if __name__ == "__main__":
    main()
