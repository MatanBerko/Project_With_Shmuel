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
Bands, for CNM and WNM only (as fig_velocity): filled = the alpha method's
mass-weighted 15-85% range of the per-cell quantity; hatched = the PDF
method over b = 1 .. 1/3.

Three stacked panels on a shared signed-z axis:
  (a) non-thermal 1D velocity: sigma_nt (alpha) and sigma_turb_pdf, log y
  (b) c_s = sqrt(sum P_th / sum rho), log y, SAME range as (a)  (both methods)
  (c) M^2 (1D): <alpha_cell - 1> and M2_pdf = M3D^2 / 3, linear y

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
    robust_log_ylim,
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
FIG_W, FIG_H = fv.FIG_W, fv.FIG_H
Z_RANGE, Z_TICKS = fv.Z_RANGE, fv.Z_TICKS
PHASE_COLORS = fv.PHASE_COLORS
BAND_PHASES = fv.BAND_PHASES
BAND_ALPHA = fv.BAND_ALPHA
BAND_LEGEND_COLOR = fv.BAND_LEGEND_COLOR
LINE_LW = fv.LINE_LW
NEUTRAL_KW = fv.NEUTRAL_KW
ZMID_KW, ZMID_LABEL = fv.ZMID_KW, fv.ZMID_LABEL
TEXT_FS = fv.TEXT_FS
LOG_PAD_DEX = fv.LOG_PAD_DEX
PDF_LS = (0, (4.0, 2.0))              # dashed: the density-PDF method
B_LINE, B_LO, B_HI = "b0p4", "b1", "b1_3"   # line and hatched band (b = 1 gives the lowest M^2)
HATCH = {"CNM": "////", "WNM": "\\\\\\\\"}
HATCH_ALPHA = 0.55
HATCH_LW = 0.5
METHOD_LABELS = ("from the weight (alpha)", "from the density-PDF width")
# per panel: (alpha mean, alpha band source/key or None, PDF key or None, y label)
PANELS = (
    (("vel", "sigma_nt"), ("pdf", "sigma_nt_cell"), "sigma_turb_pdf",
     r"$\sigma_\mathrm{nt}$ [km s$^{-1}$]"),
    (("vel", "sigma_th"), ("vel", "cs_cell"), None, r"$c_\mathrm{s}$ [km s$^{-1}$]"),
    (("vel", "M2"), ("vel", "M2_cell"), "M2_pdf", r"$\mathcal{M}^2$ (1D)"),
)
PHASE_LEGEND = dict(panel=1, loc="lower left", anchor=(0.0, 0.30))
METHOD_LEGEND = dict(panel=1, loc="lower right", anchor=(1.0, 0.30))


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

    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, FIG_H), sharex=True)
    drawn = []
    for ax, ((src, key), (bsrc, band), pkey, ylabel) in zip(axes, PANELS):
        vals = []
        for k in BAND_PHASES:
            lo, hi = m(bsrc, k, f"{band}_p_lo"), m(bsrc, k, f"{band}_p_hi")
            ax.fill_between(z, lo, hi, where=np.isfinite(lo) & np.isfinite(hi),
                            color=PHASE_COLORS[k], alpha=BAND_ALPHA[k], lw=0, zorder=1)
            vals += [lo, hi]
            if pkey:
                plo, phi = m("pdf", k, f"{pkey}_{B_LO}"), m("pdf", k, f"{pkey}_{B_HI}")
                ax.fill_between(z, plo, phi, where=np.isfinite(plo) & np.isfinite(phi),
                                facecolor="none", edgecolor=PHASE_COLORS[k], hatch=HATCH[k],
                                alpha=HATCH_ALPHA, lw=0, zorder=2)
                vals += [plo, phi]
        for k in BAND_PHASES:
            ax.plot(z, m(src, k, key), color=PHASE_COLORS[k], lw=LINE_LW, zorder=4)
            vals.append(m(src, k, key))
            if pkey:
                ax.plot(z, m("pdf", k, f"{pkey}_{B_LINE}"), color=PHASE_COLORS[k], lw=LINE_LW,
                        ls=PDF_LS, zorder=4)
                vals.append(m("pdf", k, f"{pkey}_{B_LINE}"))
        ax.plot(z, m(src, "NEUTRAL", key), zorder=5, **NEUTRAL_KW)
        vals.append(m(src, "NEUTRAL", key))
        if pkey:
            ax.plot(z, m("pdf", "NEUTRAL", f"{pkey}_{B_LINE}"), ls=PDF_LS, zorder=5, **NEUTRAL_KW)
            vals.append(m("pdf", "NEUTRAL", f"{pkey}_{B_LINE}"))
        drawn.append(vals)
        ax.set_ylabel(ylabel)
        ax.axvline(z_mid, **ZMID_KW)

    ax_a, ax_b, ax_c = axes
    # (a) and (b): one shared log range over everything drawn (zeros of the
    # clipped sigma_nt band are left out of the range)
    pos = [np.where(v > 0, v, np.nan) for v in drawn[0] + drawn[1]]
    lo, hi = robust_log_ylim([np.concatenate(pos)], lo_pct=0.0, hi_pct=100.0, pad_dex=LOG_PAD_DEX)
    for ax in (ax_a, ax_b):
        ax.set_yscale("log")
        ax.set_ylim(lo, hi)
    lo_c, hi_c = robust_linear_ylim([np.concatenate(drawn[2])], lo_pct=0.0, hi_pct=100.0,
                                    include_zero=True)
    ax_c.set_ylim(lo_c, hi_c)
    guide_line(ax_c, 0.0)

    ax_a.text(z_mid + 6, 0.97, ZMID_LABEL, transform=ax_a.get_xaxis_transform(),
              ha="left", va="top", fontsize=TEXT_FS, color="0.35", zorder=6)
    for ax, letter in zip(axes, "abc"):
        panel_label(ax, letter, loc="upper left")

    leg_kw = dict(frameon=False, fontsize=TEXT_FS, handlelength=2.2)
    ph_handles = [Patch(facecolor=PHASE_COLORS["CNM"], lw=0),
                  Patch(facecolor=PHASE_COLORS["WNM"], lw=0),
                  Line2D([], [], **NEUTRAL_KW)]
    leg1 = axes[PHASE_LEGEND["panel"]].legend(
        ph_handles, ["CNM", "WNM", "all neutral"], loc=PHASE_LEGEND["loc"],
        bbox_to_anchor=PHASE_LEGEND["anchor"], **leg_kw)
    axes[PHASE_LEGEND["panel"]].add_artist(leg1)
    me_handles = [Line2D([], [], color="0.25", lw=LINE_LW),
                  Line2D([], [], color="0.25", lw=LINE_LW, ls=PDF_LS),
                  Patch(facecolor=BAND_LEGEND_COLOR, alpha=0.4, lw=0),
                  Patch(facecolor="none", edgecolor="0.35", hatch="////", lw=0)]
    axes[METHOD_LEGEND["panel"]].legend(
        me_handles, [METHOD_LABELS[0], METHOD_LABELS[1] + r" ($b = 0.4$)",
                     "15–85%, mass-weighted", r"$b = 1/3$ – $1$"],
        loc=METHOD_LEGEND["loc"], bbox_to_anchor=METHOD_LEGEND["anchor"], **leg_kw)

    ax_c.set_xlim(*Z_RANGE)
    ax_c.set_xticks(Z_TICKS)
    ax_c.set_xlabel(r"$z$ [pc]")
    provisional_label(fig, loc="lower right")
    print(f"  (a),(b) y {lo:.3f} .. {hi:.3f} km/s; (c) y {lo_c:.3f} .. {hi_c:.3f}")
    # where the PDF line (b = 0.4) rises above the alpha line, inside the drawn run
    for (src, key), _, pkey, _ in PANELS:
        if not pkey:
            continue
        for k in ("CNM", "WNM", "NEUTRAL"):
            a, p = m(src, k, key), m("pdf", k, f"{pkey}_{B_LINE}")
            ok = np.isfinite(a) & np.isfinite(p)
            up = ok & (p > a)
            print(f"  {k} {pkey}: PDF > alpha in {int(up.sum())}/{int(ok.sum())} bins")
    save_figure(fig, NAME)


if __name__ == "__main__":
    main()
