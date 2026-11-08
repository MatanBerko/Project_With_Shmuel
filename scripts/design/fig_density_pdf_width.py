"""
fig_density_pdf_width -- appendix diagnostic for Fig. 6: the log-density width.

Reads ONLY cache/turbulence_pdf/turbulence_pdf.npz
(scripts/compute/compute_turbulence_pdf.py).

  (a) sigma_s(z) = std ln(n_H / <n>_V) per 4 pc bin for CNM, WNM (phase
      colours) and all neutral (black); solid = native 2 pc cells, dashed =
      after 2 x 2 x 2 block averaging (sigma_s_4pc). Drawn over the
      fig_velocity visibility run.
  (b)-(e) the volume-weighted PDF of s at |z - z_mid| = 0, 100, 200, 300 pc
      (the union of the 4 pc bins containing z_mid +- d), per sample, with a
      Gaussian of the same <s> and sigma_s (thin dotted) for reference.
      A sample is shown only where it has >= N_MIN_CELLS cells.
"""

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.plotting.style import (  # noqa: E402
    apply_paper_style,
    panel_label,
    provisional_label,
    save_figure,
)

_SPEC = importlib.util.spec_from_file_location(
    "fig_velocity", Path(__file__).resolve().parent / "fig_velocity.py")
fv = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fv)

# ═══════════════════════════════════════════════════════════════════════
# ██  CONTROL BLOCK
# ═══════════════════════════════════════════════════════════════════════
CACHE = Path("cache/turbulence_pdf/turbulence_pdf.npz")
NAME = "fig_density_pdf_width"
FIG_W, FIG_H = 5.0, 4.6
Z_RANGE, Z_TICKS = fv.Z_RANGE, fv.Z_TICKS
COLORS = {**fv.PHASE_COLORS, "NEUTRAL": "black"}
LW = {"CNM": fv.LINE_LW, "WNM": fv.LINE_LW, "NEUTRAL": 0.9}
SAMPLES = ("CNM", "WNM", "NEUTRAL")
LABELS = {"CNM": "CNM", "WNM": "WNM", "NEUTRAL": "all neutral"}
DASH = (0, (4.0, 2.0))
GAUSS_KW = dict(lw=0.6, ls=":")
S_RANGE = (-6.0, 5.0)
PDF_YLIM = (1e-4, 40.0)            # headroom above the peaks for the labels
TEXT_FS = fv.TEXT_FS


def main():
    apply_paper_style()
    d = np.load(CACHE, allow_pickle=False)
    z = d["z_pc"]
    z_mid = float(d["z_mid_pc"][0])
    n_min = int(d["n_min_cells"][0])
    run = {k: fv.drawn_run({"n_min_cells": d["n_min_cells"], f"{k}__n_cells": d[f"{k}__n"],
                            f"{k}__vol_frac": d[f"{k}__vol_frac"]}, z, z_mid, k)
           for k in SAMPLES}

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = fig.add_gridspec(2, 4, height_ratios=(1.15, 1.0))
    ax = fig.add_subplot(gs[0, :])
    for k in SAMPLES:
        ax.plot(z, np.where(run[k], d[f"{k}__sigma_s"], np.nan), color=COLORS[k], lw=LW[k])
        ax.plot(z, np.where(run[k], d[f"{k}__sigma_s_4pc"], np.nan), color=COLORS[k], lw=LW[k],
                ls=DASH)
    ax.axvline(z_mid, **fv.ZMID_KW)
    ax.text(z_mid + 6, 0.97, fv.ZMID_LABEL, transform=ax.get_xaxis_transform(), ha="left",
            va="top", fontsize=TEXT_FS, color="0.35")
    ax.set_xlim(*Z_RANGE)
    ax.set_xticks(Z_TICKS)
    ax.set_ylim(0.0, None)
    ax.set_xlabel(r"$z$ [pc]")
    ax.set_ylabel(r"$\sigma_s$")
    panel_label(ax, "a", loc="upper left")
    handles = [Line2D([], [], color=COLORS[k], lw=LW[k]) for k in SAMPLES] + [
        Line2D([], [], color="0.3", lw=1.0), Line2D([], [], color="0.3", lw=1.0, ls=DASH)]
    ax.legend(handles, [LABELS[k] for k in SAMPLES] + ["2 pc cells", "4 pc blocks"],
              loc="lower center", ncols=5, frameon=False, fontsize=TEXT_FS, handlelength=2.0,
              columnspacing=1.0)

    edges = d["s_edges"]
    ctr = 0.5 * (edges[:-1] + edges[1:])
    ds = np.diff(edges)
    s_fine = np.linspace(*S_RANGE, 400)
    for j, dz in enumerate(d["pdfz__offsets_pc"]):
        a = fig.add_subplot(gs[1, j], sharey=None if j == 0 else fig.axes[1])
        for k in SAMPLES:
            n = d[f"pdfz__{k}__n"][j]
            if n < n_min:
                continue
            h = d[f"pdfz__{k}__s_hist"][j]
            a.step(ctr, np.where(h > 0, h / (n * ds), np.nan), where="mid", color=COLORS[k],
                   lw=LW[k] * 0.8)
            mu, sg = d[f"pdfz__{k}__mean_s"][j], d[f"pdfz__{k}__sigma_s"][j]
            a.plot(s_fine, np.exp(-0.5 * ((s_fine - mu) / sg) ** 2) / (sg * np.sqrt(2 * np.pi)),
                   color=COLORS[k], **GAUSS_KW)
        a.set_yscale("log")
        a.set_ylim(*PDF_YLIM)
        a.set_xlim(*S_RANGE)
        a.set_xticks((-4, -2, 0, 2, 4))
        a.set_xlabel(r"$s = \ln(n/\langle n\rangle)$")
        if j == 0:
            a.set_ylabel(r"PDF of $s$")
        else:
            a.tick_params(labelleft=False)
        a.text(0.97, 0.965, rf"$|z-z_\mathrm{{mid}}|$" + f"\n= {dz:.0f} pc", transform=a.transAxes,
               ha="right", va="top", fontsize=TEXT_FS - 0.5, linespacing=1.1)
        panel_label(a, "bcde"[j], loc="upper left")
    provisional_label(fig, loc="upper right")
    i_mid = int(np.argmin(np.abs(z - z_mid)))
    for k in SAMPLES:
        print(f"  {k}: sigma_s_4pc/sigma_s at the z_mid bin = "
              f"{d[f'{k}__sigma_s_4pc'][i_mid] / d[f'{k}__sigma_s'][i_mid]:.4f}")
    save_figure(fig, NAME)


if __name__ == "__main__":
    main()
