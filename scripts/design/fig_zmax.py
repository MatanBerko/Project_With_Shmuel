"""
fig_zmax -- the density profile with full-range layer fits, the fit-range
chi2 and alpha vs the integration height, restricted to the cube
(|z| <= 750 pc). No map query anywhere.

Reads (design only, nothing recomputed):
  cache/layer_fit/layer_fit.npz  (scripts/validation/layer_fit.py) -- the cube
      profile in 4 pc bins of |z - z_mid|, its 16-square error, and the fits
      for z_max = 100..750 pc;
  cache/zmax/zmax.npz            (scripts/compute/compute_zmax.py) -- the
      headline <alpha_cell> vs z_top, used only for z_top <= 750 pc (cube only).
Three stacked panels:
  (a) <n_H>_area at signed z, line + 16-square band, z_mid marked; for each
      model and side ONE solid curve: the best fit over the full range
      (layer_fit's fit at the last z_max, 750 pc), drawn over the fitted bins.
  (b) reduced chi2 vs z_max = 100..750 pc, log y: solid = upper, dashed = lower.
  (c) <alpha_cell> vs z_top = 300..750 pc, labels at 400 and 750 pc.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.conventions import COLOR_ALPHA  # noqa: E402
from src.plotting.style import (  # noqa: E402
    apply_paper_style,
    panel_label,
    provisional_label,
    save_figure,
)

# ═══════════════════════════════════════════════════════════════════════
# ██  CONTROL BLOCK
# ═══════════════════════════════════════════════════════════════════════
LAYER_FIT_CACHE = Path("cache/layer_fit/layer_fit.npz")
ZMAX_CACHE = Path("cache/zmax/zmax.npz")
NAME = "fig_zmax"
FIG_W, FIG_H = 5.0, 6.8
CUBE_EDGE_PC = 750.0
Z_RANGE = (-CUBE_EDGE_PC, CUBE_EDGE_PC)
Z_TICKS = (-750, -500, -250, 0, 250, 500, 750)
ZMAX_RANGE = (100.0, 750.0)
ZTOP_RANGE = (300.0, 750.0)
N_YLIM = (5e-3, 3.0)
MODEL_COLORS = {"exp": "#0072B2", "gauss": "#D55E00", "sech2": "#009E73"}   # as fig_layer_fit
MODEL_LABELS = {"exp": "exp", "gauss": "Gaussian", "sech2": r"sech$^2$"}
SIDE_LS = {"upper": "-", "lower": (0, (4.0, 2.0))}
SIGN = {"upper": 1.0, "lower": -1.0}
DATA_KW = dict(color="0.15", lw=1.0)
BAND_KW = dict(color="0.6", alpha=0.35, lw=0)
FIT_LW = 1.0
ZMID_KW = dict(color="0.45", lw=0.7, ls=":", zorder=1)
ALPHA_LW = 1.6
ALPHA_MARKS_PC = (400.0, 750.0)
TEXT_FS = 6.5


def model_density(model, u, n0, h):
    if model == "exp":
        return n0 * np.exp(-u / h)
    if model == "gauss":
        return n0 * np.exp(-u ** 2 / (2.0 * h ** 2))
    return n0 / np.cosh(np.minimum(u / h, 350.0)) ** 2


def main():
    apply_paper_style()
    d = np.load(LAYER_FIT_CACHE, allow_pickle=False)
    zc = np.load(ZMAX_CACHE, allow_pickle=False)
    z_mid = float(d["z_mid"][0])
    models = [str(m) for m in d["models"]]
    zg = d["z_max_grid"]
    if not np.isclose(zg[-1], CUBE_EDGE_PC):
        raise RuntimeError("layer_fit's last z_max is not 750 pc")

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        3, 1, figsize=(FIG_W, FIG_H), gridspec_kw=dict(height_ratios=(1.35, 1.0, 0.9)))

    # ------------------------------------------------------------------ (a)
    ax_a.axvline(z_mid, **ZMID_KW)
    for side in ("upper", "lower"):
        s = SIGN[side]
        u, n, e, hi = d[f"{side}_u"], d[f"{side}_nH"], d[f"{side}_err_log"], d[f"{side}_bin_hi"]
        zs = z_mid + s * u
        ax_a.fill_between(zs, n * 10 ** (-e), n * 10 ** e, **BAND_KW, zorder=2)
        ax_a.plot(zs, n, **DATA_KW, zorder=3)
        u_fit = float(u[hi <= zg[-1] + 1e-9].max())        # last bin entering the fit
        uu = np.linspace(0.0, u_fit, 500)
        for m in models:
            n0, h = d[f"{side}_{m}_n0"][-1], d[f"{side}_{m}_h"][-1]
            ax_a.plot(z_mid + s * uu, model_density(m, uu, n0, h), color=MODEL_COLORS[m],
                      lw=FIT_LW, zorder=4)
    ax_a.set_yscale("log")
    ax_a.set_ylim(*N_YLIM)
    ax_a.set_xlim(*Z_RANGE)
    ax_a.set_xticks(Z_TICKS)
    ax_a.set_xlabel(r"$z$  [pc]", labelpad=1)
    ax_a.set_ylabel(r"$\langle n_{\rm H}\rangle_{\rm area}$ [cm$^{-3}$]")
    panel_label(ax_a, "a", loc="upper left")
    ax_a.text(z_mid + 8, N_YLIM[0] * 1.25, r"$z_{\rm mid}$", fontsize=TEXT_FS, color="0.35",
              ha="left", va="bottom")
    ax_a.legend([Line2D([], [], **DATA_KW), plt.Rectangle((0, 0), 1, 1, **BAND_KW)]
                + [Line2D([], [], color=MODEL_COLORS[m], lw=FIT_LW) for m in models],
                ["data (4 pc bins)", "16-square error"]
                + [f"{MODEL_LABELS[m]} best fit" for m in models],
                loc="upper right", bbox_to_anchor=(1.0, 0.95), frameon=False,
                fontsize=TEXT_FS, handlelength=1.6, borderaxespad=0.3)

    # ------------------------------------------------------------------ (b)
    for side in ("upper", "lower"):
        for m in models:
            ax_b.plot(zg, d[f"{side}_{m}_rchi2"], color=MODEL_COLORS[m], ls=SIDE_LS[side], lw=1.0)
    ax_b.set_xlim(*ZMAX_RANGE)
    ax_b.set_yscale("log")
    ax_b.set_ylabel(r"reduced $\chi^2$")
    ax_b.set_xlabel(r"fit range $z_{\max}$ = max $|z - z_{\rm mid}|$  [pc]", labelpad=1)
    panel_label(ax_b, "b", loc="upper left")
    ax_b.legend([Line2D([], [], color=MODEL_COLORS[m], lw=1.0) for m in models]
                + [Line2D([], [], color="0.3", ls=SIDE_LS[s], lw=1.0) for s in ("upper", "lower")],
                [MODEL_LABELS[m] for m in models] + ["upper", "lower"],
                loc="lower right", ncol=2, frameon=False, fontsize=TEXT_FS, handlelength=2.0,
                columnspacing=1.0, borderaxespad=0.3)

    # ------------------------------------------------------------------ (c)
    zt_all = zc["z_top_grid"]
    cube = (zt_all >= ZTOP_RANGE[0] - 1e-9) & (zt_all <= ZTOP_RANGE[1] + 1e-9)
    zt, a = zt_all[cube], zc["zt_alpha_cell_mean"][cube]
    ax_c.plot(zt, a, color=COLOR_ALPHA, lw=ALPHA_LW, zorder=3)
    for zz in ALPHA_MARKS_PC:
        j = int(np.argmin(np.abs(zt - zz)))
        ax_c.plot(zt[j], a[j], marker="o", ms=3.6, mfc=COLOR_ALPHA, mec="white", mew=0.5,
                  ls="none", zorder=4)
        edge = zz == ZTOP_RANGE[1]
        lab = f"{a[j]:.2f}" + ("\n(fiducial)" if edge else "")
        ax_c.annotate(lab, (zt[j], a[j]), xytext=(-5 if edge else 4, -9 if edge else -5), textcoords="offset points",
                      ha="right" if edge else "left", va="top", fontsize=TEXT_FS, color="0.15",
                      linespacing=0.9)
    ax_c.set_xlim(*ZTOP_RANGE)
    lo, hi = float(a.min()), float(a.max())
    ax_c.set_ylim(lo - 0.15 * (hi - lo), hi + 0.12 * (hi - lo))
    ax_c.set_ylabel(r"midplane $\langle\alpha_{\rm cell}\rangle$")
    ax_c.set_xlabel(r"integration height $z_{\rm top}$ ($\rho = 0$ at $|z| > z_{\rm top}$)  [pc]",
                    labelpad=1)
    panel_label(ax_c, "c", loc="upper left")

    fig.align_ylabels((ax_a, ax_b, ax_c))
    provisional_label(fig, loc="upper right")
    save_figure(fig, NAME)


if __name__ == "__main__":
    main()
