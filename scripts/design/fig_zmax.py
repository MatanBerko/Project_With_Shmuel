"""
fig_zmax -- single- and two-component layer fits to the density profile,
their chi2 vs fit range, and alpha vs the integration height; cube only
(|z| <= 750 pc). No map query anywhere.

Reads (design only, nothing recomputed):
  cache/layer_fit/layer_fit.npz  -- the cube profile in 4 pc bins of
      |z - z_mid|, its 16-square error, the single fits (z_max = 100..750);
  cache/zmax/zmax.npz            -- the double fits (dbl_* keys) and the
      headline <alpha_cell> vs z_top (used for z_top <= 750 pc only).
Layout, two columns x three rows, row 3 spanning both:
  (a) single component: data line + 16-square band, z_mid; layer_fit's fits
      of exp / Gaussian / sech2 at z_max = SINGLE_ZMAX, solid over the fitted
      bins, dashed beyond to the cube edge if SINGLE_EXTRAPOLATE; thin lines
      at z_mid +- SINGLE_ZMAX.
  (b) two components: same data; full-range best double fits, solid, with
      the two components thin and dotted in the same colour.
  (c) single: reduced chi2 vs z_max, log y, solid upper / dashed lower;
      thin line at SINGLE_ZMAX.
  (d) double: the same for every converged fit; constrained fits at normal
      width, the rest (a parameter at a bound, or sigma_h2/h2 >= 0.5) thin
      and faded.
  (e) <alpha_cell> vs z_top = 300..750 pc; labels at 400 and 750 pc.
(a)/(b) and (c)/(d) share their y axes.
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
SINGLE_ZMAX = 400.0                   # pc: the single fits shown in (a)
SINGLE_EXTRAPOLATE = True             # dashed continuation beyond SINGLE_ZMAX
FIG_W, FIG_H = 7.0, 7.0
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
COMP_KW = dict(lw=0.6, ls=(0, (1.0, 1.4)))
ZMID_KW = dict(color="0.45", lw=0.7, ls=":", zorder=1)
ZMAX_LINE_KW = dict(color="0.45", lw=0.5, ls="-", zorder=1)
EXTRAP_LS = (0, (3.0, 1.6))
LOOSE_KW = dict(lw=0.5, alpha=0.35, zorder=2)   # (d): fits that are not constrained
ALPHA_LW = 1.6
ALPHA_MARKS_PC = (400.0, 750.0)
TEXT_FS = 6.5
TITLE_FS = 8.0


def shape(model, x):
    if model == "exp":
        return np.exp(-x)
    if model == "gauss":
        return np.exp(-0.5 * x ** 2)
    return 1.0 / np.cosh(np.minimum(x, 350.0)) ** 2


def draw_profile(ax, d, z_mid):
    ax.axvline(z_mid, **ZMID_KW)
    for side in ("upper", "lower"):
        s = SIGN[side]
        u, n, e = d[f"{side}_u"], d[f"{side}_nH"], d[f"{side}_err_log"]
        zs = z_mid + s * u
        ax.fill_between(zs, n * 10 ** (-e), n * 10 ** e, **BAND_KW, zorder=2)
        ax.plot(zs, n, **DATA_KW, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(*N_YLIM)
    ax.set_xlim(*Z_RANGE)
    ax.set_xticks(Z_TICKS)
    ax.set_xlabel(r"$z$  [pc]", labelpad=1)
    ax.text(z_mid + 8, N_YLIM[0] * 1.25, r"$z_{\rm mid}$", fontsize=TEXT_FS, color="0.35",
            ha="left", va="bottom")


def fit_u(d, side, z_max):
    """u over the bins that enter the fit at z_max."""
    u, hi = d[f"{side}_u"], d[f"{side}_bin_hi"]
    return np.linspace(0.0, float(u[hi <= z_max + 1e-9].max()), 500)


def main():
    apply_paper_style()
    d = np.load(LAYER_FIT_CACHE, allow_pickle=False)
    zc = np.load(ZMAX_CACHE, allow_pickle=False)
    z_mid = float(d["z_mid"][0])
    models = [str(m) for m in d["models"]]
    zg = d["z_max_grid"]
    zg2 = zc["dbl_z_max_grid"]
    if not (np.isclose(zg[-1], CUBE_EDGE_PC) and np.isclose(zg2[-1], CUBE_EDGE_PC)):
        raise RuntimeError("the last z_max is not 750 pc")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = fig.add_gridspec(3, 2, height_ratios=(1.25, 1.0, 0.85))
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1], sharey=ax_c)
    ax_e = fig.add_subplot(gs[2, :])

    # ------------------------------------------------------------- (a) single
    i_s = int(np.argmin(np.abs(zg - SINGLE_ZMAX)))
    if not np.isclose(zg[i_s], SINGLE_ZMAX):
        raise RuntimeError(f"SINGLE_ZMAX = {SINGLE_ZMAX} is not on layer_fit's z_max grid")
    draw_profile(ax_a, d, z_mid)
    for s in (-1.0, 1.0):
        ax_a.axvline(z_mid + s * SINGLE_ZMAX, ymax=0.6, **ZMAX_LINE_KW)   # stops below the legend
    for side in ("upper", "lower"):
        s, uu = SIGN[side], fit_u(d, side, SINGLE_ZMAX)
        u_edge = abs((Z_RANGE[1] if s > 0 else Z_RANGE[0]) - z_mid)
        for m in models:
            n0, h = d[f"{side}_{m}_n0"][i_s], d[f"{side}_{m}_h"][i_s]
            ax_a.plot(z_mid + s * uu, n0 * shape(m, uu / h), color=MODEL_COLORS[m], lw=FIT_LW, zorder=4)
            if SINGLE_EXTRAPOLATE:
                ue = np.linspace(uu[-1], u_edge, 300)
                ax_a.plot(z_mid + s * ue, n0 * shape(m, ue / h), color=MODEL_COLORS[m], lw=FIT_LW,
                          ls=EXTRAP_LS, zorder=4)
    ax_a.set_ylabel(r"$\langle n_{\rm H}\rangle_{\rm area}$ [cm$^{-3}$]")
    ax_a.set_title("single component", fontsize=TITLE_FS, pad=4)
    panel_label(ax_a, "a", loc="upper left")
    style_h = [Line2D([], [], color="0.3", lw=FIT_LW)]
    style_l = ["fitted range"]
    if SINGLE_EXTRAPOLATE:
        style_h.append(Line2D([], [], color="0.3", lw=FIT_LW, ls=EXTRAP_LS))
        style_l.append("extrapolated")
    ax_a.legend([Line2D([], [], **DATA_KW), plt.Rectangle((0, 0), 1, 1, **BAND_KW)]
                + [Line2D([], [], color=MODEL_COLORS[m], lw=FIT_LW) for m in models] + style_h,
                ["data (4 pc bins)", "16-square error"] + [MODEL_LABELS[m] for m in models] + style_l,
                loc="upper right", frameon=False, fontsize=TEXT_FS - 0.5, handlelength=1.5,
                borderaxespad=0.2, labelspacing=0.3)

    # ------------------------------------------------------------- (b) double
    draw_profile(ax_b, d, z_mid)
    for side in ("upper", "lower"):
        s, uu = SIGN[side], fit_u(d, side, zg2[-1])
        for m in models:
            k = f"dbl_{side}_{m}_"
            n1, h1, n2, h2 = (float(zc[k + p][-1]) for p in ("n1", "h1", "n2", "h2"))
            c1, c2 = n1 * shape(m, uu / h1), n2 * shape(m, uu / h2)
            col = MODEL_COLORS[m]
            ax_b.plot(z_mid + s * uu, c1 + c2, color=col, lw=FIT_LW, zorder=4)
            ax_b.plot(z_mid + s * uu, c1, color=col, **COMP_KW, zorder=4)
            ax_b.plot(z_mid + s * uu, c2, color=col, **COMP_KW, zorder=4)
    ax_b.tick_params(labelleft=False)
    ax_b.set_title("two components", fontsize=TITLE_FS, pad=4)
    panel_label(ax_b, "b", loc="upper left")
    ax_b.legend([Line2D([], [], color="0.3", lw=FIT_LW), Line2D([], [], color="0.3", **COMP_KW)],
                ["sum", "components"], loc="upper right", frameon=False, fontsize=TEXT_FS - 0.5,
                handlelength=1.8, borderaxespad=0.2, labelspacing=0.3)

    # --------------------------------------------------------- (c), (d) chi2
    for side in ("upper", "lower"):
        for m in models:
            ax_c.plot(zg, d[f"{side}_{m}_rchi2"], color=MODEL_COLORS[m], ls=SIDE_LS[side], lw=1.0)
            k = f"dbl_{side}_{m}_"
            r_all = np.where(zc[k + "converged"], zc[k + "rchi2"], np.nan)
            r_con = np.where(zc[k + "constrained"], zc[k + "rchi2"], np.nan)
            ax_d.plot(zg2, r_all, color=MODEL_COLORS[m], ls=SIDE_LS[side], **LOOSE_KW)
            ax_d.plot(zg2, r_con, color=MODEL_COLORS[m], ls=SIDE_LS[side], lw=1.0, zorder=3)
    for ax in (ax_c, ax_d):
        ax.set_xlim(*ZMAX_RANGE)
        ax.set_yscale("log")
        ax.set_xlabel(r"fit range $z_{\max}$ = max $|z - z_{\rm mid}|$  [pc]", labelpad=1)
    ax_c.axvline(SINGLE_ZMAX, **ZMAX_LINE_KW)
    ax_d.tick_params(labelleft=False)
    ax_c.set_ylabel(r"reduced $\chi^2$")
    panel_label(ax_c, "c", loc="upper left")
    panel_label(ax_d, "d", loc="upper left")
    ax_c.legend([Line2D([], [], color=MODEL_COLORS[m], lw=1.0) for m in models]
                + [Line2D([], [], color="0.3", ls=SIDE_LS[s], lw=1.0) for s in ("upper", "lower")],
                [MODEL_LABELS[m] for m in models] + ["upper", "lower"],
                loc="lower right", ncol=2, frameon=False, fontsize=TEXT_FS - 0.5, handlelength=2.0,
                columnspacing=0.8, borderaxespad=0.2, labelspacing=0.3)
    ax_d.legend([Line2D([], [], color="0.3", lw=1.0), Line2D([], [], color="0.3", **LOOSE_KW)],
                ["constrained", r"parameter at a bound or $h_2$ unconstrained"],
                loc="upper right", frameon=False, fontsize=TEXT_FS - 0.5, handlelength=2.0,
                borderaxespad=0.2, labelspacing=0.3)

    # ------------------------------------------------------------- (e) alpha
    zt_all = zc["z_top_grid"]
    cube = (zt_all >= ZTOP_RANGE[0] - 1e-9) & (zt_all <= ZTOP_RANGE[1] + 1e-9)
    zt, a = zt_all[cube], zc["zt_alpha_cell_mean"][cube]
    ax_e.plot(zt, a, color=COLOR_ALPHA, lw=ALPHA_LW, zorder=3)
    for zz in ALPHA_MARKS_PC:
        j = int(np.argmin(np.abs(zt - zz)))
        ax_e.plot(zt[j], a[j], marker="o", ms=3.6, mfc=COLOR_ALPHA, mec="white", mew=0.5,
                  ls="none", zorder=4)
        edge = zz == ZTOP_RANGE[1]
        lab = f"{a[j]:.2f}" + (" (fiducial)" if edge else "")
        ax_e.annotate(lab, (zt[j], a[j]), xytext=(-6, 3) if edge else (5, -6),
                      textcoords="offset points", ha="right" if edge else "left",
                      va="bottom" if edge else "top",
                      fontsize=TEXT_FS, color="0.15")
    ax_e.set_xlim(*ZTOP_RANGE)
    lo, hi = float(a.min()), float(a.max())
    ax_e.set_ylim(lo - 0.15 * (hi - lo), hi + 0.12 * (hi - lo))
    ax_e.set_ylabel(r"midplane $\langle\alpha_{\rm cell}\rangle$")
    ax_e.set_xlabel(r"integration height $z_{\rm top}$ ($\rho = 0$ at $|z| > z_{\rm top}$)  [pc]",
                    labelpad=1)
    panel_label(ax_e, "e", loc="upper left")

    fig.align_ylabels((ax_a, ax_c, ax_e))
    provisional_label(fig, loc="upper right")
    save_figure(fig, NAME)


if __name__ == "__main__":
    main()
