"""
fig_zmax -- the density profile, the fit-range chi2 and alpha vs the
integration height, all to the map's extent (|z| <= 1000 pc).

Reads ONLY cache/zmax/zmax.npz (scripts/compute/compute_zmax.py).
Three stacked panels:
  (a) <n_H>_area in 4 pc bins of |z - z_mid| on each side, at signed z:
      the cube (line, 16-square band) and the map query beyond the cube
      edge (markers, same error as bars). The exp / Gaussian / sech2 fits at
      each model's knee on each side, solid inside the knee, dashed beyond.
      Light shading |z - z_mid| <= 400 pc (the analysis box), thin lines at
      z = +-750 pc (cube edge), dotted z_mid.
  (b) reduced chi2 vs fit range z_max, log y: solid = upper, dashed = lower;
      knees as small markers; vertical line at 750 pc.
  (c) the headline <alpha_cell> vs z_top: solid while only the cube enters
      (z_top <= 750), dashed where the query enters; labelled markers at
      400, 750 and 1000 pc; the knee range of (b) as a light band.
(b) and (c) share the x axis so the knee band lines up with (b).
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
CACHE = Path("cache/zmax/zmax.npz")
NAME = "fig_zmax"
FIG_W, FIG_H = 5.0, 6.8
Z_RANGE = (-1020.0, 1020.0)
Z_TICKS = (-1000, -750, -500, -250, 0, 250, 500, 750, 1000)
X_RANGE_BC = (100.0, 1000.0)
N_YLIM = (5e-3, 3.0)
BOX_HALF_PC = 400.0
CUBE_EDGE_PC = 750.0
MODEL_COLORS = {"exp": "#0072B2", "gauss": "#D55E00", "sech2": "#009E73"}   # as fig_layer_fit
MODEL_LABELS = {"exp": "exp", "gauss": "Gaussian", "sech2": r"sech$^2$"}
SIDE_LS = {"upper": "-", "lower": (0, (4.0, 2.0))}
SIGN = {"upper": 1.0, "lower": -1.0}
BEYOND_LS = (0, (3.0, 1.6))
DATA_KW = dict(color="0.15", lw=1.0)
BAND_KW = dict(color="0.6", alpha=0.35, lw=0)
QUERY_KW = dict(marker="o", ms=1.8, mfc="none", mec="0.15", mew=0.5, ls="none")
QUERY_ERR_KW = dict(ecolor="0.55", elinewidth=0.4, capsize=0)
BOX_KW = dict(color="0.93", lw=0, zorder=0)
EDGE_KW = dict(color="0.35", lw=0.5, ls="-", zorder=1)
ZMID_KW = dict(color="0.45", lw=0.7, ls=":", zorder=1)
KNEE_BAND_KW = dict(color="0.90", lw=0, zorder=0)
ALPHA_LW = 1.6
ALPHA_MARKS_PC = (400.0, 750.0, 1000.0)
TEXT_FS = 6.5


def model_density(model, u, n0, h):
    if model == "exp":
        return n0 * np.exp(-u / h)
    if model == "gauss":
        return n0 * np.exp(-u ** 2 / (2.0 * h ** 2))
    return n0 / np.cosh(np.minimum(u / h, 350.0)) ** 2


def main():
    apply_paper_style()
    d = np.load(CACHE, allow_pickle=False)
    z_mid = float(d["z_mid"][0])
    models = [str(m) for m in d["models"]]
    zg = d["z_max_grid"]

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        3, 1, figsize=(FIG_W, FIG_H), gridspec_kw=dict(height_ratios=(1.35, 1.0, 0.9)))
    ax_b.sharex(ax_c)

    # ------------------------------------------------------------------ (a)
    ax_a.axvspan(z_mid - BOX_HALF_PC, z_mid + BOX_HALF_PC, **BOX_KW)
    for s in (-1.0, 1.0):
        ax_a.axvline(s * CUBE_EDGE_PC, **EDGE_KW)
    ax_a.axvline(z_mid, **ZMID_KW)
    for side in ("upper", "lower"):
        s = SIGN[side]
        u, n, e, q = (d[f"{side}_u"], d[f"{side}_nH"], d[f"{side}_err_log"], d[f"{side}_is_query"])
        zs = z_mid + s * u
        c = ~q
        ax_a.fill_between(zs[c], n[c] * 10 ** (-e[c]), n[c] * 10 ** e[c], **BAND_KW, zorder=2)
        ax_a.plot(zs[c], n[c], **DATA_KW, zorder=3)
        ax_a.errorbar(zs[q], n[q], yerr=[n[q] * (1 - 10 ** (-e[q])), n[q] * (10 ** e[q] - 1)],
                      **QUERY_KW, **QUERY_ERR_KW, zorder=3)
        u_end = abs((Z_RANGE[1] if s > 0 else Z_RANGE[0]) - z_mid)
        for m in models:
            zk = float(d[f"{side}_{m}_knee"][0])
            if not np.isfinite(zk):
                continue
            i = int(np.argmin(np.abs(zg - zk)))
            n0, h = d[f"{side}_{m}_n0"][i], d[f"{side}_{m}_h"][i]
            for lo, hi, ls in ((0.0, zk, "-"), (zk, u_end, BEYOND_LS)):
                uu = np.linspace(lo, hi, 400)
                ax_a.plot(z_mid + s * uu, model_density(m, uu, n0, h), color=MODEL_COLORS[m],
                          lw=0.9, ls=ls, zorder=4)
    ax_a.set_yscale("log")
    ax_a.set_ylim(*N_YLIM)
    ax_a.set_xlim(*Z_RANGE)
    ax_a.set_xticks(Z_TICKS)
    ax_a.set_xlabel(r"$z$  [pc]", labelpad=1)
    ax_a.set_ylabel(r"$\langle n_{\rm H}\rangle_{\rm area}$ [cm$^{-3}$]")
    panel_label(ax_a, "a", loc="upper left")
    ax_a.text(z_mid + 8, 1.6e-2, r"$z_{\rm mid}$", fontsize=TEXT_FS, color="0.35",
              ha="left", va="bottom")
    ax_a.text(CUBE_EDGE_PC - 8, N_YLIM[0] * 1.25, "cube edge", fontsize=TEXT_FS - 0.5, color="0.35",
              ha="right", va="bottom")
    leg = ax_a.legend([Line2D([], [], **DATA_KW), plt.Rectangle((0, 0), 1, 1, **BAND_KW),
                       Line2D([], [], **QUERY_KW)],
                      ["cube (4 pc bins)", "16-square error", "map query"],
                      loc="upper left", bbox_to_anchor=(0.0, 0.88), frameon=False,
                      fontsize=TEXT_FS, handlelength=1.6, borderaxespad=0.3)
    ax_a.add_artist(leg)
    ax_a.legend([Line2D([], [], color=MODEL_COLORS[m], lw=0.9) for m in models]
                + [Line2D([], [], color="0.3", lw=0.9, ls=BEYOND_LS)],
                [f"{MODEL_LABELS[m]} at its knee" for m in models] + ["beyond the knee"],
                loc="upper right", bbox_to_anchor=(1.0, 0.95), frameon=False,
                fontsize=TEXT_FS, handlelength=1.6, borderaxespad=0.3)

    # ------------------------------------------------------------------ (b)
    for side in ("upper", "lower"):
        for m in models:
            r = d[f"{side}_{m}_rchi2"]
            ax_b.plot(zg, r, color=MODEL_COLORS[m], ls=SIDE_LS[side], lw=1.0, zorder=3)
            zk = float(d[f"{side}_{m}_knee"][0])
            if np.isfinite(zk):
                i = int(np.argmin(np.abs(zg - zk)))
                ax_b.plot(zg[i], r[i], marker="o" if side == "upper" else "s", ms=3.8,
                          mfc=MODEL_COLORS[m], mec="white", mew=0.4, ls="none", zorder=4)
    ax_b.axvline(CUBE_EDGE_PC, **EDGE_KW)
    ax_b.set_yscale("log")
    ax_b.set_ylabel(r"reduced $\chi^2$")
    ax_b.set_xlabel(r"fit range $z_{\max}$ = max $|z - z_{\rm mid}|$  [pc]", labelpad=1)
    ax_b.tick_params(labelbottom=True)
    panel_label(ax_b, "b", loc="upper left")
    ax_b.legend([Line2D([], [], color=MODEL_COLORS[m], lw=1.0) for m in models]
                + [Line2D([], [], color="0.3", ls=SIDE_LS[s], lw=1.0) for s in ("upper", "lower")]
                + [Line2D([], [], marker="o", ms=3.0, mfc="0.3", mec="white", ls="none")],
                [MODEL_LABELS[m] for m in models] + ["upper", "lower", "knee"],
                loc="lower right", ncol=2, frameon=False, fontsize=TEXT_FS, handlelength=2.0,
                columnspacing=1.0, borderaxespad=0.3)

    # ------------------------------------------------------------------ (c)
    zt, a = d["z_top_grid"], d["zt_alpha_cell_mean"]
    k_lo, k_hi = (float(v) for v in d["knee_range"])
    ax_c.axvspan(k_lo, k_hi, **KNEE_BAND_KW)
    ax_c.text(0.5 * (k_lo + k_hi), 0.05, "knee range\nof (b)", transform=ax_c.get_xaxis_transform(),
              ha="center", va="bottom", fontsize=TEXT_FS - 0.5, color="0.4", linespacing=0.9)
    cube = zt <= CUBE_EDGE_PC + 1e-9
    qry = zt >= CUBE_EDGE_PC - 1e-9
    ax_c.plot(zt[cube], a[cube], color=COLOR_ALPHA, lw=ALPHA_LW, zorder=3)
    ax_c.plot(zt[qry], a[qry], color=COLOR_ALPHA, lw=ALPHA_LW, ls=BEYOND_LS, zorder=3)
    ax_c.axvline(CUBE_EDGE_PC, **EDGE_KW)
    for zz in ALPHA_MARKS_PC:
        j = int(np.argmin(np.abs(zt - zz)))
        ax_c.plot(zt[j], a[j], marker="o", ms=3.6, mfc=COLOR_ALPHA, mec="white", mew=0.5,
                  ls="none", zorder=4)
        lab = f"{a[j]:.2f}" + ("\n(fiducial)" if zz == CUBE_EDGE_PC else "")
        right = zz == 1000.0
        ax_c.annotate(lab, (zt[j], a[j]), xytext=(-3 if right else 4, -5), textcoords="offset points",
                      ha="right" if right else "left", va="top", fontsize=TEXT_FS, color="0.15", linespacing=0.9)
    ax_c.set_xlim(*X_RANGE_BC)
    lo, hi = np.nanmin(a), np.nanmax(a)
    ax_c.set_ylim(lo - 0.45 * (hi - lo), hi + 0.12 * (hi - lo))
    ax_c.set_ylabel(r"midplane $\langle\alpha_{\rm cell}\rangle$")
    ax_c.set_xlabel(r"integration height $z_{\rm top}$ ($\rho = 0$ at $|z| > z_{\rm top}$)  [pc]",
                    labelpad=1)
    panel_label(ax_c, "c", loc="upper left")
    ax_c.legend([Line2D([], [], color=COLOR_ALPHA, lw=ALPHA_LW),
                 Line2D([], [], color=COLOR_ALPHA, lw=ALPHA_LW, ls=BEYOND_LS)],
                ["cube only", "cube + map query"], loc="lower right", frameon=False,
                fontsize=TEXT_FS, handlelength=2.2, borderaxespad=0.3)

    fig.align_ylabels((ax_a, ax_b, ax_c))
    provisional_label(fig, loc="upper right")
    save_figure(fig, NAME)


if __name__ == "__main__":
    main()
