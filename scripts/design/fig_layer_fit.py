"""
fig_layer_fit -- DIAGNOSTIC figure (not a paper figure): where the
footprint-mean density stops looking like a decaying gas layer.

Reads ONLY cache/layer_fit/layer_fit.npz (scripts/validation/layer_fit.py).
Three stacked panels:
  (a) <n_H>_area in 4 pc bins of |z - z_mid| on each side, plotted at signed
      z, with the 16-square error band, and the exp / gauss / sech2 fits at
      each model's knee z_max on each side (solid inside the knee, dashed
      beyond). Dotted verticals: z_mid (grey) and each model's knee (its colour).
  (b) reduced chi2 vs z_max, log y: solid = upper side, dashed = lower side.
  (c) fitted scale height h vs z_max, same encoding.
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
    panel_label,
    provisional_label,
    save_figure,
)

# ═══════════════════════════════════════════════════════════════════════
# ██  CONTROL BLOCK
# ═══════════════════════════════════════════════════════════════════════
CACHE = Path("cache/layer_fit/layer_fit.npz")
NAME = "fig_layer_fit"
FIG_W, FIG_H = 5.0, 6.6
Z_RANGE = (-750.0, 750.0)
ZMAX_RANGE = (100.0, 750.0)
N_YLIM = (5e-3, 3.0)
MODEL_COLORS = {"exp": "#0072B2", "gauss": "#D55E00", "sech2": "#009E73"}
MODEL_LABELS = {"exp": "exp", "gauss": "Gaussian", "sech2": r"sech$^2$"}
SIDE_LS = {"upper": "-", "lower": (0, (4.0, 2.0))}
SIGN = {"upper": 1.0, "lower": -1.0}
DATA_KW = dict(color="0.15", lw=1.0)
BAND_KW = dict(color="0.6", alpha=0.35, lw=0)
TEXT_FS = 6.5


def main():
    apply_paper_style()
    d = np.load(CACHE, allow_pickle=False)
    z_mid = float(d["z_mid"][0])
    models = [str(m) for m in d["models"]]
    zg = d["z_max_grid"]

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        3, 1, figsize=(FIG_W, FIG_H), gridspec_kw=dict(height_ratios=(1.3, 1.0, 1.0)))

    # ------------------------------------------------------------------ (a)
    for side in ("upper", "lower"):
        s = SIGN[side]
        u, n, e = d[f"{side}_u"], d[f"{side}_nH"], d[f"{side}_err_log"]
        zs = z_mid + s * u
        ax_a.fill_between(zs, n * 10 ** (-e), n * 10 ** e, **BAND_KW)
        ax_a.plot(zs, n, **DATA_KW, zorder=3)
        u_end = abs((Z_RANGE[1] if s > 0 else Z_RANGE[0]) - z_mid)
        for m in models:
            zk = d[f"{side}_{m}_knee"][0]      # (knee, min rchi2, z_max at min)
            if not np.isfinite(zk):
                continue
            i = int(np.argmin(np.abs(zg - zk)))
            n0, h = d[f"{side}_{m}_n0"][i], d[f"{side}_{m}_h"][i]
            col = MODEL_COLORS[m]
            for lo, hi, ls in ((0.0, zk, "-"), (zk, u_end, (0, (3.0, 1.6)))):
                uu = np.linspace(lo, hi, 300)
                ax_a.plot(z_mid + s * uu, model_density(m, uu, n0, h), color=col, lw=1.0,
                          ls=ls, zorder=4)
            ax_a.axvline(z_mid + s * zk, color=col, lw=0.7, ls=":", zorder=1)
    ax_a.axvline(z_mid, color="0.45", lw=0.7, ls=":", zorder=1)
    ax_a.set_yscale("log")
    ax_a.set_ylim(*N_YLIM)
    ax_a.set_xlim(*Z_RANGE)
    ax_a.set_xticks((-750, -500, -250, 0, 250, 500, 750))
    ax_a.set_xlabel(r"$z$  [pc]", labelpad=1)
    ax_a.set_ylabel(r"$\langle n_{\rm H}\rangle$ [cm$^{-3}$]")
    panel_label(ax_a, "a", loc="upper left")
    # upper corners: empty above the plateau on both sides of the peak
    leg = ax_a.legend([Line2D([], [], **DATA_KW), plt.Rectangle((0, 0), 1, 1, **BAND_KW)],
                      ["data (4 pc bins)", "16-square error"],
                      loc="upper left", bbox_to_anchor=(0.0, 0.88), frameon=False,
                      fontsize=TEXT_FS, handlelength=1.8)
    ax_a.add_artist(leg)
    ax_a.legend([Line2D([], [], color=MODEL_COLORS[m], lw=1.0) for m in models]
                + [Line2D([], [], color="0.3", lw=1.0, ls=(0, (3.0, 1.6)))],
                [f"{MODEL_LABELS[m]} fit at its knee" for m in models] + ["beyond the knee"],
                loc="upper right", bbox_to_anchor=(1.0, 0.95), frameon=False,
                fontsize=TEXT_FS, handlelength=1.8)

    # ------------------------------------------------------------- (b), (c)
    for ax, key in ((ax_b, "rchi2"), (ax_c, "h")):
        for side in ("upper", "lower"):
            for m in models:
                ax.plot(zg, d[f"{side}_{m}_{key}"], color=MODEL_COLORS[m], ls=SIDE_LS[side], lw=1.0)
        ax.set_xlim(*ZMAX_RANGE)
        ax.axvline(400.0, color="0.55", lw=0.6, ls=":", zorder=0)
    ax_b.set_yscale("log")
    ax_b.set_ylabel(r"reduced $\chi^2$")
    ax_b.set_xticklabels([])
    panel_label(ax_b, "b", loc="upper left")
    ax_c.set_ylabel(r"fitted $h$ [pc]")
    ax_c.set_xlabel(r"fit range $z_{\max}$ = max $|z - z_{\rm mid}|$  [pc]")
    panel_label(ax_c, "c", loc="upper left")
    side_handles = [Line2D([], [], color="0.3", ls=SIDE_LS[s], lw=1.0) for s in ("upper", "lower")]
    model_handles = [Line2D([], [], color=MODEL_COLORS[m], lw=1.0) for m in models]
    ax_b.legend(model_handles + side_handles,
                [MODEL_LABELS[m] for m in models] + [r"upper ($z > z_{\rm mid}$)",
                                                     r"lower ($z < z_{\rm mid}$)"],
                loc="lower right", ncol=2, frameon=False, fontsize=TEXT_FS, handlelength=2.2)

    fig.align_ylabels((ax_a, ax_b, ax_c))
    provisional_label(fig, loc="upper right")
    save_figure(fig, NAME)


def model_density(model, u, n0, h):
    if model == "exp":
        return n0 * np.exp(-u / h)
    if model == "gauss":
        return n0 * np.exp(-u ** 2 / (2.0 * h ** 2))
    return n0 / np.cosh(np.minimum(u / h, 350.0)) ** 2


if __name__ == "__main__":
    main()
