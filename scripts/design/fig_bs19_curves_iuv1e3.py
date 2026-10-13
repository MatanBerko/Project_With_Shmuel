"""
fig_bs19_curves_iuv1e3 -- the BS19 thermal-equilibrium curves at
I_UV = 1e3: P/k_B vs n and T vs n, and nothing else.

Two stacked panels sharing the n axis, over the table's full density
range, one plain dark curve each. The base style is the Extended Data
Fig. 1 script's (research_10.0/2_plots_for_Shmuel/Bialy_model_figure.py,
read only, untouched): 5 in wide, serif / dejavuserif, axes.linewidth
0.8, inward ticks mirrored on all four sides with log minor ticks, the
x label on the bottom panel only, and its axis-label text.

Deliberately NOT drawn: phase colours and segmentation, turning-point
markers, the n_W,max / n_C,min annotations, the I_UV = 1 reference
curve, the connector lines, the phase-classification panel, guide lines,
shading and legends. At I_UV = 1e3 the equilibrium curve has no
dP/dn = 0 turning points -- P(n) = n*T is strictly monotonic, the
radiation field having washed the S-curve out entirely -- so there is no
two-phase structure for any of that to describe. The script still checks
that (via src.physics.thermal) and prints the answer, but the figure
makes no claim about phases either way.

P/k here is n_H*T, the BS19 convention (`thermal.p_nT`), NOT the
physical 1.1 n_H k T: the table is tabulated in the n*T convention.

The table path comes from src.config_loader and the physics from
src.physics.thermal -- no hardcoded paths, nothing reimplemented.
Reading the .mat directly is a deliberate exception to the "design
scripts read only caches" rule: the subject of this figure IS the model
table, not anything derived from the density cube.

No PROVISIONAL label: that marker is about the f98 density cube's
provenance, and nothing here comes from the cube.
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import LogLocator  # noqa: E402
from scipy.io import loadmat  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config_loader import load_resolved_config  # noqa: E402
from src.physics import thermal  # noqa: E402
from src.plotting.style import save_figure  # noqa: E402

NAME = "fig_bs19_curves_iuv1e3"

# ═══════════════════════════════════════════════════════════════════════
# ██  FIGURE GEOMETRY  (from the original)
# ═══════════════════════════════════════════════════════════════════════
FIG_W = 5.0
FIG_H = 6.3
DPI = 600
HSPACE = 0.08
ADJUST = dict(left=0.17, right=0.97, top=0.97, bottom=0.09, hspace=0.08)

# ═══════════════════════════════════════════════════════════════════════
# ██  CURVE
# ═══════════════════════════════════════════════════════════════════════
IUV_MAIN = 1.0e3
CURVE_COLOR = "black"
CURVE_LW = 2.2
CURVE_ZO = 5

# Fitted to the curve, in decades, rather than hardcoded: the two panels
# span very different ranges and the point is simply to see the curve.
Y_PAD_DEX = 0.06

# ═══════════════════════════════════════════════════════════════════════
# ██  LABELS  (font sizes from the original)
# ═══════════════════════════════════════════════════════════════════════
LABEL_FS = 11
TICK_FS = 9
IUV_LABEL_FS = 9.5
IUV_LABEL_XY = (0.04, 0.94)

XLABEL_N = r"$n\ (\mathrm{cm}^{-3})$"
YLABEL_P = r"$P/k_B\ (\mathrm{K\ cm}^{-3})$"
YLABEL_T = r"$T\ (\mathrm{K})$"


# ═══════════════════════════════════════════════════════════════════════
# ██  DATA
# ═══════════════════════════════════════════════════════════════════════
def load_table():
    """(n grid, I_UV grid, T_2d_n) from the BS19 .mat, n ascending."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = loadmat(str(load_resolved_config()["bs19_mat_path"]))
    iuv = np.array(d["IUV_"]).squeeze().astype(float).ravel()
    n = np.array(d["n_"]).squeeze().astype(float).ravel()
    t2n = np.array(d["T_2d_n"]).astype(float)
    if np.any(np.diff(n) <= 0):
        order = np.argsort(n)
        n, t2n = n[order], t2n[:, order]
    return n, iuv, t2n


def curve(n_grid, iuv_grid, t_2d_n, iuv_value, tol=1e-9):
    """(T, P, provenance) at one I_UV.

    I_UV = 1e3 is an exact grid row (the grid is 10^(3 - 0.2k),
    k = 0..30), so no interpolation happens here; the log10(I_UV)-linear
    branch exists for any other value, and anything outside the
    tabulated range is refused rather than extrapolated.
    """
    lo, hi = iuv_grid.min(), iuv_grid.max()
    if not (lo <= iuv_value <= hi):
        raise ValueError(
            f"I_UV = {iuv_value:g} is outside the table's range "
            f"[{lo:g}, {hi:g}]; refusing to extrapolate.")

    j = int(np.argmin(np.abs(np.log10(iuv_grid) - np.log10(iuv_value))))
    if abs(np.log10(iuv_grid[j]) - np.log10(iuv_value)) < tol:
        t = t_2d_n[j]
        prov = f"exact grid row {j} (I_UV = {iuv_grid[j]:g})"
    else:
        order = np.argsort(iuv_grid)
        li = np.log10(iuv_grid[order])
        k = int(np.searchsorted(li, np.log10(iuv_value)))
        a, b = order[k - 1], order[k]
        w = ((np.log10(iuv_value) - np.log10(iuv_grid[a]))
             / (np.log10(iuv_grid[b]) - np.log10(iuv_grid[a])))
        t = (1.0 - w) * t_2d_n[a] + w * t_2d_n[b]
        prov = (f"log10(I_UV)-linear blend of rows {a} (I_UV = "
                f"{iuv_grid[a]:g}) and {b} (I_UV = {iuv_grid[b]:g}), w = {w:.3f}")

    return t, thermal.p_nT(n_grid, t), prov


# ═══════════════════════════════════════════════════════════════════════
# ██  STYLE
# ═══════════════════════════════════════════════════════════════════════
def set_log_minorticks(ax):
    """The original's minor-tick setup, verbatim."""
    subs = np.arange(2, 10) * 0.1
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=subs, numticks=20))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=subs, numticks=20))


def label_short_log_axis(ax, lo, hi, min_decade_ticks=3):
    """Label 1-2-5 per decade when too few decades fall inside the range.

    The T panel spans 263-8678 K, inside which the only decade tick is
    10^3 -- one labelled tick for the whole axis. The P panel spans five
    decades and needs nothing. So this triggers on the span rather than
    being applied blanket, and leaves the original's tick styling (and
    its unlabelled log minor ticks) otherwise untouched.
    """
    decades = [10.0 ** k for k in range(int(np.floor(np.log10(lo))),
                                          int(np.ceil(np.log10(hi))) + 1)]
    if sum(lo <= d <= hi for d in decades) >= min_decade_ticks:
        return
    ticks = [m * d for d in decades for m in (1, 2, 5) if lo <= m * d <= hi]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])


def fitted_ylim(y, pad_dex=Y_PAD_DEX):
    v = np.asarray(y, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    return 10.0 ** (np.log10(v.min()) - pad_dex), 10.0 ** (np.log10(v.max()) + pad_dex)


def style_axis(ax, n_lim, ylabel, ylim, bottom_panel):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(n_lim)
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(axis="both", which="both", direction="in", labelsize=TICK_FS)
    ax.tick_params(right=True, top=True, which="both")
    if bottom_panel:
        ax.set_xlabel(XLABEL_N, fontsize=LABEL_FS)
    else:
        ax.tick_params(axis="x", which="both", labelbottom=False)
    set_log_minorticks(ax)
    label_short_log_axis(ax, *ylim)


# ═══════════════════════════════════════════════════════════════════════
# ██  MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.8,
    })

    n_grid, iuv_grid, t_2d_n = load_table()
    t, p, prov = curve(n_grid, iuv_grid, t_2d_n, IUV_MAIN)
    n_lim = (float(n_grid.min()), float(n_grid.max()))

    print(f"I_UV = {IUV_MAIN:g}: {prov}")
    print(f"  n   {n_lim[0]:.4g} .. {n_lim[1]:.4g} cm^-3 ({n_grid.size} points)")
    print(f"  P/k {p.min():.4g} .. {p.max():.4g} K cm^-3")
    print(f"  T   {t.min():.4g} .. {t.max():.4g} K")
    # Not drawn, only reported: there is no two-phase structure at this I_UV.
    tp = thermal.locate_scurve_turning_points(n_grid, t)
    print(f"  dP/dn = 0 turning points: {tp if tp is not None else 'none (monotonic)'}")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 1, hspace=HSPACE, figure=fig)
    ax_p = fig.add_subplot(gs[0])
    ax_t = fig.add_subplot(gs[1])
    for ax in (ax_p, ax_t):
        ax.set_facecolor("white")

    ax_p.plot(n_grid, p, color=CURVE_COLOR, lw=CURVE_LW, ls="solid", zorder=CURVE_ZO)
    ax_t.plot(n_grid, t, color=CURVE_COLOR, lw=CURVE_LW, ls="solid", zorder=CURVE_ZO)

    style_axis(ax_p, n_lim, YLABEL_P, fitted_ylim(p), bottom_panel=False)
    style_axis(ax_t, n_lim, YLABEL_T, fitted_ylim(t), bottom_panel=True)
    fig.subplots_adjust(**ADJUST)

    ax_p.text(*IUV_LABEL_XY, r"$I_{\rm UV}=10^{3}$", fontsize=IUV_LABEL_FS,
              color="black", ha="left", va="top", transform=ax_p.transAxes,
              zorder=20)

    save_figure(fig, NAME, dpi=DPI)


if __name__ == "__main__":
    main()
