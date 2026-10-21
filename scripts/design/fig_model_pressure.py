"""
fig_model_pressure -- paper Fig. 1: the BS19 thermal-equilibrium model framed
in PRESSURE, with the real midplane gas on top.

Reads ONLY cache/model/model_pressure.npz (scripts/compute/compute_model_pressure.py).

PRESSURE CONVENTION: every pressure is the physical P_th/k = 1.1 n_H T
(helium included), INCLUDING the BS19 curves, P_min and P_max (the table's
n*T times 1.1), so that they share an axis with P_tot.

I_UV is F_UV / F_UV,sun with F_UV,sun = 2.7e-3 erg cm^-2 s^-1 (Shelest+26,
Methods) -- the Draine-field normalisation -- and is labelled that way.

  (a) P_th/k vs n_H at four I_UV spanning the midplane gas: stable branches
      solid, the unstable branch between the dP/dn = 0 turning points
      dashed, P_max and P_min marked with dots.
  (b) the two-phase band P_min(I_UV) - P_max(I_UV), with the midplane
      area-averaged <P_tot> and <1/P_th>^-1 as horizontal lines; a strip
      underneath with the mass-weighted I_UV distribution of the midplane
      gas (15-85 percentile range marked).
  (c) P_th/k over the (n_H, I_UV) plane with the phase boundaries
      n_W,max(I_UV) and n_C,min(I_UV), and the midplane gas as contours
      enclosing 50/80/95% of its mass.

Base style: the Extended Data Fig. 1 script's (as in
fig_bs19_curves_iuv1e3.py): serif / dejavuserif, axes.linewidth 0.8,
inward ticks mirrored on all four sides, log minor ticks.

LAYOUT "row" (kept): 1 x 3 at 7.1 in. LAYOUT "2+1": (a)|(b) on top, (c)
full width underneath. Run with --layout 2+1 to see the alternative.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.conventions import COLOR_PTH, COLOR_PTOT  # noqa: E402
from src.plotting.style import (  # noqa: E402
    apply_paper_style,
    panel_label,
    provisional_label,
    save_figure,
)

CACHE = Path("cache/model/model_pressure.npz")
NAME = "fig_model_pressure"
FIG_W = 7.1

# ---- axis ranges -----------------------------------------------------------
P_LIM = (1.5e2, 3e4)          # shared by (a) and (b)
N_LIM_A = (1e-2, 1e3)
IUV_LIM_B = (0.1, 10.0)
C_MARGIN_DEX = 0.25           # (c) range = 99.5% gas mass extent + this margin
C_IUV_RESAMPLE = 600          # display rows for the (c) image

# ---- styling ---------------------------------------------------------------
CURVE_CMAP = plt.cm.YlGnBu
CURVE_CMAP_RANGE = (0.45, 0.95)   # light -> dark with increasing I_UV
BAND_FILL = "0.88"
BAND_EDGE = "0.45"
GAS_LEVELS = (0.95, 0.80, 0.50)   # enclosed mass fractions, outer -> inner
GAS_COLOR = "white"
BOUNDARY_KW = dict(color="black", lw=1.1)
TEXT_FS = 6.5
LABEL_FS = 8.5
XLABEL_N = r"$n_{\rm H}$ [cm$^{-3}$]"
YLABEL_P = r"$P_{\rm th}/k$ [K cm$^{-3}$]"
LABEL_IUV = r"$I_{\rm UV}$ [$F_{\rm UV,\odot}$]"


def ed_style():
    apply_paper_style()
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.8, "axes.spines.top": True, "axes.spines.right": True,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "figure.constrained_layout.use": False,
    })


def log_axes(ax, x=True, y=True):
    subs = np.arange(2, 10) * 0.1
    if x:
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=subs, numticks=20))
        ax.xaxis.set_minor_formatter(NullFormatter())
    if y:
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=subs, numticks=20))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=7.5)


def enclosing_levels(H, fractions):
    """Density thresholds whose super-level sets enclose the given mass fractions."""
    v = np.sort(H.ravel())[::-1]
    c = np.cumsum(v) / v.sum()
    return [float(v[min(np.searchsorted(c, f), v.size - 1)]) for f in fractions]


def round100(v):
    return f"{int(round(v / 100.0) * 100):,}".replace(",", r"\,")


# ---------------------------------------------------------------------------
def panel_a(ax, d):
    k_n = len(d["iuv_curves"])
    cols = CURVE_CMAP(np.linspace(*CURVE_CMAP_RANGE, k_n))
    n = d["n_grid"]
    handles = []
    for k, (iv, c) in enumerate(zip(d["iuv_curves"], cols)):
        P = d[f"curve{k}__P"]
        nw, nc, pw, pc = d[f"curve{k}__turning"]
        stable = (n <= nw) | (n >= nc)
        warm, cold, unst = n <= nw, n >= nc, (n >= nw) & (n <= nc)
        for m, ls in ((warm, "-"), (cold, "-"), (unst, (0, (3.0, 1.6)))):
            ax.plot(n[m], P[m], color=c, lw=1.3, ls=ls, zorder=3)
        ax.plot([nw, nc], [pw, pc], ls="none", marker="o", ms=2.8, mfc=c, mec="black",
                mew=0.4, zorder=4)
        del stable
        handles.append(Line2D([], [], color=c, lw=1.3, label=rf"$I_{{\rm UV}} = {iv:g}$"))
    log_axes(ax)
    ax.set_xlim(N_LIM_A)
    ax.set_ylim(P_LIM)
    ax.set_xlabel(XLABEL_N, fontsize=LABEL_FS)
    ax.set_ylabel(YLABEL_P, fontsize=LABEL_FS)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.90), frameon=False,
              fontsize=TEXT_FS, handlelength=1.6, labelspacing=0.25, borderaxespad=0.3)
    ax.text(0.97, 0.04, r"dots: $P_{\rm max}$, $P_{\rm min}$" "\n" "dashed: unstable",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=TEXT_FS - 0.5,
            color="0.3")


def panel_b(ax, ax_s, d):
    iuv = d["iuv_fine"]
    pmin, pmax = d["P_min_fine"], d["P_max_fine"]
    ax.fill_between(iuv, pmin, pmax, color=BAND_FILL, lw=0, zorder=1)
    for p in (pmin, pmax):
        ax.plot(iuv, p, color=BAND_EDGE, lw=0.6, zorder=2)
    lx = 10 ** (0.55 * np.log10(IUV_LIM_B[0]) + 0.45 * np.log10(IUV_LIM_B[1]))
    li = np.searchsorted(iuv, lx)
    ax.text(lx, np.sqrt(pmin[li] * pmax[li]), "two-phase", ha="center", va="center",
            fontsize=TEXT_FS, color="0.25", zorder=3)
    ax.text(0.45, 0.95, "CNM only", transform=ax.transAxes, ha="center", va="top",
            fontsize=TEXT_FS, style="italic", color="0.45")
    ax.text(0.96, 0.04, "WNM only", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=TEXT_FS, style="italic", color="0.45")
    for val, c, lab in ((float(d["midplane_Ptot_area"][0]), COLOR_PTOT,
                         r"$\langle P_{\rm tot}\rangle$"),
                        (float(d["midplane_Pth_inv_inv"][0]), COLOR_PTH,
                         r"$\langle 1/P_{\rm th}\rangle^{-1}$")):
        ax.axhline(val, color=c, lw=1.2, zorder=4)
        ax.text(0.97, val * 1.08, f"{lab} $\\approx {round100(val)}$", color=c,
                transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=TEXT_FS, zorder=5)
    log_axes(ax)
    ax.set_xlim(IUV_LIM_B)
    ax.set_ylim(P_LIM)
    ax.tick_params(labelbottom=False, labelleft=False)

    # strip: mass-weighted I_UV distribution of the midplane gas
    e = 10 ** d["slab_logiuv_edges"]
    h = d["slab_H1_iuv_mass"]
    ax_s.stairs(h, e, fill=True, color="0.55", lw=0)
    p15, _, p85 = d["iuv_pct"]
    ax_s.axvspan(p15, p85, color=COLOR_PTH, alpha=0.12, lw=0)
    for p in (p15, p85):
        ax_s.axvline(p, color=COLOR_PTH, lw=0.6)
    log_axes(ax_s, y=False)
    ax_s.set_xlim(IUV_LIM_B)
    ax_s.set_ylim(0, h.max() * 1.15)
    ax_s.set_yticks([])
    ax_s.tick_params(axis="y", which="both", left=False, right=False)
    ax_s.set_xlabel(LABEL_IUV, fontsize=LABEL_FS)
    ax_s.text(0.97, 0.88, "midplane gas\n15-85%", transform=ax_s.transAxes, ha="right",
              va="top", fontsize=TEXT_FS - 1, color="0.3", linespacing=0.9)


def panel_c(fig, ax, cax, d):
    ln_e, li_e = d["slab_logn_edges"], d["slab_logiuv_edges"]
    H = gaussian_filter(d["slab_H2_mass"], 1.0)
    # (c) range: 99.5% of the gas mass, plus a margin
    lev995 = enclosing_levels(H, (0.995,))[0]
    m = H >= lev995
    rows, cols = np.where(m)
    xlim = 10 ** np.array([ln_e[rows.min()] - C_MARGIN_DEX, ln_e[rows.max() + 1] + C_MARGIN_DEX])
    ylim = 10 ** np.array([li_e[cols.min()] - C_MARGIN_DEX, li_e[cols.max() + 1] + C_MARGIN_DEX])
    n, iv = d["n_grid"], d["iuv_grid"]
    in_x, in_y = (n >= xlim[0]) & (n <= xlim[1]), (iv >= ylim[0] / 2) & (iv <= ylim[1] * 2)
    P = d["P_grid"]
    vis = P[np.ix_(in_y, in_x)]
    norm = LogNorm(10 ** np.floor(np.log10(np.nanmin(vis))), 10 ** np.ceil(np.log10(np.nanmax(vis))))
    # The table has 31 I_UV rows 0.2 dex apart; drawn as-is they show as
    # horizontal bands. Resample log P linearly in log I_UV (the same
    # interpolation the curves in (a) use) onto a fine grid for display.
    li_fine = np.linspace(np.log10(iv[0]), np.log10(iv[-1]), C_IUV_RESAMPLE)
    logP = np.log10(P)
    P_fine = 10 ** np.array([np.interp(li_fine, np.log10(iv), logP[:, j])
                              for j in range(n.size)]).T
    im = ax.pcolormesh(n, 10 ** li_fine, P_fine, cmap="cividis", norm=norm,
                       shading="nearest", rasterized=True, zorder=0)
    # phase boundaries
    ax.plot(d["n_w_max_fine"], d["iuv_fine"], **BOUNDARY_KW, zorder=3)
    ax.plot(d["n_c_min_fine"], d["iuv_fine"], **BOUNDARY_KW, zorder=3)
    # midplane gas
    nc = 10 ** (0.5 * (ln_e[:-1] + ln_e[1:]))
    ic = 10 ** (0.5 * (li_e[:-1] + li_e[1:]))
    levels = enclosing_levels(H, GAS_LEVELS)
    ax.contour(nc, ic, H.T, levels=levels, colors=GAS_COLOR, linewidths=(0.5, 0.7, 0.9),
               zorder=4)
    log_axes(ax)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(XLABEL_N, fontsize=LABEL_FS)
    ax.set_ylabel(LABEL_IUV, fontsize=LABEL_FS)
    # region labels at the top of the panel, placed between the boundaries
    y_lab = 10 ** (np.log10(ylim[1]) - 0.17 * np.log10(ylim[1] / ylim[0]))
    j = np.searchsorted(d["iuv_fine"], y_lab)
    nw, ncm = d["n_w_max_fine"][j], d["n_c_min_fine"][j]
    # UNM sits a little toward n_C,min: both boundaries lean right with
    # height, so the geometric mean crowds the n_W,max line at the text's base
    x_unm = 10 ** (0.4 * np.log10(nw) + 0.6 * np.log10(ncm))
    for x, lab in ((np.sqrt(xlim[0] * nw), "WNM"), (x_unm, "UNM"),
                   (np.sqrt(ncm * xlim[1]), "CNM")):
        ax.text(x, y_lab, lab, ha="center", va="center", fontsize=TEXT_FS, color="white",
                fontweight="bold", zorder=6)
    ax.legend(handles=[Line2D([], [], color=GAS_COLOR, lw=0.8)],
              labels=[r"midplane gas, $|z| \leq 30$ pc" "\n" "(50, 80, 95% of mass)"],
              loc="lower right", frameon=True, fontsize=TEXT_FS - 0.5, handlelength=1.4,
              facecolor="0.35", edgecolor="none", framealpha=0.8, labelcolor="white",
              borderpad=0.35)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(YLABEL_P, fontsize=LABEL_FS - 0.5, labelpad=2)
    cb.ax.tick_params(labelsize=7, direction="in", which="both")
    cb.outline.set_linewidth(0.6)
    return xlim, ylim


# ---------------------------------------------------------------------------
def build(layout):
    ed_style()
    d = np.load(CACHE, allow_pickle=False)
    if layout == "row":
        L, GAP_AB, GAP_BC, CBW, CBGAP, R = 0.50, 0.07, 0.56, 0.07, 0.05, 0.50
        w = (FIG_W - L - GAP_AB - GAP_BC - CBGAP - CBW - R) / 3
        hm, hs, gs = w, 0.17 * w, 0.03
        B, T = 0.36, 0.17       # T: room for PROVISIONAL above the colourbar
        fig_h = B + hs + gs + hm + T
        fig = plt.figure(figsize=(FIG_W, fig_h))

        def r(x, y, ww, hh):
            return (x / FIG_W, y / fig_h, ww / FIG_W, hh / fig_h)
        xa, xb = L, L + w + GAP_AB
        xc = xb + w + GAP_BC
        ax_a = fig.add_axes(r(xa, B + hs + gs, w, hm))
        ax_b = fig.add_axes(r(xb, B + hs + gs, w, hm))
        ax_s = fig.add_axes(r(xb, B, w, hs))
        ax_c = fig.add_axes(r(xc, B, w, hm + gs + hs))
        cax = fig.add_axes(r(xc + w + CBGAP, B, CBW, hm + gs + hs))
    else:   # "2+1"
        L, GAP_AB, R, CBW, CBGAP = 0.55, 0.07, 0.55, 0.08, 0.06
        w = (FIG_W - L - GAP_AB - R) / 2
        hm, hs, gs = 0.75 * w, 0.15 * 0.75 * w, 0.03
        wc = FIG_W - L - R - CBGAP - CBW + 0.0
        hc = 2.2
        B, MID, T = 0.38, 0.62, 0.08
        fig_h = B + hc + MID + hs + gs + hm + T
        fig = plt.figure(figsize=(FIG_W, fig_h))

        def r(x, y, ww, hh):
            return (x / FIG_W, y / fig_h, ww / FIG_W, hh / fig_h)
        yt = B + hc + MID
        ax_a = fig.add_axes(r(L, yt + hs + gs, w, hm))
        ax_b = fig.add_axes(r(L + w + GAP_AB, yt + hs + gs, w, hm))
        ax_s = fig.add_axes(r(L + w + GAP_AB, yt, w, hs))
        ax_c = fig.add_axes(r(L, B, wc, hc))
        cax = fig.add_axes(r(L + wc + CBGAP, B, CBW, hc))

    panel_a(ax_a, d)
    panel_b(ax_b, ax_s, d)
    xlim, ylim = panel_c(fig, ax_c, cax, d)
    for ax, letter in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c")):
        panel_label(ax, letter, loc="upper left", size=8.5, pad=0.035)
    provisional_label(fig, loc="upper right")
    print(f"  (c) ranges: n_H {xlim[0]:.3g} .. {xlim[1]:.3g} cm^-3, "
          f"I_UV {ylim[0]:.3g} .. {ylim[1]:.3g}")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=("row", "2+1"), default="row")
    args = ap.parse_args()
    fig = build(args.layout)
    save_figure(fig, NAME if args.layout == "row" else NAME + "_2plus1")


if __name__ == "__main__":
    main()
