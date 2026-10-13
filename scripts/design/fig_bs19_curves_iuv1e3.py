"""
fig_bs19_curves_iuv1e3 -- the BS19 thermal-equilibrium model at
I_UV = 1e3, drawn in the design of the Extended Data Fig. 1 script
(research_10.0/2_plots_for_Shmuel/Bialy_model_figure.py).

Three stacked panels sharing the n axis:
  (a) P/k_B vs n   -- the original's own top panel
  (b) T vs n       -- same styling, plus light 300 K / 6000 K guides
  (c) I_UV vs n    -- the original's phase-classification panel, with
                      I_UV = 1e3 marked and I_UV = 1 marked lighter

Design taken from the original and reproduced here: the stacked layout
and its subplots_adjust margins, the 5.0 in width and the per-panel
aspect, the serif / dejavuserif rcParams and axes.linewidth, the phase
colours and their 0.30 background alpha, the 2.5 pt curve split into
WNM / UNM / CNM segments with the unstable stretch dashed, the
white-filled open turning-point markers and their n_W,max / n_C,min
annotations, the bold rotated in-panel phase labels, the "BS19's model"
corner label, the black turning-point loci with their rotated labels,
the "Phase classification" label, the dashed I_UV guide line, the dashed
vertical connectors running from the turning points down through every
panel and across the gaps, the x range, the inward ticks mirrored on all
four sides with log minor ticks, and the label / tick font sizes. There
is no legend, exactly as in the original: everything is labelled in the
panel.

What changed, and only this: the I_UV shown. The main curve is
I_UV = 1e3; I_UV = 1 is kept as the reference curve, and it is the one
that carries the phase segmentation, the markers and the phase labels.

Why: at I_UV = 1e3 the equilibrium curve has NO dP/dn = 0 turning
points. P(n) = n*T is strictly monotonic -- the radiation field washes
the S-curve out entirely -- so there is no thermally unstable range and
no two-phase medium. The original script's own turning-point finder
reaches the same conclusion on this row, independently of the project's.
Panel (c) shows the reason directly: the two-phase wedge narrows with
rising I_UV and closes at I_UV ~ 158, far below 1e3, so the 1e3 line
lies in the single-phase region above it.

The I_UV = 1e3 curve is therefore drawn UNSEGMENTED, in the single
colour its classification gives. With no thermal instability the gas
sits on one stable branch, and at 1800-7100 K over this n window that
branch is the warm one -- so it takes the WNM colour. Nothing is drawn
on it that does not exist there: no segments, no markers, no unstable
shading, no P_min / P_max.

Turning points come from src.physics.thermal.locate_scurve_turning_points
(linear root refinement on the raw curve), NOT from the original's
Savitzky-Golay smoothing. The two agree to about one grid cell at
I_UV = 1 (1.0994 vs 1.0715 and 7.4738 vs 7.4131 cm^-3, i.e. 2.6% and
0.8%, against a 2.3% grid spacing).

P/k here is n_H*T, the BS19 convention (`thermal.p_nT`), NOT the
physical 1.1 n_H k T: the table's P grid and its turning points are
tabulated in the n*T convention, so that is the only one that may be
drawn against them.

Data path comes from src.config_loader and the physics from
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import LogLocator  # noqa: E402
from scipy.io import loadmat  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config_loader import load_resolved_config  # noqa: E402
from src.conventions import (  # noqa: E402
    CNM,
    CNM_TEMP_MAX_K,
    UNM,
    WNM,
    WNM_TEMP_MIN_K,
)
from src.physics import thermal  # noqa: E402
from src.plotting.style import save_figure  # noqa: E402

NAME = "fig_bs19_curves_iuv1e3"

# ═══════════════════════════════════════════════════════════════════════
# ██  FIGURE GEOMETRY  (from the original; height scaled 2 -> 3 panels
# ██  so each panel keeps the original's aspect exactly)
# ═══════════════════════════════════════════════════════════════════════
FIG_W = 5.0
FIG_H = 9.45
DPI = 600
HSPACE = 0.10
ADJUST = dict(left=0.15, right=0.97, top=0.97, bottom=0.06, hspace=0.08)

# ═══════════════════════════════════════════════════════════════════════
# ██  PHASE COLORS  (identical values to the original's, taken from
# ██  src.conventions so this repo keeps one definition of them)
# ═══════════════════════════════════════════════════════════════════════
CNM_COLOR = np.array(CNM)
UNM_COLOR = np.array(UNM)
WNM_COLOR = np.array(WNM)

# ═══════════════════════════════════════════════════════════════════════
# ██  AXIS LIMITS
# ═══════════════════════════════════════════════════════════════════════
# n is the original's exactly. P is NOT, and cannot be: the original's
# (8e2, 1.2e4) is tuned to the I_UV = 1 curve, and over this same n
# window the I_UV = 1e3 curve runs 698 -> 8.8e4, off both ends. I_UV is
# likewise widened from the original's (1e-1, 1e1) because the line this
# figure exists to mark sits at 1e3.
N_LIM = (1e-1, 5e1)
P_LIM = (6e2, 1.0e5)
T_LIM = (7e1, 1.2e4)
IUV_LIM = (1e-1, 3e3)

# ═══════════════════════════════════════════════════════════════════════
# ██  CURVES
# ═══════════════════════════════════════════════════════════════════════
IUV_MAIN = 1.0e3
IUV_REF = 1.0

IUV_COLOR = "dimgray"
IUV_LW = 2.5
# The reference curve is drawn thinner than the main one. The original has
# only one curve and so only one width; with two, and with both carrying a
# WNM-red stretch that nearly coincides below n ~ 1, weight is what keeps
# "which curve am I looking at" answerable at a glance.
REF_LW = 1.8
IUV_ZO = 5
UNSTABLE_LS = "dashed"
# The I_UV = 1e3 curve: one stable branch, warm -- see the module docstring.
MAIN_COLOR = WNM_COLOR

SHOW_MARKERS = True
MARKER = "o"
MARKER_SIZE = 5.5
MARKER_LW = 1.0

# ═══════════════════════════════════════════════════════════════════════
# ██  TEMPERATURE GUIDES  (panel b)
# ═══════════════════════════════════════════════════════════════════════
TGUIDE_COLOR = "0.72"
TGUIDE_LW = 0.8
TGUIDE_FS = 7.0

# ═══════════════════════════════════════════════════════════════════════
# ██  LABELS  (font sizes from the original)
# ═══════════════════════════════════════════════════════════════════════
LABEL_FS = 11
TICK_FS = 9
PHASE_FS = 9
ANNOT_FS = 10.0
ANNOT_COL = "black"

# (anchor n, multiplicative offset off the curve) for the bold rotated
# phase labels. The original hardcodes an absolute (n, P) per label;
# with a different y range those land in the wrong place, so the anchor
# in n is kept and the height is taken off the curve itself.
PN_LABEL_WNM = (0.42, 1.90)
PN_LABEL_UNM = (2.75, 0.55)
PN_LABEL_CNM = (26.0, 1.75)

PN_ANNOT_NWMAX_POS = (0.80, 1.50)
PN_ANNOT_NCMIN_POS = (1.18, 0.88)

# Upper LEFT, not the original's upper right: at I_UV = 1e3 the curve
# climbs into the top-right corner the original leaves empty.
PN_MODEL_LABEL_X = 0.03
PN_MODEL_LABEL_Y = 0.96
PN_MODEL_LABEL_FS = 8.5

# in-panel curve labels, the original's idiom (no legend anywhere)
CURVE_LABEL_FS = 8.0
# (anchor n, offset) per panel: the two curves sit differently in P and in T
MAIN_LABEL_P = (5.0, 1.50)
REF_LABEL_P = (0.20, 0.56)
MAIN_LABEL_T = (12.0, 0.70)
REF_LABEL_T = (0.20, 1.17)

XLABEL_N = r"$n\ (\mathrm{cm}^{-3})$"
YLABEL_P = r"$P/k_B\ (\mathrm{K\ cm}^{-3})$"
YLABEL_T = r"$T\ (\mathrm{K})$"
YLABEL_IUV = r"$I_{\rm UV}$"

# ═══════════════════════════════════════════════════════════════════════
# ██  IUV-n PANEL  (from the original)
# ═══════════════════════════════════════════════════════════════════════
BG_ALPHA = 0.30
LOCI_COLOR = "black"
LOCI_LW = 1.6

IUV_REF_LINE_COLOR = "dimgray"
IUV_REF_LINE_LW = 0.9
IUV_REF_LINE_LS = "--"
IUV_MAIN_LINE_COLOR = "black"
IUV_MAIN_LINE_LW = 1.5
IUV_MAIN_LINE_LS = "--"

IUVN_LABEL_WNM = (0.45, 6.0)
IUVN_LABEL_UNM = (6.3, 6.0)
IUVN_LABEL_CNM = (30.0, 6.0)
LOCI_LABEL_IUV_W = 0.50
LOCI_LABEL_IUV_C = 0.50
LOCI_ANNOT_FS = 10.0

PHASE_CLASS_LABEL = (40.0, 0.125)
PHASE_CLASS_FS = 8.5

NO_UNSTABLE_FS = 8.0

# ═══════════════════════════════════════════════════════════════════════
# ██  VERTICAL CONNECTOR LINES  (from the original)
# ═══════════════════════════════════════════════════════════════════════
CONNECTOR_COLOR = "dimgray"
CONNECTOR_LW = 0.8
CONNECTOR_LS = (0, (4, 3))


# ═══════════════════════════════════════════════════════════════════════
# ██  DATA
# ═══════════════════════════════════════════════════════════════════════
def load_table():
    """(n grid, I_UV grid, T_2d_n) from the BS19 .mat, n ascending.

    Path via src.config_loader, never hardcoded. The original builds a
    bicubic RectBivariateSpline over (log I, log n) so it can evaluate
    any I_UV; both I_UV values here are exact grid rows (the grid is
    10^(3 - 0.2k), k = 0..30), and an interpolating spline reproduces
    its own nodes, so reading the row directly is the same number
    without the machinery.
    """
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
    """(T, P, turning_points, provenance) at one I_UV.

    turning_points is thermal.locate_scurve_turning_points' result --
    (n_W,max, n_C,min, P at each) or None when the curve has no
    dP/dn = 0 sign changes, which is the case at I_UV = 1e3. Refuses to
    extrapolate outside the tabulated I_UV range.
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

    return (t, thermal.p_nT(n_grid, t),
            thermal.locate_scurve_turning_points(n_grid, t), prov)


def loci_from_thermal():
    """The turning-point loci n_W,max(I_UV), n_C,min(I_UV).

    From src.physics.thermal.turning_point_densities -- the project's
    own finder, not the original's Savitzky-Golay one. Rows with no
    S-curve come back NaN and are dropped here rather than filled, so
    the loci simply STOP at the highest I_UV that has a two-phase range.
    That stop is the point of panel (c).
    """
    iuv, n_w, n_c, _, _ = thermal.turning_point_densities(
        load_resolved_config()["bs19_mat_path"])
    good = np.isfinite(n_w) & np.isfinite(n_c)
    order = np.argsort(iuv[good])
    return iuv[good][order], n_w[good][order], n_c[good][order]


# ═══════════════════════════════════════════════════════════════════════
# ██  HELPERS
# ═══════════════════════════════════════════════════════════════════════
def set_log_minorticks(ax):
    """The original's minor-tick setup, verbatim."""
    subs = np.arange(2, 10) * 0.1
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=subs, numticks=20))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=subs, numticks=20))


def style_axis(ax, ylabel, ylim, bottom_panel):
    """Everything the original applies to each of its panels."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(N_LIM)
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(axis="both", which="both", direction="in", labelsize=TICK_FS)
    ax.tick_params(right=True, top=True, which="both")
    if bottom_panel:
        ax.set_xlabel(XLABEL_N, fontsize=LABEL_FS)
    else:
        ax.tick_params(axis="x", which="both", labelbottom=False)
    set_log_minorticks(ax)


def log_interp(x_arr, y_arr, x_at):
    """y(x_at) by linear interpolation in log-log."""
    return 10.0 ** np.interp(np.log10(x_at), np.log10(x_arr),
                             np.log10(np.clip(y_arr, 1e-30, None)))


def screen_angle(ax, x_arr, y_arr, x_at, frac=0.12):
    """Rotation in degrees laying text along a curve as DRAWN.

    The original hardcodes a rotation per label, tuned to its own axis
    ranges and panel aspect. Two of this figure's three y ranges differ
    (see AXIS LIMITS), which changes every on-screen slope, so the angle
    is measured off the rendered geometry rather than re-guessed.
    """
    lx0, lx1 = np.log10(ax.get_xlim())
    dlx = frac * (lx1 - lx0)
    xa = 10.0 ** (np.log10(x_at) - dlx / 2)
    xb = 10.0 ** (np.log10(x_at) + dlx / 2)
    ya, yb = log_interp(x_arr, y_arr, xa), log_interp(x_arr, y_arr, xb)
    ly0, ly1 = np.log10(ax.get_ylim())
    bbox = ax.get_window_extent()
    dx_px = (np.log10(xb) - np.log10(xa)) / (lx1 - lx0) * bbox.width
    dy_px = (np.log10(yb) - np.log10(ya)) / (ly1 - ly0) * bbox.height
    return np.degrees(np.arctan2(dy_px, dx_px))


def screen_angle_xy(ax, x_arr, y_arr, y_at, frac=0.12):
    """screen_angle for a curve parameterised by its y (the loci)."""
    ly0, ly1 = np.log10(ax.get_ylim())
    dly = frac * (ly1 - ly0)
    ya = 10.0 ** (np.log10(y_at) - dly / 2)
    yb = 10.0 ** (np.log10(y_at) + dly / 2)
    xa, xb = log_interp(y_arr, x_arr, ya), log_interp(y_arr, x_arr, yb)
    lx0, lx1 = np.log10(ax.get_xlim())
    bbox = ax.get_window_extent()
    dx_px = (np.log10(xb) - np.log10(xa)) / (lx1 - lx0) * bbox.width
    dy_px = (np.log10(yb) - np.log10(ya)) / (ly1 - ly0) * bbox.height
    return np.degrees(np.arctan2(dy_px, dx_px))


# ═══════════════════════════════════════════════════════════════════════
# ██  DRAW A CURVE PANEL  (a and b)
# ═══════════════════════════════════════════════════════════════════════
def draw_curve_panel(ax, n_grid, y_main, y_ref, tp_ref, phase_labels,
                       main_label, ref_label):
    """The I_UV = 1e3 curve plus the segmented I_UV = 1 reference curve,
    in the original's P-n panel style.

    The main curve is one unsegmented line: at I_UV = 1e3 there are no
    turning points, so there is nothing to segment it at.
    """
    ax.plot(n_grid, y_main, color=MAIN_COLOR, lw=IUV_LW, ls="solid",
            zorder=IUV_ZO)

    n_w, n_c = tp_ref[0], tp_ref[1]
    lo = int(np.searchsorted(n_grid, n_w))
    hi = int(np.searchsorted(n_grid, n_c))
    for ns, ys, unstable, color in (
        (n_grid[:lo + 1], y_ref[:lo + 1], False, WNM_COLOR),
        (n_grid[lo:hi + 1], y_ref[lo:hi + 1], True, IUV_COLOR),
        (n_grid[hi:], y_ref[hi:], False, CNM_COLOR),
    ):
        ax.plot(ns, ys, color=color, lw=IUV_LW,
                ls=UNSTABLE_LS if unstable else "solid", zorder=IUV_ZO)

    if SHOW_MARKERS:
        for n_val in (n_w, n_c):
            ax.plot(n_val, log_interp(n_grid, y_ref, n_val), marker=MARKER,
                    ms=MARKER_SIZE, color=IUV_COLOR, mfc="white",
                    mew=MARKER_LW, ls="none", zorder=IUV_ZO + 1)

    if phase_labels:
        for (n_l, off), txt, col in (
            (PN_LABEL_WNM, "WNM", WNM_COLOR),
            (PN_LABEL_UNM, "UNM", IUV_COLOR),
            (PN_LABEL_CNM, "CNM", CNM_COLOR),
        ):
            ax.text(n_l, log_interp(n_grid, y_ref, n_l) * off, txt,
                    fontsize=PHASE_FS, color=col,
                    rotation=screen_angle(ax, n_grid, y_ref, n_l),
                    rotation_mode="anchor", ha="center", va="center",
                    fontweight="bold", zorder=15)

        ax.text(n_w * PN_ANNOT_NWMAX_POS[0],
                log_interp(n_grid, y_ref, n_w) * PN_ANNOT_NWMAX_POS[1],
                r"$n_{W,\rm max}$", fontsize=ANNOT_FS, color=ANNOT_COL,
                ha="center", va="bottom", zorder=20)
        ax.text(n_c * PN_ANNOT_NCMIN_POS[0],
                log_interp(n_grid, y_ref, n_c) * PN_ANNOT_NCMIN_POS[1],
                r"$n_{C,\rm min}$", fontsize=ANNOT_FS, color=ANNOT_COL,
                ha="left", va="center", zorder=20)

    # both curves named in the panel -- the original's idiom, no legend
    for (n_l, off), y_arr, txt in ((main_label, y_main, r"$I_{\rm UV}=10^{3}$"),
                                     (ref_label, y_ref, r"$I_{\rm UV}=1$")):
        ax.text(n_l, log_interp(n_grid, y_arr, n_l) * off, txt,
                fontsize=CURVE_LABEL_FS, color=ANNOT_COL,
                rotation=screen_angle(ax, n_grid, y_arr, n_l),
                rotation_mode="anchor", ha="center", va="center", zorder=20)


# ═══════════════════════════════════════════════════════════════════════
# ██  DRAW THE IUV-n PANEL  (c)
# ═══════════════════════════════════════════════════════════════════════
def draw_iuvn_panel(ax, iuv_loci, n_w_loci, n_c_loci):
    """The original's phase-classification panel, over a wider I_UV range.

    The shaded wedges and the loci are drawn ONLY where the table
    actually has a two-phase range. Above the last S-curve row the
    original's constant fill_value would extend them flat; here they
    simply stop, because that is what the model does -- and it is why
    the I_UV = 1e3 line has no unstable range to cross.
    """
    iuv_top = float(iuv_loci.max())
    iuv_bg = np.logspace(np.log10(IUV_LIM[0]), np.log10(iuv_top), 600)
    n_wb = log_interp(iuv_loci, n_w_loci, iuv_bg)
    n_cb = log_interp(iuv_loci, n_c_loci, iuv_bg)

    ax.fill_betweenx(iuv_bg, N_LIM[0] / 20, n_wb, color=WNM_COLOR,
                     alpha=BG_ALPHA, zorder=0, linewidth=0)
    ax.fill_betweenx(iuv_bg, n_cb, N_LIM[1] * 20, color=CNM_COLOR,
                     alpha=BG_ALPHA, zorder=1, linewidth=0)
    ax.fill_betweenx(iuv_bg, n_wb, n_cb, color=UNM_COLOR,
                     alpha=BG_ALPHA, zorder=2, linewidth=0)

    ax.plot(n_wb, iuv_bg, color=LOCI_COLOR, lw=LOCI_LW, ls="-", zorder=5)
    ax.plot(n_cb, iuv_bg, color=LOCI_COLOR, lw=LOCI_LW, ls="-", zorder=5)
    # the wedge closes here: mark where the two-phase range ends
    ax.plot([n_wb[-1], n_cb[-1]], [iuv_top, iuv_top], color=LOCI_COLOR,
            lw=LOCI_LW, ls="-", zorder=5)

    for y_at, arr, txt in (
        (LOCI_LABEL_IUV_W, n_w_loci, r"$n_{W,\rm max}(I_{\rm UV})$"),
        (LOCI_LABEL_IUV_C, n_c_loci, r"$n_{C,\rm min}(I_{\rm UV})$"),
    ):
        ax.text(log_interp(iuv_loci, arr, y_at) * 1.38, y_at, txt,
                fontsize=LOCI_ANNOT_FS, color=ANNOT_COL,
                rotation=screen_angle_xy(ax, arr, iuv_loci, y_at),
                rotation_mode="anchor", ha="left", va="center", zorder=12)

    ax.axhline(y=IUV_REF, color=IUV_REF_LINE_COLOR, lw=IUV_REF_LINE_LW,
               ls=IUV_REF_LINE_LS, zorder=4)
    ax.text(N_LIM[1] * 0.80, IUV_REF * 0.80, r"$I_{\rm UV}=1$",
            fontsize=CURVE_LABEL_FS, color=ANNOT_COL, ha="right", va="top",
            zorder=12)

    ax.axhline(y=IUV_MAIN, color=IUV_MAIN_LINE_COLOR, lw=IUV_MAIN_LINE_LW,
               ls=IUV_MAIN_LINE_LS, zorder=6)
    ax.text(N_LIM[1] * 0.80, IUV_MAIN * 1.25, r"$I_{\rm UV}=10^{3}$",
            fontsize=CURVE_LABEL_FS + 1.0, color=ANNOT_COL, ha="right",
            va="bottom", zorder=12, fontweight="bold")

    # Two lines, right-aligned: the single-phase strip above the wedge is
    # crossed by both connectors, and its right end is the only part wide
    # enough to hold the note without running into one.
    ax.text(N_LIM[1] * 0.92, 330.0,
            f"no unstable range\nabove $I_{{\\rm UV}}\\approx{iuv_top:.0f}$",
            fontsize=NO_UNSTABLE_FS, color=ANNOT_COL, ha="right",
            va="center", linespacing=1.35, zorder=12)

    for (n_l, i_l), txt, arr in (
        (IUVN_LABEL_WNM, "warm", n_w_loci),
        (IUVN_LABEL_UNM, "unstable", n_w_loci),
        (IUVN_LABEL_CNM, "cold", n_c_loci),
    ):
        ax.text(n_l, i_l, txt, fontsize=PHASE_FS, color="black",
                rotation=screen_angle_xy(ax, arr, iuv_loci, i_l),
                rotation_mode="anchor", ha="center", va="center",
                fontweight="bold", zorder=10)

    ax.text(*PHASE_CLASS_LABEL, "Phase classification", fontsize=PHASE_CLASS_FS,
            color="black", ha="right", va="bottom", zorder=10)


# ═══════════════════════════════════════════════════════════════════════
# ██  CONNECTORS  (the original's, extended to three panels)
# ═══════════════════════════════════════════════════════════════════════
def draw_connectors(axes, n_grid, y_by_ax, tp_ref):
    ax_p, ax_t, ax_i = axes
    for n_val in tp_ref[:2]:
        if not (N_LIM[0] <= n_val <= N_LIM[1]):
            continue
        ax_p.plot([n_val, n_val],
                  [log_interp(n_grid, y_by_ax[0], n_val), P_LIM[0]],
                  color=CONNECTOR_COLOR, lw=CONNECTOR_LW, ls=CONNECTOR_LS,
                  zorder=0, clip_on=True)
        ax_t.plot([n_val, n_val], [T_LIM[1], T_LIM[0]],
                  color=CONNECTOR_COLOR, lw=CONNECTOR_LW, ls=CONNECTOR_LS,
                  zorder=0, clip_on=True)
        ax_i.plot([n_val, n_val], [IUV_LIM[1], IUV_REF],
                  color=CONNECTOR_COLOR, lw=CONNECTOR_LW, ls=CONNECTOR_LS,
                  zorder=7, clip_on=True)


def gap_line(fig, ax_above, ax_below, n_val):
    """The original's _gap_line: continue a connector across a panel gap."""
    pos_a, pos_b = ax_above.get_position(), ax_below.get_position()
    lx0, lx1 = np.log10(N_LIM)
    xf = (np.log10(n_val) - lx0) / (lx1 - lx0)
    xfig = pos_a.x0 + xf * pos_a.width
    fig.add_artist(Line2D([xfig, xfig], [pos_a.y0, pos_b.y1],
                          transform=fig.transFigure, color=CONNECTOR_COLOR,
                          lw=CONNECTOR_LW, ls=CONNECTOR_LS, zorder=10,
                          clip_on=False))


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
    t_main, p_main, tp_main, prov_main = curve(n_grid, iuv_grid, t_2d_n, IUV_MAIN)
    t_ref, p_ref, tp_ref, prov_ref = curve(n_grid, iuv_grid, t_2d_n, IUV_REF)
    iuv_loci, n_w_loci, n_c_loci = loci_from_thermal()

    print(f"I_UV = {IUV_MAIN:g}: {prov_main}")
    print(f"I_UV = {IUV_REF:g}:    {prov_ref}")
    print(f"I_UV = {IUV_MAIN:g}: turning points -> {tp_main}")
    print(f"I_UV = {IUV_REF:g}:    n_W,max = {tp_ref[0]:.4f}, "
          f"n_C,min = {tp_ref[1]:.4f} cm^-3 "
          f"(P = {tp_ref[2]:.1f}, {tp_ref[3]:.1f} K cm^-3)")
    print(f"loci: {len(iuv_loci)} rows with an S-curve, "
          f"I_UV {iuv_loci.min():g} .. {iuv_loci.max():g}")
    if tp_main is not None:
        raise RuntimeError(
            f"I_UV = {IUV_MAIN:g} now reports turning points; this figure is "
            "built on their absence (the main curve is drawn unsegmented and "
            "unmarked). Rebuild it deliberately rather than let the claim go "
            "stale.")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(3, 1, hspace=HSPACE, figure=fig)
    ax_p = fig.add_subplot(gs[0])
    ax_t = fig.add_subplot(gs[1])
    ax_i = fig.add_subplot(gs[2])
    for ax in (ax_p, ax_t, ax_i):
        ax.set_facecolor("white")

    style_axis(ax_p, YLABEL_P, P_LIM, bottom_panel=False)
    style_axis(ax_t, YLABEL_T, T_LIM, bottom_panel=False)
    style_axis(ax_i, YLABEL_IUV, IUV_LIM, bottom_panel=True)
    fig.subplots_adjust(**ADJUST)

    draw_curve_panel(ax_p, n_grid, p_main, p_ref, tp_ref, phase_labels=True,
                     main_label=MAIN_LABEL_P, ref_label=REF_LABEL_P)

    # Guides labelled at the RIGHT edge: at n = 50 both curves are clear of
    # both cuts, which is not true at the left edge where they bunch up.
    for t_cut in (CNM_TEMP_MAX_K, WNM_TEMP_MIN_K):
        ax_t.axhline(t_cut, color=TGUIDE_COLOR, lw=TGUIDE_LW, zorder=0)
        ax_t.text(N_LIM[1] * 0.95, t_cut * 1.07, f"{t_cut:g} K",
                  fontsize=TGUIDE_FS, color="0.45", ha="right", va="bottom",
                  zorder=3)
    draw_curve_panel(ax_t, n_grid, t_main, t_ref, tp_ref, phase_labels=False,
                     main_label=MAIN_LABEL_T, ref_label=REF_LABEL_T)

    draw_iuvn_panel(ax_i, iuv_loci, n_w_loci, n_c_loci)

    ax_p.text(PN_MODEL_LABEL_X, PN_MODEL_LABEL_Y,
              "BS19's model\n$(I_{\\rm UV}=10^{3})$",
              fontsize=PN_MODEL_LABEL_FS, color="black", ha="left", va="top",
              transform=ax_p.transAxes, zorder=20)

    draw_connectors((ax_p, ax_t, ax_i), n_grid, (p_ref, t_ref, None), tp_ref)
    for n_val in tp_ref[:2]:
        if N_LIM[0] <= n_val <= N_LIM[1]:
            gap_line(fig, ax_p, ax_t, n_val)
            gap_line(fig, ax_t, ax_i, n_val)

    save_figure(fig, NAME, dpi=DPI)


if __name__ == "__main__":
    main()
