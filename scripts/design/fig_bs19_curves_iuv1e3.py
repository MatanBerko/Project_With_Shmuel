"""
fig_bs19_curves_iuv1e3 -- the BS19 thermal-equilibrium curves T(n) and
P/k(n) at I_UV = 1e3, with I_UV = 1 for reference.

Reads the BS19 table (Temp_Bialy_interp.mat) through src.physics.thermal
and src.config_loader. This is a deliberate exception to the "design
scripts read only caches" rule: the subject of this figure IS the model
table, not anything derived from the density cube. No cube data and no
cache is touched, and no physics is computed here that
src.physics.thermal does not already provide.

P/k on this figure is n_H * T -- the BS19 convention (`thermal.p_nT`),
NOT the physical 1.1 n_H k T. The table's own P_ grid, and therefore
P_min/P_max, are tabulated in the n*T convention, so this is the only
one that may be drawn against them.

THE HEADLINE CAVEAT, which the figure states on its face
---------------------------------------------------------
At I_UV = 1e3 the BS19 equilibrium curve has NO dP/dn = 0 turning points
at all: P(n) = n*T is strictly monotonic, so there is no thermally
unstable range and no two-phase medium. The radiation field is strong
enough to wash the S-curve out entirely.

That means P_min, P_max, n_W,max and n_C,min DO NOT EXIST at I_UV = 1e3.
What `build_pmin_pmax()` and `build_phase_density_bounds()` return there
is their documented constant EDGE FILL -- the value carried over from the
highest tabulated I_UV that still has an S-curve:

    n_W,max, n_C,min   edge fill from I_UV = 158.5  (last S-curve row of
                       the T_2d_n tabulation)
    P_min, P_max       edge fill from I_UV = 63.1   (last row of the
                       T_2d_P tabulation that still has a dn/dP sign
                       change; a different row, because these come from
                       a different tabulation of the same curve)

They are drawn because they were asked for, but in a distinct "carried
over" style and labelled as such, because a reader who took them for
features of the I_UV = 1e3 curve would be reading a 6-16x weaker
radiation field's numbers off this one. The genuine turning points of the
I_UV = 1 curve are marked on that curve for contrast -- at I_UV = 1 the
S-curve is real and the two-phase range is visible as the non-monotonic
stretch of panel (b).

No PROVISIONAL label: that marker is about the f98 density cube's
provenance (orientation, n_H/A' factor), and nothing here comes from the
cube.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.io  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config_loader import load_resolved_config  # noqa: E402
from src.conventions import (  # noqa: E402
    CNM_TEMP_MAX_K,
    COLOR_PMAX,
    COLOR_PMIN,
    WNM_TEMP_MIN_K,
)
from src.physics import thermal  # noqa: E402
from src.plotting.style import (  # noqa: E402
    annotate,
    apply_paper_style,
    guide_line,
    panel_label,
    save_figure,
)

NAME = "fig_bs19_curves_iuv1e3"
IUV_MAIN = 1.0e3
IUV_REF = 1.0

MAIN_COLOR = "#1a1a2e"
REF_COLOR = "0.60"
UNSTABLE_COLOR = "#8c6d1f"
UNSTABLE_FILL_ALPHA = 0.13


def load_curves(mat_path):
    """(n grid, {I_UV: T(n)}) straight off the table, plus the I_UV grid.

    Both requested I_UV values are exact grid rows (the grid is
    10^(3 - 0.2k), k = 0..30), so no interpolation is needed; this
    function nevertheless reports which row it used, and refuses rather
    than extrapolating if an asked-for I_UV falls outside the table.
    """
    d = scipy.io.loadmat(str(mat_path))
    iuv_grid = np.array(d["IUV_"]).squeeze().astype(float).ravel()
    n_grid = np.array(d["n_"]).squeeze().astype(float).ravel()
    t_2d_n = np.array(d["T_2d_n"]).astype(float)
    if np.any(np.diff(n_grid) <= 0):
        order = np.argsort(n_grid)
        n_grid = n_grid[order]
        t_2d_n = t_2d_n[:, order]
    return iuv_grid, n_grid, t_2d_n


def temperature_row(iuv_grid, t_2d_n, iuv_value, tol=1e-9):
    """T(n) at iuv_value: the exact grid row, or a log10-linear blend.

    Returns (T, provenance_string). Raises if iuv_value is outside the
    table -- extrapolating a thermochemical equilibrium curve past the
    grid it was solved on is not a thing this script is allowed to do.
    """
    lo, hi = iuv_grid.min(), iuv_grid.max()
    if not (lo <= iuv_value <= hi):
        raise ValueError(
            f"I_UV = {iuv_value:g} is outside the table's range "
            f"[{lo:g}, {hi:g}]; refusing to extrapolate.")

    j = int(np.argmin(np.abs(np.log10(iuv_grid) - np.log10(iuv_value))))
    if abs(np.log10(iuv_grid[j]) - np.log10(iuv_value)) < tol:
        return t_2d_n[j], f"exact grid row {j} (I_UV = {iuv_grid[j]:g})"

    order = np.argsort(iuv_grid)
    li = np.log10(iuv_grid[order])
    k = int(np.searchsorted(li, np.log10(iuv_value)))
    a, b = order[k - 1], order[k]
    w = ((np.log10(iuv_value) - np.log10(iuv_grid[a]))
         / (np.log10(iuv_grid[b]) - np.log10(iuv_grid[a])))
    t = (1.0 - w) * t_2d_n[a] + w * t_2d_n[b]
    return t, (f"log10(I_UV)-linear blend of rows {a} (I_UV = "
               f"{iuv_grid[a]:g}) and {b} (I_UV = {iuv_grid[b]:g}), w = {w:.3f}")


def main():
    apply_paper_style()
    mat_path = load_resolved_config()["bs19_mat_path"]

    iuv_grid, n_grid, t_2d_n = load_curves(mat_path)
    t_main, prov_main = temperature_row(iuv_grid, t_2d_n, IUV_MAIN)
    t_ref, prov_ref = temperature_row(iuv_grid, t_2d_n, IUV_REF)
    p_main = thermal.p_nT(n_grid, t_main)
    p_ref = thermal.p_nT(n_grid, t_ref)
    print(f"I_UV = {IUV_MAIN:g}: {prov_main}")
    print(f"I_UV = {IUV_REF:g}: {prov_ref}")

    # The project's own interpolators, at both I_UV values.
    pmm = thermal.build_pmin_pmax(mat_path)
    pdb = thermal.build_phase_density_bounds(mat_path)
    marks = {v: dict(p_min=float(pmm.p_min(v)), p_max=float(pmm.p_max(v)),
                     n_w=float(pdb.n_w_max(v)), n_c=float(pdb.n_c_min(v)))
             for v in (IUV_MAIN, IUV_REF)}

    # Which rows those values actually came from, so the edge fill can be
    # named on the figure instead of silently presented as a measurement.
    iuv_raw, nw_raw, nc_raw, _, _ = thermal.turning_point_densities(mat_path)
    last_n_row = float(iuv_raw[np.isfinite(nw_raw)].max())
    pmin_finite = np.array([np.isfinite(pmm.p_min(v)) for v in iuv_raw])
    # P_min/P_max come from the P tabulation; find its last S-curve row the
    # same way build_pmin_pmax does, by asking where the interpolator stops
    # changing with I_UV.
    last_p_row = _last_pmin_row(mat_path)

    main_has_scurve = _has_turning_points(n_grid, t_main)
    ref_has_scurve = _has_turning_points(n_grid, t_ref)
    print(f"I_UV = {IUV_MAIN:g}: S-curve present? {main_has_scurve}")
    print(f"I_UV = {IUV_REF:g}: S-curve present? {ref_has_scurve}")
    print(f"last tabulated S-curve row: n-grid I_UV = {last_n_row:g}, "
          f"P-grid I_UV = {last_p_row:g}")
    print(f"marks: {marks}")

    # ------------------------------------------------------------------
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 4.15), sharex=True)
    # rect is (left, bottom, WIDTH, HEIGHT): the strip for the legend and
    # the two-line caveat has to come out of the height as well.
    fig.get_layout_engine().set(rect=(0.0, 0.175, 1.0, 0.825))

    m = marks[IUV_MAIN]
    for ax in (ax_a, ax_b):
        ax.axvspan(m["n_w"], m["n_c"], color=UNSTABLE_COLOR,
                   alpha=UNSTABLE_FILL_ALPHA, linewidth=0, zorder=1)
        for xv in (m["n_w"], m["n_c"]):
            ax.axvline(xv, color=UNSTABLE_COLOR, linestyle=(0, (3, 2)),
                       linewidth=0.9, zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(n_grid.min(), n_grid.max())
        ax.set_xlabel(r"$n_\mathrm{H}$  [cm$^{-3}$]")

    # ---------------------------------------------------------------- (a)
    guide_line(ax_a, CNM_TEMP_MAX_K, label=f"{CNM_TEMP_MAX_K:g} K",
               label_x=0.02, label_loc="above")
    guide_line(ax_a, WNM_TEMP_MIN_K, label=f"{WNM_TEMP_MIN_K:g} K",
               label_x=0.02, label_loc="below")
    ax_a.plot(n_grid, t_ref, color=REF_COLOR, linewidth=1.0, zorder=3)
    ax_a.plot(n_grid, t_main, color=MAIN_COLOR, linewidth=2.0, zorder=5)
    _mark_reference_turning_points(ax_a, n_grid, t_ref, marks[IUV_REF], t_ref)
    ax_a.set_ylabel(r"$T$  [K]")
    ax_a.set_ylim(_log_span([t_main, t_ref], pad=0.10, head=0.30))
    panel_label(ax_a, "a", loc="lower left")
    annotate(ax_a, r"$I_\mathrm{UV}=10^{3}$", loc="upper right", size=7.5)

    # ---------------------------------------------------------------- (b)
    for key, col in (("p_min", COLOR_PMIN), ("p_max", COLOR_PMAX)):
        ax_b.axhline(m[key], color=col, linestyle=(0, (3, 2)), linewidth=0.9,
                     zorder=2)
    # At the edge-fill row the two are only a couple of percent apart, so
    # the lines coincide on a log axis and two separate labels would print
    # on top of each other. One label for the pair, with the gap quoted --
    # which is itself the informative bit: the two-phase range has very
    # nearly closed by the last I_UV that still has one.
    gap = m["p_max"] / m["p_min"] - 1.0
    ax_b.annotate(rf"$P_\mathrm{{min}},P_\mathrm{{max}}$ ({gap * 100:.1f}% apart)",
                  xy=(0.03, m["p_max"]), xycoords=("axes fraction", "data"),
                  xytext=(0, 3), textcoords="offset points", ha="left",
                  va="bottom", fontsize=6.0, color=COLOR_PMAX, zorder=9)
    ax_b.plot(n_grid, p_ref, color=REF_COLOR, linewidth=1.0, zorder=3)
    ax_b.plot(n_grid, p_main, color=MAIN_COLOR, linewidth=2.0, zorder=5)
    _mark_reference_turning_points(ax_b, n_grid, p_ref, marks[IUV_REF], t_ref,
                                   as_pressure=True)
    ax_b.set_ylabel(r"$P/k_\mathrm{B}=n_\mathrm{H}T$  [K cm$^{-3}$]")
    ax_b.set_ylim(_log_span([p_main, p_ref], pad=0.10, head=0.30))
    # lower RIGHT: both curves climb to the top right, so the bottom-right
    # corner is the only empty one in this panel.
    panel_label(ax_b, "b", loc="lower right")
    annotate(ax_b, r"$I_\mathrm{UV}=10^{3}$", loc="upper left", size=7.5,
             pad=0.055)

    # ------------------------------------------------------------- legend
    handles = [
        Line2D([], [], color=MAIN_COLOR, linewidth=2.0),
        Line2D([], [], color=REF_COLOR, linewidth=1.0),
        Line2D([], [], color=REF_COLOR, marker="o", markersize=3.5,
               linestyle="none"),
        Patch(facecolor=UNSTABLE_COLOR, alpha=UNSTABLE_FILL_ALPHA, linewidth=0),
    ]
    labels = [
        r"$I_\mathrm{UV}=10^{3}$",
        r"$I_\mathrm{UV}=1$",
        r"$I_\mathrm{UV}=1$ turning points",
        "carried-over unstable range",
    ]
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 0.105), ncols=4, columnspacing=1.5)

    fig.text(
        0.5, 0.058,
        r"At $I_\mathrm{UV}=10^{3}$ the equilibrium curve is MONOTONIC: no "
        r"d$P$/d$n=0$ turning points, hence no thermally unstable range.",
        ha="center", va="bottom", fontsize=6.3, color="0.25")
    fig.text(
        0.5, 0.008,
        f"The marked $n_{{\\mathrm{{W,max}}}}$, $n_{{\\mathrm{{C,min}}}}$ (carried over "
        f"from $I_\\mathrm{{UV}}={last_n_row:.0f}$) and $P_\\mathrm{{min}}$, "
        f"$P_\\mathrm{{max}}$ (from $I_\\mathrm{{UV}}={last_p_row:.0f}$) are the "
        f"interpolators' constant edge fill, not features of this curve.",
        ha="center", va="bottom", fontsize=6.3, color="0.25")

    save_figure(fig, NAME)


# ---------------------------------------------------------------------------
def _has_turning_points(n_grid, t_row):
    return thermal.locate_scurve_turning_points(n_grid, t_row) is not None


def _last_pmin_row(mat_path):
    """Highest tabulated I_UV whose T_2d_P row still has a dn/dP sign change.

    build_pmin_pmax() does not expose this, and it is a different row from
    the n-grid one, so it is recovered here by the same criterion that
    function uses rather than hardcoded.
    """
    d = scipy.io.loadmat(str(mat_path))
    iuv = np.array(d["IUV_"]).squeeze().astype(float).ravel()
    p_grid = np.array(d["P_"]).squeeze().astype(float).ravel()
    t_2d_p = np.array(d["T_2d_P"]).astype(float)
    if np.any(np.diff(p_grid) <= 0):
        order = np.argsort(p_grid)
        p_grid, t_2d_p = p_grid[order], t_2d_p[:, order]
    best = -np.inf
    for i in range(len(iuv)):
        n_row = np.clip(p_grid / np.clip(t_2d_p[i], 1e-30, None), 1e-30, None)
        s = np.sign(np.gradient(n_row, p_grid))
        s[s == 0] = 1
        if np.any(s[:-1] * s[1:] < 0):
            best = max(best, iuv[i])
    return float(best)


def _mark_reference_turning_points(ax, n_grid, y_curve, ref_marks, t_ref,
                                     as_pressure=False):
    """Small dots where the I_UV = 1 curve's REAL turning points sit.

    The contrast is the point of the figure: these exist, the I_UV = 1e3
    ones do not.
    """
    for key in ("n_w", "n_c"):
        x = ref_marks[key]
        y = np.interp(np.log10(x), np.log10(n_grid), y_curve)
        ax.plot([x], [y], marker="o", markersize=3.5, color=REF_COLOR,
                linestyle="none", zorder=6)


def _log_span(arrays, pad=0.08, head=0.0):
    """(lo, hi) spanning every positive finite value, padded in decades."""
    v = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    v = v[np.isfinite(v) & (v > 0)]
    return 10.0 ** (np.log10(v.min()) - pad), 10.0 ** (np.log10(v.max()) + pad + head)


if __name__ == "__main__":
    main()
