"""
fig_maps -- Fig. 2 of the paper: maps of the local ISM in TRUE Galactic
orientation (+x toward the Galactic centre, +y toward l = 90, +z toward
the NGP; Sun at the origin).

Reads ONLY cache/maps/maps_slices.npz (scripts/compute/compute_maps.py),
whose planes are already re-oriented through
src.physics.loading.to_galactic_xy.

2 rows x 4 columns at full AAS two-column width:
  columns  n_H [cm^-3] | I_UV (dimensionless) | P_th/k [K cm^-3] | phase (dPdn, HIM separate)
  row 1    the z = 0 plane (x-y)
  row 2    the y = 0 plane (x-z)
Equal aspect everywhere, so row 2 is 0.8x the height of row 1. The layout
is placed by hand in inches (no layout engine): equal-aspect image panels
leave gaps a layout engine cannot close, and placing them explicitly is
what makes the colourbars exactly as wide as their columns.

Colour ranges: n_H and I_UV use the 1st and 99th percentiles over BOTH
planes, rounded outward to half-decades (the same percentiles
results/maps_summary.txt prints, recomputed here from the same cached
arrays), with extend="both" since values exist beyond them. P_th/k is
fixed at 10^2-10^4 (extend="min"): the 1-99 range reaches down to 10^1,
which spends a decade of the colour scale on the rare lowest-pressure
cells.

Phase: HIM is drawn as a light tint of the project HIM colour so the
neutral phases read first; it is a flag (n_H T < 0.5 P_min), not a
measured hot phase. There is no Sun marker: the axes are heliocentric and
the caption (results/fig_maps_caption.txt) says so.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, to_rgb  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.conventions import CNM, HIM, UNM, WNM  # noqa: E402
from src.plotting.style import (  # noqa: E402
    annotate,
    apply_paper_style,
    panel_label,
    provisional_label,
    save_figure,
)

CACHE = Path("cache/maps/maps_slices.npz")
NAME = "fig_maps"
TICKS = (-400, -200, 0, 200, 400)

# column specs: (cache key, colormap, colourbar label)
COLUMNS = (
    ("n_H", "magma", r"$n_{\rm H}$ [cm$^{-3}$]"),
    # dimensionless: F_UV / F_UV,sun, F_UV,sun = 2.7e-3 erg cm^-2 s^-1 (Shelest+26)
    ("I_UV", "viridis", r"$I_{\rm UV}$"),
    ("P_th", "cividis", r"$P_{\rm th}/k$ [K cm$^{-3}$]"),
    ("phase_dpdn", None, "phase"),
)
# HIM blended with white (35% colour / 65% white) so the neutral phases read first
HIM_TINT_FRACTION = 0.35
HIM_TINT = tuple(HIM_TINT_FRACTION * c + (1.0 - HIM_TINT_FRACTION)
                 for c in to_rgb(HIM))
PHASE_CMAP = ListedColormap([CNM, UNM, WNM, HIM_TINT])
PHASE_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)
PHASE_LABELS = ("CNM", "UNM", "WNM", "HIM (flag)")
P_TH_RANGE = (1e2, 1e4)
CB_EXTEND = {"n_H": "both", "I_UV": "both", "P_th": "min", "phase_dpdn": "neither"}

# geometry [inches]
FIG_W = 7.1
LEFT, RIGHT = 0.46, 0.06       # room for y tick labels + y label / right edge
COL_GAP = 0.06                 # no tick labels between columns
ROW_GAP = 0.20                 # x tick labels of row 1
BOTTOM = 0.40                  # x tick labels + x label of row 2
CB_GAP = 0.05                  # colourbar to row 1
CB_H = 0.07                    # colourbar thickness
CB_TOP = 0.40                  # colourbar ticks + label above the bar


def half_decade_range(*arrays):
    """1st-99th percentile over all arrays, rounded outward to half-decades."""
    v = np.concatenate([np.asarray(a, float).ravel() for a in arrays])
    v = v[np.isfinite(v) & (v > 0)]
    lo, hi = np.percentile(v, (1, 99))
    return (10 ** (np.floor(2 * np.log10(lo)) / 2), 10 ** (np.ceil(2 * np.log10(hi)) / 2),
            lo, hi)


def decade_ticks(cax, vmin, vmax):
    """Labelled ticks at whole decades, unlabelled minor ticks at half-decades.

    The two end labels are aligned inward (left / right) so they stay
    inside their own column instead of colliding with the neighbouring
    colourbar's end label across the narrow column gap.
    """
    lo, hi = np.log10(vmin), np.log10(vmax)
    ks = range(int(np.ceil(lo)), int(np.floor(hi)) + 1)
    cax.set_xticks([10.0 ** k for k in ks], labels=[rf"$10^{{{k}}}$" for k in ks])
    minor = [10.0 ** (k + 0.5) for k in range(int(np.floor(lo)), int(np.ceil(hi)))
             if lo <= k + 0.5 <= hi]
    cax.xaxis.set_minor_locator(mticker.FixedLocator(minor))
    cax.xaxis.set_minor_formatter(mticker.NullFormatter())
    labels = cax.get_xticklabels()
    if np.isclose(np.log10(cax.get_xticks()[0]), lo):
        labels[0].set_horizontalalignment("left")
    if np.isclose(np.log10(cax.get_xticks()[-1]), hi):
        labels[-1].set_horizontalalignment("right")


def main():
    apply_paper_style()
    d = np.load(CACHE, allow_pickle=False)
    x, y = d["x_pc"], d["y_pc"]
    xz_x, z = d["x_xz_pc"], d["z_xz_pc"]
    dx = float(x[1] - x[0])
    ext = {"xy": (x[0] - dx / 2, x[-1] + dx / 2, y[0] - dx / 2, y[-1] + dx / 2),
           "xz": (xz_x[0] - dx / 2, xz_x[-1] + dx / 2, z[0] - dx / 2, z[-1] + dx / 2)}
    plane_coord = {"xy": rf"$z = {float(d['z_xy_pc'][0]):g}$ pc",
                   "xz": rf"$y = {0.0 - float(d['cube_x_c_of_xz_plane'][0]):g}$ pc"}

    norms, ranges = {}, {}
    for key, _, _ in COLUMNS[:3]:
        vmin, vmax, p1, p99 = half_decade_range(d[f"xy__{key}"], d[f"xz__{key}"])
        if key == "P_th":
            vmin, vmax = P_TH_RANGE
        norms[key] = LogNorm(vmin, vmax)
        ranges[key] = (vmin, vmax, p1, p99)
        print(f"  {key:5s} p1 {p1:.4g}  p99 {p99:.4g}  ->  colour range {vmin:.4g} .. {vmax:.4g}")

    # ---- geometry --------------------------------------------------------
    pw = (FIG_W - LEFT - RIGHT - 3 * COL_GAP) / 4          # panel width
    h1 = pw * (ext["xy"][3] - ext["xy"][2]) / (ext["xy"][1] - ext["xy"][0])
    h2 = pw * (ext["xz"][3] - ext["xz"][2]) / (ext["xz"][1] - ext["xz"][0])
    fig_h = BOTTOM + h2 + ROW_GAP + h1 + CB_GAP + CB_H + CB_TOP
    fig = plt.figure(figsize=(FIG_W, fig_h))
    fig.set_layout_engine("none")   # hand-placed axes; the paper style enables constrained

    def rect(x0, y0, w, h):
        return (x0 / FIG_W, y0 / fig_h, w / FIG_W, h / fig_h)

    y_row2 = BOTTOM
    y_row1 = BOTTOM + h2 + ROW_GAP
    y_cb = y_row1 + h1 + CB_GAP
    axes = {}
    for col, (key, cmap, cblabel) in enumerate(COLUMNS):
        x0 = LEFT + col * (pw + COL_GAP)
        for row, (tag, y0, h) in enumerate((("xy", y_row1, h1), ("xz", y_row2, h2))):
            ax = fig.add_axes(rect(x0, y0, pw, h))
            axes[(row, col)] = ax
            if key == "phase_dpdn":
                im = ax.imshow(d[f"{tag}__{key}"], origin="lower", extent=ext[tag],
                               cmap=PHASE_CMAP, norm=PHASE_NORM, interpolation="nearest",
                               rasterized=True)
            else:
                im = ax.imshow(d[f"{tag}__{key}"], origin="lower", extent=ext[tag],
                               cmap=cmap, norm=norms[key], interpolation="nearest",
                               rasterized=True)
            ax.set_aspect("equal")
            ax.set_xlim(ext[tag][:2])
            ax.set_ylim(ext[tag][2:])
            for side in ("top", "right"):
                ax.spines[side].set_visible(True)
            ax.set_xticks(TICKS)
            ax.set_yticks(TICKS if tag == "xy" else (-400, -200, 0, 200, 400))
            ax.tick_params(direction="in", length=2.5, width=0.5, top=True, right=True,
                           color="white" if key != "phase_dpdn" else "0.2",
                           labelsize=7)
            ax.tick_params(labelleft=(col == 0), labelbottom=True)
            if col == 0:
                ax.set_ylabel(r"$y$ [pc]" if tag == "xy" else r"$z$ [pc]", labelpad=1)
            if row == 1:
                ax.set_xlabel(r"$x$ [pc]", labelpad=1)
            # (a)-(d) across row 1, (e)-(h) across row 2
            panel_label(ax, "abcdefgh"[4 * row + col], loc="upper left", size=7.5, pad=0.03)
            annotate(ax, plane_coord[tag], loc="upper right", size=6.0, pad=0.03)
        # ---- one colourbar per column, above row 1 ------------------------
        cax = fig.add_axes(rect(x0, y_cb, pw, CB_H))
        # the extend triangles are drawn INSIDE cax, so the bar including
        # them stays exactly as wide as its column
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend=CB_EXTEND[key])
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
        cax.tick_params(which="both", direction="out", labelsize=6.5, pad=1)
        cax.tick_params(which="major", length=2.5, width=0.5)
        cax.tick_params(which="minor", length=1.5, width=0.4)
        cb.outline.set_linewidth(0.5)
        if key == "phase_dpdn":
            cb.set_ticks(range(4))
            # slightly smaller than the decade labels: four categories,
            # one of them two words, in a 1.6 in bar
            cb.set_ticklabels(PHASE_LABELS, fontsize=6.0)
            cax.tick_params(which="major", length=0)
            cax.xaxis.set_minor_locator(mticker.NullLocator())
            cb.set_label("phase (dP/dn)", fontsize=7.5, labelpad=2)
        else:
            decade_ticks(cax, norms[key].vmin, norms[key].vmax)
            cb.set_label(cblabel, fontsize=7.5, labelpad=2)

    # ---- "-> GC" on panel (a) only ---------------------------------------
    ax_a = axes[(0, 0)]
    outline = [pe.withStroke(linewidth=1.6, foreground="black")]
    ax_a.annotate("", xy=(0.97, 0.07), xytext=(0.77, 0.07), xycoords="axes fraction",
                  arrowprops=dict(arrowstyle="-|>", color="white", lw=0.9,
                                  mutation_scale=7, path_effects=outline))
    ax_a.text(0.75, 0.07, "GC", transform=ax_a.transAxes, ha="right", va="center",
              fontsize=6.0, color="white", path_effects=outline)

    provisional_label(fig, loc="upper right")
    save_figure(fig, NAME)
    return ranges


if __name__ == "__main__":
    main()
