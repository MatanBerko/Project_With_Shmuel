"""
fig_maps_shelest_check -- the Fig. 2 planes in the layout of Shelest+26
Fig. 3, so the corrected XY orientation can be compared by eye against
the published paper.

Reads ONLY cache/maps/maps_slices.npz (scripts/compute/compute_maps.py).

2 rows x 3 columns: n (left), I_UV (middle), T as a DISCRETE 3-colour
phase map (right; CNM blue T <= 300 K, UNM yellow 300-6000 K, WNM black
T > 6000 K, HIM not separated -- Shelest+26's colouring). Top row z = 0
(x horizontal, y vertical); bottom row y = 0 (x horizontal, z vertical).
Sun at the origin. A comparison figure only: plain but clear.

  --mapping neg_swap  (default) true Galactic orientation
                      -> figures/fig_maps_shelest_check.{png,pdf}
  --mapping identity  the cube read AS STORED (no reorientation): cube
                      plane z = 0 with x_c horizontal and y_c vertical, and
                      cube plane y_c = 0 with x_c horizontal and z vertical
                      -> figures/fig_maps_shelest_check_asis.{png,pdf}
  --pair              just the z = 0 density panel, true (left) vs as
                      stored (right), same colour range
                      -> figures/fig_maps_shelest_check_pair.png
The as-stored planes exist only to test whether a published figure was
made from the cube without reorientation.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.conventions import CUBE_XY_MAPPING_IDENTITY, CUBE_XY_MAPPING_NEG_SWAP  # noqa: E402
from src.plotting.style import (  # noqa: E402
    FIGURES_DIR,
    apply_paper_style,
    provisional_label,
    save_figure,
)

CACHE = Path("cache/maps/maps_slices.npz")
NAME = "fig_maps_shelest_check"
# Shelest+26 Fig. 3 colours: CNM blue, UNM yellow, WNM black
TCUT_CMAP = ListedColormap(["#1f5fbf", "#f2d22e", "black"])
TCUT_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)
N_NORM = (1e-3, 1e1)

# everything that differs between the two readings of the cube
VIEWS = {
    CUBE_XY_MAPPING_NEG_SWAP: dict(
        name=NAME, prefix="", x="x_pc", y="y_pc", xz_x="x_xz_pc", z="z_xz_pc",
        xlab=r"$x$ [pc]", ylab=r"$y$ [pc]", tags=("z = 0", "y = 0"),
        note="Layout of Shelest+26 Fig. 3, true Galactic orientation "
             r"(+$x$ toward the Galactic centre, +$y$ toward $l = 90^\circ$); Sun at the star."),
    CUBE_XY_MAPPING_IDENTITY: dict(
        name=NAME + "_asis", prefix="asis_", x="asis_x_c_pc", y="asis_y_c_pc",
        xz_x="asis_x_c_pc", z="asis_z_xz_pc",
        xlab=r"$x_{\rm cube}$ [pc]", ylab=r"$y_{\rm cube}$ [pc]",
        tags=(r"$z = 0$", r"$y_{\rm cube} = 0$"),
        note="cube read as stored (no reorientation)"),
}


def _extents(d, v):
    x, y, xz_x, z = d[v["x"]], d[v["y"]], d[v["xz_x"]], d[v["z"]]
    h = 0.5 * float(x[1] - x[0])
    return ((x[0] - h, x[-1] + h, y[0] - h, y[-1] + h),
            (xz_x[0] - h, xz_x[-1] + h, z[0] - h, z[-1] + h))


def check_figure(d, mapping):
    v = VIEWS[mapping]
    ext_xy, ext_xz = _extents(d, v)

    fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.6), layout="none",
                             gridspec_kw=dict(height_ratios=(1.0, 0.8)))
    fig.subplots_adjust(left=0.08, right=0.93, bottom=0.11, top=0.93,
                        wspace=0.45, hspace=0.22)
    specs = [("n_H", "magma", r"$n$ [cm$^{-3}$]", LogNorm(*N_NORM)),
             ("I_UV", "viridis", r"$I_{\rm UV}$ [$G_0$]", LogNorm(1e-1, 1e1)),
             ("phase_tcut", TCUT_CMAP, "phase (T cuts)", TCUT_NORM)]
    for row, (tag, ext, ylab, label) in enumerate(
            (("xy", ext_xy, v["ylab"], v["tags"][0]),
             ("xz", ext_xz, r"$z$ [pc]", v["tags"][1]))):
        for col, (key, cmap, lab, norm) in enumerate(specs):
            ax = axes[row, col]
            im = ax.imshow(d[f"{v['prefix']}{tag}__{key}"], origin="lower", extent=ext,
                           cmap=cmap, norm=norm, interpolation="nearest", rasterized=True)
            ax.set_aspect("equal")
            ax.plot(0, 0, marker="*", ms=7, mfc="white", mec="black", mew=0.6)
            ax.set_xlabel(v["xlab"])
            if col == 0:      # same axis across a row: label it once
                ax.set_ylabel(ylab)
            ax.set_xticks((-400, 0, 400))
            ax.set_yticks((-400, -200, 0, 200, 400))
            # a colourbar exactly as tall as its (equal-aspect) panel
            cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
            cb = fig.colorbar(im, cax=cax)
            if key == "phase_tcut":
                cb.set_ticks([0, 1, 2])
                cb.set_ticklabels(["CNM", "UNM", "WNM"])
            else:
                # on top of the bar, so it cannot run into the next column
                cb.ax.set_title(lab, fontsize=7, pad=4)
            ax.text(0.03, 0.97, label, transform=ax.transAxes,
                    ha="left", va="top", fontsize=7,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))
    fig.text(0.5, 0.005, v["note"], ha="center", va="bottom", fontsize=6.5, color="0.3")
    provisional_label(fig, loc="upper right")
    save_figure(fig, v["name"])


def pair_figure(d):
    """z = 0 density, true orientation (left) vs as stored (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0), layout="none")
    fig.subplots_adjust(left=0.09, right=0.88, bottom=0.17, top=0.90, wspace=0.30)
    norm = LogNorm(*N_NORM)
    for ax, mapping, title in zip(axes, (CUBE_XY_MAPPING_NEG_SWAP, CUBE_XY_MAPPING_IDENTITY),
                                  ("true orientation", "as stored")):
        v = VIEWS[mapping]
        ext_xy, _ = _extents(d, v)
        im = ax.imshow(d[f"{v['prefix']}xy__n_H"], origin="lower", extent=ext_xy,
                       cmap="magma", norm=norm, interpolation="nearest", rasterized=True)
        ax.set_aspect("equal")
        ax.plot(0, 0, marker="*", ms=7, mfc="white", mec="black", mew=0.6)
        ax.set_xlabel(v["xlab"])
        ax.set_ylabel(v["ylab"])
        ax.set_xticks((-400, 0, 400))
        ax.set_yticks((-400, 0, 400))
        ax.text(0.03, 0.97, f"{title}, z = 0", transform=ax.transAxes, ha="left", va="top",
                fontsize=7, bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))
    cax = fig.add_axes((0.90, 0.17, 0.018, 0.73))
    fig.colorbar(im, cax=cax).ax.set_title(r"$n$ [cm$^{-3}$]", fontsize=7, pad=4)
    provisional_label(fig, loc="upper right")
    path = Path(FIGURES_DIR) / f"{NAME}_pair.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--mapping", choices=tuple(VIEWS), default=CUBE_XY_MAPPING_NEG_SWAP)
    ap.add_argument("--pair", action="store_true")
    args = ap.parse_args()
    apply_paper_style()
    d = np.load(CACHE, allow_pickle=False)
    if args.pair:
        pair_figure(d)
    else:
        check_figure(d, args.mapping)


if __name__ == "__main__":
    main()
