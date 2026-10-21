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
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.plotting.style import apply_paper_style, provisional_label, save_figure  # noqa: E402

CACHE = Path("cache/maps/maps_slices.npz")
NAME = "fig_maps_shelest_check"
# Shelest+26 Fig. 3 colours: CNM blue, UNM yellow, WNM black
TCUT_CMAP = ListedColormap(["#1f5fbf", "#f2d22e", "black"])
TCUT_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)


def main():
    apply_paper_style()
    d = np.load(CACHE, allow_pickle=False)
    x, y, xz_x, z = d["x_pc"], d["y_pc"], d["x_xz_pc"], d["z_xz_pc"]
    ext_xy = (x[0] - 1, x[-1] + 1, y[0] - 1, y[-1] + 1)
    ext_xz = (xz_x[0] - 1, xz_x[-1] + 1, z[0] - 1, z[-1] + 1)

    fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.6), layout="none",
                             gridspec_kw=dict(height_ratios=(1.0, 0.8)))
    fig.subplots_adjust(left=0.08, right=0.93, bottom=0.11, top=0.93,
                        wspace=0.45, hspace=0.22)
    specs = [("n_H", "magma", r"$n$ [cm$^{-3}$]", LogNorm(1e-3, 1e1)),
             ("I_UV", "viridis", r"$I_{\rm UV}$ [$G_0$]", LogNorm(1e-1, 1e1)),
             ("phase_tcut", TCUT_CMAP, "phase (T cuts)", TCUT_NORM)]
    for row, (tag, ext, ylab) in enumerate((("xy", ext_xy, r"$y$ [pc]"),
                                            ("xz", ext_xz, r"$z$ [pc]"))):
        for col, (key, cmap, lab, norm) in enumerate(specs):
            ax = axes[row, col]
            im = ax.imshow(d[f"{tag}__{key}"], origin="lower", extent=ext, cmap=cmap,
                           norm=norm, interpolation="nearest", rasterized=True)
            ax.set_aspect("equal")
            ax.plot(0, 0, marker="*", ms=7, mfc="white", mec="black", mew=0.6)
            ax.set_xlabel(r"$x$ [pc]")
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
            ax.text(0.03, 0.97, "z = 0" if tag == "xy" else "y = 0", transform=ax.transAxes,
                    ha="left", va="top", fontsize=7,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))
    fig.text(0.5, 0.005, "Layout of Shelest+26 Fig. 3, true Galactic orientation "
             r"(+$x$ toward the Galactic centre, +$y$ toward $l = 90^\circ$); Sun at the star.",
             ha="center", va="bottom", fontsize=6.5, color="0.3")
    provisional_label(fig, loc="upper right")
    save_figure(fig, NAME)


if __name__ == "__main__":
    main()
