"""
Design script for the gas surface density Sigma_gas(x, y) map.

Loads cache/sigma_gas_edenhofer.npz and produces a single 2D map plot.
Presentation only — no data-filtering or masking logic here (see
compute_sigma_gas.py for that).
"""

import matplotlib.pyplot as plt
import numpy as np

from src.conventions import apply_style

CACHE_PATH = "cache/sigma_gas_edenhofer.npz"
OUTPUT_PATH = "figures/sigma_gas_edenhofer.png"


def main():
    data = np.load(CACHE_PATH)
    sigma_gas = data["sigma_gas"]
    x_pc = data["x_pc"]
    y_pc = data["y_pc"]
    mean = float(data["mean"])
    median = float(data["median"])

    apply_style()
    fig, ax = plt.subplots(figsize=(6, 5.5))

    extent = [x_pc.min(), x_pc.max(), y_pc.min(), y_pc.max()]
    im = ax.imshow(
        sigma_gas,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="viridis",
    )
    ax.set_xlabel("x (pc)")
    ax.set_ylabel("y (pc)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\Sigma_{\rm gas}$ (M$_\odot$/pc$^2$)")

    ax.text(
        0.03, 0.97,
        f"mean = {mean:.2f}\nmedian = {median:.2f}",
        transform=ax.transAxes,
        ha="left", va="top",
        color="white",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
