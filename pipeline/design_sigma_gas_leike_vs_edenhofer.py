"""
Design script comparing Sigma_gas distributions derived from two independent
3D dust maps: Edenhofer et al. (2023) and Leike, Glatzle & Ensslin (2020),
both on the same +-400 pc square footprint in x, y and z restricted to
+-270 pc (Leike2020's actual data extent -- see compute_sigma_gas_leike_vs_edenhofer.py
for why this is a partial-column Sigma_gas, not this project's usual
full-column definition).

Loads cache/sigma_gas_leike_vs_edenhofer.npz and produces a single-panel
figure with two smoothed histogram lines. Presentation only — no
data-filtering or masking logic here (see
compute_sigma_gas_leike_vs_edenhofer.py for that).
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.conventions import COLOR_ALPHA, COLOR_PTOT, apply_style

CACHE_PATH = "cache/sigma_gas_leike_vs_edenhofer.npz"
OUTPUT_PATH = "figures/sigma_gas_leike_vs_edenhofer.png"

N_BINS = 50
SMOOTHING_SIGMA_BINS = 1.5


def smoothed_line(values, bins):
    counts, edges = np.histogram(values, bins=bins)
    bin_centers = np.sqrt(edges[:-1] * edges[1:])  # geometric mean, for log-spaced bins
    smoothed_counts = gaussian_filter1d(counts.astype(float), sigma=SMOOTHING_SIGMA_BINS)
    return bin_centers, smoothed_counts


def main():
    data = np.load(CACHE_PATH)
    sigma_eden = data["sigma_gas_eden"].flatten()
    sigma_leike = data["sigma_gas_leike"].flatten()
    sigma_eden = sigma_eden[np.isfinite(sigma_eden)]
    sigma_leike = sigma_leike[np.isfinite(sigma_leike)]
    mean_eden = float(data["mean_eden"])
    median_eden = float(data["median_eden"])
    mean_leike = float(data["mean_leike"])
    median_leike = float(data["median_leike"])

    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))

    combined_min = min(sigma_eden.min(), sigma_leike.min())
    combined_max = max(sigma_eden.max(), sigma_leike.max())
    bins = np.logspace(np.log10(combined_min), np.log10(combined_max), N_BINS)

    x_eden, y_eden = smoothed_line(sigma_eden, bins)
    x_leike, y_leike = smoothed_line(sigma_leike, bins)

    ax.plot(x_eden, y_eden, color=COLOR_PTOT, linewidth=1.5,
            label=r"Edenhofer2023 ($\pm$400pc xy, $\pm$270pc z, 1652 cm$^{-3}$/E)")
    ax.plot(x_leike, y_leike, color=COLOR_ALPHA, linewidth=1.5,
            label=r"Leike2020 ($\pm$400pc xy, $\pm$270pc z, 880 cm$^{-3}$ x s$_x$)")
    ax.set_xscale("log")

    ax.axvline(mean_eden, color=COLOR_PTOT, linestyle="--",
               label=f"Edenhofer mean = {mean_eden:.2f}")
    ax.axvline(median_eden, color=COLOR_PTOT, linestyle=":",
               label=f"Edenhofer median = {median_eden:.2f}")
    ax.axvline(mean_leike, color=COLOR_ALPHA, linestyle="--",
               label=f"Leike mean = {mean_leike:.2f}")
    ax.axvline(median_leike, color=COLOR_ALPHA, linestyle=":",
               label=f"Leike median = {median_leike:.2f}")

    ax.set_xlabel(r"$\Sigma_{\rm gas}$, $|z|\leq270$pc only (M$_\odot$/pc$^2$)")
    ax.set_ylabel("Number of pixels")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
