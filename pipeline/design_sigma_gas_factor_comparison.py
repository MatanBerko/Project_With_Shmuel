"""
Design script comparing Sigma_gas distributions from the old vs new
dust-extinction-to-n_H conversion factor.

Loads cache/sigma_gas_factor_comparison.npz and produces a single-panel
figure with two smoothed histogram lines. Presentation only — no
data-filtering or masking logic here (see
compute_sigma_gas_factor_comparison.py for that).
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.conventions import COLOR_ALPHA, COLOR_PTOT, apply_style

CACHE_PATH = "cache/sigma_gas_factor_comparison.npz"
OUTPUT_PATH = "figures/sigma_gas_factor_comparison.png"

N_BINS = 50
SMOOTHING_SIGMA_BINS = 1.5


def smoothed_line(values, bins):
    counts, edges = np.histogram(values, bins=bins)
    bin_centers = np.sqrt(edges[:-1] * edges[1:])  # geometric mean, for log-spaced bins
    smoothed_counts = gaussian_filter1d(counts.astype(float), sigma=SMOOTHING_SIGMA_BINS)
    return bin_centers, smoothed_counts


def main():
    data = np.load(CACHE_PATH)
    sigma_old = data["sigma_gas_old"].flatten()
    sigma_new = data["sigma_gas_new"].flatten()
    sigma_old = sigma_old[np.isfinite(sigma_old)]
    sigma_new = sigma_new[np.isfinite(sigma_new)]

    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))

    combined_min = min(sigma_old.min(), sigma_new.min())
    combined_max = max(sigma_old.max(), sigma_new.max())
    bins = np.logspace(np.log10(combined_min), np.log10(combined_max), N_BINS)

    x_old, y_old = smoothed_line(sigma_old, bins)
    x_new, y_new = smoothed_line(sigma_new, bins)

    ax.plot(x_old, y_old, color=COLOR_PTOT, linewidth=1.5,
            label="Old (1652 cm$^{-3}$/E, Zucker+21/O'Neill+24)")
    ax.plot(x_new, y_new, color=COLOR_ALPHA, linewidth=1.5,
            label="New (2700 cm$^{-3}$/E, McCallum+26)")
    ax.set_xscale("log")

    ax.set_xlabel(r"$\Sigma_{\rm gas}$ (M$_\odot$/pc$^2$)")
    ax.set_ylabel("Number of pixels")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
