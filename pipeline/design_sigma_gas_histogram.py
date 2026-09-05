"""
Design script for a histogram of the flattened Sigma_gas(x,y) map.

Loads cache/sigma_gas_edenhofer.npz and produces a single-panel histogram.
Presentation only — no data-filtering or masking logic here (see
compute_sigma_gas.py for that). Mean/median are read from the cache
(saved by compute_sigma_gas.py) rather than recomputed here, for
consistency with the map figure.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.conventions import apply_style

CACHE_PATH = "cache/sigma_gas_edenhofer.npz"
OUTPUT_PATH = "figures/sigma_gas_histogram.png"

N_BINS = 50


def main():
    data = np.load(CACHE_PATH)
    sigma_gas = data["sigma_gas"].flatten()
    mean = float(data["mean"])
    median = float(data["median"])

    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Sigma_gas spans two orders of magnitude and is right-skewed (mean > median),
    # so log-spaced bins on a log x-axis resolve the shape far better than linear.
    bins = np.logspace(np.log10(sigma_gas.min()), np.log10(sigma_gas.max()), N_BINS)
    ax.hist(sigma_gas, bins=bins, color="steelblue", edgecolor="none")
    ax.set_xscale("log")

    ax.axvline(mean, color="#c0392b", linestyle="--", label=f"Mean = {mean:.2f}")
    ax.axvline(median, color="#2980b9", linestyle=":", label=f"Median = {median:.2f}")

    ax.set_xlabel(r"$\Sigma_{\rm gas}$ (M$_\odot$/pc$^2$)")
    ax.set_ylabel("Number of pixels")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
