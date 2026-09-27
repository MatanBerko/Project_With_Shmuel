"""
Plot for the Sigma_gas offset diagnostic. Loads ONLY
cache/diagnostics/sigma_offset_results.npz (no dustmaps query, no other
pipeline import) and produces a compact 2-panel figure:

  (a) log10 Sigma_gas PDFs for the four (z-range x footprint) configs,
      with each config's median marked, plus thin reference lines at the
      OLD figure's reported median (log10 3.3 = 0.52) and the current
      repo figure's median (log10 4.77).
  (b) an XY map of 1 - Sigma_400/Sigma_750, i.e. the fractional column
      mass contributed by the 400 < |z| <= 750 pc shell alone.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy.ndimage import uniform_filter1d

RESULTS_NPZ_PATH = Path("cache/diagnostics/sigma_offset_results.npz")
OUTPUT_PNG_PATH = Path("cache/diagnostics/sigma_offset_diagnostic.png")
OUTPUT_PDF_PATH = Path("cache/diagnostics/sigma_offset_diagnostic.pdf")

COVERAGE_THRESHOLD = 0.9
LOG_MEDIAN_OLD_FIG = np.log10(3.3)
LOG_MEDIAN_NEW_FIG = np.log10(4.77)
N_BINS = 40
SMOOTH_WINDOW_BINS = 3

CONFIG_COLORS = {
    "OLD (400, cylinder)": "#c0392b",
    "(400, square)": "#e67e22",
    "(750, cylinder)": "#2980b9",
    "(750, square) = NEW": "#27ae60",
}


def smoothed_pdf(log_values, bins):
    counts, edges = np.histogram(log_values, bins=bins, density=True)
    smoothed = uniform_filter1d(counts, size=SMOOTH_WINDOW_BINS, mode="nearest")
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, smoothed


def main():
    d = np.load(RESULTS_NPZ_PATH)
    sigma_400 = d["sigma_400"]
    sigma_750 = d["sigma_750"]
    coverage_400 = d["coverage_400"]
    coverage_750 = d["coverage_750"]
    cylinder_mask = d["cylinder_mask"]
    x_pc = d["x_pc"]
    y_pc = d["y_pc"]

    cov_ok_400 = coverage_400 >= COVERAGE_THRESHOLD
    cov_ok_750 = coverage_750 >= COVERAGE_THRESHOLD

    configs = {
        "OLD (400, cylinder)": (sigma_400[cylinder_mask & cov_ok_400],
                                 float(d["old_400_cylinder__median"])),
        "(400, square)": (sigma_400[cov_ok_400],
                           float(d["400_square__median"])),
        "(750, cylinder)": (sigma_750[cylinder_mask & cov_ok_750],
                             float(d["750_cylinder__median"])),
        "(750, square) = NEW": (sigma_750[cov_ok_750],
                                 float(d["750_square__expected_new__median"])),
    }

    rcParams["font.family"] = "serif"

    fig, (ax_pdf, ax_map) = plt.subplots(1, 2, figsize=(10, 4.2))

    all_log_vals = np.concatenate(
        [np.log10(vals[vals > 0]) for vals, _ in configs.values()]
    )
    bins = np.linspace(all_log_vals.min(), all_log_vals.max(), N_BINS)

    for label, (vals, median) in configs.items():
        log_vals = np.log10(vals[vals > 0])
        centers, pdf = smoothed_pdf(log_vals, bins)
        color = CONFIG_COLORS[label]
        ax_pdf.plot(centers, pdf, color=color, linewidth=1.5, label=f"{label}: med={median:.2f}")
        ax_pdf.axvline(np.log10(median), color=color, linestyle=":", linewidth=1.0)

    ax_pdf.axvline(LOG_MEDIAN_OLD_FIG, color="0.5", linewidth=0.8, zorder=0)
    ax_pdf.text(LOG_MEDIAN_OLD_FIG, ax_pdf.get_ylim()[1] * 0.05, "old fig.",
                color="0.5", fontsize=8, ha="right", va="bottom", rotation=90)
    ax_pdf.axvline(LOG_MEDIAN_NEW_FIG, color="0.5", linewidth=0.8, zorder=0)
    ax_pdf.text(LOG_MEDIAN_NEW_FIG, ax_pdf.get_ylim()[1] * 0.05, "new fig.",
                color="0.5", fontsize=8, ha="left", va="bottom", rotation=90)

    ax_pdf.set_xlabel(r"$\log_{10}\,\Sigma_{\rm gas}$ (M$_\odot$/pc$^2$)")
    ax_pdf.set_ylabel("PDF")
    ax_pdf.legend(frameon=False, fontsize=7, loc="upper right")

    frac_extra = 1.0 - sigma_400 / sigma_750
    im = ax_map.imshow(
        frac_extra, origin="lower",
        extent=[x_pc.min(), x_pc.max(), y_pc.min(), y_pc.max()],
        cmap="magma", vmin=0.0, vmax=0.5, aspect="equal",
    )
    theta = np.linspace(0, 2 * np.pi, 200)
    ax_map.plot(500 * np.cos(theta), 500 * np.sin(theta), "w--", linewidth=1.0)
    ax_map.text(0.03, 0.96, "400 < |z| < 750 pc", transform=ax_map.transAxes,
                color="white", fontsize=8, ha="left", va="top")
    ax_map.set_xlabel("x (pc)")
    ax_map.set_ylabel("y (pc)")

    cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.set_label(r"$1 - \Sigma_{400}/\Sigma_{750}$")

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG_PATH, dpi=300)
    fig.savefig(OUTPUT_PDF_PATH)
    print(f"Saved {OUTPUT_PNG_PATH}")
    print(f"Saved {OUTPUT_PDF_PATH}")


if __name__ == "__main__":
    main()
