"""
Shared vocabulary for the alpha paper project: colors, thresholds, and plot
style used consistently across all figures and analysis scripts.

This module holds constants only — no computation or data logic.
"""

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# ISM phase colors (RGB tuples, 0-1 scale, except HIM which is a hex string)
# ---------------------------------------------------------------------------
CNM = (37 / 255, 85 / 255, 115 / 255)   # Cold Neutral Medium
UNM = (228 / 255, 210 / 255, 132 / 255)  # Unstable Neutral Medium
WNM = (194 / 255, 83 / 255, 81 / 255)   # Warm Neutral Medium
HIM = "#7b5ea7"                          # Hot Ionized Medium

# ---------------------------------------------------------------------------
# Model / quantity colors (hex strings)
# ---------------------------------------------------------------------------
COLOR_PTH = "#1a1a2e"     # thermal pressure
COLOR_PTOT = "#c0392b"    # total pressure
COLOR_PMIN = "#2e8b7a"    # minimum pressure bound
COLOR_PMAX = "#c8553d"    # maximum pressure bound
COLOR_ALPHA = "#2980b9"   # alpha = P_tot / P_th
COLOR_RPP = "#d35400"     # R_pp (or related ratio quantity)

# ---------------------------------------------------------------------------
# HIM classification threshold
# ---------------------------------------------------------------------------
# HIM is defined as Pth < HIM_THRESHOLD_FACTOR * Pmin. This strict factor of
# 0.5 (rather than the standard Pth < Pmin) was validated to produce zero
# CNM/UNM contamination in the HIM classification; the standard Pth < Pmin
# threshold was found to misclassify ~50% of CNM cells as HIM.
HIM_THRESHOLD_FACTOR = 0.5

# ---------------------------------------------------------------------------
# Phase temperature cuts
# ---------------------------------------------------------------------------
# 600 K is this project's convention for the CNM upper temperature bound.
# A companion paper instead uses 300 K. This discrepancy is a known open
# inconsistency to resolve between the two works, not something to silently
# change here.
CNM_TEMP_MAX_K = 600
WNM_TEMP_MIN_K = 6000

# ---------------------------------------------------------------------------
# Spatial domain constant
# ---------------------------------------------------------------------------
# All statistics are restricted to R <= 500 pc. The underlying domain data
# extends further, but boundary noise beyond this radius corrupts results.
R_MAX_PC = 500


def apply_style() -> None:
    """Apply this project's publication-quality matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "figure.figsize": (6, 4.5),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
        }
    )
