"""
Shared vocabulary for the alpha paper project: colors, thresholds, physical
constants, and plot style used consistently across all figures and analysis
scripts.

This module holds constants only — no computation or data logic. Physics
formulas that use these constants live in src/physics/.
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
# SETTLED (core-physics branch, alpha paper): CNM T < 300 K, UNM 300-6000 K,
# WNM T > 6000 K. This supersedes an earlier 600 K CNM bound that was used
# nowhere in the four reference compute scripts being ported
# (fig2_histograms/computing_data.py, fig3_slices/compute_data.py,
# fig4_vertical_profiles/compute_data.py,
# fig4b_velocity_dispertion_Mach_number/compute_data.py all use 300/6000
# K); it matched a companion paper's convention instead. That companion-
# paper discrepancy is unresolved and out of scope here — this constant now
# reflects only this project's reference implementations.
CNM_TEMP_MAX_K = 300
WNM_TEMP_MIN_K = 6000

# ---------------------------------------------------------------------------
# Spatial domain constants
# ---------------------------------------------------------------------------
# SETTLED (core-physics branch): the primary footprint is now a +-500 pc
# SQUARE in XY (|x|<=500, |y|<=500; Shelest+26), not a cylinder. R_MAX_PC is
# kept only for reproducing the OLD reference scripts' R<=500 pc cylinder in
# scripts/validation/regression_vs_old.py -- new code should use
# XY_HALF_RANGE_PC / footprint_mask() from src.physics.loading instead.
R_MAX_PC = 500
XY_HALF_RANGE_PC = 500.0

# ---------------------------------------------------------------------------
# Physical constants (ported verbatim from the four reference compute
# scripts, which all agree exactly on every value below)
# ---------------------------------------------------------------------------
K_B = 1.380649e-16  # erg / K
M_H = 1.6735575e-24  # g
MU = 1.4  # mean molecular weight per H (helium included), rho = MU * M_H * n_H
PC_CM = 3.085677581491367e18  # cm per pc
M_SUN_G = 1.989e33  # g per solar mass

IUV_FLOOR = 1e-30
P_FLOOR = 1e-30
N_FLOOR = 1e-12

# ---------------------------------------------------------------------------
# HIM / phase-substitution temperature
# ---------------------------------------------------------------------------
T_HIM_K = 1.0e6  # K, assigned temperature for HIM_A/HIM_B substituted cells

# ---------------------------------------------------------------------------
# External gravity (Guo+20 Milky-Way stellar disk + dark-matter halo)
# ---------------------------------------------------------------------------
SIGMA_STAR_MSUN_PC2 = 38.4  # Msun/pc^2
Z_H_PC = 413.0  # pc
RHO_DM_MSUN_PC3 = 0.0133  # Msun/pc^3
G_PC_MSUN_KMS = 4.30091e-3  # (pc/Msun) * (km/s)^2, gravitational constant
KM2S2_PER_PC_TO_CGS = 1.0e10 / PC_CM  # (km/s)^2 -> cm^2/s^2, divided by pc-in-cm


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
