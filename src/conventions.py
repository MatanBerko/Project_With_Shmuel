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
# Phase classification scheme (Shelest+26 switch)
# ---------------------------------------------------------------------------
# Shelest et al. 2026 (arXiv:2607.15352) classify phases by DENSITY against
# the two turning points of the BS19 thermal-equilibrium S-curve, not by
# temperature:
#   warm     n <  n_W,max(I_UV)
#   unstable n_W,max(I_UV) <= n <= n_C,min(I_UV)
#   cold     n >  n_C,min(I_UV)
# where n_W,max and n_C,min are the densities at the two dP/dn = 0 turning
# points of the equilibrium curve P(n) at that I_UV (see
# src.physics.thermal.build_phase_density_bounds). This is now the DEFAULT.
# "temperature" (CNM_TEMP_MAX_K / WNM_TEMP_MIN_K above, the behavior of all
# four reference scripts) is kept available behind the switch.
#
# The HIM flag (P_th < HIM_THRESHOLD_FACTOR * P_min) is SEPARATE from and
# unchanged by this switch, and takes precedence under both schemes.
PHASE_SCHEME_DPDN = "dPdn"
PHASE_SCHEME_TEMPERATURE = "temperature"
PHASE_SCHEME_DEFAULT = PHASE_SCHEME_DPDN
PHASE_SCHEMES = (PHASE_SCHEME_DPDN, PHASE_SCHEME_TEMPERATURE)

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

# STATS_BOX (Shelest+26, confirmed by the lead author): the ANALYSIS VOLUME
# is a 1 kpc x 1 kpc x 800 pc box centred on the Sun --
#     |x| <= 500 pc, |y| <= 500 pc, |z| <= 400 pc.
# Every statistic, profile and PDF is computed over this box ONLY.
#
# Two things deliberately do NOT use the box, and must not be "tidied" into
# it:
#  1. The P_tot hydrostatic integral still uses the FULL z column the cube
#     provides (+-750 pc for the f98 cube), with P = 0 fixed at the cube's
#     top and bottom edge. Clipping the integration to |z| <= 400 would put
#     the P = 0 boundary inside the gas layer and systematically understate
#     P_tot everywhere in the box.
#  2. The footprint average that sources "footprint_mean" self-gravity is
#     taken over |x|, |y| <= 500 at EVERY z the cube provides, for the same
#     reason: g_gas(z) inside the box depends on the gas column outside it.
# Sigma_gas likewise remains a full-column integral over |x|, |y| <= 500.
STATS_BOX_XY_HALF_RANGE_PC = 500.0
STATS_BOX_Z_HALF_RANGE_PC = 400.0

# Single-height statistics (Shelest+26): 60 pc-thick slabs, |z - z_c| <= 30,
# centred on z_c = 0, 150, 300 pc. This supersedes the earlier 50 pc-thick
# (+-25 pc) slab convention; neither is what the four reference scripts did
# (they all used a single nearest 2 pc z-plane via nearest_idx()).
SLAB_CENTERS_PC = (0.0, 150.0, 300.0)
SLAB_HALF_THICKNESS_PC = 30.0

# Vertical profiles (Shelest+26): one point PER GRID PLANE within the box
# (no z binning at all), superseding the earlier 10 pc bins.
VERTICAL_PROFILE_PER_PLANE = True

# ---------------------------------------------------------------------------
# Percentile convention (Shelest+26 switch)
# ---------------------------------------------------------------------------
# Shelest et al. 2026 report 15th/85th percentiles; "16_84" (the nominal
# 1-sigma pair used up to Phase B) is kept available behind the switch.
PERCENTILE_SCHEME_15_85 = "15_85"
PERCENTILE_SCHEME_16_84 = "16_84"
PERCENTILE_SCHEME_DEFAULT = PERCENTILE_SCHEME_15_85
PERCENTILE_LEVELS_BY_SCHEME = {
    PERCENTILE_SCHEME_15_85: (15.0, 85.0),
    PERCENTILE_SCHEME_16_84: (16.0, 84.0),
}

# ---------------------------------------------------------------------------
# Density cube provenance (PROVISIONAL)
# ---------------------------------------------------------------------------
# The cube currently in use (f98_sm_opt2_edendist.zarr, a single
# config/local_config.yaml entry: zarr_filename) has n_H = 1653 * A'
# baked into its "density" field, where A' is the Edenhofer+23
# differential extinction [E pc^-1]. Every number produced from it is
# PROVISIONAL: a correctly oriented version of the same cube, and later a
# final Porter-FUV cube with n_H = 1727 * A', will replace it. Only the one
# config entry changes when that happens -- no code path hardcodes a cube
# path or a conversion factor other than these two constants.
N_H_PER_EXTINCTION_F98 = 1653.0       # current, provisional cube
N_H_PER_EXTINCTION_PORTER_FUV = 1727.0  # final cube, pending
PROVISIONAL_CUBE_HEADER = (
    "PROVISIONAL: f98 cube, n_H = 1653 A'; final cube pending"
)

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

# ---------------------------------------------------------------------------
# Gas self-gravity mode (Step 1b)
# ---------------------------------------------------------------------------
# SETTLED: Phase B found the per-column ("infinite slab per line of sight")
# self-gravity inflates mass-weighted alpha by x1.9-2.6 near the midplane,
# because it lets a single compact dense cloud source its own huge local
# g_gas, over-weighting exactly the clumps that mass-weighting already
# emphasizes. Guo+20's external field is a smooth, horizontally uniform
# disk model -- so the physically consistent gas self-gravity to add
# alongside it is likewise horizontally averaged over the footprint, not
# a per-sightline slab. "footprint_mean" is now the default; "per_column"
# (the original Step 1 behavior, unchanged) is kept as an explicit
# sensitivity option.
SELF_GRAVITY_MODE_FOOTPRINT_MEAN = "footprint_mean"
SELF_GRAVITY_MODE_PER_COLUMN = "per_column"
SELF_GRAVITY_MODE_OFF = "off"
SELF_GRAVITY_MODE_DEFAULT = SELF_GRAVITY_MODE_FOOTPRINT_MEAN


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
