"""
HIM classification, phase classification (CNM/UNM/WNM/HIM), and the
RAW/HIM_A/HIM_B density+pressure variant construction.

Ported from (all four reference scripts agree on the HIM threshold and the
HIM_A/HIM_B substitution formula):
  fig2_histograms/computing_data.py:191-207 full_z_n() (HIM_A/HIM_B inline)
  fig3_slices/compute_data.py:245-265 full_z_n()
  fig4_vertical_profiles/compute_data.py:194-205 _apply_model()
  fig4b_velocity_dispertion_Mach_number/compute_data.py: HIM mask only
    (:220), no HIM_A/HIM_B substitution needed there.

No "MASKED" variant is defined in any of the four reference scripts (or
anywhere else in research_10.0) -- grepped exhaustively, no match. Per the
task's own instruction ("MASKED: port the old definition; if none exists,
STOP and report"), MASKED is NOT implemented here. Only RAW, HIM_A, HIM_B
are provided. See the final report for this flag.

Phase thresholds (T_CNM=300K, T_WNM=6000K) and HIM_THRESHOLD_FACTOR=0.5
come from src.conventions, matching fig4_vertical_profiles/compute_data.py
and fig4b_velocity_dispertion_Mach_number/compute_data.py exactly (fig2 and
fig3 don't classify phases at all, only the HIM flag).

PHASE_SCHEME (Shelest+26) -- two neutral-phase classifications
--------------------------------------------------------------
classify_phases() dispatches on PHASE_SCHEME (src.conventions):

  "dPdn" (NEW DEFAULT, Shelest et al. 2026, arXiv:2607.15352) -- classify
    by DENSITY against the two dP/dn = 0 turning points of the BS19
    thermal-equilibrium S-curve at that cell's I_UV:
        warm     (WNM)  n <  n_W,max(I_UV)
        unstable (UNM)  n_W,max(I_UV) <= n <= n_C,min(I_UV)
        cold     (CNM)  n >  n_C,min(I_UV)
    n_W,max and n_C,min come from src.physics.thermal.
    build_phase_density_bounds(); see that module for how they are located
    and interpolated. The boundaries are I_UV-dependent, so unlike the
    temperature cuts they move from cell to cell.

  "temperature" (the four reference scripts' behavior, unchanged) --
    classify by TEMPERATURE against two fixed cuts: CNM T <= 300 K,
    UNM 300-6000 K, WNM T > 6000 K.

The HIM flag is SEPARATE from and unaffected by this switch: it is always
P_th < HIM_THRESHOLD_FACTOR * P_min(I_UV), it is computed once by
him_flag(), and it takes precedence over the neutral classification under
BOTH schemes (an HIM cell is PHASE_HIM, never CNM/UNM/WNM).

Which density: the dPdn scheme is a statement about the OBSERVED gas, so
it is applied to the observed density (n_raw), not to a variant's
HIM-substituted n_model. Since HIM cells are relabelled PHASE_HIM anyway
-- and HIM cells are the only ones whose density a variant changes -- both
choices give an identical phase_flag; using n_raw just makes that
independence explicit rather than accidental, and keeps one phase_flag
valid for all three variants.
"""

from dataclasses import dataclass

import numpy as np

from src.conventions import (
    CNM_TEMP_MAX_K,
    HIM_THRESHOLD_FACTOR,
    PHASE_SCHEME_DEFAULT,
    PHASE_SCHEME_DPDN,
    PHASE_SCHEME_TEMPERATURE,
    T_HIM_K,
    WNM_TEMP_MIN_K,
)

VARIANTS = ("RAW", "HIM_A", "HIM_B")

# Phase flag codes (int8), stored in cache/core/alpha_core.zarr as
# phase_flag_dpdn / phase_flag_temperature (one array per PHASE_SCHEME).
PHASE_CNM = 0
PHASE_UNM = 1
PHASE_WNM = 2
PHASE_HIM = 3


def him_flag(Pth_raw: np.ndarray, Pmin: np.ndarray) -> np.ndarray:
    """HIM: P_th < HIM_THRESHOLD_FACTOR * P_min (strict), and P_min finite."""
    return (Pth_raw < HIM_THRESHOLD_FACTOR * Pmin) & np.isfinite(Pmin)


def phase_flag_temperature(T_K: np.ndarray, him: np.ndarray) -> np.ndarray:
    """int8 phase flag from the fixed TEMPERATURE cuts (300 K / 6000 K).

    CNM=0, UNM=1, WNM=2, HIM=3. HIM takes precedence. This is the behavior
    of all four reference scripts, kept available behind PHASE_SCHEME.
    """
    flag = np.where(T_K <= CNM_TEMP_MAX_K, PHASE_CNM,
                     np.where(T_K <= WNM_TEMP_MIN_K, PHASE_UNM, PHASE_WNM))
    flag = np.where(him, PHASE_HIM, flag)
    return flag.astype(np.int8)


# Pre-switch name, kept so existing callers and the Phase A regression
# script (scripts/validation/regression_vs_old.py, which must keep
# reproducing the OLD temperature behavior) are unaffected by the switch.
phase_flag = phase_flag_temperature


def phase_flag_dpdn(n_cm3: np.ndarray, n_w_max: np.ndarray, n_c_min: np.ndarray,
                      him: np.ndarray) -> np.ndarray:
    """int8 phase flag from the Shelest+26 dP/dn = 0 DENSITY boundaries.

    n_cm3:   observed density [cm^-3].
    n_w_max: n_W,max(I_UV) evaluated at each cell's I_UV -- the density at
             the warm-branch dP/dn = 0 turning point.
    n_c_min: n_C,min(I_UV) evaluated at each cell's I_UV -- the density at
             the cold-branch dP/dn = 0 turning point.
    him:     the (scheme-independent) HIM flag, which takes precedence.

    WNM: n < n_W,max.  CNM: n > n_C,min.  UNM: in between (inclusive of
    both boundaries, so the three classes partition every finite density
    with no gap and no overlap).

    Cells where a boundary is not finite (no S-curve at that I_UV, or a
    non-finite density) are left UNM rather than silently pushed to a
    stable phase -- UNM is the "cannot be called either" class, and a
    boundary-free I_UV is exactly that case. dpdn_unresolved_fraction()
    below reports how often this happens so it is never an invisible default.
    """
    n_cm3 = np.asarray(n_cm3)
    resolvable = np.isfinite(n_cm3) & np.isfinite(n_w_max) & np.isfinite(n_c_min)
    with np.errstate(invalid="ignore"):
        is_warm = resolvable & (n_cm3 < n_w_max)
        is_cold = resolvable & (n_cm3 > n_c_min)
    flag = np.where(is_warm, PHASE_WNM, np.where(is_cold, PHASE_CNM, PHASE_UNM))
    flag = np.where(him, PHASE_HIM, flag)
    return flag.astype(np.int8)


def dpdn_unresolved_fraction(n_cm3: np.ndarray, n_w_max: np.ndarray,
                               n_c_min: np.ndarray, him: np.ndarray) -> float:
    """Fraction of NON-HIM cells that phase_flag_dpdn() had to default to
    UNM because a boundary (or the density) was not finite. Reported, not
    swallowed -- see phase_flag_dpdn's docstring.
    """
    neutral = ~np.asarray(him)
    n_neutral = int(neutral.sum())
    if n_neutral == 0:
        return float("nan")
    unresolved = neutral & ~(np.isfinite(n_cm3) & np.isfinite(n_w_max) & np.isfinite(n_c_min))
    return float(unresolved.sum()) / n_neutral


def classify_phases(him: np.ndarray, scheme: str = PHASE_SCHEME_DEFAULT, *,
                      T_K: np.ndarray | None = None, n_cm3: np.ndarray | None = None,
                      n_w_max: np.ndarray | None = None,
                      n_c_min: np.ndarray | None = None) -> np.ndarray:
    """int8 phase flag under the requested PHASE_SCHEME.

    scheme="dPdn" (default) needs n_cm3, n_w_max, n_c_min;
    scheme="temperature" needs T_K. The HIM flag is passed in either way
    and always wins.
    """
    if scheme == PHASE_SCHEME_TEMPERATURE:
        if T_K is None:
            raise ValueError(f"scheme={scheme!r} requires T_K.")
        return phase_flag_temperature(T_K, him)
    if scheme == PHASE_SCHEME_DPDN:
        if n_cm3 is None or n_w_max is None or n_c_min is None:
            raise ValueError(f"scheme={scheme!r} requires n_cm3, n_w_max and n_c_min.")
        return phase_flag_dpdn(n_cm3, n_w_max, n_c_min, him)
    raise ValueError(
        f"Unknown phase scheme: {scheme!r}. Expected "
        f"{PHASE_SCHEME_DPDN!r} or {PHASE_SCHEME_TEMPERATURE!r}."
    )


@dataclass
class Variant:
    n_model: np.ndarray
    Pth_model: np.ndarray


def apply_variant(variant: str, n_raw: np.ndarray, Pth_raw: np.ndarray,
                    him: np.ndarray, Pmin: np.ndarray, Pmax: np.ndarray) -> Variant:
    """Build the model-corrected density and P_th for one variant.

    RAW:   unchanged.
    HIM_A: HIM cells -> P_th = P_min, n = P_min / (k_B-free "T_HIM" convention:
           n = P_min / T_HIM, matching the reference scripts' n=P/T_HIM
           shorthand where "P" there is already P_th/k_B in K cm^-3).
    HIM_B: HIM cells -> P_th = P_max, n = P_max / T_HIM.
    """
    if variant == "RAW":
        return Variant(n_model=n_raw.copy(), Pth_model=Pth_raw.copy())
    if variant == "HIM_A":
        n_model = n_raw.copy()
        Pth_model = Pth_raw.copy()
        n_model[him] = Pmin[him] / T_HIM_K
        Pth_model[him] = Pmin[him]
        return Variant(n_model=n_model, Pth_model=Pth_model)
    if variant == "HIM_B":
        n_model = n_raw.copy()
        Pth_model = Pth_raw.copy()
        n_model[him] = Pmax[him] / T_HIM_K
        Pth_model[him] = Pmax[him]
        return Variant(n_model=n_model, Pth_model=Pth_model)
    raise ValueError(
        f"Unknown variant: {variant!r}. Only {VARIANTS} are implemented "
        "-- no MASKED definition exists in any reference script (see module "
        "docstring)."
    )
