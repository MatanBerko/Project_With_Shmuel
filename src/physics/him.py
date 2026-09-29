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
"""

from dataclasses import dataclass

import numpy as np

from src.conventions import CNM_TEMP_MAX_K, HIM_THRESHOLD_FACTOR, T_HIM_K, WNM_TEMP_MIN_K

VARIANTS = ("RAW", "HIM_A", "HIM_B")

# Phase flag codes (int8), stored in cache/core/alpha_core.zarr's phase_flag.
PHASE_CNM = 0
PHASE_UNM = 1
PHASE_WNM = 2
PHASE_HIM = 3


def him_flag(Pth_raw: np.ndarray, Pmin: np.ndarray) -> np.ndarray:
    """HIM: P_th < HIM_THRESHOLD_FACTOR * P_min (strict), and P_min finite."""
    return (Pth_raw < HIM_THRESHOLD_FACTOR * Pmin) & np.isfinite(Pmin)


def phase_flag(T_K: np.ndarray, him: np.ndarray) -> np.ndarray:
    """int8 phase flag: CNM=0, UNM=1, WNM=2, HIM=3. HIM takes precedence."""
    flag = np.where(T_K <= CNM_TEMP_MAX_K, PHASE_CNM,
                     np.where(T_K <= WNM_TEMP_MIN_K, PHASE_UNM, PHASE_WNM))
    flag = np.where(him, PHASE_HIM, flag)
    return flag.astype(np.int8)


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
