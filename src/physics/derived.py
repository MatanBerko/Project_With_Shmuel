"""
Derived quantities: Sigma_gas(x,y), alpha = P_tot/P_th, sigma_eff, Mach.

Sigma_gas ported from pipeline/compute_sigma_gas.py's unit chain (also
matching fig2_histograms/computing_data.py:217-224 and
fig3_slices/compute_data.py:276-283, modulo the DZ_PC=1.0 bug those two
had -- this uses the real z_pc trapezoid, matching the SETTLED convention).

alpha ported from fig4_vertical_profiles/compute_data.py:277
    alpha_arr = np.where(Pth_model > P_FLOOR, Ptot_kB / Pth_model, np.nan)

sigma_eff / Mach ported verbatim from
fig4b_velocity_dispertion_Mach_number/compute_data.py:127-171
    sigma_eff_kmps(), mach_number(), sound_speed_kmps()
with mu = 1.4 in k_B/(mu*m_H), matching KB_OVER_MU_MH there exactly.
"""

import numpy as np

from src.conventions import K_B, M_H, M_SUN_G, MU, PC_CM, P_FLOOR

KB_OVER_MU_MH = K_B / (MU * M_H)  # erg g^-1 K^-1
KMS = 1.0e5  # cm/s per km/s


def sigma_gas_map(z_pc: np.ndarray, n_cm3: np.ndarray) -> np.ndarray:
    """Sigma_gas(x,y) [Msun/pc^2]: full-column trapezoid of n_cm3 against
    the real z_pc, axis=0 is z. n_cm3 shape (Nz, Ny, Nx) or (Nz,).
    """
    N_H_column_pc_cm3 = np.trapezoid(n_cm3, x=z_pc, axis=0)
    N_H_cm2 = N_H_column_pc_cm3 * PC_CM
    Sigma_g_cm2 = MU * M_H * N_H_cm2
    return Sigma_g_cm2 * (PC_CM ** 2) / M_SUN_G


def alpha(Ptot_kB: np.ndarray, Pth_kB: np.ndarray) -> np.ndarray:
    """alpha = P_tot / P_th, NaN where P_th <= P_FLOOR."""
    return np.where(Pth_kB > P_FLOOR, Ptot_kB / Pth_kB, np.nan)


def sigma_eff_kmps(alpha_val: np.ndarray, T_K: np.ndarray) -> np.ndarray:
    """Effective non-thermal 3D velocity dispersion [km/s].
    sigma_eff = sqrt(3*(alpha-1)*k_B*T/(mu*m_H))
    """
    alpha_val = np.asarray(alpha_val, dtype=float)
    T_K = np.asarray(T_K, dtype=float)
    dA = np.maximum(alpha_val - 1.0, 0.0)
    return np.where(
        np.isfinite(alpha_val) & np.isfinite(T_K) & (T_K > 0),
        np.sqrt(3.0 * dA * KB_OVER_MU_MH * T_K) / KMS,
        np.nan,
    )


def mach_number(alpha_val: np.ndarray) -> np.ndarray:
    """Turbulent Mach number M = sigma_eff/c_s = sqrt(3*(alpha-1))."""
    alpha_val = np.asarray(alpha_val, dtype=float)
    dA = np.maximum(alpha_val - 1.0, 0.0)
    return np.where(np.isfinite(alpha_val), np.sqrt(3.0 * dA), np.nan)


def sound_speed_kmps(T_K: np.ndarray) -> np.ndarray:
    """Isothermal sound speed [km/s]: c_s = sqrt(k_B*T/(mu*m_H))."""
    T_K = np.asarray(T_K, dtype=float)
    return np.where(T_K > 0, np.sqrt(KB_OVER_MU_MH * T_K) / KMS, np.nan)
