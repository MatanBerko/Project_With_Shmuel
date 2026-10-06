"""
Derived quantities: Sigma_gas(x,y), alpha = P_tot/p_th_phys, the sound
speed, the two velocity dispersions, and Mach.

Sigma_gas ported from pipeline/compute_sigma_gas.py's unit chain (also
matching fig2_histograms/computing_data.py:217-224 and
fig3_slices/compute_data.py:276-283, modulo the DZ_PC=1.0 bug those two
had -- this uses the real z_pc trapezoid, matching the SETTLED convention).

alpha ported from fig4_vertical_profiles/compute_data.py:277
    alpha_arr = np.where(Pth_model > P_FLOOR, Ptot_kB / Pth_model, np.nan)
Its DENOMINATOR changed in Step 1d: it is now p_th_phys (= 1.1 n_H T),
not p_nT (= n_H T). The formula is untouched; what is handed to it is
not. See src.physics.thermal for why.

Step 1d: the sound speed and both dispersions
---------------------------------------------
Thermal pressure counts particles, so the isothermal sound speed of
neutral gas is

    c_s^2 = PARTICLES_PER_H_NEUTRAL * k_B T / (MU * m_H)
          = 1.1 k_B T / (1.4 m_H) = k_B T / (1.273 m_H),

i.e. the familiar mean mass per particle of ~1.27 m_H for neutral atomic
gas with helium. Under THERMAL_PRESSURE_CONVENTION = "nT" the factor is
1 and this collapses to the pre-Step-1d k_B T / (1.4 m_H).

TWO dispersions are provided, because they answer different questions and
the pre-Step-1d code reported the second one under the first one's name:

    sigma_eff_kmps  = sqrt(alpha) * c_s        (NEW definition)
        The TOTAL effective dispersion: sigma_eff^2 = alpha c_s^2 =
        P_tot / rho exactly, i.e. the dispersion that would supply the
        whole hydrostatic pressure. Note that every particle-count factor
        cancels out of it -- alpha carries 1/f and c_s^2 carries f -- so
        sigma_eff is IDENTICAL under both conventions, to numerical
        precision. That is a property worth having: it is a statement
        about P_tot and rho, neither of which this step changed.

    sigma_nt_kmps   = sqrt(3 (alpha - 1)) * c_s = Mach * c_s
        The NON-THERMAL (turbulent) 3D dispersion, which is what
        fig4b_velocity_dispertion_Mach_number/compute_data.py:127-171
        computed and what Step 1c reported as "sigma_eff". Kept under its
        own, accurate name rather than dropped, so the old number stays
        reproducible. Unlike sigma_eff this is convention-DEPENDENT: the
        "- 1" breaks the cancellation.

Mach = sqrt(3 (alpha - 1)) is unchanged, and remains the turbulent Mach
number sigma_nt / c_s (not sigma_eff / c_s, which is sqrt(alpha)).
"""

import numpy as np

from src.conventions import (
    K_B,
    M_H,
    M_SUN_G,
    MU,
    PC_CM,
    P_FLOOR,
    THERMAL_PRESSURE_CONVENTION_DEFAULT,
)
from src.physics.thermal import particles_per_h

KB_OVER_MU_MH = K_B / (MU * M_H)  # erg g^-1 K^-1 -- MASS factor only
KMS = 1.0e5  # cm/s per km/s


def sigma_gas_map(z_pc: np.ndarray, n_cm3: np.ndarray) -> np.ndarray:
    """Sigma_gas(x,y) [Msun/pc^2]: full-column trapezoid of n_cm3 against
    the real z_pc, axis=0 is z. n_cm3 shape (Nz, Ny, Nx) or (Nz,).
    """
    N_H_column_pc_cm3 = np.trapezoid(n_cm3, x=z_pc, axis=0)
    N_H_cm2 = N_H_column_pc_cm3 * PC_CM
    Sigma_g_cm2 = MU * M_H * N_H_cm2
    return Sigma_g_cm2 * (PC_CM ** 2) / M_SUN_G


def alpha(Ptot_kB: np.ndarray, p_th_phys_kB: np.ndarray) -> np.ndarray:
    """alpha = P_tot / p_th_phys, NaN where p_th_phys <= P_FLOOR.

    The denominator must be the PHYSICAL thermal pressure (1.1 n_H T for
    neutral gas; src.physics.thermal.p_th_phys_over_kb, or
    src.physics.him.apply_variant's p_th_phys for HIM cells) -- NOT p_nT.
    Handing p_nT to this function is the pre-Step-1d bug, and inflates
    alpha by 1.1.
    """
    return np.where(p_th_phys_kB > P_FLOOR, Ptot_kB / p_th_phys_kB, np.nan)


def sound_speed_kmps(T_K: np.ndarray,
                       convention: str = THERMAL_PRESSURE_CONVENTION_DEFAULT) -> np.ndarray:
    """Isothermal sound speed of neutral gas [km/s]:
    c_s = sqrt(PARTICLES_PER_H_NEUTRAL * k_B * T / (MU * m_H)).
    """
    f_neutral, _ = particles_per_h(convention)
    T_K = np.asarray(T_K, dtype=float)
    return np.where(T_K > 0, np.sqrt(f_neutral * KB_OVER_MU_MH * T_K) / KMS, np.nan)


def sigma_eff_kmps(alpha_val: np.ndarray, T_K: np.ndarray,
                     convention: str = THERMAL_PRESSURE_CONVENTION_DEFAULT) -> np.ndarray:
    """TOTAL effective velocity dispersion [km/s]: sigma_eff = sqrt(alpha) * c_s.

    sigma_eff^2 = alpha c_s^2 = P_tot / rho. Every particle-count factor
    cancels, so this is the same number under both conventions (see the
    module docstring). For the non-thermal part use sigma_nt_kmps().
    """
    f_neutral, _ = particles_per_h(convention)
    alpha_val = np.asarray(alpha_val, dtype=float)
    T_K = np.asarray(T_K, dtype=float)
    A = np.maximum(alpha_val, 0.0)
    return np.where(
        np.isfinite(alpha_val) & np.isfinite(T_K) & (T_K > 0),
        np.sqrt(A * f_neutral * KB_OVER_MU_MH * T_K) / KMS,
        np.nan,
    )


def sigma_nt_kmps(alpha_val: np.ndarray, T_K: np.ndarray,
                    convention: str = THERMAL_PRESSURE_CONVENTION_DEFAULT) -> np.ndarray:
    """NON-THERMAL (turbulent) 3D velocity dispersion [km/s]:
    sigma_nt = sqrt(3*(alpha-1)) * c_s = Mach * c_s.

    This is what fig4b_velocity_dispertion_Mach_number/compute_data.py
    computed and what Step 1c reported as "sigma_eff". Convention-
    dependent, because the "- 1" stops the particle-count factor
    cancelling.
    """
    f_neutral, _ = particles_per_h(convention)
    alpha_val = np.asarray(alpha_val, dtype=float)
    T_K = np.asarray(T_K, dtype=float)
    dA = np.maximum(alpha_val - 1.0, 0.0)
    return np.where(
        np.isfinite(alpha_val) & np.isfinite(T_K) & (T_K > 0),
        np.sqrt(3.0 * dA * f_neutral * KB_OVER_MU_MH * T_K) / KMS,
        np.nan,
    )


def mach_number(alpha_val: np.ndarray) -> np.ndarray:
    """Turbulent Mach number M = sigma_nt/c_s = sqrt(3*(alpha-1)).

    Unchanged by Step 1d as a formula, but its INPUT alpha is now built
    on p_th_phys, so the number it returns does change.
    """
    alpha_val = np.asarray(alpha_val, dtype=float)
    dA = np.maximum(alpha_val - 1.0, 0.0)
    return np.where(np.isfinite(alpha_val), np.sqrt(3.0 * dA), np.nan)
