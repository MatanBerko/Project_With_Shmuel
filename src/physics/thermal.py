"""
BS19 thermal model: P_min(I_UV) and P_max(I_UV) interpolators, and the
P_th formula.

Ported verbatim (same algorithm, same variables) from all four reference
scripts, which agree exactly with each other:
  research_10.0/fig2_histograms/computing_data.py:78-124 build_pmin_pmax_functions()
  research_10.0/fig3_slices/compute_data.py:81-134 build_pmin_pmax_functions()
  research_10.0/fig4_vertical_profiles/compute_data.py:94-139 build_pmin_pmax()
  research_10.0/fig4b_velocity_dispertion_Mach_number/compute_data.py:86-122 build_pmin_func()
    (fig4b only needs P_min, not P_max, but its algorithm for P_min is identical)

Method: for each tabulated I_UV row, invert n(P) = P / T(P) (T from the
BS19 T_2d_P(I_UV, P) grid) to find n(P)'s turning points (where dn/dP
changes sign): the first sign change is P_min (WNM/UNM boundary), the last
is P_max (UNM/CNM boundary), each refined by linear root-finding between
the two bracketing P-grid points. P_min(I_UV) and P_max(I_UV) are then each
a log-log LINEAR interpolation (scipy.interpolate.interp1d, kind="linear")
over the tabulated turning points, with a constant fill (edge value) for
out-of-range I_UV.

NOTE on T(n, I_UV): the four reference scripts do NOT re-derive voxel
temperature from the BS19 table -- they take T directly from the zarr's
own precomputed "T" field. The BS19 table (via its T_2d_P(I_UV, P) grid)
is used here only to build the P_min/P_max(I_UV) interpolators used for
HIM classification and the HIM_A/HIM_B substitution, matching that source
usage exactly.

P_th formula (Eq. 1 of the paper draft, also pipeline/compute_fig1.py:14):
    P_th / k_B = n * T   [K cm^-3]
"Pth" as stored/returned throughout the reference scripts and here is
already P_th/k_B (n*T with n in cm^-3, T in K) -- k_B is only used
separately to convert the hydrostatic integral's cgs pressure into P/k_B.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.io
from scipy.interpolate import interp1d

from src.conventions import IUV_FLOOR


@dataclass
class PminPmaxFunctions:
    p_min: Callable[[np.ndarray], np.ndarray]
    p_max: Callable[[np.ndarray], np.ndarray]


def p_th_over_kb(n_cm3: np.ndarray, T_K: np.ndarray) -> np.ndarray:
    """P_th / k_B [K cm^-3] = n * T."""
    return n_cm3 * T_K


def _lin_root(P_grid: np.ndarray, dndP: np.ndarray, j: int) -> float:
    x0, x1 = P_grid[j], P_grid[j + 1]
    y0, y1 = dndP[j], dndP[j + 1]
    return x0 - y0 * (x1 - x0) / (y1 - y0) if (y1 - y0) != 0 else x0


def _make_interp(IUV_grid: np.ndarray, P_arr: np.ndarray):
    good = np.isfinite(P_arr) & (P_arr > 0) & (IUV_grid > 0)
    logI = np.log10(IUV_grid[good])
    logP = np.log10(P_arr[good])
    idx = np.argsort(logI)
    f = interp1d(
        logI[idx], logP[idx], kind="linear",
        bounds_error=False,
        fill_value=(logP[idx][0], logP[idx][-1]),
    )
    return lambda Iuv: 10.0 ** f(
        np.log10(np.clip(np.asarray(Iuv, dtype=float), IUV_FLOOR, None))
    )


def build_pmin_pmax(mat_path) -> PminPmaxFunctions:
    """Build P_min(I_UV) and P_max(I_UV) interpolators from Temp_Bialy_interp.mat.

    mat_path resolved by the caller via src.config_loader (cfg["bs19_mat_path"]).
    """
    d = scipy.io.loadmat(str(mat_path))

    IUV_grid = np.array(d["IUV_"]).squeeze().astype(float).ravel()
    P_grid = np.array(d["P_"]).squeeze().astype(float).ravel()
    T_2d_P = np.array(d["T_2d_P"]).astype(float)

    if np.any(np.diff(P_grid) <= 0):
        order = np.argsort(P_grid)
        P_grid = P_grid[order]
        T_2d_P = T_2d_P[:, order]

    Pmin_list, Pmax_list = [], []
    for i in range(len(IUV_grid)):
        T_row = np.clip(T_2d_P[i, :], 1e-30, None)
        n_row = np.clip(P_grid / T_row, 1e-30, None)
        dn = np.gradient(n_row, P_grid)
        s = np.sign(dn)
        s[s == 0] = 1
        ch = np.where(s[:-1] * s[1:] < 0)[0]

        if ch.size == 0:
            Pmin_list.append(np.nan)
            Pmax_list.append(np.nan)
            continue

        Pmin_list.append(float(_lin_root(P_grid, dn, int(ch[0]))))
        Pmax_list.append(float(_lin_root(P_grid, dn, int(ch[-1]))))

    Pmin_arr = np.array(Pmin_list, dtype=float)
    Pmax_arr = np.array(Pmax_list, dtype=float)

    return PminPmaxFunctions(
        p_min=_make_interp(IUV_grid, Pmin_arr),
        p_max=_make_interp(IUV_grid, Pmax_arr),
    )
