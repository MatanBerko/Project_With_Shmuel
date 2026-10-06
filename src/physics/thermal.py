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

Shelest+26 phase boundaries (build_phase_density_bounds)
--------------------------------------------------------
Shelest et al. 2026 classify phases by DENSITY against the two turning
points of the same BS19 equilibrium S-curve: n_W,max(I_UV) and
n_C,min(I_UV), the densities where dP/dn = 0.

These are read off the .mat file's OTHER tabulation of the same curve,
T_2d_n(I_UV, n) on the log-uniform n_ grid (591 points), rather than off
T_2d_P(I_UV, P) used for P_min/P_max above. Reason: the equilibrium curve
is single-valued in n (one equilibrium T per density) but TRIPLE-valued in
P across the two-phase range, and T_2d_P -- being tabulated on a P grid --
jumps between branches unpredictably there (at I_UV = 1 it flips from
T ~ 8400 K to T ~ 266 K between two adjacent P samples, and dn/dP changes
sign 18 times inside the two-phase range). P_min/P_max survive that
because they only need the FIRST and LAST such sign change -- i.e. the
lowest P at which a cold root appears and the highest P at which a warm
root still exists -- but the DENSITY at a turning point cannot be read
reliably off a branch-jumping table. P(n) = n * T_2d_n(I_UV, n) has no
such ambiguity: at I_UV = 1 it has exactly two clean dP/dn sign changes.

Consistency: the turning-point pressures of the n-grid curve reproduce the
P-grid P_min/P_max to ~5% (I_UV = 1: 2445 vs 2570 and 7191 vs 6880
K cm^-3) -- the same two physical points, located by two different
estimators on two tabulations of one curve. P_min/P_max (and therefore the
HIM flag) are left EXACTLY as they were; nothing above this section
changed.

Method (reported, not implicit):
  * For each tabulated I_UV, form P(n) = n * T_2d_n(I_UV, n) on the
    log-uniform n_ grid and take dP/d(log n) via np.gradient.
  * Locate sign changes of dP/d(log n). The FIRST is the warm-branch
    turning point (local P maximum, the highest pressure at which warm gas
    exists) -> n_W,max; the LAST is the cold-branch turning point (local P
    minimum, the lowest pressure at which cold gas exists) -> n_C,min.
    This is the same first/last convention build_pmin_pmax() already uses,
    so the two pair up: n_W,max <-> P_max and n_C,min <-> P_min.
  * Each is refined by linear root-finding on dP/d(log n) between the two
    bracketing log n samples (the same _lin_root() used for P_min/P_max,
    applied in log n since the n grid is log-uniform).
  * n_W,max(I_UV) and n_C,min(I_UV) are then log-log LINEAR interpolations
    (interp1d, kind="linear") in log10 I_UV, with constant edge fill for
    out-of-range I_UV -- identical machinery (_make_interp) to P_min/P_max.
  * For I_UV >~ 250 the S-curve flattens out entirely (no thermal
    instability, a single stable phase): no turning points exist, those
    rows are dropped as NaN, and the edge fill of the highest I_UV that
    still has an S-curve applies -- exactly how build_pmin_pmax() already
    treats the same rows.
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


@dataclass
class PhaseDensityBounds:
    """Shelest+26 density phase boundaries as functions of I_UV.

    n_w_max(I_UV): density at the warm-branch dP/dn = 0 turning point.
    n_c_min(I_UV): density at the cold-branch dP/dn = 0 turning point.
    Cells with n < n_w_max are warm, n > n_c_min are cold, in between is
    thermally unstable. Always n_w_max <= n_c_min.
    """
    n_w_max: Callable[[np.ndarray], np.ndarray]
    n_c_min: Callable[[np.ndarray], np.ndarray]
    method: str


PHASE_DENSITY_BOUNDS_METHOD = (
    "dP/dn = 0 turning points of P(n) = n * T_2d_n(I_UV, n) on the BS19 "
    "log-uniform n_ grid; first sign change of dP/d(log n) -> n_W,max "
    "(warm-branch turning point), last -> n_C,min (cold-branch turning "
    "point), each refined by linear root-finding in log n; both then "
    "interpolated linearly in log10(I_UV) vs log10(n) with constant edge "
    "fill. I_UV rows with no S-curve (I_UV >~ 250) are dropped as NaN."
)


def locate_scurve_turning_points(n_grid: np.ndarray, T_of_n: np.ndarray):
    """The two dP/dn = 0 turning points of P(n) = n * T_of_n(n).

    n_grid must be ascending and positive; T_of_n is the equilibrium
    temperature at each of those densities (single-valued, which is why the
    n tabulation is used here and not the P one -- see module docstring).
    Returns (n_W_max, n_C_min, P_at_W_max, P_at_C_min), or None when the
    curve has fewer than two dP/dn sign changes (no thermal instability at
    this I_UV).

    Derivatives are taken against log n (the BS19 n grid is log-uniform).
    That changes nothing about WHERE dP/dn vanishes -- d(log n) and dn
    differ by the strictly positive factor n * ln(10) -- but it makes the
    linear root refinement between samples well-conditioned.
    """
    log_n = np.log10(n_grid)
    P = n_grid * np.clip(T_of_n, 1e-30, None)
    dP = np.gradient(P, log_n)

    s = np.sign(dP)
    s[s == 0] = 1
    ch = np.where(s[:-1] * s[1:] < 0)[0]
    if ch.size < 2:
        return None

    log_n_w = _lin_root(log_n, dP, int(ch[0]))    # warm-branch turning point
    log_n_c = _lin_root(log_n, dP, int(ch[-1]))   # cold-branch turning point
    return (
        10.0 ** log_n_w,
        10.0 ** log_n_c,
        float(np.interp(log_n_w, log_n, P)),
        float(np.interp(log_n_c, log_n, P)),
    )


def turning_point_densities(mat_path):
    """Raw, un-interpolated (I_UV grid, n_W,max, n_C,min, P_at_each) arrays.

    Exposed separately from build_phase_density_bounds() so the tabulated
    turning points can be inspected and cross-checked against P_min/P_max
    directly. Rows with no S-curve come back as NaN.
    """
    d = scipy.io.loadmat(str(mat_path))
    IUV_grid = np.array(d["IUV_"]).squeeze().astype(float).ravel()
    n_grid = np.array(d["n_"]).squeeze().astype(float).ravel()
    T_2d_n = np.array(d["T_2d_n"]).astype(float)

    if np.any(np.diff(n_grid) <= 0):
        order = np.argsort(n_grid)
        n_grid = n_grid[order]
        T_2d_n = T_2d_n[:, order]

    n_w_max = np.full(len(IUV_grid), np.nan)
    n_c_min = np.full(len(IUV_grid), np.nan)
    P_at_w_max = np.full(len(IUV_grid), np.nan)
    P_at_c_min = np.full(len(IUV_grid), np.nan)

    for i in range(len(IUV_grid)):
        res = locate_scurve_turning_points(n_grid, T_2d_n[i, :])
        if res is None:
            continue
        n_w_max[i], n_c_min[i], P_at_w_max[i], P_at_c_min[i] = res

    return IUV_grid, n_w_max, n_c_min, P_at_w_max, P_at_c_min


def build_phase_density_bounds(mat_path) -> PhaseDensityBounds:
    """Build n_W,max(I_UV) and n_C,min(I_UV) interpolators (Shelest+26).

    mat_path is resolved by the caller via src.config_loader
    (cfg["bs19_mat_path"]), the same as build_pmin_pmax().
    """
    IUV_grid, n_w_max_arr, n_c_min_arr, _, _ = turning_point_densities(mat_path)
    return PhaseDensityBounds(
        n_w_max=_make_interp(IUV_grid, n_w_max_arr),
        n_c_min=_make_interp(IUV_grid, n_c_min_arr),
        method=PHASE_DENSITY_BOUNDS_METHOD,
    )
