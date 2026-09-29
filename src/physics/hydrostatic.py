"""
Hydrostatic P_tot integration: P_tot(z) = integral of rho*g from z out to
the boundary, with P=0 fixed at the outer edge of the Zarr z domain and
mirrored for z<0.

Boundary convention -- OLD vs NEW:
  OLD (fig2_histograms/computing_data.py:129-165 compute_Ptot_slices(),
       fig3_slices/compute_data.py:139-181 compute_Ptot_slices(),
       fig4_vertical_profiles/compute_data.py:210-273 compute_profiles()):
    all three restrict to `pos = np.where(z_pc >= 0)`, set P=0 at the
    LARGEST z (the top of the Zarr volume, `pos[j_top]`), and integrate
    DOWNWARD (decreasing z) via a manual trapezoidal step
    `P_j = P_above + 0.5*(integ_j+integ_above)*dz[j]`. None of the three
    ever computes z<0 -- their only callers request z in {0,150,300}, all
    non-negative, so a z<0 boundary/mirror convention was simply never
    needed or defined by the old code.
  NEW (this module, required for full |z|<=500 vertical profiles): the
    identical algorithm is applied independently to the z<=0 half, with
    P=0 fixed at the SMALLEST z (the bottom of the Zarr volume) and
    integrating upward (increasing z) toward the plane. This is the literal
    generalization of the old top-down algorithm, applied symmetrically to
    both signs, each anchored to its own physical domain edge. At the
    shared z=0 point, the positive-side (top-down) value is used, matching
    the only convention the old code ever exercised.

Trapezoidal integration is always against the real z_pc coordinate values
(never a hardcoded voxel size).
"""

import numpy as np

from src.conventions import K_B, PC_CM


def _align_leading(arr: np.ndarray, ndim: int) -> np.ndarray:
    """Reshape a (N,) array to (N, 1, 1, ...) with `ndim` total dims, so it
    broadcasts against an (N, ...) array along the leading axis.
    """
    if arr.ndim == ndim:
        return arr
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D array or one with ndim={ndim}, got shape {arr.shape}")
    return arr.reshape((arr.shape[0],) + (1,) * (ndim - 1))


def integrate_ptot_halfcolumn(z_ordered: np.ndarray, rho_ordered: np.ndarray,
                                 g_ordered: np.ndarray) -> np.ndarray:
    """P_tot/k_B [K cm^-3] along one half-column, in the SAME order as the
    inputs: z_ordered[0] must be the outer domain boundary (P=0 there),
    monotonically approaching the plane (z=0) at the last index.

    rho_ordered: (Nz, ...) mass density [g/cm^3].
    g_ordered:   (Nz,) or (Nz, ...) gravity [cm/s^2].
    """
    g_ordered = _align_leading(np.asarray(g_ordered), rho_ordered.ndim)
    integ = rho_ordered * g_ordered  # erg/cm^4 equivalent (rho*g), per z

    dz_cm = np.abs(np.diff(z_ordered)) * PC_CM  # physical thickness, always >= 0
    dz_cm = _align_leading(dz_cm, rho_ordered.ndim)

    step = 0.5 * (integ[1:] + integ[:-1]) * dz_cm  # trapezoidal increments
    P_cgs = np.concatenate([np.zeros_like(integ[:1]), np.cumsum(step, axis=0)], axis=0)
    return P_cgs / K_B


def p_tot_kb_full_column(z_pc: np.ndarray, rho_g_cm3: np.ndarray,
                            g_cgs: np.ndarray) -> np.ndarray:
    """P_tot/k_B [K cm^-3] for every z in z_pc (both signs), P=0 at the top
    AND bottom edges of the Zarr volume (mirrored halves, see module
    docstring). Output is aligned to the ORIGINAL order of z_pc.

    z_pc: (Nz,) real coordinate values, both signs present.
    rho_g_cm3, g_cgs: (Nz, ...) or (Nz,), aligned to z_pc along axis 0.
    """
    rho_g_cm3 = np.asarray(rho_g_cm3)
    g_cgs = _align_leading(np.asarray(g_cgs), rho_g_cm3.ndim)

    order = np.argsort(z_pc)
    inv_order = np.argsort(order)
    z_sorted = z_pc[order]
    rho_sorted = rho_g_cm3[order]
    g_sorted = g_cgs[order]

    pos_idx = np.where(z_sorted >= 0)[0]  # ascending: 0 ... +zmax
    neg_idx = np.where(z_sorted <= 0)[0]  # ascending: -zmax ... 0

    # Positive half: reverse to (top boundary -> plane) order, integrate,
    # then reverse back to ascending (0 -> top) order.
    z_pos_outer_to_plane = z_sorted[pos_idx][::-1]
    rho_pos_outer_to_plane = rho_sorted[pos_idx][::-1]
    g_pos_outer_to_plane = g_sorted[pos_idx][::-1]
    P_pos_outer_to_plane = integrate_ptot_halfcolumn(
        z_pos_outer_to_plane, rho_pos_outer_to_plane, g_pos_outer_to_plane)
    P_pos_ascending = P_pos_outer_to_plane[::-1]  # aligned to pos_idx (0 -> top)

    # Negative half: already (bottom boundary -> plane) order (ascending).
    z_neg_outer_to_plane = z_sorted[neg_idx]
    rho_neg_outer_to_plane = rho_sorted[neg_idx]
    g_neg_outer_to_plane = g_sorted[neg_idx]
    P_neg_ascending = integrate_ptot_halfcolumn(
        z_neg_outer_to_plane, rho_neg_outer_to_plane, g_neg_outer_to_plane)

    P_sorted = np.empty_like(rho_sorted)
    P_sorted[neg_idx] = P_neg_ascending  # z<=0, will be overwritten at z=0
    P_sorted[pos_idx] = P_pos_ascending  # z>=0, including z=0 (wins the overlap)

    return P_sorted[inv_order]
