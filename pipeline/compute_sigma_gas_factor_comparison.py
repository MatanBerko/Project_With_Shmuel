"""
Compute script comparing two published dust-extinction-to-n_H conversion
factors, applied to the SAME raw quantity: the native Edenhofer et al.
(2023/2024) differential extinction dE/ds [E pc^-1] (E = Zhang, Green &
Rix 2023 extinction unit), queried directly via the dustmaps package
(dustmaps.edenhofer2023.Edenhofer2023Query, integrated=False) rather than
reusing the pre-baked "density" array already in f98_sm_opt2_edendist.zarr
(that array was built with some unknown, possibly-old conversion baked in
-- this script recomputes n_H from the raw extinction map instead, so both
factors are applied to an identical, unbaked input).

    n_H_old = 1652 * dE/ds cm^-3   [Zucker+21 / O'Neill+24 / McCallum+25]
    n_H_new = 2700 * dE/ds cm^-3   [McCallum et al. 2026, arXiv:2607.07451]

NOTE on missing data: the Edenhofer map has a real inner distance cutoff
at ~68.6 pc from the Sun (confirmed via q.distance_bounds) -- there is no
reconstructed dust density closer than that. Any (x, y) column whose
line of sight passes within 68.6 pc of the origin will contain NaN values
near z=0, and the trapezoidal z-integration will correctly propagate that
NaN across the whole column (no interpolation/extrapolation is invented
to fill this gap). This produces a small NaN disk near the map center in
both Sigma_gas_old and Sigma_gas_new -- expected, not a bug, and it
affects both versions identically since they share the same raw grid.

Grid convention matches compute_sigma_gas.py / the existing zarr file:
heliocentric Galactic Cartesian coordinates, 2 pc voxel spacing, the same
+-500 pc square footprint in x, y (reusing the zarr's own x_pc/y_pc
coordinate arrays), full z range (-750 to 750 pc).
"""

import time

import astropy.units as u
import numpy as np
import zarr
from astropy.coordinates import Galactic, SkyCoord
from dustmaps.edenhofer2023 import Edenhofer2023Query

from src.config_loader import load_resolved_config

CACHE_PATH = "cache/sigma_gas_factor_comparison.npz"

# Conversion factors: n_H [cm^-3] = FACTOR * dE/ds [E pc^-1]
FACTOR_OLD = 1652.0  # Zucker+21 / O'Neill+24 / McCallum+25 convention
FACTOR_NEW = 2700.0  # McCallum et al. 2026, arXiv:2607.07451

# Same physical constants as compute_sigma_gas.py (reused exactly, not redefined)
MU = 1.4  # mean molecular weight per H, project convention
M_H = 1.6726219e-24  # grams (hydrogen mass)
PC_TO_CM = 3.0856775814913673e18  # cm per pc
MSUN_G = 1.98892e33  # grams per solar mass


def sigma_gas_from_n_H_column(N_H_column_pc_cm3):
    """Same unit chain as compute_sigma_gas.py: cm^-3*pc -> cm^-2 -> g/cm^2 -> Msun/pc^2."""
    N_H_cm2 = N_H_column_pc_cm3 * PC_TO_CM
    Sigma_g_cm2 = MU * M_H * N_H_cm2
    return Sigma_g_cm2 * (PC_TO_CM ** 2) / MSUN_G


def main():
    cfg = load_resolved_config()
    store = zarr.open(str(cfg["zarr_path"]), mode="r")

    x_pc = store["x_pc"][:]
    y_pc = store["y_pc"][:]
    z_pc = store["z_pc"][:]

    # Same +-500 pc square footprint index lookup as compute_sigma_gas.py
    x_idx = np.where(np.abs(x_pc) <= 500)[0]
    y_idx = np.where(np.abs(y_pc) <= 500)[0]
    x_lo, x_hi = x_idx.min(), x_idx.max() + 1
    y_lo, y_hi = y_idx.min(), y_idx.max() + 1
    x_sub = x_pc[x_lo:x_hi]
    y_sub = y_pc[y_lo:y_hi]

    print(f"x_pc range [{x_sub.min()}, {x_sub.max()}], {len(x_sub)} points")
    print(f"y_pc range [{y_sub.min()}, {y_sub.max()}], {len(y_sub)} points")
    print(f"z_pc range [{z_pc.min()}, {z_pc.max()}], {len(z_pc)} points")

    print("Loading Edenhofer2023Query (integrated=False)...")
    t0 = time.time()
    query = Edenhofer2023Query(integrated=False)
    print(f"  loaded in {time.time() - t0:.1f}s")
    print(f"  map distance_bounds: [{query.distance_bounds.min()}, "
          f"{query.distance_bounds.max()}]")

    X, Y = np.meshgrid(x_sub, y_sub)  # shape (y_sub, x_sub)

    # Query dE/ds slice-by-slice over z: vectorized per slice (not a
    # per-point Python loop), chunked over the z axis to bound memory.
    n_z = len(z_pc)
    dEds = np.empty((n_z, len(y_sub), len(x_sub)), dtype=np.float64)
    t_start = time.time()
    for iz, z_val in enumerate(z_pc):
        Z = np.full_like(X, z_val)
        gal = Galactic(u=X * u.pc, v=Y * u.pc, w=Z * u.pc, representation_type="cartesian")
        coords = SkyCoord(gal)
        coords.representation_type = "spherical"
        dEds[iz] = query.query(coords)

        elapsed = time.time() - t_start
        if elapsed > 30 and (iz % 50 == 0 or iz == n_z - 1):
            rate = (iz + 1) / elapsed
            eta = (n_z - iz - 1) / rate if rate > 0 else float("nan")
            print(f"  z-slice {iz + 1}/{n_z}, elapsed {elapsed:.0f}s, ETA {eta:.0f}s")

    print(f"Finished querying dE/ds grid in {time.time() - t_start:.1f}s, "
          f"shape {dEds.shape}, NaN fraction {np.mean(np.isnan(dEds)):.4%}")

    n_H_old = FACTOR_OLD * dEds
    n_H_new = FACTOR_NEW * dEds

    # Integrate over the FULL z range via trapezoidal rule against real z_pc
    # values, same method as compute_sigma_gas.py. NaNs (inside the map's
    # ~68.6 pc inner cutoff) propagate through the whole column, by design.
    N_H_column_old = np.trapezoid(n_H_old, x=z_pc, axis=0)
    N_H_column_new = np.trapezoid(n_H_new, x=z_pc, axis=0)

    sigma_gas_old = sigma_gas_from_n_H_column(N_H_column_old)
    sigma_gas_new = sigma_gas_from_n_H_column(N_H_column_new)
    ratio_new_old = sigma_gas_new / sigma_gas_old

    stats = {
        "mean_old": float(np.nanmean(sigma_gas_old)),
        "median_old": float(np.nanmedian(sigma_gas_old)),
        "mean_new": float(np.nanmean(sigma_gas_new)),
        "median_new": float(np.nanmedian(sigma_gas_new)),
        "mean_ratio_new_old": float(np.nanmean(ratio_new_old)),
        "median_ratio_new_old": float(np.nanmedian(ratio_new_old)),
        "std_ratio_new_old": float(np.nanstd(ratio_new_old)),
    }
    print("Summary statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v:.6g}")
    print(f"  expected ratio (2700/1652): {FACTOR_NEW / FACTOR_OLD:.6g}")

    np.savez(
        CACHE_PATH,
        sigma_gas_old=sigma_gas_old,
        sigma_gas_new=sigma_gas_new,
        ratio_new_old=ratio_new_old,
        x_pc=x_sub,
        y_pc=y_sub,
        **stats,
    )
    print(f"Saved {CACHE_PATH}")


if __name__ == "__main__":
    main()
