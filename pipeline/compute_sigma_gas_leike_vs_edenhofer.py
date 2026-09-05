"""
Compute script comparing Sigma_gas(x, y) derived from two independent 3D dust
maps -- Edenhofer et al. (2023) and Leike, Glatzle & Ensslin (2020) -- on the
SAME +-400 pc square footprint in x, y (not the +-500 pc footprint used
elsewhere in this project). 400 pc was chosen (rather than 500 pc) because
it is Leike2020's own reliable coverage radius; Edenhofer's reliable radius
(~500 pc) comfortably covers this smaller footprint too.

IMPORTANT DEVIATION FROM THIS PROJECT'S USUAL "FULL Z RANGE" CONVENTION:
Leike2020's actual stored HDF5 box is NOT a cube -- it spans +-370 pc in x
and y, but only +-270 pc in z (confirmed directly from Leike2020Query's
_xyz0=(-370,-370,-270) and _shape=(740,740,540)). This is a hard data
extent limit (out-of-bounds queries return NaN), not a gradual quality
falloff. Integrating over this project's usual full z range (+-750 pc)
made EVERY (x, y) column NaN for Leike2020, since every line of sight
would cross z values that simply do not exist in the file. So both maps
here are integrated over z in [-270, 270] pc ONLY -- a genuinely fair,
directly comparable PARTIAL-column Sigma_gas for both maps, not the
full-column Sigma_gas used elsewhere in this project. Do not compare
these Sigma_gas values directly against compute_sigma_gas.py's or
compute_sigma_gas_factor_comparison.py's full-column outputs.

Conversion factors (each map's own established baseline, not a
cross-methodology choice -- we are comparing MAPS here, not conversion
factors, so each uses its own standard literature coefficient):
    Edenhofer: n_H = 1652 * dE/ds   (O'Neill+24 baseline factor, this
               project's established "old" fiducial choice)
    Leike2020: n_H = 880 * s_x      (Zucker+21, verified directly against
               the map's own embedded HDF5 metadata and a numerical
               benchmark at the Perseus molecular cloud in a prior
               inspection -- s_x is natively e-folds/pc, applied with no
               extra rescaling)

NaN handling: neither map's missing/out-of-range data is filled or
interpolated. A NaN in either map's raw query output propagates through
that column's trapezoidal z-integration untouched (default numpy
behavior), so any (x, y) column with a gap shows up as NaN in the
resulting Sigma_gas map for that source, not an invented value.
"""

import time

import astropy.units as u
import numpy as np
import zarr
from astropy.coordinates import Galactic, SkyCoord
from dustmaps.edenhofer2023 import Edenhofer2023Query
from dustmaps.leike2020 import Leike2020Query

from src.config_loader import load_resolved_config

CACHE_PATH = "cache/sigma_gas_leike_vs_edenhofer.npz"
FOOTPRINT_PC = 400.0
Z_LIMIT_PC = 270.0  # Leike2020's hard z-extent limit; both maps integrated only within this

FACTOR_EDENHOFER = 1652.0  # O'Neill+24 baseline, n_H = 1652 * dE/ds
FACTOR_LEIKE = 880.0  # Zucker+21, n_H = 880 * s_x (s_x already e-folds/pc)

# Same physical constants as compute_sigma_gas.py (reused exactly, not redefined)
MU = 1.4
M_H = 1.6726219e-24
PC_TO_CM = 3.0856775814913673e18
MSUN_G = 1.98892e33


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

    x_idx = np.where(np.abs(x_pc) <= FOOTPRINT_PC)[0]
    y_idx = np.where(np.abs(y_pc) <= FOOTPRINT_PC)[0]
    x_lo, x_hi = x_idx.min(), x_idx.max() + 1
    y_lo, y_hi = y_idx.min(), y_idx.max() + 1
    x_sub = x_pc[x_lo:x_hi]
    y_sub = y_pc[y_lo:y_hi]

    # Leike2020's box is a half-open interval [-270, 270) pc in z (confirmed
    # empirically: z=+270.0 pc queries as NaN, out of bounds by exactly one
    # floor()-indexing step, while z=-270.0 pc is valid) -- so the upper
    # bound must be strictly excluded, not <=, or every column's trapezoidal
    # integral gets NaN-contaminated by this one shared boundary z-slice.
    z_idx = np.where((z_pc >= -Z_LIMIT_PC) & (z_pc < Z_LIMIT_PC))[0]
    z_lo, z_hi = z_idx.min(), z_idx.max() + 1
    z_sub = z_pc[z_lo:z_hi]

    print(f"x_pc range [{x_sub.min()}, {x_sub.max()}], {len(x_sub)} points")
    print(f"y_pc range [{y_sub.min()}, {y_sub.max()}], {len(y_sub)} points")
    print(f"z_pc range [{z_sub.min()}, {z_sub.max()}], {len(z_sub)} points "
          f"(restricted to Leike2020's z-extent, not the usual full +-750pc)")

    print("Loading Edenhofer2023Query (integrated=False)...")
    t0 = time.time()
    eden_query = Edenhofer2023Query(integrated=False)
    print(f"  loaded in {time.time() - t0:.1f}s")

    print("Loading Leike2020Query...")
    t0 = time.time()
    leike_query = Leike2020Query()
    print(f"  loaded in {time.time() - t0:.1f}s")

    X, Y = np.meshgrid(x_sub, y_sub)  # shape (y_sub, x_sub)
    n_z = len(z_sub)
    dEds = np.empty((n_z, len(y_sub), len(x_sub)), dtype=np.float64)
    s_x = np.empty((n_z, len(y_sub), len(x_sub)), dtype=np.float64)

    t_start = time.time()
    for iz, z_val in enumerate(z_sub):
        Z = np.full_like(X, z_val)
        gal = Galactic(u=X * u.pc, v=Y * u.pc, w=Z * u.pc, representation_type="cartesian")
        coords = SkyCoord(gal)
        coords.representation_type = "spherical"  # required by Edenhofer's ensure_flat_galactic

        dEds[iz] = eden_query.query(coords)
        s_x[iz] = leike_query.query(coords)

        elapsed = time.time() - t_start
        if elapsed > 30 and (iz % 50 == 0 or iz == n_z - 1):
            rate = (iz + 1) / elapsed
            eta = (n_z - iz - 1) / rate if rate > 0 else float("nan")
            print(f"  z-slice {iz + 1}/{n_z}, elapsed {elapsed:.0f}s, ETA {eta:.0f}s")

    print(f"Finished querying both maps in {time.time() - t_start:.1f}s")
    print(f"  Edenhofer NaN fraction: {np.mean(np.isnan(dEds)):.4%}")
    print(f"  Leike2020 NaN fraction: {np.mean(np.isnan(s_x)):.4%}")

    n_H_eden = FACTOR_EDENHOFER * dEds
    n_H_leike = FACTOR_LEIKE * s_x

    N_H_column_eden = np.trapezoid(n_H_eden, x=z_sub, axis=0)
    N_H_column_leike = np.trapezoid(n_H_leike, x=z_sub, axis=0)

    sigma_gas_eden = sigma_gas_from_n_H_column(N_H_column_eden)
    sigma_gas_leike = sigma_gas_from_n_H_column(N_H_column_leike)

    stats = {
        "mean_eden": float(np.nanmean(sigma_gas_eden)),
        "median_eden": float(np.nanmedian(sigma_gas_eden)),
        "nan_frac_eden": float(np.mean(np.isnan(sigma_gas_eden))),
        "mean_leike": float(np.nanmean(sigma_gas_leike)),
        "median_leike": float(np.nanmedian(sigma_gas_leike)),
        "nan_frac_leike": float(np.mean(np.isnan(sigma_gas_leike))),
    }
    print("Summary statistics (Sigma_gas, Msun/pc^2):")
    for k, v in stats.items():
        print(f"  {k}: {v:.6g}")

    np.savez(
        CACHE_PATH,
        sigma_gas_eden=sigma_gas_eden,
        sigma_gas_leike=sigma_gas_leike,
        x_pc=x_sub,
        y_pc=y_sub,
        **stats,
    )
    print(f"Saved {CACHE_PATH}")


if __name__ == "__main__":
    main()
