"""
Compute script for the gas surface density Sigma_gas(x, y).

ASSUMPTION FLAGGED FOR REVIEW: The "density" array in f98_sm_opt2_edendist.zarr
is treated here as already being hydrogen number density n_H in cm^-3 (NOT a
raw dust-extinction quantity requiring a further conversion factor). This is
inferred from prior inspection (a related array's 'density_kind': 'number'
metadata, and a value range matching realistic ISM densities) but is not
100% confirmed by an explicit units attribute on "density" itself. If this
assumption turns out to be wrong, this entire script's output is invalid and
needs to be redone with the correct conversion applied.

Method: for each (y, x) grid point within a +-500 pc square footprint
(|x| <= 500 pc and |y| <= 500 pc), integrate n_H over the full z_pc range
(-750 to 750 pc) via trapezoidal integration against the real z_pc
coordinate array, then convert the resulting column number density to a
mass surface density in Msun/pc^2.
"""

import dask.array as da
import numpy as np
import zarr

from src.config_loader import load_resolved_config

CACHE_PATH = "cache/sigma_gas_edenhofer.npz"

# Physical constants / conversion factors (project convention, per Eq. derivation)
MU = 1.4  # mean molecular weight per H, project convention
M_H = 1.6726219e-24  # grams (hydrogen mass)
PC_TO_CM = 3.0856775814913673e18  # cm per pc
MSUN_G = 1.98892e33  # grams per solar mass


def main():
    cfg = load_resolved_config()
    store = zarr.open(str(cfg["zarr_path"]), mode="r")

    x_pc = store["x_pc"][:]
    y_pc = store["y_pc"][:]
    z_pc = store["z_pc"][:]

    # Index range for the +-500 pc square footprint in x and y, found via the
    # real coordinate arrays before touching the (751, 1001, 1001) density array.
    x_mask = np.abs(x_pc) <= 500
    y_mask = np.abs(y_pc) <= 500
    x_idx = np.where(x_mask)[0]
    y_idx = np.where(y_mask)[0]
    x_lo, x_hi = x_idx.min(), x_idx.max() + 1
    y_lo, y_hi = y_idx.min(), y_idx.max() + 1

    x_sub = x_pc[x_lo:x_hi]
    y_sub = y_pc[y_lo:y_hi]

    print(f"x index range: [{x_lo}, {x_hi}) -> {len(x_sub)} points, "
          f"x_pc range [{x_sub.min()}, {x_sub.max()}]")
    print(f"y index range: [{y_lo}, {y_hi}) -> {len(y_sub)} points, "
          f"y_pc range [{y_sub.min()}, {y_sub.max()}]")
    print(f"z: full range, {len(z_pc)} points, z_pc range [{z_pc.min()}, {z_pc.max()}]")

    density = da.from_zarr(store["density"])
    # axis order is (z, y, x) per _ARRAY_DIMENSIONS metadata
    sub = density[:, y_lo:y_hi, x_lo:x_hi]
    print(f"selected sub-volume shape: {sub.shape}, "
          f"~{sub.nbytes / 1e6:.1f} MB (float32)")

    sub = sub.compute()  # (z, y_sub, x_sub)

    # Column number density N_H [cm^-3 * pc]: integrate n_H over full z range
    # against the actual z_pc coordinate values (trapezoidal rule), axis=0 is z.
    N_H_column_pc_cm3 = np.trapezoid(sub, x=z_pc, axis=0)  # shape (y_sub, x_sub)

    # Unit chain: cm^-3*pc -> cm^-2 -> g/cm^2 -> Msun/pc^2
    N_H_cm2 = N_H_column_pc_cm3 * PC_TO_CM  # cm^-3*pc -> cm^-2 (column number density)
    Sigma_g_cm2 = MU * M_H * N_H_cm2  # g/cm^2 (mass surface density, cgs)
    Sigma_gas_Msun_pc2 = Sigma_g_cm2 * (PC_TO_CM ** 2) / MSUN_G  # -> Msun/pc^2

    stats = {
        "mean": float(np.mean(Sigma_gas_Msun_pc2)),
        "median": float(np.median(Sigma_gas_Msun_pc2)),
        "min": float(np.min(Sigma_gas_Msun_pc2)),
        "max": float(np.max(Sigma_gas_Msun_pc2)),
    }
    print("Sigma_gas [Msun/pc^2] summary statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v:.6g}")

    np.savez(
        CACHE_PATH,
        sigma_gas=Sigma_gas_Msun_pc2,
        x_pc=x_sub,
        y_pc=y_sub,
        mean=stats["mean"],
        median=stats["median"],
        min=stats["min"],
        max=stats["max"],
    )
    print(f"Saved {CACHE_PATH}")


if __name__ == "__main__":
    main()
