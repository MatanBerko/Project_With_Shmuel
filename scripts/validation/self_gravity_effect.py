"""
Quantify the effect of the NEW gas self-gravity term: P_tot and alpha
ratios (self-gravity ON / OFF) vs |z|, for all three variants. Reads only
cache/core/alpha_core.zarr (produced by pipeline/compute_all.py) -- no
recomputation of any physics here.
"""

from pathlib import Path

import numpy as np
import zarr

VARIANTS = ("RAW", "HIM_A", "HIM_B")
ALPHA_CORE_ZARR_PATH = Path("cache/core/alpha_core.zarr")
OUT_NPZ_PATH = Path("cache/core/self_gravity_effect.npz")
OUT_TXT_PATH = Path("results/self_gravity_effect.txt")

BIN_WIDTH_PC = 10.0
MAX_ABS_Z_PC = 500.0
REPORT_Z_VALUES_PC = (0.0, 100.0, 300.0)


def main():
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("cache/core").mkdir(parents=True, exist_ok=True)

    g = zarr.open_group(str(ALPHA_CORE_ZARR_PATH), mode="r")
    x_pc = np.asarray(g["x_pc"][:])
    y_pc = np.asarray(g["y_pc"][:])
    z_pc = np.asarray(g["z_pc"][:])
    X, Y = np.meshgrid(x_pc, y_pc)
    footprint = (np.abs(X) <= 500.0) & (np.abs(Y) <= 500.0)

    edges = np.arange(0.0, MAX_ABS_Z_PC + BIN_WIDTH_PC / 2, BIN_WIDTH_PC)
    n_bins = len(edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    abs_z = np.abs(z_pc)

    npz_out = {"z_bin_centers": centers}
    lines = ["Self-gravity effect: P_tot and alpha ratios (ON/OFF) vs |z|",
             "=" * 90]

    for variant in VARIANTS:
        vg = g[variant]
        phase_flag = np.asarray(vg["phase_flag"][:])
        neutral = phase_flag != 3  # PHASE_HIM = 3

        Ptot_off = np.asarray(vg["self_gravity_off"]["Ptot"][:], dtype=np.float64)
        Ptot_on = np.asarray(vg["self_gravity_on"]["Ptot"][:], dtype=np.float64)
        alpha_off = np.asarray(vg["self_gravity_off"]["alpha"][:], dtype=np.float64)
        alpha_on = np.asarray(vg["self_gravity_on"]["alpha"][:], dtype=np.float64)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_ptot = np.where(Ptot_off > 0, Ptot_on / Ptot_off, np.nan)
            ratio_alpha = np.where(alpha_off > 0, alpha_on / alpha_off, np.nan)

        med_ptot = np.full(n_bins, np.nan)
        med_alpha = np.full(n_bins, np.nan)
        mean_ptot = np.full(n_bins, np.nan)
        mean_alpha = np.full(n_bins, np.nan)

        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            plane_mask = (abs_z >= lo) & (abs_z < hi if b < n_bins - 1 else abs_z <= hi)
            idxs = np.where(plane_mask)[0]
            if idxs.size == 0:
                continue
            m = neutral[idxs][:, footprint]
            rp = ratio_ptot[idxs][:, footprint][m]
            ra = ratio_alpha[idxs][:, footprint][m]
            rp = rp[np.isfinite(rp)]
            ra = ra[np.isfinite(ra)]
            if rp.size:
                med_ptot[b] = float(np.median(rp))
                mean_ptot[b] = float(np.mean(rp))
            if ra.size:
                med_alpha[b] = float(np.median(ra))
                mean_alpha[b] = float(np.mean(ra))

        npz_out[f"{variant}__ratio_Ptot_median"] = med_ptot
        npz_out[f"{variant}__ratio_Ptot_mean"] = mean_ptot
        npz_out[f"{variant}__ratio_alpha_median"] = med_alpha
        npz_out[f"{variant}__ratio_alpha_mean"] = mean_alpha

        lines.append(f"\nVariant {variant}:")
        lines.append(f"{'|z| (pc)':>10}{'P_tot ratio (median)':>24}{'alpha ratio (median)':>24}")
        for zv in REPORT_Z_VALUES_PC:
            b = min(int(zv // BIN_WIDTH_PC), n_bins - 1)
            lines.append(f"{centers[b]:>10.1f}{med_ptot[b]:>24.6f}{med_alpha[b]:>24.6f}")

    report = "\n".join(lines)
    print(report)

    np.savez(OUT_NPZ_PATH, **npz_out)
    OUT_TXT_PATH.write_text(report + "\n")
    print(f"\nSaved {OUT_NPZ_PATH}")
    print(f"Saved {OUT_TXT_PATH}")


if __name__ == "__main__":
    main()
