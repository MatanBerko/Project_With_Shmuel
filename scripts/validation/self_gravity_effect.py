"""
Step 1b: quantify the effect of gas self-gravity in BOTH modes
(footprint_mean = new default, per_column = sensitivity option) against
off, for all three variants. Reads only cache/core/alpha_core.zarr
(produced by pipeline/compute_all.py) -- no recomputation of any physics
here.

For each variant and each ON mode (mean, column), the per-cell ratio
alpha_on / alpha_off is computed (Ptot ratio is identical, since
alpha = Ptot/Pth and Pth doesn't depend on self-gravity), then reduced to
four statistics -- volume-weighted median, volume-weighted mean,
mass-weighted median, mass-weighted mean -- at three 50pc-thick z-slabs
(0, 150, 300 pc, matching pipeline/compute_all.py's SLAB_HALF_THICKNESS_PC)
and over the full |z| <= 500 pc range. HIM-flagged cells are excluded
throughout (same convention as pipeline/compute_all.py).
"""

from pathlib import Path

import numpy as np
import zarr

VARIANTS = ("RAW", "HIM_A", "HIM_B")
ON_MODES = ("mean", "column")  # vs "off"
ALPHA_CORE_ZARR_PATH = Path("cache/core/alpha_core.zarr")
OUT_NPZ_PATH = Path("cache/core/self_gravity_effect.npz")
OUT_TXT_PATH = Path("results/self_gravity_effect.txt")

SLAB_CENTERS_PC = (0.0, 150.0, 300.0)
SLAB_HALF_THICKNESS_PC = 25.0
MAX_ABS_Z_PC = 500.0

PHASE_HIM = 3


def _weighted_median(v, w):
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not m.any():
        return float("nan")
    v, w = v[m], w[m]
    idx = np.argsort(v)
    v, w = v[idx], w[idx]
    cw = np.cumsum(w)
    i = np.searchsorted(cw, 0.5 * cw[-1], side="left")
    i = min(max(int(i), 0), v.size - 1)
    return float(v[i])


def _weighted_mean(v, w):
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not m.any():
        return float("nan")
    return float(np.average(v[m], weights=w[m]))


def _four_stats(ratio_sub, neutral_sub, n_model_sub):
    ratio_flat = ratio_sub.ravel()
    neutral_flat = neutral_sub.ravel()
    n_flat = n_model_sub.ravel().astype(np.float64)

    w_vol = np.where(neutral_flat, 1.0, 0.0)
    w_mw = np.where(neutral_flat & np.isfinite(n_flat), n_flat, 0.0)

    return {
        "vol_median": _weighted_median(ratio_flat, w_vol),
        "vol_mean": _weighted_mean(ratio_flat, w_vol),
        "mw_median": _weighted_median(ratio_flat, w_mw),
        "mw_mean": _weighted_mean(ratio_flat, w_mw),
    }


def main():
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("cache/core").mkdir(parents=True, exist_ok=True)

    g = zarr.open_group(str(ALPHA_CORE_ZARR_PATH), mode="r")
    z_pc = np.asarray(g["z_pc"][:], dtype=np.float64)

    npz_out = {"z_pc": z_pc}
    lines = ["Self-gravity effect: alpha ratio (self-gravity ON / OFF), both modes",
             "ON modes: mean = footprint_mean (default), column = per_column (sensitivity option)",
             "Ratios computed per-cell (alpha_on/alpha_off), then reduced with 4 statistics.",
             "=" * 100]

    for variant in VARIANTS:
        vg = g[variant]
        phase_flag = np.asarray(vg["phase_flag"][:])
        neutral = phase_flag != PHASE_HIM
        n_model = np.asarray(vg["n_model"][:], dtype=np.float32)

        alpha_off = np.asarray(vg["self_gravity_off"]["alpha"][:], dtype=np.float64)

        lines.append(f"\nVariant {variant}:")
        for mode in ON_MODES:
            alpha_on = np.asarray(vg[f"self_gravity_{mode}"]["alpha"][:], dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(alpha_off > 0, alpha_on / alpha_off, np.nan)

            lines.append(f"  mode={mode}:")
            lines.append(f"    {'range':>14}{'vol_median':>14}{'vol_mean':>14}{'mw_median':>14}{'mw_mean':>14}")

            for zc in SLAB_CENTERS_PC:
                lo, hi = zc - SLAB_HALF_THICKNESS_PC, zc + SLAB_HALF_THICKNESS_PC
                idxs = np.where((z_pc >= lo) & (z_pc <= hi))[0]
                stats = _four_stats(ratio[idxs], neutral[idxs], n_model[idxs])
                label = f"z={zc:.0f}pc"
                lines.append(f"    {label:>14}{stats['vol_median']:>14.4f}{stats['vol_mean']:>14.4f}"
                             f"{stats['mw_median']:>14.4f}{stats['mw_mean']:>14.4f}")
                for k, val in stats.items():
                    npz_out[f"{variant}__{mode}__z{int(zc)}__{k}"] = val

            idxs_full = np.where(np.abs(z_pc) <= MAX_ABS_Z_PC)[0]
            stats_full = _four_stats(ratio[idxs_full], neutral[idxs_full], n_model[idxs_full])
            lines.append(f"    {'|z|<=500pc':>14}{stats_full['vol_median']:>14.4f}{stats_full['vol_mean']:>14.4f}"
                         f"{stats_full['mw_median']:>14.4f}{stats_full['mw_mean']:>14.4f}")
            for k, val in stats_full.items():
                npz_out[f"{variant}__{mode}__absz500__{k}"] = val

        del phase_flag, neutral, n_model, alpha_off

    report = "\n".join(lines)
    print(report)

    np.savez(OUT_NPZ_PATH, **npz_out)
    OUT_TXT_PATH.write_text(report + "\n")
    print(f"\nSaved {OUT_NPZ_PATH}")
    print(f"Saved {OUT_TXT_PATH}")


if __name__ == "__main__":
    main()
