"""
Step 1i guard: MIDPLANE_MODE = "sun" must reproduce the pre-Step-1i pipeline
EXACTLY.

Run once, BEFORE the balanced rebuild, while cache/core/alpha_core.zarr still
holds the Sun-centred build:
  1. every variant x self-gravity setting: P_tot and alpha recomputed in
     memory by the NEW code in "sun" mode (pipeline.compute_all.
     gravity_for_setting + _ptot_tiled_arrays) must equal the cached cubes
     bit for bit;
  2. RAW's finalize in "sun" mode must reproduce, row for row and value for
     value, the numbers the OLD code wrote on this machine
     (cache/sun_reference/_finalize_partial/RAW_numbers.pkl), and every
     summary array it had (RAW_summary.npz), bit for bit.
Writes results/midplane_sun_reproduction.txt. Nothing is written to the cache.
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pipeline.compute_all as pipe  # noqa: E402
from src.conventions import MIDPLANE_MODE_SUN, PROVISIONAL_CUBE_HEADER  # noqa: E402
from src.physics.loading import footprint_mask, open_zarr  # noqa: E402

SUN_REF_DIR = Path("cache/sun_reference/_finalize_partial")
OUT_TXT_PATH = Path("results/midplane_sun_reproduction.txt")


def same_value(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return (a == b) or (np.isnan(a) and np.isnan(b))
    return a == b


def main():
    t0 = time.time()
    core = zarr.open_group(str(pipe.ALPHA_CORE_ZARR_PATH), mode="r")
    z_src = np.asarray(open_zarr().z_pc, dtype=np.float64)       # what build_stage uses
    L = [PROVISIONAL_CUBE_HEADER,
         "Step 1i guard: MIDPLANE_MODE = 'sun' reproduces the pre-Step-1i pipeline",
         "=" * 90, "", "1. P_tot / alpha cubes, recomputed in memory in 'sun' mode vs the cached build:"]
    for variant in pipe.VARIANTS:
        vg = core[variant]
        for sg in pipe.SELF_GRAVITY_SETTINGS:
            sgg = vg[f"self_gravity_{sg}"]
            if pipe._group_midplane_mode(sgg) != MIDPLANE_MODE_SUN:
                raise RuntimeError("cache is not the Sun-centred build -- run this BEFORE rebuilding")
            z_mid, g, _ = pipe.gravity_for_setting(core, vg, z_src, sg, MIDPLANE_MODE_SUN)
            P, a = pipe._ptot_tiled_arrays(vg, z_src, g, z_split=z_mid)
            same_p = np.array_equal(P, np.asarray(sgg["Ptot"][:]), equal_nan=True)
            same_a = np.array_equal(a, np.asarray(sgg["alpha"][:]), equal_nan=True)
            line = f"   {variant:<6} sg {sg:<5} z_mid {z_mid:+.1f}  Ptot bitwise {same_p}  alpha bitwise {same_a}"
            print(line)
            L.append(line)
            if not (same_p and same_a):
                raise RuntimeError("sun mode does NOT reproduce the cached cubes -- STOPPING")
            del P, a, g

    print("\n2. RAW finalize in 'sun' mode vs the old code's partial...")
    x_sub = np.asarray(core["x_pc"][:], dtype=np.float64)
    y_sub = np.asarray(core["y_pc"][:], dtype=np.float64)
    z_full = np.asarray(core["z_pc"][:], dtype=np.float64)
    footprint = footprint_mask(x_sub, y_sub, pipe.FOOTPRINT_HALF_RANGE_PC)
    import dask.array as da
    zsl, _ = pipe._box_z_slice(z_full)
    grid = open_zarr()
    x_idx = np.where(np.abs(grid.x_pc) <= pipe.FOOTPRINT_HALF_RANGE_PC)[0]
    y_idx = np.where(np.abs(grid.y_pc) <= pipe.FOOTPRINT_HALF_RANGE_PC)[0]
    iuv_box = da.from_zarr(grid.store["Iuv_final"])[
        zsl, y_idx.min():y_idx.max() + 1, x_idx.min():x_idx.max() + 1].compute().astype(np.float32)
    summ, rows = pipe._finalize_one_variant("RAW", core, footprint, z_full, iuv_box,
                                            midplane_mode=MIDPLANE_MODE_SUN)
    with open(SUN_REF_DIR / "RAW_numbers.pkl", "rb") as f:
        ref_rows = pickle.load(f)
    if len(rows) != len(ref_rows):
        raise RuntimeError(f"row count {len(rows)} != {len(ref_rows)}")
    n_bad = sum(1 for r, q in zip(rows, ref_rows)
                if len(r) != len(q) or not all(same_value(x, y) for x, y in zip(r, q)))
    ref_summ = np.load(SUN_REF_DIR / "RAW_summary.npz", allow_pickle=False)
    bad_keys = [k for k in ref_summ.files
                if k not in summ or not np.array_equal(np.asarray(summ[k]), ref_summ[k],
                                                       equal_nan=ref_summ[k].dtype.kind == "f")]
    extra = sorted(set(summ) - set(ref_summ.files))
    L += ["", "2. RAW finalize in 'sun' mode vs the old code's partial on this machine:",
          f"   numbers_table rows: {len(rows)} vs {len(ref_rows)}; rows differing in any field "
          f"(text or value, exact): {n_bad}",
          f"   summary arrays: {len(ref_summ.files)} in the reference; differing (bitwise): {len(bad_keys)}",
          f"   new keys (Step 1i bookkeeping only): {extra}", "",
          f"(runtime {time.time() - t0:.1f}s)"]
    OUT_TXT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    if n_bad or bad_keys:
        raise RuntimeError("sun-mode finalize does NOT reproduce the old numbers -- STOPPING")


if __name__ == "__main__":
    main()
