"""
Compute ALL numbers the alpha paper needs, in one pass over the density
cube, using ONLY src.physics (no reimplemented physics here). No figures
are produced by this script.

Variants: RAW, HIM_A, HIM_B. MASKED is NOT computed -- no MASKED
definition exists in any of the four reference scripts (see Phase A
report); this is a deliberate, reported omission.

Outputs:
  cache/core/alpha_core.zarr   -- 3D fields per variant x self-gravity
                                   setting (float32, chunked along z)
  cache/core/summary.npz       -- vertical profiles, 1D/2D PDFs, midplane
                                   slices, Sigma_gas maps
  results/numbers_table.csv/.txt -- one row per quantity x variant x
                                   self-gravity setting
"""

import ctypes
import gc
import time
from pathlib import Path

import numpy as np
import zarr

from src.config_loader import load_resolved_config
from src.conventions import CNM_TEMP_MAX_K, K_B, M_H, MU, WNM_TEMP_MIN_K, XY_HALF_RANGE_PC, \
    SELF_GRAVITY_MODE_FOOTPRINT_MEAN, SELF_GRAVITY_MODE_OFF, SELF_GRAVITY_MODE_PER_COLUMN
from src.physics import derived, gravity, hydrostatic, thermal
from src.physics.him import PHASE_CNM, PHASE_HIM, PHASE_UNM, PHASE_WNM, apply_variant, him_flag, phase_flag
from src.physics.loading import footprint_mask, load_xy_subset_coords, open_zarr
from src.physics.stats import mass_weighted_stats, volume_weighted_stats

# ============================================================================
# Control block
# ============================================================================
VARIANTS = ("RAW", "HIM_A", "HIM_B")  # MASKED omitted -- no definition exists (see report)

# Step 1b: self-gravity settings are now off | mean | column (was off/on,
# i.e. False/True -- "on" always meant per-column). "mean"
# (footprint_mean) is the new default per-paper convention; "column"
# (per_column) is kept as an explicit sensitivity option.
SELF_GRAVITY_SETTINGS = ("off", "mean", "column")
SELF_GRAVITY_MODE_BY_SETTING = {
    "off": SELF_GRAVITY_MODE_OFF,
    "mean": SELF_GRAVITY_MODE_FOOTPRINT_MEAN,
    "column": SELF_GRAVITY_MODE_PER_COLUMN,
}
# A Step-1 run may have cached self_gravity_off / self_gravity_on groups
# (on == per_column). Reuse them under the new names instead of
# recomputing -- the underlying physics is byte-for-byte identical.
LEGACY_GROUP_NAME_BY_SETTING = {"off": "self_gravity_off", "column": "self_gravity_on"}

SELF_GRAVITY_DENSITY = "same_as_weight"  # "same_as_weight" (default) | "observed"

FOOTPRINT_HALF_RANGE_PC = XY_HALF_RANGE_PC  # +-500 pc square

VERTICAL_BIN_WIDTH_PC = 10.0  # signed-z and |z| profile bins
VERTICAL_PROFILE_MAX_ABS_Z_PC = 500.0

SLAB_CENTERS_PC = (0.0, 150.0, 300.0)
SLAB_HALF_THICKNESS_PC = 25.0  # 50 pc thick slabs (see report: NOT what the
# 4 reference scripts did -- they all used a single nearest z-plane via
# nearest_idx(), e.g. fig4_vertical_profiles/compute_data.py:341. No 50pc
# slab definition exists in any of them; this is a NEW, settled choice.

PDF_N_BINS = 60  # log-binned 1D PDFs
PDF_2D_N_BINS = 50  # 2D mass-weighted PDFs

CACHE_DIR = Path("cache/core")
RESULTS_DIR = Path("results")
ALPHA_CORE_ZARR_PATH = CACHE_DIR / "alpha_core.zarr"
SUMMARY_NPZ_PATH = CACHE_DIR / "summary.npz"
SUMMARY_NPZ_SIZE_LIMIT_MB = 50
NUMBERS_TABLE_CSV_PATH = RESULTS_DIR / "numbers_table.csv"
NUMBERS_TABLE_TXT_PATH = RESULTS_DIR / "numbers_table.txt"

FLOAT_DTYPE = np.float32
CHUNK_Z = 32  # zarr chunking along z


def peak_working_set_mb():
    """Windows-only peak working set, best-effort (returns None elsewhere)."""
    try:
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        counters = PROCESS_MEMORY_COUNTERS()
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), ctypes.sizeof(counters))
        if not ok:
            return None
        return counters.PeakWorkingSetSize / (1024 ** 2)
    except Exception:
        return None


PHASE_NAMES = {PHASE_CNM: "CNM", PHASE_UNM: "UNM", PHASE_WNM: "WNM", PHASE_HIM: "HIM"}
PHASE_CODES_NEUTRAL = ((PHASE_CNM, "CNM"), (PHASE_UNM, "UNM"), (PHASE_WNM, "WNM"))
PHASE_CODES_ALL = PHASE_CODES_NEUTRAL + ((PHASE_HIM, "HIM"),)


# ============================================================================
# Vertical profiles (10 pc bins, |z| <= 500), signed and folded
# ============================================================================
def _bin_edges(use_abs: bool) -> np.ndarray:
    if use_abs:
        return np.arange(0.0, VERTICAL_PROFILE_MAX_ABS_Z_PC + VERTICAL_BIN_WIDTH_PC / 2, VERTICAL_BIN_WIDTH_PC)
    return np.arange(-VERTICAL_PROFILE_MAX_ABS_Z_PC, VERTICAL_PROFILE_MAX_ABS_Z_PC + VERTICAL_BIN_WIDTH_PC / 2,
                      VERTICAL_BIN_WIDTH_PC)


def compute_vertical_profile(z_pc, Pth, Ptot, alpha_arr, him_cube, phase_cube, T_cube,
                               n_model, footprint, use_abs: bool):
    edges = _bin_edges(use_abs)
    n_bins = len(edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    z_for_binning = np.abs(z_pc) if use_abs else z_pc

    out = {}
    for qty in ("Pth", "Ptot", "alpha"):
        for wt in ("vol", "mw"):
            for s in ("median", "mean", "p16", "p84"):
                out[f"{qty}_{wt}_{s}"] = np.full(n_bins, np.nan)
    for _, ph in PHASE_CODES_ALL:
        out[f"f_{ph}_vol"] = np.full(n_bins, np.nan)
        out[f"f_{ph}_mw"] = np.full(n_bins, np.nan)
    for ph in ("CNM", "UNM", "WNM", "total"):
        out[f"sigma_eff_{ph}"] = np.full(n_bins, np.nan)
    out["mach"] = np.full(n_bins, np.nan)

    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b < n_bins - 1:
            plane_mask = (z_for_binning >= lo) & (z_for_binning < hi)
        else:
            plane_mask = (z_for_binning >= lo) & (z_for_binning <= hi)
        idxs = np.where(plane_mask)[0]
        if idxs.size == 0:
            continue

        him_sub = him_cube[idxs][:, footprint]
        phase_sub = phase_cube[idxs][:, footprint]
        n_sub = n_model[idxs][:, footprint].astype(np.float64)
        T_sub = T_cube[idxs][:, footprint].astype(np.float64)
        Pth_sub = Pth[idxs][:, footprint].astype(np.float64)
        Ptot_sub = Ptot[idxs][:, footprint].astype(np.float64)
        alpha_sub = alpha_arr[idxs][:, footprint].astype(np.float64)
        neutral = ~him_sub

        for qty, arr_sub in (("Pth", Pth_sub), ("Ptot", Ptot_sub), ("alpha", alpha_sub)):
            s_vol = volume_weighted_stats(arr_sub, neutral)
            s_mw = mass_weighted_stats(arr_sub, neutral, n_sub)
            out[f"{qty}_vol_median"][b] = s_vol.median
            out[f"{qty}_vol_mean"][b] = s_vol.arithmetic_mean
            out[f"{qty}_vol_p16"][b] = s_vol.p16
            out[f"{qty}_vol_p84"][b] = s_vol.p84
            out[f"{qty}_mw_median"][b] = s_mw.median
            out[f"{qty}_mw_mean"][b] = s_mw.arithmetic_mean
            out[f"{qty}_mw_p16"][b] = s_mw.p16
            out[f"{qty}_mw_p84"][b] = s_mw.p84

        total_cells = phase_sub.size
        n_weight_sub = np.where(np.isfinite(n_sub), n_sub, 0.0)
        W = n_weight_sub.sum()
        for code, ph in PHASE_CODES_ALL:
            m = phase_sub == code
            out[f"f_{ph}_vol"][b] = m.sum() / total_cells if total_cells > 0 else np.nan
            out[f"f_{ph}_mw"][b] = n_weight_sub[m].sum() / W if W > 0 else np.nan

        alpha_bin_mean = out["alpha_vol_mean"][b]
        out["mach"][b] = float(derived.mach_number(np.array([alpha_bin_mean]))[0])
        for code, ph in PHASE_CODES_NEUTRAL:
            m = neutral & (phase_sub == code)
            T_ph_mean = float(T_sub[m].mean()) if m.any() else np.nan
            out[f"sigma_eff_{ph}"][b] = float(
                derived.sigma_eff_kmps(np.array([alpha_bin_mean]), np.array([T_ph_mean]))[0])
        T_tot_mean = float(T_sub[neutral].mean()) if neutral.any() else np.nan
        out["sigma_eff_total"][b] = float(
            derived.sigma_eff_kmps(np.array([alpha_bin_mean]), np.array([T_tot_mean]))[0])

    out["bin_centers"] = centers
    out["bin_edges"] = edges
    return out


# ============================================================================
# 1D / 2D PDFs at slabs (z0 +- SLAB_HALF_THICKNESS_PC)
# ============================================================================
def slab_plane_indices(z_pc, z_center):
    lo, hi = z_center - SLAB_HALF_THICKNESS_PC, z_center + SLAB_HALF_THICKNESS_PC
    return np.where((z_pc >= lo) & (z_pc <= hi))[0]


def log_pdf_1d(values, weights, n_bins=PDF_N_BINS):
    v = values[np.isfinite(values) & (values > 0)]
    w = weights[np.isfinite(values) & (values > 0)] if weights is not None else None
    if v.size == 0:
        return np.array([]), np.array([])
    bins = np.logspace(np.log10(v.min()), np.log10(v.max()), n_bins)
    counts, edges = np.histogram(v, bins=bins, weights=w)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return centers, counts


def log_pdf_2d(x, y, w, n_bins=PDF_2D_N_BINS):
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & np.isfinite(w) & (w > 0)
    x, y, w = x[m], y[m], w[m]
    if x.size == 0:
        return np.array([]), np.array([]), np.zeros((0, 0))
    xbins = np.logspace(np.log10(x.min()), np.log10(x.max()), n_bins)
    ybins = np.logspace(np.log10(y.min()), np.log10(y.max()), n_bins)
    H, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ybins], weights=w)
    return xedges, yedges, H


# ============================================================================
# Stage 1: build -- writes ONLY cache/core/alpha_core.zarr. Resumable and
# split-able by variant, since this is the expensive part (BS19
# interpolation + hydrostatic integration over the full cube) and the one
# that needs to survive being interrupted (e.g. a background-task wall
# clock limit) across multiple invocations.
# ============================================================================
def build_stage(variants):
    t_start = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Opening zarr, loading full-column XY-footprint sub-cubes (density, T, Iuv_final)...")
    grid = open_zarr()
    x_sub, y_sub = load_xy_subset_coords(grid, FOOTPRINT_HALF_RANGE_PC)
    z_pc = grid.z_pc

    import dask.array as da
    x_lo, x_hi = int(np.where(np.abs(grid.x_pc) <= FOOTPRINT_HALF_RANGE_PC)[0].min()), \
        int(np.where(np.abs(grid.x_pc) <= FOOTPRINT_HALF_RANGE_PC)[0].max()) + 1
    y_lo, y_hi = int(np.where(np.abs(grid.y_pc) <= FOOTPRINT_HALF_RANGE_PC)[0].min()), \
        int(np.where(np.abs(grid.y_pc) <= FOOTPRINT_HALF_RANGE_PC)[0].max()) + 1
    n_raw = da.from_zarr(grid.store["density"])[:, y_lo:y_hi, x_lo:x_hi].compute().astype(np.float32)
    T = da.from_zarr(grid.store["T"])[:, y_lo:y_hi, x_lo:x_hi].compute().astype(np.float32)
    Iuv = da.from_zarr(grid.store["Iuv_final"])[:, y_lo:y_hi, x_lo:x_hi].compute().astype(np.float32)
    print(f"  sub-cube shape {n_raw.shape}, ~{n_raw.nbytes / 1e6:.1f} MB each (x3)")

    cfg = load_resolved_config()
    pminmax = thermal.build_pmin_pmax(cfg["bs19_mat_path"])
    print("Evaluating P_min(I_UV)/P_max(I_UV) over the full sub-cube...")
    Pmin = pminmax.p_min(Iuv).astype(np.float32)
    Pmax = pminmax.p_max(Iuv).astype(np.float32)
    del Iuv
    Pth_raw = (n_raw.astype(np.float64) * T.astype(np.float64)).astype(np.float32)
    him_cube = him_flag(Pth_raw, Pmin)
    phase_cube = phase_flag(T, him_cube)
    print(f"  HIM fraction (whole sub-cube): {him_cube.mean():.4%}")

    # mode="a": RESUMABLE -- a prior run may have already written some
    # variants (this is a long job; a killed/interrupted run should not
    # lose completed variants). Existing groups are loaded back rather
    # than recomputed.
    core_store = zarr.open_group(str(ALPHA_CORE_ZARR_PATH), mode="a")
    if "x_pc" not in core_store:
        core_store.attrs["variants"] = list(VARIANTS)
        core_store.attrs["self_gravity_settings"] = list(SELF_GRAVITY_SETTINGS)
        core_store.attrs["self_gravity_density"] = SELF_GRAVITY_DENSITY
        core_store.attrs["footprint_half_range_pc"] = FOOTPRINT_HALF_RANGE_PC
        core_store.attrs["masked_variant"] = "not implemented -- no MASKED definition in any reference script"
        core_store.create_array("x_pc", data=x_sub.astype(np.float32))
        core_store.create_array("y_pc", data=y_sub.astype(np.float32))
        core_store.create_array("z_pc", data=z_pc.astype(np.float32))

    for variant in variants:
        print(f"\n=== Variant {variant} ===")
        t_v = time.time()

        # Resumability granularity is the BASE arrays (n_model/Pth/T/
        # phase_flag/Sigma_gas), separate from each self_gravity setting
        # (checked individually below) -- a run killed partway through
        # self_gravity_on must not throw away an already-complete
        # self_gravity_off (or the base arrays).
        base_cached = (
            variant in core_store
            and "n_model" in core_store[variant]
            and "Pth" in core_store[variant]
            and "Sigma_gas" in core_store[variant]
        )
        if base_cached:
            print(f"  {variant} base arrays already in {ALPHA_CORE_ZARR_PATH} -- loading instead of recomputing")
            vg = core_store[variant]
            n_model = np.asarray(vg["n_model"][:], dtype=np.float32)
            Pth_model = np.asarray(vg["Pth"][:], dtype=np.float32)
            sigma_gas_variant = np.asarray(vg["Sigma_gas"][:], dtype=np.float32)
        else:
            if variant in core_store:
                print(f"  {variant} present but base arrays incomplete (killed mid-run?) -- discarding and redoing")
                del core_store[variant]
            vr = apply_variant(variant, n_raw.astype(np.float64), Pth_raw.astype(np.float64),
                                 him_cube, Pmin.astype(np.float64), Pmax.astype(np.float64))
            n_model = vr.n_model.astype(np.float32)
            Pth_model = vr.Pth_model.astype(np.float32)
            sigma_gas_variant = derived.sigma_gas_map(z_pc, n_model.astype(np.float64)).astype(np.float32)

            vg = core_store.require_group(variant)
            vg.create_array("n_model", data=n_model, chunks=(CHUNK_Z, n_model.shape[1], n_model.shape[2]))
            vg.create_array("Pth", data=Pth_model, chunks=(CHUNK_Z, Pth_model.shape[1], Pth_model.shape[2]))
            vg.create_array("T", data=T, chunks=(CHUNK_Z, T.shape[1], T.shape[2]))
            vg.create_array("phase_flag", data=phase_cube.astype(np.int8),
                              chunks=(CHUNK_Z, phase_cube.shape[1], phase_cube.shape[2]))
            vg.create_array("Sigma_gas", data=sigma_gas_variant)

        rho = (MU * M_H * n_model.astype(np.float64))
        # Already restricted to the +-500pc square footprint by the x_lo:x_hi/
        # y_lo:y_hi slice above, so "footprint_mean" here averages over the
        # entire loaded XY plane.
        full_footprint_mask = np.ones((n_model.shape[1], n_model.shape[2]), dtype=bool)

        for sg_key in SELF_GRAVITY_SETTINGS:
            mode = SELF_GRAVITY_MODE_BY_SETTING[sg_key]
            sg_group_name = f"self_gravity_{sg_key}"
            sg_cached = sg_group_name in vg and "Ptot" in vg[sg_group_name] and "alpha" in vg[sg_group_name]

            if sg_cached:
                print(f"  self_gravity={sg_key}: already complete in zarr -- skipping")
                continue

            if sg_group_name in vg:
                print(f"  self_gravity={sg_key}: partial group in zarr (killed mid-write?) -- discarding")
                del vg[sg_group_name]

            legacy_name = LEGACY_GROUP_NAME_BY_SETTING.get(sg_key)
            legacy_cached = (
                legacy_name is not None and legacy_name in vg
                and "Ptot" in vg[legacy_name] and "alpha" in vg[legacy_name]
            )
            if legacy_cached:
                print(f"  self_gravity={sg_key}: migrating cached legacy group '{legacy_name}' "
                      f"(Step 1 behavior is unchanged for off/column) -- no recompute")
                Ptot_kB = np.asarray(vg[legacy_name]["Ptot"][:], dtype=np.float32)
                alpha_arr = np.asarray(vg[legacy_name]["alpha"][:], dtype=np.float32)
                sgg = vg.require_group(sg_group_name)
                sgg.create_array("Ptot", data=Ptot_kB, chunks=(CHUNK_Z, Ptot_kB.shape[1], Ptot_kB.shape[2]))
                sgg.create_array("alpha", data=alpha_arr, chunks=(CHUNK_Z, alpha_arr.shape[1], alpha_arr.shape[2]))
                print(f"  self_gravity={sg_key}: written to zarr (migrated)")
                del Ptot_kB, alpha_arr
                gc.collect()
                continue

            print(f"  self_gravity={sg_key}: P_tot integration...")
            gravity_density = None
            if mode != SELF_GRAVITY_MODE_OFF:
                gravity_density = (n_model if SELF_GRAVITY_DENSITY == "same_as_weight" else n_raw).astype(np.float64)
            g, nan_fraction = gravity.g_total_cgs(z_pc, gravity_density, mode=mode, footprint_mask=full_footprint_mask)
            Ptot_kB = hydrostatic.p_tot_kb_full_column(z_pc, rho, g).astype(np.float32)
            alpha_arr = derived.alpha(Ptot_kB.astype(np.float64), Pth_model.astype(np.float64)).astype(np.float32)
            del g

            sgg = vg.require_group(sg_group_name)
            sgg.create_array("Ptot", data=Ptot_kB, chunks=(CHUNK_Z, Ptot_kB.shape[1], Ptot_kB.shape[2]))
            sgg.create_array("alpha", data=alpha_arr, chunks=(CHUNK_Z, alpha_arr.shape[1], alpha_arr.shape[2]))
            if nan_fraction is not None:
                sgg.create_array("footprint_nan_fraction", data=nan_fraction.astype(np.float32))
            print(f"  self_gravity={sg_key}: written to zarr")

            del Ptot_kB, alpha_arr
            gc.collect()
        del n_model, Pth_model, rho
        gc.collect()
        peak_mb = peak_working_set_mb()
        print(f"  variant {variant} done in {time.time() - t_v:.1f}s"
              + (f", working set {peak_mb:.0f} MB" if peak_mb is not None else ""))

    elapsed = time.time() - t_start
    peak_mb = peak_working_set_mb()
    print(f"\nbuild_stage runtime: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    if peak_mb is not None:
        print(f"Peak working set: {peak_mb:.0f} MB")
    print(f"Saved {ALPHA_CORE_ZARR_PATH}")


# ============================================================================
# Stage 2: finalize -- reads the COMPLETE cache/core/alpha_core.zarr (all
# variants) back, one variant at a time (cheap: no BS19/HIM/hydrostatic
# recomputation), and produces cache/core/summary.npz +
# results/numbers_table.{csv,txt}. Re-reads I_UV from the source zarr
# (not persisted in alpha_core.zarr) for the I_UV PDFs.
# ============================================================================
PARTIAL_DIR = CACHE_DIR / "_finalize_partial"


def _finalize_one_variant(variant, core_store, footprint, z_pc, Iuv):
    """All the per-variant summary/PDF/vertical-profile/numbers-table work,
    returning (summary_partial, numbers_rows_partial) for just this variant.
    """
    summary = {}
    numbers_rows = []
    print(f"\n=== Finalizing variant {variant} ===")
    t_v = time.time()
    vg = core_store[variant]
    n_model = np.asarray(vg["n_model"][:], dtype=np.float32)
    Pth_model = np.asarray(vg["Pth"][:], dtype=np.float32)
    T = np.asarray(vg["T"][:], dtype=np.float32)
    phase_cube = np.asarray(vg["phase_flag"][:], dtype=np.int8)
    him_cube = phase_cube == PHASE_HIM
    sigma_gas_variant = np.asarray(vg["Sigma_gas"][:], dtype=np.float32)

    summary[f"{variant}__Sigma_gas_map"] = sigma_gas_variant
    s_sigma = volume_weighted_stats(sigma_gas_variant, footprint)
    numbers_rows.append((variant, "n/a", "Sigma_gas", "vol", "median", s_sigma.median,
                           "Msun/pc^2", "Full-column trapezoid of variant density, +-500pc square footprint, unweighted median"))
    numbers_rows.append((variant, "n/a", "Sigma_gas", "vol", "mean", s_sigma.arithmetic_mean,
                           "Msun/pc^2", "Full-column trapezoid of variant density, +-500pc square footprint, arithmetic mean"))

    # 1D PDFs (self-gravity independent: n, T, Iuv, Pth)
    for qty, arr in (("n", n_model), ("T", T), ("Iuv", Iuv), ("Pth", Pth_model)):
        for zc in SLAB_CENTERS_PC:
            idxs = slab_plane_indices(z_pc, zc)
            sub = arr[idxs][:, footprint]
            neutral = ~him_cube[idxs][:, footprint]
            centers, counts = log_pdf_1d(sub[neutral].ravel(), None)
            summary[f"{variant}__pdf1d__{qty}__z{int(zc)}__centers"] = centers
            summary[f"{variant}__pdf1d__{qty}__z{int(zc)}__counts"] = counts

    # 2D mass-weighted PDFs (n,Pth) and (Iuv,n) -- self-gravity independent
    for (qx, xarr, qy, yarr) in (("n", n_model, "Pth", Pth_model), ("Iuv", Iuv, "n", n_model)):
        for zc in SLAB_CENTERS_PC:
            idxs = slab_plane_indices(z_pc, zc)
            neutral = ~him_cube[idxs][:, footprint]
            xv = xarr[idxs][:, footprint][neutral]
            yv = yarr[idxs][:, footprint][neutral]
            wv = n_model[idxs][:, footprint][neutral]
            xedges, yedges, H = log_pdf_2d(xv.astype(np.float64), yv.astype(np.float64), wv.astype(np.float64))
            summary[f"{variant}__pdf2d__{qx}_{qy}__z{int(zc)}__xedges"] = xedges
            summary[f"{variant}__pdf2d__{qx}_{qy}__z{int(zc)}__yedges"] = yedges
            summary[f"{variant}__pdf2d__{qx}_{qy}__z{int(zc)}__H"] = H

    for sg_key in SELF_GRAVITY_SETTINGS:
        sgg = vg[f"self_gravity_{sg_key}"]
        Ptot_kB = np.asarray(sgg["Ptot"][:], dtype=np.float32)
        alpha_arr = np.asarray(sgg["alpha"][:], dtype=np.float32)
        if "footprint_nan_fraction" in sgg:
            summary[f"{variant}__sg{sg_key}__footprint_nan_fraction"] = np.asarray(
                sgg["footprint_nan_fraction"][:], dtype=np.float32)

        for zc in SLAB_CENTERS_PC:
            iz = int(np.argmin(np.abs(z_pc - zc)))
            summary[f"{variant}__sg{sg_key}__slice__n__z{int(zc)}"] = n_model[iz]
            summary[f"{variant}__sg{sg_key}__slice__T__z{int(zc)}"] = T[iz]
            summary[f"{variant}__sg{sg_key}__slice__Pth__z{int(zc)}"] = Pth_model[iz]
            summary[f"{variant}__sg{sg_key}__slice__Ptot__z{int(zc)}"] = Ptot_kB[iz]
            summary[f"{variant}__sg{sg_key}__slice__alpha__z{int(zc)}"] = alpha_arr[iz]

        for qty, arr in (("Ptot", Ptot_kB), ("alpha", alpha_arr)):
            for zc in SLAB_CENTERS_PC:
                idxs = slab_plane_indices(z_pc, zc)
                sub = arr[idxs][:, footprint]
                neutral = ~him_cube[idxs][:, footprint]
                centers, counts = log_pdf_1d(sub[neutral].ravel(), None)
                summary[f"{variant}__sg{sg_key}__pdf1d__{qty}__z{int(zc)}__centers"] = centers
                summary[f"{variant}__sg{sg_key}__pdf1d__{qty}__z{int(zc)}__counts"] = counts

        print(f"  self_gravity={sg_key}: vertical profiles (signed z and |z|)...")
        prof_signed = compute_vertical_profile(z_pc, Pth_model, Ptot_kB, alpha_arr, him_cube, phase_cube,
                                                  T, n_model, footprint, use_abs=False)
        prof_abs = compute_vertical_profile(z_pc, Pth_model, Ptot_kB, alpha_arr, him_cube, phase_cube,
                                               T, n_model, footprint, use_abs=True)
        for k, v in prof_signed.items():
            summary[f"{variant}__sg{sg_key}__profile_signedz__{k}"] = v
        for k, v in prof_abs.items():
            summary[f"{variant}__sg{sg_key}__profile_absz__{k}"] = v

        for zc in SLAB_CENTERS_PC:
            idxs = slab_plane_indices(z_pc, zc)
            for qty, arr in (("Pth", Pth_model), ("Ptot", Ptot_kB), ("alpha", alpha_arr)):
                sub = arr[idxs][:, footprint].astype(np.float64)
                neutral = ~him_cube[idxs][:, footprint]
                n_sub = n_model[idxs][:, footprint].astype(np.float64)
                for wt, stats_fn in (("vol", lambda: volume_weighted_stats(sub, neutral)),
                                       ("mw", lambda: mass_weighted_stats(sub, neutral, n_sub))):
                    s = stats_fn()
                    for stat_name, val in (("median", s.median), ("mean", s.arithmetic_mean),
                                             ("p16", s.p16), ("p84", s.p84)):
                        numbers_rows.append((
                            variant, sg_key, qty, wt, f"z{int(zc)}_{stat_name}", val,
                            "K cm^-3" if qty in ("Pth", "Ptot") else "dimensionless",
                            f"{qty} {stat_name}, {wt}-weighted, {zc:.0f}+-{SLAB_HALF_THICKNESS_PC:.0f}pc slab, neutral cells, +-500pc square"
                        ))

        for qty in ("Pth", "Ptot", "alpha"):
            for wt in ("vol", "mw"):
                for stat_name in ("median", "mean", "p16", "p84"):
                    val = float(np.nanmean(prof_abs[f"{qty}_{wt}_{stat_name}"]))
                    numbers_rows.append((
                        variant, sg_key, qty, wt, f"absz500_{stat_name}", val,
                        "K cm^-3" if qty in ("Pth", "Ptot") else "dimensionless",
                        f"{qty} {stat_name}, {wt}-weighted, bin-average over |z|<=500pc profile, neutral cells"
                    ))

        footprint_full = np.broadcast_to(footprint, phase_cube.shape)
        n_weight = np.where(np.isfinite(n_model), n_model, 0.0).astype(np.float64)
        W = n_weight[footprint_full].sum()
        for code, ph in PHASE_CODES_ALL:
            m_all = (phase_cube == code) & footprint_full
            frac_vol = m_all.sum() / footprint_full.sum()
            frac_mw = n_weight[m_all].sum() / W if W > 0 else float("nan")
            numbers_rows.append((variant, sg_key, f"phase_fraction_{ph}", "vol", "full_volume", frac_vol,
                                   "dimensionless", f"Volume fraction of {ph}, full +-500pc square x +-750pc column"))
            numbers_rows.append((variant, sg_key, f"phase_fraction_{ph}", "mw", "full_volume", frac_mw,
                                   "dimensionless", f"Mass fraction of {ph}, full +-500pc square x +-750pc column"))

        iz_mid = int(np.argmin(np.abs(z_pc)))
        alpha_mid_vol = volume_weighted_stats(alpha_arr[iz_mid], footprint & ~him_cube[iz_mid])
        mach_mid = float(derived.mach_number(np.array([alpha_mid_vol.arithmetic_mean]))[0])
        T_neutral_mid = T[iz_mid][footprint & ~him_cube[iz_mid]].astype(np.float64)
        sigma_eff_mid = float(derived.sigma_eff_kmps(
            np.array([alpha_mid_vol.arithmetic_mean]),
            np.array([T_neutral_mid.mean() if T_neutral_mid.size else np.nan]))[0])
        numbers_rows.append((variant, sg_key, "Mach", "vol", "midplane", mach_mid,
                               "dimensionless", "Mach number at z=0, from volume-weighted mean alpha, neutral cells"))
        numbers_rows.append((variant, sg_key, "sigma_eff", "vol", "midplane", sigma_eff_mid,
                               "km/s", "Effective turbulent velocity dispersion at z=0, neutral cells, total (all phases)"))

        del Ptot_kB, alpha_arr
    del n_model, Pth_model, T, phase_cube, him_cube
    gc.collect()
    print(f"  variant {variant} finalized in {time.time() - t_v:.1f}s")
    return summary, numbers_rows


# ============================================================================
# Stage 2 driver: resumable per-variant finalize + merge. Each variant's
# partial result is pickled to cache/core/_finalize_partial/ (gitignored,
# not a deliverable) so a run that hits a wall-clock limit partway through
# doesn't lose already-finalized variants. Once all of VARIANTS has a
# partial, results are merged into summary.npz + numbers_table.{csv,txt}.
# ============================================================================
def finalize_stage(variants):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)

    core_store = zarr.open_group(str(ALPHA_CORE_ZARR_PATH), mode="r")
    for variant in variants:
        missing = variant not in core_store or any(
            f"self_gravity_{sg_key}" not in core_store[variant] for sg_key in SELF_GRAVITY_SETTINGS
        )
        if missing:
            raise RuntimeError(
                f"{ALPHA_CORE_ZARR_PATH} is missing a complete '{variant}' group -- "
                f"run `python pipeline/compute_all.py build {variant}` first."
            )

    x_sub = np.asarray(core_store["x_pc"][:], dtype=np.float64)
    y_sub = np.asarray(core_store["y_pc"][:], dtype=np.float64)
    z_pc = np.asarray(core_store["z_pc"][:], dtype=np.float64)
    footprint = footprint_mask(x_sub, y_sub, FOOTPRINT_HALF_RANGE_PC)

    variants_needing_work = [
        v for v in variants
        if not ((PARTIAL_DIR / f"{v}_summary.npz").exists() and (PARTIAL_DIR / f"{v}_numbers.pkl").exists())
    ]

    if variants_needing_work:
        print("Re-loading I_UV from the source zarr (not persisted in alpha_core.zarr)...")
        import dask.array as da
        grid = open_zarr()
        x_lo, x_hi = int(np.where(np.abs(grid.x_pc) <= FOOTPRINT_HALF_RANGE_PC)[0].min()), \
            int(np.where(np.abs(grid.x_pc) <= FOOTPRINT_HALF_RANGE_PC)[0].max()) + 1
        y_lo, y_hi = int(np.where(np.abs(grid.y_pc) <= FOOTPRINT_HALF_RANGE_PC)[0].min()), \
            int(np.where(np.abs(grid.y_pc) <= FOOTPRINT_HALF_RANGE_PC)[0].max()) + 1
        Iuv = da.from_zarr(grid.store["Iuv_final"])[:, y_lo:y_hi, x_lo:x_hi].compute().astype(np.float32)
    else:
        Iuv = None

    import pickle
    for variant in variants:
        summary_path = PARTIAL_DIR / f"{variant}_summary.npz"
        numbers_path = PARTIAL_DIR / f"{variant}_numbers.pkl"
        if summary_path.exists() and numbers_path.exists():
            print(f"{variant}: finalize partial already cached, skipping")
            continue
        summary_partial, numbers_partial = _finalize_one_variant(variant, core_store, footprint, z_pc, Iuv)
        np.savez_compressed(summary_path, **summary_partial)
        with open(numbers_path, "wb") as f:
            pickle.dump(numbers_partial, f)
        del summary_partial, numbers_partial
        gc.collect()

    have_all = all(
        (PARTIAL_DIR / f"{v}_summary.npz").exists() and (PARTIAL_DIR / f"{v}_numbers.pkl").exists()
        for v in VARIANTS
    )
    if not have_all:
        missing = [v for v in VARIANTS if not (PARTIAL_DIR / f"{v}_summary.npz").exists()]
        print(f"\nNot all variants finalized yet (missing: {missing}); skipping merge. "
              f"Re-run `finalize` for the remaining variant(s).")
        return

    finalize_merge()


def finalize_merge():
    t_start = time.time()
    import pickle

    core_store = zarr.open_group(str(ALPHA_CORE_ZARR_PATH), mode="r")
    summary = {
        "x_pc": np.asarray(core_store["x_pc"][:]),
        "y_pc": np.asarray(core_store["y_pc"][:]),
        "z_pc": np.asarray(core_store["z_pc"][:]),
    }
    numbers_rows = []
    for variant in VARIANTS:
        with np.load(PARTIAL_DIR / f"{variant}_summary.npz") as d:
            for k in d.files:
                summary[k] = d[k]
        with open(PARTIAL_DIR / f"{variant}_numbers.pkl", "rb") as f:
            numbers_rows.extend(pickle.load(f))

    print("\nWriting cache/core/summary.npz...")
    import io
    buf = io.BytesIO()
    np.savez_compressed(buf, **summary)
    size_mb = buf.tell() / 1e6
    print(f"  summary payload size: {size_mb:.1f} MB")
    with open(SUMMARY_NPZ_PATH, "wb") as f:
        f.write(buf.getvalue())
    if size_mb >= SUMMARY_NPZ_SIZE_LIMIT_MB:
        print(f"  >= {SUMMARY_NPZ_SIZE_LIMIT_MB} MB: will be left gitignored (cache/**/*.npz), not committed.")
    else:
        print(f"  < {SUMMARY_NPZ_SIZE_LIMIT_MB} MB: eligible to commit (still matches cache/**/*.npz gitignore -- "
              f"needs an explicit negation to actually commit).")

    print("Writing results/numbers_table.csv and .txt...")
    import csv
    with open(NUMBERS_TABLE_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "self_gravity", "quantity", "weighting", "stat", "value", "units", "definition"])
        for row in numbers_rows:
            writer.writerow(row)
    with open(NUMBERS_TABLE_TXT_PATH, "w") as f:
        f.write("Alpha paper -- headline numbers table\n")
        f.write("MASKED variant: not implemented -- no MASKED definition exists in any reference script.\n")
        f.write("self_gravity column: off | mean (footprint_mean, DEFAULT) | column (per_column, sensitivity option)\n")
        f.write("  -- see results/README.md for full column definitions.\n")
        f.write("=" * 120 + "\n")
        f.write(f"{'variant':<8}{'self_grav':<9}{'quantity':<22}{'weight':<7}{'stat':<20}{'value':>12} {'units':<16} definition\n")
        f.write("-" * 120 + "\n")
        for row in numbers_rows:
            variant, sg_key, qty, wt, stat, val, units, definition = row
            f.write(f"{variant:<8}{sg_key:<8}{qty:<22}{wt:<6}{stat:<20}{val: .6g} {units:<16} {definition}\n")

    elapsed = time.time() - t_start
    peak_mb = peak_working_set_mb()
    print(f"\nfinalize_merge runtime: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    if peak_mb is not None:
        print(f"Peak working set: {peak_mb:.0f} MB")
    print(f"Saved {SUMMARY_NPZ_PATH}")
    print(f"Saved {NUMBERS_TABLE_CSV_PATH}")
    print(f"Saved {NUMBERS_TABLE_TXT_PATH}")


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    mode = args[0] if args else "all"

    if mode == "build":
        variants_to_run = args[1:] or list(VARIANTS)
        build_stage(variants_to_run)
    elif mode == "finalize":
        variants_to_run = args[1:] or list(VARIANTS)
        finalize_stage(variants_to_run)
    elif mode == "merge":
        finalize_merge()
    elif mode == "all":
        build_stage(list(VARIANTS))
        finalize_stage(list(VARIANTS))
    else:
        raise SystemExit(
            f"Unknown mode {mode!r}. Usage: compute_all.py [build [VARIANT ...] | finalize [VARIANT ...] | merge | all]")
