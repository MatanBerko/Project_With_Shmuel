"""
Compute ALL numbers the alpha paper needs, in one pass over the density
cube, using ONLY src.physics (no reimplemented physics here). No figures
are produced by this script.

Variants: RAW, HIM_A, HIM_B. MASKED is NOT computed -- no MASKED
definition exists in any of the four reference scripts (see Phase A
report); this is a deliberate, reported omission.

Step 1c-prep (Shelest et al. 2026, arXiv:2607.15352) conventions
----------------------------------------------------------------
Every one of these is a switch in src.conventions, with the previous
behavior still reachable:

  * STATS_BOX -- the analysis volume is the 1 kpc x 1 kpc x 800 pc box
    centred on the Sun: |x|, |y| <= 500 pc and |z| <= 400 pc. All
    statistics, vertical profiles and PDFs use this box and nothing else.
    Two things deliberately keep the full z column the cube provides:
      - the P_tot hydrostatic integral (P = 0 stays pinned at the cube's
        top and bottom edge, +-750 pc for the f98 cube; clipping it to the
        box would put the zero-pressure boundary inside the gas layer);
      - the footprint average that sources "mean" self-gravity (g_gas(z)
        inside the box depends on gas outside it).
    Sigma_gas likewise stays a full-column integral over |x|, |y| <= 500.

  * PHASE_SCHEME -- "dPdn" (new default) classifies the neutral phases by
    density against the BS19 S-curve's two dP/dn = 0 turning points,
    n_W,max(I_UV) and n_C,min(I_UV); "temperature" is the old 300/6000 K
    cut. BOTH are computed and stored here, and both appear in the
    numbers table under the new phase_scheme column, so the two can be
    compared directly. The HIM flag (P_th < 0.5 P_min) is separate from
    and unchanged by this switch, and wins under both.

  * Slabs -- single-height statistics use |z - z_c| <= 30 pc (60 pc
    thick) at z_c = 0, 150, 300 pc (was +-25 pc).

  * Vertical profiles -- one point per GRID PLANE inside the box, no z
    binning at all (was 10 pc bins over |z| <= 500).

  * Percentiles -- 15th/85th (was 16th/84th), switchable.

PROVISIONAL: the cube in use is the f98 cube (n_H = 1653 A'), which is
known to need re-orienting; a re-oriented version and later a final
Porter-FUV cube (n_H = 1727 A') will replace it via the single
local_config.yaml zarr_filename entry. Every output file carries that
caveat as a header line.

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
from src.conventions import (
    M_H,
    MU,
    PERCENTILE_SCHEME_DEFAULT,
    PHASE_SCHEME_DEFAULT,
    PHASE_SCHEME_DPDN,
    PHASE_SCHEME_TEMPERATURE,
    PROVISIONAL_CUBE_HEADER,
    SELF_GRAVITY_MODE_FOOTPRINT_MEAN,
    SELF_GRAVITY_MODE_OFF,
    SELF_GRAVITY_MODE_PER_COLUMN,
    SLAB_CENTERS_PC,
    SLAB_HALF_THICKNESS_PC,
    STATS_BOX_XY_HALF_RANGE_PC,
    STATS_BOX_Z_HALF_RANGE_PC,
    XY_HALF_RANGE_PC,
)
from src.physics import derived, gravity, hydrostatic, thermal
from src.physics.him import (
    PHASE_CNM,
    PHASE_HIM,
    PHASE_UNM,
    PHASE_WNM,
    apply_variant,
    dpdn_unresolved_fraction,
    him_flag,
    phase_flag_dpdn,
    phase_flag_temperature,
)
from src.physics.loading import (
    footprint_mask,
    load_xy_subset_coords,
    open_zarr,
    slab_z_indices,
    stats_box_z_indices,
)
from src.physics.stats import mass_weighted_stats, percentile_levels, volume_weighted_stats

# ============================================================================
# Control block
# ============================================================================
VARIANTS = ("RAW", "HIM_A", "HIM_B")  # MASKED omitted -- no definition exists (see report)

# Step 1b: self-gravity settings are off | mean | column (was off/on, i.e.
# False/True -- "on" always meant per-column). "mean" (footprint_mean) is
# the per-paper default.
#
# Step 1c-prep: "column" is DROPPED FROM THIS RUN. The code path, its
# conventions entry, its tests and its cached zarr groups are all
# untouched -- adding "column" back to this tuple is the only change
# needed to sweep it again. It was dropped because Phase B already
# answered the question it was there to answer (per-column self-gravity
# inflates mass-weighted alpha by x1.9-2.6 near the midplane and is not
# the fiducial), so recomputing it here would triple the finalize cost for
# a sensitivity number that is not changing.
SELF_GRAVITY_SETTINGS = ("off", "mean")
SELF_GRAVITY_MODE_BY_SETTING = {
    "off": SELF_GRAVITY_MODE_OFF,
    "mean": SELF_GRAVITY_MODE_FOOTPRINT_MEAN,
    "column": SELF_GRAVITY_MODE_PER_COLUMN,  # kept: see note above
}
# A Step-1 run may have cached self_gravity_off / self_gravity_on groups
# (on == per_column). Reuse them under the new names instead of
# recomputing -- the underlying physics is byte-for-byte identical.
LEGACY_GROUP_NAME_BY_SETTING = {"off": "self_gravity_off", "column": "self_gravity_on"}

SELF_GRAVITY_DENSITY = "same_as_weight"  # "same_as_weight" (default) | "observed"

# XY footprint: used for LOADING, for Sigma_gas, for the self-gravity
# footprint average, and as the XY half of the STATS_BOX (they are the
# same +-500 pc square -- STATS_BOX only adds the |z| <= 400 pc bound).
FOOTPRINT_HALF_RANGE_PC = XY_HALF_RANGE_PC
assert STATS_BOX_XY_HALF_RANGE_PC == FOOTPRINT_HALF_RANGE_PC, (
    "STATS_BOX's XY extent and the loading footprint must be the same square."
)

# STATS_BOX: statistics/profiles/PDFs only. NOT the P_tot integration, NOT
# the self-gravity footprint average -- both of those use the full z
# column the cube provides. See the module docstring.
STATS_BOX_Z_HALF_PC = STATS_BOX_Z_HALF_RANGE_PC

PERCENTILE_SCHEME = PERCENTILE_SCHEME_DEFAULT  # "15_85" (Shelest+26) | "16_84"
PCT_LO, PCT_HI = percentile_levels(PERCENTILE_SCHEME)

# Both schemes are computed and reported so they can be compared; the
# DEFAULT one is the headline. The HIM flag is independent of both.
PHASE_SCHEMES_COMPUTED = (PHASE_SCHEME_DPDN, PHASE_SCHEME_TEMPERATURE)
PHASE_SCHEME_HEADLINE = PHASE_SCHEME_DEFAULT

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

# Per-variant base arrays that must all be present for a cached variant to
# be reusable without recomputation. "phase_flag" (singular, pre-Step-1c)
# is deliberately NOT in this list: it recorded one scheme with no record
# of which, so it is replaced rather than trusted.
BASE_ARRAYS = ("n_model", "Pth", "T", "Sigma_gas", "him",
               "phase_flag_dpdn", "phase_flag_temperature")
STALE_BASE_ARRAYS = ("phase_flag",)

PHASE_FLAG_ARRAY_BY_SCHEME = {
    PHASE_SCHEME_DPDN: "phase_flag_dpdn",
    PHASE_SCHEME_TEMPERATURE: "phase_flag_temperature",
}


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

STAT_NAMES = ("median", "mean", f"p{PCT_LO:g}", f"p{PCT_HI:g}")


def _stat_pairs(s):
    """(stat_name, value) pairs for one WeightedStats, with the percentile
    names taken from the stats object itself so a stored number can never
    be mislabelled if PERCENTILE_SCHEME changes."""
    return (("median", s.median), ("mean", s.arithmetic_mean),
            (s.lo_label, s.p_lo), (s.hi_label, s.p_hi))


# ============================================================================
# Vertical profiles: ONE POINT PER GRID PLANE inside the STATS_BOX
# (Shelest+26 -- no z binning). `use_abs` folds the +z and -z planes of the
# same |z| together into a single sample; the signed version keeps them
# separate. Both are produced: signed shows north/south asymmetry, folded
# is the higher-S/N profile.
# ============================================================================
def _profile_plane_groups(z_pc, use_abs: bool):
    """[(z_value, plane_indices), ...] -- one entry per profile point.

    Signed: one plane per entry, ascending z, |z| <= STATS_BOX_Z_HALF_PC.
    Folded: one entry per distinct |z|, pairing the +z and -z planes.
    """
    box_idx = stats_box_z_indices(z_pc, STATS_BOX_Z_HALF_PC)
    if not use_abs:
        order = box_idx[np.argsort(z_pc[box_idx])]
        return [(float(z_pc[i]), np.array([i])) for i in order]

    abs_z = np.abs(z_pc[box_idx])
    out = []
    for z_val in np.unique(np.round(abs_z, 6)):
        sel = box_idx[np.isclose(abs_z, z_val, atol=1e-6)]
        out.append((float(z_val), sel))
    return out


def compute_vertical_profile(z_pc, Pth, Ptot, alpha_arr, him_cube, phase_cubes, T_cube,
                               n_model, footprint, use_abs: bool):
    """Per-plane vertical profiles of Pth/Ptot/alpha stats, phase fractions
    (one set per PHASE_SCHEME) and Mach/sigma_eff, inside the STATS_BOX.

    phase_cubes: {scheme_name: int8 phase flag cube}. All arrays are
    already restricted to the STATS_BOX in z by the caller, and indices
    here are into that restricted array.
    """
    groups = _profile_plane_groups(z_pc, use_abs)
    n_pts = len(groups)
    centers = np.array([g[0] for g in groups], dtype=float)

    out = {}
    for qty in ("Pth", "Ptot", "alpha"):
        for wt in ("vol", "mw"):
            for s in STAT_NAMES:
                out[f"{qty}_{wt}_{s}"] = np.full(n_pts, np.nan)
    for scheme in phase_cubes:
        for _, ph in PHASE_CODES_ALL:
            out[f"f_{ph}_vol__{scheme}"] = np.full(n_pts, np.nan)
            out[f"f_{ph}_mw__{scheme}"] = np.full(n_pts, np.nan)
    for ph in ("CNM", "UNM", "WNM", "total"):
        out[f"sigma_eff_{ph}"] = np.full(n_pts, np.nan)
    out["mach"] = np.full(n_pts, np.nan)
    out["n_cells"] = np.zeros(n_pts, dtype=np.int64)

    headline_phase = phase_cubes[PHASE_SCHEME_HEADLINE]

    for b, (_, idxs) in enumerate(groups):
        him_sub = him_cube[idxs][:, footprint]
        n_sub = n_model[idxs][:, footprint].astype(np.float64)
        T_sub = T_cube[idxs][:, footprint].astype(np.float64)
        Pth_sub = Pth[idxs][:, footprint].astype(np.float64)
        Ptot_sub = Ptot[idxs][:, footprint].astype(np.float64)
        alpha_sub = alpha_arr[idxs][:, footprint].astype(np.float64)
        neutral = ~him_sub
        out["n_cells"][b] = him_sub.size

        for qty, arr_sub in (("Pth", Pth_sub), ("Ptot", Ptot_sub), ("alpha", alpha_sub)):
            s_vol = volume_weighted_stats(arr_sub, neutral, PERCENTILE_SCHEME)
            s_mw = mass_weighted_stats(arr_sub, neutral, n_sub, PERCENTILE_SCHEME)
            for wt, s in (("vol", s_vol), ("mw", s_mw)):
                for stat_name, val in _stat_pairs(s):
                    out[f"{qty}_{wt}_{stat_name}"][b] = val

        n_weight_sub = np.where(np.isfinite(n_sub), n_sub, 0.0)
        W = n_weight_sub.sum()
        for scheme, phase_cube in phase_cubes.items():
            phase_sub = phase_cube[idxs][:, footprint]
            total_cells = phase_sub.size
            for code, ph in PHASE_CODES_ALL:
                m = phase_sub == code
                out[f"f_{ph}_vol__{scheme}"][b] = m.sum() / total_cells if total_cells > 0 else np.nan
                out[f"f_{ph}_mw__{scheme}"][b] = n_weight_sub[m].sum() / W if W > 0 else np.nan

        alpha_plane_mean = out["alpha_vol_mean"][b]
        out["mach"][b] = float(derived.mach_number(np.array([alpha_plane_mean]))[0])
        phase_sub_headline = headline_phase[idxs][:, footprint]
        for code, ph in PHASE_CODES_NEUTRAL:
            m = neutral & (phase_sub_headline == code)
            T_ph_mean = float(T_sub[m].mean()) if m.any() else np.nan
            out[f"sigma_eff_{ph}"][b] = float(
                derived.sigma_eff_kmps(np.array([alpha_plane_mean]), np.array([T_ph_mean]))[0])
        T_tot_mean = float(T_sub[neutral].mean()) if neutral.any() else np.nan
        out["sigma_eff_total"][b] = float(
            derived.sigma_eff_kmps(np.array([alpha_plane_mean]), np.array([T_tot_mean]))[0])

    out["z_pc"] = centers
    return out


# ============================================================================
# 1D / 2D PDFs at the 60 pc-thick slabs (|z - z_c| <= SLAB_HALF_THICKNESS_PC)
# ============================================================================
def slab_plane_indices(z_pc, z_center):
    """Slab plane indices, clipped to the STATS_BOX (src.physics.loading)."""
    return slab_z_indices(z_pc, z_center, SLAB_HALF_THICKNESS_PC, STATS_BOX_Z_HALF_PC)


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
Z_CHUNK_CLASSIFY = 48  # z planes per chunk when classifying (memory bound)


def _him_chunked(n_raw, T, Pmin, chunk=Z_CHUNK_CLASSIFY):
    """HIM flag over the whole cube, a z-chunk at a time.

    Element for element identical to him_flag(Pth_raw, Pmin) on the full
    cube -- Pth is formed in float64 and rounded to float32 exactly as
    before, just never materialised for all 751 planes at once. That
    matters here only for peak memory (a full float32 Pth cube is ~0.75
    GB); it must NOT change the flag, because the cached HIM_A/HIM_B
    densities were built from it.
    """
    him = np.empty(n_raw.shape, dtype=bool)
    for lo in range(0, n_raw.shape[0], chunk):
        hi = min(lo + chunk, n_raw.shape[0])
        pth_c = (n_raw[lo:hi].astype(np.float64) * T[lo:hi].astype(np.float64)).astype(np.float32)
        him[lo:hi] = him_flag(pth_c, Pmin[lo:hi])
    return him


def _classify_dpdn_chunked(n_raw, Iuv, him, bounds, chunk=Z_CHUNK_CLASSIFY):
    """dPdn phase flag over the whole cube, a z-chunk at a time.

    n_W,max(I_UV) and n_C,min(I_UV) are evaluated per chunk and thrown
    away, instead of building two more full float32 cubes (~1.5 GB
    together) just to compare against the density once.

    Returns (int8 phase cube, fraction of neutral cells with no finite
    boundary).
    """
    out = np.empty(n_raw.shape, dtype=np.int8)
    unresolved = 0
    neutral_total = 0
    for lo in range(0, n_raw.shape[0], chunk):
        hi = min(lo + chunk, n_raw.shape[0])
        iuv_c = Iuv[lo:hi]
        nw = bounds.n_w_max(iuv_c).astype(np.float32)
        nc = bounds.n_c_min(iuv_c).astype(np.float32)
        him_c = him[lo:hi]
        out[lo:hi] = phase_flag_dpdn(n_raw[lo:hi], nw, nc, him_c)
        neutral_total += int(him_c.size - him_c.sum())
        unresolved += int(round(
            dpdn_unresolved_fraction(n_raw[lo:hi], nw, nc, him_c)
            * (him_c.size - him_c.sum())))
    return out, (unresolved / neutral_total if neutral_total else float("nan"))


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

    # mode="a": RESUMABLE -- a prior run may have already written some
    # variants (this is a long job; a killed/interrupted run should not
    # lose completed variants). Existing groups are loaded back rather
    # than recomputed. Opened BEFORE the expensive arrays are built so the
    # remaining work is known up front and nothing is computed that the
    # cache already has -- this machine does not have the headroom to hold
    # every intermediate at once.
    core_store = zarr.open_group(str(ALPHA_CORE_ZARR_PATH), mode="a")
    core_arrays = ("n_model", "Pth", "T", "Sigma_gas")
    needs_core_build = [
        v for v in variants
        if v not in core_store or any(a not in core_store[v] for a in core_arrays)
    ]
    sg_work = {
        v: [k for k in SELF_GRAVITY_SETTINGS
            if v in core_store
            and not (f"self_gravity_{k}" in core_store[v]
                     and "Ptot" in core_store[v][f"self_gravity_{k}"]
                     and "alpha" in core_store[v][f"self_gravity_{k}"])]
        for v in variants
    }
    for v in needs_core_build:
        sg_work[v] = list(SELF_GRAVITY_SETTINGS)
    print(f"  variants needing a core (re)build: {needs_core_build or 'none'}")
    print(f"  self-gravity work remaining: "
          f"{ {v: w for v, w in sg_work.items() if w} or 'none' }")

    cfg = load_resolved_config()
    pminmax = thermal.build_pmin_pmax(cfg["bs19_mat_path"])
    print("Evaluating P_min(I_UV) over the full sub-cube (HIM threshold)...")
    Pmin = pminmax.p_min(Iuv).astype(np.float32)

    print("Classifying HIM (P_th < 0.5 P_min), chunked over z...")
    him_cube = _him_chunked(n_raw, T, Pmin)
    print(f"  HIM fraction (whole sub-cube): {him_cube.mean():.4%}")

    # Both phase schemes, computed once and stored side by side, on the
    # OBSERVED density: HIM cells (the only ones a variant's density
    # substitution touches) are relabelled PHASE_HIM anyway, so one
    # phase_flag per scheme is valid for all three variants.
    print("Evaluating Shelest+26 dP/dn phase boundaries n_W,max(I_UV)/n_C,min(I_UV)...")
    bounds = thermal.build_phase_density_bounds(cfg["bs19_mat_path"])
    print(f"  method: {bounds.method}")
    phase_dpdn, unresolved = _classify_dpdn_chunked(n_raw, Iuv, him_cube, bounds)
    print(f"  dPdn: neutral cells with no finite boundary (defaulted to UNM): {unresolved:.4%}")
    del Iuv
    gc.collect()

    phase_cubes = {
        PHASE_SCHEME_DPDN: phase_dpdn,
        PHASE_SCHEME_TEMPERATURE: phase_flag_temperature(T, him_cube),
    }
    del phase_dpdn
    for scheme, cube in phase_cubes.items():
        fracs = ", ".join(f"{ph} {np.mean(cube == code):.2%}" for code, ph in PHASE_CODES_ALL)
        print(f"  {scheme:12s} whole sub-cube: {fracs}")

    # P_max and the full float32 P_th cube are needed ONLY by
    # apply_variant(), i.e. only when a variant's core arrays have to be
    # (re)built. Skip them otherwise -- together they are ~1.5 GB.
    if needs_core_build:
        print("Evaluating P_max(I_UV) and P_th for the variant construction...")
        Iuv = da.from_zarr(grid.store["Iuv_final"])[:, y_lo:y_hi, x_lo:x_hi].compute().astype(np.float32)
        Pmax = pminmax.p_max(Iuv).astype(np.float32)
        del Iuv
        Pth_raw = (n_raw.astype(np.float64) * T.astype(np.float64)).astype(np.float32)
    else:
        Pmax = None
        Pth_raw = None
        print("All variants' core arrays are cached -- skipping P_max/P_th (not needed).")
    gc.collect()

    if "x_pc" not in core_store:
        core_store.create_array("x_pc", data=x_sub.astype(np.float32))
        core_store.create_array("y_pc", data=y_sub.astype(np.float32))
        core_store.create_array("z_pc", data=z_pc.astype(np.float32))
    core_store.attrs["variants"] = list(VARIANTS)
    core_store.attrs["self_gravity_settings"] = list(SELF_GRAVITY_SETTINGS)
    core_store.attrs["self_gravity_density"] = SELF_GRAVITY_DENSITY
    core_store.attrs["footprint_half_range_pc"] = FOOTPRINT_HALF_RANGE_PC
    core_store.attrs["stats_box_xy_half_range_pc"] = STATS_BOX_XY_HALF_RANGE_PC
    core_store.attrs["stats_box_z_half_range_pc"] = STATS_BOX_Z_HALF_PC
    core_store.attrs["phase_schemes"] = list(PHASE_SCHEMES_COMPUTED)
    core_store.attrs["phase_scheme_headline"] = PHASE_SCHEME_HEADLINE
    core_store.attrs["phase_density_bounds_method"] = bounds.method
    core_store.attrs["dpdn_unresolved_fraction"] = float(unresolved)
    core_store.attrs["percentile_scheme"] = PERCENTILE_SCHEME
    core_store.attrs["slab_half_thickness_pc"] = SLAB_HALF_THICKNESS_PC
    core_store.attrs["provisional"] = PROVISIONAL_CUBE_HEADER
    core_store.attrs["masked_variant"] = "not implemented -- no MASKED definition in any reference script"

    for variant in variants:
        print(f"\n=== Variant {variant} ===")
        t_v = time.time()

        if variant in needs_core_build:
            if variant in core_store:
                print(f"  {variant} present but core arrays incomplete (killed mid-run?) -- discarding and redoing")
                del core_store[variant]
            vr = apply_variant(variant, n_raw.astype(np.float64), Pth_raw.astype(np.float64),
                                 him_cube, Pmin.astype(np.float64), Pmax.astype(np.float64))
            n_model = vr.n_model.astype(np.float32)
            Pth_model = vr.Pth_model.astype(np.float32)
            del vr
            sigma_gas_variant = derived.sigma_gas_map(z_pc, n_model.astype(np.float64)).astype(np.float32)

            vg = core_store.require_group(variant)
            vg.create_array("n_model", data=n_model, chunks=(CHUNK_Z, n_model.shape[1], n_model.shape[2]))
            vg.create_array("Pth", data=Pth_model, chunks=(CHUNK_Z, Pth_model.shape[1], Pth_model.shape[2]))
            vg.create_array("T", data=T, chunks=(CHUNK_Z, T.shape[1], T.shape[2]))
            vg.create_array("Sigma_gas", data=sigma_gas_variant)
            del sigma_gas_variant
        else:
            print(f"  {variant} core arrays already in {ALPHA_CORE_ZARR_PATH} -- not recomputed")
            vg = core_store[variant]
            n_model = None
            Pth_model = None

        # --- Step 1c him/phase arrays: added in place, so an existing
        # ~5 GB of valid core cache is never discarded just to record a
        # new classification of it.
        for stale in STALE_BASE_ARRAYS:
            if stale in vg:
                print(f"  dropping stale pre-Step-1c array '{stale}' (its scheme was not recorded)")
                del vg[stale]
        if "him" not in vg:
            print("  writing 'him' (scheme-independent HIM flag)")
            vg.create_array("him", data=him_cube.astype(np.int8),
                              chunks=(CHUNK_Z, him_cube.shape[1], him_cube.shape[2]))
        for scheme, arr_name in PHASE_FLAG_ARRAY_BY_SCHEME.items():
            if arr_name not in vg:
                print(f"  writing '{arr_name}'")
                cube = phase_cubes[scheme]
                vg.create_array(arr_name, data=cube,
                                  chunks=(CHUNK_Z, cube.shape[1], cube.shape[2]))

        pending_sg = sg_work.get(variant, [])
        if not pending_sg:
            print("  all self-gravity settings already complete in zarr -- nothing to integrate")
        else:
            if n_model is None:
                n_model = np.asarray(vg["n_model"][:], dtype=np.float32)
                Pth_model = np.asarray(vg["Pth"][:], dtype=np.float32)
            rho = (MU * M_H * n_model.astype(np.float64))
            # Already restricted to the +-500pc square footprint by the
            # x_lo:x_hi / y_lo:y_hi slice above, so "footprint_mean" here
            # averages over the entire loaded XY plane -- at every z the
            # cube provides, which is exactly the Shelest+26 convention
            # (the STATS_BOX |z| <= 400 pc bound is NOT applied to
            # self-gravity; see the module docstring).
            full_footprint_mask = np.ones((n_model.shape[1], n_model.shape[2]), dtype=bool)

            for sg_key in pending_sg:
                mode = SELF_GRAVITY_MODE_BY_SETTING[sg_key]
                sg_group_name = f"self_gravity_{sg_key}"

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
                else:
                    print(f"  self_gravity={sg_key}: P_tot integration...")
                    gravity_density = None
                    if mode != SELF_GRAVITY_MODE_OFF:
                        gravity_density = (n_model if SELF_GRAVITY_DENSITY == "same_as_weight"
                                           else n_raw).astype(np.float64)
                    g, nan_fraction = gravity.g_total_cgs(z_pc, gravity_density, mode=mode,
                                                            footprint_mask=full_footprint_mask)
                    Ptot_kB = hydrostatic.p_tot_kb_full_column(z_pc, rho, g).astype(np.float32)
                    alpha_arr = derived.alpha(Ptot_kB.astype(np.float64),
                                                Pth_model.astype(np.float64)).astype(np.float32)
                    del g, gravity_density

                sgg = vg.require_group(sg_group_name)
                sgg.create_array("Ptot", data=Ptot_kB, chunks=(CHUNK_Z, Ptot_kB.shape[1], Ptot_kB.shape[2]))
                sgg.create_array("alpha", data=alpha_arr, chunks=(CHUNK_Z, alpha_arr.shape[1], alpha_arr.shape[2]))
                if not legacy_cached and nan_fraction is not None:
                    sgg.create_array("footprint_nan_fraction", data=nan_fraction.astype(np.float32))
                print(f"  self_gravity={sg_key}: written to zarr")

                del Ptot_kB, alpha_arr
                gc.collect()
            del rho
        del n_model, Pth_model
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
#
# Everything here reads ONLY the STATS_BOX z range: the 3D arrays are
# sliced to |z| <= STATS_BOX_Z_HALF_PC on the way in, which both enforces
# the Shelest+26 analysis volume and cuts finalize's memory by ~47%.
# Full-column quantities (Sigma_gas, footprint_nan_fraction) are already
# reduced in the cache and are read whole.
# ============================================================================
PARTIAL_DIR = CACHE_DIR / "_finalize_partial"


def _box_z_slice(z_pc_full):
    """(slice, z_pc_in_box) for the contiguous STATS_BOX z range."""
    idx = stats_box_z_indices(z_pc_full, STATS_BOX_Z_HALF_PC)
    lo, hi = int(idx.min()), int(idx.max()) + 1
    if hi - lo != idx.size:
        raise ValueError("STATS_BOX z selection is not contiguous in the cube's z ordering.")
    return slice(lo, hi), z_pc_full[lo:hi]


def _finalize_one_variant(variant, core_store, footprint, z_pc_full, Iuv_box):
    """All the per-variant summary/PDF/vertical-profile/numbers-table work,
    returning (summary_partial, numbers_rows_partial) for just this variant.
    """
    summary = {}
    numbers_rows = []
    print(f"\n=== Finalizing variant {variant} ===")
    t_v = time.time()
    zsl, z_pc = _box_z_slice(z_pc_full)
    vg = core_store[variant]
    n_model = np.asarray(vg["n_model"][zsl], dtype=np.float32)
    Pth_model = np.asarray(vg["Pth"][zsl], dtype=np.float32)
    T = np.asarray(vg["T"][zsl], dtype=np.float32)
    him_cube = np.asarray(vg["him"][zsl]).astype(bool)
    phase_cubes = {
        scheme: np.asarray(vg[arr_name][zsl], dtype=np.int8)
        for scheme, arr_name in PHASE_FLAG_ARRAY_BY_SCHEME.items()
    }
    sigma_gas_variant = np.asarray(vg["Sigma_gas"][:], dtype=np.float32)
    print(f"  STATS_BOX: {len(z_pc)} z planes, z in [{z_pc.min():.0f}, {z_pc.max():.0f}] pc, "
          f"{footprint.sum()} XY cells per plane")

    # Sigma_gas: FULL-column integral over the +-500 pc square -- the
    # STATS_BOX z bound deliberately does not apply (see module docstring).
    summary[f"{variant}__Sigma_gas_map"] = sigma_gas_variant
    s_sigma = volume_weighted_stats(sigma_gas_variant, footprint, PERCENTILE_SCHEME)
    for stat_name, val in _stat_pairs(s_sigma):
        numbers_rows.append((variant, "n/a", "n/a", "Sigma_gas", "vol", stat_name, val,
                               "Msun/pc^2",
                               f"Full-column (+-750pc) trapezoid of variant density, +-500pc square "
                               f"footprint, unweighted {stat_name}"))

    # 1D PDFs (self-gravity independent: n, T, Iuv, Pth)
    for qty, arr in (("n", n_model), ("T", T), ("Iuv", Iuv_box), ("Pth", Pth_model)):
        for zc in SLAB_CENTERS_PC:
            idxs = slab_plane_indices(z_pc, zc)
            sub = arr[idxs][:, footprint]
            neutral = ~him_cube[idxs][:, footprint]
            centers, counts = log_pdf_1d(sub[neutral].ravel(), None)
            summary[f"{variant}__pdf1d__{qty}__z{int(zc)}__centers"] = centers
            summary[f"{variant}__pdf1d__{qty}__z{int(zc)}__counts"] = counts

    # 2D mass-weighted PDFs (n,Pth) and (Iuv,n) -- self-gravity independent
    for (qx, xarr, qy, yarr) in (("n", n_model, "Pth", Pth_model), ("Iuv", Iuv_box, "n", n_model)):
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

    # ---- Phase fractions: self-gravity independent, emitted ONCE (not per
    # self-gravity setting) under both PHASE_SCHEMEs, at each slab and over
    # the whole STATS_BOX. HIM is included here; everywhere else HIM cells
    # are excluded from statistics.
    n_weight = np.where(np.isfinite(n_model), n_model, 0.0).astype(np.float64)
    for scheme, phase_cube in phase_cubes.items():
        for label, idxs, where in (
            *[(f"z{int(zc)}", slab_plane_indices(z_pc, zc),
               f"{zc:.0f}+-{SLAB_HALF_THICKNESS_PC:.0f}pc slab") for zc in SLAB_CENTERS_PC],
            ("box", np.arange(len(z_pc)), "STATS_BOX (|x|,|y|<=500, |z|<=400pc)"),
        ):
            phase_sub = phase_cube[idxs][:, footprint]
            w_sub = n_weight[idxs][:, footprint]
            total_cells = phase_sub.size
            W = w_sub.sum()
            for code, ph in PHASE_CODES_ALL:
                m = phase_sub == code
                numbers_rows.append((
                    variant, "n/a", scheme, f"phase_fraction_{ph}", "vol", label,
                    float(m.sum() / total_cells) if total_cells else float("nan"),
                    "dimensionless", f"Volume fraction of {ph}, {where}, {scheme} scheme"))
                numbers_rows.append((
                    variant, "n/a", scheme, f"phase_fraction_{ph}", "mw", label,
                    float(w_sub[m].sum() / W) if W > 0 else float("nan"),
                    "dimensionless", f"Mass fraction of {ph}, {where}, {scheme} scheme"))

    for sg_key in SELF_GRAVITY_SETTINGS:
        sgg = vg[f"self_gravity_{sg_key}"]
        Ptot_kB = np.asarray(sgg["Ptot"][zsl], dtype=np.float32)
        alpha_arr = np.asarray(sgg["alpha"][zsl], dtype=np.float32)
        if "footprint_nan_fraction" in sgg:
            # full-column diagnostic, read whole on purpose
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

        print(f"  self_gravity={sg_key}: per-plane vertical profiles (signed z and |z|)...")
        prof_signed = compute_vertical_profile(z_pc, Pth_model, Ptot_kB, alpha_arr, him_cube, phase_cubes,
                                                  T, n_model, footprint, use_abs=False)
        prof_abs = compute_vertical_profile(z_pc, Pth_model, Ptot_kB, alpha_arr, him_cube, phase_cubes,
                                               T, n_model, footprint, use_abs=True)
        for k, v in prof_signed.items():
            summary[f"{variant}__sg{sg_key}__profile_signedz__{k}"] = v
        for k, v in prof_abs.items():
            summary[f"{variant}__sg{sg_key}__profile_absz__{k}"] = v

        # ---- Slab statistics (60 pc-thick slabs) and the whole-box statistic.
        # "box" is a DIRECT statistic over every neutral cell in the
        # STATS_BOX -- not an average of the vertical profile (which is
        # what the retired absz500_* rows were). One number, one
        # population, no double reduction.
        for label, idxs, where in (
            *[(f"z{int(zc)}", slab_plane_indices(z_pc, zc),
               f"{zc:.0f}+-{SLAB_HALF_THICKNESS_PC:.0f}pc slab, +-500pc square") for zc in SLAB_CENTERS_PC],
            ("box", np.arange(len(z_pc)), "STATS_BOX (|x|,|y|<=500, |z|<=400pc)"),
        ):
            neutral = ~him_cube[idxs][:, footprint]
            n_sub = n_model[idxs][:, footprint].astype(np.float64)
            for qty, arr in (("Pth", Pth_model), ("Ptot", Ptot_kB), ("alpha", alpha_arr)):
                sub = arr[idxs][:, footprint].astype(np.float64)
                for wt, s in (("vol", volume_weighted_stats(sub, neutral, PERCENTILE_SCHEME)),
                                ("mw", mass_weighted_stats(sub, neutral, n_sub, PERCENTILE_SCHEME))):
                    for stat_name, val in _stat_pairs(s):
                        numbers_rows.append((
                            variant, sg_key, "n/a", qty, wt, f"{label}_{stat_name}", val,
                            "K cm^-3" if qty in ("Pth", "Ptot") else "dimensionless",
                            f"{qty} {stat_name}, {wt}-weighted, {where}, neutral (non-HIM) cells"
                        ))

        iz_mid = int(np.argmin(np.abs(z_pc)))
        alpha_mid_vol = volume_weighted_stats(alpha_arr[iz_mid], footprint & ~him_cube[iz_mid],
                                                PERCENTILE_SCHEME)
        mach_mid = float(derived.mach_number(np.array([alpha_mid_vol.arithmetic_mean]))[0])
        T_neutral_mid = T[iz_mid][footprint & ~him_cube[iz_mid]].astype(np.float64)
        sigma_eff_mid = float(derived.sigma_eff_kmps(
            np.array([alpha_mid_vol.arithmetic_mean]),
            np.array([T_neutral_mid.mean() if T_neutral_mid.size else np.nan]))[0])
        numbers_rows.append((variant, sg_key, "n/a", "Mach", "vol", "midplane", mach_mid,
                               "dimensionless", "Mach number at the z=0 plane, from volume-weighted mean alpha, neutral cells"))
        numbers_rows.append((variant, sg_key, "n/a", "sigma_eff", "vol", "midplane", sigma_eff_mid,
                               "km/s", "Effective turbulent velocity dispersion at the z=0 plane, neutral cells, all neutral phases"))

        del Ptot_kB, alpha_arr
    del n_model, Pth_model, T, phase_cubes, him_cube
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
        if variant not in core_store:
            raise RuntimeError(
                f"{ALPHA_CORE_ZARR_PATH} has no '{variant}' group -- "
                f"run `python pipeline/compute_all.py build {variant}` first.")
        missing_base = [a for a in BASE_ARRAYS if a not in core_store[variant]]
        missing_sg = [k for k in SELF_GRAVITY_SETTINGS if f"self_gravity_{k}" not in core_store[variant]]
        if missing_base or missing_sg:
            raise RuntimeError(
                f"{ALPHA_CORE_ZARR_PATH}'s '{variant}' group is incomplete "
                f"(missing arrays {missing_base}, missing self-gravity groups {missing_sg}) -- "
                f"run `python pipeline/compute_all.py build {variant}` first.")

    x_sub = np.asarray(core_store["x_pc"][:], dtype=np.float64)
    y_sub = np.asarray(core_store["y_pc"][:], dtype=np.float64)
    z_pc_full = np.asarray(core_store["z_pc"][:], dtype=np.float64)
    footprint = footprint_mask(x_sub, y_sub, FOOTPRINT_HALF_RANGE_PC)

    variants_needing_work = [
        v for v in variants
        if not ((PARTIAL_DIR / f"{v}_summary.npz").exists() and (PARTIAL_DIR / f"{v}_numbers.pkl").exists())
    ]

    if variants_needing_work:
        print("Re-loading I_UV from the source zarr (not persisted in alpha_core.zarr), STATS_BOX only...")
        import dask.array as da
        zsl, _ = _box_z_slice(z_pc_full)
        grid = open_zarr()
        x_lo, x_hi = int(np.where(np.abs(grid.x_pc) <= FOOTPRINT_HALF_RANGE_PC)[0].min()), \
            int(np.where(np.abs(grid.x_pc) <= FOOTPRINT_HALF_RANGE_PC)[0].max()) + 1
        y_lo, y_hi = int(np.where(np.abs(grid.y_pc) <= FOOTPRINT_HALF_RANGE_PC)[0].min()), \
            int(np.where(np.abs(grid.y_pc) <= FOOTPRINT_HALF_RANGE_PC)[0].max()) + 1
        Iuv_box = da.from_zarr(grid.store["Iuv_final"])[zsl, y_lo:y_hi, x_lo:x_hi].compute().astype(np.float32)
    else:
        Iuv_box = None

    import pickle
    for variant in variants:
        summary_path = PARTIAL_DIR / f"{variant}_summary.npz"
        numbers_path = PARTIAL_DIR / f"{variant}_numbers.pkl"
        if summary_path.exists() and numbers_path.exists():
            print(f"{variant}: finalize partial already cached, skipping")
            continue
        summary_partial, numbers_partial = _finalize_one_variant(
            variant, core_store, footprint, z_pc_full, Iuv_box)
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


NUMBERS_TABLE_COLUMNS = ["variant", "self_gravity", "phase_scheme", "quantity",
                         "weighting", "stat", "value", "units", "definition"]


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
        f.write(f"# {PROVISIONAL_CUBE_HEADER}\n")
        writer = csv.writer(f)
        writer.writerow(NUMBERS_TABLE_COLUMNS)
        for row in numbers_rows:
            writer.writerow(row)
    with open(NUMBERS_TABLE_TXT_PATH, "w") as f:
        f.write(f"{PROVISIONAL_CUBE_HEADER}\n")
        f.write("Alpha paper -- headline numbers table (Shelest+26 conventions)\n")
        f.write(f"STATS_BOX: |x|,|y| <= {STATS_BOX_XY_HALF_RANGE_PC:.0f} pc, |z| <= {STATS_BOX_Z_HALF_PC:.0f} pc. "
                f"Slabs: |z-z_c| <= {SLAB_HALF_THICKNESS_PC:.0f} pc. "
                f"Percentiles: {PCT_LO:g}/{PCT_HI:g}.\n")
        f.write(f"phase_scheme column: {PHASE_SCHEME_DPDN} (headline) | {PHASE_SCHEME_TEMPERATURE}; "
                f"n/a where the quantity does not depend on the scheme.\n")
        f.write("MASKED variant: not implemented -- no MASKED definition exists in any reference script.\n")
        f.write("self_gravity column: off | mean (footprint_mean, DEFAULT); column (per_column) not run "
                "in this sweep -- code retained.\n")
        f.write("  -- see results/README.md for full column definitions.\n")
        f.write("=" * 132 + "\n")
        f.write(f"{'variant':<8}{'self_grav':<10}{'phase_sch':<13}{'quantity':<22}{'weight':<7}"
                f"{'stat':<16}{'value':>12} {'units':<16} definition\n")
        f.write("-" * 132 + "\n")
        for row in numbers_rows:
            variant, sg_key, scheme, qty, wt, stat, val, units, definition = row
            f.write(f"{variant:<8}{sg_key:<10}{scheme:<13}{qty:<22}{wt:<7}{stat:<16}"
                    f"{val: .6g} {units:<16} {definition}\n")

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
