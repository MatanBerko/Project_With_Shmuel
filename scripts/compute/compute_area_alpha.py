"""
Step 2d: AREA-AVERAGED P_tot and alpha -- the PI's physical definition --
plus a sub-area scale test.

Why this script exists
----------------------
P_tot is a MACROSCOPIC pressure. Vertical hydrostatic balance between
pressure and weight holds only on average over a large area: Ostriker &
Kim (2022) horizontally average both the weight and the midplane
pressures before forming any ratio. Everything in this project so far
computed P_tot per (x,y) column and then reduced alpha over CELLS, which
answers a different question ("what does a typical cell look like"). This
script computes the area average first and takes the ratio second.

The identity that makes it cheap
--------------------------------
g(z) is identical for every (x,y) column under both fiducial self-gravity
settings ("off" = g_ext only; "mean" = g_ext + the footprint-averaged
g_gas), and P_tot is LINEAR in rho. So the horizontal average of the
per-column P_tot is exactly the hydrostatic integral of the horizontally
averaged density:

    <P_tot>_area(z) = integral_z^top <rho>_area g dz'

Part (b) below verifies that numerically rather than asserting it: the
plane means of the CACHED per-column P_tot are compared against
hydrostatic.p_tot_kb_full_column() applied to the 1-D footprint-mean
density profile. The two go through completely different code paths (one
is a mean of 251001 independently integrated columns, the other a single
integral of one profile), so agreement to float precision is a real
check on both the cache and the reasoning above. If it fails, the script
raises instead of reporting.

Definitions (one number per height, or per patch)
-------------------------------------------------
  <P_tot>_area(z)      plain (equal-weight) mean of the per-column P_tot
                       over ALL cells in the plane, HIM-flagged cells
                       INCLUDED. P_tot is an integral of the density
                       field, and the fiducial treatment keeps the
                       flagged cells' observed mass in the hydrostatic
                       weight -- so there is nothing to exclude here.
  <P_th>_neutral(z)    mean of p_th_phys over NON-HIM cells only, both
                       volume-weighted (plain mean) and mass-weighted
                       (weights n_H). The flagged cells are excluded
                       because their pressure is what the flag says is
                       not neutral-medium pressure.
  alpha_area(z)        <P_tot>_area / <P_th>_neutral, for both weightings.

Note the deliberate asymmetry: the numerator averages over every cell,
the denominator over the neutral cells. That is the fiducial treatment of
this project ("flagged cells excluded from statistics, their mass kept in
the weight") carried over to area averages, not a new choice.

The sub-area scale test
-----------------------
"Large enough area" is an assumption, so it is measured. The footprint is
split into N x N equal-width square patches for N = 2, 4, 8 (sides 500,
250, 125 pc) and the same three quantities are formed inside each patch
from that patch's cells only. g(z) is NOT recomputed per patch -- it is a
property of the whole disc, and recomputing it per patch would be a
different physical model, not a scale test of this one. The spread of
alpha_area across patches at a given height is then how much the answer
depends on how much sky you average over.

Patch edges sit at exact multiples of 1000/N pc from -500 pc, left-closed
/ right-open except the last, which is closed -- the same convention
profile_bin_edges() uses in z. The cube's 501 x 501 footprint has an ODD
number of 2 pc cells per axis, so at N = 2 the patches hold 250 and 251
cells per axis (equal AREA, 1 cell different in count); the quantities
here are all means, so that is immaterial, but it is why patch cell
counts are reported.

COMPUTE script: may import src.physics and read the 3D cache. The figure
that consumes it (scripts/design/fig_area_alpha.py) reads only the .npz
written here. NO PHYSICS IS CHANGED OR RECOMPUTED HERE: P_tot, alpha, the
HIM flag and p_th_phys are all read from the cache. The only thing
recomputed is g(z) itself, which the cache does not store -- and that is
done through the same src.physics.gravity call the build stage made,
from the same cached n_model, so it is a re-derivation rather than a
second implementation.

Step 2e: the per-cell alpha
---------------------------
The PI's per-cell definition sits next to alpha_area:

  alpha_cell(x,y,z)    <P_tot>_area / p_th_phys(x,y,z) for every NON-HIM
                       cell, with <P_tot>_area the group's own number (one
                       per 4 pc bin or 60 pc slab, the same one alpha_area
                       uses).
  P_th,harm            [sum(w/p_th)/sum(w)]^-1, the harmonic mean.
  <alpha_cell>         <P_tot>_area / P_th,harm.
  median, 15/85 pct    of alpha_cell, nearest-rank like everywhere else.

all for w = 1 and w = n_H. Because the numerator is constant over a
group, harmonic <= arithmetic gives <alpha_cell> >= alpha_area, and the
rank-based median of alpha_cell is <P_tot>_area over the (mirrored-rank)
median of p_th. Both are checked on the real cells and the script raises
if either fails (check_cell_alpha). A per-PLANE <P_tot>_area variant is
reported as a diagnostic only.

Variant RAW only, both self-gravity settings (mean = fiducial, off).
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.conventions import (  # noqa: E402
    MU,
    M_H,
    PERCENTILE_SCHEME_DEFAULT,
    PROFILE_BIN_PC,
    PROVISIONAL_CUBE_HEADER,
    SELF_GRAVITY_MODE_FOOTPRINT_MEAN,
    SELF_GRAVITY_MODE_OFF,
    SLAB_CENTERS_PC,
    SLAB_HALF_THICKNESS_PC,
    STATS_BOX_XY_HALF_RANGE_PC,
    STATS_BOX_Z_HALF_RANGE_PC,
    THERMAL_PRESSURE_HEADER,
    XY_HALF_RANGE_PC,
)
from src.conventions import MIDPLANE_MODE, MIDPLANE_MODE_BALANCED  # noqa: E402
from src.physics import gravity, hydrostatic, midplane  # noqa: E402
from src.physics.loading import open_zarr, slab_z_indices  # noqa: E402
from src.physics.stats import percentile_levels, weighted_stats  # noqa: E402

ALPHA_CORE_ZARR_PATH = Path("cache/core/alpha_core.zarr")
SUMMARY_NPZ_PATH = Path("cache/core/summary.npz")
NUMBERS_TABLE_CSV = Path("results/numbers_table.csv")
OUT_NPZ_PATH = Path("cache/core/area_alpha.npz")
OUT_TXT_PATH = Path("results/area_alpha.txt")
REFERENCE_CSV = Path("results/reference/ok22_tigress_R8.csv")

VARIANT = "RAW"
SELF_GRAVITY_SETTINGS = ("mean", "off")
SELF_GRAVITY_MODE_BY_SETTING = {
    "mean": SELF_GRAVITY_MODE_FOOTPRINT_MEAN,
    "off": SELF_GRAVITY_MODE_OFF,
}
# N = 1 is the whole footprint. It is computed through the patch machinery
# as well as through the independent plain-mean path, which is what
# tests/test_area_alpha.py uses to check the two against each other.
PATCH_NS = (1, 2, 4, 8)
SCALE_TEST_NS = (2, 4, 8)  # the ones reported as a spread

# z planes read per pass. The cache is chunked (32, 501, 501), so this is
# a whole number of chunks and nothing is decompressed twice.
Z_CHUNK = 64

# Agreement required between the plane means of the cached per-column
# P_tot and the integral of the footprint-mean density. The cache stores
# P_tot as float32, so the floor is that storage precision averaged over
# ~2.5e5 columns -- measured at 6e-8 here, not 1e-15. 1e-6 leaves two
# orders of headroom above that floor while still being tight enough to
# matter: reconstructing g(z) against the cache's FLOAT32 z_pc copy
# instead of the source grid build_stage uses (see Z_PC_NOTE) shows up as
# 7e-6 and is caught, and a genuinely wrong g or boundary convention
# would be >= 1e-2.
VALIDATION_MAX_REL_DIFF = 1e-6

# Why the z grid comes from the SOURCE zarr and not from the cache.
#
# alpha_core.zarr stores z_pc as float32 (pipeline/compute_all.py writes
# `z_pc.astype(np.float32)`), but build_stage integrates against
# `grid.z_pc`, the source cube's float64 grid. Those two agree to 2e-13
# pc, which sounds irrelevant and is not: the float64 grid is
# antisymmetric only to 2e-13, so inside
# gravity._enclosed_column_from_profile the `searchsorted(z_sorted,
# |z|)` lookup lands one index off at 42 of the 751 planes, moving
# Sigma_gas(|z'|<|z|) by up to 7e-4 there. That propagates through g_gas
# into P_tot as a smooth ~1e-6 relative bias at every height.
#
# So reproducing the cached P_tot requires the grid the cache was built
# with, full stop. This is NOT a bug being worked around -- changing the
# grid would change P_tot, which is out of scope here, and the
# difference is far below any physical significance. It is recorded
# because it is exactly the kind of thing that makes a validation read
# "1e-6, good enough" when it should read "1e-8".
Z_PC_NOTE = ("z grid read from the source zarr (float64), matching "
               "build_stage; the cached z_pc copy is float32")

REPORT_SLAB_LABELS = tuple(f"z{int(zc)}" for zc in SLAB_CENTERS_PC)


# ---------------------------------------------------------------------------
# patches
# ---------------------------------------------------------------------------
def patch_ids(x_pc, y_pc, n_per_side, half_range_pc=XY_HALF_RANGE_PC):
    """(patch_id (Ny, Nx) int, side_pc) for an n x n split of the footprint.

    Edges at exact multiples of 2*half_range/n from -half_range, bins
    left-closed / right-open except the last, which is closed -- the same
    convention profile_bin_edges() uses in z, so the +500 pc edge planes
    are not silently dropped. patch_id = iy * n + ix.
    """
    side = 2.0 * half_range_pc / n_per_side

    def _ids(c):
        k = np.floor((np.asarray(c, dtype=float) + half_range_pc) / side)
        return np.clip(k, 0, n_per_side - 1).astype(np.int64)

    ix = _ids(x_pc)
    iy = _ids(y_pc)
    return (iy[:, None] * n_per_side + ix[None, :]), side


def patch_edges(n_per_side, half_range_pc=XY_HALF_RANGE_PC):
    """The 1-D patch edges, for the record in the .npz."""
    return np.linspace(-half_range_pc, half_range_pc, n_per_side + 1)


# ---------------------------------------------------------------------------
# the six sums every area average is built from
# ---------------------------------------------------------------------------
# Each group (a 4 pc z bin or a 60 pc slab) x each patch accumulates:
#   n_cells     cells in the group/patch, every cell counted
#   sum_ptot    sum of the per-column P_tot over ALL those cells
#   n_neutral   non-HIM cells among them
#   sum_pth     sum of p_th_phys over the NON-HIM cells
#   sum_n       sum of n_H         over the NON-HIM cells
#   sum_n_pth   sum of n_H*p_th_phys over the NON-HIM cells
# from which
#   <P_tot>_area      = sum_ptot / n_cells
#   <P_th>_vol        = sum_pth / n_neutral
#   <P_th>_mw         = sum_n_pth / sum_n
# Sums, not means, so a group's planes can be accumulated across z chunks.
SUM_KEYS = ("n_cells", "sum_ptot", "n_neutral", "sum_pth", "sum_n", "sum_n_pth")


def _empty_sums(shape):
    return {k: np.zeros(shape, dtype=np.float64) for k in SUM_KEYS}


def _accumulate_plane_full(acc, gids, ptot, pth, n_H, neutral):
    """The whole-footprint path: plain np.sum over the plane.

    Deliberately NOT the bincount path below, so the N = 1 patch result
    and this one are two independent computations of the same number and
    the test comparing them is a real constraint.
    """
    vals = {
        "n_cells": float(ptot.size),
        "sum_ptot": float(ptot.sum(dtype=np.float64)),
        "n_neutral": float(neutral.sum(dtype=np.int64)),
        "sum_pth": float(pth[neutral].sum(dtype=np.float64)),
        "sum_n": float(n_H[neutral].sum(dtype=np.float64)),
        "sum_n_pth": float((n_H[neutral] * pth[neutral]).sum(dtype=np.float64)),
    }
    for g in gids:
        for k, v in vals.items():
            acc[k][g] += v


def _accumulate_plane_patches(acc, gids, pid_flat, n_patches, cnt_pid,
                                ptot, pth, n_H, neutral):
    """The per-patch path: one np.bincount per sum, over the patch ids."""
    w_pth = np.where(neutral, pth, 0.0)
    w_n = np.where(neutral, n_H, 0.0)
    per_patch = {
        "n_cells": cnt_pid,
        "sum_ptot": np.bincount(pid_flat, weights=ptot, minlength=n_patches),
        "n_neutral": np.bincount(pid_flat, weights=neutral.astype(np.float64),
                                   minlength=n_patches),
        "sum_pth": np.bincount(pid_flat, weights=w_pth, minlength=n_patches),
        "sum_n": np.bincount(pid_flat, weights=w_n, minlength=n_patches),
        "sum_n_pth": np.bincount(pid_flat, weights=w_n * w_pth, minlength=n_patches),
    }
    for g in gids:
        for k, v in per_patch.items():
            acc[k][g] += v


def area_averages_from_sums(s):
    """The three area averages and the two alphas, from the six sums.

    Works elementwise on arrays of any shape, so it serves the whole
    footprint, the per-patch arrays and the tests' toy cubes alike.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        p_tot_area = np.where(s["n_cells"] > 0, s["sum_ptot"] / s["n_cells"], np.nan)
        pth_vol = np.where(s["n_neutral"] > 0, s["sum_pth"] / s["n_neutral"], np.nan)
        pth_mw = np.where(s["sum_n"] > 0, s["sum_n_pth"] / s["sum_n"], np.nan)
        return {
            "Ptot_area": p_tot_area,
            "Pth_neutral_vol": pth_vol,
            "Pth_neutral_mw": pth_mw,
            "alpha_area_vol": p_tot_area / pth_vol,
            "alpha_area_mw": p_tot_area / pth_mw,
        }


def across_patch_spread(values, scheme=PERCENTILE_SCHEME_DEFAULT):
    """(median, p_lo, p_hi, n_finite) across the patches of one group.

    Uses the project's own nearest-rank weighted estimator with w = 1 per
    finite patch, so a patch spread and a cell percentile elsewhere in
    the project come from the same formula. Patches with no neutral cells
    (which happens at large |z|, where ~97% of the volume is flagged)
    contribute NaN and are dropped by weighted_stats' finite mask.
    """
    v = np.asarray(values, dtype=float).ravel()
    st = weighted_stats(v, np.ones_like(v), scheme)
    return st.median, st.p_lo, st.p_hi, st.n


# ---------------------------------------------------------------------------
# Step 2e: the per-cell alpha against the area-averaged P_tot
# ---------------------------------------------------------------------------
# alpha_cell = <P_tot>_area / p_th_phys for every non-HIM cell of a group,
# with <P_tot>_area the group's own (one number per bin or slab, as
# reported above). Because the numerator is a constant over the group,
# every statistic of alpha_cell is a statistic of 1/p_th scaled by it:
#   <alpha_cell>      = <P_tot>_area * <1/p_th>_w = <P_tot>_area / P_th,harm
#   median(alpha_cell) = <P_tot>_area / median(p_th)   (rank-based: 1/x is
#                        monotone decreasing, so ranks reverse)
# Both are verified on the real cells in check_cell_alpha() rather than
# used as shortcuts: alpha_cell is formed explicitly and reduced with the
# project's own weighted_stats().
CELL_ALPHA_KEYS = ("Pth_harm", "Pth_median", "alpha_cell_mean",
                   "alpha_cell_mean_direct", "alpha_cell_median",
                   "alpha_cell_p_lo", "alpha_cell_p_hi", "n_cells_used")

# Required agreement between <P_tot>_area * <1/p_th> and the direct weighted
# mean of the formed alpha_cell array. Pure float64 round-off over <= 1e7
# terms; a real discrepancy would be O(1).
CELL_ALPHA_MEAN_RTOL = 1e-10
# Required agreement between median(alpha_cell) and <P_tot>_area divided by
# the order statistic of p_th the nearest-rank median maps onto (one float64
# division and its inverse).
CELL_ALPHA_MEDIAN_RTOL = 1e-12


def cell_alpha_stats(p_th, w, p_tot_area, scheme=PERCENTILE_SCHEME_DEFAULT):
    """The Step 2e statistics of alpha_cell = p_tot_area / p_th over one group.

    p_th and w are the group's NON-HIM cells (the caller selects them);
    p_tot_area is the group's scalar <P_tot>_area. Cells with non-finite
    or non-positive p_th or w are dropped, as weighted_stats() does.

    Returns a dict over CELL_ALPHA_KEYS:
      Pth_harm               [sum(w/p_th)/sum(w)]^-1
      Pth_median             weighted median of p_th
      alpha_cell_mean        p_tot_area / Pth_harm
      alpha_cell_mean_direct weighted mean of the formed alpha_cell array --
                             the same number by a different route, kept so
                             the identity can be checked
      alpha_cell_median, _p_lo, _p_hi   weighted median/percentiles of alpha_cell
      n_cells_used
    """
    p = np.asarray(p_th, dtype=float).ravel()
    ww = np.asarray(w, dtype=float).ravel()
    m = np.isfinite(p) & np.isfinite(ww) & (ww > 0) & (p > 0)
    if not m.any() or not np.isfinite(p_tot_area):
        return {k: (0 if k == "n_cells_used" else np.nan) for k in CELL_ALPHA_KEYS}
    p, ww = p[m], ww[m]
    pth_harm = 1.0 / (np.sum(ww / p) / np.sum(ww))
    st_a = weighted_stats(p_tot_area / p, ww, scheme)
    st_p = weighted_stats(p, ww, scheme)
    return {
        "Pth_harm": float(pth_harm),
        "Pth_median": st_p.median,
        "alpha_cell_mean": float(p_tot_area / pth_harm),
        "alpha_cell_mean_direct": st_a.arithmetic_mean,
        "alpha_cell_median": st_a.median,
        "alpha_cell_p_lo": st_a.p_lo,
        "alpha_cell_p_hi": st_a.p_hi,
        "n_cells_used": st_a.n,
    }


def median_identity_residual(p_th, p_tot_area, alpha_median):
    """How far the volume-weighted median(alpha_cell) is from p_tot_area/median(p_th).

    Returns (rel_diff_vs_median, rel_diff_vs_mapped, n). The nearest-rank
    estimator with w = 1 picks sorted index i = ceil(n/2) - 1. Reversing
    the order (1/x) maps that onto p_th's sorted index n - 1 - i, which
    IS i for odd n and is i + 1 for even n -- one order statistic above
    the p_th median (README, Step 1e). So:
      rel_diff_vs_mapped  against p_th[n-1-i]: must be round-off for any n
      rel_diff_vs_median  against p_th's own median: round-off for odd n,
                          the gap to the neighbouring order statistic for
                          even n (reported, not required to vanish)
    """
    p = np.asarray(p_th, dtype=float).ravel()
    p = np.sort(p[np.isfinite(p) & (p > 0)])
    n = p.size
    if n == 0:
        return np.nan, np.nan, 0
    i = int(np.ceil(0.5 * n)) - 1
    rel_med = abs(alpha_median - p_tot_area / p[i]) / alpha_median
    rel_map = abs(alpha_median - p_tot_area / p[n - 1 - i]) / alpha_median
    return float(rel_med), float(rel_map), n


def check_cell_alpha(stats_by_key, alpha_area_by_key, median_residuals):
    """The two in-script checks of Step 2e (plus the mean identity). Raises.

    stats_by_key       {(sg, weight): {CELL_ALPHA_KEY: (n_groups,) array}}
    alpha_area_by_key  {(sg, weight): (n_groups,) alpha_area}
    median_residuals   {sg: (n_groups, 2) [rel_vs_median, rel_vs_mapped]} (vol)
    Returns a summary dict for the report.
    """
    summary = {}
    for key, st in stats_by_key.items():
        a_mean, a_direct = st["alpha_cell_mean"], st["alpha_cell_mean_direct"]
        fin = np.isfinite(a_mean) & np.isfinite(a_direct)
        rel = np.abs(a_mean[fin] - a_direct[fin]) / a_mean[fin]
        max_rel = float(rel.max()) if rel.size else np.nan
        if rel.size and max_rel > CELL_ALPHA_MEAN_RTOL:
            raise RuntimeError(
                f"<alpha_cell> identity FAILED for {key}: <P_tot>_area/P_th,harm and "
                f"the direct mean of alpha_cell differ by {max_rel:.3e} "
                f"(> {CELL_ALPHA_MEAN_RTOL:.0e}). STOPPING.")
        a_area = alpha_area_by_key[key]
        both = np.isfinite(a_mean) & np.isfinite(a_area)
        # harmonic <= arithmetic, so <alpha_cell> >= alpha_area; the 1e-12
        # allows for the two being formed by different float64 sums when
        # p_th is uniform and they are mathematically equal
        ratio = a_mean[both] / a_area[both]
        if ratio.size and (ratio < 1.0 - 1e-12).any():
            bad = int(np.sum(ratio < 1.0 - 1e-12))
            raise RuntimeError(
                f"<alpha_cell> >= alpha_area FAILED for {key} in {bad} groups "
                f"(min ratio {ratio.min():.6f}). Harmonic mean exceeded the "
                "arithmetic mean -- impossible for positive p_th, so the two "
                "were not formed over the same cells. STOPPING.")
        summary[key] = {"mean_identity_max_rel": max_rel,
                        "min_ratio_mean_to_area": float(ratio.min()) if ratio.size else np.nan,
                        "n_groups": int(both.sum())}
    for sg_key, res in median_residuals.items():
        fin = np.isfinite(res[:, 1])
        max_map = float(res[fin, 1].max()) if fin.any() else np.nan
        if fin.any() and max_map > CELL_ALPHA_MEDIAN_RTOL:
            raise RuntimeError(
                f"median(alpha_cell) identity FAILED for self_gravity={sg_key}: "
                f"median(alpha_cell) and <P_tot>_area / p_th at the mirrored rank "
                f"differ by {max_map:.3e} (> {CELL_ALPHA_MEDIAN_RTOL:.0e}). STOPPING.")
        summary[("median", sg_key)] = {
            "max_rel_vs_mapped": max_map,
            "max_rel_vs_median": float(np.nanmax(res[:, 0])),
            "n_groups": int(fin.sum())}
    return summary


# ---------------------------------------------------------------------------
# the existing cell-based numbers, for comparison
# ---------------------------------------------------------------------------
def cell_based_profiles(sg_key, selection="exclude_him_flag"):
    """The existing mass-weighted cell-based alphas on the same 4 pc bins.

    Read, not recomputed: these are exactly the numbers the numbers table
    and every earlier figure report, so the comparison in the report is
    against the project's own published values.
    """
    d = np.load(SUMMARY_NPZ_PATH, allow_pickle=False)
    stem = f"{VARIANT}__sg{sg_key}__sel{selection}__profile_signedz"
    out = {
        "z_pc": np.asarray(d[f"{stem}__z_pc"], dtype=float),
        "alpha_ratio_of_means_mw": np.asarray(
            d[f"{stem}__alpha_ratio_of_means_mw"], dtype=float),
        "alpha_median_of_ratios_mw": np.asarray(
            d[f"{stem}__alpha_median_of_ratios_mw"], dtype=float),
    }
    d.close()
    return out


def cell_based_slabs(sg_key, selection="exclude_him_flag"):
    """The same two estimators at the three slabs, from numbers_table.csv."""
    wanted = {"alpha_ratio_of_means", "alpha_median_of_ratios"}
    out = {}
    with open(NUMBERS_TABLE_CSV, newline="") as f:
        rows = csv.DictReader([ln for ln in f if not ln.startswith("#")])
        for r in rows:
            if (r["variant"] == VARIANT and r["self_gravity"] == sg_key
                    and r["cell_selection"] == selection
                    and r["quantity"] in wanted and r["weighting"] == "mw"
                    and r["stat"] in REPORT_SLAB_LABELS):
                out[(r["quantity"], r["stat"])] = float(r["value"])
    missing = [(q, s) for q in sorted(wanted) for s in REPORT_SLAB_LABELS
               if (q, s) not in out]
    if missing:
        raise RuntimeError(
            f"numbers_table.csv has no {missing} rows for self_gravity={sg_key} -- "
            "the cell-based comparison column cannot be filled.")
    return out


def load_reference(path=REFERENCE_CSV):
    """The TIGRESS R8 values, as {quantity: value}."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader([ln for ln in f if not ln.startswith("#")]))
    return {r["quantity"]: float(r["value"]) for r in rows}


# ---------------------------------------------------------------------------
# g(z), and the footprint-mean density profile it needs
# ---------------------------------------------------------------------------
def footprint_mean_density_chunked(n_arr, z_chunk=Z_CHUNK):
    """<n_H>_area(z) [cm^-3] over the FULL z column, read in z chunks.

    The cache's footprint is already the +-500 pc square, so the mean is
    over every cell of each plane. Streamed because the float64 cube is
    1.5 GB and this machine does not have it to spare; the per-plane mean
    is a per-row reduction either way, so chunking changes nothing but
    peak memory.
    """
    nz = n_arr.shape[0]
    rho_bar = np.empty(nz, dtype=np.float64)
    for lo in range(0, nz, z_chunk):
        hi = min(lo + z_chunk, nz)
        chunk = np.asarray(n_arr[lo:hi], dtype=np.float64).reshape(hi - lo, -1)
        rho_bar[lo:hi] = chunk.mean(axis=1)
        del chunk
    return rho_bar


def g_for_setting(sg_key, z_pc_full, n_bar_cm3, midplane_mode=MIDPLANE_MODE):
    """(g [cm/s^2], nan_fraction, z_mid [pc]) on the full z column for one
    self-gravity setting.

    The cache does not store g, so it is re-derived here through
    src.physics.midplane.midplane_gravity -- the same physics the build
    stage uses -- rather than reimplemented. Both fiducial settings depend
    on the density cube ONLY through its footprint mean per z plane, which
    is what n_bar_cm3 is. "sun": z_mid = 0 and g from gravity.g_total_cgs
    on a (Nz, 1, 1) cube of that profile (the pre-Step-1i call). "balanced"
    (Step 1i): z_mid from the profile and g centred on it.
    """
    mode = SELF_GRAVITY_MODE_BY_SETTING[sg_key]
    z_mid, g, nan_fraction = midplane.midplane_gravity(z_pc_full, n_bar_cm3, mode, midplane_mode)
    return np.asarray(g, dtype=np.float64), nan_fraction, z_mid


def plane_mean_ptot_chunked(ptot_arr, z_chunk=Z_CHUNK):
    """<P_tot>_area(z) [K cm^-3] over the FULL z column, from the cache."""
    nz = ptot_arr.shape[0]
    out = np.empty(nz, dtype=np.float64)
    for lo in range(0, nz, z_chunk):
        hi = min(lo + z_chunk, nz)
        chunk = np.asarray(ptot_arr[lo:hi], dtype=np.float64).reshape(hi - lo, -1)
        out[lo:hi] = chunk.mean(axis=1)
        del chunk
    return out


def validate_area_ptot(z_pc_full, n_bar_cm3, g_cgs, ptot_plane_mean, z_split=0.0):
    """Part (b): the area-average identity, checked numerically.

    Returns (predicted, rel_diff, max_rel_diff). `predicted` is the
    hydrostatic integral of the footprint-mean density against the same
    g, through the same integrator and the same P = 0 boundary
    convention the cache was built with.
    """
    rho_bar_cgs = MU * M_H * n_bar_cm3
    predicted = hydrostatic.p_tot_kb_full_column(z_pc_full, rho_bar_cgs, g_cgs, z_split=z_split)
    predicted = np.asarray(predicted, dtype=np.float64)
    denom = np.where(np.abs(ptot_plane_mean) > 0, np.abs(ptot_plane_mean), np.nan)
    rel = np.abs(predicted - ptot_plane_mean) / denom
    return predicted, rel, float(np.nanmax(rel))


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    OUT_NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pct_lo, pct_hi = percentile_levels(PERCENTILE_SCHEME_DEFAULT)
    ref = load_reference()

    import pipeline.compute_all as pipe
    core = zarr.open_group(str(ALPHA_CORE_ZARR_PATH), mode="r")
    x_pc = np.asarray(core["x_pc"][:], dtype=np.float64)
    y_pc = np.asarray(core["y_pc"][:], dtype=np.float64)
    # The source grid, not the cache's float32 copy of it -- see Z_PC_NOTE.
    z_full = np.asarray(open_zarr().z_pc, dtype=np.float64)
    z_cached = np.asarray(core["z_pc"][:], dtype=np.float64)
    if z_full.shape != z_cached.shape or not np.allclose(z_full, z_cached, atol=1e-6):
        raise RuntimeError(
            "The source zarr's z grid does not match the one the cache was built "
            "from -- every cached P_tot would be integrated against a different "
            "column than the one read here.")
    zsl, z_box = pipe._box_z_slice(z_full)
    vg = core[VARIANT]

    print(f"Variant {VARIANT}; cube {vg['n_model'].shape}; "
          f"footprint |x|,|y| <= {STATS_BOX_XY_HALF_RANGE_PC:.0f} pc, "
          f"box |z| <= {STATS_BOX_Z_HALF_RANGE_PC:.0f} pc "
          f"({len(z_box)} planes of {len(z_full)})")

    # ---- groups: the 4 pc signed bins, then the three 60 pc slabs ---------
    bin_groups = pipe.profile_bin_groups(z_box, use_abs=False)
    n_bins = len(bin_groups)
    # Step 1i: the midplane slab follows z_mid (per self-gravity setting, as
    # built and recorded in the cache). Both settings' z_mid select the same
    # planes -- checked, so one set of slab groups serves both.
    z_mid_cache = {k: float(vg[f"self_gravity_{k}"].attrs.get("z_mid_pc", 0.0))
                   for k in SELF_GRAVITY_SETTINGS}
    built_modes = {vg[f"self_gravity_{k}"].attrs.get("midplane_mode", "sun")
                   for k in SELF_GRAVITY_SETTINGS}
    if built_modes != {MIDPLANE_MODE}:
        raise RuntimeError(f"cache built with midplane {built_modes}, MIDPLANE_MODE is "
                           f"{MIDPLANE_MODE!r} -- rebuild first")
    slab_by_sg = {k: [slab_z_indices(z_box, midplane.slab_center(zc, z_mid_cache[k], MIDPLANE_MODE))
                      for zc in SLAB_CENTERS_PC] for k in SELF_GRAVITY_SETTINGS}
    if not all(np.array_equal(a, b) for a, b in zip(*slab_by_sg.values())):
        raise RuntimeError("the self-gravity settings' z_mid select different slab planes")
    slab_groups = [(float(zc), idx) for zc, idx in zip(SLAB_CENTERS_PC, slab_by_sg["mean"])]
    print(f"  midplane {MIDPLANE_MODE}: z_mid = "
          + ", ".join(f"{k} {v:+.4f}" for k, v in z_mid_cache.items()) + " pc; midplane slab "
          f"z = {z_box[slab_groups[0][1]].min():g}..{z_box[slab_groups[0][1]].max():g} pc")
    n_groups = n_bins + len(slab_groups)
    # A box plane can belong to one bin and (independently) to one slab.
    plane_groups = [[] for _ in range(len(z_box))]
    for b, (_, idxs) in enumerate(bin_groups):
        for i in idxs:
            plane_groups[int(i)].append(b)
    for s, (_, idxs) in enumerate(slab_groups):
        for i in idxs:
            plane_groups[int(i)].append(n_bins + s)
    print(f"  {n_bins} signed {PROFILE_BIN_PC:.0f} pc bins + "
          f"{len(slab_groups)} slabs (+-{SLAB_HALF_THICKNESS_PC:.0f} pc) "
          f"at z = {[int(c) for c, _ in slab_groups]}")

    # ---- patch id maps ---------------------------------------------------
    pid_by_n, side_by_n, cnt_by_n = {}, {}, {}
    for N in PATCH_NS:
        pid, side = patch_ids(x_pc, y_pc, N)
        pid_by_n[N] = pid.ravel()
        side_by_n[N] = side
        cnt_by_n[N] = np.bincount(pid_by_n[N], minlength=N * N).astype(np.float64)
        print(f"  N = {N}: {N * N} patches of {side:.0f} pc, "
              f"{int(cnt_by_n[N].min())}-{int(cnt_by_n[N].max())} cells each")

    out = {
        "z_pc": np.array([c for c, _ in bin_groups], dtype=float),
        "bin_edges": pipe.profile_bin_edges(use_abs=False),
        "bin_width_pc": np.array([PROFILE_BIN_PC], dtype=float),
        "slab_centers_pc": np.array([c for c, _ in slab_groups], dtype=float),
        "slab_half_thickness_pc": np.array([SLAB_HALF_THICKNESS_PC], dtype=float),
        "slab_labels": np.array(REPORT_SLAB_LABELS),
        "percentile_levels": np.array([pct_lo, pct_hi], dtype=float),
        "patch_ns": np.array(PATCH_NS, dtype=int),
        "scale_test_ns": np.array(SCALE_TEST_NS, dtype=int),
        "patch_sides_pc": np.array([side_by_n[N] for N in PATCH_NS], dtype=float),
        "variant": np.array([VARIANT]),
        "self_gravity_settings": np.array(list(SELF_GRAVITY_SETTINGS)),
        "z_pc_full": z_full,
        "provisional": np.array([PROVISIONAL_CUBE_HEADER]),
        "thermal_pressure_note": np.array([THERMAL_PRESSURE_HEADER]),
        "z_pc_note": np.array([Z_PC_NOTE]),
        "validation_max_rel_diff_threshold": np.array([VALIDATION_MAX_REL_DIFF]),
    }
    for N in PATCH_NS:
        out[f"patch{N}__edges_pc"] = patch_edges(N)
        out[f"patch{N}__n_cells_per_patch"] = cnt_by_n[N]

    # ---- the footprint-mean density profile (self-gravity source) --------
    print("\nFootprint-mean density profile over the full z column...")
    n_bar = footprint_mean_density_chunked(vg["n_model"])
    out["n_bar_cm3_full"] = n_bar
    print(f"  <n_H>_area: midplane {n_bar[np.argmin(np.abs(z_full))]:.4f}, "
          f"max {n_bar.max():.4f} cm^-3")

    him_arr = vg["him"]
    pth_arr = vg["p_th_phys"]
    n_arr = vg["n_model"]
    validation = {}

    for sg_key in SELF_GRAVITY_SETTINGS:
        print(f"\n=== self-gravity {sg_key} ===")
        ptot_arr = vg[f"self_gravity_{sg_key}"]["Ptot"]

        # ---- (a) + (b): the plane means, and the identity check ----------
        g_cgs, nan_fraction, z_mid = g_for_setting(sg_key, z_full, n_bar)
        if abs(z_mid - z_mid_cache[sg_key]) > 1e-6:
            raise RuntimeError(f"z_mid re-derived here ({z_mid:+.6f}) != the cache's "
                               f"({z_mid_cache[sg_key]:+.6f}) for self_gravity={sg_key}")
        out[f"{sg_key}__z_mid_pc"] = np.array([z_mid])
        out[f"{sg_key}__g_cgs_full"] = g_cgs
        if nan_fraction is not None:
            out[f"{sg_key}__footprint_nan_fraction_full"] = np.asarray(
                nan_fraction, dtype=float)
        print(f"  g(z): midplane {g_cgs[np.argmin(np.abs(z_full))]:.4e}, "
              f"edge {g_cgs[-1]:.4e} cm/s^2")

        ptot_plane_mean = plane_mean_ptot_chunked(ptot_arr)
        predicted, rel, max_rel = validate_area_ptot(
            z_full, n_bar, g_cgs, ptot_plane_mean, z_split=z_mid)
        if MIDPLANE_MODE == MIDPLANE_MODE_BALANCED:
            top, bot = midplane.branch_pressures(z_full, MU * M_H * n_bar, g_cgs, z_mid)
            out[f"{sg_key}__P_at_z_mid_branches"] = np.array([top, bot])
            print(f"  P at z_mid: top-down {top:.3f}, bottom-up {bot:.3f} (rel {abs(top / bot - 1):.1e})")
        out[f"{sg_key}__ptot_plane_mean_full"] = ptot_plane_mean
        out[f"{sg_key}__ptot_from_rho_bar_full"] = predicted
        out[f"{sg_key}__validation_rel_diff_full"] = rel
        out[f"{sg_key}__validation_max_rel_diff"] = np.array([max_rel])
        in_box = np.abs(z_full) <= STATS_BOX_Z_HALF_RANGE_PC
        max_rel_box = float(np.nanmax(rel[in_box]))
        out[f"{sg_key}__validation_max_rel_diff_box"] = np.array([max_rel_box])
        validation[sg_key] = (max_rel, max_rel_box)
        print(f"  validation: max |rel diff| = {max_rel:.3e} over the full "
              f"column, {max_rel_box:.3e} inside the box")
        if not np.isfinite(max_rel) or max_rel > VALIDATION_MAX_REL_DIFF:
            raise RuntimeError(
                f"Area-average identity FAILED for self_gravity={sg_key}: max "
                f"relative difference {max_rel:.3e} exceeds "
                f"{VALIDATION_MAX_REL_DIFF:.0e}. <P_tot>_area (the plane mean of "
                "the cached per-column P_tot) and the hydrostatic integral of "
                "the footprint-mean density must agree to numerical precision, "
                "because g(z) is the same for every column and P_tot is linear "
                "in rho. A real discrepancy means the cached P_tot was not "
                "integrated against this g, or not with this boundary "
                "convention -- not a precision issue. STOPPING.")

        # ---- (c) + (d): per-group area averages, whole footprint and -----
        #                 per patch, accumulated over z chunks
        acc_full = _empty_sums((n_groups,))
        acc_patch = {N: _empty_sums((n_groups, N * N)) for N in PATCH_NS}
        print(f"  reducing {len(z_box)} box planes "
              f"({(len(z_box) + Z_CHUNK - 1) // Z_CHUNK} chunks)...")
        for lo in range(0, len(z_box), Z_CHUNK):
            hi = min(lo + Z_CHUNK, len(z_box))
            sl = slice(zsl.start + lo, zsl.start + hi)
            ptot_c = np.asarray(ptot_arr[sl], dtype=np.float64)
            pth_c = np.asarray(pth_arr[sl], dtype=np.float64)
            n_c = np.asarray(n_arr[sl], dtype=np.float64)
            him_c = np.asarray(him_arr[sl]).astype(bool)
            for k in range(hi - lo):
                gids = plane_groups[lo + k]
                if not gids:
                    continue
                ptot_p = ptot_c[k].ravel()
                pth_p = pth_c[k].ravel()
                n_p = n_c[k].ravel()
                neutral_p = ~him_c[k].ravel()
                _accumulate_plane_full(acc_full, gids, ptot_p, pth_p, n_p, neutral_p)
                for N in PATCH_NS:
                    _accumulate_plane_patches(
                        acc_patch[N], gids, pid_by_n[N], N * N, cnt_by_n[N],
                        ptot_p, pth_p, n_p, neutral_p)
            del ptot_c, pth_c, n_c, him_c

        full = area_averages_from_sums(acc_full)
        for key, arr in full.items():
            out[f"{sg_key}__{key}"] = arr[:n_bins]
            out[f"slab__{sg_key}__{key}"] = arr[n_bins:]
        out[f"{sg_key}__n_cells"] = acc_full["n_cells"][:n_bins]
        out[f"{sg_key}__n_neutral"] = acc_full["n_neutral"][:n_bins]
        out[f"slab__{sg_key}__n_cells"] = acc_full["n_cells"][n_bins:]
        out[f"slab__{sg_key}__n_neutral"] = acc_full["n_neutral"][n_bins:]

        for N in PATCH_NS:
            per_patch = area_averages_from_sums(acc_patch[N])
            for key, arr in per_patch.items():
                out[f"{sg_key}__patch{N}__{key}"] = arr[:n_bins]
                out[f"slab__{sg_key}__patch{N}__{key}"] = arr[n_bins:]
            # the spread across patches, per group
            for key in ("alpha_area_vol", "alpha_area_mw", "Ptot_area",
                        "Pth_neutral_vol", "Pth_neutral_mw"):
                med = np.empty(n_groups)
                plo = np.empty(n_groups)
                phi = np.empty(n_groups)
                nfin = np.zeros(n_groups, dtype=np.int64)
                for g in range(n_groups):
                    med[g], plo[g], phi[g], nfin[g] = across_patch_spread(
                        per_patch[key][g])
                for stat, arr in (("median", med), ("p_lo", plo),
                                    ("p_hi", phi), ("n_finite", nfin)):
                    out[f"{sg_key}__patch{N}__{key}_{stat}"] = arr[:n_bins]
                    out[f"slab__{sg_key}__patch{N}__{key}_{stat}"] = arr[n_bins:]
            print(f"    N = {N}: alpha_area_vol across patches at z = 0 -> "
                  f"median {out[f'slab__{sg_key}__patch{N}__alpha_area_vol_median'][0]:.3f}, "
                  f"{pct_lo:g}-{pct_hi:g} "
                  f"[{out[f'slab__{sg_key}__patch{N}__alpha_area_vol_p_lo'][0]:.3f}, "
                  f"{out[f'slab__{sg_key}__patch{N}__alpha_area_vol_p_hi'][0]:.3f}]")

        # ---- (e) the existing cell-based numbers, read not recomputed -----
        cb = cell_based_profiles(sg_key)
        if not np.allclose(cb["z_pc"], out["z_pc"]):
            raise RuntimeError(
                "The cached cell-based profile is on a different z grid than the "
                "bins computed here -- the comparison would be off by a bin.")
        out[f"{sg_key}__cellbased_alpha_ratio_of_means_mw"] = cb["alpha_ratio_of_means_mw"]
        out[f"{sg_key}__cellbased_alpha_median_of_ratios_mw"] = cb["alpha_median_of_ratios_mw"]
        cbs = cell_based_slabs(sg_key)
        for qty in ("alpha_ratio_of_means", "alpha_median_of_ratios"):
            out[f"slab__{sg_key}__cellbased_{qty}_mw"] = np.array(
                [cbs[(qty, s)] for s in REPORT_SLAB_LABELS], dtype=float)

        print(f"  z = 0 slab: <P_tot>_area "
              f"{out[f'slab__{sg_key}__Ptot_area'][0]:.1f}, <P_th>_neutral vol "
              f"{out[f'slab__{sg_key}__Pth_neutral_vol'][0]:.1f} / mw "
              f"{out[f'slab__{sg_key}__Pth_neutral_mw'][0]:.1f} K cm^-3 -> "
              f"alpha_area vol {out[f'slab__{sg_key}__alpha_area_vol'][0]:.3f} / mw "
              f"{out[f'slab__{sg_key}__alpha_area_mw'][0]:.3f}")

    # ---- Step 2e: per-cell alpha against <P_tot>_area -------------------
    # Needs every non-HIM cell of a group at once (medians), so the box is
    # read whole as float32 -- ~0.9 GB, as pipeline finalize already does.
    print("\n=== Step 2e: per-cell alpha_cell = <P_tot>_area / p_th_phys ===")
    pth_box = np.asarray(pth_arr[zsl], dtype=np.float32).reshape(len(z_box), -1)
    n_box = np.asarray(n_arr[zsl], dtype=np.float32).reshape(len(z_box), -1)
    him_box = np.asarray(him_arr[zsl]).astype(bool).reshape(len(z_box), -1)
    all_groups = [idxs for _, idxs in bin_groups] + [idxs for _, idxs in slab_groups]
    weights = ("vol", "mw")
    cell_stats = {(sg, wt): {k: np.full(n_groups, np.nan) for k in CELL_ALPHA_KEYS}
                  for sg in SELF_GRAVITY_SETTINGS for wt in weights}
    # diagnostic: alpha_cell with the PER-PLANE <P_tot>_area instead of the
    # group's one number (the two differ only through P_tot's variation
    # across the group's planes)
    plane_stats = {(sg, wt): {k: np.full(n_groups, np.nan)
                              for k in ("median", "mean")}
                   for sg in SELF_GRAVITY_SETTINGS for wt in weights}
    median_res = {sg: np.full((n_groups, 2), np.nan) for sg in SELF_GRAVITY_SETTINGS}
    ptot_group = {sg: np.concatenate([out[f"{sg}__Ptot_area"],
                                       out[f"slab__{sg}__Ptot_area"]])
                  for sg in SELF_GRAVITY_SETTINGS}
    ptot_plane_box = {sg: out[f"{sg}__ptot_plane_mean_full"][zsl]
                      for sg in SELF_GRAVITY_SETTINGS}
    for g, idxs in enumerate(all_groups):
        idxs = np.asarray(idxs, dtype=int)
        if idxs.size == 0:
            continue
        neutral = ~him_box[idxs]
        p_g = pth_box[idxs][neutral].astype(np.float64)
        n_g = n_box[idxs][neutral].astype(np.float64)
        plane_of_cell = np.broadcast_to(idxs[:, None], neutral.shape)[neutral]
        w_by = {"vol": np.ones_like(p_g), "mw": n_g}
        for sg in SELF_GRAVITY_SETTINGS:
            P = float(ptot_group[sg][g])
            p_plane = ptot_plane_box[sg][plane_of_cell]
            for wt in weights:
                st = cell_alpha_stats(p_g, w_by[wt], P)
                for k, v in st.items():
                    cell_stats[(sg, wt)][k][g] = v
                ps = weighted_stats(p_plane / p_g, w_by[wt])
                plane_stats[(sg, wt)]["median"][g] = ps.median
                plane_stats[(sg, wt)]["mean"][g] = ps.arithmetic_mean
            r_med, r_map, _ = median_identity_residual(
                p_g, P, cell_stats[(sg, "vol")]["alpha_cell_median"][g])
            median_res[sg][g] = (r_med, r_map)
    del pth_box, n_box, him_box

    alpha_area_by_key = {
        (sg, wt): np.concatenate([out[f"{sg}__alpha_area_{wt}"],
                                  out[f"slab__{sg}__alpha_area_{wt}"]])
        for sg in SELF_GRAVITY_SETTINGS for wt in weights}
    cell_checks = check_cell_alpha(cell_stats, alpha_area_by_key, median_res)
    for (sg, wt), st in cell_stats.items():
        for k, arr in st.items():
            out[f"{sg}__{k}_{wt}"] = arr[:n_bins]
            out[f"slab__{sg}__{k}_{wt}"] = arr[n_bins:]
        for k, arr in plane_stats[(sg, wt)].items():
            out[f"{sg}__alpha_cell_perplane_{k}_{wt}"] = arr[:n_bins]
            out[f"slab__{sg}__alpha_cell_perplane_{k}_{wt}"] = arr[n_bins:]
    for sg in SELF_GRAVITY_SETTINGS:
        out[f"{sg}__median_identity_rel_vs_median_vol"] = median_res[sg][:n_bins, 0]
        out[f"slab__{sg}__median_identity_rel_vs_median_vol"] = median_res[sg][n_bins:, 0]
        out[f"{sg}__median_identity_rel_vs_mapped_vol"] = median_res[sg][:n_bins, 1]
        out[f"slab__{sg}__median_identity_rel_vs_mapped_vol"] = median_res[sg][n_bins:, 1]
    for (sg, wt), c in ((k, v) for k, v in cell_checks.items() if k[0] != "median"):
        print(f"  {sg:>4} {wt:>3}: <alpha_cell> = <P_tot>/P_th,harm to "
              f"{c['mean_identity_max_rel']:.1e}; min <alpha_cell>/alpha_area "
              f"{c['min_ratio_mean_to_area']:.4f} over {c['n_groups']} groups")
    for sg in SELF_GRAVITY_SETTINGS:
        c = cell_checks[("median", sg)]
        print(f"  {sg:>4} vol: median identity, mirrored rank {c['max_rel_vs_mapped']:.1e}, "
              f"vs p_th median {c['max_rel_vs_median']:.1e}")

    for q, v in ref.items():
        out[f"ref_{q}"] = np.array([v])
    np.savez_compressed(OUT_NPZ_PATH, **out)
    print(f"\nSaved {OUT_NPZ_PATH}")

    write_report(out, validation, ref, pct_lo, pct_hi, t0, cell_checks)


# ---------------------------------------------------------------------------
def cell_alpha_report_lines(out, cell_checks, pct_lo, pct_hi):
    """Section 6 of results/area_alpha.txt: the Step 2e per-cell alpha."""
    L = [
        "-" * 100,
        "6. PER-CELL alpha AGAINST THE AREA-AVERAGED P_tot (Step 2e)",
        "-" * 100,
        "  alpha_cell(x,y,z) = <P_tot>_area / p_th_phys(x,y,z)   for every NON-HIM cell,",
        "  with <P_tot>_area the group's own number from section 2 / 4 (one per 4 pc bin or",
        "  60 pc slab). Over each group's non-HIM cells, w = 1 (vol) and w = n_H (mw):",
        "    P_th,harm      = [sum(w/p_th)/sum(w)]^-1           (harmonic mean)",
        "    <alpha_cell>   = <P_tot>_area / P_th,harm          (= <P_tot>_area * <1/p_th>)",
        f"    median, p{pct_lo:g}, p{pct_hi:g} of alpha_cell   (nearest-rank, as everywhere else)",
        "  alpha_area = <P_tot>_area / <P_th> (arithmetic) is unchanged and shown alongside.",
        "  Since harmonic <= arithmetic, <alpha_cell> >= alpha_area always; they are equal",
        "  only if p_th is uniform over the group.",
        "",
    ]
    for wt, wlabel in (("vol", "VOLUME-weighted (w = 1)"), ("mw", "MASS-weighted (w = n_H)")):
        L += [
            wlabel,
            f"{'z':>6}{'sg':>6}{'<P_tot>_area':>14}{'<P_th> arith':>14}{'P_th,harm':>12}"
            f"{'a_area':>9}{'<a_cell>':>10}{'med a_cell':>12}"
            f"{f'[p{pct_lo:g}, p{pct_hi:g}]':>18}",
            f"{'[pc]':>6}{'':>6}{'[K cm^-3]':>14}{'[K cm^-3]':>14}{'[K cm^-3]':>12}",
            "-" * 101,
        ]
        for i, zc in enumerate(out["slab_centers_pc"]):
            for sg in SELF_GRAVITY_SETTINGS:
                s = f"slab__{sg}__"
                L.append(
                    f"{slab_label(zc):>6}{sg:>6}{out[s + 'Ptot_area'][i]:>14.1f}"
                    f"{out[s + f'Pth_neutral_{wt}'][i]:>14.1f}"
                    f"{out[s + f'Pth_harm_{wt}'][i]:>12.1f}"
                    f"{out[s + f'alpha_area_{wt}'][i]:>9.3f}"
                    f"{out[s + f'alpha_cell_mean_{wt}'][i]:>10.3f}"
                    f"{out[s + f'alpha_cell_median_{wt}'][i]:>12.3f}"
                    f"    [{out[s + f'alpha_cell_p_lo_{wt}'][i]:>5.3f}, "
                    f"{out[s + f'alpha_cell_p_hi_{wt}'][i]:>6.3f}]")
            L.append("")
        L.append("")

    L += [
        "In-script checks (the script raises if any fails):",
        f"  (i)  median(alpha_cell) = <P_tot>_area / median(p_th), vol-weighted, every bin and slab.",
        "       The nearest-rank median with w = 1 sits at sorted index ceil(n/2)-1; reversing",
        "       the order through 1/p_th maps it onto p_th's index n-ceil(n/2), which is the",
        "       p_th median itself for odd n and the next order statistic up for even n. The",
        f"       check requires agreement at that mirrored rank to {CELL_ALPHA_MEDIAN_RTOL:.0e}:",
    ]
    for sg in SELF_GRAVITY_SETTINGS:
        c = cell_checks[("median", sg)]
        L.append(f"         {sg:>4}: max |rel diff| at the mirrored rank {c['max_rel_vs_mapped']:.2e}; "
                 f"against p_th's own median {c['max_rel_vs_median']:.2e}  "
                 f"({c['n_groups']} groups)")
    L += [
        "       (the second number is the one-order-statistic gap for even n).",
        "  (ii) <alpha_cell> >= alpha_area, both weightings, every bin and slab; and",
        "       <P_tot>_area/P_th,harm equals the directly formed mean of alpha_cell to "
        f"{CELL_ALPHA_MEAN_RTOL:.0e}:",
    ]
    for (sg, wt), c in ((k, v) for k, v in cell_checks.items() if k[0] != "median"):
        L.append(f"         {sg:>4} {wt:>3}: min <alpha_cell>/alpha_area = "
                 f"{c['min_ratio_mean_to_area']:.4f}; mean identity {c['mean_identity_max_rel']:.2e}"
                 f"  ({c['n_groups']} groups)")
    L += [
        "",
        "Diagnostic: <P_tot>_area per PLANE instead of per group. Within a 4 pc bin (2 planes)",
        "the two definitions coincide to the plane-to-plane change in <P_tot>_area; across a",
        "60 pc slab <P_tot>_area itself changes, so the per-plane version is a different",
        "(z-mixed) statistic. Self-gravity = mean, volume-weighted:",
        f"{'z':>6}{'med (group P)':>16}{'med (plane P)':>16}{'mean (group P)':>16}{'mean (plane P)':>16}",
    ]
    for i, zc in enumerate(out["slab_centers_pc"]):
        s = "slab__mean__"
        L.append(f"{slab_label(zc):>6}{out[s + 'alpha_cell_median_vol'][i]:>16.3f}"
                 f"{out[s + 'alpha_cell_perplane_median_vol'][i]:>16.3f}"
                 f"{out[s + 'alpha_cell_mean_vol'][i]:>16.3f}"
                 f"{out[s + 'alpha_cell_perplane_mean_vol'][i]:>16.3f}")
    with np.errstate(invalid="ignore"):
        d_bins = np.abs(out["mean__alpha_cell_perplane_median_vol"]
                        / out["mean__alpha_cell_median_vol"] - 1.0)
    L.append(f"  4 pc bins: max |rel diff| of the median between the two = {np.nanmax(d_bins):.2e}")
    L += [
        "",
        "Why <alpha_cell> sits above alpha_area: the mean of 1/p_th is dominated by the",
        "low-pressure neutral cells, so the harmonic mean P_th,harm falls below the",
        "arithmetic <P_th>, by an amount set by the spread of p_th within the group.",
    ]
    return L


def slab_label(zc):
    """'mid' for the midplane slab in balanced mode (it is centred on z_mid,
    not z = 0), else the slab centre in pc."""
    return "mid" if (MIDPLANE_MODE == MIDPLANE_MODE_BALANCED and zc == 0.0) else f"{zc:.0f}"


def write_report(out, validation, ref, pct_lo, pct_hi, t0, cell_checks):
    L = [
        PROVISIONAL_CUBE_HEADER,
        THERMAL_PRESSURE_HEADER,
        "Step 2d: AREA-AVERAGED P_tot and alpha, and the sub-area scale test",
        "Step 2e: the PER-CELL alpha against <P_tot>_area (section 6)",
        "=" * 100,
        "",
        "P_tot is a macroscopic pressure: vertical balance between pressure and weight holds",
        "only on average over a large area (Ostriker & Kim 2022 horizontally average both the",
        "weight and the midplane pressures). Everything else in this project reduces alpha over",
        "CELLS; here the area average is taken FIRST and the ratio SECOND.",
        "",
        f"  <P_tot>_area(z)    plain mean of the per-column P_tot over ALL cells in the plane,",
        "                     HIM-flagged cells INCLUDED (their observed mass is in the",
        "                     hydrostatic weight, so there is nothing to exclude).",
        "  <P_th>_neutral(z)  mean of p_th_phys over NON-HIM cells, volume-weighted (plain",
        "                     mean) and mass-weighted (weights n_H).",
        "  alpha_area(z)      <P_tot>_area / <P_th>_neutral, both weightings.",
        "",
        f"RAW only. Footprint |x|,|y| <= {STATS_BOX_XY_HALF_RANGE_PC:.0f} pc, box |z| <= "
        f"{STATS_BOX_Z_HALF_RANGE_PC:.0f} pc, {PROFILE_BIN_PC:.0f} pc bins, slabs "
        f"+-{SLAB_HALF_THICKNESS_PC:.0f} pc.",
        "Nothing here recomputes P_tot, alpha, p_th_phys or the HIM flag -- all are read from",
        "cache/core/alpha_core.zarr. g(z) is the only quantity re-derived, through the same",
        "src.physics.midplane call the build stage makes, because the cache does not store it.",
        "",
        f"MIDPLANE: {MIDPLANE_MODE}" + (
            f" (Step 1i). z_mid = {float(out['mean__z_mid_pc'][0]):+.3f} pc (self-gravity mean), "
            f"{float(out['off__z_mid_pc'][0]):+.3f} pc (off);"
            if MIDPLANE_MODE == MIDPLANE_MODE_BALANCED else "."),
        *(["  P_tot is integrated toward z_mid from both cube edges with g centred on z_mid. The",
           f"  'mid' slab is |z - z_mid| <= {SLAB_HALF_THICKNESS_PC:.0f} pc; z = 150 and 300 are unchanged.",
           "  P at z_mid, top-down vs bottom-up (area average): "
           + "; ".join(f"{k} {out[f'{k}__P_at_z_mid_branches'][0]:.2f} / "
                       f"{out[f'{k}__P_at_z_mid_branches'][1]:.2f}" for k in SELF_GRAVITY_SETTINGS)]
          if MIDPLANE_MODE == MIDPLANE_MODE_BALANCED else []),
        "",
        "-" * 100,
        "1. VALIDATION of the area-average identity",
        "-" * 100,
        "g(z) is identical for every (x,y) column under both settings and P_tot is linear in",
        "rho, so the plane mean of the per-column P_tot must equal the hydrostatic integral of",
        "the footprint-mean density:  <P_tot>_area(z) = int_z^top <rho>_area g dz'.",
        "The two sides are computed by completely different routes (a mean over 251001",
        "independently integrated columns vs a single integral of one 1-D profile).",
        "",
        f"{'self-gravity':>14}{'max |rel diff| full column':>30}{'max |rel diff| in box':>24}",
        "-" * 68,
    ]
    for sg_key in SELF_GRAVITY_SETTINGS:
        full_rel, box_rel = validation[sg_key]
        L.append(f"{sg_key:>14}{full_rel:>30.3e}{box_rel:>24.3e}")
    L += [
        "-" * 68,
        "Both are at the floor set by the cache storing P_tot as float32 (the threshold that",
        f"would have raised is {VALIDATION_MAX_REL_DIFF:.0e}). The identity holds exactly; what is",
        "being measured here is round-off.",
        "",
        "One detail worth recording, because it is the sort that quietly turns a 1e-8 check into",
        "a 1e-6 one: g(z) must be re-derived against the SOURCE cube's float64 z grid, which is",
        "what build_stage integrates against, not against the float32 copy of it stored in",
        "alpha_core.zarr. The two agree to 2e-13 pc, but the float64 grid is antisymmetric only",
        "to that same 2e-13, which shifts the searchsorted lookup inside the enclosed-column",
        "integral by one index at 42 of the 751 planes and moves g_gas enough to bias P_tot by",
        "~7e-6 everywhere. Using the cached copy raises the 'mean' number above to 7.0e-06 --",
        "still small, still wrong, and caught by the threshold.",
        "",
        "-" * 100,
        "2. AREA AVERAGES AND alpha_area AT THE THREE SLABS",
        "-" * 100,
        "The last column is the EXISTING cell-based mass-weighted ratio of means, read from",
        "results/numbers_table.csv -- the same quantity this project has been reporting.",
        "",
        f"{'z':>6}{'sg':>6}{'<P_tot>_area':>14}{'<P_th> vol':>12}{'<P_th> mw':>12}"
        f"{'a_area vol':>12}{'a_area mw':>12}{'cell mw RoM':>13}{'cell mw med':>13}",
        f"{'[pc]':>6}{'':>6}{'[K cm^-3]':>14}{'[K cm^-3]':>12}{'[K cm^-3]':>12}"
        f"{'':>12}{'':>12}{'':>13}{'':>13}",
        "-" * 100,
    ]
    for i, zc in enumerate(out["slab_centers_pc"]):
        for sg_key in SELF_GRAVITY_SETTINGS:
            L.append(
                f"{slab_label(zc):>6}{sg_key:>6}"
                f"{out[f'slab__{sg_key}__Ptot_area'][i]:>14.1f}"
                f"{out[f'slab__{sg_key}__Pth_neutral_vol'][i]:>12.1f}"
                f"{out[f'slab__{sg_key}__Pth_neutral_mw'][i]:>12.1f}"
                f"{out[f'slab__{sg_key}__alpha_area_vol'][i]:>12.3f}"
                f"{out[f'slab__{sg_key}__alpha_area_mw'][i]:>12.3f}"
                f"{out[f'slab__{sg_key}__cellbased_alpha_ratio_of_means_mw'][i]:>13.3f}"
                f"{out[f'slab__{sg_key}__cellbased_alpha_median_of_ratios_mw'][i]:>13.3f}")
        L.append("")
    L += [
        "-" * 100,
        "Why alpha_area (vol) sits BELOW the cell-based mass-weighted numbers: the numerator is",
        "an average over every cell while the cell-based mass-weighted estimators weight by n_H,",
        "which emphasises the dense midplane clumps where alpha is largest, and the ratio of",
        "means additionally weights each cell by its own P_th.",
        "",
        "-" * 100,
        "3. SUB-AREA SCALE TEST -- alpha_area inside N x N patches",
        "-" * 100,
        "The footprint is split into N x N equal-width square patches and the SAME three",
        "quantities are formed from each patch's own cells. g(z) is NOT recomputed per patch: it",
        "is a property of the whole disc, and changing it per patch would be a different",
        "physical model rather than a scale test of this one. Reported: median and",
        f"{pct_lo:g}-{pct_hi:g} percentiles ACROSS patches (nearest-rank, equal weight per patch).",
        "",
    ]
    side_pc = {int(N): float(s) for N, s in
               zip(out["patch_ns"], out["patch_sides_pc"])}
    for weight, wlabel in (("vol", "volume-weighted P_th"),
                             ("mw", "mass-weighted P_th")):
        header = f"{'z':>6}{'sg':>6}{'full 1000 pc':>14}"
        for N in SCALE_TEST_NS:
            header += f"{f'N={N} ({side_pc[N]:.0f} pc)':>28}"
        L += [
            f"alpha_area, {wlabel}",
            header,
            "-" * (26 + 28 * len(SCALE_TEST_NS)),
        ]
        for i, zc in enumerate(out["slab_centers_pc"]):
            for sg_key in SELF_GRAVITY_SETTINGS:
                row = (f"{slab_label(zc):>6}{sg_key:>6}"
                       f"{out[f'slab__{sg_key}__alpha_area_{weight}'][i]:>14.3f}")
                for N in SCALE_TEST_NS:
                    stem = f"slab__{sg_key}__patch{N}__alpha_area_{weight}"
                    row += (f"{out[stem + '_median'][i]:>12.3f}"
                            f"  [{out[stem + '_p_lo'][i]:>6.3f},{out[stem + '_p_hi'][i]:>7.3f}]")
                L.append(row)
            L.append("")
        L.append("")

    # the full-footprint profile, printed at a readable stride
    L += [
        "-" * 100,
        "4. PROFILE (every 10th 4 pc bin; the full 200-bin arrays are in "
        "cache/core/area_alpha.npz)",
        "-" * 100,
        f"{'z':>7}{'<P_tot>_area':>14}{'<P_th> vol':>12}{'<P_th> mw':>12}"
        f"{'a_area vol':>12}{'a_area mw':>12}{'cell mw RoM':>13}{'neutral frac':>14}",
        f"{'[pc]':>7}{'[K cm^-3]':>14}{'[K cm^-3]':>12}{'[K cm^-3]':>12}"
        f"{'':>12}{'':>12}{'':>13}{'':>14}",
        "-" * 96,
    ]
    sg_key = "mean"
    z = out["z_pc"]
    for i in range(0, len(z), 10):
        nfrac = (out[f"{sg_key}__n_neutral"][i] / out[f"{sg_key}__n_cells"][i]
                 if out[f"{sg_key}__n_cells"][i] else np.nan)
        L.append(
            f"{z[i]:>7.0f}{out[f'{sg_key}__Ptot_area'][i]:>14.1f}"
            f"{out[f'{sg_key}__Pth_neutral_vol'][i]:>12.1f}"
            f"{out[f'{sg_key}__Pth_neutral_mw'][i]:>12.1f}"
            f"{out[f'{sg_key}__alpha_area_vol'][i]:>12.3f}"
            f"{out[f'{sg_key}__alpha_area_mw'][i]:>12.3f}"
            f"{out[f'{sg_key}__cellbased_alpha_ratio_of_means_mw'][i]:>13.3f}"
            f"{nfrac:>14.4f}")
    L += [
        "-" * 96,
        "  (self-gravity = mean, the fiducial setting; 'neutral frac' is the fraction of the",
        "   bin's cells that are NOT HIM-flagged, i.e. the population behind <P_th>_neutral)",
        "",
        "-" * 100,
        "5. TIGRESS R8 midplane reference (Ostriker & Kim 2022, Table 2; 2p gas, T < 2e4 K)",
        "-" * 100,
        f"  P_tot_2p {ref['P_tot_2p']:>10.4g} K cm^-3    P_th_2p {ref['P_th_2p']:>10.4g} K cm^-3"
        f"    alpha_2p {ref['alpha_2p']:>7.3g}",
        f"  R8 has Sigma_gas = {ref['Sigma_gas']:.2f} Msun/pc^2, about twice this cube's 4.81, and",
        "  absolute pressures scale roughly as Sigma_gas^2 in a self-regulated disc -- so the",
        "  pressures are NOT expected to match. alpha is a ratio and CAN be compared.",
        "",
    ]
    L += cell_alpha_report_lines(out, cell_checks, pct_lo, pct_hi)
    L += [
        "",
        f"(runtime {time.time() - t0:.1f}s)",
    ]
    report = "\n".join(L)
    print()
    print(report)
    OUT_TXT_PATH.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved {OUT_TXT_PATH}")


if __name__ == "__main__":
    main()
