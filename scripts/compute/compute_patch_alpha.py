"""
Step 2f: RANDOM-PATCH analysis of the area-averaged pressures and alpha.

Why this script exists
----------------------
Steps 2d/2e take every average over the whole 1 x 1 kpc footprint. How
much of that answer survives on smaller areas is measured here: N_PATCH
square patches per size are dropped at random positions inside the
footprint, and the same three area quantities are formed from each
patch's own cells, per 4 pc z bin and per 60 pc slab.

Definitions (patch i, one z group = a 4 pc bin or a slab)
--------------------------------------------------------
  <P_tot>_i          plain mean of the cached per-column P_tot over ALL
                     cells of patch i in the group (HIM included).
  <1/P_th>^-1_i      [mean of 1/p_th_phys over the patch's NON-HIM cells]^-1
                     (vol), and [sum(n/p_th)/sum(n)]^-1 (mw).
  <alpha>_i          <P_tot>_i * <1/P_th>_i, both weightings.
  alpha_area_i       <P_tot>_i / <P_th>_i, arithmetic, vol only (comparison).

Patches with no non-HIM cell in a group get NaN there; how often is
reported. g(z) is NOT recomputed per patch -- P_tot is read from the
cache, integrated against the whole-disc g -- so this is a test of how
the AVERAGING area changes the answer, not of a different physical model.

Patches
-------
L = 500 pc (250 x 250 cells, fiducial) and L = 250 pc (125 x 125 cells),
N_PATCH each, lower-left cell index drawn uniformly over every integer
offset that keeps the patch inside the 501 x 501 footprint, with the
fixed PATCH_SEED. The patches overlap heavily (only 4 resp. 16 disjoint
ones fit); the overlap statistics are reported so the spread across
patches is not mistaken for 1000 independent samples.

How it is computed
------------------
For every z plane, 2-D summed-area tables (float64) of P_tot (per
self-gravity setting), the non-HIM indicator, (1/p_th)*[non-HIM],
n*[non-HIM], (n/p_th)*[non-HIM] and p_th*[non-HIM]; each patch sum is
then 4 lookups. Planes are combined into groups by SUMMING the per-plane
sums, never by averaging plane means. The whole footprint is evaluated
through the same lookups as one extra patch and must reproduce the Step
2e numbers in cache/core/area_alpha.npz to float precision; 20 random
patches are also summed by brute force and must agree. Either failure
raises.

COMPUTE script: reads cache/core/alpha_core.zarr and area_alpha.npz,
writes cache/core/patch_alpha.npz and results/patch_alpha.txt. No physics
is changed or recomputed. Variant RAW only.
"""

import sys
import time
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.conventions import (  # noqa: E402
    PERCENTILE_SCHEME_DEFAULT,
    PROFILE_BIN_PC,
    PROVISIONAL_CUBE_HEADER,
    SLAB_CENTERS_PC,
    SLAB_HALF_THICKNESS_PC,
    STATS_BOX_XY_HALF_RANGE_PC,
    STATS_BOX_Z_HALF_RANGE_PC,
    THERMAL_PRESSURE_HEADER,
)
from src.physics.loading import open_zarr, slab_z_indices  # noqa: E402
from src.physics.stats import percentile_levels, weighted_stats  # noqa: E402

ALPHA_CORE_ZARR_PATH = Path("cache/core/alpha_core.zarr")
AREA_ALPHA_NPZ_PATH = Path("cache/core/area_alpha.npz")
OUT_NPZ_PATH = Path("cache/core/patch_alpha.npz")
OUT_TXT_PATH = Path("results/patch_alpha.txt")

VARIANT = "RAW"
SELF_GRAVITY_SETTINGS = ("mean", "off")
WEIGHTS = ("vol", "mw")

PATCH_SEED = 20261020
N_PATCH = 1000
PATCH_SIZES_PC = (500.0, 250.0)     # the first is the fiducial
N_BRUTE_FORCE = 20
Z_CHUNK = 64

# The whole-footprint patch against Step 2e, and the brute-force patches
# against the summed-area lookups. Both are float64 sums of the same
# float32 cells in a different order; 1e-12 is far above that round-off
# (~1e-15) and far below any real mismatch (a dropped row is ~1e-3).
CHECK_RTOL = 1e-12

REPORT_SLAB_LABELS = tuple(f"z{int(zc)}" for zc in SLAB_CENTERS_PC)

# per-plane fields the summed-area tables are built from; P_tot is added
# per self-gravity setting as "ptot_<sg>"
CELL_FIELDS = ("neutral", "inv_pth", "n", "n_inv_pth", "pth")
SUM_KEYS = tuple(f"ptot_{sg}" for sg in SELF_GRAVITY_SETTINGS) + CELL_FIELDS

QUANTITIES = ("Ptot", "Pth_inv_inv_vol", "Pth_inv_inv_mw", "alpha_vol",
              "alpha_mw", "alpha_area_vol")


# ---------------------------------------------------------------------------
# patches
# ---------------------------------------------------------------------------
def draw_patch_offsets(n_footprint, n_cells_side, n_patch=N_PATCH, seed=PATCH_SEED):
    """(n_patch, 2) int array of (iy, ix) lower-left cell indices.

    Uniform over every integer offset 0 .. n_footprint - n_cells_side
    inclusive, so every patch lies fully inside the footprint.
    """
    if n_cells_side > n_footprint:
        raise ValueError("patch larger than the footprint")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_footprint - n_cells_side + 1, size=(n_patch, 2))


def coverage_map(offsets, n_cells_side, n_footprint):
    """How many patches contain each footprint cell, (n_footprint, n_footprint).

    Built with a 2-D difference array, so it is exact and O(n_patch).
    Uniform offsets with the patch fully inside the footprint cover the
    centre far more often than the edges: a corner cell is inside a patch
    only for offset (0, 0).
    """
    d = np.zeros((n_footprint + 1, n_footprint + 1), dtype=np.int64)
    m = n_cells_side
    for iy, ix in offsets:
        d[iy, ix] += 1
        d[iy, ix + m] -= 1
        d[iy + m, ix] -= 1
        d[iy + m, ix + m] += 1
    return np.cumsum(np.cumsum(d, axis=0), axis=1)[:n_footprint, :n_footprint]


def overlap_fractions(offsets, n_cells_side):
    """Area-overlap fraction of every unordered pair of distinct patches."""
    iy, ix = offsets[:, 0], offsets[:, 1]
    a, b = np.triu_indices(len(offsets), k=1)
    oy = np.clip(n_cells_side - np.abs(iy[a] - iy[b]), 0, None)
    ox = np.clip(n_cells_side - np.abs(ix[a] - ix[b]), 0, None)
    return (oy * ox) / float(n_cells_side * n_cells_side)


# ---------------------------------------------------------------------------
# summed-area tables
# ---------------------------------------------------------------------------
def summed_area_table(a):
    """(Ny+1, Nx+1) float64 table S with S[j, i] = sum(a[:j, :i])."""
    a = np.asarray(a, dtype=np.float64)
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    np.cumsum(np.cumsum(a, axis=0), axis=1, out=s[1:, 1:])
    return s


def patch_sums(sat, offsets, n_cells_side):
    """Sum of the original array over each square patch, 4 lookups each."""
    iy, ix = offsets[:, 0], offsets[:, 1]
    m = n_cells_side
    return sat[iy + m, ix + m] - sat[iy, ix + m] - sat[iy + m, ix] + sat[iy, ix]


def plane_fields(ptot_by_sg, pth, n_h, him):
    """The per-cell arrays whose patch sums every quantity is built from."""
    neutral = ~np.asarray(him, dtype=bool)
    pth = np.asarray(pth, dtype=np.float64)
    n_h = np.asarray(n_h, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(neutral, 1.0 / pth, 0.0)
    f = {f"ptot_{sg}": np.asarray(p, dtype=np.float64) for sg, p in ptot_by_sg.items()}
    f.update({
        "neutral": neutral.astype(np.float64),
        "inv_pth": inv,
        "n": np.where(neutral, n_h, 0.0),
        "n_inv_pth": np.where(neutral, n_h, 0.0) * inv,
        "pth": np.where(neutral, pth, 0.0),
    })
    return f


def accumulate_plane(acc, gids, sats, offsets, n_cells_side):
    """Add one plane's patch sums to every group the plane belongs to.

    sats maps each SUM_KEY to that plane's summed-area table (built once
    per plane and shared by every patch size). acc[key] has shape
    (n_groups, n_patch); groups are combined by summing per-plane sums,
    so a group's mean is over all its cells at once.
    """
    for key, sat in sats.items():
        ps = patch_sums(sat, offsets, n_cells_side)
        for g in gids:
            acc[key][g] += ps


def quantities_from_sums(s, n_cells_per_patch_plane, n_planes):
    """The patch quantities from the accumulated sums.

    n_planes is (n_groups,), the number of z planes in each group, so the
    total cell count of a patch in a group is n_cells_per_patch_plane *
    n_planes. Groups without a non-HIM cell give NaN for every P_th-based
    quantity.
    """
    n_all = (n_cells_per_patch_plane * np.asarray(n_planes, dtype=float))[:, None]
    nn, inv, sn, sninv, spth = (s["neutral"], s["inv_pth"], s["n"],
                                s["n_inv_pth"], s["pth"])
    with np.errstate(invalid="ignore", divide="ignore"):
        pinv_vol = np.where(nn > 0, nn / inv, np.nan)
        # nn is an exact integer count; sn and sninv carry summed-area
        # round-off and are not exactly 0 for an all-HIM patch
        pinv_mw = np.where(nn > 0, sn / sninv, np.nan)
        pth_ar = np.where(nn > 0, spth / nn, np.nan)
        out = {}
        for sg in SELF_GRAVITY_SETTINGS:
            ptot = np.where(n_all > 0, s[f"ptot_{sg}"] / n_all, np.nan)
            out[sg] = {
                "Ptot": ptot,
                "Pth_inv_inv_vol": pinv_vol,
                "Pth_inv_inv_mw": pinv_mw,
                "alpha_vol": ptot / pinv_vol,
                "alpha_mw": ptot / pinv_mw,
                "alpha_area_vol": ptot / pth_ar,
            }
    return out


def across_patches(values, scheme=PERCENTILE_SCHEME_DEFAULT):
    """(mean, median, p_lo, p_hi, n_nan) per group over the patch axis (-1).

    Equal weight per patch, NaN patches dropped; percentiles are the
    project's nearest-rank estimator, as everywhere else.
    """
    v = np.asarray(values, dtype=float)
    out = np.full((v.shape[0], 4), np.nan)
    n_nan = np.sum(~np.isfinite(v), axis=1)
    for g in range(v.shape[0]):
        st = weighted_stats(v[g], np.ones(v.shape[1]), scheme)
        if st.n:
            out[g] = (st.arithmetic_mean, st.median, st.p_lo, st.p_hi)
    return out, n_nan


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    pct_lo, pct_hi = percentile_levels(PERCENTILE_SCHEME_DEFAULT)
    import pipeline.compute_all as pipe

    core = zarr.open_group(str(ALPHA_CORE_ZARR_PATH), mode="r")
    x_pc = np.asarray(core["x_pc"][:], dtype=np.float64)
    y_pc = np.asarray(core["y_pc"][:], dtype=np.float64)
    if x_pc.size != y_pc.size:
        raise RuntimeError("non-square footprint; the patch offsets assume a square one")
    n_fp = x_pc.size
    dx = float(np.median(np.diff(x_pc)))
    z_full = np.asarray(open_zarr().z_pc, dtype=np.float64)
    zsl, z_box = pipe._box_z_slice(z_full)
    vg = core[VARIANT]

    bin_groups = pipe.profile_bin_groups(z_box, use_abs=False)
    n_bins = len(bin_groups)
    slab_groups = [(float(zc), slab_z_indices(z_box, zc)) for zc in SLAB_CENTERS_PC]
    groups = [idx for _, idx in bin_groups] + [idx for _, idx in slab_groups]
    n_groups = len(groups)
    n_planes = np.array([len(i) for i in groups], dtype=float)
    plane_groups = [[] for _ in range(len(z_box))]
    for g, idxs in enumerate(groups):
        for i in idxs:
            plane_groups[int(i)].append(g)

    sizes = {}
    for L in PATCH_SIZES_PC:
        m = int(round(L / dx))
        offs = draw_patch_offsets(n_fp, m)
        sizes[L] = (m, offs)
        print(f"L = {L:.0f} pc: {m} x {m} cells, {N_PATCH} patches, offsets "
              f"0..{n_fp - m}, seed {PATCH_SEED}")
    # the whole footprint as one patch, through the same lookups
    full_offs = np.zeros((1, 2), dtype=int)
    rng_bf = np.random.default_rng(PATCH_SEED + 1)
    bf_pick = {L: rng_bf.choice(N_PATCH, size=N_BRUTE_FORCE // len(PATCH_SIZES_PC),
                                replace=False) for L in PATCH_SIZES_PC}

    acc = {L: {k: np.zeros((n_groups, N_PATCH)) for k in SUM_KEYS} for L in PATCH_SIZES_PC}
    acc_full = {k: np.zeros((n_groups, 1)) for k in SUM_KEYS}
    acc_bf = {L: {k: np.zeros((n_groups, len(bf_pick[L]))) for k in SUM_KEYS}
              for L in PATCH_SIZES_PC}

    ptot_arrs = {sg: vg[f"self_gravity_{sg}"]["Ptot"] for sg in SELF_GRAVITY_SETTINGS}
    print(f"Reducing {len(z_box)} box planes into {n_bins} bins + {len(slab_groups)} slabs...")
    for lo in range(0, len(z_box), Z_CHUNK):
        hi = min(lo + Z_CHUNK, len(z_box))
        sl = slice(zsl.start + lo, zsl.start + hi)
        ptot_c = {sg: np.asarray(a[sl], dtype=np.float64) for sg, a in ptot_arrs.items()}
        pth_c = np.asarray(vg["p_th_phys"][sl], dtype=np.float64)
        n_c = np.asarray(vg["n_model"][sl], dtype=np.float64)
        him_c = np.asarray(vg["him"][sl]).astype(bool)
        for k in range(hi - lo):
            gids = plane_groups[lo + k]
            if not gids:
                continue
            f = plane_fields({sg: p[k] for sg, p in ptot_c.items()},
                             pth_c[k], n_c[k], him_c[k])
            sats = {key: summed_area_table(a) for key, a in f.items()}
            accumulate_plane(acc_full, gids, sats, full_offs, n_fp)
            for L, (m, offs) in sizes.items():
                accumulate_plane(acc[L], gids, sats, offs, m)
                for key, a in f.items():
                    bf = np.array([a[iy:iy + m, ix:ix + m].sum()
                                   for iy, ix in offs[bf_pick[L]]])
                    for g in gids:
                        acc_bf[L][key][g] += bf
        del ptot_c, pth_c, n_c, him_c

    # ---- check 1: brute force vs summed-area lookups ---------------------
    bf_max = 0.0
    for L, (m, offs) in sizes.items():
        for key in SUM_KEYS:
            a = acc[L][key][:, bf_pick[L]]
            b = acc_bf[L][key]
            # relative to the group's whole-footprint sum: summed-area
            # round-off scales with the table, not with the patch, and a
            # patch sum can legitimately be 0 (no non-HIM cell)
            scale = np.maximum(np.abs(acc_full[key]), 1e-300)
            rel = np.abs(a - b) / scale
            bf_max = max(bf_max, float(rel.max()))
    print(f"check: summed-area vs brute force, {N_BRUTE_FORCE} patches, "
          f"max rel diff {bf_max:.2e}")
    if bf_max > CHECK_RTOL:
        raise RuntimeError(f"Summed-area patch sums disagree with brute force by "
                           f"{bf_max:.3e} (> {CHECK_RTOL:.0e}). STOPPING.")

    # ---- check 2: whole footprint vs Step 2e -----------------------------
    full = quantities_from_sums(acc_full, n_fp * n_fp, n_planes)
    ref = np.load(AREA_ALPHA_NPZ_PATH, allow_pickle=False)
    ref_map = {"Ptot": "Ptot_area", "Pth_inv_inv_vol": "Pth_harm_vol",
               "Pth_inv_inv_mw": "Pth_harm_mw", "alpha_vol": "alpha_cell_mean_vol",
               "alpha_mw": "alpha_cell_mean_mw", "alpha_area_vol": "alpha_area_vol"}
    full_max = 0.0
    out = {}
    for sg in SELF_GRAVITY_SETTINGS:
        for q, rk in ref_map.items():
            r = np.concatenate([ref[f"{sg}__{rk}"], ref[f"slab__{sg}__{rk}"]])
            v = full[sg][q][:, 0]
            fin = np.isfinite(r)
            if not np.array_equal(fin, np.isfinite(v)):
                raise RuntimeError(f"full-footprint {q} ({sg}) NaN pattern differs from Step 2e")
            rel = np.abs(v[fin] - r[fin]) / np.abs(r[fin])
            full_max = max(full_max, float(rel.max()))
            out[f"full__{sg}__{q}"] = v[:n_bins]
            out[f"slab__full__{sg}__{q}"] = v[n_bins:]
    print(f"check: whole footprint vs Step 2e, max rel diff {full_max:.2e}")
    if full_max > CHECK_RTOL:
        raise RuntimeError(f"The whole-footprint patch does not reproduce Step 2e "
                           f"(max rel diff {full_max:.3e} > {CHECK_RTOL:.0e}). STOPPING.")

    # ---- per-patch quantities and their spread ---------------------------
    out.update({
        "z_pc": np.array([c for c, _ in bin_groups], dtype=float),
        "bin_edges": pipe.profile_bin_edges(use_abs=False),
        "slab_centers_pc": np.array(SLAB_CENTERS_PC, dtype=float),
        "slab_half_thickness_pc": np.array([SLAB_HALF_THICKNESS_PC]),
        "percentile_levels": np.array([pct_lo, pct_hi]),
        "patch_sizes_pc": np.array(PATCH_SIZES_PC),
        "patch_seed": np.array([PATCH_SEED]),
        "n_patch": np.array([N_PATCH]),
        "n_footprint_cells_side": np.array([n_fp]),
        "cell_pc": np.array([dx]),
        "variant": np.array([VARIANT]),
        "provisional": np.array([PROVISIONAL_CUBE_HEADER]),
        "check_bruteforce_max_rel": np.array([bf_max]),
        "check_full_footprint_max_rel": np.array([full_max]),
        "ref_alpha_2p": ref["ref_alpha_2p"],
    })
    overlap = {}
    nan_counts = {}
    for L, (m, offs) in sizes.items():
        tag = f"L{int(L)}"
        out[f"{tag}__offsets_iy_ix"] = offs
        out[f"{tag}__n_cells_side"] = np.array([m])
        ov = overlap_fractions(offs, m)
        overlap[L] = (float(ov.mean()), float(ov.min()), float(np.mean(ov == 0)),
                      (n_fp // m) ** 2, ov.size)
        out[f"{tag}__overlap_mean_min_fraczero_ndisjoint"] = np.array(overlap[L][:4])
        cov = coverage_map(offs, m, n_fp)
        out[f"{tag}__coverage"] = cov.astype(np.int32)
        c = n_fp // 2
        overlap[L] = overlap[L] + ((int(cov[c, c]), int(cov[0, c]), int(cov[0, 0]),
                                    float(np.mean(cov == 0)),
                                    float(np.mean(cov[n_fp // 4:3 * n_fp // 4,
                                                      n_fp // 4:3 * n_fp // 4])
                                          / cov.mean())),)
        q = quantities_from_sums(acc[L], m * m, n_planes)
        for sg in SELF_GRAVITY_SETTINGS:
            for name, arr in q[sg].items():
                out[f"{tag}__{sg}__{name}"] = arr[:n_bins]
                out[f"slab__{tag}__{sg}__{name}"] = arr[n_bins:]
                st, nn = across_patches(arr)
                for j, s in enumerate(("mean", "median", "p_lo", "p_hi")):
                    out[f"{tag}__{sg}__{name}__{s}"] = st[:n_bins, j]
                    out[f"slab__{tag}__{sg}__{name}__{s}"] = st[n_bins:, j]
                out[f"{tag}__{sg}__{name}__n_nan"] = nn[:n_bins]
                out[f"slab__{tag}__{sg}__{name}__n_nan"] = nn[n_bins:]
        nan_counts[L] = {w: out[f"{tag}__mean__Pth_inv_inv_{w}__n_nan"] for w in WEIGHTS}
        nan_counts[L]["slab"] = out[f"slab__{tag}__mean__Pth_inv_inv_vol__n_nan"]
        print(f"L = {L:.0f}: overlap mean {overlap[L][0]:.3f}, min {overlap[L][1]:.3f}; "
              f"NaN patch-bins (vol) {int(nan_counts[L]['vol'].sum())} of "
              f"{n_bins * N_PATCH}")

    OUT_NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ_PATH, **out)
    print(f"Saved {OUT_NPZ_PATH}")
    write_report(out, sizes, overlap, nan_counts, bf_max, full_max, pct_lo, pct_hi,
                 n_bins, t0)


# ---------------------------------------------------------------------------
LABELS = {
    "Ptot": "<P_tot>_i [K cm^-3]",
    "Pth_inv_inv_vol": "<1/P_th>^-1_i vol [K cm^-3]",
    "Pth_inv_inv_mw": "<1/P_th>^-1_i mw [K cm^-3]",
    "alpha_vol": "<alpha>_i vol",
    "alpha_mw": "<alpha>_i mw",
    "alpha_area_vol": "alpha_area_i = <P_tot>_i/<P_th>_i (vol, arithmetic)",
}


def _fmt(v, q):
    return f"{v:>9.1f}" if not q.startswith("alpha") else f"{v:>9.3f}"


def write_report(out, sizes, overlap, nan_counts, bf_max, full_max, pct_lo, pct_hi,
                 n_bins, t0):
    L_ = [
        PROVISIONAL_CUBE_HEADER,
        THERMAL_PRESSURE_HEADER,
        "Step 2f: RANDOM-PATCH area averages of P_tot, <1/P_th>^-1 and alpha",
        "=" * 100,
        "",
        "For each of N_PATCH random square patches and each z group (4 pc bin or 60 pc slab):",
        "  <P_tot>_i       plain mean of the per-column P_tot over ALL the patch's cells (HIM incl.)",
        "  <1/P_th>^-1_i   [mean of 1/p_th over the patch's NON-HIM cells]^-1 (vol);",
        "                  [sum(n/p_th)/sum(n)]^-1 (mw)",
        "  <alpha>_i       <P_tot>_i * <1/P_th>_i, both weightings",
        "  alpha_area_i    <P_tot>_i / <P_th>_i (arithmetic, vol) -- for comparison",
        f"Statistics ACROSS patches: mean, median, p{pct_lo:g}, p{pct_hi:g} (equal weight per",
        "patch, nearest-rank, NaN patches dropped). 'full' = the whole 1 x 1 kpc footprint,",
        "i.e. the Step 2e number.",
        "",
        f"RAW only. Footprint |x|,|y| <= {STATS_BOX_XY_HALF_RANGE_PC:.0f} pc "
        f"({int(out['n_footprint_cells_side'][0])} x {int(out['n_footprint_cells_side'][0])} "
        f"columns of {out['cell_pc'][0]:g} pc), box |z| <= {STATS_BOX_Z_HALF_RANGE_PC:.0f} pc, "
        f"{PROFILE_BIN_PC:.0f} pc bins, slabs +-{SLAB_HALF_THICKNESS_PC:.0f} pc.",
        f"{N_PATCH} patches per size, lower-left cell uniform over all valid integer offsets,",
        f"seed {PATCH_SEED} (offsets stored in cache/core/patch_alpha.npz). g(z) is the",
        "whole-disc one; P_tot is read from the cache, never recomputed per patch.",
        "",
        "-" * 100,
        "1. CHECKS (the script raises if either fails)",
        "-" * 100,
        f"  whole footprint through the summed-area lookups vs Step 2e (<P_tot>_area,",
        f"  <1/P_th>^-1 vol/mw, <alpha_cell> vol/mw, alpha_area vol), every bin and slab, both",
        f"  self-gravity settings: max |rel diff| {full_max:.2e}   (threshold {CHECK_RTOL:.0e})",
        f"  summed-area lookups vs brute-force sums, {N_BRUTE_FORCE} random patches, every sum,",
        f"  bin and slab: max |rel diff| {bf_max:.2e}   (threshold {CHECK_RTOL:.0e})",
        "",
        "-" * 100,
        "2. PATCH OVERLAP and NaN counts",
        "-" * 100,
        f"{'L [pc]':>8}{'cells':>8}{'pairs':>10}{'mean overlap':>14}{'min overlap':>13}"
        f"{'frac pairs disjoint':>21}{'disjoint that fit':>19}",
    ]
    for Lp, (m, _) in sizes.items():
        mean_ov, min_ov, f0, nd, npairs, _ = overlap[Lp]
        L_.append(f"{Lp:>8.0f}{m:>8d}{npairs:>10d}{mean_ov:>14.3f}{min_ov:>13.3f}"
                  f"{f0:>21.4f}{nd:>19d}")
    L_ += [
        "  Overlap = shared area / patch area, over ALL pairs of the patches (not a sample).",
        "  'disjoint that fit' = floor(501/cells)^2, the most non-overlapping patches possible.",
        "",
        "  COVERAGE -- how many of the patches contain a given footprint cell:",
        f"{'L [pc]':>10}{'centre':>9}{'edge-mid':>10}{'corner':>8}{'never covered':>15}"
        f"{'central 250 pc square / footprint mean':>40}",
    ]
    for Lp in sizes:
        cc, ce, ck, f_never, central = overlap[Lp][5]
        L_.append(f"{Lp:>10.0f}{cc:>9d}{ce:>10d}{ck:>8d}{f_never:>15.4f}{central:>40.2f}")
    L_ += [
        "  Uniform offsets with every patch fully inside the footprint sample the CENTRE far",
        "  more often than the edges (a corner cell is only in the patch at offset (0,0)). The",
        "  across-patch statistics therefore describe mostly the central part of the footprint,",
        "  while 'full' weights every cell equally. Where the two disagree -- e.g. the full",
        "  <1/P_th>^-1 at z = 0 sitting below most patches -- the footprint's outer cells differ",
        "  from its centre; it is not an error in either number (the full value is checked",
        "  against Step 2e above).",
        "",
        "  NaN patch-groups (no non-HIM cell in that patch and group), self-gravity-independent:",
    ]
    for Lp in sizes:
        nc = nan_counts[Lp]
        worst = int(np.argmax(nc["vol"]))
        where = (f"worst bin z = {out['z_pc'][worst]:.0f} pc ({int(nc['vol'][worst])} patches)"
                 if nc["vol"].any() else "none in any bin")
        L_.append(
            f"    L = {Lp:.0f} pc: vol {int(nc['vol'].sum())} / mw {int(nc['mw'].sum())} of "
            f"{n_bins * N_PATCH} patch-bins; {where}; slabs {list(map(int, nc['slab']))}")
    L_.append("")

    sec = 3
    for Lp, (m, _) in sizes.items():
        tag = f"L{int(Lp)}"
        L_ += ["-" * 100,
               f"{sec}. SLABS, L = {Lp:.0f} pc ({m} x {m} cells)",
               "-" * 100]
        sec += 1
        for sg in SELF_GRAVITY_SETTINGS:
            L_ += [f"self-gravity {sg}",
                   f"{'z':>5}  {'quantity':<52}{'mean':>9}{'median':>9}"
                   f"{f'p{pct_lo:g}':>9}{f'p{pct_hi:g}':>9}{'full':>9}{'NaN':>6}"]
            for i, zc in enumerate(out["slab_centers_pc"]):
                for q in QUANTITIES:
                    s = f"slab__{tag}__{sg}__{q}__"
                    L_.append(
                        f"{zc:>5.0f}  {LABELS[q]:<52}"
                        + "".join(_fmt(out[s + k][i], q) for k in ("mean", "median", "p_lo", "p_hi"))
                        + _fmt(out[f"slab__full__{sg}__{q}"][i], q)
                        + f"{int(out[s + 'n_nan'][i]):>6d}")
                L_.append("")
    for Lp, (m, _) in sizes.items():
        tag = f"L{int(Lp)}"
        for sg in SELF_GRAVITY_SETTINGS:
            for w in WEIGHTS:
                qs = ("Ptot", f"Pth_inv_inv_{w}", f"alpha_{w}")
                L_ += ["-" * 100,
                       f"{sec}. PROFILE, L = {Lp:.0f} pc, self-gravity {sg}, {w}-weighted P_th "
                       f"(every 4 pc bin)",
                       "-" * 100,
                       f"{'':>6}{'<P_tot>_i':^45}{'<1/P_th>^-1_i':^45}{'<alpha>_i':^45}",
                       f"{'z':>6}" + (f"{'mean':>9}{'med':>9}{f'p{pct_lo:g}':>9}"
                                      f"{f'p{pct_hi:g}':>9}{'full':>9}") * 3 + f"{'NaN':>6}"]
                sec += 1
                for i, z in enumerate(out["z_pc"]):
                    row = f"{z:>6.0f}"
                    for q in qs:
                        s = f"{tag}__{sg}__{q}__"
                        row += "".join(_fmt(out[s + k][i], q)
                                       for k in ("mean", "median", "p_lo", "p_hi"))
                        row += _fmt(out[f"full__{sg}__{q}"][i], q)
                    row += f"{int(out[f'{tag}__{sg}__{qs[1]}__n_nan'][i]):>6d}"
                    L_.append(row)
                L_.append("")
    L_ += [f"(runtime {time.time() - t0:.1f}s)"]
    report = "\n".join(L_)
    OUT_TXT_PATH.write_text(report + "\n", encoding="utf-8")
    print("\n".join(L_[:80]))
    print(f"\nSaved {OUT_TXT_PATH}")


if __name__ == "__main__":
    main()
