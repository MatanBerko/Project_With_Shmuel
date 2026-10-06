"""
Is the density in HIM-flagged cells measured, or is it the prior?

HIM-flagged cells sit at n_H ~ 0.02-0.04 cm^-3, close to the Edenhofer
map's high-|z| floor. If their density is prior-dominated rather than
constrained by data, then everything built on them -- the flag itself,
and HIM_A/HIM_B's substitution -- is being driven by the reconstruction's
prior, not by observations. This script measures that directly, per
voxel, as the posterior signal-to-noise

    SNR = posterior mean / posterior standard deviation

of the Edenhofer et al. (2023) differential extinction.

TWO TESTS, ONE OF THEM SHAPE-INDEPENDENT
----------------------------------------
    SNR = mean / std                 -- how many posterior sigmas the
                                        value sits above zero.
    detected: mean - 2*std > 0       -- a hard "clearly detected" test,
                                        phrased without dividing, so it
                                        needs no assumption about the
                                        posterior's shape.

Wherever std > 0 these are algebraically the same cut: mean - 2 std > 0
is exactly SNR > 2. They are reported separately anyway because the
detection form is evaluated straight from mean and std and therefore also
covers the degenerate voxels where std = 0 (SNR undefined there, but the
detection test is not), and because it is the form that does not invite a
Gaussian reading of the ratio.

WHICH MAP FLAVOUR, AND WHY NOT THE SAMPLES
------------------------------------------
Edenhofer2023Query(integrated=False, flavor="main"), queried with
mode="mean" and mode="std". That is the SAME map the cube was built from
(pipeline/compute_sigma_gas_factor_comparison.py and
scripts/validation/check_orientation.py both use it) and the same
posterior; "mean" and "std" are the published per-voxel first two moments
of exactly the posterior whose samples load_samples=True would give.

load_samples=True is NOT used because samples_healpix.fits is not
downloaded (dustmaps raises FileNotFoundError), and fetching it is a
19 GB download (dustmaps.edenhofer2023.fetch(fetch_samples=True)). This
is not a memory problem -- it is a missing file -- and the quantity this
script needs, mean/std, is available without it. What the samples would
add is the SHAPE of each voxel's posterior beyond its first two moments
(whether a low-SNR voxel is log-normal-ish or piled against a floor) and
the ability to propagate the posterior's spatial correlations through the
column integral. Neither changes the SNR numbers below. If that shape is
wanted later, the only requirement is the 19 GB fetch.

ORIENTATION
-----------
The f98 cube is mirrored in XY relative to Galactic coordinates. The
mapping is READ from results/orientation_check.txt (the "best mapping"
line), never hardcoded, and is then verified before use: the parsed
mapping must reproduce a high cube-vs-map correlation while the identity
mapping does not. If that check fails the script stops rather than
quietly querying the wrong sightlines.

SAMPLING
--------
A stratified random sample over the STATS_BOX: every 20 pc |z| bin gets
the same number of HIM-flagged and non-flagged cells, so the high-|z|
bins (where non-flagged cells are rare) and the midplane (where flagged
cells are a minority of the mass) are both resolved. Because the sample
is stratified, every box-wide number is computed with STRATUM WEIGHTS
(population count / sampled count), so it estimates the population and
not the sample.
"""

import re
import sys
import time
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.conventions import (  # noqa: E402
    N_H_PER_EXTINCTION_F98,
    PROVISIONAL_CUBE_HEADER,
    STATS_BOX_XY_HALF_RANGE_PC,
    STATS_BOX_Z_HALF_RANGE_PC,
)
from src.physics.loading import open_zarr  # noqa: E402

ALPHA_CORE_ZARR_PATH = Path("cache/core/alpha_core.zarr")
ORIENTATION_TXT = Path("results/orientation_check.txt")
OUT_NPZ_PATH = Path("cache/validation/posterior_snr.npz")
OUT_TXT_PATH = Path("results/posterior_snr.txt")

VARIANT = "RAW"              # observed density, no substitution
ABS_Z_BIN_PC = 20.0          # |z| bins for the reported table
N_PER_STRATUM = 500          # per |z| bin per class -> 20 bins x 2 x 500 = 20000
N_VERIFY = 2000              # points used for the orientation re-check
RANDOM_SEED = 20261006
SNR_THRESHOLDS = (1.0, 2.0)
AXIS_NAMES = ("x", "y", "z")


# ---------------------------------------------------------------------------
# orientation: read it, build the transform, verify it
# ---------------------------------------------------------------------------
def read_best_mapping(path=ORIENTATION_TXT):
    """Parse the "best mapping (-y,-x,+z): r = 0.94" line.

    Returns (label, M) where M is the 3x3 signed permutation with
    cube_coords = M @ true_coords -- the direction check_orientation.py
    scanned in (it read the cube at M(p) while querying the map at p).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run scripts/validation/check_orientation.py first; "
            "the XY mapping must be read from its output, not assumed.")
    text = path.read_text(encoding="utf-8")
    m = re.search(r"best mapping \(([^)]*)\):\s*r\s*=\s*([-\d.]+)", text)
    if m is None:
        raise ValueError(f"No 'best mapping (...)' line in {path}")
    label, r_best = m.group(1), float(m.group(2))

    M = np.zeros((3, 3), dtype=float)
    for k, part in enumerate(label.split(",")):
        part = part.strip()
        sign = -1.0 if part[0] == "-" else 1.0
        axis = AXIS_NAMES.index(part[-1])
        M[k, axis] = sign
    # a signed permutation is orthogonal, so the inverse is the transpose
    if not np.allclose(M @ M.T, np.eye(3)):
        raise ValueError(f"Parsed mapping ({label}) is not a signed permutation.")
    return f"({label})", M, r_best


def cube_to_true(M, cube_xyz):
    """True Galactic (x, y, z) for cube coordinates. cube = M @ true, so
    true = M^T @ cube."""
    return cube_xyz @ M  # (N,3) @ (3,3) == (M.T @ cube^T)^T


def pearson_log(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if m.sum() < 3:
        return float("nan"), int(m.sum())
    return float(np.corrcoef(np.log10(a[m]), np.log10(b[m]))[0, 1]), int(m.sum())


def query_map(query, xyz, mode, chunk=200_000):
    """dustmaps query at heliocentric Galactic cartesian points, chunked."""
    import astropy.units as u
    from astropy.coordinates import Galactic, SkyCoord

    out = np.empty(len(xyz), dtype=float)
    for lo in range(0, len(xyz), chunk):
        hi = min(lo + chunk, len(xyz))
        p = xyz[lo:hi]
        gal = Galactic(u=p[:, 0] * u.pc, v=p[:, 1] * u.pc, w=p[:, 2] * u.pc,
                       representation_type="cartesian")
        c = SkyCoord(gal)
        c.representation_type = "spherical"
        out[lo:hi] = np.asarray(query.query(c, mode=mode), dtype=float)
    return out


# ---------------------------------------------------------------------------
def weighted_quantiles(values, weights, qs):
    """Weighted quantiles by the same nearest-rank rule the pipeline uses."""
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not m.any():
        return [float("nan")] * len(qs)
    v, w = v[m], w[m]
    o = np.argsort(v)
    v, w = v[o], w[o]
    cw = np.cumsum(w)
    out = []
    for q in qs:
        i = int(np.searchsorted(cw, q * cw[-1], side="left"))
        out.append(float(v[min(max(i, 0), v.size - 1)]))
    return out


def weighted_fraction(mask, weights):
    w = np.asarray(weights, float)
    m = np.isfinite(w) & (w > 0)
    if not m.any():
        return float("nan")
    return float(np.sum(w[m] * np.asarray(mask, bool)[m]) / np.sum(w[m]))


def main():
    t0 = time.time()
    OUT_NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    label, M, r_best_reported = read_best_mapping()
    print(f"Orientation read from {ORIENTATION_TXT}: best mapping {label} "
          f"(r = {r_best_reported:.4f} there)")

    # ---- cube arrays over the STATS_BOX
    import pipeline.compute_all as pipe
    core = zarr.open_group(str(ALPHA_CORE_ZARR_PATH), mode="r")
    x_pc = np.asarray(core["x_pc"][:], float)
    y_pc = np.asarray(core["y_pc"][:], float)
    z_full = np.asarray(core["z_pc"][:], float)
    zsl, z_pc = pipe._box_z_slice(z_full)

    ix = np.where(np.abs(x_pc) <= STATS_BOX_XY_HALF_RANGE_PC)[0]
    iy = np.where(np.abs(y_pc) <= STATS_BOX_XY_HALF_RANGE_PC)[0]
    x_sub, y_sub = x_pc[ix], y_pc[iy]

    vg = core[VARIANT]
    print("Loading RAW n_H and the HIM flag over the STATS_BOX...")
    n_H = np.asarray(vg["n_model"][zsl], dtype=np.float32)[:, iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1]
    him = np.asarray(vg["him"][zsl]).astype(bool)[:, iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1]
    nz, ny, nx = n_H.shape
    print(f"  box {n_H.shape}; flagged fraction {him.mean():.4f}")

    # ---- stratified sample: equal counts per (|z| bin, class)
    abs_z = np.abs(z_pc)
    n_bins = int(round(STATS_BOX_Z_HALF_RANGE_PC / ABS_Z_BIN_PC))
    bin_edges = np.arange(n_bins + 1, dtype=float) * ABS_Z_BIN_PC
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    samples = []   # (iz, iy, ix, bin, is_flagged, stratum_weight)
    pop_counts = np.zeros((n_bins, 2), dtype=np.int64)
    samp_counts = np.zeros((n_bins, 2), dtype=np.int64)

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        sel_planes = np.where((abs_z >= lo) & ((abs_z < hi) if b < n_bins - 1
                                               else (abs_z <= hi)))[0]
        if sel_planes.size == 0:
            continue
        him_b = him[sel_planes]
        for cls, want_flagged in ((1, True), (0, False)):
            mask = him_b if want_flagged else ~him_b
            n_pop = int(mask.sum())
            pop_counts[b, cls] = n_pop
            if n_pop == 0:
                continue
            take = min(N_PER_STRATUM, n_pop)
            flat = np.flatnonzero(mask.ravel())
            pick = rng.choice(flat, size=take, replace=False)
            p_iz, p_iy, p_ix = np.unravel_index(pick, mask.shape)
            samp_counts[b, cls] = take
            w = n_pop / take
            for k in range(take):
                samples.append((sel_planes[p_iz[k]], p_iy[k], p_ix[k], b, cls, w))

    samples = np.array(samples, dtype=float)
    s_iz = samples[:, 0].astype(int)
    s_iy = samples[:, 1].astype(int)
    s_ix = samples[:, 2].astype(int)
    s_bin = samples[:, 3].astype(int)
    s_flag = samples[:, 4].astype(bool)
    s_wstrat = samples[:, 5]
    n_samp = len(samples)
    print(f"  stratified sample: {n_samp} cells "
          f"({n_bins} |z| bins x 2 classes, up to {N_PER_STRATUM} each)")

    cube_xyz = np.stack([x_sub[s_ix], y_sub[s_iy], z_pc[s_iz]], axis=1)
    true_xyz = cube_to_true(M, cube_xyz)
    s_n_H = n_H[s_iz, s_iy, s_ix].astype(np.float64)

    # ---- the map
    print("Loading Edenhofer2023Query(integrated=False, flavor='main')...")
    from dustmaps.edenhofer2023 import Edenhofer2023Query
    t_q = time.time()
    query = Edenhofer2023Query(integrated=False)
    print(f"  loaded in {time.time() - t_q:.1f}s")

    # ---- VERIFY the orientation before trusting it
    v = rng.choice(n_samp, size=min(N_VERIFY, n_samp), replace=False)
    mean_mapped = query_map(query, true_xyz[v], "mean")
    mean_identity = query_map(query, cube_xyz[v], "mean")
    r_mapped, n_used = pearson_log(s_n_H[v], mean_mapped)
    r_identity, _ = pearson_log(s_n_H[v], mean_identity)
    print(f"Orientation re-check on {n_used} points: r(mapped {label}) = {r_mapped:.4f}, "
          f"r(identity) = {r_identity:.4f}")
    if not (r_mapped > r_identity + 0.2):
        raise RuntimeError(
            f"The mapping read from {ORIENTATION_TXT} ({label}) does not beat the identity "
            f"(r {r_mapped:.4f} vs {r_identity:.4f}). Refusing to query the wrong sightlines.")

    # ---- the real query
    print(f"Querying mean and std at {n_samp} points...")
    mean = query_map(query, true_xyz, "mean")
    std = query_map(query, true_xyz, "std")
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(std > 0, mean / std, np.nan)
    # Shape-independent detection test, formed WITHOUT dividing so it is
    # also defined on voxels with std = 0 (where the SNR is not).
    detected = np.isfinite(mean) & np.isfinite(std) & ((mean - 2.0 * std) > 0.0)
    usable = np.isfinite(mean) & np.isfinite(std)
    print(f"  usable (finite mean and std) for {usable.sum()}/{n_samp} cells "
          f"({1 - usable.mean():.2%} unusable: inside the map's inner cutoff)")
    n_zero_std = int((usable & ~(std > 0)).sum())
    if n_zero_std:
        print(f"  {n_zero_std} usable cells have std = 0 -- SNR undefined there, "
              f"detection test still defined")

    # ---- sanity: cube density vs 1653 x posterior mean
    ratio = np.where(mean > 0, s_n_H / mean, np.nan)
    r_sanity, n_sanity = pearson_log(s_n_H, mean)
    med_ratio = float(np.nanmedian(ratio))
    print(f"Sanity: n_H(cube) / A'(map) median {med_ratio:.1f} "
          f"(expected {N_H_PER_EXTINCTION_F98:.0f}), log-log r = {r_sanity:.4f} "
          f"on {n_sanity} cells")

    # ---- per |z| bin x class table
    res = {}
    for cls, name in ((1, "flagged"), (0, "neutral")):
        for k in ("snr_median", "snr_p15", "snr_p85", "n_sampled"):
            res[f"{k}__{name}"] = np.full(n_bins, np.nan)
        for wname in ("vol", "mw"):
            res[f"frac_detected_{wname}__{name}"] = np.full(n_bins, np.nan)
            for thr in SNR_THRESHOLDS:
                res[f"frac_snr_lt{thr:g}_{wname}__{name}"] = np.full(n_bins, np.nan)

    for b in range(n_bins):
        for cls, name in ((1, "flagged"), (0, "neutral")):
            sel = (s_bin == b) & (s_flag == (cls == 1)) & usable
            res[f"n_sampled__{name}"][b] = int(sel.sum())
            if not sel.any():
                continue
            w_vol = np.ones(sel.sum())
            w_mw = s_n_H[sel]
            q = weighted_quantiles(snr[sel], w_vol, (0.15, 0.50, 0.85))
            res[f"snr_p15__{name}"][b], res[f"snr_median__{name}"][b], \
                res[f"snr_p85__{name}"][b] = q
            for thr in SNR_THRESHOLDS:
                below = snr[sel] < thr
                res[f"frac_snr_lt{thr:g}_vol__{name}"][b] = weighted_fraction(below, w_vol)
                res[f"frac_snr_lt{thr:g}_mw__{name}"][b] = weighted_fraction(below, w_mw)
            res[f"frac_detected_vol__{name}"][b] = weighted_fraction(detected[sel], w_vol)
            res[f"frac_detected_mw__{name}"][b] = weighted_fraction(detected[sel], w_mw)

    # ---- box-wide, STRATUM-WEIGHTED so the stratification does not bias it
    box = {}
    w_vol_box = np.where(usable, s_wstrat, 0.0)
    w_mw_box = np.where(usable, s_wstrat * s_n_H, 0.0)
    for thr in SNR_THRESHOLDS:
        below = snr < thr
        box[f"box_frac_snr_lt{thr:g}_vol"] = weighted_fraction(below, w_vol_box)
        box[f"box_frac_snr_lt{thr:g}_mw"] = weighted_fraction(below, w_mw_box)
    box["box_frac_detected_vol"] = weighted_fraction(detected, w_vol_box)
    box["box_frac_detected_mw"] = weighted_fraction(detected, w_mw_box)
    box["box_snr_median_vol"] = weighted_quantiles(snr, w_vol_box, (0.5,))[0]
    box["box_snr_median_mw"] = weighted_quantiles(snr, w_mw_box, (0.5,))[0]

    # ---- save
    out = dict(res)
    out.update({k: np.array([v], dtype=float) for k, v in box.items()})
    out["abs_z_pc"] = bin_centers
    out["bin_edges"] = bin_edges
    out["bin_width_pc"] = np.array([ABS_Z_BIN_PC], dtype=float)
    out["snr_thresholds"] = np.array(SNR_THRESHOLDS, dtype=float)
    out["pop_counts"] = pop_counts
    out["samp_counts"] = samp_counts
    out["sanity_median_ratio"] = np.array([med_ratio])
    out["sanity_log_r"] = np.array([r_sanity])
    out["orientation_r_mapped"] = np.array([r_mapped])
    out["orientation_r_identity"] = np.array([r_identity])
    np.savez_compressed(OUT_NPZ_PATH, **out)
    print(f"Saved {OUT_NPZ_PATH}")

    # ---- report
    L = [
        PROVISIONAL_CUBE_HEADER,
        "Posterior signal-to-noise of the Edenhofer+23 density, per |z| bin",
        "=" * 104,
        "Map:        Edenhofer2023Query(integrated=False, flavor='main'), modes 'mean' and 'std'.",
        "",
        "            SNR = posterior mean / posterior standard deviation, per voxel, where the",
        "            STD IS THE PUBLISHED PER-VOXEL POSTERIOR STANDARD DEVIATION FROM THE MAP",
        "            ITSELF -- it is NOT computed from posterior samples. load_samples=True is",
        "            not used: samples_healpix.fits is not downloaded, and fetching it is a 19 GB",
        "            download. mean and std are the published first two moments of exactly the",
        "            posterior those samples would come from, so the SNR below is the same",
        "            quantity; what the samples would add is the SHAPE of each voxel's posterior",
        "            beyond its first two moments.",
        "",
        "            'detected' = (mean - 2*std) > 0: a hard detection test phrased without",
        "            dividing, so it assumes nothing about that shape. Wherever std > 0 it is",
        "            algebraically the same cut as SNR > 2 -- i.e. detected = 1 - f(SNR<2) -- and",
        "            it is reported separately because it is also defined on voxels with std = 0,",
        "            where the SNR is not.",
        f"Orientation: {label}, read from {ORIENTATION_TXT} and re-checked here --",
        f"            r(mapped) = {r_mapped:.4f} vs r(identity) = {r_identity:.4f} on {n_used} cells.",
        f"Sample:     {n_samp} cells, stratified: {n_bins} |z| bins x "
        f"{{flagged, non-flagged}} x up to {N_PER_STRATUM}.",
        f"            Box-wide numbers use stratum weights (population/sampled), so they estimate",
        f"            the population and not the sample.",
        f"Sanity:     n_H(cube) / A'(map) median {med_ratio:.1f} (expected "
        f"{N_H_PER_EXTINCTION_F98:.0f}), log-log r = {r_sanity:.4f}.",
        "",
        f"{'|z| bin':>12} | {'HIM-FLAGGED':^61} | {'NON-FLAGGED':^61}",
        f"{'[pc]':>12} | {'med SNR':>8}{'15-85':>14}{'f<1 vol':>9}{'f<1 mw':>9}"
        f"{'det vol':>9}{'det mw':>9} | "
        f"{'med SNR':>8}{'15-85':>14}{'f<1 vol':>9}{'f<1 mw':>9}"
        f"{'det vol':>9}{'det mw':>9}",
        "-" * 140,
    ]
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        cells = []
        for name in ("flagged", "neutral"):
            cells.append(
                f"{res[f'snr_median__{name}'][b]:>8.2f}"
                f"{res[f'snr_p15__{name}'][b]:>7.2f}-{res[f'snr_p85__{name}'][b]:<6.2f}"
                f"{res[f'frac_snr_lt1_vol__{name}'][b]:>9.3f}"
                f"{res[f'frac_snr_lt1_mw__{name}'][b]:>9.3f}"
                f"{res[f'frac_detected_vol__{name}'][b]:>9.3f}"
                f"{res[f'frac_detected_mw__{name}'][b]:>9.3f}")
        L.append(f"{lo:>5.0f}-{hi:<6.0f} | {cells[0]} | {cells[1]}")
    L += [
        "-" * 140,
        "  f<1 = fraction with SNR < 1.  det = fraction with mean - 2*std > 0 "
        "(= 1 - fraction with SNR < 2).",
        "",
        "Box-wide (stratum-weighted, STATS_BOX |x|,|y| <= 500, |z| <= 400 pc):",
        f"  median SNR                     volume-weighted {box['box_snr_median_vol']:.2f}, "
        f"mass-weighted {box['box_snr_median_mw']:.2f}",
    ]
    for thr in SNR_THRESHOLDS:
        L.append(f"  fraction with SNR < {thr:g}            "
                 f"volume {box[f'box_frac_snr_lt{thr:g}_vol']:.4f}, "
                 f"MASS {box[f'box_frac_snr_lt{thr:g}_mw']:.4f}")
    L.append(f"  fraction clearly detected      "
             f"volume {box['box_frac_detected_vol']:.4f}, "
             f"MASS {box['box_frac_detected_mw']:.4f}   (mean - 2*std > 0)")
    L += [
        "",
        "  The MASS fractions are the headline: they are the share of the observed column mass",
        "  in the box whose density is not individually constrained by the data at that level.",
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
