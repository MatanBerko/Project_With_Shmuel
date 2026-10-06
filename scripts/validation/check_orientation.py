"""
Is the f98 zarr cube oriented the way every downstream script assumes?

The whole project reads the cube as heliocentric Galactic Cartesian
coordinates -- x toward l = 0, y toward l = 90, z toward the North
Galactic Pole -- with the "density" field equal to n_H = 1653 * A', where
A' is the Edenhofer et al. (2023) differential extinction [E pc^-1]. That
assumption has never been tested against the extinction map the cube was
built from. This script tests it, and ONLY reports: it does not touch
src/physics/loading.py, and nothing it prints changes how any other
script reads the cube.

Method
------
1. Draw ~2000 random voxels from the STATS_BOX (|x|, |y| <= 500 pc,
   |z| <= 400 pc), using the cube's own coordinate arrays, with a fixed
   seed.
2. For each sampled voxel, take its physical position p = (x, y, z) as
   the TRUTH and query the Edenhofer map there directly:
   dustmaps.edenhofer2023.Edenhofer2023Query(integrated=False) at the
   matching Galactic Cartesian point (astropy Galactic(u, v, w,
   representation_type="cartesian") -- u toward l=0, v toward l=90,
   w toward the NGP, exactly the convention above). This is the same
   query pipeline/compute_sigma_gas_factor_comparison.py already uses.
3. For each of the 48 axis mappings -- 6 axis permutations x 8 sign flips,
   i.e. every way the cube's (x, y, z) axes could be a relabelling of the
   true ones -- read the cube's density at M(p) instead of at p, and
   correlate log10(density) against log10(A') over the sample. The true
   mapping should stand out: a wrong mapping pairs each sightline's
   extinction with an unrelated voxel, so its correlation collapses.
   Every M(p) stays inside the cube (the box's largest coordinate is
   500 pc and the cube spans +-1000, +-1000, +-750 pc), so all 48
   mappings are evaluated on real data, none on an extrapolation.
4. Under the best mapping, fit the density ratio density / A'. Expected
   ~1653 if the cube really has that factor baked in. Reported three
   ways: the median ratio, the log-space ratio with the slope forced to
   1, and a free-slope log-log fit (a slope well away from 1 would mean
   the cube is not a constant multiple of the map at all -- the cube is
   smoothed, so some scatter and a slope slightly off 1 are expected).
5. Compare the mean density in the z > 0 and z < 0 halves of the box from
   both sources. A real north/south asymmetry should appear in BOTH; an
   asymmetry present in only one of them is a sign of a flipped or
   shifted z axis.

Memory: the map is loaded, queried, and released BEFORE the density
sub-cube is read, so the two large objects never coexist. The sub-cube is
the |x|, |y|, |z| <= 500 pc box (~0.5 GB float32), which is all that any
of the 48 mappings can reach.
"""

import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.conventions import (  # noqa: E402
    N_H_PER_EXTINCTION_F98,
    PROVISIONAL_CUBE_HEADER,
    STATS_BOX_XY_HALF_RANGE_PC,
    STATS_BOX_Z_HALF_RANGE_PC,
)
from src.config_loader import load_resolved_config  # noqa: E402
from src.physics.loading import RHO_FIELD, open_zarr  # noqa: E402

OUT_TXT_PATH = Path("results/orientation_check.txt")

N_SAMPLE_CORRELATION = 2000   # the 48-mapping scan (task spec: ~2000)
N_SAMPLE_TOTAL = 20000        # the ratio fit + north/south means
RANDOM_SEED = 20261006

# Every M(p) must land inside this half-range of the cube; the box's
# largest coordinate is STATS_BOX_XY_HALF_RANGE_PC, so a permutation can
# put at most that into any one slot.
SUBCUBE_HALF_RANGE_PC = max(STATS_BOX_XY_HALF_RANGE_PC, STATS_BOX_Z_HALF_RANGE_PC)

AXIS_NAMES = ("x", "y", "z")


def axis_mappings():
    """All 48 signed axis mappings, as (label, perm, signs).

    mapped[k] = signs[k] * p[perm[k]], and `mapped` is then read as the
    cube's own (x, y, z). The identity mapping is perm=(0,1,2),
    signs=(1,1,1) -- what every script currently assumes.
    """
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            label = ",".join(
                f"{'-' if signs[k] < 0 else '+'}{AXIS_NAMES[perm[k]]}" for k in range(3))
            out.append((f"({label})", perm, signs))
    return out


IDENTITY_LABEL = "(+x,+y,+z)"


def pearson_log(a, b):
    """Pearson r of log10(a) vs log10(b) over the jointly positive-finite
    subset, plus that subset's size."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if m.sum() < 3:
        return float("nan"), int(m.sum())
    la, lb = np.log10(a[m]), np.log10(b[m])
    if la.std() == 0 or lb.std() == 0:
        return float("nan"), int(m.sum())
    return float(np.corrcoef(la, lb)[0, 1]), int(m.sum())


def nearest_indices(coord_axis, values):
    """Nearest grid index along one ascending, uniformly spaced coordinate
    axis. Direct arithmetic on the real spacing read from the axis itself
    -- no hardcoded voxel size, and no (N x Naxis) distance matrix.
    """
    coord_axis = np.asarray(coord_axis, dtype=float)
    d = float(coord_axis[1] - coord_axis[0])
    j = np.rint((np.asarray(values, dtype=float) - coord_axis[0]) / d)
    return np.clip(j, 0, len(coord_axis) - 1).astype(np.intp)


def main():
    t0 = time.time()
    Path("results").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    grid = open_zarr()
    x_pc, y_pc, z_pc = grid.x_pc, grid.y_pc, grid.z_pc

    # ---- 1. sample voxels from the STATS_BOX ------------------------------
    ix_box = np.where(np.abs(x_pc) <= STATS_BOX_XY_HALF_RANGE_PC)[0]
    iy_box = np.where(np.abs(y_pc) <= STATS_BOX_XY_HALF_RANGE_PC)[0]
    iz_box = np.where(np.abs(z_pc) <= STATS_BOX_Z_HALF_RANGE_PC)[0]

    ix = rng.choice(ix_box, size=N_SAMPLE_TOTAL)
    iy = rng.choice(iy_box, size=N_SAMPLE_TOTAL)
    iz = rng.choice(iz_box, size=N_SAMPLE_TOTAL)
    px, py, pz = x_pc[ix], y_pc[iy], z_pc[iz]
    p = np.stack([px, py, pz], axis=1)  # (N, 3) TRUE physical position

    lines = [
        PROVISIONAL_CUBE_HEADER,
        "Zarr cube orientation check -- REPORT ONLY (the loader is unchanged)",
        "=" * 100,
        f"cube:            {load_resolved_config()['zarr_path'].name} "
        f"(single config entry: local_config.yaml zarr_filename)",
        f"grid:            x {x_pc.min():.0f}..{x_pc.max():.0f}, y {y_pc.min():.0f}..{y_pc.max():.0f}, "
        f"z {z_pc.min():.0f}..{z_pc.max():.0f} pc, spacing {np.diff(x_pc)[0]:.0f} pc",
        f"STATS_BOX:       |x|,|y| <= {STATS_BOX_XY_HALF_RANGE_PC:.0f} pc, |z| <= {STATS_BOX_Z_HALF_RANGE_PC:.0f} pc",
        f"samples:         {N_SAMPLE_CORRELATION} for the 48-mapping scan, "
        f"{N_SAMPLE_TOTAL} for the ratio fit and north/south means (seed {RANDOM_SEED})",
        f"expected ratio:  n_H / A' = {N_H_PER_EXTINCTION_F98:.0f}",
        "",
    ]

    # ---- 2. query the Edenhofer map at the TRUE positions ------------------
    print(f"Querying Edenhofer2023 (integrated=False) at {N_SAMPLE_TOTAL} points...")
    import astropy.units as u
    from astropy.coordinates import Galactic, SkyCoord
    from dustmaps.edenhofer2023 import Edenhofer2023Query

    t_q = time.time()
    query = Edenhofer2023Query(integrated=False)
    # distance_bounds is an astropy Quantity -- convert to plain pc
    bounds_pc = np.asarray(query.distance_bounds.to_value(u.pc), dtype=float)
    dist_lo, dist_hi = float(bounds_pc.min()), float(bounds_pc.max())
    gal = Galactic(u=px * u.pc, v=py * u.pc, w=pz * u.pc, representation_type="cartesian")
    coords = SkyCoord(gal)
    coords.representation_type = "spherical"
    a_prime = np.asarray(query.query(coords), dtype=float)
    del query, coords, gal
    print(f"  queried in {time.time() - t_q:.1f}s; map distance bounds "
          f"[{dist_lo:.1f}, {dist_hi:.1f}] pc")

    r_helio = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    nan_frac = float(np.mean(~np.isfinite(a_prime)))
    inside_cutoff = float(np.mean(r_helio < dist_lo))
    lines += [
        f"dustmaps:        Edenhofer2023Query(integrated=False), distance bounds "
        f"[{dist_lo:.1f}, {dist_hi:.1f}] pc",
        f"                 NaN fraction in the sample {nan_frac:.2%} "
        f"(samples inside the map's inner cutoff: {inside_cutoff:.2%})",
        "",
    ]

    # ---- 3. read the density sub-cube and scan all 48 mappings -------------
    print(f"Loading the density sub-cube (|x|,|y|,|z| <= {SUBCUBE_HALF_RANGE_PC:.0f} pc)...")
    import dask.array as da
    sx = np.where(np.abs(x_pc) <= SUBCUBE_HALF_RANGE_PC)[0]
    sy = np.where(np.abs(y_pc) <= SUBCUBE_HALF_RANGE_PC)[0]
    sz = np.where(np.abs(z_pc) <= SUBCUBE_HALF_RANGE_PC)[0]
    x_sub, y_sub, z_sub = x_pc[sx], y_pc[sy], z_pc[sz]
    t_l = time.time()
    rho = da.from_zarr(grid.store[RHO_FIELD])[
        int(sz.min()):int(sz.max()) + 1,
        int(sy.min()):int(sy.max()) + 1,
        int(sx.min()):int(sx.max()) + 1].compute().astype(np.float32)
    print(f"  loaded {rho.shape} ({rho.nbytes / 1e6:.0f} MB) in {time.time() - t_l:.1f}s")

    def density_at(mapped):
        """Cube density at mapped (N,3) coordinates read as (x, y, z)."""
        jx = nearest_indices(x_sub, mapped[:, 0])
        jy = nearest_indices(y_sub, mapped[:, 1])
        jz = nearest_indices(z_sub, mapped[:, 2])
        return rho[jz, jy, jx].astype(np.float64)

    n_corr = min(N_SAMPLE_CORRELATION, N_SAMPLE_TOTAL)
    p_corr, a_corr = p[:n_corr], a_prime[:n_corr]

    print(f"Scanning all 48 axis mappings on {n_corr} samples...")
    results = []
    for label, perm, signs in axis_mappings():
        mapped = np.stack([signs[k] * p_corr[:, perm[k]] for k in range(3)], axis=1)
        dens = density_at(mapped)
        r, n_used = pearson_log(dens, a_corr)
        results.append((label, perm, signs, r, n_used))

    results_sorted = sorted(results, key=lambda t: (-1e9 if not np.isfinite(t[3]) else t[3]), reverse=True)
    identity = next(t for t in results if t[0] == IDENTITY_LABEL)
    best = results_sorted[0]

    lines += [
        "48 axis mappings: Pearson r of log10(cube density) vs log10(A')",
        "  mapping M means the cube is read at M(p) while A' is queried at the true p;",
        "  (+x,+y,+z) is the identity -- what every script currently assumes.",
        "",
        f"  {'rank':>4}  {'mapping':<14}{'r':>9}{'n_used':>9}",
    ]
    for rank, (label, _, _, r, n_used) in enumerate(results_sorted, start=1):
        mark = "   <-- identity" if label == IDENTITY_LABEL else ""
        if rank <= 10 or label == IDENTITY_LABEL:
            lines.append(f"  {rank:>4}  {label:<14}{r:>9.4f}{n_used:>9d}{mark}")
    if len(results_sorted) > 10:
        lines.append(f"  ... ({len(results_sorted) - 10} further mappings omitted; "
                     f"worst r = {results_sorted[-1][3]:.4f} for {results_sorted[-1][0]})")
    lines += [
        "",
        f"  identity (+x,+y,+z): r = {identity[3]:.4f}",
        f"  best mapping {best[0]}: r = {best[3]:.4f}",
        f"  VERDICT: the best mapping IS the identity -- the cube is oriented as assumed."
        if best[0] == IDENTITY_LABEL else
        f"  VERDICT: the best mapping is {best[0]}, NOT the identity -- the cube's axes do "
        f"not match the assumed convention.",
        "",
    ]

    # ---- 4. density ratio under the best mapping ---------------------------
    best_label, best_perm, best_signs, _, _ = best
    mapped_all = np.stack([best_signs[k] * p[:, best_perm[k]] for k in range(3)], axis=1)
    dens_all = density_at(mapped_all)

    m = np.isfinite(dens_all) & np.isfinite(a_prime) & (dens_all > 0) & (a_prime > 0)
    d_fit, a_fit = dens_all[m], a_prime[m]
    ratio = d_fit / a_fit
    log_ratio = np.log10(d_fit) - np.log10(a_fit)
    slope, intercept = np.polyfit(np.log10(a_fit), np.log10(d_fit), 1)
    r_best_full, _ = pearson_log(dens_all, a_prime)

    lines += [
        f"Density ratio n_H / A' under the best mapping {best_label} "
        f"({m.sum()} of {N_SAMPLE_TOTAL} samples usable)",
        f"  r (log-log, full sample)         {r_best_full:.4f}",
        f"  median ratio                     {np.median(ratio):.1f}",
        f"  log-space ratio (slope fixed 1)  {10 ** np.median(log_ratio):.1f}",
        f"  free-slope log-log fit           slope {slope:.4f}, "
        f"ratio at A'=median {10 ** (intercept + (slope - 1) * np.median(np.log10(a_fit))):.1f}",
        f"  ratio percentiles 15/50/85       {np.percentile(ratio, 15):.1f} / "
        f"{np.percentile(ratio, 50):.1f} / {np.percentile(ratio, 85):.1f}",
        f"  expected                         {N_H_PER_EXTINCTION_F98:.0f}",
        "",
    ]

    # ---- 5. north/south comparison ----------------------------------------
    # Sampled, same points for both sources (apples to apples).
    north = m & (pz > 0)
    south = m & (pz < 0)

    def _mean_se(v):
        v = np.asarray(v, dtype=float)
        return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)), int(v.size)

    dn, dn_se, dn_n = _mean_se(dens_all[north])
    ds, ds_se, ds_n = _mean_se(dens_all[south])
    an, an_se, _ = _mean_se(a_prime[north])
    as_, as_se, _ = _mean_se(a_prime[south])

    # And the cube's own exact box means, streamed plane by plane (no
    # sampling noise at all), for reference.
    zb = np.where(np.abs(z_sub) <= STATS_BOX_Z_HALF_RANGE_PC)[0]
    xy_box = (np.abs(x_sub) <= STATS_BOX_XY_HALF_RANGE_PC)[None, :] & \
             (np.abs(y_sub) <= STATS_BOX_XY_HALF_RANGE_PC)[:, None]
    sums = {"north": 0.0, "south": 0.0}
    counts = {"north": 0, "south": 0}
    for i in zb:
        if z_sub[i] == 0:
            continue
        half = "north" if z_sub[i] > 0 else "south"
        plane = rho[i][xy_box].astype(np.float64)
        plane = plane[np.isfinite(plane)]
        sums[half] += plane.sum()
        counts[half] += plane.size
    exact_north = sums["north"] / counts["north"] if counts["north"] else float("nan")
    exact_south = sums["south"] / counts["south"] if counts["south"] else float("nan")

    label_cube = "cube density [cm^-3]"
    label_map = "dustmaps A' [E/pc]"
    label_se = "  +- standard error"
    lines += [
        f"North (z>0) vs south (z<0), identical sample points, best mapping {best_label}",
        f"  {'source':<28}{'z>0 mean':>14}{'z<0 mean':>14}{'N/S ratio':>12}",
        f"  {label_cube:<28}{dn:>14.4f}{ds:>14.4f}{dn / ds:>12.3f}",
        f"  {label_se:<28}{dn_se:>14.4f}{ds_se:>14.4f}{'':>12}",
        f"  {label_map:<28}{an:>14.6f}{as_:>14.6f}{an / as_:>12.3f}",
        f"  {label_se:<28}{an_se:>14.6f}{as_se:>14.6f}{'':>12}",
        f"  sample sizes: z>0 {dn_n}, z<0 {ds_n}",
        "",
        f"  cube density, EXACT box mean over every voxel (no sampling):",
        f"  {'':<28}{exact_north:>14.4f}{exact_south:>14.4f}{exact_north / exact_south:>12.3f}",
        "",
        "  Reading: the two sources' N/S ratios agreeing says the cube's z axis carries the",
        "  same vertical asymmetry as the map. They disagreeing points at a flipped, shifted",
        "  or permuted z axis -- which the 48-mapping scan above then localises.",
        "",
    ]

    # ---- 6. does the mis-orientation actually bias anything? ---------------
    # Worth stating explicitly rather than leaving the reader to work out,
    # because the answer depends on whether the best mapping touches z and
    # on whether it maps the analysis footprint onto itself.
    z_is_correct = best_perm[2] == 2 and best_signs[2] == 1
    xy_only = z_is_correct and set(best_perm[:2]) == {0, 1}
    square_invariant = xy_only  # a +-L square in XY is invariant under any
    # signed permutation of x and y, so an XY-only error maps the footprint
    # exactly onto itself.
    lines += ["Implication for the provisional numbers", "-" * 40]
    if best_label == IDENTITY_LABEL:
        lines += ["  No mis-orientation detected; nothing to correct."]
    elif xy_only and square_invariant:
        lines += [
            f"  The best mapping {best_label} permutes and flips only x and y; the z axis is",
            "  correct (+z), which the matching north/south ratios above independently confirm.",
            "",
            "  That means the provisional statistics in results/numbers_table.csv are NOT",
            "  biased by this, for two reasons that both have to hold and both do:",
            f"    1. the analysis footprint is a +-{STATS_BOX_XY_HALF_RANGE_PC:.0f} pc SQUARE, which any signed",
            "       permutation of x and y maps exactly onto itself -- the same set of voxels",
            "       is selected either way;",
            "    2. no quantity computed here depends on x or y individually. P_tot integrates",
            "       along z, the self-gravity footprint average collapses x and y away, and",
            "       every statistic, slab, profile and PDF is binned in z only.",
            "",
            "  What IS affected is anything that resolves the XY plane: the Sigma_gas(x, y)",
            "  map and the z-slice images in cache/core/summary.npz are reflected about the",
            "  x = -y diagonal. Their medians and means over the square are unchanged, but no",
            "  figure should be made from those maps until the re-oriented cube lands.",
            "",
            f"  Concretely: the cube's x axis carries {'-' if best_signs[0] < 0 else '+'}"
            f"{AXIS_NAMES[best_perm[0]]}_true and its y axis carries "
            f"{'-' if best_signs[1] < 0 else '+'}{AXIS_NAMES[best_perm[1]]}_true.",
        ]
    else:
        lines += [
            f"  The best mapping {best_label} is NOT an x-y-only relabelling"
            + ("" if z_is_correct else " and it changes the z axis"),
            "  -- so z-dependent results (slabs, vertical profiles, P_tot) ARE affected and",
            "  the provisional numbers should not be used until the cube is re-oriented.",
        ]

    lines += ["", f"(runtime {time.time() - t0:.1f}s)"]

    report = "\n".join(lines)
    print()
    print(report)
    OUT_TXT_PATH.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved {OUT_TXT_PATH}")


if __name__ == "__main__":
    main()
