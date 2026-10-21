"""
DIAGNOSTIC: how much of the midplane <P_tot>_area comes from the thin, high-|z|
gas tail, and how well is that tail measured?

results/ptot_budget.txt (Fig. 3) found that gas at |z| > 400 pc supplies 34.9%
of P(0+) and 23.2% of P(0-) -- outside the statistics box but inside the
integration column (the full cube, z = -750..+750 pc, P = 0 at both edges).

  (A) Posterior SNR of the Edenhofer+23 map out to the cube edge, per 20 pc
      |z| bin -- is the tail measured or prior-dominated?
  (B) Cut-height dependence: P(0+-) and the |z| <= 30 pc slab alpha_area_vol
      with the density set to zero beyond +-z_top.
  (C) Weight beyond the cube edge: C1 a direct Edenhofer+23 query for
      750 < |z| <= 1000 pc, C2 an exponential fit to the cube's 400-750 pc tail
      extended to 1500 pc.

Measures only: no physics, cache, existing result or existing figure changes.
Fiducial as Fig. 3 (RAW, all footprint cells incl. HIM-flagged, footprint_mean
self-gravity, Sun-centred midplane, the SOURCE cube's float64 z grid). The
1-D machinery is reused from scripts/compute/compute_ptot_budget.py, the map
machinery from scripts/validation/posterior_snr.py (imported, not edited).

Writes results/high_z_tail.txt and cache/high_z_tail/high_z_tail.npz.
"""

import importlib.util
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
from src.physics import gravity, hydrostatic  # noqa: E402
from src.physics.loading import open_zarr, slab_z_indices  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════
# ██  CONTROL BLOCK
# ═══════════════════════════════════════════════════════════════════════
VARIANT = "RAW"
ALPHA_CORE_ZARR_PATH = Path("cache/core/alpha_core.zarr")
AREA_ALPHA_NPZ_PATH = Path("cache/core/area_alpha.npz")
PTOT_BUDGET_NPZ_PATH = Path("cache/ptot_budget/ptot_budget.npz")
POSTERIOR_SNR_TXT_PATH = Path("results/posterior_snr.txt")   # its cache is not on disk
OUT_NPZ_PATH = Path("cache/high_z_tail/high_z_tail.npz")
OUT_TXT_PATH = Path("results/high_z_tail.txt")

# (A)
ABS_Z_MAX_PC = 760.0                 # last bin 740-760 (the cube ends at 750)
# (B)
Z_TOPS_PC = (300.0, 400.0, 500.0, 600.0, 750.0)
SLAB_HALF_PC = 30.0
# (C1)
C1_XY_STEP_PC = 10.0
C1_Z_STEP_PC = 10.0
C1_Z_MAX_PC = 1000.0
C1_CONTINUITY_Z_PC = (740.0, 750.0, 760.0)   # queried as well, for the 740-760 check
C1_CONTINUITY_TOL = 0.20             # report; > factor 2 -> STOP
C1_SNR_BIN_PC = 50.0
# (C2)
C2_FIT_RANGE_PC = (400.0, 750.0)
C2_Z_MAX_PC = 1500.0
C2_Z_STEP_PC = 2.0
# assertions
REPRO_RTOL = 1e-6

_ROOT = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cpb = _load("compute_ptot_budget", "scripts/compute/compute_ptot_budget.py")
psnr = _load("posterior_snr", "scripts/validation/posterior_snr.py")


# ═══════════════════════════════════════════════════════════════════════
# ██  1-D HELPERS (on top of compute_ptot_budget / src.physics)
# ═══════════════════════════════════════════════════════════════════════
def budget(z, nH, slab_idx=None):
    """P(0+), P(0-), their self-gravity parts and the full P_sum(z) profile
    for a density profile nH on a z grid symmetric about 0 (gas self-gravity
    from THIS profile, stars/DM Guo+20 unchanged; P = 0 at the grid edges)."""
    rho = gravity.RHO_PER_N * nH
    sig = cpb.sigma_components(z, nH)
    g_tot = cpb.g_from_sigma(sig["star"] + sig["dm"] + sig["gas"])
    g_gas = cpb.g_from_sigma(sig["gas"])
    top, bot = cpb.pressure_branches(z, rho, g_tot)
    sg_top, sg_bot = cpb.pressure_branches(z, rho, g_gas)
    P = hydrostatic.p_tot_kb_full_column(z, rho, g_tot)
    out = {"P0p": top, "P0m": bot, "sg0p": sg_top, "sg0m": sg_bot, "P": P,
           "sigma_total": float(sig["gas"][-1])}
    if slab_idx is not None:
        out["slab_P"] = float(np.mean(P[slab_idx]))
    return out


def extend_profile(z_cube, nH_cube, z_ext_pos, n_ext_pos, z_ext_neg, n_ext_neg):
    """Cube profile plus a tail on each side.

    Each tail grid STARTS AT THE CUBE EDGE (|z| = 750) with the tail's own
    density there, so the edge point appears twice: once with the cube's
    density, once with the tail's. The zero-width interval between them
    carries no weight, so the cube part is integrated exactly as in Fig. 3
    (over [-750, 750]) and the tail purely from its own profile. Without
    the duplicate, the first tail interval would be a trapezoid from the
    cube's density down to the tail's -- for a zero tail that alone adds a
    half cell of column and weight that Fig. 3 does not have (1e-3 in P).
    """
    if not (np.isclose(z_ext_pos[0], z_cube[-1]) and np.isclose(z_ext_neg[0], -z_cube[0])):
        raise RuntimeError("each tail grid must start at the cube edge")
    z = np.concatenate([-z_ext_neg[::-1], z_cube, z_ext_pos])
    n = np.concatenate([n_ext_neg[::-1], nH_cube, n_ext_pos])
    if not np.allclose(z, -z[::-1], atol=1e-9):
        raise RuntimeError("extended grid is not symmetric about 0")
    return z, n


def read_posterior_snr_table(path=POSTERIOR_SNR_TXT_PATH):
    """{class: {"med": [...], "det_mw": [...]}} for the 0-400 pc bins of
    results/posterior_snr.txt (two decimals / three decimals as printed)."""
    out = {"flagged": {"med": [], "det_mw": []}, "neutral": {"med": [], "det_mw": []}}
    for line in path.read_text().splitlines():
        parts = line.split("|")
        if len(parts) != 3 or "-" not in parts[0] or not parts[0].strip()[0].isdigit():
            continue
        for name, part in (("flagged", parts[1]), ("neutral", parts[2])):
            tok = part.split()
            out[name]["med"].append(float(tok[0]))
            out[name]["det_mw"].append(float(tok[5]))
    return {k: {q: np.array(v) for q, v in d.items()} for k, d in out.items()}


def column_msun(z, nH):
    return float(np.trapezoid(nH * gravity.RHO_MSUN_PC3_PER_N, z))


# ═══════════════════════════════════════════════════════════════════════
# ██  MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    checks = []

    def check(name, measured, threshold):
        checks.append((name, measured, threshold))
        print(f"  check {name}: {measured:.3e} (threshold {threshold:.0e})")
        if not (np.isfinite(measured) and measured <= threshold):
            raise RuntimeError(f"ASSERTION FAILED -- {name}: {measured:.3e}. STOPPING.")

    pb = np.load(PTOT_BUDGET_NPZ_PATH, allow_pickle=False)
    aa = np.load(AREA_ALPHA_NPZ_PATH, allow_pickle=False)
    z = np.asarray(open_zarr().z_pc, dtype=np.float64)        # SOURCE float64 grid
    if not np.array_equal(pb["z_pc"], z):
        raise RuntimeError("ptot_budget.npz z grid differs from the source grid")
    nH = pb["nH_area"]
    P0p_fig3, P0m_fig3 = (float(v) for v in pb["P0_total"])
    slab_idx = slab_z_indices(z, 0.0, SLAB_HALF_PC)
    pth_slab = float(aa["slab__mean__Pth_neutral_vol"][0])
    alpha_slab_2e = float(aa["slab__mean__alpha_area_vol"][0])
    ptot_slab_2e = float(aa["slab__mean__Ptot_area"][0])
    out = {"z_pc": z, "nH_area": nH}

    # ================================================================= (B)
    print("\n(B) cut-height dependence")
    cut = {}
    for zt in Z_TOPS_PC:
        n_t = np.where(np.abs(z) <= zt, nH, 0.0)
        r = budget(z, n_t, slab_idx)
        r["alpha_slab"] = r["slab_P"] / pth_slab
        r["sigma_within"] = column_msun(z, n_t)
        cut[zt] = r
        out[f"cut{int(zt)}__P"] = r["P"]
    full = cut[Z_TOPS_PC[-1]]
    check("z_top = 750: P(0+) vs Fig. 3 (rel)", abs(full["P0p"] / P0p_fig3 - 1), REPRO_RTOL)
    check("z_top = 750: P(0-) vs Fig. 3 (rel)", abs(full["P0m"] / P0m_fig3 - 1), REPRO_RTOL)
    check("z_top = 750: slab <P_tot>_area vs area_alpha (rel)",
          abs(full["slab_P"] / ptot_slab_2e - 1), REPRO_RTOL)
    check("z_top = 750: slab alpha_area_vol vs area_alpha (rel)",
          abs(full["alpha_slab"] / alpha_slab_2e - 1), REPRO_RTOL)

    # ================================================================= (C2)
    print("\n(C2) exponential fit to the 400-750 pc tail")
    c2 = {}
    z_ext = np.arange(z[-1], C2_Z_MAX_PC + 0.5 * C2_Z_STEP_PC, C2_Z_STEP_PC)   # starts at 750
    for side, sgn in (("pos", 1.0), ("neg", -1.0)):
        m = (sgn * z >= C2_FIT_RANGE_PC[0]) & (sgn * z <= C2_FIT_RANGE_PC[1])
        slope, icpt = np.polyfit(np.abs(z[m]), np.log(nH[m]), 1)
        c2[side] = {"n0": float(np.exp(icpt)), "h": float(-1.0 / slope),
                    "n_ext": np.exp(icpt) * np.exp(-z_ext * (-slope)),
                    "rms_dex": float(np.std(np.log10(nH[m]) - (icpt + slope * np.abs(z[m]))
                                            / np.log(10)))}
        print(f"  {side}: n0 = {c2[side]['n0']:.4f} cm^-3, h = {c2[side]['h']:.1f} pc")
    z2, n2 = extend_profile(z, nH, z_ext, c2["pos"]["n_ext"], z_ext, c2["neg"]["n_ext"])
    # zero-tail control on the SAME extended grid must reproduce Fig. 3
    z2_0, n2_0 = extend_profile(z, nH, z_ext, 0 * z_ext, z_ext, 0 * z_ext)
    r0 = budget(z2_0, n2_0)
    check("extended grid, tail = 0: P(0+) vs Fig. 3 (rel)", abs(r0["P0p"] / P0p_fig3 - 1), REPRO_RTOL)
    check("extended grid, tail = 0: P(0-) vs Fig. 3 (rel)", abs(r0["P0m"] / P0m_fig3 - 1), REPRO_RTOL)
    slab2 = np.where(np.abs(z2) <= SLAB_HALF_PC)[0]
    r2 = budget(z2, n2, slab2)
    r2["alpha_slab"] = r2["slab_P"] / pth_slab
    c2_extra = {side: column_msun(z_ext, c2[side]["n_ext"]) for side in ("pos", "neg")}
    out.update({"c2_z": z2, "c2_nH": n2, "c2_P": r2["P"],
                "c2_n0_h_pos": np.array([c2["pos"]["n0"], c2["pos"]["h"]]),
                "c2_n0_h_neg": np.array([c2["neg"]["n0"], c2["neg"]["h"]])})

    # ================================================================= (A) + (C1): the map
    print("\n(A) posterior SNR, stratified sample over the full cube z range")
    label, M, _ = psnr.read_best_mapping()
    core = zarr.open_group(str(ALPHA_CORE_ZARR_PATH), mode="r")
    x_pc = np.asarray(core["x_pc"][:], float)
    y_pc = np.asarray(core["y_pc"][:], float)
    vg = core[VARIANT]
    n_cube = np.asarray(vg["n_model"][:], dtype=np.float32)
    him = np.asarray(vg["him"][:]).astype(bool)
    # Stratify on the SAME z values posterior_snr.py binned on -- the cache's
    # z_pc copy. The planes sit exactly on the 20 pc bin edges, and the source
    # float64 grid is antisymmetric only to ~2e-13, so binning on it moves
    # some edge planes into the neighbouring bin and the draws diverge from
    # posterior_snr's. This is sampling only: every z INTEGRAL in this script
    # uses the source float64 grid.
    z_strat = np.asarray(core["z_pc"][:], dtype=np.float64)
    if not np.allclose(z_strat, z, atol=1e-6):
        raise RuntimeError("cached z_pc does not match the source grid")
    abs_z = np.abs(z_strat)
    n_bins = int(round(ABS_Z_MAX_PC / psnr.ABS_Z_BIN_PC))
    edges = np.arange(n_bins + 1, dtype=float) * psnr.ABS_Z_BIN_PC
    centers = 0.5 * (edges[:-1] + edges[1:])

    # --- stratified sampling: copied from posterior_snr.py main() (same seed,
    #     same per-stratum count, same bin order), only extended to 760 pc
    rng = np.random.default_rng(psnr.RANDOM_SEED)
    samples, pop = [], np.zeros((n_bins, 2), dtype=np.int64)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        planes = np.where((abs_z >= lo) & ((abs_z < hi) if b < n_bins - 1 else (abs_z <= hi)))[0]
        if planes.size == 0:
            continue
        him_b = him[planes]
        for cls, flagged in ((1, True), (0, False)):
            mask = him_b if flagged else ~him_b
            n_pop = int(mask.sum())
            pop[b, cls] = n_pop
            if n_pop == 0:
                continue
            take = min(psnr.N_PER_STRATUM, n_pop)
            pick = rng.choice(np.flatnonzero(mask.ravel()), size=take, replace=False)
            p_iz, p_iy, p_ix = np.unravel_index(pick, mask.shape)
            for k in range(take):
                samples.append((planes[p_iz[k]], p_iy[k], p_ix[k], b, cls, n_pop / take))
    s = np.array(samples, dtype=float)
    s_iz, s_iy, s_ix = s[:, 0].astype(int), s[:, 1].astype(int), s[:, 2].astype(int)
    s_bin, s_flag, s_w = s[:, 3].astype(int), s[:, 4].astype(bool), s[:, 5]
    cube_xyz = np.stack([x_pc[s_ix], y_pc[s_iy], z_strat[s_iz]], axis=1)
    true_xyz = psnr.cube_to_true(M, cube_xyz)
    s_n = n_cube[s_iz, s_iy, s_ix].astype(np.float64)
    del n_cube, him
    print(f"  {len(s)} sampled cells, {n_bins} |z| bins")

    from dustmaps.edenhofer2023 import Edenhofer2023Query
    query = Edenhofer2023Query(integrated=False, flavor="main")
    v = rng.choice(len(s), size=min(psnr.N_VERIFY, len(s)), replace=False)
    r_map, _ = psnr.pearson_log(s_n[v], psnr.query_map(query, true_xyz[v], "mean"))
    r_id, _ = psnr.pearson_log(s_n[v], psnr.query_map(query, cube_xyz[v], "mean"))
    print(f"  orientation re-check: r(mapped {label}) = {r_map:.4f}, r(identity) = {r_id:.4f}")
    if not r_map > r_id + 0.2:
        raise RuntimeError("orientation check failed -- STOPPING")
    mean = psnr.query_map(query, true_xyz, "mean")
    std = psnr.query_map(query, true_xyz, "std")
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(std > 0, mean / std, np.nan)
    usable = np.isfinite(mean) & np.isfinite(std)
    detected = usable & ((mean - 2.0 * std) > 0)

    groups = {"all": np.ones(len(s), bool), "flagged": s_flag, "neutral": ~s_flag}
    A = {g: {k: np.full(n_bins, np.nan) for k in
             ("med", "p15", "p85", "f1_vol", "f1_mw", "det_vol", "det_mw", "n")} for g in groups}
    for b in range(n_bins):
        for g, gm in groups.items():
            sel = (s_bin == b) & gm & usable
            if not sel.any():
                continue
            # within a bin the classes are stratified, so "all" uses stratum
            # weights; within one class every sampled cell has the same weight
            wv = s_w[sel] if g == "all" else np.ones(sel.sum())
            wm = wv * s_n[sel]
            q = psnr.weighted_quantiles(snr[sel], wv, (0.15, 0.5, 0.85))
            A[g]["p15"][b], A[g]["med"][b], A[g]["p85"][b] = q
            A[g]["f1_vol"][b] = psnr.weighted_fraction(snr[sel] < 1, wv)
            A[g]["f1_mw"][b] = psnr.weighted_fraction(snr[sel] < 1, wm)
            A[g]["det_vol"][b] = psnr.weighted_fraction(detected[sel], wv)
            A[g]["det_mw"][b] = psnr.weighted_fraction(detected[sel], wm)
            A[g]["n"][b] = sel.sum()
    zone = {"0-400": centers < STATS_BOX_Z_HALF_RANGE_PC, "400-750": centers > STATS_BOX_Z_HALF_RANGE_PC}
    shares = {}
    for zn, bm in zone.items():
        sel = np.isin(s_bin, np.where(bm)[0]) & usable
        wm = s_w[sel] * s_n[sel]
        shares[zn] = {"lt1": psnr.weighted_fraction(snr[sel] < 1, wm),
                      "lt2": psnr.weighted_fraction(snr[sel] < 2, wm),
                      "med_mw": psnr.weighted_quantiles(snr[sel], wm, (0.5,))[0],
                      "med_vol": psnr.weighted_quantiles(snr[sel], s_w[sel], (0.5,))[0]}
    # comparison with posterior_snr.txt's 0-400 bins
    ref = read_posterior_snr_table()
    cmp_rows = []
    for name in ("flagged", "neutral"):
        nref = ref[name]["med"].size
        d_med = np.abs(np.round(A[name]["med"][:nref], 2) - ref[name]["med"])
        d_det = np.abs(np.round(A[name]["det_mw"][:nref], 3) - ref[name]["det_mw"])
        cmp_rows.append((name, float(d_med[:nref - 1].max()), float(d_det[:nref - 1].max()),
                         float(d_med[nref - 1]), float(d_det[nref - 1])))
    for g in groups:
        for k, arr in A[g].items():
            out[f"A__{g}__{k}"] = arr
    out["A_abs_z_pc"] = centers

    # ---- C1: direct query beyond the cube edge
    print("\n(C1) direct Edenhofer query, 750 < |z| <= 1000 pc")
    xy = np.arange(-STATS_BOX_XY_HALF_RANGE_PC, STATS_BOX_XY_HALF_RANGE_PC + 0.5, C1_XY_STEP_PC)
    zq = np.arange(z[-1] + C1_Z_STEP_PC, C1_Z_MAX_PC + 0.5, C1_Z_STEP_PC)
    zlev = np.unique(np.concatenate([np.array(C1_CONTINUITY_Z_PC), zq]))
    X, Y = np.meshgrid(xy, xy)
    bounds = np.asarray(query.distance_bounds.to_value("pc"), dtype=float)
    c1 = {}
    for side, sgn in (("pos", 1.0), ("neg", -1.0)):
        rows = {"z": [], "nH": [], "cov": [], "snr_med": [], "snr_all": [], "zz": []}
        for zz in zlev:
            pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, sgn * zz)], axis=1)
            d = np.linalg.norm(pts, axis=1)
            inside = (d >= bounds.min()) & (d <= bounds.max())
            mq = np.full(len(pts), np.nan)
            sq = np.full(len(pts), np.nan)
            if inside.any():
                mq[inside] = psnr.query_map(query, pts[inside], "mean")
                sq[inside] = psnr.query_map(query, pts[inside], "std")
            ok = np.isfinite(mq) & np.isfinite(sq)
            rows["z"].append(zz)
            rows["cov"].append(float(ok.mean()))
            rows["nH"].append(float(N_H_PER_EXTINCTION_F98 * np.mean(mq[ok])) if ok.any() else np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                sn = np.where(sq[ok] > 0, mq[ok] / sq[ok], np.nan)
            rows["snr_all"].append(sn)
            rows["zz"].append(np.full(sn.size, zz))
        c1[side] = {k: np.array(v) for k, v in rows.items() if k not in ("snr_all", "zz")}
        c1[side]["snr"] = np.concatenate(rows["snr_all"])
        c1[side]["snr_z"] = np.concatenate(rows["zz"])
        # continuity: query at 740/750 vs the cube's planes 740..750
        cz = (np.abs(z) >= 740) & (np.abs(z) <= 750) & (np.sign(z) == sgn)
        qz = np.isin(c1[side]["z"], (740.0, 750.0))
        c1[side]["continuity"] = float(np.mean(c1[side]["nH"][qz]) / np.mean(nH[cz]))
        print(f"  {side}: coverage {c1[side]['cov'].min():.3f}-{c1[side]['cov'].max():.3f}, "
              f"continuity query/cube at 740-750 = {c1[side]['continuity']:.3f}")
        if not 0.5 <= c1[side]["continuity"] <= 2.0:
            raise RuntimeError(f"C1 continuity off by more than a factor of 2 on the {side} side "
                               f"({c1[side]['continuity']:.3f}) -- STOPPING before using C1.")
    # extended profile: cube to 750, then the query beyond
    ext = {}
    for side in ("pos", "neg"):
        sel = c1[side]["z"] >= z[-1]                 # starts at 750 (queried)
        ext[side] = (c1[side]["z"][sel], c1[side]["nH"][sel])
    z1, n1 = extend_profile(z, nH, ext["pos"][0], ext["pos"][1], ext["neg"][0], ext["neg"][1])
    slab1 = np.where(np.abs(z1) <= SLAB_HALF_PC)[0]
    r1 = budget(z1, n1, slab1)
    r1["alpha_slab"] = r1["slab_P"] / pth_slab
    z1_0, n1_0 = extend_profile(z, nH, ext["pos"][0], 0 * ext["pos"][1], ext["neg"][0], 0 * ext["neg"][1])
    r10 = budget(z1_0, n1_0)
    check("C1 grid, tail = 0: P(0+) vs Fig. 3 (rel)", abs(r10["P0p"] / P0p_fig3 - 1), REPRO_RTOL)
    check("C1 grid, tail = 0: P(0-) vs Fig. 3 (rel)", abs(r10["P0m"] / P0m_fig3 - 1), REPRO_RTOL)
    c1_extra = {side: column_msun(*ext[side]) for side in ("pos", "neg")}
    for side in ("pos", "neg"):
        out[f"c1_{side}_z"] = c1[side]["z"]
        out[f"c1_{side}_nH"] = c1[side]["nH"]
        out[f"c1_{side}_cov"] = c1[side]["cov"]
        out[f"c1_{side}_snr"] = c1[side]["snr"]
        out[f"c1_{side}_snr_z"] = c1[side]["snr_z"]
    out["c1_z"], out["c1_P"] = z1, r1["P"]
    out["fig3_P"] = pb["P_sum"]
    out["P0_fig3"] = np.array([P0p_fig3, P0m_fig3])
    out["P0_c2"] = np.array([r2["P0p"], r2["P0m"]])
    out["P0_c1"] = np.array([r1["P0p"], r1["P0m"]])
    out["provisional"] = np.array([PROVISIONAL_CUBE_HEADER])
    OUT_NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ_PATH, **out)
    print(f"Saved {OUT_NPZ_PATH}")

    # ================================================================= report
    pct = lambda a, b: f"{(a / b - 1):+.1%}"  # noqa: E731
    L = [
        PROVISIONAL_CUBE_HEADER,
        "DIAGNOSTIC: the high-|z| gas tail -- how well measured, and how much of P_tot it carries",
        "=" * 104,
        "Fiducial as Fig. 3: RAW, all footprint cells (HIM included), |x|,|y| <= 500 pc, self-gravity",
        "footprint_mean, Sun-centred midplane, the source cube's float64 z grid "
        f"({z[0]:g}..{z[-1]:g} pc).",
        f"Fig. 3 reference: P(0+) = {P0p_fig3:.1f}, P(0-) = {P0m_fig3:.1f} K cm^-3; slab (|z| <= "
        f"{SLAB_HALF_PC:g} pc) <P_tot>_area = {ptot_slab_2e:.1f}, <P_th>_neutral,vol = {pth_slab:.1f},",
        f"alpha_area_vol = {alpha_slab_2e:.4f} (area_alpha.txt).",
        "Measures only -- no physics, cache, existing result or figure is changed.",
        "",
        "-" * 104,
        "ASSERTIONS (the script raises if any fails)",
        "-" * 104,
    ]
    for name, meas, thr in checks:
        L.append(f"  {meas:>10.2e}  (<= {thr:.0e})  {name}")
    L += [
        "",
        "-" * 104,
        "(A) POSTERIOR SNR OF THE MAP, per 20 pc |z| bin, out to the cube edge",
        "-" * 104,
        "  posterior_snr.py's machinery: Edenhofer2023Query(integrated=False, flavor='main'), mean and",
        f"  std, orientation {label} (re-checked here: r = {r_map:.4f} vs identity {r_id:.4f}), same seed,",
        f"  {psnr.N_PER_STRATUM} cells per (|z| bin, flagged / non-flagged) stratum. 'all' combines the two",
        "  classes with stratum weights. det = mean - 2 std > 0.",
        "",
        f"{'|z| bin':>10} | {'ALL: med SNR  15-85      f<1 mw  det vol  det mw':<48}| "
        f"{'FLAGGED med':>11}{'f<1 mw':>8}{'det mw':>8} | {'NON-FL med':>10}{'f<1 mw':>8}{'det mw':>8}",
        "-" * 118,
    ]
    for b in range(n_bins):
        a, f, ne = A["all"], A["flagged"], A["neutral"]
        L.append(f"{edges[b]:>4.0f}-{edges[b + 1]:<4.0f}  | {a['med'][b]:>8.2f}  {a['p15'][b]:>4.2f}-"
                 f"{a['p85'][b]:<4.2f}  {a['f1_mw'][b]:>9.3f}{a['det_vol'][b]:>9.3f}{a['det_mw'][b]:>8.3f}  | "
                 f"{f['med'][b]:>11.2f}{f['f1_mw'][b]:>8.3f}{f['det_mw'][b]:>8.3f} | "
                 f"{ne['med'][b]:>10.2f}{ne['f1_mw'][b]:>8.3f}{ne['det_mw'][b]:>8.3f}")
    L += [
        "  (the per-class f<1 vol and det vol columns are in the cache; f<1 = fraction with SNR < 1)",
        "",
        "  MASS-weighted share of the column with low SNR (all cells, stratum x n_H weights):",
        f"{'':>4}{'zone':>10}{'median SNR mw':>15}{'median SNR vol':>16}{'SNR < 1':>10}{'SNR < 2':>10}",
    ]
    for zn in ("0-400", "400-750"):
        sh = shares[zn]
        L.append(f"{'':>4}{zn:>10}{sh['med_mw']:>15.2f}{sh['med_vol']:>16.2f}{sh['lt1']:>10.4f}{sh['lt2']:>10.4f}")
    L += [
        "",
        "  Reproduction of results/posterior_snr.txt (0-400 pc), compared at its printed precision",
        "  (the posterior_snr cache is not on disk): same seed and bin order, so bins 0-380 draw the",
        "  SAME cells; bin 380-400 does not (there it is the last bin and closed at 400, here open).",
    ]
    for name, dm, dd, dm_last, dd_last in cmp_rows:
        L.append(f"    {name:>8}: bins 0-380 max |delta| median SNR {dm:.3f}, det mw {dd:.3f};"
                 f"  bin 380-400: {dm_last:.3f}, {dd_last:.3f}")
    L += [
        "",
        "-" * 104,
        "(B) CUT-HEIGHT DEPENDENCE: rho = 0 beyond +-z_top (gas self-gravity from the truncated profile)",
        "-" * 104,
        f"{'z_top':>7}{'P(0+)':>10}{'P(0-)':>10}{'mean':>10}{'sg(0+)':>9}{'sg(0-)':>9}"
        f"{'slab <P_tot>':>14}{'slab alpha':>12}{'Sigma_gas':>11}",
        f"{'[pc]':>7}{'---------------- [K cm^-3] ----------------':>48}{'':>14}{'':>12}{'[Msun/pc2]':>11}",
    ]
    for zt in Z_TOPS_PC:
        r = cut[zt]
        L.append(f"{zt:>7.0f}{r['P0p']:>10.1f}{r['P0m']:>10.1f}{0.5 * (r['P0p'] + r['P0m']):>10.1f}"
                 f"{r['sg0p']:>9.1f}{r['sg0m']:>9.1f}{r['slab_P']:>14.1f}{r['alpha_slab']:>12.4f}"
                 f"{r['sigma_within']:>11.3f}")
    a400, a750 = cut[400.0]["alpha_slab"], cut[750.0]["alpha_slab"]
    L += [
        f"  slab alpha_area_vol: {a400:.4f} at z_top = 400 vs {a750:.4f} at 750 "
        f"({pct(a400, a750)} relative to the fiducial)",
        "",
        "-" * 104,
        "(C) WEIGHT BEYOND THE CUBE EDGE",
        "-" * 104,
        f"  C1 direct query: x,y step {C1_XY_STEP_PC:g} pc over the footprint, z step {C1_Z_STEP_PC:g} pc, "
        f"map distance range [{bounds.min():.1f}, {bounds.max():.1f}] pc,",
        f"  n_H = {N_H_PER_EXTINCTION_F98:.0f} A' (the cube's conversion).",
        f"{'':>4}{'side':>6}{'covered':>14}{'continuity (query/cube, 740-750 pc)':>38}",
    ]
    for side in ("pos", "neg"):
        L.append(f"{'':>4}{side:>6}{c1[side]['cov'].min():>7.3f}-{c1[side]['cov'].max():<6.3f}"
                 f"{c1[side]['continuity']:>38.3f}")
    L += ["", f"{'':>4}{'|z| bin':>12}{'<n_H> pos':>11}{'<n_H> neg':>11}{'med SNR pos':>13}{'med SNR neg':>13}"]
    for lo in np.arange(z[-1], C1_Z_MAX_PC, C1_SNR_BIN_PC):
        hi = lo + C1_SNR_BIN_PC
        row = f"{'':>4}{lo:>5.0f}-{hi:<6.0f}"
        for side in ("pos", "neg"):
            m = (c1[side]["z"] > lo) & (c1[side]["z"] <= hi)
            row += f"{np.mean(c1[side]['nH'][m]):>11.4f}"
        for side in ("pos", "neg"):
            m = (c1[side]["snr_z"] > lo) & (c1[side]["snr_z"] <= hi)
            row += f"{np.nanmedian(c1[side]['snr'][m]):>13.2f}"
        L.append(row)
    L += [
        "",
        f"  C2 exponential fit n0 exp(-|z|/h) to the cube's <n_H>_area over {C2_FIT_RANGE_PC[0]:g} <= |z| <= "
        f"{C2_FIT_RANGE_PC[1]:g} pc (log-linear least squares), extended to {C2_Z_MAX_PC:g} pc:",
    ]
    for side in ("pos", "neg"):
        L.append(f"    {side}: n0 = {c2[side]['n0']:.4f} cm^-3, h = {c2[side]['h']:.1f} pc, "
                 f"rms residual {c2[side]['rms_dex']:.3f} dex")
    L += [
        "",
        f"{'':>4}{'estimate':>10}{'extra col pos':>15}{'extra col neg':>15}{'P(0+)':>10}{'P(0-)':>10}"
        f"{'dP(0+)':>9}{'dP(0-)':>9}{'slab alpha':>12}{'d alpha':>9}",
        f"{'':>4}{'':>10}{'[Msun/pc2]':>15}{'[Msun/pc2]':>15}",
        f"{'':>4}{'Fig. 3':>10}{'0':>15}{'0':>15}{P0p_fig3:>10.1f}{P0m_fig3:>10.1f}{'':>9}{'':>9}"
        f"{a750:>12.4f}",
    ]
    for lab, r, ex in (("C1 (1000)", r1, c1_extra), ("C2 (1500)", r2, c2_extra)):
        L.append(f"{'':>4}{lab:>10}{ex['pos']:>15.3f}{ex['neg']:>15.3f}{r['P0p']:>10.1f}{r['P0m']:>10.1f}"
                 f"{pct(r['P0p'], P0p_fig3):>9}{pct(r['P0m'], P0m_fig3):>9}{r['alpha_slab']:>12.4f}"
                 f"{pct(r['alpha_slab'], a750):>9}")
    L += [
        "  (gas self-gravity from the extended profile; Guo stars/DM unchanged; P = 0 at the new edge;",
        "   slab alpha = slab mean of the new <P_tot>_area / the existing slab <P_th>_neutral,vol)",
        "",
        f"(runtime {time.time() - t0:.1f}s)",
    ]
    OUT_TXT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
