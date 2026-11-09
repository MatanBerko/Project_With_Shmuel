"""
compute_zmax -- the fit-range / integration-height figure (fig_zmax), all to
the map's extent (|z| <= 1000 pc). No extrapolation beyond the data.

Combines scripts/validation/layer_fit.py (single-layer fits, reduced chi2 vs
z_max, knee rule) and scripts/validation/high_z_tail.py (map query beyond the
cube, W / P_tot with gas self-gravity, slab alpha), whose functions are
IMPORTED, not re-implemented and not modified:

  layer_fit    plane_square_means, bin_side, log_errors, fit_one, knee,
               LOG_MODEL, MODELS, BOUNDS, BIN_PC, KNEE_* (its U_MAX_PC is
               set to 1000 pc on the imported module object, the file is
               untouched)
  high_z_tail  budget, extend_profile, column_msun, the C1 query settings
               (Edenhofer2023Query(integrated=False, flavor='main'), mode
               'mean', heliocentric Galactic points, n_H = 1653 A')
  posterior_snr  query_map, read_best_mapping, cube_to_true
  compute_area_alpha  patch_ids, g_for_setting
  src.physics.midplane / hydrostatic (the balanced-midplane P_tot)

Steps
-----
1. Profile beyond the cube: map query over the full footprint on the cube's
   own x,y grid (2 pc), z = +-(754, 758, ...) pc in 4 pc steps from the cube
   edge (|z| = 750) to u = |z - z_mid| = 1000 pc on each side (z = +998 on
   the upper side, z = -1010 on the lower one), plus every 10 pc z_top node
   above 750 that is not on the 4 pc grid (used only by step 4). <n_H>_area
   and the 16-square means per plane, exactly as layer_fit.
2. Cube/query mismatch: the same query on the cube's own 2 pc grid at the
   cube's planes 600 <= |z| <= 750, both orientations, the 10 pc x,y
   subsample, and the query/cube ratio vs heliocentric distance.
3. Chi2 vs fit range: layer_fit's fits on the combined profile (cube for
   |z| <= 750, query beyond), z_max = 100..1000 pc in 10 pc steps.
4. Alpha vs integration height: rho = 0 beyond |z| = z_top (Sun frame, as
   high_z_tail table B), z_top = 300..1000 pc in 10 pc steps, gas
   self-gravity from the truncated profile:
     alpha_area_vol (high_z_tail definition): slab |z| <= 30 pc mean of the
        Sun-centred P_tot (z_split = 0) / slab <P_th>_neutral,vol;
     <alpha_cell> (Fig. 4 headline): slab |z - z_mid| <= 30 pc mean of the
        balanced-midplane P_tot (z_mid re-found for each truncated profile)
        times the slab <1/P_th> over the non-HIM cells.
   A third column, alpha_area_vol on the balanced slab, bridges the two.

Writes cache/zmax/zmax.npz and results/zmax.txt. Changes nothing else.
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.conventions import N_H_PER_EXTINCTION_F98, PROVISIONAL_CUBE_HEADER  # noqa: E402
from src.conventions import MIDPLANE_MODE, MIDPLANE_MODE_BALANCED  # noqa: E402
from src.physics import gravity, hydrostatic  # noqa: E402
from src.physics.loading import open_zarr  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════
# ██  CONTROL BLOCK
# ═══════════════════════════════════════════════════════════════════════
VARIANT = "RAW"
ALPHA_CORE_ZARR_PATH = Path("cache/core/alpha_core.zarr")
AREA_ALPHA_NPZ_PATH = Path("cache/core/area_alpha.npz")
PTOT_BUDGET_NPZ_PATH = Path("cache/ptot_budget/ptot_budget.npz")
LAYER_FIT_NPZ_PATH = Path("cache/layer_fit/layer_fit.npz")
HIGH_Z_TAIL_NPZ_PATH = Path("cache/high_z_tail/high_z_tail.npz")
OUT_NPZ_PATH = Path("cache/zmax/zmax.npz")
OUT_TXT_PATH = Path("results/zmax.txt")

CUBE_EDGE_PC = 750.0
U_MAX_PC = 1000.0                    # profile and fits out to u = |z - z_mid| = 1000
Q_Z_STEP_PC = 4.0
Z_MAX_GRID_PC = np.arange(100.0, U_MAX_PC + 1e-9, 10.0)
Z_TOP_GRID_PC = np.arange(300.0, 1000.0 + 1e-9, 10.0)
Z_TOP_TABLE_PC = (300.0, 400.0, 500.0, 600.0, 750.0, 850.0, 1000.0)
TABLE_B_PC = (300.0, 400.0, 500.0, 600.0, 750.0)
SLAB_HALF_PC = 30.0
PTH_PLANES_ABS_Z_PC = 90.0           # p_th sums for planes |z| <= this (slab + z_mid shifts)
# mismatch probe
MM_ABS_Z_PC = (600.0, 750.0)
MM_DIST_PLANES_PC = np.arange(100.0, 750.0 + 1e-9, 50.0)
MM_DIST_EDGES_PC = np.arange(0.0, 1300.0 + 1e-9, 100.0)
MM_COARSE_STEP_PC = 10.0             # the C1 x,y step, as a subsample of the 2 pc grid
# high_z_tail printed values (table B and C1), for the printed-precision check
TABLE_B_PRINTED = {300.0: (3974.7, 5663.4, 4705.5, 1.4019), 400.0: (4727.2, 6587.2, 5540.9, 1.6508),
                   500.0: (5425.1, 7112.1, 6155.1, 1.8338), 600.0: (6156.8, 7686.1, 6810.5, 2.0291),
                   750.0: (7245.4, 8570.5, 7800.3, 2.3240)}
C1_PRINTED = (8587.7, 9665.9, 2.6888)
PTH_SUN_PRINTED = 3356.5
REPRO_RTOL = 1e-6

_ROOT = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lf = _load("layer_fit", "scripts/validation/layer_fit.py")
hzt = _load("high_z_tail", "scripts/validation/high_z_tail.py")
caa = lf.caa
psnr = hzt.psnr
lf.U_MAX_PC = U_MAX_PC               # bin_side's bin count; module object only


# ═══════════════════════════════════════════════════════════════════════
# ██  HELPERS
# ═══════════════════════════════════════════════════════════════════════
def query_planes(query, z_list, x_pc, y_pc, pid, n_sq, with_std=True):
    """Per plane: <n_H>_area, coverage, the 16 square means (C1 settings)."""
    X, Y = np.meshgrid(x_pc, y_pc)                     # [iy, ix], as the cube
    bounds = np.asarray(query.distance_bounds.to_value("pc"), dtype=float)
    nH, cov, sq, dmax = [], [], [], []
    for zz in z_list:
        pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, zz)], axis=1)
        d = np.linalg.norm(pts, axis=1)
        inside = (d >= bounds.min()) & (d <= bounds.max())
        m = np.full(len(pts), np.nan)
        s = np.full(len(pts), np.nan)
        m[inside] = psnr.query_map(query, pts[inside], "mean")
        if with_std:
            s[inside] = psnr.query_map(query, pts[inside], "std")
            ok = np.isfinite(m) & np.isfinite(s)
        else:
            ok = np.isfinite(m)
        n = N_H_PER_EXTINCTION_F98 * m
        cov.append(float(ok.mean()))
        nH.append(float(np.mean(n[ok])) if ok.any() else np.nan)
        w = np.where(ok, n, 0.0)
        sq.append(np.bincount(pid[ok], weights=w[ok], minlength=n_sq)
                  / np.bincount(pid[ok], minlength=n_sq))
        dmax.append(float(d.max()))
    return np.array(nH), np.array(cov), np.array(sq), np.array(dmax), bounds


def pth_plane_sums(core, z_src):
    """Per plane |z| <= PTH_PLANES_ABS_Z_PC: non-HIM count, sum p_th, and for
    the cells with finite p_th > 0 their count and sum 1/p_th."""
    idx = np.where(np.abs(z_src) <= PTH_PLANES_ABS_Z_PC)[0]
    vg = core[VARIANT]
    rows = []
    for i in idx:
        p = np.asarray(vg["p_th_phys"][i], dtype=np.float64).ravel()
        neu = ~np.asarray(vg["him"][i]).astype(bool).ravel()
        good = neu & np.isfinite(p) & (p > 0)
        rows.append((neu.sum(), p[neu].sum(), good.sum(), (1.0 / p[good]).sum()))
    r = np.array(rows, dtype=np.float64)
    return {"z": z_src[idx], "n_neutral": r[:, 0], "sum_pth": r[:, 1],
            "n_inv": r[:, 2], "sum_inv": r[:, 3]}


def slab_pth(ps, z_center):
    """(<P_th>_neutral,vol, <1/P_th>^-1) over the planes |z - z_center| <= 30."""
    m = np.abs(ps["z"] - z_center) <= SLAB_HALF_PC
    if abs(z_center) + SLAB_HALF_PC > PTH_PLANES_ABS_Z_PC:
        raise RuntimeError("slab outside the planes whose p_th sums were read")
    return (ps["sum_pth"][m].sum() / ps["n_neutral"][m].sum(),
            ps["n_inv"][m].sum() / ps["sum_inv"][m].sum())


def balanced_P(z, nH):
    """(z_mid, P(z)) for the balanced midplane with footprint-mean gas
    self-gravity -- the Fig. 3 / Fig. 4 route (compute_area_alpha.g_for_setting,
    then p_tot_kb_full_column split at z_mid)."""
    g, _, z_mid = caa.g_for_setting("mean", z, nH)
    P = hydrostatic.p_tot_kb_full_column(z, gravity.RHO_PER_N * nH, g, z_split=z_mid)
    return z_mid, P


def truncate(z, n, z_top, is_tail):
    """rho = 0 beyond |z| = z_top. The tail grids start with their own node at
    |z| = 750, so for z_top <= 750 the whole tail is zero (table B exactly)."""
    keep = (np.abs(z) <= z_top + 1e-9) & ~(is_tail & (z_top <= CUBE_EDGE_PC + 1e-9))
    return np.where(keep, n, 0.0)


# ═══════════════════════════════════════════════════════════════════════
# ██  MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    checks = []

    def check(name, measured, threshold):
        checks.append((name, measured, threshold))
        print(f"  check {name}: {measured:.3e} (<= {threshold:.0e})")
        if not (np.isfinite(measured) and measured <= threshold):
            raise RuntimeError(f"ASSERTION FAILED -- {name}: {measured:.3e} > {threshold:.0e}. STOPPING.")

    if MIDPLANE_MODE != MIDPLANE_MODE_BALANCED:
        raise RuntimeError("built for the balanced midplane (MIDPLANE_MODE)")

    # ---------------------------------------------------------------- cube
    z = np.asarray(open_zarr().z_pc, dtype=np.float64)                 # SOURCE float64 grid
    pb = np.load(PTOT_BUDGET_NPZ_PATH, allow_pickle=False)
    aa = np.load(AREA_ALPHA_NPZ_PATH, allow_pickle=False)
    lfc = np.load(LAYER_FIT_NPZ_PATH, allow_pickle=False)
    hzc = np.load(HIGH_Z_TAIL_NPZ_PATH, allow_pickle=False)
    if not np.array_equal(pb["z_pc"], z):
        raise RuntimeError("ptot_budget z grid differs from the source grid")
    nH = pb["nH_area"]
    check("cube <n_H>_area: ptot_budget vs layer_fit cache (max rel)",
          float(np.max(np.abs(nH / lfc["nH_area"] - 1))), 1e-15)
    check("cube <n_H>_area: ptot_budget vs high_z_tail cache (max rel)",
          float(np.max(np.abs(nH / hzc["nH_area"] - 1))), 1e-15)
    z_mid0 = float(lfc["z_mid"][0])
    core = zarr.open_group(str(ALPHA_CORE_ZARR_PATH), mode="r")
    x_pc = np.asarray(core["x_pc"][:], dtype=np.float64)
    y_pc = np.asarray(core["y_pc"][:], dtype=np.float64)
    n_arr = core[VARIANT]["n_model"]
    print("cube 16-square means ...")
    sq_cube, cnt = lf.plane_square_means(core, n_arr)
    check("cube 16-square err_log per plane vs layer_fit cache (max abs, dex)",
          float(np.nanmax(np.abs(
              np.nanstd(np.log10(sq_cube), axis=1, ddof=1) / 4.0 - lfc["err_log_plane"]))), 1e-12)
    pid2d, _ = caa.patch_ids(x_pc, y_pc, lf.N_SQUARES_PER_SIDE)
    pid = pid2d.ravel()
    n_sq = lf.N_SQUARES_PER_SIDE ** 2

    from dustmaps.edenhofer2023 import Edenhofer2023Query
    query = Edenhofer2023Query(integrated=False, flavor="main")

    # ============================================================ 1. query profile
    print("\n(1) map query beyond the cube edge")
    tq = time.time()
    q_up = np.arange(CUBE_EDGE_PC + Q_Z_STEP_PC, z_mid0 + U_MAX_PC + 1e-9, Q_Z_STEP_PC)
    q_lo = -np.arange(CUBE_EDGE_PC + Q_Z_STEP_PC, -(z_mid0 - U_MAX_PC) + 1e-9, Q_Z_STEP_PC)
    # z_top nodes (alpha integration only) not on the 4 pc grid, plus the edge itself
    zt_ext = Z_TOP_GRID_PC[Z_TOP_GRID_PC > CUBE_EDGE_PC]
    # integration tail (step 4): one node set for both sides (extend_profile needs a grid
    # symmetric about 0): |z| = 750, the 4 pc grid to 1000, and every 10 pc z_top node
    z_int = np.unique(np.round(np.concatenate([
        [CUBE_EDGE_PC], np.arange(CUBE_EDGE_PC + Q_Z_STEP_PC, 1000.0 + 1e-9, Q_Z_STEP_PC), zt_ext]), 9))
    n_extra_nodes = int(sum(not np.any(np.isclose(v, np.arange(CUBE_EDGE_PC, 1000.0 + 1e-9, Q_Z_STEP_PC)))
                            for v in zt_ext))
    Q = {}
    for side, zl in (("upper", q_up), ("lower", q_lo), ("upper_int", z_int), ("lower_int", -z_int)):
        nq, cov, sq, dmax, bounds = query_planes(query, zl, x_pc, y_pc, pid, n_sq)
        Q[side] = {"z": zl, "nH": nq, "cov": cov, "sq": sq, "dmax": dmax}
        print(f"  {side}: {zl.size} planes |z| {np.abs(zl).min():.0f}-{np.abs(zl).max():.0f}, "
              f"coverage {cov.min():.4f}-{cov.max():.4f}, max distance {dmax.max():.1f} pc")
    t_query = time.time() - tq
    cov_min = min(Q[s]["cov"].min() for s in Q)
    if cov_min < 1.0:
        raise RuntimeError(f"coverage drops below 1.0 (min {cov_min}) -- STOPPING (truncate the profile)")
    z_full_cov = float(np.sqrt(bounds.max() ** 2 - 2 * 500.0 ** 2))

    # combined profile for the fits: cube planes |z| <= 750, query beyond (4 pc planes)
    zc = np.concatenate([q_lo[::-1], z, q_up])
    nc = np.concatenate([Q["lower"]["nH"][::-1], nH, Q["upper"]["nH"]])
    sqc = np.concatenate([Q["lower"]["sq"][::-1], sq_cube, Q["upper"]["sq"]])
    src = np.concatenate([np.ones(q_lo.size), np.zeros(z.size), np.ones(q_up.size)])  # 1 = query
    if np.any(np.diff(zc) <= 0):
        raise RuntimeError("combined z grid is not ascending")

    # ============================================================ 2. mismatch
    print("\n(2) cube/query mismatch on the cube's own grid")
    label, M, _ = psnr.read_best_mapping()
    X, Y = np.meshgrid(x_pc, y_pc)
    coarse = ((np.abs(np.round(X / MM_COARSE_STEP_PC) * MM_COARSE_STEP_PC - X) < 1e-9)
              & (np.abs(np.round(Y / MM_COARSE_STEP_PC) * MM_COARSE_STEP_PC - Y) < 1e-9)).ravel()
    MM = {}
    for side, sgn in (("upper", 1.0), ("lower", -1.0)):
        idx = np.where((sgn * z >= MM_ABS_Z_PC[0] - 1e-9) & (sgn * z <= MM_ABS_Z_PC[1] + 1e-9))[0]
        rows = []
        for i in idx:
            cube = np.asarray(n_arr[i], dtype=np.float64).ravel()
            pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z[i])], axis=1)
            m_id = N_H_PER_EXTINCTION_F98 * psnr.query_map(query, pts, "mean")
            m_tr = N_H_PER_EXTINCTION_F98 * psnr.query_map(query, psnr.cube_to_true(M, pts), "mean")
            ok = np.isfinite(m_id) & np.isfinite(m_tr)
            r_tr = np.corrcoef(np.log10(cube[ok]), np.log10(m_tr[ok]))[0, 1]
            rows.append((z[i], cube.mean(), m_id[ok].mean(), m_tr[ok].mean(),
                         m_id[ok & coarse].mean(), cube[coarse].mean(), r_tr,
                         np.median(m_tr[ok] / cube[ok])))
        MM[side] = np.array(rows)
    # ratio vs heliocentric distance (orientation-mapped, per-bin sum ratio)
    nb = MM_DIST_EDGES_PC.size - 1
    MMd = {}
    for side, sgn in (("upper", 1.0), ("lower", -1.0)):
        num, den = np.zeros(nb), np.zeros(nb)
        for zz in MM_DIST_PLANES_PC:
            i = int(np.argmin(np.abs(z - sgn * zz)))
            cube = np.asarray(n_arr[i], dtype=np.float64).ravel()
            pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z[i])], axis=1)
            m = N_H_PER_EXTINCTION_F98 * psnr.query_map(query, psnr.cube_to_true(M, pts), "mean")
            k = np.digitize(np.linalg.norm(pts, axis=1), MM_DIST_EDGES_PC) - 1
            ok = np.isfinite(m)
            num += np.bincount(k[ok], m[ok], nb)
            den += np.bincount(k[ok], cube[ok], nb)
        with np.errstate(invalid="ignore", divide="ignore"):
            MMd[side] = np.where(den > 0, num / den, np.nan)

    # ============================================================ 3. fits
    print("\n(3) layer fits on the combined profile")
    out = {"z_pc": zc, "nH_area": nc, "source_is_query": src, "z_mid": np.array([z_mid0]),
           "z_max_grid": Z_MAX_GRID_PC, "models": np.array(lf.MODELS),
           "provisional": np.array([PROVISIONAL_CUBE_HEADER])}
    with np.errstate(divide="ignore", invalid="ignore"):
        lg_pl = np.where(sqc > 0, np.log10(np.where(sqc > 0, sqc, 1.0)), np.nan)
    out["err_log_plane"] = np.nanstd(lg_pl, axis=1, ddof=1) / np.sqrt(np.sum(sqc > 0, axis=1))
    res, knees, failures, dropped_all, nq_bins = {}, {}, [], {}, {}
    for side in lf.SIDES:
        u, n_bin, sq_bin, hi_edge, npl = lf.bin_side(zc, z_mid0, nc, sqc, side)
        sel = zc > z_mid0 if side == "upper" else zc < z_mid0
        k = np.floor(np.abs(zc[sel] - z_mid0) / lf.BIN_PC).astype(int)
        nbins = int(np.ceil(U_MAX_PC / lf.BIN_PC))
        cnt_b = np.bincount(k, minlength=nbins)[:nbins]
        frac_q = (np.bincount(k, weights=src[sel], minlength=nbins)[:nbins]
                  / np.maximum(cnt_b, 1))[cnt_b > 0]          # bin_side keeps the same bins
        if np.any((frac_q > 0) & (frac_q < 1)):
            raise RuntimeError(f"{side}: a 4 pc bin mixes cube and query planes")
        err, n_used, dropped = lf.log_errors(sq_bin)
        dropped_all[side] = dropped
        if not (np.all(n_bin > 0) and np.all(np.isfinite(err)) and np.all(err > 0)):
            raise RuntimeError(f"{side}: a bin has non-positive density or no usable error")
        nq_bins[side] = (int((frac_q == 1).sum()), float(u[frac_q == 1].min()), float(u[-1]))
        logn = np.log10(n_bin)
        out.update({f"{side}_u": u, f"{side}_nH": n_bin, f"{side}_err_log": err,
                    f"{side}_bin_hi": hi_edge, f"{side}_is_query": frac_q == 1,
                    f"{side}_n_planes": npl})
        for model in lf.MODELS:
            cols = {kk: np.full(Z_MAX_GRID_PC.size, np.nan)
                    for kk in ("n0", "h", "chi2", "rchi2", "N_bins")}
            for i, zm in enumerate(Z_MAX_GRID_PC):
                m = hi_edge <= zm + 1e-9
                N = int(m.sum())
                n0, h, chi2, status = lf.fit_one(model, u[m], logn[m], err[m], n_bin[0])
                cols["N_bins"][i] = N
                if status != "ok":
                    failures.append((side, model, zm, status))
                if np.isfinite(n0):
                    cols["n0"][i], cols["h"][i], cols["chi2"][i] = n0, h, chi2
                    cols["rchi2"][i] = chi2 / (N - 2)
            for kk, v in cols.items():
                out[f"{side}_{model}_{kk}"] = v
            res[(side, model)] = cols
            knees[(side, model)] = lf.knee(Z_MAX_GRID_PC, cols["rchi2"])
            out[f"{side}_{model}_knee"] = np.array(knees[(side, model)])
            # reproduction of layer_fit where every bin is a cube bin
            last_cube_hi = float(hi_edge[~(frac_q == 1)].max())
            same = Z_MAX_GRID_PC <= min(last_cube_hi, 750.0) + 1e-9
            ref = lfc[f"{side}_{model}_rchi2"][:same.sum()]
            check(f"{side} {model}: rchi2 vs layer_fit, z_max <= {Z_MAX_GRID_PC[same][-1]:.0f} (max rel)",
                  float(np.max(np.abs(cols["rchi2"][same] / ref - 1))), REPRO_RTOL)
            check(f"{side} {model}: knee vs layer_fit (abs pc)",
                  abs(knees[(side, model)][0] - float(lfc[f"{side}_{model}_knee"][0]))
                  if np.isfinite(knees[(side, model)][0]) else 0.0, 0.0)

    # ============================================================ 4. alpha vs z_top
    print("\n(4) alpha vs integration height")
    ps = pth_plane_sums(core, z)
    pth_sun_vol, pth_sun_harm = slab_pth(ps, 0.0)
    pth_mid_vol, pth_mid_harm = slab_pth(ps, z_mid0)
    check("slab <P_th>_neutral,vol at z_mid vs area_alpha (rel)",
          abs(pth_mid_vol / float(aa["slab__mean__Pth_neutral_vol"][0]) - 1), REPRO_RTOL)
    check("slab <1/P_th>^-1 at z_mid vs area_alpha (rel)",
          abs(pth_mid_harm / float(aa["slab__mean__Pth_harm_vol"][0]) - 1), REPRO_RTOL)
    check("Sun-centred slab <P_th>_neutral,vol vs high_z_tail printed 3356.5 (abs, printed precision)",
          abs(round(pth_sun_vol, 1) - PTH_SUN_PRINTED), 1e-9)

    # extended grid: cube + (4 pc query planes U z_top nodes), each tail starting at the edge
    zp, np_ = np.abs(Q["upper_int"]["z"]), Q["upper_int"]["nH"]     # |z| ascending from 750
    zn, nn_ = np.abs(Q["lower_int"]["z"]), Q["lower_int"]["nH"]
    zE, nE = hzt.extend_profile(z, nH, zp, np_, zn, nn_)
    on_cube = np.concatenate([np.zeros(zn.size, bool), np.ones(z.size, bool), np.zeros(zp.size, bool)])
    is_tail = ~on_cube
    slab_sun = np.where(np.abs(zE) <= SLAB_HALF_PC)[0]
    if not np.array_equal(zE[slab_sun], z[np.abs(z) <= SLAB_HALF_PC]):
        raise RuntimeError("Sun-centred slab planes differ from the cube's")
    A = {k: np.full(Z_TOP_GRID_PC.size, np.nan) for k in
         ("alpha_area_vol_sun", "alpha_cell_mean", "alpha_area_vol_mid", "P0p", "P0m",
          "slab_P_sun", "slab_P_mid", "z_mid", "sigma_gas", "n_slab_planes_mid")}
    P_store = {}
    for i, zt in enumerate(Z_TOP_GRID_PC):
        nt = truncate(zE, nE, zt, is_tail)
        r = hzt.budget(zE, nt, slab_sun)
        A["P0p"][i], A["P0m"][i], A["slab_P_sun"][i] = r["P0p"], r["P0m"], r["slab_P"]
        A["alpha_area_vol_sun"][i] = r["slab_P"] / pth_sun_vol
        A["sigma_gas"][i] = hzt.column_msun(zE, nt)
        zm_t, P_t = balanced_P(zE, nt)
        sl = np.abs(zE - zm_t) <= SLAB_HALF_PC
        pv, ph = slab_pth(ps, zm_t)
        A["z_mid"][i] = zm_t
        A["slab_P_mid"][i] = float(np.mean(P_t[sl]))
        A["alpha_cell_mean"][i] = A["slab_P_mid"][i] / ph
        A["alpha_area_vol_mid"][i] = A["slab_P_mid"][i] / pv
        A["n_slab_planes_mid"][i] = sl.sum()
        if zt in Z_TOP_TABLE_PC:
            P_store[zt] = r["P"]
    iT = {zt: int(np.argmin(np.abs(Z_TOP_GRID_PC - zt))) for zt in Z_TOP_TABLE_PC}

    # -- reproduction of high_z_tail table B (identical inputs: the cube profile)
    zs = hzc["z_pc"]
    for zt in TABLE_B_PC:
        Pc = hzc[f"cut{int(zt)}__P"]
        check(f"table B z_top = {zt:.0f}: P(z) on the cube planes vs high_z_tail cache (max rel)",
              float(np.max(np.abs(P_store[zt][on_cube] / np.where(Pc > 0, Pc, np.nan) - 1)[Pc > 0])),
              REPRO_RTOL)
        check(f"table B z_top = {zt:.0f}: slab <P_tot> vs high_z_tail cache (rel)",
              abs(A["slab_P_sun"][iT[zt]] / float(np.mean(Pc[np.abs(zs) <= SLAB_HALF_PC])) - 1), REPRO_RTOL)
        j = int(np.argmin(np.abs(Z_TOP_GRID_PC - zt)))
        pp, pm, sp, al = TABLE_B_PRINTED[zt]
        worst = max(abs(round(A["P0p"][j], 1) - pp), abs(round(A["P0m"][j], 1) - pm),
                    abs(round(A["slab_P_sun"][j], 1) - sp), abs(round(A["alpha_area_vol_sun"][j], 4) - al))
        check(f"table B z_top = {zt:.0f}: P(0+), P(0-), slab P, alpha vs printed (abs, printed precision)",
              worst, 1e-9)
    # -- C1 (its own 10 pc tail from the high_z_tail cache: identical inputs)
    c1t = {s: (hzc[f"c1_{s}_z"], hzc[f"c1_{s}_nH"]) for s in ("pos", "neg")}
    selp, seln = c1t["pos"][0] >= CUBE_EDGE_PC, c1t["neg"][0] >= CUBE_EDGE_PC
    z1, n1 = hzt.extend_profile(z, nH, c1t["pos"][0][selp], c1t["pos"][1][selp],
                                c1t["neg"][0][seln], c1t["neg"][1][seln])
    r1 = hzt.budget(z1, n1, np.where(np.abs(z1) <= SLAB_HALF_PC)[0])
    a1 = r1["slab_P"] / pth_sun_vol
    P0c1 = hzc["P0_c1"]
    check("C1 (cached 10 pc tail): P(0+) vs high_z_tail cache (rel)", abs(r1["P0p"] / P0c1[0] - 1), REPRO_RTOL)
    check("C1 (cached 10 pc tail): P(0-) vs high_z_tail cache (rel)", abs(r1["P0m"] / P0c1[1] - 1), REPRO_RTOL)
    check("C1 (cached 10 pc tail): P(z) vs high_z_tail cache c1_P (max rel)",
          float(np.nanmax(np.abs(r1["P"] / np.where(hzc["c1_P"] > 0, hzc["c1_P"], np.nan) - 1))), REPRO_RTOL)
    check("C1 (cached 10 pc tail): P(0+), P(0-), alpha vs printed (abs, printed precision)",
          max(abs(round(r1["P0p"], 1) - C1_PRINTED[0]), abs(round(r1["P0m"], 1) - C1_PRINTED[1]),
              abs(round(a1, 4) - C1_PRINTED[2])), 1e-9)
    # -- the fiducial (z_top = 750) Fig. 3 / Fig. 4 numbers on the balanced route
    j750 = iT[750.0]
    check("z_top = 750: z_mid vs ptot_budget (abs pc)", abs(A["z_mid"][j750] - float(pb["z_mid"][0])), 1e-6)
    check("z_top = 750: balanced slab <W> vs area_alpha (rel)",
          abs(A["slab_P_mid"][j750] / float(aa["slab__mean__Ptot_area"][0]) - 1), REPRO_RTOL)
    check("z_top = 750: <alpha_cell> vs area_alpha slab (rel)",
          abs(A["alpha_cell_mean"][j750] / float(aa["slab__mean__alpha_cell_mean_vol"][0]) - 1), REPRO_RTOL)
    check("z_top = 750: balanced alpha_area_vol vs area_alpha slab (rel)",
          abs(A["alpha_area_vol_mid"][j750] / float(aa["slab__mean__alpha_area_vol"][0]) - 1), REPRO_RTOL)
    # -- this grid at z_top = 1000 vs C1 (different query grid: reported, not asserted)
    j1000 = iT[1000.0]
    c1_vs = {"P0p": A["P0p"][j1000] / P0c1[0] - 1, "P0m": A["P0m"][j1000] / P0c1[1] - 1,
             "alpha": A["alpha_area_vol_sun"][j1000] / a1 - 1}

    for k, v in A.items():
        out[f"zt_{k}"] = v
    out["z_top_grid"] = Z_TOP_GRID_PC
    out["knee_range"] = np.array([np.nanmin([knees[k][0] for k in knees]),
                                  np.nanmax([knees[k][0] for k in knees])])
    for side in ("upper", "lower"):
        out[f"q_{side}_z"], out[f"q_{side}_nH"] = Q[side]["z"], Q[side]["nH"]
        out[f"q_{side}_cov"] = Q[side]["cov"]
        out[f"mm_{side}"] = MM[side]
        out[f"mm_dist_{side}"] = MMd[side]
    out["mm_dist_edges"] = MM_DIST_EDGES_PC
    out["pth_slab"] = np.array([pth_sun_vol, pth_sun_harm, pth_mid_vol, pth_mid_harm])
    OUT_NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ_PATH, **out)
    print(f"Saved {OUT_NPZ_PATH}")

    # ================================================================= report
    W = 104
    L = [PROVISIONAL_CUBE_HEADER,
         "z_max: density profile, chi2 vs fit range and alpha vs integration height to |z| = 1000 pc",
         "=" * W,
         "Measures only -- no physics, cache, existing result or figure is changed. No extrapolation:",
         "every number uses the cube (|z| <= 750 pc) and the Edenhofer+23 map query beyond it.",
         "",
         "-" * W, "ASSERTIONS (the script raises if any fails)", "-" * W]
    L += [f"  {m:>10.2e}  (<= {t:.0e})  {n}" for n, m, t in checks]

    L += ["", "-" * W, "(1) PROFILE BEYOND THE CUBE: map query", "-" * W,
          "  C1 query settings (high_z_tail): Edenhofer2023Query(integrated=False, flavor='main'), mode",
          f"  'mean', heliocentric Galactic (x, y, z), n_H = {N_H_PER_EXTINCTION_F98:.0f} A'; a point counts as covered when",
          f"  mean and std are finite. Map distance range [{bounds.min():.1f}, {bounds.max():.1f}] pc.",
          f"  x,y: the cube's own 2 pc grid over |x|,|y| <= 500 pc ({x_pc.size} x {y_pc.size} points per plane;",
          "  the 2 pc step is the finest available and ran in "
          f"{t_query:.0f} s for all planes, so no coarser step was needed).",
          f"  z: 4 pc steps from the cube edge, upper z = {q_up[0]:g}..{q_up[-1]:g} pc ({q_up.size} planes), lower",
          f"  z = {q_lo[0]:g}..{q_lo[-1]:g} pc ({q_lo.size} planes), i.e. to u = |z - z_mid| = 1000 pc on each side",
          f"  (z_mid = {z_mid0:.4f} pc); for the integration in (4) a common node set on both sides:",
          f"  |z| = 750, the 4 pc grid 754..998 and {n_extra_nodes} z_top nodes (10 pc steps) not on it, to |z| = 1000.",
          "  The cube is used for |z| <= 750 (Sun frame), the query beyond; so on the upper side the cube",
          f"  reaches u = {CUBE_EDGE_PC - z_mid0:.1f} pc and on the lower side u = {CUBE_EDGE_PC + z_mid0:.1f} pc. No 4 pc bin mixes",
          "  cube and query planes.",
          f"  COVERAGE: 1.0000 on every queried plane (min {cov_min:.4f}); the farthest point (a footprint corner",
          f"  at |z| = {max(np.abs(q_lo).max(), q_up.max()):g}) is at {max(Q[s]['dmax'].max() for s in Q):.1f} pc. The footprint is fully covered",
          f"  up to |z| = sqrt({bounds.max():.1f}^2 - 2 x 500^2) = {z_full_cov:.1f} pc.",
          "  <n_H>_area and the 16-square error exactly as layer_fit (same squares: compute_area_alpha.",
          "  patch_ids on the same x,y grid). Under the cube's orientation mapping (-y,-x,+z) the footprint",
          "  maps onto itself and the 16 squares onto the 16 squares, so neither <n_H>_area nor the error",
          "  depends on the mapping; the query is made at the cube's (x, y) as C1 did.",
          "",
          f"  {'u bin':>10}{'upper <n_H>':>13}{'err dex':>9}{'lower <n_H>':>13}{'err dex':>9}   (query bins, 50 pc means)"]
    for lo in np.arange(750.0, 1000.0, 50.0):
        row = f"  {lo:>5.0f}-{lo + 50:<4.0f}"
        for side in ("upper", "lower"):
            u, nb_, eb, isq = out[f"{side}_u"], out[f"{side}_nH"], out[f"{side}_err_log"], out[f"{side}_is_query"]
            m = (u >= lo) & (u < lo + 50) & isq
            row += f"{np.mean(nb_[m]):>13.4f}{np.mean(eb[m]):>9.4f}" if m.any() else f"{'--':>13}{'--':>9}"
        L.append(row)

    mmu, mml = MM["upper"], MM["lower"]

    def mmrow(a, lo, hi):
        m = (np.abs(a[:, 0]) >= lo) & (np.abs(a[:, 0]) <= hi)
        return (a[m, 2].sum() / a[m, 1].sum(), a[m, 3].sum() / a[m, 1].sum(),
                a[m, 4].sum() / a[m, 5].sum(), np.mean(a[m, 6]), np.median(a[m, 7]))
    L += ["", "-" * W, "(2) CUBE / QUERY MISMATCH (report only; nothing is rescaled)", "-" * W,
          "  The query with the C1 settings at the cube's own (x, y, z) nodes: every 2 pc cell of the",
          "  footprint on the cube's planes, compared with the cube's <n_H>_area on the same planes.",
          "  ratio = sum(query) / sum(cube) over the planes of the range.",
          f"  {'side':>6}{'|z| range':>12}{'identity':>10}{'mapped':>9}{'10 pc':>8}{'r(log)':>8}{'cell med':>10}",
          f"  {'':>6}{'':>12}{'2 pc':>10}{label:>9}{'subsamp':>8}{'mapped':>8}{'ratio':>10}"]
    for side, a in (("upper", mmu), ("lower", mml)):
        for lo, hi in ((600.0, 750.0), (700.0, 750.0), (740.0, 750.0)):
            ri, rt, rc, rr, rm = mmrow(a, lo, hi)
            L.append(f"  {side:>6}{lo:>7.0f}-{hi:<4.0f}{ri:>10.3f}{rt:>9.3f}{rc:>8.3f}{rr:>8.3f}{rm:>10.3f}")
    L += ["  (identity = query at the cube's (x, y) as C1; mapped = at the true position under the",
          "   orientation mapping; 10 pc subsamp = query and cube both on the 10 pc sub-grid; r(log) =",
          "   Pearson r of log10 n per cell, mapped; cell med = median per-cell query/cube, mapped)",
          "", "  query/cube vs heliocentric distance d (mapped orientation; planes |z| = 100..750 pc every",
          "  50 pc, all footprint cells; per-bin sum ratio):",
          "  " + f"{'d [pc]':>8}" + "".join(f"{e:>7.0f}" for e in MM_DIST_EDGES_PC[:-1])]
    for side in ("upper", "lower"):
        L.append("  " + f"{side:>8}" + "".join(f"{v:>7.3f}" if np.isfinite(v) else f"{'--':>7}"
                                               for v in MMd[side]))
    L += ["  FINDING: the ratio is the same with and without the orientation mapping (the footprint maps",
          "  onto itself) and at 2 pc and 10 pc x,y steps, so neither the x,y sampling step nor the",
          "  orientation causes it. It depends on heliocentric distance, alike on both sides: 0.92-1.05 for",
          "  d = 200-700 pc, falling to 0.67-0.87 beyond d = 700-800 pc. A distance-dependent difference",
          "  between two renderings of the same posterior points to the reconstruction itself: the cube",
          "  was built from 'localob_edensphere2kpc_edendist.dat' (a 2 kpc sphere), the query uses the",
          "  'main' (1.25 kpc) flavour, whose outer shells the high-|z| footprint corners approach",
          "  (d = 1023 pc at |z| = 740). The query's own interpolation (HEALPix bilinear on the sky, linear",
          "  in distance between shells) applies at every d and cannot by itself produce the trend. The",
          "  'less_data_but_2kpc' flavour is not on disk, so the flavour cause is likely, not demonstrated.",
          ]

    L += ["", "-" * W, "(3) CHI2 VS FIT RANGE: layer_fit's fits on the combined profile", "-" * W,
          "  As layer_fit: 4 pc bins in u = |z - z_mid|, upper / lower separately; exp, Gaussian and sech2",
          "  centred on z_mid, free n0 and h, fitted in log10 space (curve_fit trf, bounds n0 in",
          f"  [{lf.BOUNDS[0][0]:g}, {lf.BOUNDS[1][0]:g}] cm^-3, h in [{lf.BOUNDS[0][1]:g}, {lf.BOUNDS[1][1]:g}] pc); error = 16-square std(log10)/sqrt(16);",
          "  reduced chi2 = chi2/(N_bins - 2) over the bins with upper edge <= z_max; z_max = 100..1000 pc.",
          f"  Knee: the first z_max at or beyond the minimum of reduced chi2 over z_max <= {lf.KNEE_SEARCH_MAX_PC:g} pc",
          f"  where reduced chi2 exceeds {lf.KNEE_FACTOR:g}x that minimum.",
          f"  Query bins: upper {nq_bins['upper'][0]} (u = {nq_bins['upper'][1]:.1f}..{nq_bins['upper'][2]:.1f}), "
          f"lower {nq_bins['lower'][0]} (u = {nq_bins['lower'][1]:.1f}..{nq_bins['lower'][2]:.1f}).",
          "  Fits failed or at a bound:"]
    if failures:
        L += [f"    {s:5s} {m:5s} z_max = {zm:5.0f}: {st}" for s, m, zm, st in failures]
    else:
        L.append("    none -- every fit converged inside the bounds.")
    for side in lf.SIDES:
        L.append(f"  {side}: squares dropped in {len(dropped_all[side])} (bin, square) pairs")
    L += ["", f"  {'side':5s} {'model':5s} {'min rchi2':>9s} {'at':>5s} {'knee':>5s} {'h@knee':>8s} "
          f"{'rchi2@750':>9s} {'rchi2@1000':>10s} {'h@1000':>8s}"]
    for side in lf.SIDES:
        for model in lf.MODELS:
            zk, rmin, zmin = knees[(side, model)]
            c = res[(side, model)]
            i750 = int(np.argmin(np.abs(Z_MAX_GRID_PC - 750)))
            ik = int(np.argmin(np.abs(Z_MAX_GRID_PC - zk))) if np.isfinite(zk) else None
            L.append(f"  {side:5s} {model:5s} {rmin:9.3f} {zmin:5.0f} "
                     + (f"{zk:5.0f} {c['h'][ik]:8.1f}" if ik is not None else f"{'none':>5} {'':>8}")
                     + f" {c['rchi2'][i750]:9.2f} {c['rchi2'][-1]:10.2f} {c['h'][-1]:8.1f}")
    L += ["", f"  reduced chi2 vs z_max (every 50 pc):",
          "  " + f"{'z_max':>6}" + "".join(f"{s[:2]}-{m:>5s}" .rjust(10) for s in lf.SIDES for m in lf.MODELS)]
    for i, zm in enumerate(Z_MAX_GRID_PC):
        if zm % 50 == 0:
            L.append("  " + f"{zm:>6.0f}" + "".join(f"{res[(s, m)]['rchi2'][i]:>10.2f}"
                                                   for s in lf.SIDES for m in lf.MODELS))

    L += ["", "-" * W, "(4) ALPHA VS INTEGRATION HEIGHT: rho = 0 beyond |z| = z_top", "-" * W,
          "  rho = 0 where |z| > z_top (Sun frame, as high_z_tail table B); cube for |z| <= 750, the (1)",
          "  query beyond (4 pc planes plus a node at every 10 pc z_top, so each z_top is a grid node as in",
          "  table B); each tail starts at the cube edge (high_z_tail.extend_profile). Gas self-gravity",
          "  from the truncated profile; Guo+20 stars and DM unchanged; P = 0 at the grid edges.",
          f"  alpha_area_vol (high_z_tail definition): mean over |z| <= 30 pc of the Sun-centred P_tot",
          f"     (high_z_tail.budget, z_split = 0) / slab <P_th>_neutral,vol = {pth_sun_vol:.2f} K cm^-3.",
          f"  <alpha_cell> (Fig. 4 headline): mean over |z - z_mid| <= 30 pc of the balanced-midplane P_tot",
          "     (compute_area_alpha.g_for_setting + p_tot_kb_full_column split at z_mid, z_mid re-found for",
          f"     each truncated profile) / the slab <1/P_th>^-1 over its non-HIM cells ({pth_mid_harm:.2f} K cm^-3 at",
          "     the fiducial z_mid).",
          "  alpha_area_vol (balanced): the same balanced slab <P_tot> / its <P_th>_neutral,vol (bridge).",
          "",
          f"  {'z_top':>6}{'z_mid':>9}{'P(0+)':>9}{'P(0-)':>9}{'slab P':>9}{'a_area,vol':>11}"
          f"{'slab <W>':>10}{'<a_cell>':>10}{'a_area,vol':>11}{'Sigma_gas':>10}",
          f"  {'[pc]':>6}{'[pc]':>9}{'(Sun-centred, K cm^-3)':>27}{'(h_z_t)':>11}{'(z_mid)':>10}"
          f"{'headline':>10}{'(z_mid)':>11}{'[Msun/pc2]':>10}"]
    for zt in Z_TOP_TABLE_PC:
        j = iT[zt]
        L.append(f"  {zt:>6.0f}{A['z_mid'][j]:>9.3f}{A['P0p'][j]:>9.1f}{A['P0m'][j]:>9.1f}"
                 f"{A['slab_P_sun'][j]:>9.1f}{A['alpha_area_vol_sun'][j]:>11.4f}{A['slab_P_mid'][j]:>10.1f}"
                 f"{A['alpha_cell_mean'][j]:>10.4f}{A['alpha_area_vol_mid'][j]:>11.4f}{A['sigma_gas'][j]:>10.3f}")
    a = A["alpha_cell_mean"]
    L += [f"  <alpha_cell>: {a[iT[400.0]]:.3f} at 400, {a[iT[750.0]]:.3f} at 750 (current fiducial), "
          f"{a[iT[1000.0]]:.3f} at 1000 pc",
          f"  ({a[iT[400.0]] / a[iT[750.0]] - 1:+.1%} and {a[iT[1000.0]] / a[iT[750.0]] - 1:+.1%} relative to 750).",
          f"  z_mid ranges {np.nanmin(A['z_mid']):.3f}..{np.nanmax(A['z_mid']):.3f} pc over z_top; the balanced slab holds "
          f"{int(np.nanmin(A['n_slab_planes_mid']))}..{int(np.nanmax(A['n_slab_planes_mid']))} planes.",
          "",
          "  z_top = 1000 on this grid vs C1 (high_z_tail, 10 pc x,y and z steps; different query grid):",
          f"    P(0+) {c1_vs['P0p']:+.2e}, P(0-) {c1_vs['P0m']:+.2e}, alpha_area_vol {c1_vs['alpha']:+.2e} (relative)",
          "",
          "  full grid (10 pc):",
          f"  {'z_top':>6}{'a_area,vol':>11}{'<a_cell>':>10}{'a_area,vol':>11}",
          f"  {'':>6}{'(h_z_t)':>11}{'headline':>10}{'(z_mid)':>11}"]
    for j, zt in enumerate(Z_TOP_GRID_PC):
        L.append(f"  {zt:>6.0f}{A['alpha_area_vol_sun'][j]:>11.4f}{A['alpha_cell_mean'][j]:>10.4f}"
                 f"{A['alpha_area_vol_mid'][j]:>11.4f}")
    L += ["", f"runtime {time.time() - t0:.0f} s"]
    OUT_TXT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:200]))


if __name__ == "__main__":
    main()
