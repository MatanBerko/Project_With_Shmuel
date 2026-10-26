"""
DIAGNOSTIC: where does the footprint-mean density stop looking like a
decaying gas layer?

<n_H>_area(z) falls from ~1 cm^-3 at the midplane to a plateau of
~0.02-0.03 cm^-3 at |z| > ~400 pc. This script fits three standard layer
shapes to the profile out to a moving height z_max and records how the fit
quality and the fitted scale height change with z_max. The height where the
fit quality breaks down (the "knee") is where the layer picture stops
describing the data.

Data
----
  <n_H>(z)   footprint mean over ALL cells (HIM-flagged included) of the
             cached RAW n_model, through compute_area_alpha's
             footprint_mean_density_chunked -- the same route as
             compute_ptot_budget.py / compute_area_alpha.py -- on the SOURCE
             cube's float64 z grid.
  u          |z - z_mid|, z_mid the balanced midplane (self-gravity on) from
             src.physics.midplane, as in Fig. 3.
  bins       4 pc in u, [4k, 4k + 4), separately for the UPPER side
             (z > z_mid) and the LOWER side (z < z_mid). A bin's density is
             the mean over its planes (every plane has the same cell count,
             so this is the mean over the bin's cells); its position is the
             mean u of its planes.

Errors
------
The footprint is split into the 16 disjoint 250 x 250 pc squares
(compute_area_alpha.patch_ids, N = 4). Each square's <n_H> is formed per bin
the same way; the error on log10 <n_H> in a bin is the sample standard
deviation (ddof = 1) of log10 of the square profiles divided by
sqrt(n_squares). A square with <n_H> <= 0 (or non-finite) in a bin is dropped
for that bin, n_squares is then the number kept, and the drop is reported.

Models (centred on z_mid, free n0 and h, fitted in log10 space)
---------------------------------------------------------------
  exp    n0 exp(-u/h)
  gauss  n0 exp(-u^2 / (2 h^2))
  sech2  n0 sech(u/h)^2
chi2 = sum((log10 data - log10 model)^2 / err^2) over the bins with upper
edge <= z_max; reduced chi2 = chi2 / (N_bins - 2).

Knee rule: the first z_max at or beyond the minimum of reduced chi2 over
z_max <= 400 pc where reduced chi2 exceeds 2x that minimum.

Measures only: no physics, cache, existing result or existing figure
changes. Writes cache/layer_fit/layer_fit.npz and results/layer_fit.txt.
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import zarr
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.conventions import PROVISIONAL_CUBE_HEADER  # noqa: E402
from src.conventions import MIDPLANE_MODE, MIDPLANE_MODE_BALANCED  # noqa: E402
from src.physics import midplane  # noqa: E402
from src.physics.loading import open_zarr  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════
# ██  CONTROL BLOCK
# ═══════════════════════════════════════════════════════════════════════
VARIANT = "RAW"
ALPHA_CORE_ZARR_PATH = Path("cache/core/alpha_core.zarr")
AREA_ALPHA_NPZ_PATH = Path("cache/core/area_alpha.npz")
PTOT_BUDGET_NPZ_PATH = Path("cache/ptot_budget/ptot_budget.npz")
OUT_NPZ_PATH = Path("cache/layer_fit/layer_fit.npz")
OUT_TXT_PATH = Path("results/layer_fit.txt")

BIN_PC = 4.0
U_MAX_PC = 764.0                      # bins out to the cube edge on either side
N_SQUARES_PER_SIDE = 4                # 4 x 4 = 16 squares of 250 x 250 pc
Z_MAX_GRID_PC = np.arange(100.0, 750.0 + 1e-9, 10.0)
PRED_U_PC = 750.0                     # report each fit's density here
KNEE_SEARCH_MAX_PC = 400.0
KNEE_FACTOR = 2.0
MODELS = ("exp", "gauss", "sech2")
H0_PC = {"exp": 200.0, "gauss": 150.0, "sech2": 150.0}
BOUNDS = ([1e-6, 1.0], [1e3, 1e5])    # (n0 [cm^-3], h [pc]), positive
SIDES = ("upper", "lower")
REPRO_RTOL = 1e-15                    # <n_H> vs area_alpha.npz
ZMID_TOL_PC = 1e-6                    # z_mid vs ptot_budget.npz

_SPEC = importlib.util.spec_from_file_location(
    "compute_area_alpha",
    Path(__file__).resolve().parents[1] / "compute" / "compute_area_alpha.py")
caa = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(caa)


# ═══════════════════════════════════════════════════════════════════════
# ██  MODELS (log10 of the density, written to stay finite at large u/h)
# ═══════════════════════════════════════════════════════════════════════
LOG10E = np.log10(np.e)


def log_exp(u, n0, h):
    return np.log10(n0) - (u / h) * LOG10E


def log_gauss(u, n0, h):
    return np.log10(n0) - (u ** 2 / (2.0 * h ** 2)) * LOG10E


def log_sech2(u, n0, h):
    # sech x = 2 e^-x / (1 + e^-2x)
    x = u / h
    return np.log10(n0) + 2.0 * (np.log10(2.0) - x * LOG10E - np.log10(1.0 + np.exp(-2.0 * x)))


LOG_MODEL = {"exp": log_exp, "gauss": log_gauss, "sech2": log_sech2}


# ═══════════════════════════════════════════════════════════════════════
# ██  DATA
# ═══════════════════════════════════════════════════════════════════════
def plane_square_means(core, n_arr, z_chunk=caa.Z_CHUNK):
    """(Nz, 16) per-square mean n_H of every plane, streamed in z chunks.

    x, y are the cache's own footprint coordinates, as compute_area_alpha uses."""
    x_pc = np.asarray(core["x_pc"][:], dtype=np.float64)
    y_pc = np.asarray(core["y_pc"][:], dtype=np.float64)
    pid, _ = caa.patch_ids(x_pc, y_pc, N_SQUARES_PER_SIDE)
    pid = pid.ravel()
    n_sq = N_SQUARES_PER_SIDE ** 2
    cnt = np.bincount(pid, minlength=n_sq).astype(np.float64)
    nz = n_arr.shape[0]
    out = np.empty((nz, n_sq))
    for lo in range(0, nz, z_chunk):
        hi = min(lo + z_chunk, nz)
        chunk = np.asarray(n_arr[lo:hi], dtype=np.float64).reshape(hi - lo, -1)
        for k in range(hi - lo):
            out[lo + k] = np.bincount(pid, weights=chunk[k], minlength=n_sq) / cnt
        del chunk
    return out, cnt


def bin_side(z, z_mid, nH, sq, side):
    """Bin one side into 4 pc u-bins: (u_mean, nH_bin, sq_bin, edges_hi, n_planes)."""
    sel = z > z_mid if side == "upper" else z < z_mid
    u = np.abs(z[sel] - z_mid)
    k = np.floor(u / BIN_PC).astype(np.int64)
    nb = int(np.ceil(U_MAX_PC / BIN_PC))
    npl = np.bincount(k, minlength=nb)[:nb].astype(np.float64)
    keep = npl > 0
    u_mean = np.bincount(k, weights=u, minlength=nb)[:nb]
    n_bin = np.bincount(k, weights=nH[sel], minlength=nb)[:nb]
    sq_bin = np.stack([np.bincount(k, weights=sq[sel, j], minlength=nb)[:nb]
                       for j in range(sq.shape[1])], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        u_mean, n_bin, sq_bin = u_mean / npl, n_bin / npl, sq_bin / npl[:, None]
    hi_edge = (np.arange(nb) + 1) * BIN_PC
    return u_mean[keep], n_bin[keep], sq_bin[keep], hi_edge[keep], npl[keep]


def log_errors(sq_bin):
    """(err_log10, n_used, dropped list of (bin index, square index))."""
    good = np.isfinite(sq_bin) & (sq_bin > 0)
    dropped = [(int(i), int(j)) for i, j in zip(*np.nonzero(~good))]
    with np.errstate(invalid="ignore", divide="ignore"):
        lg = np.where(good, np.log10(np.where(good, sq_bin, 1.0)), np.nan)
    n_used = good.sum(axis=1)
    err = np.nanstd(lg, axis=1, ddof=1) / np.sqrt(n_used)
    return err, n_used, dropped


# ═══════════════════════════════════════════════════════════════════════
# ██  FITS
# ═══════════════════════════════════════════════════════════════════════
def fit_one(model, u, logn, err, n0_start):
    """(n0, h, chi2, status). status 'ok', or the failure reason."""
    f = LOG_MODEL[model]
    p0 = [min(max(n0_start, BOUNDS[0][0] * 10), BOUNDS[1][0] / 10), H0_PC[model]]
    try:
        p, _ = curve_fit(f, u, logn, p0=p0, sigma=err, absolute_sigma=True,
                         bounds=BOUNDS, method="trf", max_nfev=20000)
    except (RuntimeError, ValueError) as e:
        return np.nan, np.nan, np.nan, f"FAILED: {e}"
    chi2 = float(np.sum(((logn - f(u, *p)) / err) ** 2))
    status = "ok"
    for name, val, lo, hi in (("n0", p[0], BOUNDS[0][0], BOUNDS[1][0]),
                              ("h", p[1], BOUNDS[0][1], BOUNDS[1][1])):
        if np.isclose(val, lo, rtol=1e-6) or np.isclose(val, hi, rtol=1e-6):
            status = f"AT BOUND ({name} = {val:.4g})"
    return float(p[0]), float(p[1]), chi2, status


def knee(z_max, rchi2):
    """(z_knee, rchi2_min, z_at_min) by the stated rule; z_knee NaN if never."""
    m = (z_max <= KNEE_SEARCH_MAX_PC) & np.isfinite(rchi2)
    i_min = np.flatnonzero(m)[np.argmin(rchi2[m])]
    r_min = float(rchi2[i_min])
    over = np.flatnonzero((np.arange(z_max.size) >= i_min) & (rchi2 > KNEE_FACTOR * r_min))
    return (float(z_max[over[0]]) if over.size else np.nan), r_min, float(z_max[i_min])


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
            raise RuntimeError(f"ASSERTION FAILED -- {name}: {measured:.3e} > {threshold:.0e}.")

    # ---- profile, squares, midplane ---------------------------------------
    z = np.asarray(open_zarr().z_pc, dtype=np.float64)          # SOURCE float64 grid
    if np.any(np.diff(z) <= 0):
        raise RuntimeError("source z grid is not ascending")
    core = zarr.open_group(str(ALPHA_CORE_ZARR_PATH), mode="r")
    n_arr = core[VARIANT]["n_model"]
    if n_arr.shape[0] != z.size:
        raise RuntimeError("cached n_model z length != source z grid")
    nH = caa.footprint_mean_density_chunked(n_arr)
    aa = np.load(AREA_ALPHA_NPZ_PATH, allow_pickle=False)
    check("<n_H>_area vs area_alpha.npz n_bar_cm3_full (max rel)",
          float(np.max(np.abs(nH / aa["n_bar_cm3_full"] - 1))), REPRO_RTOL)
    sq, cnt = plane_square_means(core, n_arr)
    # equal-weight mean of the squares, weighted by their cell counts = footprint mean
    check("cell-weighted mean of the 16 squares vs <n_H>_area (max rel)",
          float(np.max(np.abs(sq @ cnt / cnt.sum() / nH - 1))), 1e-12)

    if MIDPLANE_MODE != MIDPLANE_MODE_BALANCED:
        raise RuntimeError("built for the balanced midplane (MIDPLANE_MODE)")
    z_mid = midplane.find_balanced_midplane(z, nH, self_gravity=True)
    pb = np.load(PTOT_BUDGET_NPZ_PATH, allow_pickle=False)
    check("z_mid vs ptot_budget.npz z_mid [pc]", abs(z_mid - float(pb["z_mid"][0])), ZMID_TOL_PC)
    print(f"z_mid = {z_mid:.4f} pc; squares hold {int(cnt.min())}..{int(cnt.max())} cells per plane")

    # ---- bins, errors, fits ------------------------------------------------
    out = {"z_pc": z, "nH_area": nH, "z_mid": np.array([z_mid]),
           "z_max_grid": Z_MAX_GRID_PC, "models": np.array(MODELS),
           "provisional": np.array([PROVISIONAL_CUBE_HEADER])}
    # signed-z error band for the figure: per plane, the same 16-square error
    with np.errstate(divide="ignore", invalid="ignore"):
        lg_pl = np.where(sq > 0, np.log10(np.where(sq > 0, sq, 1.0)), np.nan)
    out["err_log_plane"] = np.nanstd(lg_pl, axis=1, ddof=1) / np.sqrt(np.sum(sq > 0, axis=1))

    res, dropped_all, failures, knees = {}, {}, [], {}
    for side in SIDES:
        u, n_bin, sq_bin, hi_edge, npl = bin_side(z, z_mid, nH, sq, side)
        err, n_used, dropped = log_errors(sq_bin)
        dropped_all[side] = [(float(hi_edge[i] - BIN_PC), j) for i, j in dropped]
        if not (np.all(n_bin > 0) and np.all(np.isfinite(err)) and np.all(err > 0)):
            raise RuntimeError(f"{side}: a bin has non-positive density or no usable error")
        logn = np.log10(n_bin)
        out.update({f"{side}_u": u, f"{side}_nH": n_bin, f"{side}_err_log": err,
                    f"{side}_n_squares": n_used, f"{side}_bin_hi": hi_edge,
                    f"{side}_n_planes": npl, f"{side}_sq_nH": sq_bin})
        for model in MODELS:
            cols = {k: np.full(Z_MAX_GRID_PC.size, np.nan)
                    for k in ("n0", "h", "chi2", "rchi2", "n_fit750", "N_bins")}
            for i, zm in enumerate(Z_MAX_GRID_PC):
                m = hi_edge <= zm + 1e-9
                N = int(m.sum())
                n0, h, chi2, status = fit_one(model, u[m], logn[m], err[m], n_bin[0])
                cols["N_bins"][i] = N
                if status != "ok":
                    failures.append((side, model, zm, status))
                if np.isfinite(n0):
                    cols["n0"][i], cols["h"][i], cols["chi2"][i] = n0, h, chi2
                    cols["rchi2"][i] = chi2 / (N - 2)
                    cols["n_fit750"][i] = 10 ** LOG_MODEL[model](PRED_U_PC, n0, h)
            for k, v in cols.items():
                out[f"{side}_{model}_{k}"] = v
            res[(side, model)] = cols
            knees[(side, model)] = knee(Z_MAX_GRID_PC, cols["rchi2"])
            out[f"{side}_{model}_knee"] = np.array(knees[(side, model)])

    OUT_NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ_PATH, **out)
    print(f"Saved {OUT_NPZ_PATH}")

    # ================================================================= report
    def at(side, z_target):
        uu, ee = out[f"{side}_u"], out[f"{side}_err_log"]
        i = int(np.argmin(np.abs(uu - z_target)))
        return uu[i], ee[i]

    L = [
        PROVISIONAL_CUBE_HEADER,
        "DIAGNOSTIC: the height z_max beyond which <n_H>_area(z) stops looking like a decaying layer",
        "=" * 100,
        "Measures only -- no physics, cache, W, existing result or figure is changed.",
        "",
        "METHOD",
        "  Data: footprint-mean n_H over ALL cells (HIM included), RAW, |x|,|y| <= 500 pc, the cached",
        "  n_model via compute_area_alpha.footprint_mean_density_chunked (the Fig. 3 route), on the source",
        f"  cube's float64 z grid ({z[0]:g}..{z[-1]:g} pc, {z.size} planes).",
        f"  u = |z - z_mid|, z_mid = {z_mid:.4f} pc (balanced midplane, self-gravity on).",
        f"  Bins: {BIN_PC:g} pc in u, [4k, 4k+4), upper side (z > z_mid) and lower side (z < z_mid)",
        "  separately; bin density = mean over its planes, bin position = mean u of its planes.",
        "  Models (centred on z_mid; free n0, h; fitted in log10 space with scipy curve_fit, trf,",
        f"  bounds n0 in [{BOUNDS[0][0]:g}, {BOUNDS[1][0]:g}] cm^-3, h in [{BOUNDS[0][1]:g}, {BOUNDS[1][1]:g}] pc;",
        "  start n0 = first bin's density, h = 200 pc (exp) / 150 pc (gauss, sech2)):",
        "    exp    n0 exp(-u/h)      gauss  n0 exp(-u^2/(2h^2))      sech2  n0 sech(u/h)^2",
        "  chi2 = sum((log10 data - log10 model)^2 / err^2) over bins with upper edge <= z_max;",
        "  reduced chi2 = chi2 / (N_bins - 2). z_max = 100..750 pc in 10 pc steps.",
        f"  n_fit(750) = the fit's density at u = {PRED_U_PC:g} pc.",
        "",
        "ERROR DEFINITION",
        "  The footprint is split into the 16 disjoint 250 x 250 pc squares (compute_area_alpha.patch_ids,",
        f"  N = 4; {int(cnt.min())}..{int(cnt.max())} cells per square per plane -- the 501-cell axis is odd).",
        "  err(log10 <n_H>) per bin = std(log10 of the 16 square profiles, ddof = 1) / sqrt(n_squares).",
        "  A square with <n_H> <= 0 in a bin is dropped for that bin (n_squares is then the number kept).",
    ]
    for side in SIDES:
        if dropped_all[side]:
            L.append(f"  {side}: dropped (u_lo, square) = {dropped_all[side]}")
        else:
            L.append(f"  {side}: no square dropped in any bin (all 16 squares positive everywhere).")
    for side in SIDES:
        u0, e0 = at(side, 0.0)
        u4, e4 = at(side, 400.0)
        L.append(f"  {side}: err = {e0:.4f} dex at u = {u0:.1f} pc ({100 * (10 ** e0 - 1):.1f}%), "
                 f"{e4:.4f} dex at u = {u4:.1f} pc ({100 * (10 ** e4 - 1):.1f}%)")
    L += ["", "ASSERTIONS (the script raises if any fails)"]
    for name, meas, thr in checks:
        L.append(f"  {meas:>10.2e}  (<= {thr:.0e})  {name}")

    L += ["", "FIT FAILURES / FITS AT A BOUND"]
    if failures:
        for side, model, zm, status in failures:
            L.append(f"  {side:5s} {model:5s} z_max = {zm:5.0f}: {status}")
    else:
        L.append("  none -- every fit converged inside the bounds.")

    L += ["", "KNEE",
          f"  Rule: the first z_max at or beyond the minimum of reduced chi2 over z_max <= "
          f"{KNEE_SEARCH_MAX_PC:g} pc",
          f"  where reduced chi2 exceeds {KNEE_FACTOR:g}x that minimum.",
          f"  {'side':5s} {'model':5s} {'min rchi2':>9s} {'at z_max':>8s} {'knee':>6s} "
          f"{'n0 [cm^-3]':>10s} {'h [pc]':>8s} {'rchi2':>8s} {'n_fit(750)':>10s}"]
    for side in SIDES:
        for model in MODELS:
            zk, rmin, zmin = knees[(side, model)]
            c = res[(side, model)]
            if np.isfinite(zk):
                i = int(np.argmin(np.abs(Z_MAX_GRID_PC - zk)))
                L.append(f"  {side:5s} {model:5s} {rmin:9.3f} {zmin:8.0f} {zk:6.0f} {c['n0'][i]:10.4f} "
                         f"{c['h'][i]:8.1f} {c['rchi2'][i]:8.3f} {c['n_fit750'][i]:10.3e}")
            else:
                L.append(f"  {side:5s} {model:5s} {rmin:9.3f} {zmin:8.0f}   none")

    for side in SIDES:
        for model in MODELS:
            c = res[(side, model)]
            L += ["", f"FULL TABLE -- {side} side, {model}",
                  f"  {'z_max':>5s} {'N':>4s} {'n0 [cm^-3]':>10s} {'h [pc]':>9s} {'rchi2':>9s} "
                  f"{'n_fit(750)':>10s}"]
            for i, zm in enumerate(Z_MAX_GRID_PC):
                L.append(f"  {zm:5.0f} {int(c['N_bins'][i]):4d} {c['n0'][i]:10.4f} {c['h'][i]:9.2f} "
                         f"{c['rchi2'][i]:9.3f} {c['n_fit750'][i]:10.3e}")
    L += ["", f"runtime {time.time() - t0:.0f} s"]
    OUT_TXT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_TXT_PATH}")


if __name__ == "__main__":
    main()
