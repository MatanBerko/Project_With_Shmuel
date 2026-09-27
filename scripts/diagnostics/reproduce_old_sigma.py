"""
Diagnostic: reproduce the months-old Sigma_gas figure (median ~3.3, "mean"
~3.1 Msun/pc^2) from research_10.0/fig2_histograms/{computing_data.py,
designing_histograms.py}, starting from the current repo's verified-correct
method and cumulatively re-introducing the old script's specific deviations.

Read-only with respect to research_10.0 and the rest of this repo's
pipeline: this script only reads the zarr cube and BS19 .mat file via
src.config_loader (both already used elsewhere in this repo), and does not
import, modify, or write to anything under research_10.0 or pipeline/.

Confirmed by direct inspection of the old scripts (not assumed):
  A) computing_data.py:29  DZ_PC = 1.0  (hardcoded), while the zarr's real
     z spacing is 2.0 pc (computing_data.py:219-224 sums density planes and
     multiplies by DZ_PC*PC_CM instead of trapezoidal-integrating against
     the real z_pc coordinate array) -> integral ~2x too low.
  B) computing_data.py:254-271  Sigma_gas per sightline is filtered by
     `neutral = (~him_mask[zi].ravel()) & r_mask_flat` for zi at z=0 only
     -- i.e. every sightline's *total-column* Sigma_gas is kept or dropped
     based only on whether that column's *z=0 plane* is HIM
     (Pth < 0.5*Pmin) and within R<=500 pc.
  C) The z range used for the Sigma_gas sum itself is NOT restricted --
     computing_data.py's full_z_n() iterates `for iz in
     range(len(z_pc_full))`, i.e. the entire +-750 pc column, same as the
     current repo pipeline. This hypothesis is REFUTED; kept fixed at
     +-750 pc in every row below, it is not a free variable.
  D) designing_histograms.py:283-284,321-322 the plotted quantity is
     x = log10(Sigma_gas), and weighted_mean(x, w) at line 159-165 is
     `np.average(x[fin], weights=w[fin])` -- i.e. the "mean" line is the
     (weighted) mean of log10(Sigma_gas), not of Sigma_gas itself. Because
     median commutes with a monotonic transform, the median line is
     unaffected (median(log10 Sigma) exponentiated == median(Sigma)), but
     the "mean" line is a geometric mean, not an arithmetic one.

Cumulative steps computed here, all on the SAME underlying zarr "density"
field used unmodified as n_H (no dustmaps re-derivation, matching both the
old script and the current repo's compute_sigma_gas.py):
  (0) correct   : trapezoidal integration against real z_pc, full +-750 pc
                  column, R<=500 pc cylinder footprint.
  (1) + old dz  : replace trapz with sum(n)*DZ_PC_OLD (DZ_PC_OLD=1.0, the
                  old script's hardcoded value) -- isolates cause A.
  (2) + old mask: additionally restrict to sightlines whose z=0 plane is
                  non-HIM (Pth < 0.5*Pmin) -- isolates cause B.
  (3) + old stat: additionally compute "mean" as 10**mean(log10 Sigma))
                  instead of the arithmetic mean, under both weightings
                  the old design script actually offered (unweighted
                  "sightline" and Sigma_gas-weighted "mass") -- isolates
                  cause D and settles which weighting was plotted.
"""

from pathlib import Path

import dask.array as da
import numpy as np
import scipy.io
import zarr
from scipy.interpolate import interp1d

from src.config_loader import load_resolved_config

# ---------------------------------------------------------------------------
# Control block
# ---------------------------------------------------------------------------
XY_HALF_RANGE_PC = 500.0  # footprint used everywhere below (R<=500 pc cylinder)
R_MAX_PC = 500.0
Z_PLOT_PC = 0.0  # the z-slice the old script used for the neutral/HIM mask

DZ_PC_OLD = 1.0  # old script's hardcoded (wrong) voxel size, computing_data.py:29
T_HIM = 1.0e6  # unused in the Sigma formula itself; kept for parity with the old script

# Old script's exact physical constants (computing_data.py:43-47), reused
# throughout so only the four listed causes vary between rows.
MU = 1.4
M_H = 1.6735575e-24  # g
PC_TO_CM = 3.085677581491367e18  # cm per pc
MSUN_G = 1.989e33  # g per solar mass
IUV_FLOOR = 1e-30

RHO_FIELD = "density"
IUV_FIELD = "Iuv_final"
T_FIELD = "T"

CACHE_DIR = Path("cache/diagnostics")
CUBE_CACHE_PATH = CACHE_DIR / "old_sigma_repro_cube.npz"
RESULTS_TXT_PATH = CACHE_DIR / "old_sigma_repro_stats.txt"

TARGET_MEDIAN = 3.3
TARGET_MEAN_LOG = 3.1


# ---------------------------------------------------------------------------
# Pmin(IUV)/Pmax(IUV) interpolators -- reproduces
# computing_data.py:78-124 build_pmin_pmax_functions() exactly, reading the
# BS19 .mat file via this repo's own config resolution instead of a
# hardcoded research_10.0 path.
# ---------------------------------------------------------------------------
def build_pmin_pmax_functions(mat_path):
    d = scipy.io.loadmat(str(mat_path))

    IUV_grid = np.array(d["IUV_"]).squeeze().astype(float).ravel()
    P_grid = np.array(d["P_"]).squeeze().astype(float).ravel()
    T_2d_P = np.array(d["T_2d_P"]).astype(float)

    if np.any(np.diff(P_grid) <= 0):
        order = np.argsort(P_grid)
        P_grid = P_grid[order]
        T_2d_P = T_2d_P[:, order]

    Pmin_list, Pmax_list = [], []
    for i in range(len(IUV_grid)):
        T_row = np.clip(T_2d_P[i, :], 1e-30, None)
        n_row = np.clip(P_grid / T_row, 1e-30, None)
        dn = np.gradient(n_row, P_grid)
        s = np.sign(dn)
        s[s == 0] = 1
        ch = np.where(s[:-1] * s[1:] < 0)[0]

        if ch.size == 0:
            Pmin_list.append(np.nan)
            Pmax_list.append(np.nan)
            continue

        def lin_root(j):
            x0, x1 = P_grid[j], P_grid[j + 1]
            y0, y1 = dn[j], dn[j + 1]
            return x0 - y0 * (x1 - x0) / (y1 - y0) if (y1 - y0) != 0 else x0

        Pmin_list.append(float(lin_root(int(ch[0]))))
        Pmax_list.append(float(lin_root(int(ch[-1]))))

    Pmin_arr = np.array(Pmin_list, dtype=float)
    Pmax_arr = np.array(Pmax_list, dtype=float)

    def make_interp(P_arr):
        good = np.isfinite(P_arr) & (P_arr > 0) & (IUV_grid > 0)
        logI = np.log10(IUV_grid[good])
        logP = np.log10(P_arr[good])
        idx = np.argsort(logI)
        f = interp1d(logI[idx], logP[idx], kind="linear",
                      bounds_error=False,
                      fill_value=(logP[idx][0], logP[idx][-1]))
        return lambda Iuv2d: 10.0 ** f(
            np.log10(np.clip(np.asarray(Iuv2d, float), IUV_FLOOR, None)))

    return make_interp(Pmin_arr), make_interp(Pmax_arr)


# ---------------------------------------------------------------------------
# Cube construction (query once, cache, reuse)
# ---------------------------------------------------------------------------
def build_or_load_cube():
    if CUBE_CACHE_PATH.exists():
        print(f"Loading cached sub-cube from {CUBE_CACHE_PATH}")
        d = np.load(CUBE_CACHE_PATH)
        return (d["x_pc"], d["y_pc"], d["z_pc"], d["n_sub"], d["T_sub"], d["Iuv_sub"])

    cfg = load_resolved_config()
    store = zarr.open(str(cfg["zarr_path"]), mode="r")

    x_pc = store["x_pc"][:]
    y_pc = store["y_pc"][:]
    z_pc = store["z_pc"][:]

    x_idx = np.where(np.abs(x_pc) <= XY_HALF_RANGE_PC)[0]
    y_idx = np.where(np.abs(y_pc) <= XY_HALF_RANGE_PC)[0]
    x_lo, x_hi = x_idx.min(), x_idx.max() + 1
    y_lo, y_hi = y_idx.min(), y_idx.max() + 1
    x_sub = x_pc[x_lo:x_hi]
    y_sub = y_pc[y_lo:y_hi]

    iz0 = int(np.argmin(np.abs(z_pc)))
    print(f"x_pc range [{x_sub.min()}, {x_sub.max()}], {len(x_sub)} points")
    print(f"y_pc range [{y_sub.min()}, {y_sub.max()}], {len(y_sub)} points")
    print(f"z_pc range [{z_pc.min()}, {z_pc.max()}], {len(z_pc)} points, "
          f"spacing {z_pc[1] - z_pc[0]:.3f} pc")
    print(f"z=0 plane index: {iz0} (z_pc[{iz0}]={z_pc[iz0]})")

    print(f"Loading {RHO_FIELD}/{T_FIELD}/{IUV_FIELD} sub-cubes "
          f"(full z column, +-{XY_HALF_RANGE_PC:.0f} pc XY)...")
    n_sub = da.from_zarr(store[RHO_FIELD])[:, y_lo:y_hi, x_lo:x_hi].compute().astype(np.float32)
    T_sub = da.from_zarr(store[T_FIELD])[:, y_lo:y_hi, x_lo:x_hi].compute().astype(np.float32)
    Iuv_sub = da.from_zarr(store[IUV_FIELD])[:, y_lo:y_hi, x_lo:x_hi].compute().astype(np.float32)
    print(f"  n_sub shape {n_sub.shape}, ~{n_sub.nbytes / 1e6:.1f} MB each (x3 fields)")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CUBE_CACHE_PATH, x_pc=x_sub, y_pc=y_sub, z_pc=z_pc,
              n_sub=n_sub, T_sub=T_sub, Iuv_sub=Iuv_sub)
    print(f"Cached sub-cube to {CUBE_CACHE_PATH}")
    return x_sub, y_sub, z_pc, n_sub, T_sub, Iuv_sub


# ---------------------------------------------------------------------------
# Sigma_gas unit chain (old script's exact constants)
# ---------------------------------------------------------------------------
def sigma_gas_from_N_H_column(N_H_column_pc_cm3):
    N_H_cm2 = N_H_column_pc_cm3 * PC_TO_CM
    Sigma_g_cm2 = MU * M_H * N_H_cm2
    return Sigma_g_cm2 * (PC_TO_CM ** 2) / MSUN_G


def weighted_median(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    fin = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[fin], w[fin]
    if x.size == 0:
        return np.nan
    order = np.argsort(x)
    x, w = x[order], w[order]
    cumw = np.cumsum(w)
    return float(x[np.searchsorted(cumw, cumw[-1] * 0.5)])


def weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    fin = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if fin.sum() == 0:
        return np.nan
    return float(np.average(x[fin], weights=w[fin]))


def linear_stats(sigma, mask, label):
    vals = sigma[mask]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    n = vals.size
    if n == 0:
        return {"label": label, "n": 0}
    return {
        "label": label,
        "n": n,
        "median": float(np.median(vals)),
        "arithmetic_mean": float(np.mean(vals)),
        "mean_of_log10": float(10 ** np.mean(np.log10(vals))),
    }


def format_row(stats, extra=""):
    if stats["n"] == 0:
        return f"{stats['label']:>34s}:  N=0" + extra
    return (
        f"{stats['label']:>34s}:  N={stats['n']:6d}  "
        f"median={stats['median']:7.4f}  "
        f"arith_mean={stats['arithmetic_mean']:7.4f}  "
        f"mean_of_log10={stats['mean_of_log10']:7.4f}" + extra
    )


def main():
    x_pc, y_pc, z_pc, n_sub, T_sub, Iuv_sub = build_or_load_cube()
    iz0 = int(np.argmin(np.abs(z_pc)))
    T_z0 = T_sub[iz0]
    Iuv_z0 = Iuv_sub[iz0]

    X, Y = np.meshgrid(x_pc, y_pc)
    R = np.sqrt(X ** 2 + Y ** 2)
    cylinder_mask = R <= R_MAX_PC
    print(f"R<=500 pc cylinder: {cylinder_mask.sum()} / {cylinder_mask.size} cells")

    # ---- step 0: correct trapezoidal integration, full column ----
    N_H_trapz = np.trapezoid(n_sub, x=z_pc, axis=0)  # (y, x), cm^-3*pc
    sigma_v0 = sigma_gas_from_N_H_column(N_H_trapz)

    # ---- step 1: + old dz handling (sum * DZ_PC_OLD instead of trapz) ----
    N_H_oldsum = np.sum(n_sub, axis=0) * DZ_PC_OLD  # cm^-3*pc, wrong dz
    sigma_v1 = sigma_gas_from_N_H_column(N_H_oldsum)

    # ---- step 2: + old neutral-at-z=0 mask ----
    cfg = load_resolved_config()
    Pmin_func, Pmax_func = build_pmin_pmax_functions(cfg["bs19_mat_path"])
    Pth_z0 = n_sub[iz0].astype(float) * T_z0.astype(float)
    Pmin_z0 = Pmin_func(Iuv_z0)
    him_z0 = (Pth_z0 < 0.5 * Pmin_z0) & np.isfinite(Pmin_z0)
    neutral_z0 = ~him_z0
    print(f"z=0 HIM fraction (within full XY subset): {him_z0.mean():.4%}")

    mask_step2 = cylinder_mask & neutral_z0
    print(f"cylinder & neutral-at-z=0: {mask_step2.sum()} / {cylinder_mask.sum()} "
          f"cylinder sightlines kept ({100 * mask_step2.sum() / cylinder_mask.sum():.1f}%)")

    # ---- residual check: does the old script's HIM_B density substitution
    # (applied at EVERY z-plane, not just z=0 -- computing_data.py:191-210
    # full_z_n(), used for MODEL="HIM_B" which designing_histograms.py:24
    # actually plots) close the gap left after steps 0-3? Full-column,
    # not one of the four listed causes, reported separately as a residual
    # explanation rather than folded into the main cumulative table.
    print("Residual check: computing full-column HIM_B density substitution...")
    Pth_full = n_sub.astype(np.float64) * T_sub.astype(np.float64)
    Pmin_full = Pmin_func(Iuv_sub)
    Pmax_full = Pmax_func(Iuv_sub)
    him_full = (Pth_full < 0.5 * Pmin_full) & np.isfinite(Pmin_full)
    n_HIMB = n_sub.astype(np.float64).copy()
    n_HIMB[him_full] = Pmax_full[him_full] / T_HIM
    print(f"  full-column HIM fraction: {him_full.mean():.4%}")

    N_H_oldsum_HIMB = np.sum(n_HIMB, axis=0) * DZ_PC_OLD
    sigma_v1_HIMB = sigma_gas_from_N_H_column(N_H_oldsum_HIMB)
    residual_stats = linear_stats(sigma_v1_HIMB, mask_step2,
                                    "(2r) step 2 + full-column HIM_B density subst.")

    # ---- reports (linear stats: median, arithmetic mean, mean-of-log10) ----
    rows = []
    rows.append(linear_stats(sigma_v0, cylinder_mask, "(0) correct: trapz, full column, cylinder"))
    rows.append(linear_stats(sigma_v1, cylinder_mask, "(1) + old dz (sum*1.0 pc)"))
    rows.append(linear_stats(sigma_v1, mask_step2, "(2) + old neutral-at-z=0 mask"))

    # ---- step 3: + old weighting/mean definition (log10 stats), both weightings ----
    vals_step2 = sigma_v1[mask_step2]
    vals_step2 = vals_step2[np.isfinite(vals_step2) & (vals_step2 > 0)]
    log_vals = np.log10(vals_step2)

    unweighted = np.ones_like(vals_step2)
    mass_weighted = vals_step2.copy()

    med_log_unw = 10 ** weighted_median(log_vals, unweighted)
    mean_log_unw = 10 ** weighted_mean(log_vals, unweighted)
    med_log_w = 10 ** weighted_median(log_vals, mass_weighted)
    mean_log_w = 10 ** weighted_mean(log_vals, mass_weighted)

    # residual-corrected version of step 3 (unweighted, log-mean stat, on
    # top of the full-column HIM_B substitution)
    vals_residual = sigma_v1_HIMB[mask_step2]
    vals_residual = vals_residual[np.isfinite(vals_residual) & (vals_residual > 0)]
    log_vals_residual = np.log10(vals_residual)
    med_log_residual = 10 ** weighted_median(log_vals_residual, np.ones_like(vals_residual))
    mean_log_residual = 10 ** weighted_mean(log_vals_residual, np.ones_like(vals_residual))

    lines = []
    lines.append("Old Sigma_gas figure reproduction - cumulative decomposition")
    lines.append("=" * 78)
    lines.append(f"Zarr grid spacing (from coordinate arrays): "
                 f"dx={x_pc[1]-x_pc[0]:.3f} pc, dy={y_pc[1]-y_pc[0]:.3f} pc, "
                 f"dz={z_pc[1]-z_pc[0]:.3f} pc")
    lines.append(f"Old script's hardcoded DZ_PC = {DZ_PC_OLD} pc "
                 f"(vs real dz = {z_pc[1]-z_pc[0]:.3f} pc)")
    lines.append("")
    lines.append("Cumulative steps (median / arith_mean / mean-of-log10, all in Msun/pc^2):")
    prev_median = None
    for r in rows:
        effect = ""
        if prev_median is not None and r["n"] > 0:
            effect = f"   [median x {r['median']/prev_median:.4f} vs previous row]"
        lines.append(format_row(r, effect))
        if r["n"] > 0:
            prev_median = r["median"]

    lines.append("")
    lines.append("(3) + old stat definition (mean = 10**mean(log10 Sigma)), both weightings:")
    lines.append(f"{'unweighted (sightline-weighted)':>34s}:  "
                 f"median={med_log_unw:7.4f}  mean_of_log10={mean_log_unw:7.4f}"
                 f"   [median x {med_log_unw/prev_median:.4f} vs step 2]")
    lines.append(f"{'Sigma_gas-weighted (mass-weighted)':>34s}:  "
                 f"median={med_log_w:7.4f}  mean_of_log10={mean_log_w:7.4f}"
                 f"   [median x {med_log_w/prev_median:.4f} vs step 2]")
    lines.append("")
    lines.append("Residual check -- full-column HIM_B density substitution "
                 "(not one of the 4 listed causes, applied on top of step 2):")
    lines.append(format_row(residual_stats,
                              f"   [median x {residual_stats['median']/prev_median:.4f} vs step 2]"
                              if residual_stats["n"] > 0 else ""))
    lines.append(f"{'(2r) + old stat (log-mean, unweighted)':>34s}:  "
                 f"median={med_log_residual:7.4f}  mean_of_log10={mean_log_residual:7.4f}")
    lines.append("")
    lines.append(f"Target from the old figure: median ~= {TARGET_MEDIAN}, "
                 f"\"mean\" (mean of log10 Sigma) ~= {TARGET_MEAN_LOG}")
    closer = "unweighted" if abs(med_log_unw - TARGET_MEDIAN) < abs(med_log_w - TARGET_MEDIAN) else "Sigma_gas-weighted"
    lines.append(f"Closer match to the old figure's weighting: {closer}")

    report = "\n".join(lines)
    print()
    print(report)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_TXT_PATH.write_text(report + "\n")
    print(f"\nSaved {RESULTS_TXT_PATH}")


if __name__ == "__main__":
    main()
