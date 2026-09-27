"""
Diagnostic: decompose the ~2x Sigma_gas offset between an old standalone
script's result (median ~3.3 Msun/pc^2) and this repo's
compute_sigma_gas_factor_comparison.py result (sigma_gas_old: median 4.77,
mean 6.79), which use the same n_H = ~1652-1653 * dE/ds conversion factor.

Read-only with respect to the rest of the pipeline: this script only queries
dustmaps.edenhofer2023.Edenhofer2023Query directly and writes its own cache/
results files under cache/diagnostics/. It does not import or modify any
existing pipeline/*.py script.

Method: query ONE (x, y, z) cube of the native Edenhofer dE/ds field on a
grid wide enough to cover both the OLD script's z range (+-400 pc) and the
repo's z range (+-750 pc), then slice that single cube four different ways
(z range x footprint shape) to isolate exactly how much of the offset comes
from each choice, without requerying dustmaps per configuration.
"""

import time
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import Galactic, SkyCoord
from dustmaps.edenhofer2023 import Edenhofer2023Query

# ---------------------------------------------------------------------------
# Control block
# ---------------------------------------------------------------------------
XY_HALF_RANGE_PC = 500.0
XY_SPACING_PC = 10.0  # matches the OLD script's XY grid spacing
Z_HALF_RANGE_PC = 750.0  # covers both OLD (+-400) and repo (+-750) z ranges
Z_SPACING_PC = 5.0

# Old script's exact factor (1653) is used for the OLD config; the repo's
# factor comparison script uses 1652 -- both are carried through so the
# 0.06% difference between them can be quoted, not hidden.
FACTOR_N_H_OLD = 1653.0  # cm^-3 per (E/pc), OLD standalone script
FACTOR_N_H_REPO = 1652.0  # cm^-3 per (E/pc), compute_sigma_gas_factor_comparison.py

MU = 1.4  # mean molecular weight per H (helium included), same as repo pipeline
M_H = 1.6726219e-24  # g
PC_TO_CM = 3.0856775814913673e18  # cm per pc
MSUN_G = 1.98892e33  # g per solar mass

COVERAGE_THRESHOLD = 0.9
CYLINDER_R_PC = 500.0
Z_OLD_HALF_RANGE_PC = 400.0
FLOOR_Z_ABS_MIN_PC = 600.0  # |z| >= this counts as "density floor" region
MODE_LOG_BIN_DEX = 0.05

CACHE_DIR = Path("cache/diagnostics")
CUBE_CACHE_PATH = CACHE_DIR / "sigma_offset_cube.npz"
RESULTS_NPZ_PATH = CACHE_DIR / "sigma_offset_results.npz"
RESULTS_TXT_PATH = CACHE_DIR / "sigma_offset_stats.txt"

NEW_PIPELINE_CACHE_PATH = Path("cache/sigma_gas_factor_comparison.npz")


# ---------------------------------------------------------------------------
# Cube construction (query once, cache, reuse)
# ---------------------------------------------------------------------------
def build_or_load_cube():
    if CUBE_CACHE_PATH.exists():
        print(f"Loading cached dE/ds cube from {CUBE_CACHE_PATH}")
        d = np.load(CUBE_CACHE_PATH)
        return d["x_pc"], d["y_pc"], d["z_pc"], d["dEds"]

    x_pc = np.arange(-XY_HALF_RANGE_PC, XY_HALF_RANGE_PC + XY_SPACING_PC / 2, XY_SPACING_PC)
    y_pc = np.arange(-XY_HALF_RANGE_PC, XY_HALF_RANGE_PC + XY_SPACING_PC / 2, XY_SPACING_PC)
    z_pc = np.arange(-Z_HALF_RANGE_PC, Z_HALF_RANGE_PC + Z_SPACING_PC / 2, Z_SPACING_PC)

    print(f"Grid: x {x_pc.size} pts [{x_pc.min()},{x_pc.max()}], "
          f"y {y_pc.size} pts [{y_pc.min()},{y_pc.max()}], "
          f"z {z_pc.size} pts [{z_pc.min()},{z_pc.max()}]")

    print("Loading Edenhofer2023Query (integrated=False)...")
    t0 = time.time()
    query = Edenhofer2023Query(integrated=False)
    print(f"  loaded in {time.time() - t0:.1f}s")

    X, Y = np.meshgrid(x_pc, y_pc)  # shape (y, x)
    n_z = len(z_pc)
    dEds = np.empty((n_z, len(y_pc), len(x_pc)), dtype=np.float32)

    t_start = time.time()
    for iz, z_val in enumerate(z_pc):
        Z = np.full_like(X, z_val)
        gal = Galactic(u=X * u.pc, v=Y * u.pc, w=Z * u.pc, representation_type="cartesian")
        coords = SkyCoord(gal)
        coords.representation_type = "spherical"
        # No try/except here: a query failure must raise, not be silently
        # papered over with a NaN fill.
        dEds[iz] = query.query(coords).astype(np.float32)

        elapsed = time.time() - t_start
        if iz % 25 == 0 or iz == n_z - 1:
            rate = (iz + 1) / elapsed if elapsed > 0 else float("nan")
            eta = (n_z - iz - 1) / rate if rate > 0 else float("nan")
            print(f"  z-slice {iz + 1}/{n_z}, elapsed {elapsed:.0f}s, ETA {eta:.0f}s")

    print(f"Finished querying dE/ds cube in {time.time() - t_start:.1f}s, "
          f"shape {dEds.shape}, NaN fraction {np.mean(np.isnan(dEds)):.4%}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CUBE_CACHE_PATH, x_pc=x_pc, y_pc=y_pc, z_pc=z_pc, dEds=dEds)
    print(f"Cached cube to {CUBE_CACHE_PATH}")
    return x_pc, y_pc, z_pc, dEds


# ---------------------------------------------------------------------------
# Sigma_gas from a dE/ds cube slice
# ---------------------------------------------------------------------------
def sigma_gas_from_dEds(dEds_zyx, z_sub, factor_n_h):
    """NaN -> 0 before integration (matches the OLD script); trapezoidal in z."""
    n_H = factor_n_h * np.nan_to_num(dEds_zyx, nan=0.0)
    N_H_column_pc_cm3 = np.trapezoid(n_H, x=z_sub, axis=0)  # (y, x)
    N_H_cm2 = N_H_column_pc_cm3 * PC_TO_CM
    Sigma_g_cm2 = MU * M_H * N_H_cm2
    return Sigma_g_cm2 * (PC_TO_CM ** 2) / MSUN_G


def coverage_fraction(dEds_zyx):
    """Finite fraction per column, computed on the RAW (unfilled) query output."""
    finite = np.isfinite(dEds_zyx)
    return finite.mean(axis=0)  # (y, x)


def mode_of_log_sigma(sigma_values, bin_dex=MODE_LOG_BIN_DEX):
    log_s = np.log10(sigma_values[sigma_values > 0])
    lo = np.floor(log_s.min() / bin_dex) * bin_dex
    hi = np.ceil(log_s.max() / bin_dex) * bin_dex
    bins = np.arange(lo, hi + bin_dex, bin_dex)
    counts, edges = np.histogram(log_s, bins=bins)
    peak = np.argmax(counts)
    center_log = 0.5 * (edges[peak] + edges[peak + 1])
    return float(10 ** center_log)


def compute_stats(sigma, mask, label):
    vals = sigma[mask]
    vals = vals[np.isfinite(vals)]
    n = vals.size
    if n == 0:
        return {"label": label, "n": 0}
    log_vals = np.log10(vals[vals > 0])
    return {
        "label": label,
        "n": n,
        "median": float(np.median(vals)),
        "arithmetic_mean": float(np.mean(vals)),
        "geometric_mean": float(10 ** np.mean(log_vals)),
        "mode": mode_of_log_sigma(vals),
        "frac_below_1": float(np.mean(vals < 1.0)),
    }


def format_stats_row(stats):
    if stats["n"] == 0:
        return f"{stats['label']:>28s}:  N=0 (no columns passed coverage cut)"
    return (
        f"{stats['label']:>28s}:  N={stats['n']:6d}  "
        f"median={stats['median']:7.3f}  "
        f"arith_mean={stats['arithmetic_mean']:7.3f}  "
        f"geom_mean={stats['geometric_mean']:7.3f}  "
        f"mode={stats['mode']:7.3f}  "
        f"frac(Sigma<1)={stats['frac_below_1']:.4f}"
    )


def main():
    x_pc, y_pc, z_pc, dEds = build_or_load_cube()
    X, Y = np.meshgrid(x_pc, y_pc)
    R = np.sqrt(X ** 2 + Y ** 2)
    cylinder_mask = R <= CYLINDER_R_PC
    square_mask = np.ones_like(cylinder_mask, dtype=bool)

    z_idx_400 = np.abs(z_pc) <= Z_OLD_HALF_RANGE_PC
    z_idx_750 = np.abs(z_pc) <= Z_HALF_RANGE_PC  # full cube, all True by construction

    dEds_400 = dEds[z_idx_400]
    dEds_750 = dEds[z_idx_750]
    z_sub_400 = z_pc[z_idx_400]
    z_sub_750 = z_pc[z_idx_750]

    sigma_400 = sigma_gas_from_dEds(dEds_400, z_sub_400, FACTOR_N_H_OLD)
    sigma_750 = sigma_gas_from_dEds(dEds_750, z_sub_750, FACTOR_N_H_OLD)

    coverage_400 = coverage_fraction(dEds_400)
    coverage_750 = coverage_fraction(dEds_750)

    cov_ok_400 = coverage_400 >= COVERAGE_THRESHOLD
    cov_ok_750 = coverage_750 >= COVERAGE_THRESHOLD

    configs = {
        "OLD (400, cylinder)": compute_stats(sigma_400, cylinder_mask & cov_ok_400, "OLD (400, cylinder)"),
        "OLD, no coverage cut": compute_stats(sigma_400, cylinder_mask, "OLD, no coverage cut"),
        "(400, square)": compute_stats(sigma_400, square_mask & cov_ok_400, "(400, square)"),
        "(750, cylinder)": compute_stats(sigma_750, cylinder_mask & cov_ok_750, "(750, cylinder)"),
        "(750, square) = expected NEW": compute_stats(sigma_750, square_mask & cov_ok_750, "(750, square) = expected NEW"),
    }

    # "new pipeline as-is": recompute the same 4 statistics directly from the
    # existing repo cache, for a like-for-like comparison row.
    new_pipeline_row = None
    if NEW_PIPELINE_CACHE_PATH.exists():
        d = np.load(NEW_PIPELINE_CACHE_PATH)
        sigma_new_pipeline = d["sigma_gas_old"]
        mask_all = np.isfinite(sigma_new_pipeline)
        new_pipeline_row = compute_stats(sigma_new_pipeline, mask_all, "new pipeline as-is")

    # Density floor diagnostic: median n_H at |z| >= 600 pc, and the implied
    # extra Sigma contributed by the 400 < |z| <= 750 shell.
    floor_z_mask = np.abs(z_pc) >= FLOOR_Z_ABS_MIN_PC
    dEds_floor = dEds[floor_z_mask]
    finite_floor = dEds_floor[np.isfinite(dEds_floor)]
    n_H_floor = FACTOR_N_H_OLD * finite_floor
    median_n_H_floor = float(np.median(n_H_floor)) if n_H_floor.size else float("nan")

    extra_shell_mask = (np.abs(z_pc) > Z_OLD_HALF_RANGE_PC) & (np.abs(z_pc) <= Z_HALF_RANGE_PC)
    dEds_shell = dEds[extra_shell_mask]
    z_sub_shell = z_pc[extra_shell_mask]
    # Trapezoidal integration needs each side's z's to be contiguous and
    # sorted; split into the two physically separate shells and sum.
    neg_side = z_sub_shell < 0
    pos_side = z_sub_shell > 0
    sigma_shell = np.zeros_like(sigma_750)
    if neg_side.sum() >= 2:
        sigma_shell += sigma_gas_from_dEds(dEds_shell[neg_side], z_sub_shell[neg_side], FACTOR_N_H_OLD)
    if pos_side.sum() >= 2:
        sigma_shell += sigma_gas_from_dEds(dEds_shell[pos_side], z_sub_shell[pos_side], FACTOR_N_H_OLD)

    valid_shell_cols = square_mask & cov_ok_750
    median_extra_sigma_shell = float(np.median(sigma_shell[valid_shell_cols]))

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    lines = []
    lines.append("Sigma_gas offset diagnostic - controlled decomposition")
    lines.append("=" * 72)
    lines.append(f"Grid: XY +-{XY_HALF_RANGE_PC:.0f} pc @ {XY_SPACING_PC:.0f} pc, "
                 f"Z +-{Z_HALF_RANGE_PC:.0f} pc @ {Z_SPACING_PC:.0f} pc")
    lines.append(f"n_H factor (OLD script): {FACTOR_N_H_OLD}, "
                 f"(repo factor comparison script): {FACTOR_N_H_REPO}, "
                 f"ratio {FACTOR_N_H_OLD / FACTOR_N_H_REPO:.6f}")
    lines.append("")
    lines.append("Four-way decomposition (coverage >= 0.9 applied unless noted):")
    for key in ["OLD (400, cylinder)", "OLD, no coverage cut", "(400, square)",
                "(750, cylinder)", "(750, square) = expected NEW"]:
        lines.append(format_stats_row(configs[key]))
    if new_pipeline_row is not None:
        lines.append(format_stats_row(new_pipeline_row))
    else:
        lines.append(f"{'new pipeline as-is':>28s}:  SKIPPED ({NEW_PIPELINE_CACHE_PATH} not found)")
    lines.append("")
    lines.append(f"Median n_H at |z| >= {FLOOR_Z_ABS_MIN_PC:.0f} pc (density floor): "
                 f"{median_n_H_floor:.4g} cm^-3")
    lines.append(f"Median implied extra Sigma_gas from {Z_OLD_HALF_RANGE_PC:.0f} < |z| <= "
                 f"{Z_HALF_RANGE_PC:.0f} pc shell (both sides): "
                 f"{median_extra_sigma_shell:.4f} Msun/pc^2")
    report = "\n".join(lines)
    print()
    print(report)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_TXT_PATH.write_text(report + "\n")
    print(f"\nSaved {RESULTS_TXT_PATH}")

    save_kwargs = dict(
        x_pc=x_pc, y_pc=y_pc,
        sigma_400=sigma_400, sigma_750=sigma_750,
        coverage_400=coverage_400, coverage_750=coverage_750,
        cylinder_mask=cylinder_mask,
        median_n_H_floor=median_n_H_floor,
        median_extra_sigma_shell=median_extra_sigma_shell,
    )
    for key, stats in configs.items():
        prefix = key.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "").replace("=", "").lower()
        for stat_key, stat_val in stats.items():
            if stat_key == "label":
                continue
            save_kwargs[f"{prefix}__{stat_key}"] = stat_val
    if new_pipeline_row is not None:
        for stat_key, stat_val in new_pipeline_row.items():
            if stat_key == "label":
                continue
            save_kwargs[f"new_pipeline_as_is__{stat_key}"] = stat_val

    np.savez(RESULTS_NPZ_PATH, **save_kwargs)
    print(f"Saved {RESULTS_NPZ_PATH}")


if __name__ == "__main__":
    main()
