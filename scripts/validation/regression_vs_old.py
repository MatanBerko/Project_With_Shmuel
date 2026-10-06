"""
Phase A regression: compare the new src.physics core against the OLD
reference scripts' own cached output (read-only .npz caches under
research_10.0/, never rerun or reimplemented from research_10.0 itself),
for P_th, P_tot, alpha, sigma_nt, Mach across RAW/HIM_A/HIM_B, pinned to
the pre-Step-1d "nT" thermal-pressure convention (see apply_variant call
below) since that is what the old caches were built with.

Comparison conditions (matching the old scripts exactly, so any remaining
difference is attributable to the SETTLED changes, not incidental setup
drift):
  - self-gravity OFF (g_ext only)
  - OLD R<=500 pc cylinder footprint (src.physics.loading.cylinder_mask),
    not the new square footprint
  - z>=0 half only (the only domain the old scripts ever computed), 376
    planes, z grid confirmed to match the old caches' zpos exactly
  - NEUTRAL-only (HIM-excluded) statistics, matching the old scripts

Old caches read (read-only, under research_10.0/, resolved via the
research_10.0 path this session already has on disk -- never hardcoded
into any COMMITTED script; this validation script is itself not committed
data, but note this reads outside the repo, matching the task's own
"old .npz caches read-only" instruction):
  fig4_vertical_profiles/data/profiles_{RAW,HIM_A,HIM_B}.npz
  fig4b_velocity_dispertion_Mach_number/data/sigma_mach_{RAW,HIM_A,HIM_B}.npz

Prints and writes a max/median relative-difference table per
quantity/variant to stdout (captured into the Phase A report).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config_loader import load_resolved_config
from src.physics import derived, gravity, hydrostatic, thermal
from src.physics.him import apply_variant, him_flag, phase_flag, PHASE_CNM, PHASE_UNM, PHASE_WNM
from src.physics.loading import cylinder_mask, load_xy_subset_coords, open_zarr
from src.physics.stats import mass_weighted_stats, volume_weighted_stats
from src.conventions import M_H, MU, R_MAX_PC, THERMAL_PRESSURE_CONVENTION_NT

# OLD cache location -- research_10.0, read only, resolved by literal
# relative-to-repo-root path (this script itself lives outside pipeline/,
# is not part of the committed pipeline, and never writes there).
OLD_DATA_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "research_10.0"
OLD_PROFILES_DIR = OLD_DATA_ROOT / "fig4_vertical_profiles" / "data"
OLD_SIGMA_MACH_DIR = OLD_DATA_ROOT / "fig4b_velocity_dispertion_Mach_number" / "data"

VARIANTS = ("RAW", "HIM_A", "HIM_B")


def rel_diff(new, old):
    old = np.asarray(old, dtype=float)
    new = np.asarray(new, dtype=float)
    m = np.isfinite(old) & np.isfinite(new)
    denom = np.where(np.abs(old[m]) > 1e-30, np.abs(old[m]), np.nan)
    d = np.abs(new[m] - old[m]) / denom
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmax(d)), float(np.nanmedian(d))


def main():
    grid = open_zarr()
    x_sub, y_sub = load_xy_subset_coords(grid, half_range_pc=R_MAX_PC)
    cyl = cylinder_mask(x_sub, y_sub, r_max_pc=R_MAX_PC)

    pos_mask = grid.z_pc >= 0
    zpos = np.sort(grid.z_pc[pos_mask])
    n_z = len(zpos)

    cfg = load_resolved_config()
    pminmax = thermal.build_pmin_pmax(cfg["bs19_mat_path"])

    x_lo = int(np.where(np.abs(grid.x_pc) <= R_MAX_PC)[0].min())
    x_hi = int(np.where(np.abs(grid.x_pc) <= R_MAX_PC)[0].max()) + 1
    y_lo = int(np.where(np.abs(grid.y_pc) <= R_MAX_PC)[0].min())
    y_hi = int(np.where(np.abs(grid.y_pc) <= R_MAX_PC)[0].max()) + 1

    z_idx_sorted = np.argsort(grid.z_pc)
    z_idx_pos = z_idx_sorted[grid.z_pc[z_idx_sorted] >= 0]

    print(f"Streaming {n_z} z-planes x {len(y_sub)}x{len(x_sub)} XY for RAW/HIM_A/HIM_B...")
    n_model_cubes = {v: np.empty((n_z, len(y_sub), len(x_sub)), dtype=np.float64) for v in VARIANTS}
    Pth_model_cubes = {v: np.empty((n_z, len(y_sub), len(x_sub)), dtype=np.float64) for v in VARIANTS}
    him_cube = np.empty((n_z, len(y_sub), len(x_sub)), dtype=bool)
    T_cube = np.empty((n_z, len(y_sub), len(x_sub)), dtype=np.float64)

    for jj, iz in enumerate(z_idx_pos):
        n_raw = np.asarray(grid.store["density"][iz, y_lo:y_hi, x_lo:x_hi], dtype=np.float64)
        T = np.asarray(grid.store["T"][iz, y_lo:y_hi, x_lo:x_hi], dtype=np.float64)
        Iuv = np.asarray(grid.store["Iuv_final"][iz, y_lo:y_hi, x_lo:x_hi], dtype=np.float64)
        Pth_raw = n_raw * T
        Pmin = pminmax.p_min(Iuv)
        Pmax = pminmax.p_max(Iuv)
        him = him_flag(Pth_raw, Pmin)
        him_cube[jj] = him
        T_cube[jj] = T
        for v in VARIANTS:
            # Step 1d: this script exists to reproduce the OLD scripts'
            # numbers, which predate the helium/particle-count correction,
            # so it pins the "nT" convention -- p_th = n_H*T and the old
            # n = P/T_HIM substitution. Do NOT switch this to "physical":
            # the old caches it compares against were built with nT, and
            # a 1.1x shift here would read as a regression that isn't one.
            variant = apply_variant(v, n_raw, Pth_raw, him, Pmin, Pmax,
                                    THERMAL_PRESSURE_CONVENTION_NT)
            n_model_cubes[v][jj] = variant.n_model
            Pth_model_cubes[v][jj] = variant.p_th_phys
        if jj % 50 == 0:
            print(f"  z-plane {jj + 1}/{n_z}")

    g_ext = gravity.g_ext_cgs(zpos)  # self-gravity OFF

    print("\n" + "=" * 100)
    print(f"{'variant':<8}{'quantity':<10}{'weighting':<10}{'stat':<10}"
          f"{'max_rel_diff':>16}{'median_rel_diff':>18}")
    print("=" * 100)

    results = []
    for v in VARIANTS:
        n_model = n_model_cubes[v]
        Pth_model = Pth_model_cubes[v]
        rho = MU * M_H * n_model
        Ptot_kB = hydrostatic.p_tot_kb_full_column(zpos, rho, g_ext)
        alpha_arr = derived.alpha(Ptot_kB, Pth_model)

        old_profiles = np.load(OLD_PROFILES_DIR / f"profiles_{v}.npz")
        old_sigma_mach = np.load(OLD_SIGMA_MACH_DIR / f"sigma_mach_{v}.npz")
        assert np.array_equal(old_profiles["zpos"], zpos), "z-grid mismatch vs old cache"

        new_Pth_vol_mean = np.full(n_z, np.nan)
        new_Pth_mw_mean = np.full(n_z, np.nan)
        new_Ptot_vol_mean = np.full(n_z, np.nan)
        new_Ptot_mw_mean = np.full(n_z, np.nan)
        new_alpha_vol_mean = np.full(n_z, np.nan)
        new_alpha_mw_mean = np.full(n_z, np.nan)
        new_alpha_vol_med = np.full(n_z, np.nan)

        for jj in range(n_z):
            neutral_in_R = (~him_cube[jj]) & cyl
            s_pth_vol = volume_weighted_stats(Pth_model[jj], neutral_in_R)
            s_pth_mw = mass_weighted_stats(Pth_model[jj], neutral_in_R, n_model[jj])
            s_ptot_vol = volume_weighted_stats(Ptot_kB[jj], neutral_in_R)
            s_ptot_mw = mass_weighted_stats(Ptot_kB[jj], neutral_in_R, n_model[jj])
            s_alpha_vol = volume_weighted_stats(alpha_arr[jj], neutral_in_R)
            s_alpha_mw = mass_weighted_stats(alpha_arr[jj], neutral_in_R, n_model[jj])

            new_Pth_vol_mean[jj] = s_pth_vol.arithmetic_mean
            new_Pth_mw_mean[jj] = s_pth_mw.arithmetic_mean
            new_Ptot_vol_mean[jj] = s_ptot_vol.arithmetic_mean
            new_Ptot_mw_mean[jj] = s_ptot_mw.arithmetic_mean
            new_alpha_vol_mean[jj] = s_alpha_vol.arithmetic_mean
            new_alpha_mw_mean[jj] = s_alpha_mw.arithmetic_mean
            new_alpha_vol_med[jj] = s_alpha_vol.median

        for qty, new_arr, old_key in [
            ("P_th", new_Pth_vol_mean, "Pth_vol_mean"),
            ("P_th", new_Pth_mw_mean, "Pth_mw_mean"),
            ("P_tot", new_Ptot_vol_mean, "Ptot_vol_mean"),
            ("P_tot", new_Ptot_mw_mean, "Ptot_mw_mean"),
            ("alpha", new_alpha_vol_mean, "alpha_vol_mean"),
            ("alpha", new_alpha_mw_mean, "alpha_mw_mean"),
            ("alpha", new_alpha_vol_med, "alpha_vol_med"),
        ]:
            weighting = "mw" if "mw" in old_key else "vol"
            stat = "median" if old_key.endswith("med") else "mean"
            max_d, med_d = rel_diff(new_arr, old_profiles[old_key])
            print(f"{v:<8}{qty:<10}{weighting:<10}{stat:<10}{max_d:16.6e}{med_d:18.6e}")
            results.append((v, qty, weighting, stat, max_d, med_d))

        # sigma_nt / Mach: use MY OWN alpha_vol_mean (this pipeline's
        # equivalent of what old fig4b reads from the upstream profiles
        # cache), phase classification via T_cube + him_cube, matching
        # fig4b_velocity_dispertion_Mach_number/compute_data.py's algorithm.
        phase = phase_flag(T_cube, him_cube)
        neutral = ~him_cube & cyl[None, :, :]
        new_mach = derived.mach_number(new_alpha_vol_mean)

        T_mean_total = np.full(n_z, np.nan)
        for jj in range(n_z):
            m = neutral[jj]
            if m.any():
                T_mean_total[jj] = float(T_cube[jj][m].mean())
        # The old scripts' "sigma_total" is the NON-THERMAL dispersion
        # sqrt(3(alpha-1))*c_s, which Step 1d renamed sigma_nt_kmps (the
        # name sigma_eff_kmps now means the total sqrt(alpha)*c_s). Same
        # formula, same convention, so the comparison is unchanged.
        new_sigma_total = derived.sigma_nt_kmps(new_alpha_vol_mean, T_mean_total,
                                                THERMAL_PRESSURE_CONVENTION_NT)

        max_d, med_d = rel_diff(new_mach, old_sigma_mach["mach"])
        print(f"{v:<8}{'Mach':<10}{'n/a':<10}{'value':<10}{max_d:16.6e}{med_d:18.6e}")
        results.append((v, "Mach", "n/a", "value", max_d, med_d))

        max_d, med_d = rel_diff(new_sigma_total, old_sigma_mach["sigma_total"])
        print(f"{v:<8}{'sigma_nt':<10}{'total':<10}{'value':<10}{max_d:16.6e}{med_d:18.6e}")
        results.append((v, "sigma_nt", "total", "value", max_d, med_d))

    print("=" * 100)
    return results


if __name__ == "__main__":
    main()
