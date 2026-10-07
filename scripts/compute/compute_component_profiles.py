"""
Component profiles for the simulation comparison (Alon Gurman, GHOSDT /
Gurman et al. 2025): P_th of the neutral gas, P_tot, alpha and n_H
against height, in the form a simulator can put next to their own runs.

COMPUTE script: may import src.physics and read the 3D cache. The figure
that consumes it reads only the .npz written here.

FIDUCIAL TREATMENT, unchanged from the rest of the project:
  * RAW -- observed density, no HIM_A/HIM_B substitution;
  * HIM-flagged cells EXCLUDED from every statistic;
  * their observed mass KEPT in the hydrostatic weight, so P_tot already
    includes it. Nothing here recomputes P_tot; it is read from the
    cache. This step only selects cells and reduces them.
  * self-gravity = mean (footprint-averaged), the fiducial setting.

n_H is reported BOTH ways -- over all cells and over non-HIM cells only
-- because it is the one quantity where the exclusion changes what the
number means rather than just its value. "Mean density of the box" and
"mean density of the neutral gas" are different physical questions, and
a simulator comparing against a two-phase average wants the second.

The HIM_A/HIM_B pressures (1.1 P_min, 1.1 P_max) are stored as an
ASSUMPTION, not a measurement: they are what those variants would impose
on the flagged cells, so the figure can show the range the HIM treatment
spans without pretending it was observed.
"""

import sys
import time
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config_loader import load_resolved_config  # noqa: E402
from src.conventions import (  # noqa: E402
    PARTICLES_PER_H_NEUTRAL,
    PERCENTILE_SCHEME_DEFAULT,
    PROFILE_BIN_PC,
    PROVISIONAL_CUBE_HEADER,
    STATS_BOX_XY_HALF_RANGE_PC,
    STATS_BOX_Z_HALF_RANGE_PC,
    THERMAL_PRESSURE_HEADER,
)
from src.physics import thermal  # noqa: E402
from src.physics.stats import (  # noqa: E402
    percentile_levels,
    ratio_of_means,
    weighted_stats,
)

ALPHA_CORE_ZARR_PATH = Path("cache/core/alpha_core.zarr")
OUT_NPZ_PATH = Path("cache/core/component_profiles.npz")
OUT_TXT_PATH = Path("results/component_profiles.txt")
REFERENCE_CSV = Path("results/reference/ok22_tigress_R8.csv")

VARIANT = "RAW"
SELF_GRAVITY = "mean"
REPORT_Z_PC = (-300.0, -150.0, 0.0, 150.0, 300.0)


def load_reference(path=REFERENCE_CSV):
    """The TIGRESS R8 values, as {quantity: (value, units)}."""
    import csv
    with open(path, newline="") as f:
        rows = list(csv.DictReader([ln for ln in f if not ln.startswith("#")]))
    return {r["quantity"]: (float(r["value"]), r["units"]) for r in rows}


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
    z_full = np.asarray(core["z_pc"][:], dtype=np.float64)
    zsl, z_pc = pipe._box_z_slice(z_full)
    footprint = np.ones((len(y_pc), len(x_pc)), dtype=bool)  # already the +-500 square
    groups = pipe.profile_bin_groups(z_pc, use_abs=False)
    n_bins = len(groups)

    vg = core[VARIANT]
    print("Loading RAW n_H, T, him, p_th_phys, P_tot and alpha over the STATS_BOX...")
    n_H = np.asarray(vg["n_model"][zsl], dtype=np.float32)
    T = np.asarray(vg["T"][zsl], dtype=np.float32)
    him = np.asarray(vg["him"][zsl]).astype(bool)
    p_th = np.asarray(vg["p_th_phys"][zsl], dtype=np.float32)
    p_tot = np.asarray(vg[f"self_gravity_{SELF_GRAVITY}"]["Ptot"][zsl], dtype=np.float32)
    alpha = np.asarray(vg[f"self_gravity_{SELF_GRAVITY}"]["alpha"][zsl], dtype=np.float32)
    print(f"  box {n_H.shape}; flagged fraction {him.mean():.4f}")

    # ---- P_min / P_max for the HIM_A/HIM_B assumed pressures --------------
    # Re-derived from I_UV through the pipeline's own interpolator, and the
    # flag it implies is checked against the cached one, exactly as
    # compute_raw_diagnostics does -- otherwise the assumed band would be
    # drawn from a different P_min than the flag was built with.
    print("Re-deriving P_min/P_max(I_UV) and checking the flag it implies...")
    import dask.array as da
    from src.conventions import HIM_FLAG_COEFF
    from src.physics.loading import open_zarr
    cfg = load_resolved_config()
    grid = open_zarr()
    x_lo = int(np.where(np.abs(grid.x_pc) <= STATS_BOX_XY_HALF_RANGE_PC)[0].min())
    x_hi = int(np.where(np.abs(grid.x_pc) <= STATS_BOX_XY_HALF_RANGE_PC)[0].max()) + 1
    y_lo = int(np.where(np.abs(grid.y_pc) <= STATS_BOX_XY_HALF_RANGE_PC)[0].min())
    y_hi = int(np.where(np.abs(grid.y_pc) <= STATS_BOX_XY_HALF_RANGE_PC)[0].max()) + 1
    Iuv = da.from_zarr(grid.store["Iuv_final"])[zsl, y_lo:y_hi, x_lo:x_hi].compute()
    pminmax = thermal.build_pmin_pmax(cfg["bs19_mat_path"])
    Pmin = pminmax.p_min(Iuv.astype(np.float32)).astype(np.float32)
    Pmax = pminmax.p_max(Iuv.astype(np.float32)).astype(np.float32)
    del Iuv

    pnT_chk = (n_H[:4].astype(np.float64) * T[:4].astype(np.float64)).astype(np.float32)
    him_chk = (pnT_chk < HIM_FLAG_COEFF * Pmin[:4]) & np.isfinite(Pmin[:4])
    agree = float((him_chk == him[:4]).mean())
    print(f"  re-derived flag agrees with the cached one on {agree:.6%} of a 4-plane sample")
    if agree < 1.0:
        raise RuntimeError(
            "Re-derived P_min does not reproduce the cached HIM flag -- the assumed "
            "HIM_A/HIM_B band would not be the one those variants actually impose.")

    # ---- per-bin reduction ------------------------------------------------
    keys = [
        "Pth_neutral_mw_mean", "Pth_neutral_mw_median", "Pth_neutral_mw_p15",
        "Pth_neutral_mw_p85", "Pth_neutral_vol_mean",
        "Ptot_mw_mean", "Ptot_mw_median", "Ptot_mw_p15", "Ptot_mw_p85",
        "Ptot_vol_mean",
        "alpha_ratio_of_means_vol", "alpha_ratio_of_means_mw",
        "alpha_median_of_ratios_mw", "alpha_p15_of_ratios_mw", "alpha_p85_of_ratios_mw",
        "nH_vol_mean_all", "nH_vol_median_all", "nH_vol_p15_all", "nH_vol_p85_all",
        "nH_mw_mean_all",
        "nH_vol_mean_neutral", "nH_vol_median_neutral", "nH_vol_p15_neutral",
        "nH_vol_p85_neutral", "nH_mw_mean_neutral",
        "him_assumed_Pth_A", "him_assumed_Pth_B", "him_frac_vol", "him_frac_mw",
    ]
    out = {k: np.full(n_bins, np.nan) for k in keys}
    out["n_cells"] = np.zeros(n_bins, dtype=np.int64)
    out["n_neutral"] = np.zeros(n_bins, dtype=np.int64)

    for b, (_, idxs) in enumerate(groups):
        if len(idxs) == 0:
            continue
        him_b = him[idxs][:, footprint]
        n_b = n_H[idxs][:, footprint].astype(np.float64)
        pth_b = p_th[idxs][:, footprint].astype(np.float64)
        pt_b = p_tot[idxs][:, footprint].astype(np.float64)
        a_b = alpha[idxs][:, footprint].astype(np.float64)
        neutral = ~him_b

        out["n_cells"][b] = him_b.size
        out["n_neutral"][b] = int(neutral.sum())
        w_all_vol = np.ones(n_b.shape)
        w_all_mw = np.where(np.isfinite(n_b), n_b, 0.0)
        w_vol = np.where(neutral, 1.0, 0.0)
        w_mw = np.where(neutral & np.isfinite(n_b), n_b, 0.0)

        out["him_frac_vol"][b] = float(him_b.mean())
        tot_mass = w_all_mw.sum()
        out["him_frac_mw"][b] = float(w_all_mw[him_b].sum() / tot_mass) if tot_mass else np.nan

        # --- pressures over NON-HIM cells
        for name, arr in (("Pth_neutral", pth_b), ("Ptot", pt_b)):
            st = weighted_stats(arr, w_mw, PERCENTILE_SCHEME_DEFAULT)
            out[f"{name}_mw_mean"][b] = st.arithmetic_mean
            out[f"{name}_mw_median"][b] = st.median
            out[f"{name}_mw_p15"][b] = st.p_lo
            out[f"{name}_mw_p85"][b] = st.p_hi
            out[f"{name}_vol_mean"][b] = weighted_stats(
                arr, w_vol, PERCENTILE_SCHEME_DEFAULT).arithmetic_mean

        # --- alpha over the same cells
        out["alpha_ratio_of_means_vol"][b] = ratio_of_means(pt_b, pth_b, w_vol)
        out["alpha_ratio_of_means_mw"][b] = ratio_of_means(pt_b, pth_b, w_mw)
        st_a = weighted_stats(a_b, w_mw, PERCENTILE_SCHEME_DEFAULT)
        out["alpha_median_of_ratios_mw"][b] = st_a.median
        out["alpha_p15_of_ratios_mw"][b] = st_a.p_lo
        out["alpha_p85_of_ratios_mw"][b] = st_a.p_hi

        # --- n_H, all cells and non-HIM cells
        for tag, wv, wm in (("all", w_all_vol, w_all_mw), ("neutral", w_vol, w_mw)):
            st_n = weighted_stats(n_b, wv, PERCENTILE_SCHEME_DEFAULT)
            out[f"nH_vol_mean_{tag}"][b] = st_n.arithmetic_mean
            out[f"nH_vol_median_{tag}"][b] = st_n.median
            out[f"nH_vol_p15_{tag}"][b] = st_n.p_lo
            out[f"nH_vol_p85_{tag}"][b] = st_n.p_hi
            out[f"nH_mw_mean_{tag}"][b] = weighted_stats(
                n_b, wm, PERCENTILE_SCHEME_DEFAULT).arithmetic_mean

        # --- the HIM_A/HIM_B ASSUMED pressures, median over flagged cells
        if him_b.any():
            pmin_b = Pmin[idxs][:, footprint].astype(np.float64)[him_b]
            pmax_b = Pmax[idxs][:, footprint].astype(np.float64)[him_b]
            out["him_assumed_Pth_A"][b] = float(np.median(
                PARTICLES_PER_H_NEUTRAL * pmin_b))
            out["him_assumed_Pth_B"][b] = float(np.median(
                PARTICLES_PER_H_NEUTRAL * pmax_b))

        if b % 50 == 0:
            print(f"  bin {b + 1}/{n_bins} (z = {groups[b][0]:+.0f} pc)")

    out["z_pc"] = np.array([g[0] for g in groups], dtype=float)
    out["bin_edges"] = pipe.profile_bin_edges(use_abs=False)
    out["bin_width_pc"] = np.array([PROFILE_BIN_PC], dtype=float)
    out["percentile_levels"] = np.array([pct_lo, pct_hi], dtype=float)
    out["variant"] = np.array([VARIANT])
    out["self_gravity"] = np.array([SELF_GRAVITY])
    out["provisional"] = np.array([PROVISIONAL_CUBE_HEADER])
    for q, (v, u) in ref.items():
        out[f"ref_{q}"] = np.array([v])
    np.savez_compressed(OUT_NPZ_PATH, **out)
    print(f"Saved {OUT_NPZ_PATH}")

    # ---- report ----------------------------------------------------------
    def at(z_target):
        return int(np.argmin(np.abs(out["z_pc"] - z_target)))

    L = [
        PROVISIONAL_CUBE_HEADER,
        THERMAL_PRESSURE_HEADER,
        "Component profiles for the simulation comparison (GHOSDT / Gurman et al. 2025)",
        "=" * 104,
        f"Fiducial treatment: {VARIANT} (observed density, no HIM_A/HIM_B substitution);",
        "  HIM-flagged cells EXCLUDED from every statistic; their observed mass KEPT in the",
        f"  hydrostatic weight, so P_tot (self-gravity {SELF_GRAVITY}) already includes it and is",
        "  read from the cache unchanged. STATS_BOX |x|,|y| <= "
        f"{STATS_BOX_XY_HALF_RANGE_PC:.0f} pc, |z| <= {STATS_BOX_Z_HALF_RANGE_PC:.0f} pc, "
        f"{PROFILE_BIN_PC:.0f} pc bins.",
        "",
        "COMPARABILITY: TIGRESS R8 has Sigma_gas = "
        f"{ref['Sigma_gas'][0]:.2f} Msun/pc^2, about twice this cube's "
        "4.81 Msun/pc^2.",
        "  In a self-regulated disc the absolute pressures scale roughly as Sigma_gas^2, so they",
        "  are NOT expected to match. alpha = P_tot/P_th is a ratio and CAN be compared.",
        "",
        f"{'z [pc]':>8}{'Pth,neut mw':>14}{'Ptot mw':>12}{'alpha RoM vol':>15}"
        f"{'alpha RoM mw':>14}{'alpha med mw':>14}{'nH vol mean':>13}{'nH mw mean':>12}",
        f"{'':>8}{'[K cm^-3]':>14}{'[K cm^-3]':>12}{'':>15}{'':>14}{'':>14}"
        f"{'[cm^-3]':>13}{'[cm^-3]':>12}",
        "-" * 104,
    ]
    for z_t in REPORT_Z_PC:
        i = at(z_t)
        L.append(
            f"{out['z_pc'][i]:>8.0f}{out['Pth_neutral_mw_mean'][i]:>14.1f}"
            f"{out['Ptot_mw_mean'][i]:>12.1f}{out['alpha_ratio_of_means_vol'][i]:>15.3f}"
            f"{out['alpha_ratio_of_means_mw'][i]:>14.3f}"
            f"{out['alpha_median_of_ratios_mw'][i]:>14.3f}"
            f"{out['nH_vol_mean_all'][i]:>13.4f}{out['nH_mw_mean_all'][i]:>12.4f}")
    L += [
        "-" * 104,
        "  (pressures and alpha over NON-HIM cells; nH columns over ALL cells -- the non-HIM",
        f"   nH is in the cache as nH_*_neutral and is listed below)",
        "",
        f"{'z [pc]':>8}{'nH vol mean':>14}{'nH vol med':>13}{'nH mw mean':>13}"
        f"{'HIM frac vol':>14}{'HIM frac mass':>15}",
        f"{'':>8}{'non-HIM':>14}{'non-HIM':>13}{'non-HIM':>13}{'':>14}{'':>15}",
        "-" * 77,
    ]
    for z_t in REPORT_Z_PC:
        i = at(z_t)
        L.append(
            f"{out['z_pc'][i]:>8.0f}{out['nH_vol_mean_neutral'][i]:>14.4f}"
            f"{out['nH_vol_median_neutral'][i]:>13.4f}"
            f"{out['nH_mw_mean_neutral'][i]:>13.4f}"
            f"{out['him_frac_vol'][i]:>14.4f}{out['him_frac_mw'][i]:>15.4f}")
    L += [
        "-" * 77,
        "",
        "TIGRESS R8 midplane reference (Ostriker & Kim 2022, Table 2; 2p gas, T < 2e4 K):",
        f"  P_tot_2p   {ref['P_tot_2p'][0]:>10.4g} K cm^-3      "
        f"P_th_2p    {ref['P_th_2p'][0]:>10.4g} K cm^-3",
        f"  P_turb_2p  {ref['P_turb_2p'][0]:>10.4g} K cm^-3      "
        f"Pi_mag_2p  {ref['Pi_mag_2p'][0]:>10.4g} K cm^-3",
        f"  P_tot_hot  {ref['P_tot_hot'][0]:>10.4g} K cm^-3      "
        f"P_DE       {ref['P_DE'][0]:>10.4g} K cm^-3",
        f"  n_H_2p     {ref['n_H_2p'][0]:>10.4g} cm^-3        "
        f"sigma_eff  {ref['sigma_eff_2p'][0]:>10.4g} km/s",
        f"  Sigma_gas  {ref['Sigma_gas'][0]:>10.4g} Msun/pc^2    "
        f"H_2p       {ref['H_2p'][0]:>10.4g} pc",
        f"  alpha_2p   {ref['alpha_2p'][0]:>10.4g}              (= P_tot_2p / P_th_2p)",
        "",
        "ASSUMED HIM pressures (NOT measured): what HIM_A and HIM_B would impose on the",
        "flagged cells, as 1.1 P_min and 1.1 P_max, median over the flagged cells in each bin.",
        f"  at z = 0:   HIM_A {out['him_assumed_Pth_A'][at(0.0)]:.1f}, "
        f"HIM_B {out['him_assumed_Pth_B'][at(0.0)]:.1f} K cm^-3",
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
