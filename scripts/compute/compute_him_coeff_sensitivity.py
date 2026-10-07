"""
How much does the answer depend on the HIM-flag coefficient C?

The flag is p_nT = n_H T < C * P_min(I_UV), with C = 0.5 chosen because
at that value no CNM or UNM cell is supposed to be caught. C is a choice,
not a measurement, so this sweeps it over {0.1, 0.25, 0.5, 1.0} and
reports how far the answers move -- and checks the "no contamination"
claim rather than repeating it.

COMPUTE script: may import src.physics and read the 3D cache.

WHAT IS AND IS NOT RECOMPUTED
  * The FLAG is recomputed for each C, from p_nT and a P_min re-derived
    from the source cube's I_UV through the pipeline's own interpolator
    -- with the same reproduction check compute_raw_diagnostics does: at
    C = 0.5 the recomputed flag must equal the cached one exactly, or
    this script refuses to run.
  * P_tot is NOT recomputed. Under the fiducial treatment the flagged
    cells keep their observed mass in the hydrostatic weight whichever
    cells the flag picks out, so P_tot does not depend on C at all. Only
    which cells enter a STATISTIC does.

CONTAMINATION, and why the phase cubes in the cache cannot answer it
--------------------------------------------------------------------
"Contamination" is the fraction of the cells flagged at C that are
really CNM or UNM. The cached phase_flag_* cubes cannot be used: HIM
takes precedence in them, so every cell flagged at C = 0.5 is already
labelled HIM there and its neutral class has been overwritten.

So the neutral classification is recomputed WITHOUT the HIM override, by
calling the production classifiers (src.physics.him.phase_flag_dpdn and
phase_flag_temperature) with an all-False him argument. That is the same
code the pipeline uses, evaluated with the override switched off -- not a
reimplementation.
"""

import sys
import time
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config_loader import load_resolved_config  # noqa: E402
from src.conventions import (  # noqa: E402
    HIM_FLAG_COEFF,
    HIM_FLAG_COEFF_GRID,
    PERCENTILE_SCHEME_DEFAULT,
    PHASE_SCHEME_DPDN,
    PHASE_SCHEME_TEMPERATURE,
    PROFILE_BIN_PC,
    PROVISIONAL_CUBE_HEADER,
    STATS_BOX_XY_HALF_RANGE_PC,
    STATS_BOX_Z_HALF_RANGE_PC,
    THERMAL_PRESSURE_HEADER,
)
from src.physics import thermal  # noqa: E402
from src.physics.him import (  # noqa: E402
    PHASE_CNM,
    PHASE_UNM,
    phase_flag_dpdn,
    phase_flag_temperature,
)
from src.physics.loading import open_zarr  # noqa: E402
from src.physics.stats import (  # noqa: E402
    percentile_levels,
    ratio_of_means,
    weighted_stats,
)

ALPHA_CORE_ZARR_PATH = Path("cache/core/alpha_core.zarr")
OUT_NPZ_PATH = Path("cache/core/him_coeff_sensitivity.npz")
OUT_TXT_PATH = Path("results/him_coeff_sensitivity.txt")

VARIANT = "RAW"
SELF_GRAVITY = "mean"
Z_CHUNK = 48
REPORT_Z_PC = (0.0, 150.0, 300.0)
SCHEMES = (PHASE_SCHEME_DPDN, PHASE_SCHEME_TEMPERATURE)


def main():
    t0 = time.time()
    OUT_NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pct_lo, pct_hi = percentile_levels(PERCENTILE_SCHEME_DEFAULT)

    import pipeline.compute_all as pipe
    core = zarr.open_group(str(ALPHA_CORE_ZARR_PATH), mode="r")
    z_full = np.asarray(core["z_pc"][:], dtype=np.float64)
    zsl, z_pc = pipe._box_z_slice(z_full)
    groups = pipe.profile_bin_groups(z_pc, use_abs=False)
    n_bins = len(groups)

    vg = core[VARIANT]
    print("Loading RAW n_H, T, him, p_th_phys, P_tot and alpha over the STATS_BOX...")
    n_H = np.asarray(vg["n_model"][zsl], dtype=np.float32)
    T = np.asarray(vg["T"][zsl], dtype=np.float32)
    him_cached = np.asarray(vg["him"][zsl]).astype(bool)
    p_th = np.asarray(vg["p_th_phys"][zsl], dtype=np.float32)
    p_tot = np.asarray(vg[f"self_gravity_{SELF_GRAVITY}"]["Ptot"][zsl], dtype=np.float32)
    alpha = np.asarray(vg[f"self_gravity_{SELF_GRAVITY}"]["alpha"][zsl], dtype=np.float32)
    shape = n_H.shape
    print(f"  box {shape}; cached flagged fraction {him_cached.mean():.4f}")

    # ---- P_min, the dPdn boundaries, and p_nT -----------------------------
    print("Re-deriving P_min(I_UV) and the dPdn density boundaries...")
    import dask.array as da
    cfg = load_resolved_config()
    grid = open_zarr()
    x_lo = int(np.where(np.abs(grid.x_pc) <= STATS_BOX_XY_HALF_RANGE_PC)[0].min())
    x_hi = int(np.where(np.abs(grid.x_pc) <= STATS_BOX_XY_HALF_RANGE_PC)[0].max()) + 1
    y_lo = int(np.where(np.abs(grid.y_pc) <= STATS_BOX_XY_HALF_RANGE_PC)[0].min())
    y_hi = int(np.where(np.abs(grid.y_pc) <= STATS_BOX_XY_HALF_RANGE_PC)[0].max()) + 1
    Iuv = da.from_zarr(grid.store["Iuv_final"])[zsl, y_lo:y_hi, x_lo:x_hi].compute().astype(np.float32)
    pminmax = thermal.build_pmin_pmax(cfg["bs19_mat_path"])
    Pmin = pminmax.p_min(Iuv).astype(np.float32)
    bounds = thermal.build_phase_density_bounds(cfg["bs19_mat_path"])

    # p_nT, formed exactly as the pipeline forms it (float64 then rounded
    # to float32) so the C = 0.5 reproduction check is meaningful.
    p_nT = np.empty(shape, dtype=np.float32)
    # Neutral phase labels WITHOUT the HIM override -- see module docstring.
    phase_neutral = {s: np.empty(shape, dtype=np.int8) for s in SCHEMES}
    for lo in range(0, shape[0], Z_CHUNK):
        hi = min(lo + Z_CHUNK, shape[0])
        sl = slice(lo, hi)
        p_nT[sl] = (n_H[sl].astype(np.float64) * T[sl].astype(np.float64)).astype(np.float32)
        no_him = np.zeros(n_H[sl].shape, dtype=bool)
        nw = bounds.n_w_max(Iuv[sl]).astype(np.float32)
        nc = bounds.n_c_min(Iuv[sl]).astype(np.float32)
        phase_neutral[PHASE_SCHEME_DPDN][sl] = phase_flag_dpdn(n_H[sl], nw, nc, no_him)
        phase_neutral[PHASE_SCHEME_TEMPERATURE][sl] = phase_flag_temperature(T[sl], no_him)
    del Iuv, nw, nc
    print("  p_nT and the override-free phase labels built")

    # ---- reproduction check at the fiducial C ----------------------------
    flag_fid = (p_nT < HIM_FLAG_COEFF * Pmin) & np.isfinite(Pmin)
    agree = float((flag_fid == him_cached).mean())
    print(f"  recomputed flag at C = {HIM_FLAG_COEFF} agrees with the cached one on "
          f"{agree:.8%} of the box")
    if agree < 1.0:
        raise RuntimeError(
            f"At C = {HIM_FLAG_COEFF} the recomputed flag does not reproduce the cached "
            "one; every number below would be built on a different flag than the rest "
            "of the project.")
    del flag_fid

    # ---- the EXACT coefficient at which contamination starts --------------
    # Contamination appears as soon as one CNM/UNM cell satisfies
    # p_nT < C P_min, i.e. as soon as C exceeds min(p_nT / P_min) over the
    # CNM/UNM cells. That minimum IS the critical coefficient, so it is
    # computed directly instead of being bracketed on the four-point grid
    # -- which would only ever say "somewhere between 0.5 and 1".
    print("Finding the exact C at which CNM/UNM contamination starts...")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.isfinite(Pmin) & (Pmin > 0), p_nT / Pmin, np.inf)
    C_crit = {}
    for s_name in SCHEMES:
        ph = phase_neutral[s_name]
        cnm_unm = (ph == PHASE_CNM) | (ph == PHASE_UNM)
        vals = ratio[cnm_unm]
        vals = vals[np.isfinite(vals)]
        C_crit[s_name] = float(vals.min()) if vals.size else float("nan")
        print(f"  {s_name:12s}: first CNM/UNM cell is flagged at C = {C_crit[s_name]:.4f}")
    del ratio

    # ---- sweep ------------------------------------------------------------
    C_grid = np.array(HIM_FLAG_COEFF_GRID, dtype=float)
    est = ("alpha_median_of_ratios", "alpha_mean_of_ratios", "alpha_ratio_of_means",
           "alpha_p15_of_ratios", "alpha_p85_of_ratios")
    prof = {f"{e}__{w}": np.full((len(C_grid), n_bins), np.nan)
            for e in est for w in ("vol", "mw")}
    for k in ("flag_frac_vol", "flag_frac_mw"):
        prof[k] = np.full((len(C_grid), n_bins), np.nan)
    for s in SCHEMES:
        for w in ("vol", "mw"):
            prof[f"contam_{s}__{w}"] = np.full((len(C_grid), n_bins), np.nan)

    box = {f"{e}__{w}": np.full(len(C_grid), np.nan) for e in est for w in ("vol", "mw")}
    for k in ("flag_frac_vol", "flag_frac_mw"):
        box[k] = np.full(len(C_grid), np.nan)
    for s in SCHEMES:
        for w in ("vol", "mw"):
            box[f"contam_{s}__{w}"] = np.full(len(C_grid), np.nan)

    n_w_all = np.where(np.isfinite(n_H), n_H, 0.0).astype(np.float64)

    for ci, C in enumerate(C_grid):
        flag = (p_nT < C * Pmin) & np.isfinite(Pmin)
        neutral = ~flag
        print(f"  C = {C:<5}: flagged {flag.mean():.4%} by volume")

        # --- box level
        mass_tot = n_w_all.sum()
        box["flag_frac_vol"][ci] = float(flag.mean())
        box["flag_frac_mw"][ci] = float(n_w_all[flag].sum() / mass_tot)
        for s in SCHEMES:
            ph = phase_neutral[s]
            cnm_unm = (ph == PHASE_CNM) | (ph == PHASE_UNM)
            bad = flag & cnm_unm
            n_flag = int(flag.sum())
            box[f"contam_{s}__vol"][ci] = (bad.sum() / n_flag) if n_flag else np.nan
            m_flag = n_w_all[flag].sum()
            box[f"contam_{s}__mw"][ci] = (n_w_all[bad].sum() / m_flag) if m_flag > 0 else np.nan

        a_v = alpha[neutral].astype(np.float64)
        pt_v = p_tot[neutral].astype(np.float64)
        pth_v = p_th[neutral].astype(np.float64)
        n_v = n_H[neutral].astype(np.float64)
        for w_name, w in (("vol", np.ones_like(a_v)),
                          ("mw", np.where(np.isfinite(n_v), n_v, 0.0))):
            st = weighted_stats(a_v, w, PERCENTILE_SCHEME_DEFAULT)
            box[f"alpha_median_of_ratios__{w_name}"][ci] = st.median
            box[f"alpha_mean_of_ratios__{w_name}"][ci] = st.arithmetic_mean
            box[f"alpha_p15_of_ratios__{w_name}"][ci] = st.p_lo
            box[f"alpha_p85_of_ratios__{w_name}"][ci] = st.p_hi
            box[f"alpha_ratio_of_means__{w_name}"][ci] = ratio_of_means(pt_v, pth_v, w)
        del a_v, pt_v, pth_v, n_v

        # --- per bin
        for b, (_, idxs) in enumerate(groups):
            if len(idxs) == 0:
                continue
            fl = flag[idxs]
            nb = n_H[idxs].astype(np.float64)
            wb = np.where(np.isfinite(nb), nb, 0.0)
            prof["flag_frac_vol"][ci, b] = float(fl.mean())
            mtot = wb.sum()
            prof["flag_frac_mw"][ci, b] = float(wb[fl].sum() / mtot) if mtot > 0 else np.nan
            for s in SCHEMES:
                ph = phase_neutral[s][idxs]
                bad = fl & ((ph == PHASE_CNM) | (ph == PHASE_UNM))
                nf = int(fl.sum())
                prof[f"contam_{s}__vol"][ci, b] = (bad.sum() / nf) if nf else np.nan
                mf = wb[fl].sum()
                prof[f"contam_{s}__mw"][ci, b] = (wb[bad].sum() / mf) if mf > 0 else np.nan

            nt = ~fl
            ab = alpha[idxs][nt].astype(np.float64)
            ptb = p_tot[idxs][nt].astype(np.float64)
            pthb = p_th[idxs][nt].astype(np.float64)
            nnb = nb[nt]
            for w_name, w in (("vol", np.ones_like(ab)),
                              ("mw", np.where(np.isfinite(nnb), nnb, 0.0))):
                st = weighted_stats(ab, w, PERCENTILE_SCHEME_DEFAULT)
                prof[f"alpha_median_of_ratios__{w_name}"][ci, b] = st.median
                prof[f"alpha_mean_of_ratios__{w_name}"][ci, b] = st.arithmetic_mean
                prof[f"alpha_p15_of_ratios__{w_name}"][ci, b] = st.p_lo
                prof[f"alpha_p85_of_ratios__{w_name}"][ci, b] = st.p_hi
                prof[f"alpha_ratio_of_means__{w_name}"][ci, b] = ratio_of_means(ptb, pthb, w)
        del flag, neutral

    # ---- save -------------------------------------------------------------
    out = dict(prof)
    out.update({f"box_{k}": v for k, v in box.items()})
    out.update({
        "C_grid": C_grid,
        "C_fiducial": np.array([HIM_FLAG_COEFF]),
        "z_pc": np.array([g[0] for g in groups], dtype=float),
        "bin_width_pc": np.array([PROFILE_BIN_PC], dtype=float),
        "percentile_levels": np.array([pct_lo, pct_hi], dtype=float),
        "schemes": np.array(list(SCHEMES)),
        "variant": np.array([VARIANT]),
        "self_gravity": np.array([SELF_GRAVITY]),
        "provisional": np.array([PROVISIONAL_CUBE_HEADER]),
        "flag_reproduction_agreement": np.array([agree]),
        "C_crit_dPdn": np.array([C_crit[PHASE_SCHEME_DPDN]]),
        "C_crit_temperature": np.array([C_crit[PHASE_SCHEME_TEMPERATURE]]),
    })
    np.savez_compressed(OUT_NPZ_PATH, **out)
    print(f"Saved {OUT_NPZ_PATH}")

    # ---- report -----------------------------------------------------------
    z_arr = out["z_pc"]

    def at(zt):
        return int(np.argmin(np.abs(z_arr - zt)))

    L = [
        PROVISIONAL_CUBE_HEADER,
        THERMAL_PRESSURE_HEADER,
        "Sensitivity to the HIM-flag coefficient C:  p_nT = n_H T  <  C * P_min(I_UV)",
        "=" * 104,
        f"Variant {VARIANT}, self-gravity {SELF_GRAVITY}, STATS_BOX "
        f"|x|,|y| <= {STATS_BOX_XY_HALF_RANGE_PC:.0f} pc, |z| <= "
        f"{STATS_BOX_Z_HALF_RANGE_PC:.0f} pc, {PROFILE_BIN_PC:.0f} pc bins.",
        f"Fiducial C = {HIM_FLAG_COEFF}. The recomputed flag reproduces the cached one at that",
        f"  value on {agree:.6%} of the box, so this sweep sits on the same flag as the",
        "  rest of the project.",
        "",
        "P_tot is NOT recomputed per C: under the fiducial treatment the flagged cells keep",
        "  their observed mass in the hydrostatic weight whichever cells the flag picks out,",
        "  so only which cells enter a STATISTIC depends on C.",
        "",
        "CONTAMINATION = of the cells flagged at C, the fraction that the neutral classifier",
        "  calls CNM or UNM. Computed with the HIM override switched OFF, because the cached",
        "  phase cubes have already relabelled every flagged cell as HIM.",
        "",
        "Box-level:",
        f"{'C':>6}{'flag vol':>11}{'flag mass':>11}"
        f"{'contam dPdn vol':>17}{'contam dPdn mw':>16}"
        f"{'contam T vol':>14}{'contam T mw':>13}",
        "-" * 88,
    ]
    for ci, C in enumerate(C_grid):
        L.append(
            f"{C:>6.2f}{box['flag_frac_vol'][ci]:>11.5f}{box['flag_frac_mw'][ci]:>11.5f}"
            f"{box[f'contam_{PHASE_SCHEME_DPDN}__vol'][ci]:>17.6f}"
            f"{box[f'contam_{PHASE_SCHEME_DPDN}__mw'][ci]:>16.6f}"
            f"{box[f'contam_{PHASE_SCHEME_TEMPERATURE}__vol'][ci]:>14.6f}"
            f"{box[f'contam_{PHASE_SCHEME_TEMPERATURE}__mw'][ci]:>13.6f}")
    L += [
        "-" * 88,
        "",
        "Exact onset of contamination (min of p_nT / P_min over the CNM/UNM cells -- the",
        "  smallest C at which ANY of them is flagged, not a grid bracket):",
        f"  dPdn scheme:        C = {C_crit[PHASE_SCHEME_DPDN]:.4f}",
        f"  temperature scheme: C = {C_crit[PHASE_SCHEME_TEMPERATURE]:.4f}",
        f"  So C = {HIM_FLAG_COEFF} has a factor "
        f"{C_crit[PHASE_SCHEME_DPDN] / HIM_FLAG_COEFF:.2f} of headroom before the first",
        "  CNM/UNM cell is caught.",
        "",
        f"Box-level alpha over NON-flagged cells ({pct_lo:g}/{pct_hi:g} in brackets):",
        f"{'C':>6}{'wt':>5}{'median':>10}{'mean':>10}{'rat.means':>11}"
        f"{'p' + format(pct_lo, 'g'):>9}{'p' + format(pct_hi, 'g'):>9}",
        "-" * 60,
    ]
    for ci, C in enumerate(C_grid):
        for w in ("vol", "mw"):
            L.append(
                f"{C:>6.2f}{w:>5}"
                f"{box[f'alpha_median_of_ratios__{w}'][ci]:>10.4f}"
                f"{box[f'alpha_mean_of_ratios__{w}'][ci]:>10.4f}"
                f"{box[f'alpha_ratio_of_means__{w}'][ci]:>11.4f}"
                f"{box[f'alpha_p15_of_ratios__{w}'][ci]:>9.4f}"
                f"{box[f'alpha_p85_of_ratios__{w}'][ci]:>9.4f}")
    L += ["-" * 60, ""]

    L += [f"Mass-weighted alpha at z = 0 / 150 / 300 pc:",
          f"{'C':>6}{'estimator':>24}" + "".join(f"{'z=' + format(zt, '.0f'):>12}"
                                                   for zt in REPORT_Z_PC),
          "-" * 66]
    for ci, C in enumerate(C_grid):
        for e, lab in (("alpha_median_of_ratios", "median of ratios"),
                       ("alpha_ratio_of_means", "ratio of means")):
            L.append(f"{C:>6.2f}{lab:>24}" + "".join(
                f"{prof[f'{e}__mw'][ci, at(zt)]:>12.4f}" for zt in REPORT_Z_PC))
    L += [
        "-" * 66,
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
