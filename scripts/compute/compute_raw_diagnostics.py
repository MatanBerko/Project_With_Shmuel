"""
RAW diagnostics for the HIM flag, binned in z -> cache/core/raw_diagnostics.npz.

This is a COMPUTE script, so it is allowed to import src.physics and to
read the source cube. The figure script that consumes this
(scripts/design/fig_raw_diagnostics.py) reads only the .npz it writes --
that split is the rule for this project: design scripts never compute
physics.

What goes in the cache, all per 4 pc signed-z profile bin inside the
STATS_BOX, over the +-500 pc square footprint, RAW only:

  him_frac_vol / him_frac_mw
      Fraction of cells (by volume) and of observed gas mass that the
      HIM flag catches. The flag is p_nT < 0.5 P_min(I_UV) and is
      variant-independent, so "RAW only" costs nothing.

  n_median / n_p15 / n_p85  and  T_median / T_p15 / T_p85,
  each for "flagged" and "neutral" separately
      Volume-weighted. Kept separate rather than pooled because the
      whole question the diagnostics figure asks is how the two
      populations differ.

  pnT_median / pnT_p15 / pnT_p85  and  half_pmin_median
      p_nT = n_H * T, the BS19 CLASSIFICATION convention -- NOT the
      physical p_th_phys = 1.1 n_H T that alpha uses. The flag compares
      p_nT against 0.5 P_min, so these two curves are what actually
      decides the flag, and plotting them against each other shows
      directly where the gas drops below the threshold.
      Taken over ALL cells in the bin (flagged and neutral together),
      volume-weighted: the point is to see the whole population cross
      the threshold, so conditioning on the flag would be circular.

  frac_alpha_lt1_vol / _mw
      Fraction of NON-HIM cells with alpha < 1, self-gravity = mean.
      This is exactly the fraction for which Mach and sigma_nt are
      undefined (see results/README.md).

P_min is not in alpha_core.zarr, so it is rebuilt here from the source
cube's I_UV through the same src.physics.thermal.build_pmin_pmax the
pipeline uses -- not re-derived independently, so the threshold drawn in
the figure is the threshold the flag was actually computed with.

The 4 pc bin edges come from pipeline.compute_all.profile_bin_groups, so
they are the SAME edges as the pipeline's profiles by construction
rather than by a matching constant that could drift.
"""

import sys
import time
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config_loader import load_resolved_config  # noqa: E402
from src.conventions import (  # noqa: E402
    HIM_THRESHOLD_FACTOR,
    PROFILE_BIN_PC,
    PROVISIONAL_CUBE_HEADER,
    STATS_BOX_XY_HALF_RANGE_PC,
    STATS_BOX_Z_HALF_RANGE_PC,
)
from src.physics import thermal  # noqa: E402
from src.physics.loading import footprint_mask, open_zarr  # noqa: E402
from src.physics.stats import weighted_stats  # noqa: E402

ALPHA_CORE_ZARR_PATH = Path("cache/core/alpha_core.zarr")
OUT_NPZ_PATH = Path("cache/core/raw_diagnostics.npz")

VARIANT = "RAW"          # observed density, no HIM substitution
SELF_GRAVITY = "mean"    # the fiducial setting, for the alpha < 1 fraction


def main():
    t0 = time.time()
    OUT_NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ---- the pipeline's own bin geometry, imported rather than copied
    import pipeline.compute_all as pipe

    core = zarr.open_group(str(ALPHA_CORE_ZARR_PATH), mode="r")
    x_pc = np.asarray(core["x_pc"][:], dtype=np.float64)
    y_pc = np.asarray(core["y_pc"][:], dtype=np.float64)
    z_full = np.asarray(core["z_pc"][:], dtype=np.float64)
    zsl, z_pc = pipe._box_z_slice(z_full)
    footprint = footprint_mask(x_pc, y_pc, STATS_BOX_XY_HALF_RANGE_PC)
    groups = pipe.profile_bin_groups(z_pc, use_abs=False)
    n_bins = len(groups)
    print(f"STATS_BOX: {len(z_pc)} z planes, |z| <= {STATS_BOX_Z_HALF_RANGE_PC:.0f} pc; "
          f"{n_bins} bins of {PROFILE_BIN_PC:g} pc; {int(footprint.sum())} XY cells/plane")

    vg = core[VARIANT]
    print("Loading RAW n_H, T, him and alpha (STATS_BOX only)...")
    n_H = np.asarray(vg["n_model"][zsl], dtype=np.float32)   # RAW == observed n_H
    T = np.asarray(vg["T"][zsl], dtype=np.float32)
    him = np.asarray(vg["him"][zsl]).astype(bool)
    alpha = np.asarray(vg[f"self_gravity_{SELF_GRAVITY}"]["alpha"][zsl], dtype=np.float32)

    # ---- P_min(I_UV), rebuilt with the pipeline's own interpolator
    print("Re-deriving P_min(I_UV) from the source cube (same interpolator as the pipeline)...")
    import dask.array as da
    cfg = load_resolved_config()
    grid = open_zarr()
    x_lo = int(np.where(np.abs(grid.x_pc) <= STATS_BOX_XY_HALF_RANGE_PC)[0].min())
    x_hi = int(np.where(np.abs(grid.x_pc) <= STATS_BOX_XY_HALF_RANGE_PC)[0].max()) + 1
    y_lo = int(np.where(np.abs(grid.y_pc) <= STATS_BOX_XY_HALF_RANGE_PC)[0].min())
    y_hi = int(np.where(np.abs(grid.y_pc) <= STATS_BOX_XY_HALF_RANGE_PC)[0].max()) + 1
    Iuv = da.from_zarr(grid.store["Iuv_final"])[zsl, y_lo:y_hi, x_lo:x_hi].compute()
    pminmax = thermal.build_pmin_pmax(cfg["bs19_mat_path"])
    Pmin = pminmax.p_min(Iuv.astype(np.float32)).astype(np.float32)
    del Iuv

    # Consistency check: the flag stored in the cache must be the flag
    # this P_min implies. If it is not, the threshold curve in the figure
    # would be drawn against a different P_min than the flag used.
    pnT_check = (n_H[:4].astype(np.float64) * T[:4].astype(np.float64)).astype(np.float32)
    him_check = (pnT_check < HIM_THRESHOLD_FACTOR * Pmin[:4]) & np.isfinite(Pmin[:4])
    agree = float((him_check == him[:4]).mean())
    print(f"  re-derived HIM flag agrees with the cached one on {agree:.6%} of a 4-plane sample")
    if agree < 1.0:
        raise RuntimeError(
            "Re-derived P_min does not reproduce the cached HIM flag -- the threshold "
            "curve would not be the one the flag was computed with.")

    # ---- accumulate per bin
    keys_flagstate = ("n_median", "n_p15", "n_p85", "T_median", "T_p15", "T_p85")
    out = {f"{k}__{state}": np.full(n_bins, np.nan)
           for k in keys_flagstate for state in ("flagged", "neutral")}
    for k in ("him_frac_vol", "him_frac_mw", "pnT_median", "pnT_p15", "pnT_p85",
              "half_pmin_median", "frac_alpha_lt1_vol", "frac_alpha_lt1_mw"):
        out[k] = np.full(n_bins, np.nan)
    out["n_cells"] = np.zeros(n_bins, dtype=np.int64)
    out["n_planes"] = np.zeros(n_bins, dtype=np.int64)
    out["n_flagged"] = np.zeros(n_bins, dtype=np.int64)

    z_centers = np.array([c for c, _ in groups], dtype=float)

    for b, (_, idxs) in enumerate(groups):
        out["n_planes"][b] = len(idxs)
        if len(idxs) == 0:
            continue
        him_b = him[idxs][:, footprint]
        n_b = n_H[idxs][:, footprint].astype(np.float64)
        T_b = T[idxs][:, footprint].astype(np.float64)
        a_b = alpha[idxs][:, footprint].astype(np.float64)
        pmin_b = Pmin[idxs][:, footprint].astype(np.float64)
        pnT_b = n_b * T_b
        neutral_b = ~him_b

        out["n_cells"][b] = him_b.size
        out["n_flagged"][b] = int(him_b.sum())

        # --- flag fractions, by volume and by observed mass
        out["him_frac_vol"][b] = him_b.sum() / him_b.size
        w_mass = np.where(np.isfinite(n_b), n_b, 0.0)
        total_mass = w_mass.sum()
        out["him_frac_mw"][b] = (w_mass[him_b].sum() / total_mass
                                 if total_mass > 0 else np.nan)

        # --- n_H and T, volume-weighted, flagged and neutral separately
        for state, sel in (("flagged", him_b), ("neutral", neutral_b)):
            if not sel.any():
                continue
            s_n = weighted_stats(n_b, np.where(sel, 1.0, 0.0))
            s_T = weighted_stats(T_b, np.where(sel, 1.0, 0.0))
            out[f"n_median__{state}"][b] = s_n.median
            out[f"n_p15__{state}"][b] = s_n.p_lo
            out[f"n_p85__{state}"][b] = s_n.p_hi
            out[f"T_median__{state}"][b] = s_T.median
            out[f"T_p15__{state}"][b] = s_T.p_lo
            out[f"T_p85__{state}"][b] = s_T.p_hi

        # --- the classification pair, over ALL cells in the bin
        all_w = np.ones(pnT_b.shape, dtype=np.float64)
        s_pnT = weighted_stats(pnT_b, all_w)
        out["pnT_median"][b] = s_pnT.median
        out["pnT_p15"][b] = s_pnT.p_lo
        out["pnT_p85"][b] = s_pnT.p_hi
        out["half_pmin_median"][b] = weighted_stats(
            HIM_THRESHOLD_FACTOR * pmin_b, all_w).median

        # --- alpha < 1 among NON-HIM cells (where Mach/sigma_nt die)
        w_vol = np.where(neutral_b, 1.0, 0.0)
        w_mw = np.where(neutral_b & np.isfinite(n_b), n_b, 0.0)
        for key, w in (("frac_alpha_lt1_vol", w_vol), ("frac_alpha_lt1_mw", w_mw)):
            m = np.isfinite(a_b) & (w > 0)
            out[key][b] = (float(np.sum(w[m] * (a_b[m] < 1.0)) / np.sum(w[m]))
                           if m.any() else np.nan)

        if b % 50 == 0:
            print(f"  bin {b + 1}/{n_bins} (z = {z_centers[b]:+.0f} pc)")

    out["z_pc"] = z_centers
    out["bin_edges"] = pipe.profile_bin_edges(use_abs=False)
    out["bin_width_pc"] = np.array([PROFILE_BIN_PC], dtype=float)
    out["percentile_levels"] = np.array([pipe.PCT_LO, pipe.PCT_HI], dtype=float)
    out["him_threshold_factor"] = np.array([HIM_THRESHOLD_FACTOR], dtype=float)
    out["variant"] = np.array([VARIANT])
    out["self_gravity"] = np.array([SELF_GRAVITY])
    out["provisional"] = np.array([PROVISIONAL_CUBE_HEADER])
    out["stats_box_z_half_range_pc"] = np.array([STATS_BOX_Z_HALF_RANGE_PC], dtype=float)

    np.savez_compressed(OUT_NPZ_PATH, **out)
    print(f"\nSaved {OUT_NPZ_PATH} ({OUT_NPZ_PATH.stat().st_size / 1e3:.0f} kB) "
          f"in {time.time() - t0:.1f}s")

    # a short human-readable summary, so a bad run is obvious immediately
    iz0 = int(np.argmin(np.abs(z_centers)))
    print(f"  at z ~ 0: HIM frac vol {out['him_frac_vol'][iz0]:.3f}, "
          f"mw {out['him_frac_mw'][iz0]:.3f}; "
          f"p_nT median {out['pnT_median'][iz0]:.0f} vs 0.5 P_min "
          f"{out['half_pmin_median'][iz0]:.0f} K cm^-3; "
          f"frac(alpha<1) vol {out['frac_alpha_lt1_vol'][iz0]:.3f}")
    print(f"  HIM frac vol range over the box: "
          f"{np.nanmin(out['him_frac_vol']):.3f} .. {np.nanmax(out['him_frac_vol']):.3f}")


if __name__ == "__main__":
    main()
