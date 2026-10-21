"""
Step 2g (Fig. 6): the Step 2e area-averaged alpha split by PHASE.

For each neutral phase k in {CNM, UNM, WNM}, per signed 4 pc z bin and per
60 pc slab (z = 0, 150, 300), over the whole +-500 pc footprint:

  <1/P_th>_k^-1   harmonic mean of p_th_phys over the cells of phase k
  <alpha>_k       <P_tot>_area * <1/P_th>_k -- the SAME <P_tot>_area for every
                  phase (P_tot is a property of the column, not of the cell)
  alpha_cell      <P_tot>_area / p_th_phys per cell of phase k: median, p15,
                  p85 (nearest-rank, src.physics.stats, as in Step 2e)

plus the VOLUME fraction (cell fraction) of CNM, UNM, WNM and HIM per bin.

Phases: the cached dP/dn flag, cache/core/alpha_core.zarr RAW/phase_flag_dpdn,
codes from src.physics.him: CNM = 0, UNM = 1, WNM = 2, HIM = 3. The HIM code
is the HIM flag, so CNM + UNM + WNM are exactly Step 2e's non-HIM cells.

<P_tot>_area is READ from cache/core/area_alpha.npz (Step 2e), not recomputed.
p_th_phys and the phase flag are read from alpha_core.zarr one z chunk at a
time. Self-gravity "mean" (fiducial) only, volume weighting only, RAW only.

A phase's value in a bin is NaN when the bin holds fewer than N_MIN_CELLS
cells of that phase.

Checks (raise on failure):
  a) recombination: sum_k N_k <1/P_th>_k / sum_k N_k == Step 2e's all-non-HIM
     <1/P_th> (= 1 / Pth_harm_vol) in every bin and slab, to float precision;
  b) CNM + UNM + WNM + HIM fractions sum to 1 in every bin, and the HIM volume
     fraction of the z = 0 slab equals results/numbers_table.csv's
     phase_fraction_HIM (vol, z0).
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.conventions import (  # noqa: E402
    PERCENTILE_SCHEME_DEFAULT,
    PROFILE_BIN_PC,
    PROVISIONAL_CUBE_HEADER,
    SLAB_CENTERS_PC,
    SLAB_HALF_THICKNESS_PC,
    STATS_BOX_XY_HALF_RANGE_PC,
    STATS_BOX_Z_HALF_RANGE_PC,
    THERMAL_PRESSURE_HEADER,
)
from src.physics.him import PHASE_CNM, PHASE_HIM, PHASE_UNM, PHASE_WNM  # noqa: E402
from src.physics.loading import open_zarr, slab_z_indices  # noqa: E402
from src.physics.stats import percentile_levels, weighted_stats  # noqa: E402

ALPHA_CORE_ZARR_PATH = Path("cache/core/alpha_core.zarr")
AREA_ALPHA_NPZ_PATH = Path("cache/core/area_alpha.npz")
NUMBERS_TABLE_CSV = Path("results/numbers_table.csv")
OUT_NPZ_PATH = Path("cache/core/phase_alpha.npz")
OUT_TXT_PATH = Path("results/phase_alpha.txt")

VARIANT = "RAW"
SG = "mean"
N_MIN_CELLS = 100
Z_CHUNK = 64
NEUTRAL = (("CNM", PHASE_CNM), ("UNM", PHASE_UNM), ("WNM", PHASE_WNM))
ALL_PHASES = NEUTRAL + (("HIM", PHASE_HIM),)
RECOMB_RTOL = 1e-12
HIM_FRAC_RTOL = 1e-9
STAT_KEYS = ("Pth_inv_inv", "alpha", "alpha_cell_median", "alpha_cell_p_lo",
             "alpha_cell_p_hi", "n_cells")


def phase_group_stats(p_th, p_tot_area, n_min=N_MIN_CELLS, scheme=PERCENTILE_SCHEME_DEFAULT):
    """The Step 2g statistics of one phase in one group (volume-weighted).

    Returns a dict over STAT_KEYS, plus 'sum_inv' (the unthresholded sum of
    1/p_th, for the recombination check). Fewer than n_min cells -> NaN
    statistics (sum_inv and n_cells still reported).
    """
    p = np.asarray(p_th, dtype=np.float64).ravel()
    n = p.size
    sum_inv = float(np.sum(1.0 / p)) if n else 0.0
    out = {"n_cells": n, "sum_inv": sum_inv}
    if n < n_min or not np.isfinite(p_tot_area):
        out.update({k: np.nan for k in STAT_KEYS if k != "n_cells"})
        return out
    inv_mean = sum_inv / n
    st = weighted_stats(p_tot_area / p, np.ones(n), scheme)
    out.update({
        "Pth_inv_inv": 1.0 / inv_mean,
        "alpha": p_tot_area * inv_mean,
        "alpha_cell_median": st.median,
        "alpha_cell_p_lo": st.p_lo,
        "alpha_cell_p_hi": st.p_hi,
    })
    return out


def nan_onset(z, values):
    """Smallest |z| of a NaN bin on each side: (z<0 side, z>0 side); None if none."""
    out = []
    for side in (z < 0, z > 0):
        bad = side & ~np.isfinite(values)
        out.append(float(np.min(np.abs(z[bad]))) if bad.any() else None)
    return tuple(out)


def read_him_fraction_z0(path=NUMBERS_TABLE_CSV):
    with open(path, newline="") as f:
        rows = csv.DictReader([ln for ln in f if not ln.startswith("#")])
        for r in rows:
            if (r["variant"] == VARIANT and r["phase_scheme"] == "dPdn"
                    and r["quantity"] == "phase_fraction_HIM" and r["weighting"] == "vol"
                    and r["stat"] == "z0"):
                return float(r["value"])
    raise RuntimeError("phase_fraction_HIM vol z0 (dPdn) not found in numbers_table.csv")


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    import pipeline.compute_all as pipe
    pct_lo, pct_hi = percentile_levels(PERCENTILE_SCHEME_DEFAULT)

    z_full = np.asarray(open_zarr().z_pc, dtype=np.float64)
    zsl, z_box = pipe._box_z_slice(z_full)
    bin_groups = pipe.profile_bin_groups(z_box, use_abs=False)
    slab_groups = [(float(zc), slab_z_indices(z_box, zc)) for zc in SLAB_CENTERS_PC]
    groups = [np.asarray(i, int) for _, i in bin_groups] + [np.asarray(i, int) for _, i in slab_groups]
    n_bins, n_groups = len(bin_groups), len(bin_groups) + len(slab_groups)
    z_bins = np.array([c for c, _ in bin_groups], dtype=float)

    aa = np.load(AREA_ALPHA_NPZ_PATH, allow_pickle=False)
    if not np.allclose(aa["z_pc"], z_bins):
        raise RuntimeError("area_alpha.npz bins differ from the profile bins built here")
    ptot_area = np.concatenate([aa[f"{SG}__Ptot_area"], aa[f"slab__{SG}__Ptot_area"]])
    pth_harm_2e = np.concatenate([aa[f"{SG}__Pth_harm_vol"], aa[f"slab__{SG}__Pth_harm_vol"]])
    n_cells_2e = np.concatenate([aa[f"{SG}__n_cells"], aa[f"slab__{SG}__n_cells"]])

    core = zarr.open_group(str(ALPHA_CORE_ZARR_PATH), mode="r")
    vg = core[VARIANT]
    pth_arr, ph_arr = vg["p_th_phys"], vg["phase_flag_dpdn"]
    last_plane = np.array([g.max() if g.size else -1 for g in groups])

    # per group, per neutral phase: collected p_th values until the group is complete
    pending = {g: {name: [] for name, _ in NEUTRAL} for g in range(n_groups)}
    counts = np.zeros((n_groups, len(ALL_PHASES)), dtype=np.int64)
    res = {name: {k: np.full(n_groups, np.nan) for k in STAT_KEYS + ("sum_inv",)}
           for name, _ in NEUTRAL}
    plane_groups = [[] for _ in range(len(z_box))]
    for g, idxs in enumerate(groups):
        for i in idxs:
            plane_groups[int(i)].append(g)

    print(f"Reducing {len(z_box)} box planes ({n_bins} bins + {len(slab_groups)} slabs), "
          f"chunks of {Z_CHUNK}...")
    for lo in range(0, len(z_box), Z_CHUNK):
        hi = min(lo + Z_CHUNK, len(z_box))
        sl = slice(zsl.start + lo, zsl.start + hi)
        pth_c = np.asarray(pth_arr[sl], dtype=np.float64).reshape(hi - lo, -1)
        ph_c = np.asarray(ph_arr[sl]).reshape(hi - lo, -1)
        for k in range(hi - lo):
            i_plane = lo + k
            gids = plane_groups[i_plane]
            if not gids:
                continue
            ph = ph_c[k]
            cnt = np.array([np.count_nonzero(ph == code) for _, code in ALL_PHASES])
            for g in gids:
                counts[g] += cnt
                for name, code in NEUTRAL:
                    pending[g][name].append(pth_c[k][ph == code])
            for g in gids:
                if last_plane[g] == i_plane:          # group complete: reduce and free
                    for name, _ in NEUTRAL:
                        vals = (np.concatenate(pending[g][name]) if pending[g][name]
                                else np.empty(0))
                        st = phase_group_stats(vals, ptot_area[g])
                        for key, v in st.items():
                            res[name][key][g] = v
                    pending[g] = None
        del pth_c, ph_c

    # ---- check a): recombination -------------------------------------------
    n_neu = sum(res[name]["n_cells"] for name, _ in NEUTRAL)
    inv_recomb = sum(res[name]["sum_inv"] for name, _ in NEUTRAL) / n_neu
    inv_2e = 1.0 / pth_harm_2e
    fin = np.isfinite(inv_2e)
    rel_a = np.abs(inv_recomb[fin] / inv_2e[fin] - 1.0)
    max_a = float(rel_a.max())
    print(f"check a) recombined <1/P_th> vs Step 2e: max |rel diff| {max_a:.2e} "
          f"over {int(fin.sum())} groups")
    if not np.array_equal(fin, n_neu > 0) or max_a > RECOMB_RTOL:
        raise RuntimeError(f"Recombination FAILED: {max_a:.3e} > {RECOMB_RTOL:.0e}. STOPPING.")

    # ---- check b): fractions ------------------------------------------------
    tot = counts.sum(axis=1)
    if not np.array_equal(tot, n_cells_2e.astype(np.int64)):
        raise RuntimeError("cell counts per group differ from Step 2e's n_cells")
    frac = counts / tot[:, None]
    max_b = float(np.max(np.abs(frac.sum(axis=1) - 1.0)))
    him_z0 = float(frac[n_bins + list(SLAB_CENTERS_PC).index(0.0), 3])
    him_ref = read_him_fraction_z0()
    rel_him = abs(him_z0 / him_ref - 1.0)
    print(f"check b) fractions sum to 1 within {max_b:.1e}; HIM vol fraction z0 slab "
          f"{him_z0:.7f} vs numbers_table {him_ref:.7f} (rel {rel_him:.1e})")
    if max_b > 1e-12 or rel_him > HIM_FRAC_RTOL:
        raise RuntimeError("Phase-fraction check FAILED. STOPPING.")

    # ---- save -----------------------------------------------------------------
    out = {
        "z_pc": z_bins, "bin_width_pc": np.array([PROFILE_BIN_PC]),
        "slab_centers_pc": np.array(SLAB_CENTERS_PC, dtype=float),
        "slab_half_thickness_pc": np.array([SLAB_HALF_THICKNESS_PC]),
        "phase_names": np.array([n for n, _ in ALL_PHASES]),
        "phase_codes": np.array([c for _, c in ALL_PHASES]),
        "n_min_cells": np.array([N_MIN_CELLS]),
        "percentile_levels": np.array([pct_lo, pct_hi]),
        "Ptot_area": ptot_area[:n_bins], "slab__Ptot_area": ptot_area[n_bins:],
        "Pth_inv_inv_all": pth_harm_2e[:n_bins], "slab__Pth_inv_inv_all": pth_harm_2e[n_bins:],
        "check_recombination_max_rel": np.array([max_a]),
        "check_him_fraction_z0": np.array([him_z0, him_ref]),
        "provisional": np.array([PROVISIONAL_CUBE_HEADER]),
        "self_gravity": np.array([SG]), "variant": np.array([VARIANT]),
    }
    for j, (name, _) in enumerate(ALL_PHASES):
        out[f"{name}__vol_frac"] = frac[:n_bins, j]
        out[f"slab__{name}__vol_frac"] = frac[n_bins:, j]
    for name, _ in NEUTRAL:
        for key in STAT_KEYS:
            out[f"{name}__{key}"] = res[name][key][:n_bins]
            out[f"slab__{name}__{key}"] = res[name][key][n_bins:]
    OUT_NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ_PATH, **out)
    print(f"Saved {OUT_NPZ_PATH}")

    # ---- report -------------------------------------------------------------
    L = [
        PROVISIONAL_CUBE_HEADER,
        THERMAL_PRESSURE_HEADER,
        "Step 2g (Fig. 6): area-averaged alpha split by PHASE",
        "=" * 100,
        "Per signed 4 pc z bin and per 60 pc slab, over the whole footprint "
        f"(|x|,|y| <= {STATS_BOX_XY_HALF_RANGE_PC:.0f} pc, box |z| <= "
        f"{STATS_BOX_Z_HALF_RANGE_PC:.0f} pc),",
        f"RAW, self-gravity {SG} (fiducial), VOLUME-weighted (equal weight per cell):",
        "  <1/P_th>_k^-1  harmonic mean of p_th_phys over the cells of phase k",
        "  <alpha>_k      <P_tot>_area * <1/P_th>_k -- the same <P_tot>_area for every phase,",
        "                 read from cache/core/area_alpha.npz (Step 2e), not recomputed",
        f"  alpha_cell     <P_tot>_area / p_th_phys per cell of phase k: median, p{pct_lo:g}, "
        f"p{pct_hi:g} (nearest-rank)",
        "  vol frac       cell fraction of each phase in the bin (HIM included)",
        "Phases: cached dP/dn flag (alpha_core.zarr RAW/phase_flag_dpdn), codes "
        + ", ".join(f"{n} = {c}" for n, c in ALL_PHASES)
        + " (src.physics.him).",
        f"A phase's statistics are NaN in a bin with fewer than N_MIN_CELLS = {N_MIN_CELLS} "
        "of its cells.",
        "",
        "-" * 100,
        "CHECKS (the script raises if either fails)",
        "-" * 100,
        f"  a) sum_k N_k <1/P_th>_k / sum_k N_k vs Step 2e's all-non-HIM <1/P_th>, every bin and",
        f"     slab: max |rel diff| {max_a:.2e} (threshold {RECOMB_RTOL:.0e})  PASS",
        f"  b) CNM+UNM+WNM+HIM fractions sum to 1 within {max_b:.1e} in every bin and slab; HIM",
        f"     volume fraction of the z = 0 slab {him_z0:.7f} vs numbers_table.csv {him_ref:.7f}",
        f"     (rel {rel_him:.1e})  PASS",
        "",
        "-" * 100,
        "SLABS (+-30 pc)",
        "-" * 100,
        f"{'z':>5}{'phase':>6}{'<1/P_th>_k^-1':>15}{'<alpha>_k':>11}"
        f"{'alpha_cell median [p15, p85]':>32}{'vol frac':>10}{'cells':>10}",
        f"{'[pc]':>5}{'':>6}{'[K cm^-3]':>15}",
    ]
    for i, zc in enumerate(SLAB_CENTERS_PC):
        g = n_bins + i
        L.append(f"{zc:>5.0f}{'all':>6}{pth_harm_2e[g]:>15.1f}"
                 f"{ptot_area[g] / pth_harm_2e[g]:>11.3f}{'(Step 2e, non-HIM)':>32}"
                 f"{1 - frac[g, 3]:>10.4f}{int(n_neu[g]):>10d}")
        for j, (name, _) in enumerate(NEUTRAL):
            r = res[name]
            L.append(f"{'':>5}{name:>6}{r['Pth_inv_inv'][g]:>15.1f}{r['alpha'][g]:>11.3f}"
                     f"{r['alpha_cell_median'][g]:>13.3f}  [{r['alpha_cell_p_lo'][g]:>6.3f},"
                     f"{r['alpha_cell_p_hi'][g]:>7.3f}]{frac[g, j]:>10.4f}{int(r['n_cells'][g]):>10d}")
        L.append(f"{'':>5}{'HIM':>6}{'':>15}{'':>11}{'':>32}{frac[g, 3]:>10.4f}"
                 f"{int(counts[g, 3]):>10d}")
        L.append(f"{'':>5}  <P_tot>_area = {ptot_area[g]:.1f} K cm^-3")
        L.append("")
    L += [
        "-" * 100,
        f"NaN BINS (fewer than {N_MIN_CELLS} cells of the phase), 4 pc signed bins in |z| <= "
        f"{STATS_BOX_Z_HALF_RANGE_PC:.0f} pc",
        "-" * 100,
        f"{'phase':>6}{'NaN bins':>10}{'first NaN at z < 0':>20}{'first NaN at z > 0':>20}",
    ]
    for name, _ in NEUTRAL:
        a = res[name]["alpha"][:n_bins]
        neg, pos = nan_onset(z_bins, a)
        fmt = lambda v: "none" if v is None else f"|z| = {v:.0f} pc"  # noqa: E731
        L.append(f"{name:>6}{int(np.sum(~np.isfinite(a))):>10d}{fmt(neg):>20}{fmt(pos):>20}")
    L += [
        "  ('first NaN' = the bin centre nearest the midplane that is NaN; bins beyond it may",
        "   recover where the phase reappears -- the full arrays are in phase_alpha.npz)",
        "",
        f"(runtime {time.time() - t0:.1f}s)",
    ]
    OUT_TXT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
