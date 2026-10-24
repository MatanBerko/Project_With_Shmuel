"""
Fig. 7: the missing support expressed as a VELOCITY.

alpha = W / P_th says how much more pressure the gas needs than its thermal
pressure supplies. This script turns that into the velocity dispersion the
gas would need if ALL of the missing support were motions (an upper limit
on turbulent velocities: magnetic and cosmic-ray pressure also contribute).

Definitions, per signed 4 pc z bin and per +-30 pc slab, over the whole
+-500 pc footprint, for each phase k in {CNM, UNM, WNM, NEUTRAL = all
non-HIM}, using the cells of phase k in the group:

  W                 the group's area-averaged gas weight <P_tot>_area/k_B
                    [K cm^-3] -- ONE number per group, shared by every
                    phase (the weight is a property of the column)
  <rho>_k           plain (volume) mean of rho = 1.4 m_H n_H
  sigma_eff,k       sqrt(k_B W / <rho>_k)                 [km/s]
  sigma_th,k        sqrt(sum(k_B p_th_phys) / sum(rho))   [km/s]
                    (mass-weighted isothermal c_s^2 = P_th/rho)
  sigma_nt,k        sqrt(max(sigma_eff^2 - sigma_th^2, 0)); the unclipped
                    difference sigma_eff^2 - sigma_th^2 is kept as well
  <M^2>_k           sum(rho (W/p_th_phys - 1)) / sum(rho): the mass-weighted
                    mean of the per-cell M^2 = alpha_cell - 1. Cells with
                    alpha_cell < 1 contribute NEGATIVE M^2 and are kept.
  f_m(alpha<1)      mass fraction of phase-k cells with W/p_th_phys < 1
  f_vol             N_k / N_cells of the group (HIM included in N_cells)

Per-cell distributions, for k in {CNM, WNM, NEUTRAL}: the MASS-weighted
(weights = rho) 15th and 85th percentiles of

  sigma_eff,cell    sqrt(k_B W / rho)          [km/s]
  c_s,cell          sqrt(k_B p_th_phys / rho)  [km/s]
  M2_cell           W / p_th_phys - 1

with src.physics.stats.weighted_stats (nearest rank on the weighted
cumulative distribution, PERCENTILE_SCHEME_DEFAULT = 15/85), the estimator
every other percentile in the project uses. The group means above are the
mass-weighted means of the squares of the first two (sigma_eff^2 =
k_B W sum(1/rho * rho)/sum(rho), sigma_th^2 likewise) and of M2_cell itself,
so a mean is not a percentile of the same distribution and can sit outside
the 15-85 band when the distribution is skewed.

and for ALL GAS (every cell, HIM included): sigma_eff,all = sqrt(k_B W /
<rho>_all). All velocities are 1D (vertical / line of sight). Mass
weighting follows Ostriker & Kim (2022): sigma^2 = sum(P) / sum(rho).
A phase's statistics are NaN in a group with fewer than N_MIN_CELLS of its
cells.

Slabs: |z - z_c| <= 30 pc at z_c = z_mid + {-300, -150, 0, +150, +300} pc,
selected with the same slab_z_indices(z_box, midplane.slab_center(...))
call compute_area_alpha.py uses for its midplane slab, shifted.

Inputs (nothing is rebuilt):
  cache/core/alpha_core.zarr   RAW n_model, p_th_phys, phase_flag_dpdn, him
  cache/core/area_alpha.npz    the per-plane <P_tot>_area (mean__ptot_plane_mean_full),
                               from which every group's W is the mean over its
                               planes (planes all hold the same number of cells),
                               and the Step 2e mass-weighted <alpha_cell> used
                               by assertion (i)

Assertions (raise on failure):
  (i)  <M^2>_NEUTRAL + 1 == the cached mass-weighted <alpha_cell> (Step 2e,
       full footprint) in every bin and in the midplane slab, < 1e-10 rel.
       The cached +150/+300 slabs are centred on z = 150/300, not on
       z_mid +- 150/300, so they are not comparable and are not used.
  (ii) sigma_eff,all^2 <rho>_all / k_B == W, < 1e-12 rel.
plus consistency checks that the inputs line up: W of every bin and the
midplane slab equals the cached Ptot_area; the phase flag's HIM code is the
him flag cell for cell; group cell counts equal Step 2e's.

COMPUTE script: reads the cache, writes cache/velocity/velocity.npz and
results/velocity.txt. scripts/design/fig_velocity.py reads only the npz.
"""

import sys
import time
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.conventions import (  # noqa: E402
    PERCENTILE_SCHEME_DEFAULT,
    K_B,
    M_H,
    MIDPLANE_MODE,
    MU,
    PROFILE_BIN_PC,
    PROVISIONAL_CUBE_HEADER,
    SLAB_HALF_THICKNESS_PC,
    STATS_BOX_XY_HALF_RANGE_PC,
    STATS_BOX_Z_HALF_RANGE_PC,
    THERMAL_PRESSURE_HEADER,
)
from src.physics import midplane  # noqa: E402
from src.physics.him import PHASE_CNM, PHASE_HIM, PHASE_UNM, PHASE_WNM  # noqa: E402
from src.physics.loading import open_zarr, slab_z_indices  # noqa: E402
from src.physics.stats import percentile_levels, weighted_stats  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════
# ██  CONTROL BLOCK
# ═══════════════════════════════════════════════════════════════════════
ALPHA_CORE_ZARR_PATH = Path("cache/core/alpha_core.zarr")
AREA_ALPHA_NPZ_PATH = Path("cache/core/area_alpha.npz")
OUT_NPZ_PATH = Path("cache/velocity/velocity.npz")
OUT_TXT_PATH = Path("results/velocity.txt")

VARIANT = "RAW"
SG = "mean"                                   # self-gravity footprint_mean (fiducial)
SLAB_OFFSETS_PC = (-300.0, -150.0, 0.0, 150.0, 300.0)   # slab centres relative to z_mid
N_MIN_CELLS = 100
Z_CHUNK = 64                                  # a whole number of the cache's 32-plane chunks
ALPHA_IDENTITY_RTOL = 1e-10                   # assertion (i)
W_IDENTITY_RTOL = 1e-12                       # assertion (ii), and W vs the cached Ptot_area

# Literature reference values, converted to our 1D isothermal convention below.
HT03_MT = 3.0                  # Heiles & Troland 2003, CNM turbulent Mach number (3D,
                               # C_s^2 = kT / (1.4 m_H))
MM21_SIGMA_TOTAL = (8.0, 0.6)  # Marchal & Miville-Deschenes 2021, WNM, line of sight [km/s]
MM21_SIGMA_TH = (7.0, 0.7)
MM21_SIGMA_TURB = (3.8, 0.4)
MM21_MS = (0.87, 0.15)         # 3D, ADIABATIC sound speed (gamma = 5/3)
GAMMA_ADIABATIC = 5.0 / 3.0
F_NEUTRAL = 1.1                # particles per H in p_th_phys = 1.1 n_H k T
# ═══════════════════════════════════════════════════════════════════════

PHASES = (("CNM", PHASE_CNM), ("UNM", PHASE_UNM), ("WNM", PHASE_WNM))
STAT_PHASES = ("CNM", "UNM", "WNM", "NEUTRAL")
N_CODES = 4                                   # CNM, UNM, WNM, HIM
SUM_KEYS = ("n", "s_n", "s_p", "s_n_m2", "s_n_sub1")
STAT_KEYS = ("n_cells", "vol_frac", "rho_mean", "sigma_eff", "sigma_th", "sigma_nt",
             "sigma_nt2_unclipped", "M2", "sqrt_M2", "mass_frac_alpha_lt1")
KMS = 1.0e5
PCT_PHASES = ("CNM", "WNM", "NEUTRAL")
PCT_QUANTITIES = ("sigma_eff_cell", "cs_cell", "M2_cell")


def literature_conversions():
    """The reference values in our 1D isothermal convention, with their formulas."""
    mm21_ms_lo = MM21_MS[0] - MM21_MS[1]
    mm21_ms_hi = MM21_MS[0] + MM21_MS[1]
    to_m2 = lambda ms: ms ** 2 * GAMMA_ADIABATIC / 3.0  # noqa: E731
    return {
        "HT03_M2": HT03_MT ** 2 / 3.0,
        "HT03_M2_corrected": HT03_MT ** 2 / (3.0 * F_NEUTRAL),
        "MM21_M2": to_m2(MM21_MS[0]),
        "MM21_M2_lo": to_m2(mm21_ms_lo),
        "MM21_M2_hi": to_m2(mm21_ms_hi),
        "MM21_sigma_total": np.array(MM21_SIGMA_TOTAL),
        "MM21_sigma_th": np.array(MM21_SIGMA_TH),
        "MM21_sigma_turb": np.array(MM21_SIGMA_TURB),
        "MM21_Ms": np.array(MM21_MS),
    }


# ---------------------------------------------------------------------------
# sums and statistics
# ---------------------------------------------------------------------------
def phase_sums(n_H, p_th, phase, W):
    """Per-phase-code sums over a set of cells under one shared weight W.

    n_H [cm^-3], p_th = p_th_phys [K cm^-3], phase = codes 0..3, W [K cm^-3].
    Returns {key: (N_CODES,) float64} over SUM_KEYS:
      n         cell count
      s_n       sum n_H              (mass, up to the constant 1.4 m_H)
      s_p       sum p_th
      s_n_m2    sum n_H (W/p_th - 1) (mass-weighted M^2, unnormalised)
      s_n_sub1  sum n_H over cells with W/p_th < 1
    """
    n = np.asarray(n_H, dtype=np.float64).ravel()
    p = np.asarray(p_th, dtype=np.float64).ravel()
    ph = np.asarray(phase).ravel().astype(np.int64)
    a = W / p
    bc = lambda w=None: np.bincount(ph, weights=w, minlength=N_CODES).astype(np.float64)  # noqa: E731
    return {
        "n": bc(),
        "s_n": bc(n),
        "s_p": bc(p),
        "s_n_m2": bc(n * (a - 1.0)),
        "s_n_sub1": bc(np.where(a < 1.0, n, 0.0)),
    }


def combine(sums):
    """{name: {key: scalar}} for CNM, UNM, WNM, NEUTRAL (codes 0-2) and ALL (0-3).

    Works on (..., N_CODES) arrays as well, reducing the last axis.
    """
    out = {name: {k: sums[k][..., code] for k in SUM_KEYS} for name, code in PHASES}
    out["NEUTRAL"] = {k: sums[k][..., :PHASE_HIM].sum(axis=-1) for k in SUM_KEYS}
    out["ALL"] = {k: sums[k].sum(axis=-1) for k in SUM_KEYS}
    return out


def velocity_stats(s, W, n_total, n_min=N_MIN_CELLS):
    """The velocity statistics of one phase from its sums (scalars or arrays).

    Returns ({STAT_KEY: value}, {STAT_KEY: value}) -- thresholded (NaN where
    fewer than n_min cells) and raw (no threshold; for the assertions).
    """
    W = np.asarray(W, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho_mean = MU * M_H * s["s_n"] / s["n"]
        sig_eff2 = K_B * W / rho_mean
        sig_th2 = K_B * s["s_p"] / (MU * M_H * s["s_n"])
        d2 = sig_eff2 - sig_th2
        m2 = s["s_n_m2"] / s["s_n"]
        raw = {
            "n_cells": np.asarray(s["n"], dtype=np.float64),
            "vol_frac": s["n"] / n_total,
            "rho_mean": rho_mean,
            "sigma_eff": np.sqrt(sig_eff2) / KMS,
            "sigma_th": np.sqrt(sig_th2) / KMS,
            "sigma_nt": np.sqrt(np.maximum(d2, 0.0)) / KMS,
            "sigma_nt2_unclipped": d2 / KMS ** 2,
            "M2": m2,
            "sqrt_M2": np.where(m2 > 0, np.sqrt(np.abs(m2)), np.nan),
            "mass_frac_alpha_lt1": s["s_n_sub1"] / s["s_n"],
        }
    ok = np.asarray(s["n"]) >= n_min
    thr = {k: (v if k in ("n_cells", "vol_frac") else np.where(ok, v, np.nan))
           for k, v in raw.items()}
    return thr, raw


def cell_percentiles(n_H, p_th, W, n_min=N_MIN_CELLS, scheme=PERCENTILE_SCHEME_DEFAULT):
    """Mass-weighted (w = rho) low/high percentiles of the per-cell quantities.

    n_H, p_th: the cells of one phase in one group; W: the group's weight.
    Returns {f"{q}_p_lo" / f"{q}_p_hi": value} over PCT_QUANTITIES, NaN when
    fewer than n_min cells.
    """
    n = np.asarray(n_H, dtype=np.float64).ravel()
    p = np.asarray(p_th, dtype=np.float64).ravel()
    out = {f"{q}_{e}": np.nan for q in PCT_QUANTITIES for e in ("p_lo", "p_hi")}
    if n.size < n_min:
        return out
    rho = MU * M_H * n
    cell = {
        "sigma_eff_cell": np.sqrt(K_B * W / rho) / KMS,
        "cs_cell": np.sqrt(K_B * p / rho) / KMS,
        "M2_cell": W / p - 1.0,
    }
    for q, v in cell.items():
        st = weighted_stats(v, rho, scheme)
        out[f"{q}_p_lo"], out[f"{q}_p_hi"] = st.p_lo, st.p_hi
    return out


def sigma_eff_all(s_all, W):
    """sigma_eff over ALL gas [km/s] and the <rho>_all it used."""
    rho_mean = MU * M_H * s_all["s_n"] / s_all["n"]
    return np.sqrt(K_B * W / rho_mean) / KMS, rho_mean


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    import pipeline.compute_all as pipe

    z_full = np.asarray(open_zarr().z_pc, dtype=np.float64)
    zsl, z_box = pipe._box_z_slice(z_full)
    core = zarr.open_group(str(ALPHA_CORE_ZARR_PATH), mode="r")
    vg = core[VARIANT]
    sg_attrs = vg[f"self_gravity_{SG}"].attrs
    if sg_attrs.get("midplane_mode", "sun") != MIDPLANE_MODE:
        raise RuntimeError(f"cache built with midplane {sg_attrs.get('midplane_mode')!r}, "
                           f"MIDPLANE_MODE is {MIDPLANE_MODE!r}")
    z_mid = float(sg_attrs.get("z_mid_pc", 0.0))

    aa = np.load(AREA_ALPHA_NPZ_PATH, allow_pickle=False)
    if abs(float(aa[f"{SG}__z_mid_pc"][0]) - z_mid) > 1e-9:
        raise RuntimeError("area_alpha.npz z_mid differs from the zarr's")

    # ---- groups: 4 pc signed bins, then the five slabs ----------------------
    bin_groups = pipe.profile_bin_groups(z_box, use_abs=False)
    z_bins = np.array([c for c, _ in bin_groups], dtype=float)
    if not np.allclose(aa["z_pc"], z_bins):
        raise RuntimeError("area_alpha.npz bins differ from the profile bins built here")
    mid_center = midplane.slab_center(0.0, z_mid, MIDPLANE_MODE)
    slab_centers = np.array([mid_center + off for off in SLAB_OFFSETS_PC])
    slab_idx = [slab_z_indices(z_box, zc) for zc in slab_centers]
    i_mid = SLAB_OFFSETS_PC.index(0.0)
    groups = [np.asarray(i, int) for _, i in bin_groups] + [np.asarray(i, int) for i in slab_idx]
    n_bins, n_groups = len(bin_groups), len(groups)
    plane_groups = [[] for _ in range(len(z_box))]
    for g, idxs in enumerate(groups):
        for i in idxs:
            plane_groups[int(i)].append(g)
    print(f"z_mid = {z_mid:+.4f} pc; {n_bins} bins + {len(slab_idx)} slabs at "
          + ", ".join(f"{zc:+.1f}" for zc in slab_centers) + " pc")
    for zc, idx in zip(slab_centers, slab_idx):
        print(f"  slab {zc:+8.2f}: planes z = {z_box[idx].min():+g} .. {z_box[idx].max():+g} "
              f"({idx.size} planes)")

    # ---- W per group from the cached per-plane <P_tot>_area -----------------
    ptot_plane_box = np.asarray(aa[f"{SG}__ptot_plane_mean_full"], dtype=np.float64)[zsl]
    W = np.array([ptot_plane_box[idxs].mean() for idxs in groups])
    W_cached = np.concatenate([aa[f"{SG}__Ptot_area"], aa[f"slab__{SG}__Ptot_area"][:1]])
    W_here = np.concatenate([W[:n_bins], W[n_bins + i_mid:n_bins + i_mid + 1]])
    w_check = float(np.max(np.abs(W_here / W_cached - 1.0)))
    print(f"W vs cached Ptot_area (bins + midplane slab): max rel {w_check:.2e}")
    if w_check > W_IDENTITY_RTOL:
        raise RuntimeError(f"W per group differs from the cached Ptot_area by {w_check:.3e}")

    # ---- the sums, accumulated over z chunks ---------------------------------
    acc = {k: np.zeros((n_groups, N_CODES)) for k in SUM_KEYS}
    n_arr, p_arr, ph_arr, him_arr = (vg["n_model"], vg["p_th_phys"],
                                     vg["phase_flag_dpdn"], vg["him"])
    n_bad = 0
    # neutral cells per group, held until the group's last plane is read
    last_plane = np.array([g.max() for g in groups])
    pending = {g: [] for g in range(n_groups)}
    pct = {k: {f"{q}_{e}": np.full(n_groups, np.nan) for q in PCT_QUANTITIES
               for e in ("p_lo", "p_hi")} for k in PCT_PHASES}
    print(f"Reducing {len(z_box)} box planes in chunks of {Z_CHUNK}...")
    for lo in range(0, len(z_box), Z_CHUNK):
        hi = min(lo + Z_CHUNK, len(z_box))
        sl = slice(zsl.start + lo, zsl.start + hi)
        n_c = np.asarray(n_arr[sl], dtype=np.float64).reshape(hi - lo, -1)
        p_c = np.asarray(p_arr[sl], dtype=np.float64).reshape(hi - lo, -1)
        ph_c = np.asarray(ph_arr[sl]).reshape(hi - lo, -1)
        him_c = np.asarray(him_arr[sl]).astype(bool).reshape(hi - lo, -1)
        if not np.array_equal(him_c, ph_c == PHASE_HIM):
            raise RuntimeError("phase_flag_dpdn == HIM differs from the him flag")
        n_bad += int(np.sum(~(np.isfinite(n_c) & np.isfinite(p_c) & (n_c > 0) & (p_c > 0))))
        for k in range(hi - lo):
            for g in plane_groups[lo + k]:
                s = phase_sums(n_c[k], p_c[k], ph_c[k], W[g])
                for key in SUM_KEYS:
                    acc[key][g] += s[key]
                neu = ph_c[k] != PHASE_HIM
                pending[g].append((n_c[k][neu], p_c[k][neu], ph_c[k][neu]))
                if last_plane[g] == lo + k:          # group complete: percentiles, then free
                    n_g, p_g, ph_g = (np.concatenate(a) for a in zip(*pending[g]))
                    for name, sel in (("CNM", ph_g == PHASE_CNM), ("WNM", ph_g == PHASE_WNM),
                                      ("NEUTRAL", slice(None))):
                        for key, v in cell_percentiles(n_g[sel], p_g[sel], W[g]).items():
                            pct[name][key][g] = v
                    pending[g] = None
        del n_c, p_c, ph_c, him_c
    if n_bad:
        raise RuntimeError(f"{n_bad} box cells have non-finite or non-positive n_H or p_th")

    comb = combine(acc)
    n_total = comb["ALL"]["n"]
    n_cells_cached = np.concatenate([aa[f"{SG}__n_cells"], aa[f"slab__{SG}__n_cells"][:1]])
    n_here = np.concatenate([n_total[:n_bins], n_total[n_bins + i_mid:n_bins + i_mid + 1]])
    if not np.array_equal(n_here, n_cells_cached):
        raise RuntimeError("group cell counts differ from Step 2e's n_cells")

    stats, raw = {}, {}
    for name in STAT_PHASES:
        stats[name], raw[name] = velocity_stats(comb[name], W, n_total)
    sig_all, rho_all = sigma_eff_all(comb["ALL"], W)
    him_vol = acc["n"][:, PHASE_HIM] / n_total

    # ---- assertion (i): <M^2>_NEUTRAL + 1 == cached mw <alpha_cell> ----------
    a_cached = np.concatenate([aa[f"{SG}__alpha_cell_mean_mw"],
                               aa[f"slab__{SG}__alpha_cell_mean_mw"][:1]])
    a_here = 1.0 + np.concatenate([raw["NEUTRAL"]["M2"][:n_bins],
                                   raw["NEUTRAL"]["M2"][n_bins + i_mid:n_bins + i_mid + 1]])
    fin = np.isfinite(a_cached)
    if not np.array_equal(fin, np.isfinite(a_here)):
        raise RuntimeError("assertion (i): finite groups differ between here and the cache")
    rel_i = np.abs(a_here[fin] / a_cached[fin] - 1.0)
    max_i, mid_i = float(rel_i.max()), float(rel_i[-1])
    print(f"assertion (i): <M^2>_NEUTRAL + 1 vs cached mw <alpha_cell>: max rel {max_i:.2e} "
          f"over {int(fin.sum())} groups; midplane slab {a_here[-1]:.6f} vs "
          f"{a_cached[-1]:.6f} (rel {mid_i:.2e})")
    if max_i > ALPHA_IDENTITY_RTOL:
        raise RuntimeError(f"assertion (i) FAILED: {max_i:.3e} > {ALPHA_IDENTITY_RTOL:.0e}")

    # ---- assertion (ii): sigma_eff,all^2 <rho>_all / k_B == W ---------------
    w_back = (sig_all * KMS) ** 2 * rho_all / K_B
    max_ii = float(np.max(np.abs(w_back / W - 1.0)))
    print(f"assertion (ii): sigma_eff,all^2 <rho>_all / k_B vs W: max rel {max_ii:.2e}")
    if max_ii > W_IDENTITY_RTOL:
        raise RuntimeError(f"assertion (ii) FAILED: {max_ii:.3e} > {W_IDENTITY_RTOL:.0e}")

    lit = literature_conversions()

    # ---- save ----------------------------------------------------------------
    out = {
        "z_pc": z_bins, "bin_width_pc": np.array([PROFILE_BIN_PC]),
        "z_mid_pc": np.array([z_mid]),
        "slab_offsets_pc": np.array(SLAB_OFFSETS_PC), "slab_centers_pc": slab_centers,
        "slab_half_thickness_pc": np.array([SLAB_HALF_THICKNESS_PC]),
        "n_min_cells": np.array([N_MIN_CELLS]),
        "W": W[:n_bins], "slab__W": W[n_bins:],
        "ALL__sigma_eff": sig_all[:n_bins], "slab__ALL__sigma_eff": sig_all[n_bins:],
        "ALL__rho_mean": rho_all[:n_bins], "slab__ALL__rho_mean": rho_all[n_bins:],
        "ALL__n_cells": n_total[:n_bins], "slab__ALL__n_cells": n_total[n_bins:],
        "HIM__vol_frac": him_vol[:n_bins], "slab__HIM__vol_frac": him_vol[n_bins:],
        "assert_alpha_identity_max_rel": np.array([max_i]),
        "assert_alpha_identity_mid_rel": np.array([mid_i]),
        "assert_W_identity_max_rel": np.array([max_ii]),
        "check_W_vs_cached_max_rel": np.array([w_check]),
        "provisional": np.array([PROVISIONAL_CUBE_HEADER]),
        "self_gravity": np.array([SG]), "variant": np.array([VARIANT]),
    }
    for name in STAT_PHASES:
        for key in STAT_KEYS:
            out[f"{name}__{key}"] = stats[name][key][:n_bins]
            out[f"slab__{name}__{key}"] = stats[name][key][n_bins:]
    out["percentile_levels"] = np.array(percentile_levels(PERCENTILE_SCHEME_DEFAULT))
    for name in PCT_PHASES:
        for key, arr in pct[name].items():
            out[f"{name}__{key}"] = arr[:n_bins]
            out[f"slab__{name}__{key}"] = arr[n_bins:]
    for k, v in lit.items():
        out[f"ref__{k}"] = np.atleast_1d(np.asarray(v, dtype=float))
    OUT_NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ_PATH, **out)
    print(f"Saved {OUT_NPZ_PATH}")

    write_report(out, lit, w_check, max_i, mid_i, a_here[-1], a_cached[-1], max_ii,
                 int(fin.sum()), time.time() - t0)


# ---------------------------------------------------------------------------
def write_report(out, lit, w_check, max_i, mid_i, a_mid_here, a_mid_cached, max_ii,
                 n_fin, runtime):
    L = [
        PROVISIONAL_CUBE_HEADER,
        THERMAL_PRESSURE_HEADER,
        "Fig. 7: the missing support as VELOCITY",
        "=" * 100,
        f"RAW, self-gravity {SG} (fiducial), balanced midplane z_mid = "
        f"{float(out['z_mid_pc'][0]):+.3f} pc, footprint |x|,|y| <= "
        f"{STATS_BOX_XY_HALF_RANGE_PC:.0f} pc, box |z| <= {STATS_BOX_Z_HALF_RANGE_PC:.0f} pc.",
        "Phases: cached dP/dn flag (CNM = 0, UNM = 1, WNM = 2, HIM = 3); NEUTRAL = CNM+UNM+WNM.",
        "rho = 1.4 m_H n_H;  p_th_phys = 1.1 n_H k T;  all velocities 1D (vertical / line of sight).",
        "",
        "DEFINITIONS (per 4 pc bin or +-30 pc slab; cells of phase k in the group):",
        "  W               the group's area-averaged gas weight <P_tot>_area / k_B [K cm^-3], ONE",
        "                  number shared by every phase (mean of the cached per-plane values)",
        "  <rho>_k         plain (volume) mean of rho over phase k",
        "  sigma_eff,k     sqrt(k_B W / <rho>_k)",
        "  sigma_th,k      sqrt(sum(k_B p_th_phys) / sum(rho))   (mass-weighted c_s^2 = P_th/rho)",
        "  sigma_nt,k      sqrt(max(sigma_eff^2 - sigma_th^2, 0)); 'dsig2' = the unclipped",
        "                  sigma_eff^2 - sigma_th^2 [km^2 s^-2]",
        "  <M^2>_k         sum(rho (W/p_th_phys - 1)) / sum(rho): mass-weighted mean of the per-cell",
        "                  M^2 = alpha_cell - 1; cells with alpha_cell < 1 contribute negative M^2",
        "  sqrt<M^2>       where <M^2> > 0",
        "  f_m(a<1)        mass fraction of phase-k cells with alpha_cell = W/p_th_phys < 1",
        "  f_vol           cell (volume) fraction of the group",
        "  sigma_eff,all   sqrt(k_B W / <rho>_all) over every cell, HIM included",
        "All of alpha - 1 is assigned to motions, so these are UPPER LIMITS on turbulent",
        "velocities (magnetic and cosmic-ray support also contribute).",
        f"Phase statistics are NaN where the group holds fewer than {N_MIN_CELLS} cells of the phase.",
        "Slabs: |z - z_c| <= 30 pc at z_c = z_mid + {-300, -150, 0, +150, +300} pc.",
        "",
        "-" * 100,
        "ASSERTIONS (the script raises if any fails)",
        "-" * 100,
        f"  (i)  <M^2>_NEUTRAL + 1 == cached mass-weighted <alpha_cell> (area_alpha.npz,",
        f"       {SG}__alpha_cell_mean_mw and slab__{SG}__alpha_cell_mean_mw[mid]):",
        f"       max |rel diff| {max_i:.2e} over {n_fin} groups (all bins with neutral cells +",
        f"       the midplane slab); midplane slab {a_mid_here:.6f} vs {a_mid_cached:.6f}",
        f"       (rel {mid_i:.2e}); threshold {ALPHA_IDENTITY_RTOL:.0e}.  PASS",
        "       The cached +150/+300 slabs are centred on z = 150/300 pc, not z_mid +- 150/300,",
        "       so no cached value exists for the shifted slabs; they are not part of (i).",
        f"  (ii) sigma_eff,all^2 <rho>_all / k_B == W: max |rel diff| {max_ii:.2e}, threshold",
        f"       {W_IDENTITY_RTOL:.0e}.  PASS",
        f"  input checks: W per group vs cached Ptot_area (bins + midplane slab) max rel",
        f"       {w_check:.2e}; phase flag HIM == him flag cell for cell; group cell counts ==",
        "       Step 2e n_cells; no non-positive n_H or p_th in the box.",
        "",
        "-" * 100,
        "SLAB TABLE (+-30 pc slabs)",
        "-" * 100,
        f"{'phase':>8}{'sig_eff':>9}{'sig_th':>8}{'sig_nt':>8}{'dsig2':>9}{'<M^2>':>8}"
        f"{'sqrt':>7}{'f_m(a<1)':>10}{'f_vol':>8}{'<rho>/m_H':>11}{'cells':>10}",
        f"{'':>8}{'[km/s]':>9}{'[km/s]':>8}{'[km/s]':>8}{'[km2/s2]':>9}{'':>8}{'<M^2>':>7}"
        f"{'':>10}{'':>8}{'[cm^-3]':>11}",
    ]
    for i, (off, zc) in enumerate(zip(out["slab_offsets_pc"], out["slab_centers_pc"])):
        tag = "z_mid" if off == 0 else f"z_mid {off:+.0f}"
        L.append(f"{tag} (z_c = {zc:+.2f} pc): W = {out['slab__W'][i]:.1f} K cm^-3, "
                 f"HIM f_vol = {out['slab__HIM__vol_frac'][i]:.4f}")
        for name in STAT_PHASES:
            g = lambda q: out[f"slab__{name}__{q}"][i]  # noqa: E731
            L.append(f"{name:>8}{g('sigma_eff'):>9.3f}{g('sigma_th'):>8.3f}{g('sigma_nt'):>8.3f}"
                     f"{g('sigma_nt2_unclipped'):>9.2f}{g('M2'):>8.3f}{g('sqrt_M2'):>7.3f}"
                     f"{g('mass_frac_alpha_lt1'):>10.4f}{g('vol_frac'):>8.4f}"
                     f"{g('rho_mean') / M_H:>11.4f}{int(g('n_cells')):>10d}")
        L.append(f"{'ALL':>8}{out['slab__ALL__sigma_eff'][i]:>9.3f}{'':>8}{'':>8}{'':>9}{'':>8}"
                 f"{'':>7}{'':>10}{1.0:>8.4f}{out['slab__ALL__rho_mean'][i] / M_H:>11.4f}"
                 f"{int(out['slab__ALL__n_cells'][i]):>10d}")
        L.append("")
    pl, ph_ = out["percentile_levels"]
    i_mid = int(np.where(out["slab_offsets_pc"] == 0.0)[0][0])
    L += [
        "  (<rho>/m_H includes the factor 1.4: it is 1.4 <n_H>.)",
        "",
        "-" * 100,
        f"PER-CELL DISTRIBUTIONS, midplane slab: MASS-weighted (w = rho) p{pl:g} / p{ph_:g}",
        "-" * 100,
        "  sigma_eff,cell = sqrt(k_B W / rho), c_s,cell = sqrt(k_B p_th_phys / rho) [km/s],",
        "  M2_cell = W / p_th_phys - 1; nearest rank on the mass-weighted cumulative distribution",
        "  (src.physics.stats.weighted_stats). The group mean is shown alongside: sigma_eff and",
        "  sigma_th are the RMS (mass-weighted) of the per-cell values, <M^2> their mass-weighted",
        "  mean. All bins and slabs are in the npz.",
        f"{'phase':>8}{'sig_eff':>9}{f'[p{pl:g}, p{ph_:g}]':>18}{'sig_th':>9}"
        f"{f'c_s [p{pl:g}, p{ph_:g}]':>20}{'<M^2>':>8}{f'M2 [p{pl:g}, p{ph_:g}]':>20}",
    ]
    for name in PCT_PHASES:
        g = lambda q: out[f"slab__{name}__{q}"][i_mid]  # noqa: E731
        L.append(f"{name:>8}{g('sigma_eff'):>9.3f}   [{g('sigma_eff_cell_p_lo'):>6.3f}, "
                 f"{g('sigma_eff_cell_p_hi'):>6.3f}]{g('sigma_th'):>9.3f}      "
                 f"[{g('cs_cell_p_lo'):>6.3f}, {g('cs_cell_p_hi'):>6.3f}]{g('M2'):>8.3f}      "
                 f"[{g('M2_cell_p_lo'):>6.3f}, {g('M2_cell_p_hi'):>6.3f}]")
    L += [
        "",
        "-" * 100,
        "LITERATURE REFERENCE VALUES, converted to our 1D isothermal convention",
        "-" * 100,
        "  Our convention: M^2 = sigma_turb,1D^2 / c_s^2, c_s^2 = P_th/rho = 1.1 kT / (1.4 m_H).",
        "  CNM, Heiles & Troland 2003: M_t ~ 3, 3D, with C_s^2 = kT / (1.4 m_H).",
        f"    M^2 (1D) = M_t^2 / 3                = {HT03_MT:g}^2 / 3       = {lit['HT03_M2']:.3f}",
        f"    with our 1.1 in P_th: M_t^2/(3*1.1) = {HT03_MT:g}^2 / 3.3     = "
        f"{lit['HT03_M2_corrected']:.3f}",
        "  WNM, Marchal & Miville-Deschenes 2021: line-of-sight sigma_total = "
        f"{MM21_SIGMA_TOTAL[0]} +- {MM21_SIGMA_TOTAL[1]},",
        f"    sigma_th = {MM21_SIGMA_TH[0]} +- {MM21_SIGMA_TH[1]}, sigma_turb = "
        f"{MM21_SIGMA_TURB[0]} +- {MM21_SIGMA_TURB[1]} km/s (used as-is: already 1D);",
        f"    Ms = {MM21_MS[0]} +- {MM21_MS[1]}, 3D, ADIABATIC sound speed (gamma = 5/3).",
        "    M^2 (1D isothermal) = Ms^2 * gamma / 3 = Ms^2 * 5/9:",
        f"      central (Ms = {MM21_MS[0]:.2f}) {lit['MM21_M2']:.3f};  range Ms = "
        f"{MM21_MS[0] - MM21_MS[1]:.2f} .. {MM21_MS[0] + MM21_MS[1]:.2f} -> "
        f"{lit['MM21_M2_lo']:.3f} .. {lit['MM21_M2_hi']:.3f}",
        "",
        f"Runtime: {runtime:.1f} s",
    ]
    OUT_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT_PATH.write_text("\n".join(L) + "\n")
    print(f"Wrote {OUT_TXT_PATH}")


if __name__ == "__main__":
    main()
