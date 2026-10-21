"""
Fig. 3, "Building P_tot": the 1-D weight budget of the area-averaged total
pressure, split into stars + dark matter + gas self-gravity.

Why 1-D is exact: in the fiducial setting g(z) is the same for every (x,y)
column (Guo+20 g_ext + footprint-mean gas self-gravity, src.physics.gravity),
and P_tot is linear in rho, so <P_tot>_area(z) is the hydrostatic integral of
the footprint-mean density <rho>_area(z) -- validated to ~1e-10 in
results/area_alpha.txt section 1. And P_tot is linear in g, so it splits
exactly into the three gravity components.

Fiducial: RAW, ALL footprint cells (HIM-flagged included -- their mass is in
the weight), |x|,|y| <= 500 pc, self-gravity footprint_mean, and the BALANCED
midplane (Step 1i, src.physics.midplane): P_tot is integrated toward z_mid from
both cube edges with g centred on z_mid, so <P_tot>_area is continuous there.

Route (the same as scripts/compute/compute_area_alpha.py, reused not copied):
  * z grid: the SOURCE cube's float64 z_pc (open_zarr), never the float32
    copy in alpha_core.zarr -- see area_alpha.txt section 1 for why;
  * <n_H>_area(z): compute_area_alpha.footprint_mean_density_chunked over the
    cached RAW n_model (the footprint is already the +-500 pc square);
  * every gravity term and every pressure integral through src.physics, with
    z_mid from src.physics.midplane (the pipeline's own); this script's older
    1-D root (find_zmid, below) is kept as an independent cross-check.
The Sun-centred P(0+) / P(0-) jump is recomputed alongside, for comparison.

Writes cache/ptot_budget/ptot_budget.npz and results/ptot_budget.txt.
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import zarr
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.conventions import (  # noqa: E402
    K_B,
    M_SUN_G,
    PC_CM,
    PROVISIONAL_CUBE_HEADER,
    RHO_DM_MSUN_PC3,
    SELF_GRAVITY_MODE_FOOTPRINT_MEAN,
    SIGMA_STAR_MSUN_PC2,
    STATS_BOX_XY_HALF_RANGE_PC,
    STATS_BOX_Z_HALF_RANGE_PC,
    Z_H_PC,
)
from src.conventions import MIDPLANE_MODE, MIDPLANE_MODE_BALANCED  # noqa: E402
from src.physics import gravity, hydrostatic, midplane  # noqa: E402
from src.physics.loading import open_zarr  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════
# ██  CONTROL BLOCK
# ═══════════════════════════════════════════════════════════════════════
VARIANT = "RAW"
ALPHA_CORE_ZARR_PATH = Path("cache/core/alpha_core.zarr")
AREA_ALPHA_NPZ_PATH = Path("cache/core/area_alpha.npz")
OUT_NPZ_PATH = Path("cache/ptot_budget/ptot_budget.npz")
OUT_TXT_PATH = Path("results/ptot_budget.txt")
REPORT_Z_PC = (-300.0, -150.0, 150.0, 300.0)     # plus z_mid-, z_mid+ and the cube edges
ZMID_CROSSCHECK_PC = 0.05     # official z_mid vs this script's own 1-D root
G_SPLIT_RTOL = 1e-12          # stars + DM == g_ext, components == g_total
P_CACHE_RTOL = 1e-6           # P_sum vs cached <P_tot>_area, |z| <= 400 pc
CLOSED_FORM_RTOL = 1e-2       # self-gravity closed form: discretisation level
ZMID_BRACKET_PC = 100.0       # brentq bracket +- this
ZMID_XTOL_PC = 1e-6
NUMBERS_TABLE_SIGMA_MEDIAN = 4.80595   # numbers_table.txt, Sigma_gas vol median

COMPONENTS = ("star", "dm", "gas")

_SPEC = importlib.util.spec_from_file_location(
    "compute_area_alpha", Path(__file__).resolve().parent / "compute_area_alpha.py")
caa = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(caa)


# ═══════════════════════════════════════════════════════════════════════
# ██  PHYSICS HELPERS (thin wrappers over src.physics; reused by the tests)
# ═══════════════════════════════════════════════════════════════════════
def sigma_components(z_pc, n_bar_cm3, z_ref_pc=0.0):
    """{star, dm, gas}: enclosed column Sigma_enc [Msun/pc^2] about z_ref.

    Stars and DM are Guo+20's g_ext terms written as an enclosed column
    (their sum times GRAV_PREFACTOR_CGS is gravity.g_ext_cgs). Gas is the
    enclosed column of the 1-D profile: gravity.sigma_gas_enclosed about
    z = 0, or -- only for the z_mid diagnostic, z_ref != 0 -- the column
    within |z' - z_ref| < |z - z_ref| from the same cumulative integral.
    """
    s = np.abs(np.asarray(z_pc, dtype=float) - z_ref_pc)
    star = SIGMA_STAR_MSUN_PC2 * (1.0 - np.exp(-s / Z_H_PC))
    dm = 2.0 * RHO_DM_MSUN_PC3 * s
    if z_ref_pc == 0.0:
        gas = gravity.sigma_gas_enclosed(z_pc, n_bar_cm3)
    else:
        rho = n_bar_cm3 * gravity.RHO_MSUN_PC3_PER_N
        cum = cumulative_trapezoid(rho, x=z_pc, initial=0.0)   # z_pc ascending
        hi = np.interp(np.clip(z_ref_pc + s, z_pc[0], z_pc[-1]), z_pc, cum)
        lo = np.interp(np.clip(z_ref_pc - s, z_pc[0], z_pc[-1]), z_pc, cum)
        gas = hi - lo
    return {"star": star, "dm": dm, "gas": gas}


def g_from_sigma(sigma):
    return gravity.GRAV_PREFACTOR_CGS * np.asarray(sigma)


def pressure_branches(z_pc, rho_cgs, g_cgs, z_split=0.0):
    """(P top-down at z_split, P bottom-up at z_split) [K cm^-3].

    hydrostatic.integrate_ptot_halfcolumn on each half, from its own domain
    edge to the split. z_pc ascending. For z_split between grid points the
    split point is inserted with linearly interpolated rho and g.
    """
    z = np.asarray(z_pc, dtype=float)
    rho = np.asarray(rho_cgs, dtype=float)
    g = np.asarray(g_cgs, dtype=float)
    if not np.any(np.isclose(z, z_split, atol=1e-9, rtol=0)):
        k = int(np.searchsorted(z, z_split))
        z = np.insert(z, k, z_split)
        rho = np.insert(rho, k, np.interp(z_split, z_pc, rho_cgs))
        g = np.insert(g, k, np.interp(z_split, z_pc, g_cgs))
    k0 = int(np.argmin(np.abs(z - z_split)))
    top = hydrostatic.integrate_ptot_halfcolumn(z[k0:][::-1], rho[k0:][::-1], g[k0:][::-1])[-1]
    bot = hydrostatic.integrate_ptot_halfcolumn(z[:k0 + 1], rho[:k0 + 1], g[:k0 + 1])[-1]
    return float(top), float(bot)


def closed_form_sg_kb(sigma_total_msun_pc2):
    """pi G Sigma^2 / 2 / k_B [K cm^-3] -- the mean of the two gas-self-gravity
    midplane branches for ANY profile on a domain symmetric about z = 0."""
    sigma_cgs = sigma_total_msun_pc2 * M_SUN_G / PC_CM ** 2
    return 0.25 * gravity.GRAV_PREFACTOR_CGS * sigma_total_msun_pc2 * sigma_cgs / K_B


def weight_imbalance(z_m, z_pc, n_bar_cm3, self_gravity=True):
    """(top-down - bottom-up) weight [K cm^-3] for gravity centred on z_m."""
    sig = sigma_components(z_pc, n_bar_cm3, z_ref_pc=z_m)
    tot = sig["star"] + sig["dm"] + (sig["gas"] if self_gravity else 0.0)
    rho = gravity.RHO_PER_N * n_bar_cm3
    top, bot = pressure_branches(z_pc, rho, g_from_sigma(tot), z_split=z_m)
    return top - bot


def find_zmid(z_pc, n_bar_cm3, self_gravity=True, bracket=ZMID_BRACKET_PC):
    """Balanced midplane (DIAGNOSTIC): z_m where top-down == bottom-up weight."""
    return brentq(weight_imbalance, -bracket, bracket,
                  args=(z_pc, n_bar_cm3, self_gravity), xtol=ZMID_XTOL_PC)


def column_balance_height(z_pc, n_bar_cm3):
    """z where the gas column above equals the column below."""
    cum = cumulative_trapezoid(n_bar_cm3 * gravity.RHO_MSUN_PC3_PER_N, x=z_pc, initial=0.0)
    return float(np.interp(0.5 * cum[-1], cum, z_pc))


# ═══════════════════════════════════════════════════════════════════════
# ██  MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    checks = []          # (name, measured, threshold)

    def check(name, measured, threshold):
        checks.append((name, measured, threshold))
        print(f"  check {name}: {measured:.3e} (threshold {threshold:.0e})")
        if not (np.isfinite(measured) and measured <= threshold):
            raise RuntimeError(f"ASSERTION FAILED -- {name}: {measured:.3e} > {threshold:.0e}. "
                               "STOPPING (nothing tuned).")

    # ---- a. z grid and <n_H>_area ----------------------------------------
    z = np.asarray(open_zarr().z_pc, dtype=np.float64)        # SOURCE float64 grid
    if np.any(np.diff(z) <= 0):
        raise RuntimeError("source z grid is not ascending")
    core = zarr.open_group(str(ALPHA_CORE_ZARR_PATH), mode="r")
    vg = core[VARIANT]
    n_arr = vg["n_model"]
    if n_arr.shape[0] != z.size:
        raise RuntimeError("cached n_model z length != source z grid")
    nH = caa.footprint_mean_density_chunked(n_arr)            # the area-alpha route
    nan_frac = np.empty(z.size)
    for lo in range(0, z.size, caa.Z_CHUNK):
        hi = min(lo + caa.Z_CHUNK, z.size)
        c = np.asarray(n_arr[lo:hi], dtype=np.float64).reshape(hi - lo, -1)
        nan_frac[lo:hi] = np.mean(~np.isfinite(c), axis=1)
    rho = gravity.RHO_PER_N * nH                               # g cm^-3 (MU * M_H * nH)
    rho_msun = nH * gravity.RHO_MSUN_PC3_PER_N
    aa = np.load(AREA_ALPHA_NPZ_PATH, allow_pickle=False)
    if not np.array_equal(aa["z_pc_full"], z):
        raise RuntimeError("area_alpha.npz z grid differs from the source grid")
    check("<n_H>_area vs area_alpha.npz n_bar_cm3_full (max rel)",
          float(np.max(np.abs(nH / aa["n_bar_cm3_full"] - 1))), 1e-15)
    print(f"z grid {z[0]:g}..{z[-1]:g} pc ({z.size} planes); max NaN fraction per plane "
          f"{nan_frac.max():.2e}")

    # ---- the midplane: official (src.physics.midplane) vs this script's 1-D root
    if MIDPLANE_MODE != MIDPLANE_MODE_BALANCED:
        raise RuntimeError("Fig. 3 is built for the balanced midplane (MIDPLANE_MODE)")
    zm = {"on": midplane.find_balanced_midplane(z, nH, self_gravity=True),
          "off": midplane.find_balanced_midplane(z, nH, self_gravity=False)}
    zm_diag = {"on": find_zmid(z, nH, self_gravity=True), "off": find_zmid(z, nH, self_gravity=False)}
    for k, sg in (("on", "mean"), ("off", "off")):
        check(f"official z_mid ({k}) vs this script's own 1-D root [pc]",
              abs(zm[k] - zm_diag[k]), ZMID_CROSSCHECK_PC)
        check(f"official z_mid ({k}) vs the cache's z_mid (build) [pc]",
              abs(zm[k] - float(vg[f"self_gravity_{sg}"].attrs["z_mid_pc"])), 1e-6)
    z_mid = zm["on"]
    z_colbal = midplane.column_balance_height(z, nH)

    # ---- b. gravity components, centred on z_mid -------------------------------
    s = np.abs(z - z_mid)
    sig = {"star": SIGMA_STAR_MSUN_PC2 * (1.0 - np.exp(-s / Z_H_PC)),
           "dm": 2.0 * RHO_DM_MSUN_PC3 * s,
           "gas": midplane.enclosed_column_about(z, rho_msun, z_mid)}
    sig["total"] = sig["star"] + sig["dm"] + sig["gas"]
    g = {k: g_from_sigma(v) for k, v in sig.items()}
    g_ext = gravity.g_ext_cgs(z - z_mid)
    nz = g_ext > 0
    check("g_star + g_dm == gravity.g_ext_cgs(z - z_mid) (max rel)",
          float(np.max(np.abs((g["star"] + g["dm"])[nz] / g_ext[nz] - 1))), G_SPLIT_RTOL)
    g_ref = midplane.centred_g_cgs(z, nH, z_mid, self_gravity=True)
    nz = g_ref > 0
    check("g_star + g_dm + g_gas == midplane.centred_g_cgs (max rel)",
          float(np.max(np.abs(g["total"][nz] / g_ref[nz] - 1))), G_SPLIT_RTOL)
    g_caa, _, zm_caa = caa.g_for_setting("mean", z, nH)
    check("g_total == compute_area_alpha.g_for_setting('mean') (max rel)",
          float(np.max(np.abs(g["total"][nz] / g_caa[nz] - 1))), G_SPLIT_RTOL)

    # ---- c. pressure components, split at z_mid -------------------------------------
    P = {k: hydrostatic.p_tot_kb_full_column(z, rho, g[k], z_split=z_mid) for k in COMPONENTS}
    P["sum"] = P["star"] + P["dm"] + P["gas"]
    g_off = gravity.g_ext_cgs(z - zm["off"])                     # off: its OWN z_mid
    P["off"] = hydrostatic.p_tot_kb_full_column(z, rho, g_off, z_split=zm["off"])
    box = np.abs(z) <= STATS_BOX_Z_HALF_RANGE_PC
    ptot_on, ptot_off = aa["mean__ptot_plane_mean_full"], aa["off__ptot_plane_mean_full"]
    check("P_star+P_dm+P_gas vs cached <P_tot>_area (sg mean), |z|<=400 (max rel)",
          float(np.max(np.abs(P["sum"][box] / ptot_on[box] - 1))), P_CACHE_RTOL)
    check("P (sg off, centred on its own z_mid) vs cached <P_tot>_area (sg off), |z|<=400 (max rel)",
          float(np.max(np.abs(P["off"][box] / ptot_off[box] - 1))), P_CACHE_RTOL)
    p_direct = hydrostatic.p_tot_kb_full_column(z, rho, g["total"], z_split=z_mid)
    check("sum of component P == P from summed g (max rel, linearity)",
          float(np.nanmax(np.abs(P["sum"][p_direct > 0] / p_direct[p_direct > 0] - 1))), 1e-12)

    # ---- d. both branches AT z_mid -----------------------------------------------------
    br = {k: midplane.branch_pressures(z, rho, g[k], z_mid) for k in COMPONENTS + ("total",)}
    check("total P at z_mid: top-down vs bottom-up (rel) -- continuity",
          abs(br["total"][0] / br["total"][1] - 1), 1e-6)
    check("sum of component branches == total branch, top and bottom (max rel)",
          max(abs(sum(br[k][j] for k in COMPONENTS) / br["total"][j] - 1) for j in (0, 1)), 1e-12)
    P_zmid = 0.5 * (br["total"][0] + br["total"][1])
    comp_mid = {k: 0.5 * (br[k][0] + br[k][1]) for k in COMPONENTS}   # each component's mean branch
    shares = {k: comp_mid[k] / P_zmid for k in COMPONENTS}
    # the old Sun-centred jump, recomputed live for comparison
    sig_sun = sigma_components(z, nH)
    g_sun = g_from_sigma(sig_sun["star"] + sig_sun["dm"] + sig_sun["gas"])
    sun_top, sun_bot = pressure_branches(z, rho, g_sun)

    # ---- e. closed-form self-gravity check (about z_mid) ---------------------------------
    sigma_tot = float(np.trapezoid(rho_msun, z))
    sg_mean = 0.5 * (br["gas"][0] + br["gas"][1])
    sg_closed = closed_form_sg_kb(sigma_tot)
    rel_closed = abs(sg_mean / sg_closed - 1)
    check("0.5[P_gas(z_mid+)+P_gas(z_mid-)] vs pi G Sigma^2/2 (rel)", rel_closed, CLOSED_FORM_RTOL)
    sig_map = np.asarray(vg["Sigma_gas"][:], dtype=np.float64)
    sig_map_mean, sig_map_median = float(np.nanmean(sig_map)), float(np.nanmedian(sig_map))
    rel_mean_vs_total = abs(sig_map_mean / sigma_tot - 1)

    # ---- f. column balance about z_mid and about the Sun ------------------------------------
    cum_at = lambda x: float(np.trapezoid(np.interp(np.r_[z[z < x], x], z, rho_msun),  # noqa: E731
                                          np.r_[z[z < x], x]))
    sigma_below = cum_at(z_mid)
    sigma_above = sigma_tot - sigma_below
    pos0, neg0 = z >= 0, z <= 0
    sigma_above0 = float(np.trapezoid(rho_msun[pos0], z[pos0]))
    sigma_below0 = float(np.trapezoid(rho_msun[neg0], z[neg0]))

    # ---- h. the |z| > 400 pc contribution -------------------------------------------------------
    i400p = int(np.argmin(np.abs(z - STATS_BOX_Z_HALF_RANGE_PC)))
    i400m = int(np.argmin(np.abs(z + STATS_BOX_Z_HALF_RANGE_PC)))
    f_out_top = float(P["sum"][i400p] / P_zmid)
    f_out_bot = float(P["sum"][i400m] / P_zmid)

    # ---- save ---------------------------------------------------------------------------
    out = {
        "z_pc": z, "nH_area": nH, "rho_area": rho, "nan_fraction": nan_frac,
        "P_sum": P["sum"], "P_off": P["off"],
        "ptot_area_cached_on": ptot_on, "ptot_area_cached_off": ptot_off,
        "midplane_mode": np.array([MIDPLANE_MODE]),
        "z_mid": np.array([z_mid]), "z_mid_off": np.array([zm["off"]]),
        "z_mid_diag_on": np.array([zm_diag["on"]]), "z_mid_diag_off": np.array([zm_diag["off"]]),
        "z_colbal": np.array([z_colbal]),
        "P_zmid": np.array([P_zmid]),
        "shares_zmid": np.array([shares[k] for k in COMPONENTS]),
        "P_sun_branches": np.array([sun_top, sun_bot]),
        "sigma_gas_total": np.array([sigma_tot]),
        "sigma_above": np.array([sigma_above]), "sigma_below": np.array([sigma_below]),
        "sigma_above_sun": np.array([sigma_above0]), "sigma_below_sun": np.array([sigma_below0]),
        "sigma_map_mean": np.array([sig_map_mean]), "sigma_map_median": np.array([sig_map_median]),
        "sg_branch_mean": np.array([sg_mean]), "sg_closed_form": np.array([sg_closed]),
        "f_ptot_zmid_from_above_400": np.array([f_out_top]),
        "f_ptot_zmid_from_below_400": np.array([f_out_bot]),
        "provisional": np.array([PROVISIONAL_CUBE_HEADER]),
        "variant": np.array([VARIANT]), "self_gravity_mode": np.array([SELF_GRAVITY_MODE_FOOTPRINT_MEAN]),
    }
    for k in COMPONENTS + ("total",):
        out[f"sigma_enc_{k}"] = sig[k]
        out[f"g_{k}"] = g[k]
        out[f"P0_{k}"] = np.array(br[k])                       # [top-down, bottom-up] AT z_mid
    for k in COMPONENTS:
        out[f"P_{k}"] = P[k]
    OUT_NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ_PATH, **out)
    print(f"Saved {OUT_NPZ_PATH}")

    # ---- report -----------------------------------------------------------------------
    def at(arr, zz):
        return float(np.interp(zz, z, arr))
    L = [
        PROVISIONAL_CUBE_HEADER,
        "Fig. 3 'Building P_tot': the 1-D weight budget of <P_tot>_area(z)",
        "=" * 100,
        f"RAW, ALL footprint cells (HIM included), |x|,|y| <= {STATS_BOX_XY_HALF_RANGE_PC:.0f} pc, "
        "self-gravity footprint_mean,",
        f"BALANCED midplane (Step 1i): z_mid = {z_mid:+.4f} pc; P_tot integrated toward z_mid from",
        "both cube edges with g centred on z_mid. g(z) is the same for every column, so",
        "<P_tot>_area(z) = int <rho>_area g dz exactly, and it splits exactly into stars + dark",
        "matter + gas self-gravity (P is linear in g).",
        f"z grid: the SOURCE cube's float64 z_pc, {z[0]:g} .. {z[-1]:g} pc, {z.size} planes "
        f"(spacing {z[1] - z[0]:g} pc).",
        f"<n_H>_area: compute_area_alpha.footprint_mean_density_chunked over the cached RAW n_model;",
        f"  NaN fraction per plane: max {nan_frac.max():.2e} (mean {nan_frac.mean():.2e}).",
        f"Guo+20: Sigma_star = {SIGMA_STAR_MSUN_PC2} Msun/pc^2, z_h = {Z_H_PC} pc, "
        f"rho_DM = {RHO_DM_MSUN_PC3} Msun/pc^3.",
        "",
        "-" * 100,
        "ASSERTIONS (the script raises if any fails)",
        "-" * 100,
    ]
    for name, meas, thr in checks:
        L.append(f"  {meas:>10.2e}  (<= {thr:.0e})  {name}")
    L += [
        "",
        "-" * 100,
        "THE MIDPLANE",
        "-" * 100,
        "  z_mid: where the area-averaged weight above equals the weight below, g centred on z_mid",
        "  (src.physics.midplane.find_balanced_midplane); the cross-check is this script's own 1-D",
        "  root (approximate enclosed column at the off-grid end).",
        f"    self-gravity on   z_mid = {zm['on']:+.4f} pc   (1-D cross-check {zm_diag['on']:+.4f})",
        f"    self-gravity off  z_mid = {zm['off']:+.4f} pc   (1-D cross-check {zm_diag['off']:+.4f})",
        f"  column-balance height (equal gas column above and below): {z_colbal:+.4f} pc",
        "",
        "-" * 100,
        "PROFILES  (z_mid-/z_mid+ = the bottom-up / top-down branch AT z_mid; they meet)",
        "-" * 100,
        f"{'z':>8}{'<n_H>':>10}{'Sig_star':>10}{'Sig_DM':>9}{'Sig_gas':>9}"
        f"{'P_star':>10}{'P_DM':>9}{'P_gas':>9}{'P_sum':>10}",
        f"{'[pc]':>8}{'[cm^-3]':>10}{'---- [Msun/pc^2] ----':>28}{'---- [K cm^-3] ----':>38}",
    ]
    rows = [(f"{z[0]:g}", z[0], None)] + [(f"{zz:g}", zz, None) for zz in REPORT_Z_PC[:2]]
    rows += [("z_mid-", z_mid, 1), ("z_mid+", z_mid, 0)]
    rows += [(f"{zz:g}", zz, None) for zz in REPORT_Z_PC[2:]] + [(f"{z[-1]:g}", z[-1], None)]
    for lab, zz, branch in rows:
        pv = [at(P[k], zz) for k in COMPONENTS] if branch is None else [br[k][branch] for k in COMPONENTS]
        L.append(f"{lab:>8}{at(nH, zz):>10.4f}{at(sig['star'], zz):>10.3f}{at(sig['dm'], zz):>9.3f}"
                 f"{at(sig['gas'], zz):>9.3f}{pv[0]:>10.1f}{pv[1]:>9.1f}{pv[2]:>9.1f}{sum(pv):>10.1f}")
    L += [
        "",
        "-" * 100,
        "MIDPLANE VALUE AND SHARES",
        "-" * 100,
        f"  P(z_mid) = {P_zmid:.2f} K cm^-3  (top-down {br['total'][0]:.4f}, bottom-up {br['total'][1]:.4f})",
        "  shares (each component's mean of its two branches; the components are NOT individually",
        "  balanced, only their sum is):",
    ]
    for k, lab in (("star", "stars"), ("dm", "dark matter"), ("gas", "gas self-gravity")):
        L.append(f"    {lab:<18}{comp_mid[k]:>9.1f}  ({shares[k]:.1%})   branches top {br[k][0]:.1f} / "
                 f"bottom {br[k][1]:.1f}")
    L += [
        "",
        f"  Pre-Step-1i (Sun-centred, recomputed here): P(0+) = {sun_top:.1f}, P(0-) = {sun_bot:.1f} "
        f"K cm^-3 -- the jump",
        f"  the balanced midplane removes. P(z_mid) is {P_zmid / sun_top - 1:+.1%} vs P(0+) and "
        f"{P_zmid / sun_bot - 1:+.1%} vs P(0-).",
        "",
        f"  Share of P(z_mid) from gas at z > +{STATS_BOX_Z_HALF_RANGE_PC:.0f} pc "
        f"(= P_sum(+400) / P(z_mid)): {f_out_top:.4f}",
        f"  Share of P(z_mid) from gas at z < -{STATS_BOX_Z_HALF_RANGE_PC:.0f} pc "
        f"(= P_sum(-400) / P(z_mid)): {f_out_bot:.4f}",
        f"  (outside the stats box, inside the cube, which spans z = {z[0]:g} .. {z[-1]:g} pc)",
        "",
        "-" * 100,
        "CLOSED-FORM SELF-GRAVITY CHECK (about z_mid)",
        "-" * 100,
        "  For ANY profile and ANY centre: 0.5 [P_gas top + P_gas bottom] = pi G Sigma^2 / 2.",
        f"  Sigma_gas_total (full-cube column of <rho>_area) = {sigma_tot:.4f} Msun/pc^2",
        f"  0.5 [P_gas(z_mid+) + P_gas(z_mid-)] = {sg_mean:.2f} K cm^-3",
        f"  pi G Sigma^2 / 2 / k_B             = {sg_closed:.2f} K cm^-3",
        f"  relative difference                = {rel_closed:.2e}   (threshold {CLOSED_FORM_RTOL:.0e})",
        "",
        "  Per-column Sigma_gas map over the footprint (alpha_core.zarr RAW/Sigma_gas):",
        f"    mean   {sig_map_mean:.4f} Msun/pc^2  (vs Sigma_gas_total {sigma_tot:.4f}: rel "
        f"{rel_mean_vs_total:.1e})",
        f"    median {sig_map_median:.4f} Msun/pc^2  (numbers_table.txt: {NUMBERS_TABLE_SIGMA_MEDIAN})",
        "",
        "-" * 100,
        "COLUMN BALANCE",
        "-" * 100,
        f"  about z_mid: above {sigma_above:.4f}, below {sigma_below:.4f} Msun/pc^2, "
        f"ratio {sigma_above / sigma_below:.4f}",
        f"  about z = 0: above {sigma_above0:.4f}, below {sigma_below0:.4f} Msun/pc^2, "
        f"ratio {sigma_above0 / sigma_below0:.4f}",
        "  (the weights, not the columns, balance at z_mid: g grows away from the plane, so gas",
        "   further out weighs more)",
        "",
        f"(runtime {time.time() - t0:.1f}s)",
    ]
    OUT_TXT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
