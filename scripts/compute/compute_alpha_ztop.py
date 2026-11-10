"""
compute_alpha_ztop -- DIAGNOSTIC: does the SHAPE of alpha(z) (Fig. 4)
depend on the integration height z_top?

For z_top = 400, 500, 600, 750 pc the density is set to zero at |z| > z_top
(cube only, rho_bar = the footprint-mean RAW density), and the weight W(z)
is recomputed exactly as compute_zmax does: gas self-gravity from the
truncated profile, Guo+20 stars and DM, the balanced midplane z_mid re-found
for each z_top (compute_zmax.balanced_P -> compute_area_alpha.g_for_setting
+ hydrostatic.p_tot_kb_full_column split at z_mid).

Per Fig. 4 bin (compute_area_alpha / pipeline.compute_all.profile_bin_groups:
4 pc bins in SIGNED z over the stats box |z| <= 400 pc, the bins Fig. 4's
cache is on), the headline statistics come from
compute_area_alpha.cell_alpha_stats -- the function Fig. 4's numbers come
from -- over the bin's non-HIM cells, volume-weighted, with the bin's
<W>_area = mean of W over the bin's planes:
    <alpha_cell> = <W> <1/P_th>, median alpha_cell, 15-85% range.
<1/P_th>^-1 per bin does not depend on z_top (stored once).
Midplane slab (|z - z_mid| <= 30 pc, z_mid per z_top): compute_zmax's
slab_pth / balanced-slab definition.

Checks (raise on failure):
  * z_top = 750: <alpha_cell>, median, p15, p85, <1/P_th>^-1 and <W> per bin
    vs cache/core/area_alpha.npz (Fig. 4), 1e-6 relative in every bin;
  * slab <W>, <alpha_cell> and z_mid for all four z_top vs cache/zmax/zmax.npz
    (results/zmax.txt), 1e-6 relative.

Writes cache/alpha_ztop/alpha_ztop.npz and results/alpha_ztop.txt.
Measures only: nothing existing is changed.
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.conventions import (  # noqa: E402
    PERCENTILE_SCHEME_DEFAULT,
    PROVISIONAL_CUBE_HEADER,
    STATS_BOX_Z_HALF_RANGE_PC,
)
from src.physics.loading import open_zarr  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════
# ██  CONTROL BLOCK
# ═══════════════════════════════════════════════════════════════════════
VARIANT = "RAW"
SG = "mean"                                   # footprint-mean self-gravity (fiducial)
ALPHA_CORE_ZARR_PATH = Path("cache/core/alpha_core.zarr")
AREA_ALPHA_NPZ_PATH = Path("cache/core/area_alpha.npz")
ZMAX_NPZ_PATH = Path("cache/zmax/zmax.npz")
OUT_NPZ_PATH = Path("cache/alpha_ztop/alpha_ztop.npz")
OUT_TXT_PATH = Path("results/alpha_ztop.txt")
Z_TOPS_PC = (400.0, 500.0, 600.0, 750.0)
FIDUCIAL_Z_TOP_PC = 750.0
TABLE_DZ_PC = (50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0)
SLAB_HALF_PC = 30.0
REPRO_RTOL = 1e-6
STAT_KEYS = ("alpha_cell_mean", "alpha_cell_median", "alpha_cell_p_lo", "alpha_cell_p_hi")

_ROOT = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cz = _load("compute_zmax", "scripts/compute/compute_zmax.py")
caa = cz.caa


def main():
    t0 = time.time()
    checks = []

    def check(name, measured, threshold):
        checks.append((name, measured, threshold))
        print(f"  check {name}: {measured:.3e} (<= {threshold:.0e})")
        if not (np.isfinite(measured) and measured <= threshold):
            raise RuntimeError(f"ASSERTION FAILED -- {name}: {measured:.3e}. STOPPING.")

    import pipeline.compute_all as pipe

    aa = np.load(AREA_ALPHA_NPZ_PATH, allow_pickle=False)
    zc = np.load(ZMAX_NPZ_PATH, allow_pickle=False)
    z = np.asarray(open_zarr().z_pc, dtype=np.float64)        # SOURCE float64 grid
    core = zarr.open_group(str(ALPHA_CORE_ZARR_PATH), mode="r")
    vg = core[VARIANT]
    nH = np.asarray(aa["n_bar_cm3_full"], dtype=np.float64)    # Fig. 4's self-gravity source
    if not np.array_equal(aa["z_pc_full"], z):
        raise RuntimeError("area_alpha z grid differs from the source grid")

    # ---- Fig. 4's bins and the box cells -------------------------------------
    zsl, z_box = pipe._box_z_slice(z)
    bin_groups = pipe.profile_bin_groups(z_box, use_abs=False)
    zb = np.array([c for c, _ in bin_groups], dtype=float)
    if not np.allclose(zb, aa["z_pc"]):
        raise RuntimeError("bin centres differ from area_alpha's")
    print("reading the box's p_th and HIM flag ...")
    pth_box = np.asarray(vg["p_th_phys"][zsl], dtype=np.float32).reshape(len(z_box), -1)
    him_box = np.asarray(vg["him"][zsl]).astype(bool).reshape(len(z_box), -1)
    cells = []
    for _, idxs in bin_groups:
        idxs = np.asarray(idxs, dtype=int)
        cells.append(pth_box[idxs][~him_box[idxs]].astype(np.float64))
    del pth_box, him_box
    ps = cz.pth_plane_sums(core, z)                            # for the slab

    # ---- per z_top --------------------------------------------------------------
    no_tail = np.zeros(z.size, dtype=bool)
    out = {"z_pc_full": z, "bin_z_pc": zb, "bin_edges": aa["bin_edges"],
           "z_tops_pc": np.array(Z_TOPS_PC), "provisional": np.array([PROVISIONAL_CUBE_HEADER]),
           "percentile_levels": aa["percentile_levels"]}
    pth_harm = np.full(zb.size, np.nan)
    R = {}
    for zt in Z_TOPS_PC:
        print(f"\nz_top = {zt:.0f} pc")
        nt = cz.truncate(z, nH, zt, no_tail)
        z_mid, W = cz.balanced_P(z, nt)
        Wbin = np.array([np.mean(W[zsl][np.asarray(idx, int)]) for _, idx in bin_groups])
        st = {k: np.full(zb.size, np.nan) for k in STAT_KEYS}
        for b in range(zb.size):
            s = caa.cell_alpha_stats(cells[b], np.ones_like(cells[b]), float(Wbin[b]),
                                     PERCENTILE_SCHEME_DEFAULT)
            for k in STAT_KEYS:
                st[k][b] = s[k]
            if zt == Z_TOPS_PC[0]:
                pth_harm[b] = s["Pth_harm"]
        sl = np.abs(z - z_mid) <= SLAB_HALF_PC
        pv, ph = cz.slab_pth(ps, z_mid)
        slab_W = float(np.mean(W[sl]))
        R[zt] = {"z_mid": z_mid, "W": W, "Wbin": Wbin, **st,
                 "slab_W": slab_W, "slab_alpha": slab_W / ph}
        key = f"zt{int(zt)}__"
        out[key + "z_mid"] = np.array([z_mid])
        out[key + "W_full"] = W
        out[key + "W_bin"] = Wbin
        for k in STAT_KEYS:
            out[key + k] = st[k]
        out[key + "slab_W"] = np.array([slab_W])
        out[key + "slab_alpha_cell_mean"] = np.array([slab_W / ph])
        print(f"  z_mid = {z_mid:+.4f} pc, slab <W> = {slab_W:.1f}, slab <alpha_cell> = {slab_W / ph:.4f}")
    out["Pth_harm_bin"] = pth_harm

    # ---- checks ---------------------------------------------------------------------
    f = R[FIDUCIAL_Z_TOP_PC]
    print("\nchecks")
    for k, ref in (("Wbin", "mean__Ptot_area"), ("alpha_cell_mean", "mean__alpha_cell_mean_vol"),
                   ("alpha_cell_median", "mean__alpha_cell_median_vol"),
                   ("alpha_cell_p_lo", "mean__alpha_cell_p_lo_vol"),
                   ("alpha_cell_p_hi", "mean__alpha_cell_p_hi_vol")):
        check(f"z_top = 750: {k} vs area_alpha {ref}, every bin (max rel)",
              float(np.max(np.abs(f[k] / aa[ref] - 1))), REPRO_RTOL)
    check("<1/P_th>^-1 per bin vs area_alpha mean__Pth_harm_vol (max rel)",
          float(np.max(np.abs(pth_harm / aa["mean__Pth_harm_vol"] - 1))), REPRO_RTOL)
    gz = zc["z_top_grid"]
    for zt in Z_TOPS_PC:
        j = int(np.argmin(np.abs(gz - zt)))
        check(f"z_top = {zt:.0f}: z_mid vs zmax.npz (abs pc)", abs(R[zt]["z_mid"] - zc["zt_z_mid"][j]), 1e-6)
        check(f"z_top = {zt:.0f}: slab <W> vs zmax.npz (rel)",
              abs(R[zt]["slab_W"] / zc["zt_slab_P_mid"][j] - 1), REPRO_RTOL)
        check(f"z_top = {zt:.0f}: slab <alpha_cell> vs zmax.npz (rel)",
              abs(R[zt]["slab_alpha"] / zc["zt_alpha_cell_mean"][j] - 1), REPRO_RTOL)

    OUT_NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ_PATH, **out)
    print(f"Saved {OUT_NPZ_PATH}")

    # ================================================================= report
    def at(zt, dz, sgn):
        """The bin nearest z_mid + sgn * dz (z_mid of this z_top)."""
        zz = R[zt]["z_mid"] + sgn * dz
        b = int(np.argmin(np.abs(zb - zz)))
        return b, R[zt]["alpha_cell_mean"][b]

    W_ = 104
    L = [PROVISIONAL_CUBE_HEADER,
         "DIAGNOSTIC: does the shape of alpha(z) depend on the integration height z_top?",
         "=" * W_,
         "Measures only -- no physics, cache, existing result or figure is changed. Cube only.",
         "",
         "METHOD",
         "  rho = 0 at |z| > z_top (Sun frame, as compute_zmax), footprint-mean RAW density (all cells,",
         "  HIM included). W(z) as compute_zmax: gas self-gravity from the truncated profile, Guo+20",
         "  stars and DM, balanced midplane z_mid re-found for each z_top, P = 0 at the cube edges.",
         "  Bins: Fig. 4's 4 pc bins in signed z over |z| <= 400 pc (the stats box; every z_top here",
         "  is >= 400 pc). Per bin, compute_area_alpha.cell_alpha_stats over the non-HIM cells,",
         "  volume-weighted, with <W> = the mean of W over the bin's planes:",
         "    <alpha_cell> = <W> <1/P_th> (headline), median alpha_cell, 15-85% range.",
         "  <1/P_th>^-1 per bin is independent of z_top. Because <W> is one number per bin, every",
         "  alpha_cell statistic of a bin scales with <W>: the z_top dependence is the W(z) dependence.",
         f"  Slab: |z - z_mid| <= {SLAB_HALF_PC:g} pc with this z_top's z_mid; <alpha_cell> = slab <W> /",
         "  slab <1/P_th>^-1 (compute_zmax). Table rows: the bin whose centre is nearest z_mid +- dz.",
         "",
         "-" * W_, "ASSERTIONS (the script raises if any fails)", "-" * W_]
    L += [f"  {m:>10.2e}  (<= {t:.0e})  {n}" for n, m, t in checks]
    L += ["", "-" * W_, "z_mid AND THE MIDPLANE SLAB", "-" * W_,
          f"  {'z_top':>6}{'z_mid [pc]':>12}{'slab <W>':>11}{'slab <alpha_cell>':>19}"]
    for zt in Z_TOPS_PC:
        L.append(f"  {zt:>6.0f}{R[zt]['z_mid']:>12.3f}{R[zt]['slab_W']:>11.1f}{R[zt]['slab_alpha']:>19.4f}")

    for title, fn in (
            ("<alpha_cell>, mean of the two sides", lambda zt, dz: 0.5 * (at(zt, dz, 1)[1] + at(zt, dz, -1)[1])),
            ("<alpha_cell>, upper side (z > z_mid)", lambda zt, dz: at(zt, dz, 1)[1]),
            ("<alpha_cell>, lower side (z < z_mid)", lambda zt, dz: at(zt, dz, -1)[1])):
        L += ["", "-" * W_, title, "-" * W_,
              f"  {'|z-z_mid|':>10}" + "".join(f"{f'z_top {zt:.0f}':>12}" for zt in Z_TOPS_PC)
              + "".join(f"{f'/ slab {zt:.0f}':>12}" for zt in Z_TOPS_PC)]
        L.append(f"  {'slab':>10}" + "".join(f"{R[zt]['slab_alpha']:>12.3f}" for zt in Z_TOPS_PC)
                 + "".join(f"{1.0:>12.3f}" for _ in Z_TOPS_PC))
        for dz in TABLE_DZ_PC:
            v = [fn(zt, dz) for zt in Z_TOPS_PC]
            L.append(f"  {dz:>10.0f}" + "".join(f"{x:>12.3f}" for x in v)
                     + "".join(f"{x / R[zt]['slab_alpha']:>12.3f}" for x, zt in zip(v, Z_TOPS_PC)))
    L += ["  (left block: <alpha_cell>; right block: the shape <alpha_cell>(z) / <alpha_cell>(slab))",
          "", "  Bins used (centre z [pc]) for z_top = 750; the others differ by at most one bin:",
          "  " + ", ".join(f"+-{dz:.0f}: {zb[at(750.0, dz, 1)[0]]:+.0f}/{zb[at(750.0, dz, -1)[0]]:+.0f}"
                           for dz in TABLE_DZ_PC)]

    # the shape relative to the fiducial, per bin
    L += ["", "-" * W_, "SHAPE RELATIVE TO THE FIDUCIAL: [alpha(z)/alpha(slab)]_ztop / [same]_750, per bin", "-" * W_,
          f"  over the bins with |z - z_mid(750)| <= {STATS_BOX_Z_HALF_RANGE_PC - 50:g} pc:",
          f"  {'z_top':>6}{'min':>8}{'median':>8}{'max':>8}"]
    zm750 = R[750.0]["z_mid"]
    sel = np.abs(zb - zm750) <= STATS_BOX_Z_HALF_RANGE_PC - 50.0
    s750 = R[750.0]["alpha_cell_mean"] / R[750.0]["slab_alpha"]
    for zt in Z_TOPS_PC:
        rr = (R[zt]["alpha_cell_mean"] / R[zt]["slab_alpha"] / s750)[sel]
        L.append(f"  {zt:>6.0f}{rr.min():>8.3f}{np.median(rr):>8.3f}{rr.max():>8.3f}")
    L += ["", f"runtime {time.time() - t0:.0f} s"]
    OUT_TXT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
