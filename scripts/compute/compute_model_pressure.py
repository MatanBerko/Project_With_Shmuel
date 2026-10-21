"""
Fig. 1 (pressure-framed BS19 model): everything the figure needs, in one cache.

PRESSURE CONVENTION: every pressure here is the PHYSICAL thermal pressure
P_th/k = 1.1 n_H T (helium included; src.physics.thermal.p_th_phys_over_kb),
INCLUDING the BS19 equilibrium curves, P_min and P_max: the table is
tabulated in n*T, so each of those is multiplied by 1.1 here, so that they
share an axis with P_tot. (The HIM flag itself is still evaluated in the
BS19 n*T convention, as everywhere in the project -- only what is PLOTTED
is converted.)

From the BS19 table (path via src.config_loader, physics via
src.physics.thermal):
  * P/k(n) = 1.1 n T on the table's n grid at each I_UV in IUV_CURVES,
    with that curve's own dP/dn = 0 turning points (P_max at n_W,max,
    P_min at n_C,min);
  * P_min(I_UV), P_max(I_UV) (x1.1) and n_W,max(I_UV), n_C,min(I_UV) on a
    fine log I_UV grid over the range where the S-curve exists;
  * P_th/k = 1.1 n T on the full (n, I_UV) table grid.

From the cube (RAW, fiducial: non-HIM cells, |x|,|y| <= 500 pc, |z| <= 30 pc):
  * the mass-weighted (rho = 1.4 m_H n_H) 2-D histogram in (log n, log I_UV)
    and the mass-weighted 1-D I_UV distribution, with exact weighted
    15/50/85 percentiles.
  Only that slab is read (31 planes), never the full cube. cache/core/
  summary.npz has a midplane (I_UV, n) 2-D PDF with the same definition
  (n_H-weighted, non-HIM, footprint, |z| <= 30 pc) but on 49 coarse bins
  set by the data extremes, and its 1-D I_UV PDF is VOLUME-weighted -- so
  the percentiles and the outside-S-curve mass fraction are computed from
  the cells; the cached 2-D PDF is used as a cross-check (same total, same
  distribution when re-binned onto its edges).
  The HIM flag is read from cache/core/alpha_core.zarr (RAW) and checked
  against a recomputation through src.physics.him on the same cells.

Midplane reference pressures are READ from results/area_alpha.txt (Step 2e
section 6, the |z| <= 30 pc slab, self-gravity on, volume-weighted):
<P_tot>_area and <1/P_th>^-1 (the column headed P_th,harm). Not recomputed.

Writes cache/model/model_pressure.npz and results/model_pressure.txt.
No maps, so no XY reorientation.
"""

import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config_loader import load_resolved_config  # noqa: E402
from src.conventions import (  # noqa: E402
    M_H,
    MU,
    PROVISIONAL_CUBE_HEADER,
    STATS_BOX_XY_HALF_RANGE_PC,
)
from src.conventions import MIDPLANE_MODE, MIDPLANE_MODE_BALANCED  # noqa: E402
from src.physics import midplane, thermal  # noqa: E402
from src.physics.him import him_flag  # noqa: E402
from src.physics.loading import (  # noqa: E402
    IUV_FIELD,
    RHO_FIELD,
    T_FIELD,
    open_zarr,
    slab_z_indices,
    xy_index_range,
)
from src.physics.stats import weighted_percentile  # noqa: E402

OUT_NPZ_PATH = Path("cache/model/model_pressure.npz")
OUT_TXT_PATH = Path("results/model_pressure.txt")
ALPHA_CORE_ZARR_PATH = Path("cache/core/alpha_core.zarr")
SUMMARY_NPZ_PATH = Path("cache/core/summary.npz")
AREA_ALPHA_TXT = Path("results/area_alpha.txt")

F_NEUTRAL = 1.1                    # P_th/k = 1.1 n T, the factor applied to the table
SLAB_HALF_PC = 30.0
CLEAN_IUV = (0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0)
N_IUV_CURVES = 4
PCT = (15.0, 50.0, 85.0)
DEX_N, DEX_IUV = 0.05, 0.025       # histogram bin widths
N_FINE_IUV = 400


# ---------------------------------------------------------------------------
# BS19 table
# ---------------------------------------------------------------------------
def load_table(mat_path):
    """(n grid ascending, I_UV grid ascending, T_2d_n [iuv, n])."""
    from scipy.io import loadmat
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = loadmat(str(mat_path))
    iuv = np.array(d["IUV_"]).squeeze().astype(float).ravel()
    n = np.array(d["n_"]).squeeze().astype(float).ravel()
    t = np.array(d["T_2d_n"]).astype(float)
    on, oi = np.argsort(n), np.argsort(iuv)
    return n[on], iuv[oi], t[oi][:, on]


def t_at_iuv(iuv_grid, t2d, iuv):
    """T(n) at one I_UV, linear in log10 I_UV between table rows (no extrapolation)."""
    li = np.log10(iuv_grid)
    x = np.log10(iuv)
    if not (li[0] - 1e-12 <= x <= li[-1] + 1e-12):
        raise ValueError(f"I_UV = {iuv:g} outside the table range")
    k = int(np.clip(np.searchsorted(li, x), 1, li.size - 1))
    w = (x - li[k - 1]) / (li[k] - li[k - 1])
    return (1.0 - w) * t2d[k - 1] + w * t2d[k]


def scurve_edges(n_grid, iuv_grid, t2d):
    """Lowest and highest I_UV with an S-curve, refined between table rows.

    A row "has an S-curve" when P(n) has two dP/dn = 0 turning points
    (thermal.locate_scurve_turning_points). The edge between the last row
    with one and the first without is refined on log-linearly interpolated
    curves to ~0.001 dex.
    """
    has = np.array([thermal.locate_scurve_turning_points(n_grid, t2d[i]) is not None
                    for i in range(iuv_grid.size)])
    if not has.any():
        raise RuntimeError("no S-curve anywhere in the table")
    idx = np.where(has)[0]
    if not np.all(np.diff(idx) == 1):
        raise RuntimeError("S-curve rows are not contiguous in I_UV")

    def refine(i_in, i_out):
        lo, hi = np.log10(iuv_grid[i_in]), np.log10(iuv_grid[i_out])
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            ok = thermal.locate_scurve_turning_points(
                n_grid, t_at_iuv(iuv_grid, t2d, 10 ** mid)) is not None
            lo, hi = (mid, hi) if ok else (lo, mid)
        return 10 ** lo

    i0, i1 = int(idx[0]), int(idx[-1])
    lo_edge = iuv_grid[i0] if i0 == 0 else refine(i0, i0 - 1)
    hi_edge = iuv_grid[i1] if i1 == iuv_grid.size - 1 else refine(i1, i1 + 1)
    return float(lo_edge), float(hi_edge), (float(iuv_grid[i0]), float(iuv_grid[i1]))


def choose_iuv_curves(p15, p85, n=N_IUV_CURVES, clean=CLEAN_IUV):
    """n clean values BRACKETING [p15, p85].

    The ends are the largest clean value <= p15 and the smallest >= p85;
    the inner values are log-even targets between those ends, each snapped
    to the nearest clean value not already taken. (Snapping the ends too,
    as before Step 1i, flipped the whole set from 0.3/0.5/1/2 to 0.5/1/2/3
    when p15 moved from 0.385 to 0.392 -- across the 0.3/0.5 snapping
    boundary -- and lost the low end of the range.)
    """
    clean = tuple(sorted(clean))
    lo = max((c for c in clean if c <= p15), default=clean[0])
    hi = min((c for c in clean if c >= p85), default=clean[-1])
    targets = np.logspace(np.log10(lo), np.log10(hi), n)[1:-1]
    lc = np.log10(clean)
    chosen = [lo, hi]
    for t in targets:
        order = np.argsort(np.abs(lc - np.log10(t)))
        pick = next(clean[j] for j in order if clean[j] not in chosen)
        chosen.append(pick)
    return tuple(sorted(chosen))


# ---------------------------------------------------------------------------
# midplane reference pressures (read, not recomputed)
# ---------------------------------------------------------------------------
def read_midplane_references(path=AREA_ALPHA_TXT):
    """(<P_tot>_area, <1/P_th>^-1, <alpha_cell>, header, data line): section 6, vol, z = 0, sg mean."""
    lines = path.read_text().splitlines()
    i6 = next(i for i, ln in enumerate(lines) if ln.startswith("6. PER-CELL"))
    iv = next(i for i in range(i6, len(lines)) if lines[i].startswith("VOLUME-weighted"))
    header = lines[iv + 1]
    # the midplane slab row: 'mid' (balanced, Step 1i) or '0' (Sun-centred)
    row = next(ln for ln in lines[iv:] if re.match(r"^\s+(mid|0)\s+mean\s", ln))
    cols = header.split()
    if not ("<P_tot>_area" in cols and "P_th,harm" in cols and "<a_cell>" in cols):
        raise RuntimeError(f"unexpected header in {path}: {header!r}")
    vals = row.split()
    # data tokens: z sg <P_tot>_area <P_th>_arith P_th,harm a_area <a_cell> med [p15, p85]
    return float(vals[2]), float(vals[4]), float(vals[6]), header, row


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    cfg = load_resolved_config()
    mat = cfg["bs19_mat_path"]

    # ---- the table --------------------------------------------------------
    n_grid, iuv_grid, t2d = load_table(mat)
    iuv_lo_edge, iuv_hi_edge, row_bounds = scurve_edges(n_grid, iuv_grid, t2d)
    pmm = thermal.build_pmin_pmax(mat)
    bounds = thermal.build_phase_density_bounds(mat)
    iuv_fine = np.logspace(np.log10(iuv_lo_edge), np.log10(iuv_hi_edge), N_FINE_IUV)
    print(f"S-curve exists for I_UV in [{iuv_lo_edge:.4g}, {iuv_hi_edge:.4g}] "
          f"(table rows {row_bounds[0]:g} .. {row_bounds[1]:g})")

    # ---- the midplane slab of the cube -----------------------------------
    grid = open_zarr()
    x_lo, x_hi = xy_index_range(grid.x_pc, STATS_BOX_XY_HALF_RANGE_PC)
    y_lo, y_hi = xy_index_range(grid.y_pc, STATS_BOX_XY_HALF_RANGE_PC)
    # Step 1i: the midplane slab is |z - z_mid| <= 30 pc (the fiducial
    # self-gravity setting's z_mid, as built and recorded in the cache)
    core = zarr.open_group(str(ALPHA_CORE_ZARR_PATH), mode="r")
    z_mid = float(core["RAW"]["self_gravity_mean"].attrs.get("z_mid_pc", 0.0))
    z_slab = midplane.slab_center(0.0, z_mid, MIDPLANE_MODE)
    iz = slab_z_indices(grid.z_pc, z_slab, SLAB_HALF_PC)
    zs = slice(int(iz.min()), int(iz.max()) + 1)
    if zs.stop - zs.start != iz.size:
        raise RuntimeError("slab planes are not contiguous")
    read = lambda f: np.asarray(grid.store[f][zs, y_lo:y_hi, x_lo:x_hi], dtype=np.float64)  # noqa: E731
    n_h, iuv, T = read(RHO_FIELD), read(IUV_FIELD), read(T_FIELD)

    zc = np.asarray(core["z_pc"][:], dtype=float)
    jz = np.array([int(np.argmin(np.abs(zc - z))) for z in grid.z_pc[zs]])
    if not np.allclose(zc[jz], grid.z_pc[zs], atol=1e-3) or np.any(np.diff(jz) != 1):
        raise RuntimeError("alpha_core z grid does not line up with the slab")
    him = np.asarray(core["RAW"]["him"][jz[0]:jz[-1] + 1]).astype(bool)
    him_re = him_flag(thermal.p_nT(n_h, T), pmm.p_min(iuv))
    n_disagree = int(np.sum(him != him_re))
    print(f"slab z = {grid.z_pc[zs][0]:g}..{grid.z_pc[zs][-1]:g} pc ({iz.size} planes), "
          f"{n_h.size} cells; cached vs recomputed HIM flag disagree in {n_disagree} cells")
    if n_disagree:
        raise RuntimeError("cached HIM flag does not match a recomputation -- STOPPING")

    sel = ~him & np.isfinite(n_h) & (n_h > 0) & np.isfinite(iuv) & (iuv > 0)
    n_s, iuv_s = n_h[sel], iuv[sel]
    mass = MU * M_H * n_s                         # g cm^-3 per cell; cells have equal volume
    w = mass / mass.sum()

    p15, p50, p85 = (weighted_percentile(iuv_s, mass, q) for q in PCT)
    # VOLUME-weighted (equal weight per cell) -- the weighting of the area-
    # averaged <P_tot> and <1/P_th>^-1 the figure draws next to this gas
    ones = np.ones_like(iuv_s)
    v15, v50, v85 = (weighted_percentile(iuv_s, ones, q) for q in PCT)
    # curves span the VOLUME-weighted 15-85% range (the figure's weighting);
    # the mass-weighted range gives the same four clean values
    iuv_curves = choose_iuv_curves(v15, v85)
    if iuv_curves != choose_iuv_curves(p15, p85):
        print("  note: mass- and volume-weighted ranges pick different IUV_CURVES")
    print(f"volume-weighted I_UV percentiles 15/50/85: {v15:.3f} / {v50:.3f} / {v85:.3f}")
    print(f"IUV_CURVES = {iuv_curves}")
    f_out_hi = float(w[iuv_s > iuv_hi_edge].sum())
    f_out_lo = float(w[iuv_s < iuv_lo_edge].sum())
    f_out_hi_grid = float(w[iuv_s > row_bounds[1]].sum())
    fv_out_hi = float(np.mean(iuv_s > iuv_hi_edge))
    fv_out_lo = float(np.mean(iuv_s < iuv_lo_edge))
    fv_out_hi_grid = float(np.mean(iuv_s > row_bounds[1]))

    # ---- histograms -------------------------------------------------------
    ln, li = np.log10(n_s), np.log10(iuv_s)
    n_edges = np.arange(np.floor(ln.min() / DEX_N) * DEX_N, ln.max() + DEX_N, DEX_N)
    i_edges = np.arange(np.floor(li.min() / DEX_IUV) * DEX_IUV, li.max() + DEX_IUV, DEX_IUV)
    H2, _, _ = np.histogram2d(ln, li, bins=[n_edges, i_edges], weights=w)
    H1, _ = np.histogram(li, bins=i_edges, weights=w)
    H1_vol, _ = np.histogram(li, bins=i_edges)
    H1_vol = H1_vol / H1_vol.sum()          # cell fraction per bin, like H1

    # cross-check against the pipeline's cached midplane 2-D PDF (I_UV, n)
    s = np.load(SUMMARY_NPZ_PATH)
    base = "RAW__selexclude_him_flag__pdf2d__Iuv_n__z0"
    Hc, xe, ye = s[f"{base}__H"], s[f"{base}__xedges"], s[f"{base}__yedges"]
    Hr, _, _ = np.histogram2d(iuv_s, n_s, bins=[xe, ye], weights=n_s)
    cross_rel = float(np.abs(Hr - Hc).sum() / Hc.sum())
    print(f"cross-check vs summary.npz midplane (I_UV, n) PDF: sum|diff|/sum = {cross_rel:.2e}")
    if cross_rel > 1e-5:
        raise RuntimeError("slab histogram does not reproduce the cached midplane PDF -- STOPPING")

    # ---- model curves (x1.1) -----------------------------------------------
    out = {
        "n_grid": n_grid, "iuv_grid": iuv_grid,
        "P_grid": F_NEUTRAL * n_grid[None, :] * t2d,           # [iuv, n]
        "iuv_fine": iuv_fine,
        "P_min_fine": F_NEUTRAL * pmm.p_min(iuv_fine),
        "P_max_fine": F_NEUTRAL * pmm.p_max(iuv_fine),
        "n_w_max_fine": bounds.n_w_max(iuv_fine),
        "n_c_min_fine": bounds.n_c_min(iuv_fine),
        "iuv_scurve_edges": np.array([iuv_lo_edge, iuv_hi_edge]),
        "iuv_curves": np.array(iuv_curves),
        "iuv_pct": np.array([p15, p50, p85]), "pct_levels": np.array(PCT),
        "slab_logn_edges": n_edges, "slab_logiuv_edges": i_edges,
        "slab_H2_mass": H2, "slab_H1_iuv_mass": H1,
        "f_mass_iuv_above_scurve": np.array([f_out_hi]),
        "f_mass_iuv_below_scurve": np.array([f_out_lo]),
        "f_vol_iuv_above_scurve": np.array([fv_out_hi]),
        "f_vol_iuv_below_scurve": np.array([fv_out_lo]),
        "F_NEUTRAL": np.array([F_NEUTRAL]),
        "provisional": np.array([PROVISIONAL_CUBE_HEADER]),
    }
    curve_rows = []
    for k, iv in enumerate(iuv_curves):
        tc = t_at_iuv(iuv_grid, t2d, iv)
        tp = thermal.locate_scurve_turning_points(n_grid, tc)
        out[f"curve{k}__P"] = F_NEUTRAL * n_grid * tc
        if tp is None:
            out[f"curve{k}__turning"] = np.full(4, np.nan)
        else:
            nw, nc, pw, pc = tp
            out[f"curve{k}__turning"] = np.array([nw, nc, F_NEUTRAL * pw, F_NEUTRAL * pc])
        curve_rows.append((iv, F_NEUTRAL * float(pmm.p_min(iv)), F_NEUTRAL * float(pmm.p_max(iv)),
                           out[f"curve{k}__turning"]))

    p_tot_mid, p_inv_mid, a_cell_mid, ref_header, ref_row = read_midplane_references()
    ratio_mid = p_tot_mid / p_inv_mid
    if abs(ratio_mid - a_cell_mid) > 1e-3 * a_cell_mid:
        raise RuntimeError(f"<P_tot>/<1/P_th>^-1 = {ratio_mid:.4f} does not match "
                           f"<alpha_cell> = {a_cell_mid} from area_alpha.txt")
    out["midplane_Ptot_area"] = np.array([p_tot_mid])
    out["midplane_Pth_inv_inv"] = np.array([p_inv_mid])

    # I_UV,cross: where P_max(I_UV) (x1.1, build_pmin_pmax -- the P_max the
    # HIM flag and the table in this file use) equals the midplane <P_tot>.
    # Below it <P_tot> > P_max, i.e. the total pressure is above the whole
    # two-phase band. P_max rises monotonically with I_UV over the S-curve
    # range, which is checked rather than assumed.
    pmax_fine = F_NEUTRAL * pmm.p_max(iuv_fine)
    if np.any(np.diff(pmax_fine) < 0):
        raise RuntimeError("P_max(I_UV) is not monotonic -- the crossing is not unique")
    lo, hi = np.log10(iuv_fine[0]), np.log10(iuv_fine[-1])
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if F_NEUTRAL * float(pmm.p_max(10 ** mid)) < p_tot_mid else (lo, mid)
    iuv_cross = 10 ** (0.5 * (lo + hi))
    f_below_cross = float(w[iuv_s < iuv_cross].sum())
    out["iuv_cross_pmax_eq_ptot"] = np.array([iuv_cross])
    out["f_mass_iuv_below_cross"] = np.array([f_below_cross])
    f_vol_below_cross = float(np.mean(iuv_s < iuv_cross))
    pmax_v50 = F_NEUTRAL * float(pmm.p_max(v50))
    pmin_v50 = F_NEUTRAL * float(pmm.p_min(v50))
    out["iuv_pct_vol"] = np.array([v15, v50, v85])
    out["slab_H1_iuv_vol"] = H1_vol
    out["f_vol_iuv_below_cross"] = np.array([f_vol_below_cross])
    out["P_max_at_vol_median_iuv"] = np.array([pmax_v50])

    OUT_NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ_PATH, **out)
    print(f"Saved {OUT_NPZ_PATH}")

    # ---- report -------------------------------------------------------------
    where = ("BELOW P_min" if p_tot_mid < pmin_v50 else
             "ABOVE P_max" if p_tot_mid > pmax_v50 else "INSIDE the two-phase band")
    L = [
        PROVISIONAL_CUBE_HEADER,
        "Fig. 1 (pressure-framed BS19 model) -- inputs and reference numbers",
        "=" * 96,
        "Every pressure is P_th/k = 1.1 n_H T [K cm^-3], including the BS19 curves, P_min and",
        "P_max (the table's n*T x 1.1), so they share an axis with P_tot.",
        f"I_UV is F_UV / F_UV,sun with F_UV,sun = 2.7e-3 erg cm^-2 s^-1 (Shelest+26, Methods) --",
        "the Draine-field normalisation, not Habing G_0.",
        "",
        f"BS19 S-curve (two dP/dn = 0 turning points) exists for I_UV in "
        f"[{iuv_lo_edge:.4g}, {iuv_hi_edge:.4g}]",
        f"  (tabulated rows {row_bounds[0]:g} .. {row_bounds[1]:g}; upper edge refined on "
        "log-I_UV-interpolated curves)",
        "",
        f"Midplane gas: RAW, non-HIM cells, |x|,|y| <= {STATS_BOX_XY_HALF_RANGE_PC:.0f} pc, "
        + (f"|z - z_mid| <= {SLAB_HALF_PC:.0f} pc, z_mid = {z_mid:+.3f} pc (balanced, Step 1i)"
           if MIDPLANE_MODE == MIDPLANE_MODE_BALANCED else f"|z| <= {SLAB_HALF_PC:.0f} pc")
        + f" ({iz.size} planes, z = {grid.z_pc[zs][0]:g}..{grid.z_pc[zs][-1]:g} pc, "
        f"{int(sel.sum())} cells used)",
        "  VOLUME-weighted (equal weight per cell) -- the weighting of the area-averaged",
        "  <P_tot> and <1/P_th>^-1; nearest-rank percentiles",
        f"  I_UV percentiles  p15 {v15:.3f}   p50 {v50:.3f}   p85 {v85:.3f}",
        f"  IUV_CURVES (clean values spanning p15-p85): {', '.join(f'{v:g}' for v in iuv_curves)}",
        f"  volume fraction with I_UV ABOVE the S-curve range ({iuv_hi_edge:.4g}): {fv_out_hi:.4%}"
        f"   (above the last tabulated S-curve row {row_bounds[1]:g}: {fv_out_hi_grid:.4%})",
        f"  volume fraction with I_UV BELOW the S-curve range ({iuv_lo_edge:.4g}): {fv_out_lo:.4%}",
        f"  cached HIM flag == recomputed flag in every slab cell; slab 2-D histogram "
        f"reproduces summary.npz's midplane PDF to {cross_rel:.1e}",
        "",
        "P_min and P_max (1.1 n T) at IUV_CURVES",
        f"{'I_UV':>8}{'P_min':>12}{'P_max':>12}   {'curve turning points: P_min, P_max at n_C,min, n_W,max':<}",
        f"{'':>8}{'[K cm^-3]':>12}{'[K cm^-3]':>12}",
    ]
    for iv, pmn, pmx, tp in curve_rows:
        L.append(f"{iv:>8g}{pmn:>12.1f}{pmx:>12.1f}   {tp[3]:>8.1f}, {tp[2]:>8.1f}   "
                 f"at n = {tp[1]:.3g}, {tp[0]:.3g} cm^-3")
    L += [
        "  (P_min/P_max from thermal.build_pmin_pmax -- the functions the HIM flag uses, from the",
        "   P-tabulated table; turning points from the n-tabulated curve actually plotted. The",
        "   two tabulations differ by up to ~5% at these I_UV.)",
        "",
        "Midplane reference pressures, READ from results/area_alpha.txt (section 6, the midplane",
        "slab -- |z - z_mid| <= 30 pc in balanced mode --, self-gravity on = footprint-mean,",
        "volume-weighted), not recomputed:",
        f"  <P_tot>_area   = {p_tot_mid:.1f} K cm^-3",
        f"  <1/P_th>^-1    = {p_inv_mid:.1f} K cm^-3   (the column headed P_th,harm)",
        "  header line:  " + ref_header.strip(),
        "  data line:    " + ref_row.strip(),
        f"  <P_tot>/<1/P_th>^-1 = {ratio_mid:.2f} = <alpha_cell> ({a_cell_mid:.3f}, same line, "
        "section 6, volume-weighted, midplane slab, sg = mean)",
        "",
        f"At the median midplane I_UV = {v50:.3f}: P_min = {pmin_v50:.0f}, P_max = {pmax_v50:.0f} K cm^-3;",
        f"  midplane <P_tot> = {p_tot_mid:.0f} is {where}; <1/P_th>^-1 = {p_inv_mid:.0f}.",
        "",
        f"I_UV,cross, where P_max(I_UV) = midplane <P_tot> = {p_tot_mid:.1f}: "
        f"I_UV,cross = {iuv_cross:.4f}",
        f"  (P_max x1.1 from thermal.build_pmin_pmax; for I_UV < I_UV,cross, <P_tot> > P_max)",
        f"  volume fraction of midplane gas (same slab) with I_UV < I_UV,cross: "
        f"{f_vol_below_cross:.4f}",
        "",
        f"STATEMENT: the midplane <P_tot> ({p_tot_mid:.0f} K cm^-3) is above P_max for "
        f"I_UV < {iuv_cross:.3f}, which is {f_vol_below_cross:.1%} of the neutral midplane volume.",
        "",
        f"(runtime {time.time() - t0:.1f}s)",
    ]
    OUT_TXT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
