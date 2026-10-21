"""
Fig. 2 maps: two planes of the local ISM in TRUE Galactic orientation.

Reads only two planes of the cube (never the full cube):
  (i)  the z plane nearest z = 0, over |x|,|y| <= 500 pc;
  (ii) the TRUE y = 0 plane over |x| <= 500, |z| <= 400 pc. Under the
       "neg_swap" orientation (results/orientation_check.txt) true y = -x_c,
       so true y = 0 is the CUBE plane x_c = 0 (nearest), and true x = -y_c.
Both are re-indexed into true coordinates by
src.physics.loading.to_galactic_xy (CUBE_XY_MAPPING) -- the only place this
project applies the orientation fix. No statistic, profile or alpha uses it.

For each plane: n_H, I_UV, T, P_th/k = 1.1 n T, the HIM flag
(n T < 0.5 P_min(I_UV)), the dPdn phase map (CNM/UNM/WNM/HIM, the fiducial
scheme), and the Shelest+26 Fig. 3 temperature-cut map (CNM T <= 300 K,
UNM 300 < T <= 6000 K, WNM T > 6000 K, HIM NOT separated).

Independent data check (raises -- no cache is written -- if it fails):
log10 A' from the Edenhofer map (dustmaps Edenhofer2023Query(integrated=
False), as in check_orientation.py) at 2000 random points of the z = 0
plane in TRUE coordinates is correlated with log10 n_H of the RE-ORIENTED
plane and of the un-reoriented plane at the same indices. Expected ~0.9+
vs ~0.3; a re-oriented r below MIN_R_REORIENTED stops the script.

Writes cache/maps/maps_slices.npz and results/maps_summary.txt.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config_loader import load_resolved_config  # noqa: E402
from src.conventions import (  # noqa: E402
    CNM_TEMP_MAX_K,
    CUBE_XY_MAPPING,
    PROVISIONAL_CUBE_HEADER,
    STATS_BOX_XY_HALF_RANGE_PC,
    STATS_BOX_Z_HALF_RANGE_PC,
    THERMAL_PRESSURE_HEADER,
    WNM_TEMP_MIN_K,
)
from src.physics import thermal  # noqa: E402
from src.physics.him import (  # noqa: E402
    PHASE_CNM,
    PHASE_HIM,
    PHASE_UNM,
    PHASE_WNM,
    classify_phases,
    dpdn_unresolved_fraction,
    him_flag,
    phase_flag_temperature,
)
from src.physics.loading import (  # noqa: E402
    IUV_FIELD,
    RHO_FIELD,
    T_FIELD,
    galactic_xy_coords,
    open_zarr,
    to_galactic_xy,
    xy_index_range,
)

OUT_NPZ_PATH = Path("cache/maps/maps_slices.npz")
OUT_TXT_PATH = Path("results/maps_summary.txt")

N_CHECK = 2000
CHECK_SEED = 20261021
MIN_R_REORIENTED = 0.85
PERCENTILES = (1, 15, 50, 85, 99)
PHASE_NAMES = {PHASE_CNM: "CNM", PHASE_UNM: "UNM", PHASE_WNM: "WNM", PHASE_HIM: "HIM"}
FIELDS = {"n_H": RHO_FIELD, "I_UV": IUV_FIELD, "T": T_FIELD}


def derived(n, iuv, T, pminmax, bounds):
    """P_th/k, HIM flag, dPdn phase map and temperature-cut map of one plane."""
    pmin = pminmax.p_min(iuv)
    him = him_flag(thermal.p_nT(n, T), pmin)
    nw, nc = bounds.n_w_max(iuv), bounds.n_c_min(iuv)
    return {
        "P_th": thermal.p_th_phys_over_kb(n, T),
        "him": him,
        "phase_dpdn": classify_phases(him, "dPdn", n_cm3=n, n_w_max=nw, n_c_min=nc),
        # Shelest+26 Fig. 3: temperature cuts only, HIM not separated
        "phase_tcut": phase_flag_temperature(T, np.zeros_like(him)),
        "dpdn_unresolved": dpdn_unresolved_fraction(n, nw, nc, him),
    }


def pearson_log(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    return float(np.corrcoef(np.log10(a[m]), np.log10(b[m]))[0, 1]), int(m.sum())


def main():
    t0 = time.time()
    cfg = load_resolved_config()
    grid = open_zarr()
    xc, yc, zc = grid.x_pc, grid.y_pc, grid.z_pc
    half = STATS_BOX_XY_HALF_RANGE_PC
    x_lo, x_hi = xy_index_range(xc, half)
    y_lo, y_hi = xy_index_range(yc, half)
    xc_fp, yc_fp = xc[x_lo:x_hi], yc[y_lo:y_hi]
    iz0 = int(np.argmin(np.abs(zc)))
    ixc0 = int(np.argmin(np.abs(xc)))
    z_box = np.where(np.abs(zc) <= STATS_BOX_Z_HALF_RANGE_PC)[0]
    z_lo, z_hi = int(z_box.min()), int(z_box.max()) + 1
    print(f"cube {grid.store[RHO_FIELD].shape} [z, y, x]; mapping {CUBE_XY_MAPPING}")
    print(f"XY plane: z = {zc[iz0]:g} pc (index {iz0}); XZ plane: cube x_c = {xc[ixc0]:g} pc "
          f"(index {ixc0}) = true y = {0.0 - xc[ixc0]:g} pc")

    # ---- the two planes, three fields each, straight from the store ------
    xy_cube, xz_cube = {}, {}
    for key, field in FIELDS.items():
        arr = grid.store[field]
        xy_cube[key] = np.asarray(arr[iz0, y_lo:y_hi, x_lo:x_hi], dtype=np.float64)
        xz_cube[key] = np.asarray(arr[z_lo:z_hi, y_lo:y_hi, ixc0:ixc0 + 1], dtype=np.float64)

    # ---- the cube AS STORED, for the Shelest+26 comparison only: the z = 0
    #      plane above, unchanged, and the cube plane y_c = 0 (x_c horizontal)
    iyc0 = int(np.argmin(np.abs(yc)))
    asis_xz = {key: np.asarray(grid.store[field][z_lo:z_hi, iyc0, x_lo:x_hi], dtype=np.float64)
               for key, field in FIELDS.items()}

    # ---- re-orient: [y_c, x_c] -> [y, x], and [z, y_c, x_c=0] -> [z, y=0, x]
    x_true, y_true = galactic_xy_coords(xc_fp, yc_fp)
    xy = {k: np.ascontiguousarray(to_galactic_xy(v, xc_fp, yc_fp)) for k, v in xy_cube.items()}
    x_xz, y_xz = galactic_xy_coords(xc[ixc0:ixc0 + 1], yc_fp)
    xz = {k: np.ascontiguousarray(to_galactic_xy(v, xc[ixc0:ixc0 + 1], yc_fp)[:, 0, :])
          for k, v in xz_cube.items()}
    z_xz = zc[z_lo:z_hi]

    # ---- independent check against the Edenhofer map --------------------
    print(f"Querying Edenhofer2023 at {N_CHECK} random points of the z = {zc[iz0]:g} plane...")
    import astropy.units as u
    from astropy.coordinates import Galactic, SkyCoord
    from dustmaps.edenhofer2023 import Edenhofer2023Query
    rng = np.random.default_rng(CHECK_SEED)
    ii = rng.integers(0, x_true.size, N_CHECK)
    jj = rng.integers(0, y_true.size, N_CHECK)
    px, py = x_true[ii], y_true[jj]
    query = Edenhofer2023Query(integrated=False)
    gal = Galactic(u=px * u.pc, v=py * u.pc, w=np.full(N_CHECK, zc[iz0]) * u.pc,
                   representation_type="cartesian")
    coords = SkyCoord(gal)
    coords.representation_type = "spherical"
    a_prime = np.asarray(query.query(coords), dtype=float)
    del query, coords, gal
    r_re, n_re = pearson_log(xy["n_H"][jj, ii], a_prime)
    # un-reoriented: the cube plane read as if it were already true (x_c = x, y_c = y)
    r_raw, n_raw = pearson_log(xy_cube["n_H"][jj, ii], a_prime)
    print(f"  r(log n_H, log A'): re-oriented {r_re:.3f} (n = {n_re}), "
          f"un-reoriented {r_raw:.3f} (n = {n_raw})")
    if not (r_re >= MIN_R_REORIENTED):
        raise RuntimeError(
            f"Re-oriented n_H correlates with the Edenhofer map at r = {r_re:.3f} "
            f"< {MIN_R_REORIENTED}. The orientation fix does not reproduce "
            "results/orientation_check.txt. STOPPING -- no cache or figure written.")

    # ---- derived quantities ----------------------------------------------
    pminmax = thermal.build_pmin_pmax(cfg["bs19_mat_path"])
    bounds = thermal.build_phase_density_bounds(cfg["bs19_mat_path"])
    for plane in (xy, xz):
        d = derived(plane["n_H"], plane["I_UV"], plane["T"], pminmax, bounds)
        plane["unresolved"] = d.pop("dpdn_unresolved")
        plane.update(d)

    out = {
        "x_pc": x_true, "y_pc": y_true, "z_xy_pc": np.array([zc[iz0]]),
        "x_xz_pc": x_xz, "z_xz_pc": z_xz, "y_xz_pc": y_xz,
        "cube_x_c_of_xz_plane": np.array([xc[ixc0]]),
        "mapping": np.array([CUBE_XY_MAPPING]),
        "provisional": np.array([PROVISIONAL_CUBE_HEADER]),
        "check_r_reoriented": np.array([r_re]), "check_r_unreoriented": np.array([r_raw]),
        "check_n": np.array([n_re]),
        "phase_codes": np.array([PHASE_CNM, PHASE_UNM, PHASE_WNM, PHASE_HIM]),
        "phase_names": np.array(["CNM", "UNM", "WNM", "HIM"]),
    }
    for tag, plane in (("xy", xy), ("xz", xz)):
        for k in ("n_H", "I_UV", "T", "P_th"):
            out[f"{tag}__{k}"] = plane[k].astype(np.float32)
        out[f"{tag}__him"] = plane["him"]
        out[f"{tag}__phase_dpdn"] = plane["phase_dpdn"]
        out[f"{tag}__phase_tcut"] = plane["phase_tcut"]
    # as stored (mapping "identity"): cube coordinates, NO reorientation.
    # Only for testing whether a published figure read the cube this way.
    out["asis_x_c_pc"], out["asis_y_c_pc"] = xc_fp, yc_fp
    out["asis_z_xz_pc"] = z_xz
    out["asis_y_c_of_xz_plane"] = np.array([yc[iyc0]])
    for tag, plane in (("xy", xy_cube), ("xz", asis_xz)):
        for k in ("n_H", "I_UV", "T"):
            out[f"asis_{tag}__{k}"] = plane[k].astype(np.float32)
        out[f"asis_{tag}__phase_tcut"] = phase_flag_temperature(
            plane["T"], np.zeros(plane["T"].shape, dtype=bool))
    OUT_NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ_PATH, **out)
    print(f"Saved {OUT_NPZ_PATH}")

    # ---- report ------------------------------------------------------------
    L = [
        PROVISIONAL_CUBE_HEADER,
        THERMAL_PRESSURE_HEADER,
        "Fig. 2 maps: two planes in TRUE Galactic orientation",
        "=" * 96,
        f"cube axis order [z, y, x]; orientation mapping {CUBE_XY_MAPPING!r} "
        "(src.conventions.CUBE_XY_MAPPING):",
        "  the true field at (x, y, z) is the cube value at (x_c = -y, y_c = -x, z_c = z)",
        "  (results/orientation_check.txt). Applied ONLY to these maps.",
        "",
        "Planes actually used",
        f"  XY: cube z plane {iz0}, z = {zc[iz0]:g} pc; |x|,|y| <= {half:.0f} pc "
        f"({x_true.size} x {y_true.size} cells of {x_true[1] - x_true[0]:g} pc)",
        f"  XZ: cube x_c plane {ixc0}, x_c = {xc[ixc0]:g} pc  ->  true y = {0.0 - xc[ixc0]:g} pc; "
        f"true x = -y_c over |x| <= {half:.0f} pc, z = {z_xz[0]:g}..{z_xz[-1]:g} pc "
        f"({z_xz.size} planes)",
        "",
        "Independent check: log10 n_H vs log10 A' (Edenhofer2023Query(integrated=False))",
        f"  {N_CHECK} random points of the z = {zc[iz0]:g} pc plane in TRUE coordinates "
        f"(seed {CHECK_SEED}); {n_re} usable",
        f"  re-oriented plane    r = {r_re:.3f}",
        f"  un-reoriented plane  r = {r_raw:.3f}   (cube read as if x_c = x, y_c = y)",
        f"  required: re-oriented r >= {MIN_R_REORIENTED}  ->  PASS",
        "",
        "Percentiles over every cell of each plane (equal weight per cell)",
        f"{'plane':>6}{'quantity':>19}" + "".join(f"{f'p{q}':>12}" for q in PERCENTILES),
    ]
    units = {"n_H": "n_H [cm^-3]", "I_UV": "I_UV [G_0]", "T": "T [K]",
             "P_th": "P_th/k [K cm^-3]"}
    for tag, plane in (("XY", xy), ("XZ", xz)):
        for k in ("n_H", "I_UV", "T", "P_th"):
            v = np.percentile(plane[k][np.isfinite(plane[k])], PERCENTILES)
            L.append(f"{tag:>6}{units[k]:>19}" + "".join(f"{x:>12.4g}" for x in v))
    L += [
        "",
        f"Phase AREA fractions in the z = {zc[iz0]:g} pc plane (fraction of cells)",
        f"{'scheme':>40}{'CNM':>9}{'UNM':>9}{'WNM':>9}{'HIM':>9}",
    ]
    for label, key in (("dPdn (fiducial, HIM separated)", "phase_dpdn"),
                       (f"T cuts {CNM_TEMP_MAX_K}/{WNM_TEMP_MIN_K} K, no HIM (Shelest+26)",
                        "phase_tcut")):
        fr = [np.mean(xy[key] == c) for c in (PHASE_CNM, PHASE_UNM, PHASE_WNM, PHASE_HIM)]
        L.append(f"{label:>40}" + "".join(f"{f:>9.4f}" for f in fr))
    L += [
        f"  dPdn: non-HIM cells with no finite phase boundary (left UNM): "
        f"XY {xy['unresolved']:.4%}, XZ {xz['unresolved']:.4%}",
        f"  HIM-flagged fraction: XY {xy['him'].mean():.4f}, XZ {xz['him'].mean():.4f}",
        "",
        f"(runtime {time.time() - t0:.1f}s)",
    ]
    OUT_TXT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
