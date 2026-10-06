"""
"HIM-free columns": how much of the footprint survives, and what alpha is
in what survives.

The PI's proposal (the MASKED treatment) is stricter than excluding HIM
cells: throw away every (x, y) column that passes through ANY HIM-flagged
cell, keeping only sightlines that are clean all the way up. Since the
flag catches roughly half the volume at the midplane and ~97% by
|z| = 400 pc (see results/README.md), a full-height clean column may be
vanishingly rare -- so this measures the survival rate as a function of
the height Z_clean the column has to stay clean to, and then measures
alpha in whatever survives.

This is a COMPUTE script: it may import src.physics and read the 3D
cache. The figure that consumes it reads only the .npz it writes.

What it computes, RAW only (observed density, no substitution):

  (a) z_clean(x, y): the largest Z such that NO HIM-flagged cell in that
      column has |z| <= Z. Per the task's convention this is 0 when the
      midplane cell itself is flagged -- which collides with "clean at
      the midplane but flagged at |z| = 2 pc", so a separate
      midplane_flagged map is stored and the two cases stay
      distinguishable.
  (b) the strictly-clean fraction for a range of Z_clean, plus two looser
      versions (column HIM volume fraction below 5% and 10% within
      |z| <= Z_clean), since a hard "no HIM at all" cut may leave nothing.
  (c) the three alpha estimators inside clean columns, against the same
      estimators over ALL columns with HIM cells excluded -- the
      like-for-like comparison, over the same |z| range.
  (d) a selection-bias check: are the surviving columns typical? Sigma_gas
      (full column), midplane n_H and midplane P_th, clean vs all.

  (e) Step 1h: the same questions on a GRID of allowed HIM fractions
      rather than a single "no HIM at all" cut -- f_HIM(Z) <= f_max for
      f_max in {0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7} and Z out to 400 pc,
      with alpha and the bias check for every pair that keeps at least 1%
      of the columns. The grid goes to its own cache
      (cache/core/clean_columns_grid.npz) and its own section of the same
      report.

      The grid lives in THIS script rather than a separate one that
      appends to the report. A separate appender would mean re-running
      this script silently deletes the grid section, and the report's
      contents would depend on the order the two were last run.

P_tot is read from the cache unchanged. Nothing here recomputes physics;
this step only SELECTS cells.

ORIENTATION
-----------
The cube is mirrored in XY. For the z_clean MAP to be displayable in true
Galactic coordinates, the mapping is read from
results/orientation_check.txt (never hardcoded), checked to be a signed
permutation that touches only x and y, and used to write a second,
re-oriented copy of the map into the cache. Doing it here rather than in
the figure keeps the figure a pure cache consumer, and keeps the
verification next to the data. The mapping's correctness AGAINST THE SKY
was established by check_orientation.py and independently re-verified
against dustmaps by posterior_snr.py; what is verified here is that the
array transform implements that mapping, by round-trip (it is an
involution) and by a marked-cell check.
"""

import re
import sys
import time
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.conventions import (  # noqa: E402
    PERCENTILE_SCHEME_DEFAULT,
    PROVISIONAL_CUBE_HEADER,
    STATS_BOX_XY_HALF_RANGE_PC,
    STATS_BOX_Z_HALF_RANGE_PC,
    THERMAL_PRESSURE_HEADER,
)
from src.physics.stats import (  # noqa: E402
    percentile_levels,
    ratio_of_means,
    weighted_stats,
)

ALPHA_CORE_ZARR_PATH = Path("cache/core/alpha_core.zarr")
ORIENTATION_TXT = Path("results/orientation_check.txt")
OUT_NPZ_PATH = Path("cache/core/clean_columns.npz")
OUT_GRID_NPZ_PATH = Path("cache/core/clean_columns_grid.npz")
OUT_TXT_PATH = Path("results/clean_columns.txt")

VARIANT = "RAW"
SELF_GRAVITY = "mean"
Z_CLEAN_GRID_PC = (0.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0)
Z_CLEAN_ALPHA_PC = (50.0, 100.0, 150.0, 200.0)
LOOSE_THRESHOLDS = (0.05, 0.10)

# Step 1h: the allowed-HIM-fraction grid.
F_MAX_GRID = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70)
Z_GRID_PC = (50.0, 100.0, 150.0, 200.0, 300.0, 400.0)
MIN_KEEP_FRACTION = 0.01   # alpha is only reported for selections this big
F_HIM_HIST_EDGES = np.linspace(0.0, 1.0, 51)
AXIS_NAMES = ("x", "y", "z")


# ---------------------------------------------------------------------------
# orientation
# ---------------------------------------------------------------------------
def read_best_mapping(path=ORIENTATION_TXT):
    """(label, M) with cube_coords = M @ true_coords, from the "best
    mapping (-y,-x,+z)" line of check_orientation.py's report."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run scripts/validation/check_orientation.py first; "
            "the XY mapping must be read from its output, not assumed.")
    m = re.search(r"best mapping \(([^)]*)\):", path.read_text(encoding="utf-8"))
    if m is None:
        raise ValueError(f"No 'best mapping (...)' line in {path}")
    label = m.group(1)
    M = np.zeros((3, 3), dtype=float)
    for k, part in enumerate(label.split(",")):
        part = part.strip()
        M[k, AXIS_NAMES.index(part[-1])] = -1.0 if part[0] == "-" else 1.0
    if not np.allclose(M @ M.T, np.eye(3)):
        raise ValueError(f"Parsed mapping ({label}) is not a signed permutation.")
    if not (M[2, 2] == 1.0 and M[2, 0] == 0.0 and M[2, 1] == 0.0):
        raise ValueError(
            f"Mapping ({label}) touches z. This script re-orients an XY map only; "
            "a z-mixing mapping would make the z_clean map meaningless.")
    return f"({label})", M


def to_true_orientation(M_cube, x_cube, y_cube, M):
    """Re-index an XY map from cube axes to true Galactic axes.

    M_cube[iy, ix] is the value at cube (x_cube[ix], y_cube[iy]). The
    returned array is indexed [iy_true, ix_true] on the true axes, which
    are returned alongside it.

    Derivation (no hardcoded flip): cube = M @ true, so for an output
    cell at true (xt, yt) the cube coordinates are
        x_c = M[0,0] xt + M[0,1] yt,   y_c = M[1,0] xt + M[1,1] yt
    and the value is looked up at the nearest index of each. Written as
    an explicit lookup rather than a clever transpose so it stays correct
    if the mapping ever changes to a different signed permutation.
    """
    x_true = np.asarray(x_cube, dtype=float)
    y_true = np.asarray(y_cube, dtype=float)

    XT, YT = np.meshgrid(x_true, y_true)          # (ny_true, nx_true)
    XC = M[0, 0] * XT + M[0, 1] * YT
    YC = M[1, 0] * XT + M[1, 1] * YT

    def nearest(axis, vals):
        axis = np.asarray(axis, dtype=float)
        d = float(axis[1] - axis[0])
        return np.clip(np.rint((vals - axis[0]) / d), 0, len(axis) - 1).astype(np.intp)

    return M_cube[nearest(y_cube, YC), nearest(x_cube, XC)], x_true, y_true


# ---------------------------------------------------------------------------
# column cleanliness
# ---------------------------------------------------------------------------
def column_clean_maps(him, abs_z, box_half_range_pc=STATS_BOX_Z_HALF_RANGE_PC):
    """(z_first_flag, z_clean, midplane_flagged) for every (x, y) column.

    him     : (nz, ny, nx) boolean HIM flag.
    abs_z   : (nz,) |z| of each plane, in pc. Planes may come in any order
              and both signs of z map onto the same |z|.

    z_first_flag is the PRIMITIVE and the only thing any clean/dirty test
    should be made against: the smallest |z| at which the column has a
    flagged cell, +inf if it never does.

    z_clean is the quantity asked for -- the largest Z with no flagged
    cell at |z| <= Z -- derived from it. It is deliberately NOT used for
    tests, because it is overloaded: the convention makes it 0 both when
    the midplane cell is flagged and when the column is clean at the
    midplane but flagged at the next plane out. midplane_flagged
    separates those two cases.
    """
    him = np.asarray(him, dtype=bool)
    abs_z = np.asarray(abs_z, dtype=float)
    heights = np.sort(np.unique(abs_z))
    dz = float(heights[1] - heights[0]) if heights.size > 1 else 0.0

    ny, nx = him.shape[1:]
    z_first_flag = np.full((ny, nx), np.inf, dtype=np.float64)
    still_clean = np.ones((ny, nx), dtype=bool)
    for z_val in heights:
        planes = np.where(np.isclose(abs_z, z_val))[0]
        flagged_here = him[planes].any(axis=0)
        z_first_flag[still_clean & flagged_here] = z_val
        still_clean &= ~flagged_here

    midplane_flagged = z_first_flag == 0.0
    z_clean = np.where(np.isfinite(z_first_flag),
                       np.maximum(z_first_flag - dz, 0.0),
                       box_half_range_pc)
    return z_first_flag, z_clean, midplane_flagged


def clean_fraction(z_first_flag, Z):
    """Fraction of columns with no flagged cell at |z| <= Z.

    Tested against z_first_flag, never against z_clean: at an off-grid Z
    (25 pc on a 2 pc grid) a column first flagged at |z| = 26 pc IS clean
    to 25 but has z_clean = 24, so a `z_clean >= Z` test would wrongly
    call it dirty.
    """
    return float((np.asarray(z_first_flag, dtype=float) > Z).mean())


def him_column_fraction(him, abs_z, Z):
    """Per-column HIM volume fraction within |z| <= Z."""
    planes = np.where(np.asarray(abs_z, dtype=float) <= Z)[0]
    return np.asarray(him, dtype=bool)[planes].mean(axis=0)


def contiguous_plane_slice(abs_z, Z):
    """slice of the planes with |z| <= Z.

    On a z grid running -Zbox..+Zbox the set is contiguous, so a slice
    gives VIEWS instead of the copies fancy indexing would make -- which
    matters at Z = 400 pc, where each copied float32 array is 0.4 GB.
    Contiguity is asserted rather than assumed.
    """
    idx = np.where(np.asarray(abs_z, dtype=float) <= Z)[0]
    lo, hi = int(idx.min()), int(idx.max()) + 1
    if hi - lo != idx.size:
        raise ValueError(f"planes with |z| <= {Z} are not contiguous in this z ordering")
    return slice(lo, hi)


def survival_grid(him, abs_z, f_max_grid, z_grid):
    """(f_him_by_Z, survive) for a grid of allowed HIM fractions.

    survive[i, j] is the fraction of columns whose HIM volume fraction
    within |z| <= z_grid[j] is at most f_max_grid[i].

    Monotonic in f_max by construction. NOT monotonic in Z in general:
    f_HIM(Z) is a running mean, so a column flagged only near the
    midplane has its fraction diluted as Z grows and can re-enter a
    selection it had fallen out of. The real cube does this at
    f_max = 0.7, where the kept fraction rises from 0.636 at Z = 50 pc to
    0.678 at Z = 100 pc.
    """
    f_him = {Z: him_column_fraction(him, abs_z, Z) for Z in z_grid}
    survive = np.full((len(f_max_grid), len(z_grid)), np.nan)
    for i, f_max in enumerate(f_max_grid):
        for j, Z in enumerate(z_grid):
            survive[i, j] = float((f_him[Z] <= f_max).mean())
    return f_him, survive


def compute_fhim_grid(him, abs_z, alpha, p_tot, p_th, n_H, sigma_gas, iz_mid,
                        pct_lo, pct_hi):
    """Step 1h: survival, alpha and selection bias on the (Z, f_max) grid.

    A column is kept at (Z, f_max) when its HIM VOLUME FRACTION within
    |z| <= Z is at most f_max. Unlike the strict Step 1g cut, kept columns
    generally DO contain HIM cells, so the statistics are taken over the
    NON-HIM cells inside them -- the same exclusion every other RAW number
    in this project uses.

    Returns (grid dict for the cache, report lines).
    """
    ny, nx = him.shape[1:]
    n_cols = ny * nx

    f_him, survive = survival_grid(him, abs_z, F_MAX_GRID, Z_GRID_PC)

    # distribution of f_HIM at the box edge
    f400 = f_him[max(Z_GRID_PC)].ravel()
    q = np.percentile(f400, [pct_lo, 50.0, pct_hi])
    hist, _ = np.histogram(f400, bins=F_HIM_HIST_EDGES)

    est_names = ("alpha_median_of_ratios", "alpha_mean_of_ratios",
                 "alpha_ratio_of_means", "alpha_p15_of_ratios", "alpha_p85_of_ratios")
    shape = (len(F_MAX_GRID), len(Z_GRID_PC))
    grid = {f"{e}__{wt}": np.full(shape, np.nan)
            for e in est_names for wt in ("vol", "mw")}
    for k in ("bias_Sigma_gas_median", "bias_n_mid_median", "bias_Pth_mid_median"):
        grid[k] = np.full(shape, np.nan)
    grid["n_columns_kept"] = np.zeros(shape, dtype=np.int64)
    grid["n_cells_used"] = np.zeros(shape, dtype=np.int64)
    grid["reported"] = np.zeros(shape, dtype=bool)

    n_mid = n_H[iz_mid].astype(np.float64)
    pth_mid = p_th[iz_mid].astype(np.float64)
    sg = sigma_gas.astype(np.float64)

    for j, Z in enumerate(Z_GRID_PC):
        zsl_Z = contiguous_plane_slice(abs_z, Z)
        him_s = him[zsl_Z]
        a_s, pt_s, pth_s, n_s = alpha[zsl_Z], p_tot[zsl_Z], p_th[zsl_Z], n_H[zsl_Z]
        neutral_s = ~him_s
        for i, f_max in enumerate(F_MAX_GRID):
            col_mask = f_him[Z] <= f_max
            kept = int(col_mask.sum())
            grid["n_columns_kept"][i, j] = kept
            keep_frac = kept / n_cols
            if keep_frac < MIN_KEEP_FRACTION:
                continue
            grid["reported"][i, j] = True

            sel = col_mask[None, :, :] & neutral_s
            grid["n_cells_used"][i, j] = int(sel.sum())
            a_v = a_s[sel].astype(np.float64)
            pt_v = pt_s[sel].astype(np.float64)
            pth_v = pth_s[sel].astype(np.float64)
            n_v = n_s[sel].astype(np.float64)
            del sel

            for wt in ("vol", "mw"):
                w = np.ones_like(a_v) if wt == "vol" else np.where(np.isfinite(n_v), n_v, 0.0)
                st = weighted_stats(a_v, w, PERCENTILE_SCHEME_DEFAULT)
                grid[f"alpha_median_of_ratios__{wt}"][i, j] = st.median
                grid[f"alpha_mean_of_ratios__{wt}"][i, j] = st.arithmetic_mean
                grid[f"alpha_p15_of_ratios__{wt}"][i, j] = st.p_lo
                grid[f"alpha_p85_of_ratios__{wt}"][i, j] = st.p_hi
                grid[f"alpha_ratio_of_means__{wt}"][i, j] = ratio_of_means(pt_v, pth_v, w)
            del a_v, pt_v, pth_v, n_v

            for name, arr in (("Sigma_gas", sg), ("n_mid", n_mid), ("Pth_mid", pth_mid)):
                vals = arr[col_mask]
                vals = vals[np.isfinite(vals)]
                grid[f"bias_{name}_median"][i, j] = float(np.median(vals)) if vals.size else np.nan
            print(f"    Z={Z:>5.0f} f_max={f_max:<5.2f} kept {kept:>7d} columns "
                  f"({keep_frac:.4%}), {grid['n_cells_used'][i, j]:>10d} cells")

    grid.update({
        "f_max_grid": np.array(F_MAX_GRID, dtype=float),
        "Z_grid_pc": np.array(Z_GRID_PC, dtype=float),
        "survive_fraction": survive,
        "min_keep_fraction": np.array([MIN_KEEP_FRACTION]),
        "f_him_at_box_edge_percentiles": q,
        "f_him_at_box_edge_hist": hist,
        "f_him_hist_edges": F_HIM_HIST_EDGES,
        "f_him_at_box_edge_mean": np.array([float(f400.mean())]),
        "n_columns_total": np.array([n_cols]),
        "percentile_levels": np.array([pct_lo, pct_hi], dtype=float),
    })
    for Z in Z_GRID_PC:
        grid[f"f_him_map_Z{int(Z)}"] = f_him[Z].astype(np.float32)

    # ---- report lines
    L = [
        "",
        "=" * 100,
        "(d) Step 1h: survival on a grid of ALLOWED HIM fractions",
        "=" * 100,
        "A column is kept at (Z, f_max) when its HIM VOLUME FRACTION within |z| <= Z is at",
        "most f_max. f_max = 0 is the strict cut of section (a). Kept columns generally DO",
        "contain HIM cells once f_max > 0, so the alpha statistics below are taken over the",
        "NON-HIM cells inside them -- the same exclusion every other RAW number here uses.",
        "",
        "Fraction of columns kept:",
        f"{'f_max':>8}" + "".join(f"{'Z=' + format(Z, '.0f'):>12}" for Z in Z_GRID_PC),
        "-" * (8 + 12 * len(Z_GRID_PC)),
    ]
    for i, f_max in enumerate(F_MAX_GRID):
        L.append(f"{f_max:>8.2f}" + "".join(f"{survive[i, j]:>12.5f}"
                                            for j in range(len(Z_GRID_PC))))
    L += [
        "-" * (8 + 12 * len(Z_GRID_PC)),
        "  Monotonic in f_max by construction. NOT guaranteed monotonic in Z: f_HIM(Z) is a",
        "  running mean, so a column flagged only near the midplane has its fraction DILUTED",
        "  as Z grows and can re-enter the selection.",
        "",
        f"Distribution of f_HIM at |z| <= {max(Z_GRID_PC):.0f} pc over all {n_cols} columns:",
        f"  median {q[1]:.4f}, {pct_lo:g}/{pct_hi:g} percentiles {q[0]:.4f} / {q[2]:.4f}, "
        f"mean {float(f400.mean()):.4f}",
        f"  (full histogram, {len(F_HIM_HIST_EDGES) - 1} bins over [0, 1], is in the cache)",
        "",
        f"alpha and selection bias, for every pair keeping >= {MIN_KEEP_FRACTION:.0%} of columns",
        f"{'Z':>5}{'f_max':>7}{'columns':>9}{'kept':>8}{'cells':>11}{'wt':>5}"
        f"{'median':>9}{'mean':>9}{'rat.mns':>9}{'p' + format(pct_lo, 'g'):>8}"
        f"{'p' + format(pct_hi, 'g'):>8}{'Sig_gas':>9}{'n_H(0)':>9}{'Pth(0)':>9}",
        "-" * 114,
    ]
    for j, Z in enumerate(Z_GRID_PC):
        any_row = False
        for i, f_max in enumerate(F_MAX_GRID):
            if not grid["reported"][i, j]:
                continue
            any_row = True
            for wt in ("vol", "mw"):
                L.append(
                    f"{Z:>5.0f}{f_max:>7.2f}{grid['n_columns_kept'][i, j]:>9d}"
                    f"{grid['n_columns_kept'][i, j] / n_cols:>8.4f}"
                    f"{grid['n_cells_used'][i, j]:>11d}{wt:>5}"
                    f"{grid[f'alpha_median_of_ratios__{wt}'][i, j]:>9.4f}"
                    f"{grid[f'alpha_mean_of_ratios__{wt}'][i, j]:>9.4f}"
                    f"{grid[f'alpha_ratio_of_means__{wt}'][i, j]:>9.4f}"
                    f"{grid[f'alpha_p15_of_ratios__{wt}'][i, j]:>8.4f}"
                    f"{grid[f'alpha_p85_of_ratios__{wt}'][i, j]:>8.4f}"
                    f"{grid['bias_Sigma_gas_median'][i, j]:>9.3f}"
                    f"{grid['bias_n_mid_median'][i, j]:>9.4f}"
                    f"{grid['bias_Pth_mid_median'][i, j]:>9.0f}")
        if any_row:
            L.append("-" * 114)
    L += [
        "  Sig_gas / n_H(0) / Pth(0) are MEDIANS over the kept columns, for comparison with",
        "  the all-column values in section (c): 4.806 Msun/pc^2, 0.1675 cm^-3, 1316 K cm^-3.",
        "  Pairs keeping less than the threshold are left out of this table; their survival",
        "  fractions are still in the grid above.",
    ]
    return grid, L


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    OUT_NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pct_lo, pct_hi = percentile_levels(PERCENTILE_SCHEME_DEFAULT)

    label, M = read_best_mapping()
    print(f"Orientation (for the map display only): {label}, read from {ORIENTATION_TXT}")

    import pipeline.compute_all as pipe
    core = zarr.open_group(str(ALPHA_CORE_ZARR_PATH), mode="r")
    x_pc = np.asarray(core["x_pc"][:], dtype=np.float64)
    y_pc = np.asarray(core["y_pc"][:], dtype=np.float64)
    z_full = np.asarray(core["z_pc"][:], dtype=np.float64)
    zsl, z_pc = pipe._box_z_slice(z_full)

    ix = np.where(np.abs(x_pc) <= STATS_BOX_XY_HALF_RANGE_PC)[0]
    iy = np.where(np.abs(y_pc) <= STATS_BOX_XY_HALF_RANGE_PC)[0]
    x_sub, y_sub = x_pc[ix], y_pc[iy]
    xsl = slice(int(ix.min()), int(ix.max()) + 1)
    ysl = slice(int(iy.min()), int(iy.max()) + 1)

    vg = core[VARIANT]
    print("Loading RAW him, n_H, p_th_phys and alpha over the STATS_BOX...")
    him = np.asarray(vg["him"][zsl]).astype(bool)[:, ysl, xsl]
    n_H = np.asarray(vg["n_model"][zsl], dtype=np.float32)[:, ysl, xsl]
    p_th = np.asarray(vg["p_th_phys"][zsl], dtype=np.float32)[:, ysl, xsl]
    alpha = np.asarray(vg[f"self_gravity_{SELF_GRAVITY}"]["alpha"][zsl],
                       dtype=np.float32)[:, ysl, xsl]
    p_tot = np.asarray(vg[f"self_gravity_{SELF_GRAVITY}"]["Ptot"][zsl],
                       dtype=np.float32)[:, ysl, xsl]
    sigma_gas = np.asarray(vg["Sigma_gas"][:], dtype=np.float32)[ysl, xsl]
    nz, ny, nx = him.shape
    n_cols = ny * nx
    print(f"  box {him.shape}; {n_cols} columns; flagged fraction {him.mean():.4f}")

    abs_z = np.abs(z_pc)
    dz = float(np.diff(np.sort(np.unique(abs_z)))[0])

    # ---- (a) z_clean map -------------------------------------------------
    # Walk |z| outward and record, for each column, the height at which it
    # first stops being clean. Done as a loop over the ~201 distinct |z|
    # values rather than a full (nz, ny, nx) float array, which would be
    # 0.8 GB for no reason.
    print("Building the z_clean map...")
    z_first_flag, z_clean, midplane_flagged = column_clean_maps(him, abs_z)
    never_flagged = ~np.isfinite(z_first_flag)
    print(f"  columns never flagged inside the box: {int(never_flagged.sum())} "
          f"({never_flagged.mean():.4%}); midplane flagged: {midplane_flagged.mean():.4%}")

    # ---- (b) clean fractions vs Z_clean ----------------------------------
    print("Clean fractions vs Z_clean...")
    frac_strict = np.full(len(Z_CLEAN_GRID_PC), np.nan)
    frac_loose = {t: np.full(len(Z_CLEAN_GRID_PC), np.nan) for t in LOOSE_THRESHOLDS}
    him_col_frac_by_Z = {}
    for i, Z in enumerate(Z_CLEAN_GRID_PC):
        col_frac = him_column_fraction(him, abs_z, Z)
        him_col_frac_by_Z[Z] = col_frac
        frac_strict[i] = float((col_frac == 0.0).mean())
        for t in LOOSE_THRESHOLDS:
            frac_loose[t][i] = (col_frac < t).mean()
        # cross-check the two independent routes to the same number.
        # "clean to Z" is z_first_flag > Z -- NOT z_clean >= Z, which is
        # wrong whenever Z is not a grid height (at Z = 25 pc a column
        # first flagged at |z| = 26 is clean but has z_clean = 24).
        from_map = clean_fraction(z_first_flag, Z)
        if not np.isclose(from_map, frac_strict[i], atol=1e-12):
            raise RuntimeError(
                f"z_clean map and the direct count disagree at Z={Z}: "
                f"{from_map} vs {frac_strict[i]}")

    # ---- (c) alpha in clean columns vs all columns -----------------------
    print("Alpha in clean columns vs all columns...")
    est_names = ("alpha_median_of_ratios", "alpha_mean_of_ratios",
                 "alpha_ratio_of_means", "alpha_p15_of_ratios", "alpha_p85_of_ratios")
    res = {}
    for pop in ("clean", "all"):
        for wt in ("vol", "mw"):
            for e in est_names:
                res[f"{e}__{pop}__{wt}"] = np.full(len(Z_CLEAN_ALPHA_PC), np.nan)
        res[f"n_cells__{pop}"] = np.zeros(len(Z_CLEAN_ALPHA_PC), dtype=np.int64)
        res[f"n_columns__{pop}"] = np.zeros(len(Z_CLEAN_ALPHA_PC), dtype=np.int64)

    for i, Z in enumerate(Z_CLEAN_ALPHA_PC):
        planes = np.where(abs_z <= Z)[0]
        clean_cols = ~him[planes].any(axis=0)          # (ny, nx)
        a_sub = alpha[planes].astype(np.float64)
        pt_sub = p_tot[planes].astype(np.float64)
        pth_sub = p_th[planes].astype(np.float64)
        n_sub = n_H[planes].astype(np.float64)
        him_sub = him[planes]

        # clean columns: every cell in them (there are no HIM cells there
        # by construction, so "excluding HIM" would be a no-op)
        # all columns: HIM cells excluded, which is the project default
        masks = {
            "clean": np.broadcast_to(clean_cols, him_sub.shape),
            "all": ~him_sub,
        }
        res["n_columns__clean"][i] = int(clean_cols.sum())
        res["n_columns__all"][i] = n_cols

        for pop, mask in masks.items():
            res[f"n_cells__{pop}"][i] = int(mask.sum())
            for wt in ("vol", "mw"):
                w = (np.where(mask, 1.0, 0.0) if wt == "vol"
                     else np.where(mask & np.isfinite(n_sub), n_sub, 0.0))
                st = weighted_stats(a_sub, w, PERCENTILE_SCHEME_DEFAULT)
                res[f"alpha_median_of_ratios__{pop}__{wt}"][i] = st.median
                res[f"alpha_mean_of_ratios__{pop}__{wt}"][i] = st.arithmetic_mean
                res[f"alpha_p15_of_ratios__{pop}__{wt}"][i] = st.p_lo
                res[f"alpha_p85_of_ratios__{pop}__{wt}"][i] = st.p_hi
                res[f"alpha_ratio_of_means__{pop}__{wt}"][i] = ratio_of_means(
                    pt_sub, pth_sub, w)
        print(f"  Z={Z:.0f}: clean columns {res['n_columns__clean'][i]} "
              f"({res['n_columns__clean'][i] / n_cols:.4%}), "
              f"cells clean {res['n_cells__clean'][i]} vs all {res['n_cells__all'][i]}")

    # ---- (d) selection bias ---------------------------------------------
    print("Selection-bias check...")
    iz_mid = int(np.argmin(np.abs(z_pc)))
    n_mid = n_H[iz_mid].astype(np.float64)
    pth_mid = p_th[iz_mid].astype(np.float64)
    bias = {}
    for i, Z in enumerate(Z_CLEAN_ALPHA_PC):
        planes = np.where(abs_z <= Z)[0]
        clean_cols = ~him[planes].any(axis=0)
        for name, arr in (("Sigma_gas", sigma_gas.astype(np.float64)),
                          ("n_mid", n_mid), ("Pth_mid", pth_mid)):
            for pop, sel in (("clean", clean_cols), ("all", np.ones_like(clean_cols))):
                key = f"bias_{name}_median__{pop}"
                bias.setdefault(key, np.full(len(Z_CLEAN_ALPHA_PC), np.nan))
                vals = arr[sel]
                vals = vals[np.isfinite(vals)]
                bias[key][i] = float(np.median(vals)) if vals.size else np.nan

    # ---- orientation: re-oriented map, verified -------------------------
    z_clean_true, x_true, y_true = to_true_orientation(z_clean, x_sub, y_sub, M)
    back, _, _ = to_true_orientation(z_clean_true, x_true, y_true, M)
    if not np.array_equal(back, z_clean):
        raise RuntimeError("to_true_orientation is not an involution for this mapping; "
                           "the re-oriented map cannot be trusted.")
    # marked-cell check on the real grid: a cube cell at (x_c, y_c) must
    # land at true (M^T applied), i.e. x_true = -y_c and y_true = -x_c for
    # the (-y,-x,+z) mapping -- derived from M, not assumed.
    jx, jy = int(np.argmin(np.abs(x_sub - 200.0))), int(np.argmin(np.abs(y_sub - 100.0)))
    probe = np.zeros_like(z_clean)
    probe[jy, jx] = 1.0
    probe_true, _, _ = to_true_orientation(probe, x_sub, y_sub, M)
    jt = np.argwhere(probe_true == 1.0)
    xt_expect = M[0, 0] * x_sub[jx] + M[1, 0] * y_sub[jy]
    yt_expect = M[0, 1] * x_sub[jx] + M[1, 1] * y_sub[jy]
    got = (x_true[jt[0, 1]], y_true[jt[0, 0]])
    if not np.allclose(got, (xt_expect, yt_expect)):
        raise RuntimeError(f"marked-cell check failed: {got} != {(xt_expect, yt_expect)}")
    print(f"  orientation transform verified: cube (200, 100) -> true "
          f"({got[0]:.0f}, {got[1]:.0f}) pc; involution holds")

    # ---- (e) Step 1h: the allowed-HIM-fraction grid ----------------------
    print("Allowed-HIM-fraction grid...")
    grid, grid_lines = compute_fhim_grid(
        him, abs_z, alpha, p_tot, p_th, n_H, sigma_gas, iz_mid, pct_lo, pct_hi)
    grid["provisional"] = np.array([PROVISIONAL_CUBE_HEADER])
    grid["variant"] = np.array([VARIANT])
    grid["self_gravity"] = np.array([SELF_GRAVITY])
    np.savez_compressed(OUT_GRID_NPZ_PATH, **grid)
    print(f"Saved {OUT_GRID_NPZ_PATH}")

    # ---- save ------------------------------------------------------------
    out = dict(res)
    out.update(bias)
    out.update({
        "z_clean_cube": z_clean,
        "z_first_flag_cube": z_first_flag,
        "z_clean_true": z_clean_true,
        "midplane_flagged_cube": midplane_flagged.astype(np.int8),
        "x_cube_pc": x_sub, "y_cube_pc": y_sub,
        "x_true_pc": x_true, "y_true_pc": y_true,
        "orientation_label": np.array([label]),
        "orientation_matrix": M,
        "Z_clean_grid_pc": np.array(Z_CLEAN_GRID_PC, dtype=float),
        "Z_clean_alpha_pc": np.array(Z_CLEAN_ALPHA_PC, dtype=float),
        "frac_clean_strict": frac_strict,
        "loose_thresholds": np.array(LOOSE_THRESHOLDS, dtype=float),
        "percentile_levels": np.array([pct_lo, pct_hi], dtype=float),
        "n_columns_total": np.array([n_cols]),
        "self_gravity": np.array([SELF_GRAVITY]),
        "variant": np.array([VARIANT]),
        "provisional": np.array([PROVISIONAL_CUBE_HEADER]),
    })
    for t in LOOSE_THRESHOLDS:
        out[f"frac_clean_lt{int(t * 100)}pct"] = frac_loose[t]
    np.savez_compressed(OUT_NPZ_PATH, **out)
    print(f"Saved {OUT_NPZ_PATH}")

    # ---- report ----------------------------------------------------------
    L = [
        PROVISIONAL_CUBE_HEADER,
        THERMAL_PRESSURE_HEADER,
        "HIM-free (\"clean\") columns: survival rate, and alpha in what survives",
        "=" * 100,
        f"Variant {VARIANT} (observed density), self-gravity {SELF_GRAVITY}, "
        f"STATS_BOX |x|,|y| <= {STATS_BOX_XY_HALF_RANGE_PC:.0f} pc, "
        f"|z| <= {STATS_BOX_Z_HALF_RANGE_PC:.0f} pc.",
        f"{n_cols} columns ({ny} x {nx}) on the {dz:.0f} pc grid. "
        f"HIM flag unchanged: p_nT < 0.5 P_min.",
        "A column is CLEAN to Z if no HIM-flagged cell in it has |z| <= Z.",
        "",
        "(a) fraction of columns that are clean to Z_clean",
        f"{'Z_clean [pc]':>13}{'strict':>12}{'HIM frac < 5%':>16}{'HIM frac < 10%':>16}",
        "-" * 57,
    ]
    for i, Z in enumerate(Z_CLEAN_GRID_PC):
        L.append(f"{Z:>13.0f}{frac_strict[i]:>12.5f}"
                 f"{frac_loose[0.05][i]:>16.5f}{frac_loose[0.10][i]:>16.5f}")
    L += [
        "-" * 57,
        f"  Columns never flagged anywhere in the box: {int(never_flagged.sum())} "
        f"of {n_cols} ({never_flagged.mean():.5%}).",
        "",
        "(b) alpha in clean columns vs ALL columns (HIM cells excluded), same |z| range",
        f"{'Z':>5}{'wt':>5}{'population':>12}{'columns':>10}{'cells':>12}"
        f"{'median':>10}{'mean':>10}{'rat.means':>11}"
        f"{'p' + format(pct_lo, 'g'):>9}{'p' + format(pct_hi, 'g'):>9}",
        "-" * 93,
    ]
    for i, Z in enumerate(Z_CLEAN_ALPHA_PC):
        for wt in ("vol", "mw"):
            for pop in ("clean", "all"):
                L.append(
                    f"{Z:>5.0f}{wt:>5}{pop:>12}"
                    f"{res[f'n_columns__{pop}'][i]:>10d}{res[f'n_cells__{pop}'][i]:>12d}"
                    f"{res[f'alpha_median_of_ratios__{pop}__{wt}'][i]:>10.4f}"
                    f"{res[f'alpha_mean_of_ratios__{pop}__{wt}'][i]:>10.4f}"
                    f"{res[f'alpha_ratio_of_means__{pop}__{wt}'][i]:>11.4f}"
                    f"{res[f'alpha_p15_of_ratios__{pop}__{wt}'][i]:>9.4f}"
                    f"{res[f'alpha_p85_of_ratios__{pop}__{wt}'][i]:>9.4f}")
        L.append("-" * 93)
    L += [
        "",
        "(c) selection bias: are the surviving columns typical? (medians)",
        f"{'Z':>5}{'Sigma_gas clean':>18}{'all':>10}{'ratio':>8}"
        f"{'n_H(z=0) clean':>17}{'all':>10}{'ratio':>8}"
        f"{'Pth(z=0) clean':>17}{'all':>10}{'ratio':>8}",
        "-" * 105,
    ]
    for i, Z in enumerate(Z_CLEAN_ALPHA_PC):
        row = f"{Z:>5.0f}"
        for name, fmt in (("Sigma_gas", "9.4f"), ("n_mid", "8.4f"), ("Pth_mid", "9.1f")):
            c = bias[f"bias_{name}_median__clean"][i]
            a = bias[f"bias_{name}_median__all"][i]
            width = 18 if name == "Sigma_gas" else 17
            row += f"{c:>{width}.4g}{a:>10.4g}{(c / a if a else float('nan')):>8.3f}"
        L.append(row)
    L += [
        "-" * 105,
        "  Sigma_gas is the FULL-column integral (+-750 pc) and is not restricted to |z| <= Z;",
        "  it is listed against Z only because the clean-column SET changes with Z.",
        "",
        f"Map orientation: {label}, read from {ORIENTATION_TXT}. The cache carries both the",
        "  cube-indexed z_clean map and a copy re-indexed onto true Galactic axes for display.",
    ]
    L += grid_lines
    L += [
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
