"""
Step 2d: the area-averaged P_tot / alpha and the sub-area scale test.

Three properties, each a constraint rather than a restatement of the
code:

1. THE IDENTITY. On a toy cube with a uniform g, the area average of the
   per-column P_tot equals the hydrostatic integral of the area-averaged
   density, EXACTLY. This is the thing the real script's validation step
   measures on the real cube, and it is what makes computing <P_tot>_area
   from the cached per-column P_tot legitimate in the first place. The
   toy's density varies strongly from column to column, so an
   implementation that quietly assumed a smooth field would fail it.
   The companion negative control: with a g that varies from column to
   column the identity must BREAK, otherwise the test is not testing the
   assumption it claims to.

2. N = 1 EQUALS THE FULL FOOTPRINT. The script computes the whole
   footprint two ways -- plain np.sum over the plane, and np.bincount
   over patch ids -- precisely so this can be checked. They are separate
   code paths, so agreement is evidence, not tautology.

3. NO HIM CELLS => alpha_area(vol) EQUALS THE CELL-BASED VOLUME-WEIGHTED
   RATIO OF MEANS. With every cell neutral, <P_tot>_area / <P_th>_vol =
   mean(P_tot)/mean(P_th) = sum(P_tot)/sum(P_th), which is exactly
   src.physics.stats.volume_weighted_ratio_of_means over all cells. The
   converse matters as much and is also asserted: as soon as some cells
   ARE flagged the two part company, because the numerator still averages
   over every cell while the denominator drops the flagged ones. That
   asymmetry is the fiducial treatment, not an oversight, so it is
   pinned down here.

The patch geometry is checked too (coverage, disjointness, equal side
lengths, and the odd-cell-count edge), since every number in the scale
test rests on it.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.conventions import MU, M_H
from src.physics import hydrostatic
from src.physics.stats import volume_weighted_ratio_of_means

_SPEC = importlib.util.spec_from_file_location(
    "compute_area_alpha",
    Path(__file__).resolve().parents[1] / "scripts" / "compute" / "compute_area_alpha.py")
caa = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(caa)


# ---------------------------------------------------------------------------
# a toy cube, deliberately lumpy
# ---------------------------------------------------------------------------
TOY_Z = np.arange(-20.0, 20.0 + 1.0, 2.0)   # 21 planes at 2 pc
TOY_NY, TOY_NX = 6, 6


def _toy_density(seed=11):
    """A density cube with real column-to-column structure.

    Log-normal, so a few columns carry most of the mass -- the regime
    where "average the area first" and "average the cells" genuinely
    differ, and where an identity that only held for smooth fields would
    show it.
    """
    rng = np.random.default_rng(seed)
    return np.exp(rng.normal(0.0, 1.2, size=(len(TOY_Z), TOY_NY, TOY_NX)))


def _uniform_g(value=3.0e-9):
    """g(z) the same for every column, as both fiducial settings give."""
    return np.full(len(TOY_Z), value) * (1.0 + 0.4 * np.abs(TOY_Z) / TOY_Z.max())


def _per_column_ptot(n_cm3, g_1d):
    """The cache's quantity: P_tot/k_B integrated independently per column."""
    return hydrostatic.p_tot_kb_full_column(TOY_Z, MU * M_H * n_cm3, g_1d)


# ---------------------------------------------------------------------------
# 1. the identity
# ---------------------------------------------------------------------------
def test_area_averaged_ptot_equals_integral_of_mean_density_under_uniform_g():
    n = _toy_density()
    g = _uniform_g()

    per_column = _per_column_ptot(n, g)
    area_mean = per_column.reshape(len(TOY_Z), -1).mean(axis=1)

    rho_bar = n.reshape(len(TOY_Z), -1).mean(axis=1)
    from_mean_profile = hydrostatic.p_tot_kb_full_column(
        TOY_Z, MU * M_H * rho_bar, g)

    assert np.allclose(area_mean, from_mean_profile, rtol=1e-12, atol=0.0)
    # ... and the field really is lumpy, so the agreement is not trivial:
    # the midplane P_tot must span a wide range from column to column.
    mid = per_column[len(TOY_Z) // 2].ravel()
    assert mid.max() / mid.min() > 3.0


def test_the_identity_breaks_when_g_varies_between_columns():
    """The negative control for the test above.

    If this passed too, the first test would be checking an algebraic
    identity of the integrator rather than the physical assumption that
    g(z) is horizontally uniform -- which is exactly the assumption the
    whole area-average argument rests on.
    """
    n = _toy_density()
    g_1d = _uniform_g()
    g_cube = g_1d[:, None, None] * np.linspace(0.5, 1.5, TOY_NY * TOY_NX
                                                 ).reshape(1, TOY_NY, TOY_NX)

    per_column = hydrostatic.p_tot_kb_full_column(TOY_Z, MU * M_H * n, g_cube)
    area_mean = per_column.reshape(len(TOY_Z), -1).mean(axis=1)

    rho_bar = n.reshape(len(TOY_Z), -1).mean(axis=1)
    from_mean_profile = hydrostatic.p_tot_kb_full_column(
        TOY_Z, MU * M_H * rho_bar, g_1d)

    interior = np.abs(TOY_Z) < TOY_Z.max()
    assert not np.allclose(area_mean[interior], from_mean_profile[interior],
                           rtol=1e-3)


# ---------------------------------------------------------------------------
# patch geometry
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_side", [1, 2, 4, 8])
def test_patches_tile_the_footprint_exactly_once(n_side):
    half = 500.0
    coord = np.arange(-half, half + 1.0, 2.0)   # 501 cells, as the real cube
    pid, side = caa.patch_ids(coord, coord, n_side, half_range_pc=half)

    assert side == pytest.approx(2 * half / n_side)
    assert pid.shape == (coord.size, coord.size)
    assert pid.min() == 0 and pid.max() == n_side * n_side - 1
    counts = np.bincount(pid.ravel(), minlength=n_side * n_side)
    assert counts.sum() == coord.size ** 2          # every cell, exactly once
    assert (counts > 0).all()                        # no empty patch
    # 501 is odd, so at n_side = 2 the two halves differ by one cell per
    # axis. Equal AREA, not equal count -- documented, and checked here so
    # a future change to the edge convention cannot pass unnoticed.
    per_axis = np.bincount(
        np.floor((coord + half) / side).clip(0, n_side - 1).astype(int),
        minlength=n_side)
    assert per_axis.max() - per_axis.min() <= 1


def test_last_patch_edge_is_closed():
    """The +half_range cells must land in the last patch, not past it."""
    half = 10.0
    coord = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
    pid, _ = caa.patch_ids(coord, coord, 2, half_range_pc=half)
    # x = +10 is the closed upper edge -> last patch, not an out-of-range id
    assert pid[-1, -1] == 3
    assert pid[0, 0] == 0


# ---------------------------------------------------------------------------
# 2. N = 1 equals the whole footprint
# ---------------------------------------------------------------------------
def _reduce_toy(n, ptot, p_th, him, n_side):
    """Run both of the script's accumulation paths over the toy cube.

    Returns (full_footprint_averages, patch_averages) -- the first from
    the plain np.sum path, the second from the np.bincount path.
    """
    coord_y = np.linspace(-10.0, 10.0, TOY_NY)
    coord_x = np.linspace(-10.0, 10.0, TOY_NX)
    pid, _ = caa.patch_ids(coord_x, coord_y, n_side, half_range_pc=10.0)
    pid_flat = pid.ravel()
    n_patches = n_side * n_side
    cnt_pid = np.bincount(pid_flat, minlength=n_patches).astype(float)

    acc_full = caa._empty_sums((1,))
    acc_patch = caa._empty_sums((1, n_patches))
    for k in range(len(TOY_Z)):
        args = (ptot[k].ravel(), p_th[k].ravel(), n[k].ravel(), ~him[k].ravel())
        caa._accumulate_plane_full(acc_full, [0], *args)
        caa._accumulate_plane_patches(acc_patch, [0], pid_flat, n_patches,
                                        cnt_pid, *args)
    return (caa.area_averages_from_sums(acc_full),
            caa.area_averages_from_sums(acc_patch))


def test_single_patch_reproduces_the_full_footprint():
    n = _toy_density()
    g = _uniform_g()
    ptot = _per_column_ptot(n, g)
    p_th = 1.1 * n * 8000.0
    rng = np.random.default_rng(3)
    him = rng.random(n.shape) < 0.25

    full, patch = _reduce_toy(n, ptot, p_th, him, n_side=1)
    for key in full:
        assert patch[key].shape == (1, 1)
        assert np.allclose(full[key], patch[key][:, 0], rtol=1e-12, atol=0.0), key


def test_patch_averages_recombine_into_the_full_footprint():
    """A patch split must conserve the underlying sums at every N.

    The per-patch MEANS are not averages of each other, but the sums they
    are built from are additive, so recombining them has to give the
    whole-footprint answer back. This is what makes a spread across
    patches a decomposition of the full number rather than a different
    measurement.
    """
    n = _toy_density(seed=5)
    ptot = _per_column_ptot(n, _uniform_g())
    p_th = 1.1 * n * 8000.0
    him = np.zeros(n.shape, dtype=bool)
    him[:, 0, :] = True   # one flagged row, so the two populations differ

    full, patch3 = _reduce_toy(n, ptot, p_th, him, n_side=3)
    # <P_tot>_area recombines as a cell-count-weighted mean of the patches
    coord = np.linspace(-10.0, 10.0, TOY_NX)
    pid, _ = caa.patch_ids(coord, coord, 3, half_range_pc=10.0)
    w = np.bincount(pid.ravel(), minlength=9).astype(float)
    recombined = np.average(patch3["Ptot_area"][0], weights=w)
    assert recombined == pytest.approx(full["Ptot_area"][0], rel=1e-12)


# ---------------------------------------------------------------------------
# 3. no HIM cells => alpha_area(vol) == the cell-based ratio of means
# ---------------------------------------------------------------------------
def test_with_no_him_cells_alpha_area_vol_equals_cell_based_ratio_of_means():
    n = _toy_density(seed=7)
    ptot = _per_column_ptot(n, _uniform_g())
    p_th = 1.1 * n * 8000.0
    him = np.zeros(n.shape, dtype=bool)

    full, _ = _reduce_toy(n, ptot, p_th, him, n_side=1)

    expected = volume_weighted_ratio_of_means(
        ptot.ravel(), p_th.ravel(), np.ones(ptot.size, dtype=bool))
    assert full["alpha_area_vol"][0] == pytest.approx(expected, rel=1e-12)


def test_with_him_cells_the_two_definitions_part_company():
    """The asymmetry is deliberate and must be visible.

    <P_tot>_area averages over EVERY cell (the flagged cells' observed
    mass is in the hydrostatic weight, so their P_tot is real), while
    <P_th>_neutral drops the flagged ones. The cell-based ratio of means
    over non-HIM cells drops them from BOTH. If these two ever agreed
    with flagged cells present, the numerator would have been filtered
    somewhere it should not be.
    """
    n = _toy_density(seed=9)
    ptot = _per_column_ptot(n, _uniform_g())
    p_th = 1.1 * n * 8000.0
    rng = np.random.default_rng(13)
    him = rng.random(n.shape) < 0.4

    full, _ = _reduce_toy(n, ptot, p_th, him, n_side=1)

    neutral_only = volume_weighted_ratio_of_means(
        ptot.ravel(), p_th.ravel(), ~him.ravel())
    assert full["alpha_area_vol"][0] != pytest.approx(neutral_only, rel=1e-6)


# ---------------------------------------------------------------------------
# the across-patch spread
# ---------------------------------------------------------------------------
def test_across_patch_spread_is_the_project_percentile_estimator():
    """Ordered, inclusive of NaN patches being dropped, and nearest-rank."""
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan])
    med, lo, hi, n_finite = caa.across_patch_spread(vals)
    assert n_finite == 5                      # the NaN patch is dropped
    assert lo <= med <= hi
    assert med == pytest.approx(3.0)
    # nearest-rank on 5 equal-weight values: 15% -> first, 85% -> last
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(5.0)


def test_across_patch_spread_of_all_nan_is_nan():
    med, lo, hi, n_finite = caa.across_patch_spread(np.full(4, np.nan))
    assert n_finite == 0
    assert np.isnan(med) and np.isnan(lo) and np.isnan(hi)


def test_area_averages_are_nan_where_a_patch_has_no_neutral_cells():
    """At large |z| ~97% of the volume is flagged, so empty-denominator
    patches are a real case, not a hypothetical one."""
    s = caa._empty_sums((2,))
    s["n_cells"][:] = 10.0
    s["sum_ptot"][:] = 100.0
    s["n_neutral"][0] = 4.0
    s["sum_pth"][0] = 40.0
    s["sum_n"][0] = 2.0
    s["sum_n_pth"][0] = 20.0
    # group 1 has no neutral cells at all
    out = caa.area_averages_from_sums(s)
    assert out["Ptot_area"][1] == pytest.approx(10.0)   # still defined
    assert np.isnan(out["Pth_neutral_vol"][1])
    assert np.isnan(out["Pth_neutral_mw"][1])
    assert np.isnan(out["alpha_area_vol"][1])
    assert out["alpha_area_vol"][0] == pytest.approx(10.0 / 10.0)
    assert out["alpha_area_mw"][0] == pytest.approx(10.0 / 10.0)
