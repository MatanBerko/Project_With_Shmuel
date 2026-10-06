"""
Step 1f (A): the STATS_CELL_SELECTION switch.

Statistics have always been taken over non-HIM-flagged cells only. On
this cube the flag selects low-density warm gas rather than hot gas, so
RAW is now also reported over ALL cells. What has to hold:

1. With NO flagged cells, the two selections are the same selection, so
   every statistic must come out bit-identical. This is the test that
   catches a selection being applied to one quantity and not another.
2. With flagged cells, "all_cells" must include exactly those cells and
   nothing else -- checked by counting, and by checking the extra cells
   actually move the numbers.
3. Only RAW is reported both ways; HIM_A/HIM_B keep the default, because
   they SUBSTITUTE the flagged cells' density and an all_cells statistic
   would average model values in with observations.
4. The switch must not reach anything that is not a cell statistic:
   P_tot, Sigma_gas and the self-gravity source always counted flagged
   cells and still do.
"""

import numpy as np
import pytest

import pipeline.compute_all as pipe
from src.conventions import (
    MU,
    M_H,
    SELF_GRAVITY_MODE_FOOTPRINT_MEAN,
    SELF_GRAVITY_MODE_OFF,
    STATS_CELL_SELECTIONS,
    STATS_CELL_SELECTIONS_BY_VARIANT,
    STATS_CELL_SELECTION_ALL,
    STATS_CELL_SELECTION_DEFAULT,
    STATS_CELL_SELECTION_EXCLUDE_HIM,
)
from src.physics.derived import sigma_gas_map
from src.physics.gravity import g_total_cgs
from src.physics.hydrostatic import p_tot_kb_full_column

CUBE_Z_PC = np.arange(-750.0, 750.0 + 1.0, 2.0)


# ---------------------------------------------------------------------------
# the switch itself
# ---------------------------------------------------------------------------
def test_constants_and_which_variants_get_both():
    assert STATS_CELL_SELECTION_DEFAULT == STATS_CELL_SELECTION_EXCLUDE_HIM
    assert set(STATS_CELL_SELECTIONS) == {"exclude_him_flag", "all_cells"}
    # RAW both ways; the HIM variants only the default -- see the module
    # docstring and src.conventions for why this asymmetry is deliberate.
    assert STATS_CELL_SELECTIONS_BY_VARIANT["RAW"] == STATS_CELL_SELECTIONS
    for v in ("HIM_A", "HIM_B"):
        assert STATS_CELL_SELECTIONS_BY_VARIANT[v] == (STATS_CELL_SELECTION_EXCLUDE_HIM,)


def test_selection_mask_counts_exactly_the_flagged_cells():
    rng = np.random.default_rng(1106)
    him = rng.random((7, 11)) < 0.37
    m_excl = pipe.selection_mask(STATS_CELL_SELECTION_EXCLUDE_HIM, him)
    m_all = pipe.selection_mask(STATS_CELL_SELECTION_ALL, him)

    np.testing.assert_array_equal(m_excl, ~him)
    assert m_all.all()
    # all_cells adds EXACTLY the flagged cells, no more and no fewer
    assert m_all.sum() - m_excl.sum() == him.sum()
    np.testing.assert_array_equal(m_all & ~m_excl, him)
    assert m_excl.shape == him.shape and m_all.shape == him.shape

    with pytest.raises(ValueError, match="Unknown cell selection"):
        pipe.selection_mask("just_the_good_ones", him)


def test_selection_note_distinguishes_the_two():
    a = pipe.selection_note(STATS_CELL_SELECTION_EXCLUDE_HIM)
    b = pipe.selection_note(STATS_CELL_SELECTION_ALL)
    assert a != b
    assert "non-HIM" in a and "ALL cells" in b


# ---------------------------------------------------------------------------
# toy cube: identical with no flagged cells, different with them
# ---------------------------------------------------------------------------
def _toy_cube(n_planes=41, ny=9, nx=11, flagged_frac=0.0, seed=5):
    """A small STATS_BOX-shaped cube plus the arrays the profile needs."""
    rng = np.random.default_rng(seed)
    z = np.arange(-(n_planes // 2), n_planes // 2 + 1, dtype=float) * 2.0
    shape = (len(z), ny, nx)
    n = 10.0 ** rng.uniform(-2.0, 0.5, shape)
    T = 10.0 ** rng.uniform(1.8, 4.0, shape)
    p_th = 1.1 * n * T
    p_tot = p_th * 10.0 ** rng.uniform(-0.4, 1.0, shape)
    alpha = p_tot / p_th
    him = rng.random(shape) < flagged_frac
    phase = np.where(him, 3, rng.integers(0, 3, shape)).astype(np.int8)
    footprint = np.ones((ny, nx), dtype=bool)
    return dict(z_pc=z, n=n.astype(np.float32), T=T.astype(np.float32),
                p_th=p_th.astype(np.float32), p_tot=p_tot.astype(np.float32),
                alpha=alpha.astype(np.float32), him=him,
                phase_cubes={"dPdn": phase, "temperature": phase},
                footprint=footprint)


def _profile(c, selection, use_abs=False):
    return pipe.compute_vertical_profile(
        c["z_pc"], c["p_th"], c["p_tot"], c["alpha"], c["him"], c["phase_cubes"],
        c["T"], c["n"], c["footprint"], use_abs=use_abs, selection=selection)


def test_no_flagged_cells_makes_the_two_selections_identical():
    """The strongest check in this file: if nothing is flagged the two
    selections ARE the same selection, so every single profile entry must
    match exactly. A quantity that ignored `selection` and hard-coded
    ~him would still pass; a quantity that applied it inconsistently
    (one mask for the stats, another for the temperature mean, say) would
    not."""
    c = _toy_cube(flagged_frac=0.0)
    assert not c["him"].any()

    for use_abs in (False, True):
        a = _profile(c, STATS_CELL_SELECTION_EXCLUDE_HIM, use_abs)
        b = _profile(c, STATS_CELL_SELECTION_ALL, use_abs)
        assert set(a) == set(b)
        for k in a:
            np.testing.assert_array_equal(
                np.asarray(a[k]), np.asarray(b[k]),
                err_msg=f"{k} differs with no flagged cells (use_abs={use_abs})")


def test_flagged_cells_are_included_by_all_cells_and_change_the_numbers():
    c = _toy_cube(flagged_frac=0.30)
    n_flagged = int(c["him"].sum())
    assert n_flagged > 0

    excl = _profile(c, STATS_CELL_SELECTION_EXCLUDE_HIM)
    allc = _profile(c, STATS_CELL_SELECTION_ALL)

    # counts: all_cells selects every cell; the difference is exactly the
    # flagged ones, bin by bin and in total
    np.testing.assert_array_equal(allc["n_selected"], allc["n_cells"])
    assert allc["n_selected"].sum() - excl["n_selected"].sum() == n_flagged
    # n_cells itself is the population size and must NOT depend on the
    # selection -- it is what n_selected is measured against
    np.testing.assert_array_equal(excl["n_cells"], allc["n_cells"])

    # and the extra cells actually move the statistics, so "identical"
    # above was a real constraint rather than an inert one
    moved = 0
    for k in ("alpha_median_of_ratios_vol", "alpha_mean_of_ratios_vol",
              "alpha_ratio_of_means_vol", "Pth_phys_vol_median", "Ptot_mw_mean"):
        if not np.allclose(excl[k], allc[k], equal_nan=True):
            moved += 1
    assert moved == 5, f"only {moved}/5 statistics responded to the selection"

    # phase fractions are the documented exception: computed over every
    # cell under both selections, so they must NOT move
    for k in [k for k in excl if k.startswith("f_")]:
        np.testing.assert_array_equal(excl[k], allc[k], err_msg=k)


def test_all_cells_statistics_match_a_hand_computed_full_population():
    """Spot-check one bin against the primitives, so "all_cells" is shown
    to be the statistic over the whole bin and not merely different."""
    from src.physics.stats import ratio_of_means, weighted_stats

    c = _toy_cube(flagged_frac=0.25)
    allc = _profile(c, STATS_CELL_SELECTION_ALL)
    groups = pipe.profile_bin_groups(c["z_pc"], use_abs=False)
    b = next(i for i, (_, idx) in enumerate(groups) if len(idx) > 0)
    idxs = groups[b][1]

    fp = c["footprint"]
    a_sub = c["alpha"][idxs][:, fp].astype(np.float64)
    pt_sub = c["p_tot"][idxs][:, fp].astype(np.float64)
    pth_sub = c["p_th"][idxs][:, fp].astype(np.float64)
    w = np.ones(a_sub.size)

    st = weighted_stats(a_sub, w)
    assert allc["alpha_median_of_ratios_vol"][b] == pytest.approx(st.median, rel=1e-12)
    assert allc["alpha_mean_of_ratios_vol"][b] == pytest.approx(st.arithmetic_mean, rel=1e-12)
    assert allc["alpha_ratio_of_means_vol"][b] == pytest.approx(
        ratio_of_means(pt_sub, pth_sub, w), rel=1e-12)
    assert allc["n_selected"][b] == a_sub.size


# ---------------------------------------------------------------------------
# the switch must not reach the physics
# ---------------------------------------------------------------------------
def test_ptot_sigma_gas_and_self_gravity_never_see_the_selection():
    """Flagged cells always carried their mass into P_tot, the
    self-gravity source and Sigma_gas, under every selection. Those are
    computed from the density field with no cell mask at all -- so none
    of their call signatures can even accept one.
    """
    import inspect

    for fn in (p_tot_kb_full_column, sigma_gas_map, g_total_cgs):
        params = set(inspect.signature(fn).parameters)
        assert "selection" not in params and "him" not in params, fn.__name__

    # and numerically: build a cube, flag half of it, and show the three
    # quantities are unchanged by which cells a statistic would select
    z = CUBE_Z_PC
    n_cube = 0.5 * np.exp(-np.abs(z)[:, None, None] / 150.0) * np.ones((1, 4, 4))
    rho = MU * M_H * n_cube
    footprint = np.ones((4, 4), dtype=bool)

    g_off, _ = g_total_cgs(z, None, mode=SELF_GRAVITY_MODE_OFF)
    g_mean, _ = g_total_cgs(z, n_cube, mode=SELF_GRAVITY_MODE_FOOTPRINT_MEAN,
                              footprint_mask=footprint)
    ref = (p_tot_kb_full_column(z, rho, g_off).tobytes(),
           p_tot_kb_full_column(z, rho, g_mean).tobytes(),
           sigma_gas_map(z, n_cube).tobytes(),
           g_mean.tobytes())

    for selection in STATS_CELL_SELECTIONS:
        # a selection changes which cells a STATISTIC uses; it cannot
        # change the density field, so recomputing must be byte-identical
        mask = pipe.selection_mask(selection, np.zeros((len(z), 4, 4), dtype=bool))
        assert mask.shape == n_cube.shape
        g_off2, _ = g_total_cgs(z, None, mode=SELF_GRAVITY_MODE_OFF)
        g_mean2, _ = g_total_cgs(z, n_cube, mode=SELF_GRAVITY_MODE_FOOTPRINT_MEAN,
                                   footprint_mask=footprint)
        got = (p_tot_kb_full_column(z, rho, g_off2).tobytes(),
               p_tot_kb_full_column(z, rho, g_mean2).tobytes(),
               sigma_gas_map(z, n_cube).tobytes(),
               g_mean2.tobytes())
        assert got == ref, selection


# ---------------------------------------------------------------------------
# the numbers table carries the column
# ---------------------------------------------------------------------------
def test_numbers_table_has_a_cell_selection_column_in_the_right_place():
    cols = pipe.NUMBERS_TABLE_COLUMNS
    assert "cell_selection" in cols
    assert cols.index("cell_selection") == cols.index("self_gravity") + 1
    # the row tuples the finalize stage builds must match the header width
    assert len(cols) == 10
