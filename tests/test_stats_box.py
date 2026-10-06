"""
STATS_BOX, slabs and the percentile switch.

The STATS_BOX is the Shelest+26 analysis volume: |x|, |y| <= 500 pc and
|z| <= 400 pc. What makes it easy to get wrong is that it must apply to
statistics and NOT to the P_tot integration or the self-gravity footprint
average -- so this module checks both halves of that: that the selection
helpers really do restrict to the box, and that the quantities which are
supposed to span the full column still do.
"""

import numpy as np
import pytest

from src.conventions import (
    MU,
    M_H,
    PERCENTILE_LEVELS_BY_SCHEME,
    PERCENTILE_SCHEME_15_85,
    PERCENTILE_SCHEME_16_84,
    PERCENTILE_SCHEME_DEFAULT,
    SELF_GRAVITY_MODE_FOOTPRINT_MEAN,
    SLAB_CENTERS_PC,
    SLAB_HALF_THICKNESS_PC,
    STATS_BOX_XY_HALF_RANGE_PC,
    STATS_BOX_Z_HALF_RANGE_PC,
)
from src.physics.derived import sigma_gas_map
from src.physics.gravity import g_total_cgs
from src.physics.hydrostatic import p_tot_kb_full_column
from src.physics.loading import (
    footprint_mask,
    slab_z_indices,
    stats_box_z_indices,
    stats_box_z_mask,
)
from src.physics.stats import percentile_levels, volume_weighted_stats, weighted_stats

# The real cube's z grid: -750 to +750 pc at 2 pc spacing.
CUBE_Z_PC = np.arange(-750.0, 750.0 + 1.0, 2.0)
CUBE_XY_PC = np.arange(-1000.0, 1000.0 + 1.0, 2.0)


def test_stats_box_is_the_1kpc_x_1kpc_x_800pc_box():
    assert STATS_BOX_XY_HALF_RANGE_PC == 500.0
    assert STATS_BOX_Z_HALF_RANGE_PC == 400.0

    mask = stats_box_z_mask(CUBE_Z_PC)
    z_in = CUBE_Z_PC[mask]
    assert z_in.min() == -400.0 and z_in.max() == 400.0
    assert z_in.size == 401  # 800 pc / 2 pc + 1
    # nothing outside the box slips in, nothing inside is dropped
    assert not mask[np.abs(CUBE_Z_PC) > 400.0].any()
    assert mask[np.abs(CUBE_Z_PC) <= 400.0].all()

    idx = stats_box_z_indices(CUBE_Z_PC)
    np.testing.assert_array_equal(CUBE_Z_PC[idx], z_in)
    # contiguous, which pipeline/compute_all.py relies on to slice the cache
    np.testing.assert_array_equal(np.diff(idx), np.ones(idx.size - 1, dtype=idx.dtype))


def test_stats_box_xy_footprint_is_the_500pc_square():
    m = footprint_mask(CUBE_XY_PC, CUBE_XY_PC, STATS_BOX_XY_HALF_RANGE_PC)
    X, Y = np.meshgrid(CUBE_XY_PC, CUBE_XY_PC)
    assert m.sum() == 501 * 501
    assert np.abs(X[m]).max() == 500.0 and np.abs(Y[m]).max() == 500.0
    # a square, not a cylinder: the corner (500, 500) is included
    assert m[np.argmin(np.abs(CUBE_XY_PC - 500.0)), np.argmin(np.abs(CUBE_XY_PC - 500.0))]


def test_slabs_are_60pc_thick_at_the_three_standard_heights():
    assert SLAB_HALF_THICKNESS_PC == 30.0
    assert SLAB_CENTERS_PC == (0.0, 150.0, 300.0)

    for zc in SLAB_CENTERS_PC:
        idx = slab_z_indices(CUBE_Z_PC, zc)
        z = CUBE_Z_PC[idx]
        assert z.min() == zc - 30.0 and z.max() == zc + 30.0
        assert z.size == 31  # 60 pc / 2 pc + 1
        assert np.all(np.abs(z - zc) <= SLAB_HALF_THICKNESS_PC)
        # and every plane is inside the box
        assert np.all(np.abs(z) <= STATS_BOX_Z_HALF_RANGE_PC)


def test_slab_is_clipped_to_the_box_when_it_would_overhang():
    """A slab centred near the box edge must not reach outside it."""
    idx = slab_z_indices(CUBE_Z_PC, 390.0)
    z = CUBE_Z_PC[idx]
    assert z.max() == 400.0        # clipped, not 420
    assert z.min() == 360.0        # the inner side is untouched
    assert np.all(np.abs(z) <= STATS_BOX_Z_HALF_RANGE_PC)


def test_box_excludes_the_cube_edges_the_ptot_integral_still_needs():
    """The cube supplies |z| <= 750 pc; the box keeps only |z| <= 400. The
    planes in between are exactly the ones P_tot needs and statistics must
    not use -- this pins down that they are a real, non-empty set, so the
    distinction is not vacuous."""
    outside = np.abs(CUBE_Z_PC) > STATS_BOX_Z_HALF_RANGE_PC
    assert outside.sum() == 350
    assert np.abs(CUBE_Z_PC[outside]).max() == 750.0


def test_ptot_and_sigma_gas_still_span_the_full_column_not_the_box():
    """P_tot keeps P = 0 pinned at the CUBE's edge (+-750 pc), and
    Sigma_gas stays a full-column integral. If either were silently
    clipped to the STATS_BOX, both would change -- so compare the full
    column against a box-clipped column and require them to differ, with
    the full-column version being the larger one.
    """
    n0 = 0.5
    z_full = CUBE_Z_PC
    n_full = np.full_like(z_full, n0)
    rho_full = MU * M_H * n_full
    g_full, _ = g_total_cgs(z_full, None, mode="off")
    p_full = p_tot_kb_full_column(z_full, rho_full, g_full)

    box = stats_box_z_mask(z_full)
    z_box = z_full[box]
    n_box = n_full[box]
    rho_box = MU * M_H * n_box
    g_box, _ = g_total_cgs(z_box, None, mode="off")
    p_box = p_tot_kb_full_column(z_box, rho_box, g_box)

    i_full_mid = int(np.argmin(np.abs(z_full)))
    i_box_mid = int(np.argmin(np.abs(z_box)))
    # Integrating only over the box would understate midplane P_tot badly.
    assert p_full[i_full_mid] > 1.5 * p_box[i_box_mid]

    # Sigma_gas over the full column is likewise larger than a box-clipped
    # one, and for uniform density by exactly the ratio of the two column
    # lengths -- which pins down that it integrated +-750 pc, not +-400.
    sigma_full = sigma_gas_map(z_full, n_full)
    sigma_box = sigma_gas_map(z_box, n_box)
    assert sigma_full > sigma_box
    assert np.isclose(sigma_full / sigma_box, 1500.0 / 800.0, rtol=1e-12)


def test_self_gravity_footprint_mean_uses_every_z_plane_the_cube_provides():
    """footprint_mean g_gas at the top of the box must already include the
    gas between the box edge and the cube edge -- it is sourced from the
    full column, not from the box."""
    z = CUBE_Z_PC
    n_cube = np.zeros((len(z), 5, 5))
    # put ALL the gas outside the STATS_BOX, between |z| = 500 and 600 pc
    outside = (np.abs(z) >= 500.0) & (np.abs(z) <= 600.0)
    n_cube[outside, :, :] = 2.0
    footprint = np.ones((5, 5), dtype=bool)

    g_tot, nan_frac = g_total_cgs(z, n_cube, mode=SELF_GRAVITY_MODE_FOOTPRINT_MEAN,
                                    footprint_mask=footprint)
    g_ext_only, _ = g_total_cgs(z, None, mode="off")

    assert np.allclose(nan_frac, 0.0)
    # inside the box there is no gas at all, so g_gas must be 0 there ...
    i_mid = int(np.argmin(np.abs(z)))
    assert np.isclose(g_tot[i_mid], g_ext_only[i_mid], rtol=1e-12)
    # ... but beyond 600 pc the full outside-the-box column is enclosed,
    # which a box-only computation would have missed entirely.
    i_far = int(np.argmin(np.abs(z - 700.0)))
    assert g_tot[i_far] > g_ext_only[i_far] * 1.0001


# ---------------------------------------------------------------------------
# Percentile switch
# ---------------------------------------------------------------------------
def test_default_percentile_scheme_is_shelest_15_85():
    assert PERCENTILE_SCHEME_DEFAULT == PERCENTILE_SCHEME_15_85
    assert percentile_levels() == (15.0, 85.0)
    assert percentile_levels(PERCENTILE_SCHEME_16_84) == (16.0, 84.0)
    assert set(PERCENTILE_LEVELS_BY_SCHEME) == {PERCENTILE_SCHEME_15_85, PERCENTILE_SCHEME_16_84}
    with pytest.raises(ValueError, match="Unknown percentile scheme"):
        percentile_levels("one_sigma_ish")


def test_weighted_stats_reports_the_levels_it_used_and_both_schemes_work():
    # 0..999 with uniform weight: the nearest-rank weighted percentile at
    # level q picks the first value whose cumulative weight reaches q% of
    # the total, i.e. index ceil(q/100 * 1000) - 1.
    v = np.arange(1000.0)
    w = np.ones_like(v)

    s15 = weighted_stats(v, w, PERCENTILE_SCHEME_15_85)
    assert (s15.pct_lo, s15.pct_hi) == (15.0, 85.0)
    assert (s15.lo_label, s15.hi_label) == ("p15", "p85")
    assert s15.p_lo == 149.0 and s15.p_hi == 849.0

    s16 = weighted_stats(v, w, PERCENTILE_SCHEME_16_84)
    assert (s16.pct_lo, s16.pct_hi) == (16.0, 84.0)
    assert (s16.lo_label, s16.hi_label) == ("p16", "p84")
    assert s16.p_lo == 159.0 and s16.p_hi == 839.0

    # the switch must move the numbers -- i.e. it is actually wired through
    assert s15.p_lo < s16.p_lo and s15.p_hi > s16.p_hi
    # and must not touch median or mean
    assert s15.median == s16.median
    assert s15.arithmetic_mean == s16.arithmetic_mean == pytest.approx(499.5)


def test_empty_selection_still_reports_its_percentile_levels():
    s = volume_weighted_stats(np.array([1.0, 2.0]), np.array([False, False]))
    assert s.n == 0
    assert np.isnan(s.median) and np.isnan(s.p_lo) and np.isnan(s.p_hi)
    assert (s.pct_lo, s.pct_hi) == percentile_levels()
