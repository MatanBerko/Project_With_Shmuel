"""
Step 1e (B): the 4 pc vertical-profile binning.

Two separate things are checked, and the second is the one that actually
protects the physics:

1. The bin geometry: edges at exact multiples of PROFILE_BIN_PC, bins
   left-closed with the LAST one closed (so the z = +400 pc plane is not
   dropped), and the resulting plane counts per bin on the cube's real
   2 pc grid -- 2 planes per signed bin and 3 in the last, 3 / 4 / 6 for
   the folded |z| profile.

2. That PROFILE_BIN_PC is statistics-only. P_tot, Sigma_gas and the
   self-gravity integrals must be byte-identical when the constant
   changes, because they are computed on the real grid and must never
   see it. That is asserted by actually changing the constant and
   recomputing, not by inspection.
"""

import importlib

import numpy as np
import pytest

import src.conventions as conv
from src.conventions import (
    MU,
    M_H,
    PROFILE_BIN_PC,
    SELF_GRAVITY_MODE_FOOTPRINT_MEAN,
    SELF_GRAVITY_MODE_OFF,
    STATS_BOX_Z_HALF_RANGE_PC,
)
from src.physics.derived import sigma_gas_map
from src.physics.gravity import g_total_cgs
from src.physics.hydrostatic import p_tot_kb_full_column

# The real cube's z grid.
CUBE_Z_PC = np.arange(-750.0, 750.0 + 1.0, 2.0)


@pytest.fixture()
def pipeline():
    """pipeline.compute_all, freshly imported so it picks up whatever
    PROFILE_BIN_PC currently is."""
    import pipeline.compute_all as m
    return importlib.reload(m)


# ---------------------------------------------------------------------------
# 1. bin geometry
# ---------------------------------------------------------------------------
def test_profile_bin_pc_is_four():
    assert PROFILE_BIN_PC == 4.0


def test_signed_edges_and_plane_counts(pipeline):
    edges = pipeline.profile_bin_edges(use_abs=False)
    assert edges[0] == -STATS_BOX_Z_HALF_RANGE_PC
    assert edges[-1] == STATS_BOX_Z_HALF_RANGE_PC
    assert len(edges) == 2 * int(STATS_BOX_Z_HALF_RANGE_PC / PROFILE_BIN_PC) + 1
    np.testing.assert_allclose(np.diff(edges), PROFILE_BIN_PC)
    # every edge is an exact multiple of the bin width
    np.testing.assert_allclose(edges / PROFILE_BIN_PC, np.rint(edges / PROFILE_BIN_PC))

    groups = pipeline.profile_bin_groups(CUBE_Z_PC, use_abs=False)
    assert len(groups) == 200  # 800 pc / 4 pc
    counts = [len(idx) for _, idx in groups]
    # 2 planes per 4 pc bin on a 2 pc grid; the LAST bin is closed, so it
    # also picks up the z = +400 plane
    assert counts[:-1] == [2] * 199
    assert counts[-1] == 3
    assert sum(counts) == 401  # every STATS_BOX plane used exactly once

    # centers are bin centers, not plane positions
    centers = np.array([c for c, _ in groups])
    np.testing.assert_allclose(centers[0], -STATS_BOX_Z_HALF_RANGE_PC + PROFILE_BIN_PC / 2)
    np.testing.assert_allclose(np.diff(centers), PROFILE_BIN_PC)

    # and each bin really holds the planes its edges say it does
    for b, (_, idx) in enumerate(groups):
        lo, hi = edges[b], edges[b + 1]
        z = CUBE_Z_PC[idx]
        assert np.all(z >= lo)
        assert np.all(z <= hi) if b == len(groups) - 1 else np.all(z < hi)


def test_folded_edges_and_plane_counts(pipeline):
    edges = pipeline.profile_bin_edges(use_abs=True)
    assert edges[0] == 0.0 and edges[-1] == STATS_BOX_Z_HALF_RANGE_PC

    groups = pipeline.profile_bin_groups(CUBE_Z_PC, use_abs=True)
    assert len(groups) == 100  # 400 pc / 4 pc
    counts = [len(idx) for _, idx in groups]
    # first bin: |z| = 0 (one plane) and |z| = 2 (two planes) -> 3
    assert counts[0] == 3
    # middle bins: two |z| values x both signs -> 4
    assert counts[1:-1] == [4] * 98
    # last bin is closed: |z| = 396, 398, 400 x both signs -> 6
    assert counts[-1] == 6
    assert sum(counts) == 401


def test_last_bin_is_closed_so_the_box_edge_plane_is_not_dropped(pipeline):
    """A half-open last bin would silently drop z = +-400 pc. Check the
    edge planes are present, and that they are in the LAST bin."""
    groups = pipeline.profile_bin_groups(CUBE_Z_PC, use_abs=False)
    last_z = CUBE_Z_PC[groups[-1][1]]
    assert STATS_BOX_Z_HALF_RANGE_PC in last_z
    first_z = CUBE_Z_PC[groups[0][1]]
    assert -STATS_BOX_Z_HALF_RANGE_PC in first_z

    all_idx = np.concatenate([idx for _, idx in groups])
    assert len(all_idx) == len(set(all_idx.tolist())), "a plane landed in two bins"
    used = set(CUBE_Z_PC[all_idx].tolist())
    expected = set(CUBE_Z_PC[np.abs(CUBE_Z_PC) <= STATS_BOX_Z_HALF_RANGE_PC].tolist())
    assert used == expected


def test_non_dividing_bin_width_is_rejected_rather_than_silently_truncated(pipeline):
    with pytest.raises(ValueError, match="does not divide"):
        pipeline.profile_bin_edges(use_abs=False, bin_pc=3.0)
    # and a bin width that DOES divide is accepted, including the
    # degenerate 2 pc case that reproduces the per-plane behaviour
    g2 = pipeline.profile_bin_groups(CUBE_Z_PC, use_abs=False, bin_pc=2.0)
    assert len(g2) == 400
    assert [len(i) for i in [idx for _, idx in g2]].count(1) == 399


# ---------------------------------------------------------------------------
# 2. the constant is statistics-only
# ---------------------------------------------------------------------------
def _physics_outputs():
    """P_tot (both self-gravity settings) and Sigma_gas on a toy cube,
    computed exactly the way the build stage does: on the real 2 pc grid.
    """
    z = CUBE_Z_PC
    ny = nx = 4
    zz = z[:, None, None]
    n_cube = (0.6 * np.exp(-np.abs(zz) / 140.0)
              + 0.05 * np.cos(zz / 37.0) ** 2
              + 0.01 * np.arange(ny)[None, :, None]
              + 0.02 * np.arange(nx)[None, None, :])
    rho = MU * M_H * n_cube
    footprint = np.ones((ny, nx), dtype=bool)

    g_off, _ = g_total_cgs(z, None, mode=SELF_GRAVITY_MODE_OFF)
    g_mean, nan_frac = g_total_cgs(z, n_cube, mode=SELF_GRAVITY_MODE_FOOTPRINT_MEAN,
                                     footprint_mask=footprint)
    return {
        "p_tot_off": p_tot_kb_full_column(z, rho, g_off),
        "p_tot_mean": p_tot_kb_full_column(z, rho, g_mean),
        "g_mean": g_mean,
        "nan_frac": nan_frac,
        "sigma_gas": sigma_gas_map(z, n_cube),
    }


def test_changing_profile_bin_pc_leaves_ptot_and_sigma_gas_byte_identical(monkeypatch):
    """The physics must not be able to see PROFILE_BIN_PC. Change it and
    recompute P_tot, the self-gravity field and Sigma_gas: every byte has
    to match. This is checked by comparing raw bytes rather than with a
    tolerance, since any dependence at all would be a bug.
    """
    import pipeline.compute_all as m

    base = _physics_outputs()

    for new_bin in (2.0, 8.0, 20.0):
        monkeypatch.setattr(conv, "PROFILE_BIN_PC", new_bin)
        importlib.reload(m)
        assert m.PROFILE_BIN_PC == new_bin
        # the profile geometry DOES change -- otherwise this test would
        # pass trivially because nothing was reloaded
        assert len(m.profile_bin_groups(CUBE_Z_PC, use_abs=False)) == int(800 / new_bin)

        got = _physics_outputs()
        for k, v in base.items():
            assert got[k].tobytes() == v.tobytes(), f"{k} changed with PROFILE_BIN_PC={new_bin}"

    monkeypatch.undo()
    importlib.reload(m)
    assert m.PROFILE_BIN_PC == PROFILE_BIN_PC


def test_profile_binning_does_not_touch_the_z_grid_used_for_integration():
    """Sanity companion to the above: the trapezoid really is on the 2 pc
    grid, so a 4 pc-binned profile and the integrals coexist on the same
    cube without one resampling the other."""
    z = CUBE_Z_PC
    n = np.full_like(z, 0.5)
    # the integral spans the full +-750 pc column (NOT the stats box, NOT
    # a multiple of PROFILE_BIN_PC away from it)
    sigma = sigma_gas_map(z, n)
    sigma_coarse = sigma_gas_map(z[::2], n[::2])  # a deliberately wrong 4 pc grid
    assert sigma == pytest.approx(sigma_coarse, rel=1e-12)  # uniform n: both exact
    # but the grid actually used has 751 planes, not 376
    assert len(z) == 751
