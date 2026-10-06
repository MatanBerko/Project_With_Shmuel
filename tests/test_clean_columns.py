"""
Step 1g: HIM-free ("clean") columns, and the display re-orientation.

Two things are checked, both against values worked out by hand rather
than against the code's own output:

1. The column maps. On a toy cube with flagged cells placed deliberately,
   z_first_flag, z_clean and the clean fractions must come out exactly
   as counted by hand -- including the two cases the z_clean convention
   conflates (midplane flagged vs clean-only-at-the-midplane), and the
   off-grid Z where a naive `z_clean >= Z` test gives the wrong answer.
   Strict <= loose must hold for every threshold, since a column with no
   HIM at all also has HIM fraction below any positive threshold.

2. The display re-orientation. A marked cell in a cube-indexed array
   must land at the true-Galactic position the mapping implies, and the
   transform must be an involution for this mapping.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "compute_clean_columns",
    Path(__file__).resolve().parents[1] / "scripts" / "compute" / "compute_clean_columns.py")
ccc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ccc)


# ---------------------------------------------------------------------------
# a toy cube whose answers are counted by hand
# ---------------------------------------------------------------------------
# z grid: -8 .. +8 pc at 2 pc -> |z| in {0, 2, 4, 6, 8}
TOY_Z = np.arange(-8.0, 8.0 + 1.0, 2.0)
TOY_ABS_Z = np.abs(TOY_Z)
BOX_HALF = 8.0


def _toy_him():
    """4 columns, each with a deliberately chosen first-flag height.

    col (0,0): flagged at z = +4 only            -> first flag |z| = 4
    col (0,1): flagged at the midplane z = 0     -> first flag |z| = 0
    col (1,0): flagged at z = -2 only            -> first flag |z| = 2
               (the NEGATIVE side: |z| must fold)
    col (1,1): never flagged                     -> first flag inf
    """
    him = np.zeros((len(TOY_Z), 2, 2), dtype=bool)
    him[np.argmin(np.abs(TOY_Z - 4.0)), 0, 0] = True
    him[np.argmin(np.abs(TOY_Z - 0.0)), 0, 1] = True
    him[np.argmin(np.abs(TOY_Z + 2.0)), 1, 0] = True
    return him


def test_z_first_flag_z_clean_and_midplane_flag_are_hand_counted():
    him = _toy_him()
    zf, zc, midflag = ccc.column_clean_maps(him, TOY_ABS_Z, box_half_range_pc=BOX_HALF)

    np.testing.assert_array_equal(zf, np.array([[4.0, 0.0], [2.0, np.inf]]))
    # z_clean = first flag - 2 pc, floored at 0; the box edge when never flagged
    np.testing.assert_array_equal(zc, np.array([[2.0, 0.0], [0.0, BOX_HALF]]))
    np.testing.assert_array_equal(midflag, np.array([[False, True], [False, False]]))

    # The two cases z_clean = 0 conflates are distinguished only by
    # midplane_flagged: col (0,1) is flagged AT the midplane, col (1,0) is
    # clean at the midplane and flagged at |z| = 2.
    assert zc[0, 1] == zc[1, 0] == 0.0
    assert midflag[0, 1] and not midflag[1, 0]


def test_clean_fraction_matches_hand_counts_including_off_grid_Z():
    him = _toy_him()
    zf, zc, _ = ccc.column_clean_maps(him, TOY_ABS_Z, box_half_range_pc=BOX_HALF)

    # By hand: first flags are 4, 0, 2, inf.  "clean to Z" is first > Z.
    expected = {
        0.0: 3 / 4,    # all but the midplane-flagged column
        1.0: 3 / 4,    # off grid: 2 > 1, so the |z|=2 column is still clean
        2.0: 2 / 4,    # on grid: 2 > 2 is false, that column turns dirty
        3.0: 2 / 4,    # off grid: 4 > 3, so the |z|=4 column is still clean
        4.0: 1 / 4,    # on grid: only the never-flagged column survives
        8.0: 1 / 4,
    }
    for Z, want in expected.items():
        assert ccc.clean_fraction(zf, Z) == pytest.approx(want), Z

    # The off-grid case is exactly where a `z_clean >= Z` test goes wrong:
    # at Z = 3 the column first flagged at |z| = 4 is clean, but its
    # z_clean is 2. Show the naive test would disagree, so this is a real
    # constraint and not a restatement.
    assert float((zc >= 3.0).mean()) != pytest.approx(ccc.clean_fraction(zf, 3.0))


def test_him_column_fraction_and_strict_never_exceeds_loose():
    him = _toy_him()
    zf, _, _ = ccc.column_clean_maps(him, TOY_ABS_Z, box_half_range_pc=BOX_HALF)

    # at Z = 4 the planes are |z| in {0,2,4} -> 5 planes (z=0,+-2,+-4)
    frac = ccc.him_column_fraction(him, TOY_ABS_Z, 4.0)
    np.testing.assert_allclose(frac, np.array([[1 / 5, 1 / 5], [1 / 5, 0.0]]))

    # strict <= loose, for every Z and every positive threshold: a column
    # with no HIM at all also has HIM fraction below any threshold > 0
    for Z in (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
        strict = ccc.clean_fraction(zf, Z)
        col_frac = ccc.him_column_fraction(him, TOY_ABS_Z, Z)
        assert strict == pytest.approx(float((col_frac == 0.0).mean())), Z
        for thr in (0.05, 0.10, 0.5):
            loose = float((col_frac < thr).mean())
            assert strict <= loose + 1e-12, (Z, thr, strict, loose)


def test_all_flagged_and_none_flagged_edges():
    nz = len(TOY_Z)
    none_flagged = np.zeros((nz, 2, 2), dtype=bool)
    zf, zc, mid = ccc.column_clean_maps(none_flagged, TOY_ABS_Z, box_half_range_pc=BOX_HALF)
    assert np.all(~np.isfinite(zf)) and np.all(zc == BOX_HALF) and not mid.any()
    assert ccc.clean_fraction(zf, BOX_HALF) == 1.0

    all_flagged = np.ones((nz, 2, 2), dtype=bool)
    zf, zc, mid = ccc.column_clean_maps(all_flagged, TOY_ABS_Z, box_half_range_pc=BOX_HALF)
    assert np.all(zf == 0.0) and np.all(zc == 0.0) and mid.all()
    assert ccc.clean_fraction(zf, 0.0) == 0.0


# ---------------------------------------------------------------------------
# display re-orientation
# ---------------------------------------------------------------------------
def _mapping_from_label(label):
    M = np.zeros((3, 3))
    for k, part in enumerate(label.split(",")):
        part = part.strip()
        M[k, ccc.AXIS_NAMES.index(part[-1])] = -1.0 if part[0] == "-" else 1.0
    return M


def test_marked_cell_moves_to_the_expected_true_position():
    """For the cube's actual mapping (-y,-x,+z): cube x holds -y_true and
    cube y holds -x_true, so a feature at cube (x, y) = (+200, +100) must
    appear at true (x, y) = (-100, -200)."""
    x = np.arange(-300.0, 300.0 + 1.0, 100.0)   # -300..300
    y = x.copy()
    M = _mapping_from_label("-y,-x,+z")

    cube = np.zeros((len(y), len(x)))
    jx = int(np.argmin(np.abs(x - 200.0)))
    jy = int(np.argmin(np.abs(y - 100.0)))
    cube[jy, jx] = 7.0

    true_map, x_true, y_true = ccc.to_true_orientation(cube, x, y, M)
    where = np.argwhere(true_map == 7.0)
    assert where.shape == (1, 2), "the marked cell must land in exactly one place"
    got = (x_true[where[0, 1]], y_true[where[0, 0]])
    assert got == (-100.0, -200.0), got


def test_reorientation_is_an_involution_and_preserves_the_values():
    rng = np.random.default_rng(1107)
    x = np.arange(-200.0, 200.0 + 1.0, 50.0)
    y = x.copy()
    M = _mapping_from_label("-y,-x,+z")
    cube = rng.random((len(y), len(x)))

    true_map, x_true, y_true = ccc.to_true_orientation(cube, x, y, M)
    back, _, _ = ccc.to_true_orientation(true_map, x_true, y_true, M)
    np.testing.assert_array_equal(back, cube)
    # it is a relabelling, so the multiset of values is unchanged
    np.testing.assert_array_equal(np.sort(true_map.ravel()), np.sort(cube.ravel()))
    # ... and it is NOT the identity, so the test above is not vacuous
    assert not np.array_equal(true_map, cube)


def test_identity_mapping_is_a_no_op():
    x = np.arange(-200.0, 200.0 + 1.0, 50.0)
    cube = np.arange(float(len(x) * len(x))).reshape(len(x), len(x))
    M = _mapping_from_label("+x,+y,+z")
    out, xt, yt = ccc.to_true_orientation(cube, x, x, M)
    np.testing.assert_array_equal(out, cube)
    np.testing.assert_array_equal(xt, x)


def test_read_best_mapping_rejects_a_z_mixing_mapping(tmp_path, monkeypatch):
    bad = tmp_path / "orientation_check.txt"
    bad.write_text("  best mapping (-y,-z,+x): r = 0.9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="touches z"):
        ccc.read_best_mapping(bad)

    good = tmp_path / "ok.txt"
    good.write_text("  best mapping (-y,-x,+z): r = 0.94\n", encoding="utf-8")
    label, M = ccc.read_best_mapping(good)
    assert label == "(-y,-x,+z)"
    np.testing.assert_array_equal(M, _mapping_from_label("-y,-x,+z"))

    missing = tmp_path / "nope.txt"
    with pytest.raises(FileNotFoundError, match="check_orientation"):
        ccc.read_best_mapping(missing)


# ---------------------------------------------------------------------------
# the real cache, if it has been built
# ---------------------------------------------------------------------------
def test_real_cache_is_internally_consistent():
    path = Path("cache/core/clean_columns.npz")
    if not path.exists():
        pytest.skip("run scripts/compute/compute_clean_columns.py first")
    d = np.load(path, allow_pickle=False)

    Z = d["Z_clean_grid_pc"]
    strict = d["frac_clean_strict"]
    # strict <= loose <= 1, and all monotonically non-increasing in Z
    for thr in d["loose_thresholds"]:
        loose = d[f"frac_clean_lt{int(thr * 100)}pct"]
        assert np.all(strict <= loose + 1e-12)
        assert np.all(np.diff(loose) <= 1e-12), "loose fraction must not rise with Z"
    assert np.all(np.diff(strict) <= 1e-12), "strict fraction must not rise with Z"
    assert np.all((strict >= 0) & (strict <= 1))

    # the strict fraction must agree with the stored z_first_flag map
    zf = d["z_first_flag_cube"]
    for i, zz in enumerate(Z):
        assert ccc.clean_fraction(zf, zz) == pytest.approx(strict[i], abs=1e-12), zz

    # the re-oriented map is a relabelling of the same values
    np.testing.assert_array_equal(np.sort(d["z_clean_true"].ravel()),
                                    np.sort(d["z_clean_cube"].ravel()))

    # clean columns are a subset of all columns, so their count never exceeds it
    assert np.all(d["n_columns__clean"] <= d["n_columns__all"])
    assert np.all(d["n_cells__clean"] <= d["n_cells__all"])
