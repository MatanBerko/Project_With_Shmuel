"""
to_galactic_xy: the cube -> true Galactic re-indexing for the Fig. 2 maps.

Under "neg_swap" the true field at (x, y) is the cube at (x_c = -y,
y_c = -x) (results/orientation_check.txt). A single marked voxel at cube
(x_c = +100, y_c = -40) must therefore land at true (x = +40, y = -100).
"""

import numpy as np
import pytest

from src.conventions import CUBE_XY_MAPPING
from src.physics.loading import galactic_xy_coords, to_galactic_xy

X_C = np.arange(-200.0, 200.0 + 1, 2.0)    # 201 points, symmetric
Y_C = np.arange(-120.0, 120.0 + 1, 2.0)    # 121 points: a NON-square grid
Z_C = np.array([-2.0, 0.0, 2.0])


def _marked_cube():
    cube = np.zeros((Z_C.size, Y_C.size, X_C.size))     # [z, y_c, x_c]
    iy = int(np.argmin(np.abs(Y_C - (-40.0))))
    ix = int(np.argmin(np.abs(X_C - 100.0)))
    cube[:, iy, ix] = 1.0
    return cube


def test_marked_voxel_lands_at_true_position():
    out = to_galactic_xy(_marked_cube(), X_C, Y_C, mapping="neg_swap")
    x, y = galactic_xy_coords(X_C, Y_C, mapping="neg_swap")
    assert out.shape == (Z_C.size, y.size, x.size) == (3, X_C.size, Y_C.size)
    assert np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)
    for k in range(Z_C.size):
        j_y, i_x = np.argwhere(out[k] == 1.0)[0]
        assert (x[i_x], y[j_y]) == (40.0, -100.0)
    assert out.sum() == Z_C.size


def test_general_relation_holds_everywhere():
    rng = np.random.default_rng(0)
    cube = rng.random((Y_C.size, X_C.size))
    out = to_galactic_xy(cube, X_C, Y_C, mapping="neg_swap")
    x, y = galactic_xy_coords(X_C, Y_C, mapping="neg_swap")
    for _ in range(200):
        i, j = rng.integers(x.size), rng.integers(y.size)
        ixc = int(np.argmin(np.abs(X_C + y[j])))
        iyc = int(np.argmin(np.abs(Y_C + x[i])))
        assert out[j, i] == cube[iyc, ixc]


def test_true_y0_plane_from_single_cube_x_plane():
    cube = _marked_cube()
    ix0 = int(np.argmin(np.abs(X_C)))
    cube[:, :, ix0] = Y_C[None, :]          # value = y_c along the x_c = 0 plane
    plane = to_galactic_xy(cube[:, :, ix0:ix0 + 1], X_C[ix0:ix0 + 1], Y_C, "neg_swap")
    x, y = galactic_xy_coords(X_C[ix0:ix0 + 1], Y_C, "neg_swap")
    assert plane.shape == (Z_C.size, 1, Y_C.size) and y.tolist() == [0.0]
    np.testing.assert_array_equal(plane[0, 0], -x)    # true x = -y_c


def test_identity_returns_input_unchanged():
    cube = _marked_cube()
    out = to_galactic_xy(cube, X_C, Y_C, mapping="identity")
    assert out is cube
    x, y = galactic_xy_coords(X_C, Y_C, mapping="identity")
    assert x is not None and np.array_equal(x, X_C) and np.array_equal(y, Y_C)


def test_asymmetric_grid_and_bad_mapping_are_refused():
    with pytest.raises(ValueError, match="symmetric"):
        to_galactic_xy(np.zeros((3, 4)), np.arange(4.0), np.array([-2.0, 0.0, 2.0]),
                       mapping="neg_swap")
    with pytest.raises(ValueError, match="Unknown"):
        to_galactic_xy(np.zeros((3, 3)), np.array([-2.0, 0, 2]), np.array([-2.0, 0, 2]),
                       mapping="mirror")


def test_default_mapping_is_neg_swap_for_the_f98_cube():
    assert CUBE_XY_MAPPING == "neg_swap"
