"""
Real-data check: Sigma_gas(x,y) computed from the actual project zarr cube
(observed/RAW density, +-500 pc SQUARE footprint) must match the
independently-verified reference numbers (median 4.77, arithmetic mean
6.79 Msun/pc^2) within 1%.
"""

import numpy as np

from src.physics.derived import sigma_gas_map
from src.physics.loading import footprint_mask, load_subcube, load_xy_subset_coords, open_zarr

REFERENCE_MEDIAN = 4.77
REFERENCE_MEAN = 6.79
TOLERANCE = 0.01


def test_sigma_gas_median_and_mean_match_reference():
    grid = open_zarr()
    n_cm3 = load_subcube(grid, "density", half_range_pc=500.0)
    x_sub, y_sub = load_xy_subset_coords(grid, half_range_pc=500.0)

    sigma = sigma_gas_map(grid.z_pc, n_cm3)
    mask = footprint_mask(x_sub, y_sub, half_range_pc=500.0)

    vals = sigma[mask]
    vals = vals[np.isfinite(vals)]

    median = float(np.median(vals))
    mean = float(np.mean(vals))

    assert abs(median - REFERENCE_MEDIAN) / REFERENCE_MEDIAN < TOLERANCE, median
    assert abs(mean - REFERENCE_MEAN) / REFERENCE_MEAN < TOLERANCE, mean
