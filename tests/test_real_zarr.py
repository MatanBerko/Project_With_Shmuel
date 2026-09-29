"""
Real-data check: Sigma_gas(x,y) computed from the actual project zarr cube
(observed/RAW density, +-500 pc SQUARE footprint) must match
pipeline/compute_sigma_gas.py's own cached reference numbers (median 4.80,
arithmetic mean 6.90 Msun/pc^2 -- exact cached values 4.803458,
6.904051 -- both use the zarr "density" field directly as n_H, matching
this task's SETTLED convention) within 1%.

An earlier version of this test used median=4.77/mean=6.79. Those numbers
turned out to belong to a DIFFERENT existing script,
pipeline/compute_sigma_gas_factor_comparison.py's sigma_gas_old, which
re-derives n_H from a direct dustmaps query (1652 * dE/ds) on a coarser
10 pc grid, not the zarr's own density field. That script's result is an
independent cross-check that agrees with this one to ~2%, not the primary
reference for the zarr-density convention this core implements.
"""

import numpy as np

from src.physics.derived import sigma_gas_map
from src.physics.loading import footprint_mask, load_subcube, load_xy_subset_coords, open_zarr

REFERENCE_MEDIAN = 4.80
REFERENCE_MEAN = 6.90
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
