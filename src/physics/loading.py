"""
Zarr loading helpers: coordinate arrays, XY footprint mask, and
memory-conscious (chunked, dask-backed) access to the density/T/Iuv cubes.

Grid spacing is NEVER hardcoded here -- every dz/dx/dy used downstream comes
from the coordinate arrays returned by open_zarr().
"""

from dataclasses import dataclass

import dask.array as da
import numpy as np
import zarr

from src.config_loader import load_resolved_config

RHO_FIELD = "density"
IUV_FIELD = "Iuv_final"
T_FIELD = "T"


@dataclass
class ZarrGrid:
    store: object
    x_pc: np.ndarray
    y_pc: np.ndarray
    z_pc: np.ndarray


def open_zarr() -> ZarrGrid:
    """Open the project's zarr cube and read its coordinate arrays.

    Path is resolved only via src.config_loader -- never hardcoded.
    """
    cfg = load_resolved_config()
    store = zarr.open(str(cfg["zarr_path"]), mode="r")
    x_pc = np.asarray(store["x_pc"][:], dtype=np.float64).ravel()
    y_pc = np.asarray(store["y_pc"][:], dtype=np.float64).ravel()
    z_pc = np.asarray(store["z_pc"][:], dtype=np.float64).ravel()
    return ZarrGrid(store=store, x_pc=x_pc, y_pc=y_pc, z_pc=z_pc)


def grid_spacing(coord: np.ndarray) -> float:
    """Uniform grid spacing read from a coordinate array (never hardcoded)."""
    diffs = np.diff(coord)
    spacing = float(diffs[0])
    if not np.allclose(diffs, spacing, rtol=1e-6):
        raise ValueError("Coordinate array is not uniformly spaced.")
    return spacing


def footprint_mask(x_pc: np.ndarray, y_pc: np.ndarray, half_range_pc: float = 500.0) -> np.ndarray:
    """SETTLED square footprint: |x| <= half_range_pc AND |y| <= half_range_pc.

    x_pc, y_pc are 1D coordinate arrays; returns a 2D (len(y_pc), len(x_pc))
    boolean mask, matching the zarr's (z, y, x) axis order for a single
    z-plane.
    """
    X, Y = np.meshgrid(x_pc, y_pc)  # shape (y, x)
    return (np.abs(X) <= half_range_pc) & (np.abs(Y) <= half_range_pc)


def cylinder_mask(x_pc: np.ndarray, y_pc: np.ndarray, r_max_pc: float = 500.0) -> np.ndarray:
    """OLD footprint (R <= r_max_pc), kept only for regression comparisons."""
    X, Y = np.meshgrid(x_pc, y_pc)
    return np.sqrt(X ** 2 + Y ** 2) <= r_max_pc


def xy_index_range(coord: np.ndarray, half_range_pc: float) -> tuple[int, int]:
    """Index bounds [lo, hi) of a coordinate array within +-half_range_pc.

    Mirrors the pattern already used by pipeline/compute_sigma_gas.py: find
    the index range from the real coordinate array before touching the
    (potentially large) density array.
    """
    idx = np.where(np.abs(coord) <= half_range_pc)[0]
    return int(idx.min()), int(idx.max()) + 1


def load_subcube(grid: ZarrGrid, field: str, half_range_pc: float = 500.0) -> np.ndarray:
    """Load a (Nz, Ny, Nx) sub-cube of `field`, restricted to the XY square
    footprint, as a single in-memory float32 array via dask (chunked read).
    """
    x_lo, x_hi = xy_index_range(grid.x_pc, half_range_pc)
    y_lo, y_hi = xy_index_range(grid.y_pc, half_range_pc)
    arr = da.from_zarr(grid.store[field])[:, y_lo:y_hi, x_lo:x_hi]
    return arr.compute().astype(np.float32)


def load_xy_subset_coords(grid: ZarrGrid, half_range_pc: float = 500.0):
    """x_pc, y_pc restricted to the XY square footprint index range."""
    x_lo, x_hi = xy_index_range(grid.x_pc, half_range_pc)
    y_lo, y_hi = xy_index_range(grid.y_pc, half_range_pc)
    return grid.x_pc[x_lo:x_hi], grid.y_pc[y_lo:y_hi]


def iter_z_planes(grid: ZarrGrid, field: str, half_range_pc: float = 500.0):
    """Yield (iz, z_value, plane) for one field, one z-plane at a time,
    restricted to the XY square footprint -- for memory-conscious streaming
    over the full z column without holding the whole cube in memory.
    """
    x_lo, x_hi = xy_index_range(grid.x_pc, half_range_pc)
    y_lo, y_hi = xy_index_range(grid.y_pc, half_range_pc)
    field_arr = grid.store[field]
    for iz, z_val in enumerate(grid.z_pc):
        plane = np.asarray(field_arr[iz, y_lo:y_hi, x_lo:x_hi], dtype=np.float64)
        yield iz, float(z_val), plane
