"""
Zarr loading helpers: coordinate arrays, XY footprint mask, and
memory-conscious (chunked, dask-backed) access to the density/T/Iuv cubes.

Grid spacing is NEVER hardcoded here -- every dz/dx/dy used downstream comes
from the coordinate arrays returned by open_zarr().

The cube path comes from exactly one config entry (local_config.yaml's
zarr_filename, via src.config_loader) -- open_zarr() is the only place any
code in this project opens the density cube, so swapping the provisional
f98 cube for the re-oriented one, and later for the final Porter-FUV cube,
is a one-line config change.

Two spatial selections live here and must not be conflated:
  * footprint_mask() / load_subcube() -- the +-500 pc XY SQUARE. Applied
    when loading, because every quantity (P_tot integration and
    self-gravity footprint average included) is restricted in XY.
  * stats_box_z_mask() / slab_z_indices() -- the STATS_BOX |z| <= 400 pc
    and the 60 pc single-height slabs. Applied only when REDUCING to
    statistics; the P_tot integral and the self-gravity footprint average
    deliberately keep the full z column the cube provides.
"""

from dataclasses import dataclass

import dask.array as da
import numpy as np
import zarr

from src.config_loader import load_resolved_config
from src.conventions import SLAB_HALF_THICKNESS_PC, STATS_BOX_Z_HALF_RANGE_PC

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


def stats_box_z_mask(z_pc: np.ndarray,
                       z_half_range_pc: float = STATS_BOX_Z_HALF_RANGE_PC) -> np.ndarray:
    """STATS_BOX z selection: |z| <= z_half_range_pc, as a 1D boolean mask
    over the cube's own z coordinate array.

    Paired with footprint_mask() this is the full Shelest+26 analysis
    volume (1 kpc x 1 kpc x 800 pc centred on the Sun). It is NOT applied
    to the P_tot integration or to the self-gravity footprint average,
    both of which need the full column the cube provides -- see
    STATS_BOX_* in src.conventions for why.
    """
    return np.abs(np.asarray(z_pc, dtype=float)) <= z_half_range_pc


def stats_box_z_indices(z_pc: np.ndarray,
                          z_half_range_pc: float = STATS_BOX_Z_HALF_RANGE_PC) -> np.ndarray:
    """Indices of the z planes inside the STATS_BOX, ascending."""
    return np.where(stats_box_z_mask(z_pc, z_half_range_pc))[0]


def slab_z_indices(z_pc: np.ndarray, z_center_pc: float,
                     half_thickness_pc: float = SLAB_HALF_THICKNESS_PC,
                     z_half_range_pc: float = STATS_BOX_Z_HALF_RANGE_PC) -> np.ndarray:
    """Indices of the z planes in the single-height slab
    |z - z_center_pc| <= half_thickness_pc, clipped to the STATS_BOX.

    Default half thickness is the Shelest+26 30 pc (60 pc-thick slab).
    The STATS_BOX clip is a no-op for the three standard slab centres
    (0/150/300 pc, so at most |z| = 330 <= 400) but keeps the guarantee
    that no statistic ever reads a plane outside the analysis volume.
    """
    z = np.asarray(z_pc, dtype=float)
    in_slab = np.abs(z - z_center_pc) <= half_thickness_pc
    return np.where(in_slab & stats_box_z_mask(z, z_half_range_pc))[0]


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
