"""
Sentinel-to-NaN conversion shared by every stencil-based algorithm.

Many DEM formats encode missing samples as a sentinel value (commonly
``-9999``).  If that sentinel is left in place, edge-padding plus a 3×3
stencil leaks the sentinel into neighbouring cells — producing 9999-unit
elevation drops next to lakes, voids, and data borders.  Every gradient,
curvature, ruggedness, horizon, or interpolation routine therefore needs
to normalise the input before any neighbour arithmetic runs.

The helper here is the one place that conversion happens, so each module
just calls ``to_nan_float64(dem, nodata)`` and lets the existing
``np.isfinite`` guards downstream do their job.
"""
from __future__ import annotations
import numpy as np


def to_nan_float64(dem: np.ndarray, nodata: float | None = None) -> np.ndarray:
    """Return a float64 copy of ``dem`` with the sentinel replaced by NaN.

    NaN samples in the source are preserved.  ``nodata`` may be ``None``,
    in which case the array is just upcast.  The output is always a fresh
    array so callers may mutate it without affecting the input.
    """
    z = dem.astype(np.float64, copy=True)
    if nodata is not None:
        z[z == nodata] = np.nan
    return z


def missing_mask(dem: np.ndarray, nodata: float | None = None) -> np.ndarray:
    """Boolean mask of cells that are missing (NaN or equal to ``nodata``)."""
    mask = ~np.isfinite(dem)
    if nodata is not None:
        mask = mask | (dem == nodata)
    return mask
