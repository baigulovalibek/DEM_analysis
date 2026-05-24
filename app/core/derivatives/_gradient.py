"""
Horn (1981) weighted finite-difference gradient, shared by all derivative modules.

Neighbourhood labelling (north-up raster, row 0 = north):

    z1  z2  z3      z[r-1, c-1]  z[r-1, c]  z[r-1, c+1]
    z4  z5  z6  =   z[r,   c-1]  z[r,   c]  z[r,   c+1]
    z7  z8  z9      z[r+1, c-1]  z[r+1, c]  z[r+1, c+1]

Eastward gradient  (positive = elevation increases eastward):
    dz/dx = [(z3+2z6+z9) - (z1+2z4+z7)] / (8L)

Northward gradient (positive = elevation increases northward):
    dz/dy = [(z1+2z2+z3) - (z7+2z8+z9)] / (8L)
    (rows increase southward in raster storage, so we negate the raw row-diff)
"""
from __future__ import annotations
import numpy as np

from app.core._dem_cache import (
    get_cached_gradient,
    put_cached_gradient,
    to_nan_float64_cached,
)


def horn_gradients(
    dem: np.ndarray,
    cell_size: float,
    z_factor: float = 1.0,
    nodata: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (dz_dx, dz_dy): eastward and northward gradients, same shape as dem.
    Edge cells are handled by replicating the boundary (edge-padding).

    Sentinel ``nodata`` values are converted to NaN before the stencil runs
    so that the contamination from a -9999 neighbour cannot leak into the
    derived gradient — NaN poisons its 3×3 window instead, which the
    downstream ``np.isfinite`` guards mask out correctly.

    A small process-wide LRU memoises the result against the DEM buffer's
    fingerprint plus the cell_size / z_factor / nodata triple.  Hillshade,
    slope, aspect, and multidirectional-hillshade all start from the same
    gradient, so chaining them no longer re-runs the 3×3 stencil four
    times.  Returned arrays are read-only — call ``.copy()`` if you need
    to mutate.
    """
    cached = get_cached_gradient(dem, cell_size, z_factor, nodata)
    if cached is not None:
        return cached

    z = to_nan_float64_cached(dem, nodata) * z_factor
    zp = np.pad(z, 1, mode="edge")

    z1 = zp[:-2, :-2]
    z2 = zp[:-2, 1:-1]
    z3 = zp[:-2, 2:]
    z4 = zp[1:-1, :-2]
    z6 = zp[1:-1, 2:]
    z7 = zp[2:, :-2]
    z8 = zp[2:, 1:-1]
    z9 = zp[2:, 2:]

    dz_dx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8.0 * cell_size)
    dz_dy = ((z1 + 2 * z2 + z3) - (z7 + 2 * z8 + z9)) / (8.0 * cell_size)

    put_cached_gradient(dem, cell_size, z_factor, nodata, dz_dx, dz_dy)
    return dz_dx, dz_dy
