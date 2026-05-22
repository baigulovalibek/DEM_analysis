"""
Slope — maximum rate of elevation change.

Reference: Horn (1981) — default algorithm identical to gdaldem / QGIS.
Alternative: Zevenbergen & Thorne (1987) quadratic fit (see below).
"""
from __future__ import annotations
import numpy as np
from app.core.derivatives._gradient import horn_gradients


def slope(
    dem: np.ndarray,
    cell_size: float,
    z_factor: float = 1.0,
    units: str = "degrees",
    nodata: float = None,
) -> np.ndarray:
    """
    Parameters
    ----------
    dem       : 2-D elevation array
    cell_size : horizontal resolution (same unit as dem elevations when z_factor=1)
    z_factor  : vertical exaggeration
    units     : "degrees" | "percent" | "radians"
    nodata    : masked to NaN in result

    Returns
    -------
    float32 array, same shape as dem
    """
    dz_dx, dz_dy = horn_gradients(dem, cell_size, z_factor)
    grad_mag = np.sqrt(dz_dx ** 2 + dz_dy ** 2)

    if units == "radians":
        result = np.arctan(grad_mag).astype(np.float32)
    elif units == "percent":
        result = (grad_mag * 100.0).astype(np.float32)
    else:
        result = np.degrees(np.arctan(grad_mag)).astype(np.float32)

    if nodata is not None:
        result[dem == nodata] = np.nan
    return result
