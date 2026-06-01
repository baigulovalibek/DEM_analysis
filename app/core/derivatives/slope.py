"""
Slope — maximum rate of elevation change.

Reference: Horn (1981) — default algorithm identical to gdaldem / QGIS.
Alternative: Zevenbergen & Thorne (1987) quadratic fit (see below).
"""
from __future__ import annotations
import numpy as np
from app.core.derivatives._gradient import horn_gradients
from app.core._nodata import missing_mask


def slope(
    dem: np.ndarray,
    cell_size: float,
    z_factor: float = 1.0,
    units: str = "degrees",
    nodata: float | None = None,
) -> np.ndarray:
    """
    Parameters
    ----------
    dem       : 2-D elevation array
    cell_size : horizontal resolution (same unit as dem elevations when z_factor=1)
    z_factor  : vertical exaggeration
    units     : "degrees" | "percent" | "radians"
    nodata    : masked to NaN in result (including the one-pixel rim where the
                3×3 stencil would otherwise inherit the sentinel value)

    Returns
    -------
    float32 array, same shape as dem
    """
    dz_dx, dz_dy = horn_gradients(dem, cell_size, z_factor, nodata=nodata)
    with np.errstate(invalid="ignore"):
        grad_mag = np.sqrt(dz_dx ** 2 + dz_dy ** 2)

        if units == "radians":
            result = np.arctan(grad_mag).astype(np.float32)
        elif units == "percent":
            result = (grad_mag * 100.0).astype(np.float32)
        else:
            result = np.degrees(np.arctan(grad_mag)).astype(np.float32)

    # NaN propagates from neighbouring nodata via the gradient stencil; also
    # mask the centre cell itself (Horn's stencil uses only the 8 surrounding
    # cells, so a NaN centre with valid neighbours would otherwise come back
    # with a finite slope).
    result[missing_mask(dem, nodata)] = np.nan
    result[~np.isfinite(result)] = np.nan
    return result
