"""
Analytical hillshade (shaded relief).

Reference: Horn, B.K.P. (1981). Hill shading and the reflectance map.
           Proc. IEEE 69(1), 14-47.

The Lambertian reflectance model implemented here is identical to what
gdaldem, ArcGIS, and QGIS use as their default.
"""
from __future__ import annotations
import numpy as np
from app.core.derivatives._gradient import horn_gradients


def hillshade(
    dem: np.ndarray,
    cell_size: float,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    nodata: float = None,
) -> np.ndarray:
    """
    Compute hillshade.

    Parameters
    ----------
    dem       : 2-D float elevation array (north-up, row=0 is north)
    cell_size : ground resolution in the same units as elevations (z_factor=1)
                or in horizontal units when z_factor != 1
    azimuth   : geographic azimuth of the sun (degrees, 0=N clockwise)
    altitude  : solar elevation angle above the horizon (degrees)
    z_factor  : vertical exaggeration applied before gradient computation
    nodata    : value to treat as missing (result will be 0 there)

    Returns
    -------
    uint8 array in [0, 255], same shape as dem
    """
    dz_dx, dz_dy = horn_gradients(dem, cell_size, z_factor)

    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)

    # Lambertian dot-product with un-normalised surface normal (-dzdx, -dzdy, 1)
    # L (toward sun) in (East, North, Up): (sin φ · cos α,  cos φ · cos α,  sin α)
    numerator = (
        np.sin(alt_rad)
        - np.cos(alt_rad) * (dz_dx * np.sin(az_rad) + dz_dy * np.cos(az_rad))
    )
    denominator = np.sqrt(dz_dx ** 2 + dz_dy ** 2 + 1.0)

    hs = np.maximum(0.0, numerator / denominator) * 255.0

    result = hs.astype(np.float32)
    if nodata is not None:
        result[dem == nodata] = 0.0
    return result
