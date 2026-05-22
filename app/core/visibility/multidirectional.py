"""
Multi-directional oblique-weighted hillshade.

Reference: Mark, R.K. (1992). Multidirectional, oblique-weighted, shaded-relief
           image of Hawaii. USGS Open-File Report 92-422.

Implemented as: weighted mean of 4 azimuths (225°, 270°, 315°, 360°) at a
fixed altitude, matching the gdaldem -multidirectional option.
"""
from __future__ import annotations
import numpy as np
from app.core.derivatives.hillshade import hillshade


# Mark (1992) azimuths and weights
_MARK_AZIMUTHS = [225.0, 270.0, 315.0, 360.0]
_MARK_WEIGHTS  = [0.5, 1.0, 1.0, 0.5]   # centre azimuths weighted more


def multidirectional_hillshade(
    dem: np.ndarray,
    cell_size: float,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    azimuths: list[float] = None,
    weights: list[float] = None,
    nodata: float = None,
) -> np.ndarray:
    """
    Weighted multi-azimuth hillshade.

    Returns a float32 array in [0, 255].
    """
    if azimuths is None:
        azimuths = _MARK_AZIMUTHS
    if weights is None:
        weights = _MARK_WEIGHTS

    total_w = sum(weights)
    result = np.zeros(dem.shape, dtype=np.float64)

    for az, w in zip(azimuths, weights):
        hs = hillshade(dem, cell_size, azimuth=az, altitude=altitude,
                       z_factor=z_factor, nodata=nodata)
        result += hs.astype(np.float64) * w

    return (result / total_w).astype(np.float32)
