"""
Multi-directional oblique-weighted hillshade.

Reference: Mark, R.K. (1992). Multidirectional, oblique-weighted, shaded-relief
           image of Hawaii. USGS Open-File Report 92-422.

Implemented as: weighted mean of 4 azimuths (225°, 270°, 315°, 360°) at a
fixed altitude, matching the gdaldem -multidirectional option.
"""
from __future__ import annotations
import numpy as np

from app.core.derivatives._gradient import horn_gradients


# Mark (1992) azimuths and weights
_MARK_AZIMUTHS = [225.0, 270.0, 315.0, 360.0]
_MARK_WEIGHTS  = [0.5, 1.0, 1.0, 0.5]   # centre azimuths weighted more


def multidirectional_hillshade(
    dem: np.ndarray,
    cell_size: float,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    azimuths: list[float] | None = None,
    weights: list[float] | None = None,
    nodata: float | None = None,
) -> np.ndarray:
    """
    Weighted multi-azimuth hillshade.

    Computes the Horn gradient once and reuses it for every azimuth — the
    previous implementation re-ran the gradient stencil four times.

    Returns a float32 array in [0, 255].
    """
    if azimuths is None:
        azimuths = _MARK_AZIMUTHS
    if weights is None:
        weights = _MARK_WEIGHTS

    dz_dx, dz_dy = horn_gradients(dem, cell_size, z_factor, nodata=nodata)
    alt_rad = np.radians(altitude)
    sin_alt = np.sin(alt_rad)
    cos_alt = np.cos(alt_rad)

    with np.errstate(invalid="ignore"):
        denominator = np.sqrt(dz_dx ** 2 + dz_dy ** 2 + 1.0)

        total_w = float(sum(weights))
        accum = np.zeros(dem.shape, dtype=np.float64)
        for az, w in zip(azimuths, weights):
            az_rad = np.radians(az)
            numerator = sin_alt - cos_alt * (
                dz_dx * np.sin(az_rad) + dz_dy * np.cos(az_rad)
            )
            hs = np.maximum(0.0, numerator / denominator) * 255.0
            accum += hs * w

        result = (accum / total_w).astype(np.float32)

    # Same masking convention as the per-azimuth hillshade — any NaN
    # propagated by the gradient stencil (centre nodata or any neighbour
    # nodata) renders as 0.
    result[~np.isfinite(result)] = 0.0
    return result
