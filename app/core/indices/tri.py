"""
Terrain Ruggedness Index (TRI).

Reference: Riley, S.J., DeGloria, S.D., Elliot, R. (1999).
           Intermountain Journal of Sciences 5(1-4), 23-27.

Formula: TRI = sqrt( sum_{i=1}^{8} (z_i - z_centre)^2 )
"""
from __future__ import annotations
import numpy as np


def tri(
    dem: np.ndarray,
    nodata: float = None,
) -> np.ndarray:
    """
    Riley et al. (1999) TRI over the 3×3 neighbourhood.

    Returns float32, same shape as dem.
    """
    z = dem.astype(np.float64)
    zp = np.pad(z, 1, mode="edge")

    z5 = z   # centre
    sq_sum = np.zeros_like(z)

    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1),  (1, 0), (1, 1)]

    for dr, dc in offsets:
        nbr = zp[1 + dr: 1 + dr + z.shape[0],
                 1 + dc: 1 + dc + z.shape[1]]
        sq_sum += (nbr - z5) ** 2

    result = np.sqrt(sq_sum).astype(np.float32)
    if nodata is not None:
        result[dem == nodata] = np.nan
    return result
