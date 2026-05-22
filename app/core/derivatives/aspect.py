"""
Aspect — compass direction of steepest downslope.

Reference: Horn (1981).

Convention: 0° = North, increases clockwise; flat cells → -1.

IMPORTANT: aspect is circular data.  For statistics or ML features, always
decompose via sin/cos.  The helper functions northness() and eastness() are
provided for that purpose.
"""
from __future__ import annotations
import numpy as np
from app.core.derivatives._gradient import horn_gradients


def aspect(
    dem: np.ndarray,
    cell_size: float,
    z_factor: float = 1.0,
    nodata: float = None,
    flat_threshold: float = 1e-6,
) -> np.ndarray:
    """
    Returns aspect in degrees [0, 360), -1 for flat cells.
    """
    dz_dx, dz_dy = horn_gradients(dem, cell_size, z_factor)

    # atan2(dx, -dy) gives the downslope direction in geographic convention
    # (0=North, clockwise).  Both dx and dy are eastward/northward, so:
    # downslope_east = -dz_dx,  downslope_north = -dz_dy
    # geographic angle from north: atan2(-dz_dx, dz_dy) ... let's verify sign:
    # Slope faces north (dz_dy>0, uphill is north) → downslope is south = 180°
    # atan2(-0, -dz_dy<0) = atan2(0, pos) = 0° → need 180°. So:
    # aspect = atan2(-dz_dx, -dz_dy) → with dz_dx=0, dz_dy>0:
    #          = atan2(0, -1) = π = 180° ✓
    asp = np.degrees(np.arctan2(-dz_dx, -dz_dy)) % 360.0

    flat = (np.abs(dz_dx) < flat_threshold) & (np.abs(dz_dy) < flat_threshold)
    asp[flat] = -1.0

    result = asp.astype(np.float32)
    if nodata is not None:
        result[dem == nodata] = -1.0
    return result


def northness(asp_deg: np.ndarray) -> np.ndarray:
    """cos(aspect): +1 = north-facing, -1 = south-facing."""
    valid = asp_deg >= 0
    out = np.zeros_like(asp_deg, dtype=np.float32)
    out[valid] = np.cos(np.radians(asp_deg[valid]))
    return out


def eastness(asp_deg: np.ndarray) -> np.ndarray:
    """sin(aspect): +1 = east-facing, -1 = west-facing."""
    valid = asp_deg >= 0
    out = np.zeros_like(asp_deg, dtype=np.float32)
    out[valid] = np.sin(np.radians(asp_deg[valid]))
    return out
