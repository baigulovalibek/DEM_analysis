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
from app.core._nodata import missing_mask


def aspect(
    dem: np.ndarray,
    cell_size: float,
    z_factor: float = 1.0,
    nodata: float | None = None,
    flat_threshold: float | None = None,
) -> np.ndarray:
    """
    Returns aspect in degrees [0, 360), -1 for flat cells.

    ``flat_threshold`` is a gradient magnitude (rise / run).  When left as
    ``None`` it defaults to ``1e-3 / cell_size`` — a 1 mm elevation
    difference across one cell — so the threshold scales with grid spacing
    and large genuinely-flat regions are detected even on coarse DEMs.
    Pass an explicit numeric value to override.
    """
    dz_dx, dz_dy = horn_gradients(dem, cell_size, z_factor, nodata=nodata)

    if flat_threshold is None:
        # 1 mm of relief across one cell — small enough to not eat real
        # micro-relief, large enough that mm-precision DEMs don't generate
        # random aspect on physically flat plateaux.
        flat_threshold = 1e-3 / max(cell_size, 1e-9)

    # atan2(dx, -dy) gives the downslope direction in geographic convention
    # (0=North, clockwise).  Both dx and dy are eastward/northward, so:
    # downslope_east = -dz_dx,  downslope_north = -dz_dy
    # geographic angle from north: atan2(-dz_dx, dz_dy) ... let's verify sign:
    # Slope faces north (dz_dy>0, uphill is north) → downslope is south = 180°
    # atan2(-0, -dz_dy<0) = atan2(0, pos) = 0° → need 180°. So:
    # aspect = atan2(-dz_dx, -dz_dy) → with dz_dx=0, dz_dy>0:
    #          = atan2(0, -1) = π = 180° ✓
    with np.errstate(invalid="ignore"):
        asp = np.degrees(np.arctan2(-dz_dx, -dz_dy)) % 360.0

    flat = (np.abs(dz_dx) < flat_threshold) & (np.abs(dz_dy) < flat_threshold)
    asp[flat] = -1.0

    result = asp.astype(np.float32)
    # Mask the centre cell explicitly (Horn's stencil uses only the 8
    # surrounding cells, so a NaN centre with valid neighbours would
    # otherwise inherit a computed aspect).  Neighbour-NaN propagation
    # is handled by the second pass.
    result[missing_mask(dem, nodata)] = -1.0
    result[~np.isfinite(result)] = -1.0
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
