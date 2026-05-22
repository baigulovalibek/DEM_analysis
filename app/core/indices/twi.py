"""
Topographic Wetness Index (TWI).

TWI = ln(a / tan β)

Reference: Beven & Kirkby (1979). Hydrological Sciences Bulletin 24(1), 43-69.

Notes:
  - a  = specific catchment area (flow_accum * cell_size)
  - β  = local slope in radians (floored at ε to avoid log(∞) on flat cells)
  - MFD/D∞ accumulation gives smoother, more realistic patterns than D8.
"""
from __future__ import annotations
import numpy as np
from app.core.hydrology.flow_accumulation import specific_catchment_area


def twi(
    flow_accum: np.ndarray,
    slope_rad: np.ndarray,
    cell_size: float,
    slope_min: float = 0.001,
    nodata: float = None,
) -> np.ndarray:
    """
    Parameters
    ----------
    flow_accum  : upstream contributing cell count (from accumulation)
    slope_rad   : slope in radians (same grid)
    cell_size   : ground resolution in metres
    slope_min   : minimum slope (radians) to floor flat cells

    Returns
    -------
    float32 TWI grid
    """
    a = specific_catchment_area(flow_accum, cell_size).astype(np.float64)
    beta = np.maximum(slope_rad.astype(np.float64), slope_min)
    # Guard against zero contributing area (outlet/nodata cells)
    a = np.maximum(a, cell_size)   # floor at one cell width
    result = np.log(a / np.tan(beta)).astype(np.float32)
    result[~np.isfinite(result)] = np.nan

    if nodata is not None:
        result[flow_accum == 0] = np.nan
    return result
