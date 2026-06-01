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
    float32 TWI grid.  Cells with zero contributing area or non-finite slope
    are masked to NaN — those are outlets, nodata, and slope-stencil edges
    where TWI is not physically defined.
    """
    fa = flow_accum.astype(np.float64)
    sr = slope_rad.astype(np.float64)

    a = specific_catchment_area(fa, cell_size).astype(np.float64)
    beta = np.maximum(sr, slope_min)
    a = np.maximum(a, cell_size)   # floor at one cell width

    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.log(a / np.tan(beta)).astype(np.float32)

    # A cell only has a meaningful TWI when both inputs are defined AND it
    # has positive contributing area.  Zero-accumulation cells are usually
    # outlets, nodata, or cells outside the analysis mask — masking them
    # to NaN gives consumers a single uniform missing-data indicator.
    invalid = (fa <= 0) | ~np.isfinite(sr) | ~np.isfinite(result)
    result[invalid] = np.nan
    return result
