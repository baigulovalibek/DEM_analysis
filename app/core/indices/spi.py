"""
Stream Power Index (SPI) — proxy for erosive power of overland flow.

SPI = a · tan β

Reference: Moore et al. (1991). Hydrological Processes 5(1), 3-30.
"""
from __future__ import annotations
import numpy as np
from app.core.hydrology.flow_accumulation import specific_catchment_area


def spi(
    flow_accum: np.ndarray,
    slope_rad: np.ndarray,
    cell_size: float,
    log_scale: bool = True,
) -> np.ndarray:
    """
    Parameters
    ----------
    log_scale : if True return ln(SPI+1) to compress the dynamic range

    Cells with zero contributing area or non-finite slope (outlets, nodata,
    slope-stencil edges) are masked to NaN so consumers don't mistake
    "no data" for genuinely zero erosive power.
    """
    fa = flow_accum.astype(np.float64)
    sr = slope_rad.astype(np.float64)

    a = specific_catchment_area(fa, cell_size).astype(np.float64)
    with np.errstate(invalid="ignore"):
        raw = a * np.tan(np.maximum(sr, 0.0))
        if log_scale:
            result = np.log1p(raw)
        else:
            result = raw

    invalid = (fa <= 0) | ~np.isfinite(sr) | ~np.isfinite(result)
    out = result.astype(np.float32)
    out[invalid] = np.nan
    return out
