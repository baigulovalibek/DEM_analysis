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
    """
    a = specific_catchment_area(flow_accum, cell_size).astype(np.float64)
    beta = slope_rad.astype(np.float64)
    raw = a * np.tan(np.maximum(beta, 0.0))
    if log_scale:
        result = np.log1p(raw)
    else:
        result = raw
    return result.astype(np.float32)
