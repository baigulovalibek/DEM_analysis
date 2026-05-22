"""
Topographic Position Index (TPI).

TPI = z - mean(neighbourhood)

Reference: Weiss, A. (2001). ESRI User Conference poster.
           Guisan, Weiss & Weiss (1999). Plant Ecology 143, 107-122.

The annular window (doughnut shape, inner_r to outer_r) is standard in
Jenness (2006) to separate immediate vs. regional position.

Landform classification after Weiss (2001) / De Reu et al. (2013):
  Uses TPI at two scales + slope to produce 6 classes:
  Ridge, Upper slope, Middle slope, Flat, Lower slope, Valley
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter


def tpi(
    dem: np.ndarray,
    radius: int = 5,
    annular: bool = False,
    inner_radius: int = 1,
) -> np.ndarray:
    """
    Parameters
    ----------
    dem          : elevation array
    radius       : outer neighbourhood radius in cells
    annular      : if True, use an annular (doughnut) window — excludes
                   the inner_radius cells around centre, so TPI reflects
                   landscape position rather than local microtopography
    inner_radius : inner ring excluded when annular=True

    Returns
    -------
    float32 TPI
    """
    z = dem.astype(np.float64)
    window = 2 * radius + 1

    if annular and inner_radius >= 1:
        outer_sum = uniform_filter(z, size=window, mode="nearest") * window ** 2
        inner_w = 2 * inner_radius + 1
        inner_sum = uniform_filter(z, size=inner_w, mode="nearest") * inner_w ** 2
        n_outer = window ** 2 - inner_w ** 2
        mean_z = (outer_sum - inner_sum) / max(n_outer, 1)
    else:
        mean_z = uniform_filter(z, size=window, mode="nearest")

    return (z - mean_z).astype(np.float32)


def classify_landform(
    tpi_small: np.ndarray,
    tpi_large: np.ndarray,
    slope_deg: np.ndarray,
    slope_flat: float = 6.0,
    tpi_lo: float = -1.0,
    tpi_hi: float = 1.0,
) -> np.ndarray:
    """
    Weiss (2001) 10-class landform classification.

    Classes (1-10):
    1  Canyons / deeply incised streams
    2  Midslope drainages / shallow valleys
    3  Upland drainages / headwaters
    4  U-shaped valleys
    5  Plains / flat
    6  Open slopes
    7  Upper slopes / mesas
    8  Local ridges / hills in valleys
    9  Midslope ridges / small hills
    10 Mountain tops / high ridges

    Returns int8 class grid.
    """
    ts = tpi_small
    tl = tpi_large
    s = slope_deg
    lo, hi = tpi_lo, tpi_hi
    flat = slope_flat

    cls = np.zeros(ts.shape, dtype=np.int8)

    cls[(ts <= lo) & (tl <= lo)]                                 = 1   # Canyon
    cls[(ts <= lo) & (tl > lo) & (tl < hi)]                     = 2   # Midslope drain
    cls[(ts <= lo) & (tl >= hi)]                                 = 3   # Upland drain
    cls[(ts > lo) & (ts < hi) & (tl <= lo) & (s > flat)]        = 4   # U-valley
    cls[(ts > lo) & (ts < hi) & (s <= flat)]                     = 5   # Plain
    cls[(ts > lo) & (ts < hi) & (s > flat) & (tl > lo) & (tl < hi)] = 6  # Open slope
    cls[(ts > lo) & (ts < hi) & (tl >= hi)]                     = 7   # Upper slope
    cls[(ts >= hi) & (tl <= lo)]                                 = 8   # Local ridge
    cls[(ts >= hi) & (tl > lo) & (tl < hi)]                     = 9   # Midslope ridge
    cls[(ts >= hi) & (tl >= hi)]                                 = 10  # Mountain top

    return cls
