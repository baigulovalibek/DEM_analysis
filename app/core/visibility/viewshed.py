"""
Viewshed analysis — cells visible from an observer point.

Algorithm: R3 line-of-sight (brute-force per-ray).  Each ray from observer is
sampled at sub-cell resolution; a cell is visible if the LoS slope to it exceeds
all intermediate terrain slopes.  O(n²) for n = max_radius cells.

Reference: Wang, Robinson & White (2000). PE&RS 66(1), 87-90.
           Franklin & Ray (1994). Spatial Data Handling.

Optional earth-curvature and atmospheric-refraction correction
(refraction ≈ 0.13 * curvature) is applied when requested.
"""
from __future__ import annotations
import numpy as np


REFRACTION_COEFF = 0.13   # standard atmospheric refraction coefficient


def viewshed(
    dem: np.ndarray,
    cell_size: float,
    observer_row: int,
    observer_col: int,
    observer_height: float = 1.8,
    target_height: float = 0.0,
    max_radius: float = None,
    correct_curvature: bool = True,
    nodata: float = None,
    _progress=None,
) -> np.ndarray:
    """
    Compute binary viewshed from a single observer point.

    Parameters
    ----------
    dem             : elevation grid
    cell_size       : ground resolution (metres)
    observer_row/col: integer grid coordinates of observer
    observer_height : height above ground of observer's eye (metres)
    target_height   : height above ground of target being observed
    max_radius      : maximum range (cells); None = entire grid
    correct_curvature : apply earth curvature + refraction adjustment

    Returns
    -------
    uint8 binary grid: 1 = visible, 0 = hidden/nodata
    """
    rows, cols = dem.shape
    visibility = np.zeros((rows, cols), dtype=np.uint8)

    obs_elev = dem[observer_row, observer_col]
    if nodata is not None and obs_elev == nodata:
        return visibility
    obs_elev += observer_height

    max_r = int(max_radius) if max_radius is not None else max(rows, cols)

    # Cast a ray to every target cell using Bresenham-like stepping
    for tr in range(rows):
        if _progress is not None and _progress(int(tr / max(rows, 1) * 100)):
            return visibility
        for tc in range(cols):
            if tr == observer_row and tc == observer_col:
                visibility[tr, tc] = 1
                continue
            if nodata is not None and dem[tr, tc] == nodata:
                continue

            dx = tc - observer_col
            dy = tr - observer_row
            dist_cells = np.sqrt(dx ** 2 + dy ** 2)
            if dist_cells > max_r:
                continue

            # Sample along the ray at cell-resolution steps
            n_steps = int(np.ceil(dist_cells))
            visible = True
            max_slope = -np.inf

            for step in range(1, n_steps + 1):
                frac = step / n_steps
                sr = observer_row + dy * frac
                sc = observer_col + dx * frac

                # Bilinear interpolation of terrain elevation
                r0, c0 = int(sr), int(sc)
                r1, c1 = min(r0 + 1, rows - 1), min(c0 + 1, cols - 1)
                wr, wc = sr - r0, sc - c0

                z_sample = (
                    dem[r0, c0] * (1 - wr) * (1 - wc)
                    + dem[r0, c1] * (1 - wr) * wc
                    + dem[r1, c0] * wr * (1 - wc)
                    + dem[r1, c1] * wr * wc
                )

                dist_m = step / n_steps * dist_cells * cell_size

                if correct_curvature and dist_m > 0:
                    R_earth = 6_371_000.0
                    curv_corr = (dist_m ** 2 / (2 * R_earth)) * (1 - REFRACTION_COEFF)
                    z_sample -= curv_corr

                if step < n_steps:
                    slope_to_sample = (z_sample - obs_elev) / dist_m
                    if slope_to_sample > max_slope:
                        max_slope = slope_to_sample
                else:
                    # Final target cell
                    z_target = z_sample + target_height
                    slope_to_target = (z_target - obs_elev) / dist_m
                    visible = slope_to_target >= max_slope

            if visible:
                visibility[tr, tc] = 1

    return visibility


def cumulative_viewshed(
    dem: np.ndarray,
    cell_size: float,
    observer_points: list[tuple[int, int]],
    observer_height: float = 1.8,
    **kwargs,
) -> np.ndarray:
    """
    Cumulative (total) viewshed: count how many observers can see each cell.
    Useful for site siting, archaeology, telecommunications.
    """
    accum = np.zeros(dem.shape, dtype=np.int32)
    for r, c in observer_points:
        accum += viewshed(dem, cell_size, r, c, observer_height, **kwargs).astype(np.int32)
    return accum
