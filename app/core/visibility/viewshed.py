"""
Viewshed analysis — cells visible from an observer point.

Algorithm: R3 line-of-sight (brute-force per-ray).  Each ray from observer is
sampled at sub-cell resolution; a cell is visible if the LoS slope to it exceeds
all intermediate terrain slopes.  O(n²) for n = max_radius cells.

Reference: Wang, Robinson & White (2000). PE&RS 66(1), 87-90.
           Franklin & Ray (1994). Spatial Data Handling.

Optional earth-curvature and atmospheric-refraction correction
(refraction ≈ 0.13 * curvature) is applied when requested.

When Numba is available the per-target ray sweep runs as a
``@njit(parallel=True)`` kernel with the outer target-row loop
parallelized across CPU cores.  We cap concurrency at
``min(os.cpu_count(), 8)`` so the JIT kernel doesn't starve the Qt
event loop.  See OPTIMIZATION.md §B.5 for the design.
"""
from __future__ import annotations
import math
import os

import numpy as np

from app.core._dem_cache import to_nan_float64_cached
from app.core._jit import JIT_ENABLED, njit, prange


REFRACTION_COEFF = 0.13   # standard atmospheric refraction coefficient
_R_EARTH_M = 6_371_000.0
_CURV_FACTOR_DEFAULT = (1.0 - REFRACTION_COEFF) / (2.0 * _R_EARTH_M)


_THREADS_CAPPED = False


def _cap_jit_threads() -> None:
    """Limit Numba parallelism so the viewshed kernel doesn't starve the GUI.

    Called lazily before the first JIT viewshed run.  Only fires when
    Numba is available and ``NUMBA_NUM_THREADS`` hasn't been set
    explicitly by the user.
    """
    global _THREADS_CAPPED
    if _THREADS_CAPPED or not JIT_ENABLED:
        return
    _THREADS_CAPPED = True
    if os.environ.get("NUMBA_NUM_THREADS"):
        return  # respect user override
    try:
        import numba
        cap = min(os.cpu_count() or 1, 8)
        # ``set_num_threads`` will raise if asked for more than the
        # configured max — bound by ``get_num_threads`` to be safe.
        cap = min(cap, numba.get_num_threads())
        numba.set_num_threads(cap)
    except Exception:
        pass


@njit(cache=True, parallel=True)
def _viewshed_kernel(
    z: np.ndarray,
    visibility: np.ndarray,
    observer_row: int,
    observer_col: int,
    obs_elev: float,
    target_height: float,
    max_r: int,
    cell_size: float,
    curv_factor: float,        # 0.0 disables curvature correction
    r0: int, r1: int, c0: int, c1: int,
) -> None:
    """Parallel ray-cast over target rows ``[r0, r1)`` × cols ``[c0, c1)``.

    Per-target work is independent, so ``prange`` on the outer loop
    parallelizes cleanly.  Arithmetic is in float64 throughout to match
    the pure-Python reference bit-for-bit.
    """
    rows = z.shape[0]
    cols = z.shape[1]
    for tr in prange(r0, r1):
        for tc in range(c0, c1):
            if tr == observer_row and tc == observer_col:
                visibility[tr, tc] = 1
                continue
            zt = z[tr, tc]
            if zt != zt:  # NaN check — np.isfinite would force an object call
                continue

            dx = tc - observer_col
            dy = tr - observer_row
            dist_cells = math.sqrt(dx * dx + dy * dy)
            if dist_cells > max_r:
                continue

            adx = -dx if dx < 0 else dx
            ady = -dy if dy < 0 else dy
            n_steps = adx if adx > ady else ady
            if n_steps < 1:
                n_steps = 1
            visible = True
            max_slope = -1.0e308   # -inf in float64

            for step in range(1, n_steps + 1):
                frac = step / n_steps
                sr = observer_row + dy * frac
                sc = observer_col + dx * frac

                r0_s = int(sr)
                c0_s = int(sc)
                r1_s = r0_s + 1
                if r1_s > rows - 1:
                    r1_s = rows - 1
                c1_s = c0_s + 1
                if c1_s > cols - 1:
                    c1_s = cols - 1
                wr = sr - r0_s
                wc = sc - c0_s

                z_sample = (
                    z[r0_s, c0_s] * (1.0 - wr) * (1.0 - wc)
                    + z[r0_s, c1_s] * (1.0 - wr) * wc
                    + z[r1_s, c0_s] * wr * (1.0 - wc)
                    + z[r1_s, c1_s] * wr * wc
                )

                if z_sample != z_sample:
                    visible = False
                    break

                dist_m = frac * dist_cells * cell_size

                if curv_factor > 0.0 and dist_m > 0.0:
                    z_sample -= dist_m * dist_m * curv_factor

                if step < n_steps:
                    slope_to_sample = (z_sample - obs_elev) / dist_m
                    if slope_to_sample > max_slope:
                        max_slope = slope_to_sample
                else:
                    z_target = z_sample + target_height
                    slope_to_target = (z_target - obs_elev) / dist_m
                    visible = slope_to_target >= max_slope

            if visible:
                visibility[tr, tc] = 1


def _viewshed_py(
    z: np.ndarray,
    visibility: np.ndarray,
    observer_row: int,
    observer_col: int,
    obs_elev: float,
    target_height: float,
    max_r: int,
    cell_size: float,
    curv_factor: float,
    r0: int, r1: int, c0: int, c1: int,
    _progress,
) -> bool:
    """Pure-Python viewshed driver — returns True on user cancel."""
    rows, cols = z.shape
    span = max(r1 - r0, 1)
    # Throttle progress emissions to ≤100 over the whole sweep.
    report_every = max(1, span // 100)
    next_report = r0 + report_every if _progress is not None else r1 + 1

    for tr in range(r0, r1):
        if tr >= next_report:
            if _progress(int((tr - r0) / span * 100)):
                return True
            next_report = tr + report_every
        for tc in range(c0, c1):
            if tr == observer_row and tc == observer_col:
                visibility[tr, tc] = 1
                continue
            if not np.isfinite(z[tr, tc]):
                continue

            dx = tc - observer_col
            dy = tr - observer_row
            dist_cells = math.sqrt(dx * dx + dy * dy)
            if dist_cells > max_r:
                continue

            n_steps = max(abs(dx), abs(dy))
            if n_steps < 1:
                n_steps = 1
            visible = True
            max_slope = -np.inf

            for step in range(1, n_steps + 1):
                frac = step / n_steps
                sr = observer_row + dy * frac
                sc = observer_col + dx * frac

                r0_s, c0_s = int(sr), int(sc)
                r1_s = min(r0_s + 1, rows - 1)
                c1_s = min(c0_s + 1, cols - 1)
                wr, wc = sr - r0_s, sc - c0_s

                z_sample = (
                    z[r0_s, c0_s] * (1 - wr) * (1 - wc)
                    + z[r0_s, c1_s] * (1 - wr) * wc
                    + z[r1_s, c0_s] * wr * (1 - wc)
                    + z[r1_s, c1_s] * wr * wc
                )

                if not np.isfinite(z_sample):
                    visible = False
                    break

                dist_m = frac * dist_cells * cell_size

                if curv_factor > 0.0 and dist_m > 0.0:
                    z_sample -= dist_m * dist_m * curv_factor

                if step < n_steps:
                    slope_to_sample = (z_sample - obs_elev) / dist_m
                    if slope_to_sample > max_slope:
                        max_slope = slope_to_sample
                else:
                    z_target = z_sample + target_height
                    slope_to_target = (z_target - obs_elev) / dist_m
                    visible = slope_to_target >= max_slope

            if visible:
                visibility[tr, tc] = 1

    return False


def viewshed(
    dem: np.ndarray,
    cell_size: float,
    observer_row: int,
    observer_col: int,
    observer_height: float = 1.8,
    target_height: float = 0.0,
    max_radius: float | None = None,
    correct_curvature: bool = True,
    nodata: float | None = None,
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

    # Sentinel→NaN once at the top so the bilinear interpolator below
    # cannot pull a raw -9999 value into the line-of-sight sample
    # (which would look like a deep downhill and let the ray "see"
    # past every void).
    z = to_nan_float64_cached(dem, nodata)

    obs_elev_raw = z[observer_row, observer_col]
    if not np.isfinite(obs_elev_raw):
        return visibility
    obs_elev = float(obs_elev_raw) + observer_height

    max_r = int(max_radius) if max_radius is not None else max(rows, cols)
    r0 = max(0, observer_row - max_r)
    r1 = min(rows, observer_row + max_r + 1)
    c0 = max(0, observer_col - max_r)
    c1 = min(cols, observer_col + max_r + 1)

    curv_factor = _CURV_FACTOR_DEFAULT if correct_curvature else 0.0

    if JIT_ENABLED:
        _cap_jit_threads()
        # Single 0→100 progress tick: the JIT path finishes a max_radius
        # = 1000 viewshed in a few seconds, so a granular progress bar
        # buys no UX and would force per-row synchronisation against
        # the prange kernel.
        if _progress is not None and _progress(0):
            return visibility
        _viewshed_kernel(
            z, visibility, int(observer_row), int(observer_col),
            obs_elev, float(target_height), max_r, float(cell_size),
            float(curv_factor), r0, r1, c0, c1,
        )
        if _progress is not None:
            _progress(100)
        return visibility

    cancelled = _viewshed_py(
        z, visibility, int(observer_row), int(observer_col),
        obs_elev, float(target_height), max_r, float(cell_size),
        float(curv_factor), r0, r1, c0, c1, _progress,
    )
    if cancelled:
        # Match the legacy semantics: return whatever was filled in so far.
        return visibility
    return visibility


def cumulative_viewshed(
    dem: np.ndarray,
    cell_size: float,
    observer_points: list[tuple[int, int]],
    observer_height: float = 1.8,
    _progress=None,
    **kwargs,
) -> np.ndarray:
    """
    Cumulative (total) viewshed: count how many observers can see each cell.
    Useful for site siting, archaeology, telecommunications.
    """
    accum = np.zeros(dem.shape, dtype=np.int32)
    total = max(len(observer_points), 1)
    for i, (r, c) in enumerate(observer_points):
        # Outer progress: per-observer.  The inner viewshed's own progress
        # would overwrite this, so don't pass it down.
        if _progress is not None and _progress(int(i / total * 100)):
            return accum
        accum += viewshed(dem, cell_size, r, c, observer_height, **kwargs).astype(np.int32)
    return accum
