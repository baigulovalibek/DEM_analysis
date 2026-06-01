"""
Sky-View Factor (SVF) and related horizon-based visualisation.

SVF ≈ 1 - (1/n) · Σ sin(γ_i)

where γ_i is the maximum horizon angle in search direction i.

References:
  Zakšek, K., Oštir, K., Kokalj, Ž. (2011).
      Sky-View Factor as a relief visualization technique.
      Remote Sensing 3(2), 398-415.
  Yokoyama, R., Shirasawa, M., Pike, R.J. (2002).
      Visualizing topography by openness. PE&RS 68(3), 257-265.
  Kokalj & Somrak (2019). Remote Sensing 11(7), 747.

Positive openness = mean of horizon angles above the horizontal = related to SVF.
Negative openness = same but looking downward (concavity indicator).
"""
from __future__ import annotations
import math

import numpy as np

from app.core._dem_cache import to_nan_float64_cached
from app.core._jit import JIT_ENABLED, njit, prange


_NEG_HALF_PI = -math.pi / 2


@njit(cache=True, parallel=True)
def _horizon_sweep_jit(
    z: np.ndarray,
    cell_size: float,
    dr: float,
    dc: float,
    max_radius: int,
) -> np.ndarray:
    """JIT'd single-direction horizon sweep.

    Equivalent to the NumPy path below but the per-cell ray walk stays
    in registers across the whole ``max_radius`` loop — no per-step
    full-grid allocations, no fancy-indexing memory traffic.  Parallel
    over output rows via ``prange``.

    Returns the max horizon angle (radians) per cell as float64; the
    caller downcasts to float32 once.  NaN-centre cells stay at
    ``-π/2`` (treated as "below horizon"); the public wrappers re-mask
    them to NaN at the end.

    Three tight-loop optimisations matter on big DEMs:

    1. **Track ``tan(angle)`` instead of the angle itself.**  ``math.atan2``
       costs ~30 ns; running it ``max_radius`` times per cell per direction
       (1600× on defaults) dominates the wall-clock.  We track
       ``max_tan = max(dz / dist)`` in the inner loop and call ``atan``
       *once* per cell at the end.

    2. **Compare via ``dz > max_tan * dist``**, avoiding the divide on
       samples that can't beat the running max.  Multiplies are ~4× cheaper
       than divides, and most ray steps don't update the max.

    3. **Track only above-horizon hits** (max_tan ≥ 0).  Below-horizon
       cells contribute 0 to both SVF (via ``max(γ, 0)``) and openness;
       there's no reason to carry negative angles through the loop.

    Bilinear sampling is kept (rather than dropping to nearest-cell)
    because the latter introduces a visible 0.05–0.10 mean SVF shift on
    noisy terrain — a quality regression, not just numerical noise.
    """
    rows = z.shape[0]
    cols = z.shape[1]
    horizon = np.zeros((rows, cols), dtype=np.float64)

    ray_len = math.sqrt(dr * dr + dc * dc)
    cell_ray = cell_size * ray_len   # dist = step * cell_ray
    rows_max = rows - 1
    cols_max = cols - 1

    for tr in prange(rows):
        for tc in range(cols):
            z0 = z[tr, tc]
            if z0 != z0:  # NaN — leave horizon at 0
                continue
            max_tan = 0.0
            dist = 0.0
            for step in range(1, max_radius + 1):
                # Strength-reduced step * cell_ray.  Must update every
                # iteration regardless of subsequent `continue`s so the
                # accumulator stays in sync with the ray position.
                dist += cell_ray
                fr = tr + dr * step
                fc = tc + dc * step
                # Clip to grid (matches the np.clip in the NumPy path —
                # equivalent to assuming the boundary elevation extends
                # to infinity).
                if fr < 0.0:
                    fr = 0.0
                elif fr > rows_max:
                    fr = rows_max
                if fc < 0.0:
                    fc = 0.0
                elif fc > cols_max:
                    fc = cols_max
                # int() truncates toward zero; fr/fc are clipped to
                # [0, *_max] so this matches np.floor.
                r0i = int(fr)
                c0i = int(fc)
                r1i = r0i + 1 if r0i < rows_max else r0i
                c1i = c0i + 1 if c0i < cols_max else c0i
                wr = fr - r0i
                wc = fc - c0i
                elev = (z[r0i, c0i] * (1.0 - wr) * (1.0 - wc)
                        + z[r0i, c1i] * (1.0 - wr) * wc
                        + z[r1i, c0i] * wr * (1.0 - wc)
                        + z[r1i, c1i] * wr * wc)
                if elev != elev:  # NaN sample — leaves running max unchanged
                    continue
                dz = elev - z0
                if dz <= 0.0:
                    continue   # below horizon, cannot raise max_tan
                # Cheap compare-via-multiply; only divide when we actually
                # beat the running max.
                if dz > max_tan * dist:
                    max_tan = dz / dist
            horizon[tr, tc] = math.atan(max_tan)
    return horizon


def _horizon_angles(
    dem: np.ndarray,
    cell_size: float,
    direction_deg: float,
    max_radius: int,
    _progress=None,
    _progress_start: float = 0.0,
    _progress_end: float = 0.0,
) -> tuple[np.ndarray, bool]:
    """
    For each cell, find the maximum horizon elevation angle (radians) looking
    in direction `direction_deg` (geographic: 0=N clockwise) up to `max_radius`.

    ``dem`` is expected to already have missing samples encoded as NaN —
    bilinear interpolation across NaN corners yields NaN, which is treated
    as "below horizon" and leaves the running max unchanged.

    Edges are sampled by clipping the ray to the grid boundary; this is
    equivalent to assuming the terrain continues to infinity at the boundary
    elevation, so SVF/openness slightly underestimate sky openness for cells
    near the data edge.

    Returns ``(horizon, cancelled)`` where ``horizon`` is a float32 array of
    angles in [0, π/2] and ``cancelled`` is True iff the caller signalled
    cancellation through ``_progress``.

    When Numba is available the heavy lifting runs in ``_horizon_sweep_jit``
    — typically 20-50× faster than the NumPy path on large DEMs because
    the per-cell ray walk stays in registers instead of streaming
    ``max_radius`` full-grid temporaries through memory.  The JIT path can
    only cancel between directions (each direction completes in well under
    a second for typical inputs); the NumPy fallback still checks per step
    via the ``[_progress_start, _progress_end]`` sub-range.
    """
    # ── JIT fast path ─────────────────────────────────────────────────
    if JIT_ENABLED:
        if _progress is not None and _progress_end > _progress_start:
            if _progress(int(_progress_start)):
                return np.zeros(dem.shape, dtype=np.float32), True
        angle_rad = math.radians(direction_deg)
        dc = math.sin(angle_rad)
        dr = -math.cos(angle_rad)
        z64 = dem if dem.dtype == np.float64 else dem.astype(np.float64)
        h = _horizon_sweep_jit(z64, float(cell_size), dr, dc, int(max_radius))
        return h.astype(np.float32), False

    # ── NumPy fallback ────────────────────────────────────────────────
    angle_rad = np.radians(direction_deg)
    dc = float(np.sin(angle_rad))   # eastward component
    dr = -float(np.cos(angle_rad))  # row-upward component (row 0 = north)

    # Pay the float64→float32 cast once per direction; the inner
    # ``max_radius``-step loop then runs entirely in float32.  ``z`` is
    # the cached read-only NaN-converted DEM, so we cannot mutate it.
    if dem.dtype != np.float32:
        z = dem.astype(np.float32)
    else:
        z = dem

    rows, cols = z.shape
    horizon = np.full((rows, cols), -np.pi / 2, dtype=np.float32)

    # Hoist row/col index arrays out of the step loop — they're constant
    # across the radius sweep.  For max_radius=50, n_directions=16 this
    # saves ~800 full-grid allocations per call.
    ri = np.arange(rows, dtype=np.float32)[:, None]
    ci = np.arange(cols, dtype=np.float32)[None, :]
    ray_len = np.float32(np.sqrt(dr ** 2 + dc ** 2))
    neg_half_pi = np.float32(-np.pi / 2)

    span = _progress_end - _progress_start

    with np.errstate(invalid="ignore"):
        for step in range(1, max_radius + 1):
            # Per-step progress + cancel.  Reporting from inside the
            # radius loop (rather than only between directions) keeps
            # the bar moving on big DEMs and lets the user cancel
            # within ~one full-grid pass.  The worker throttles
            # emissions to 20 Hz so a chatty caller is fine.
            if _progress is not None and span > 0.0:
                pct = _progress_start + span * (step - 1) / max_radius
                if _progress(int(pct)):
                    return horizon, True

            # Fractional shift in row and column
            frac_r = np.float32(dr * step)
            frac_c = np.float32(dc * step)

            r_sample = np.clip(ri + frac_r, 0, rows - 1)
            c_sample = np.clip(ci + frac_c, 0, cols - 1)

            # Bilinear weights
            r0 = np.floor(r_sample).astype(np.int32)
            r1 = np.minimum(r0 + 1, rows - 1)
            c0 = np.floor(c_sample).astype(np.int32)
            c1 = np.minimum(c0 + 1, cols - 1)

            wr = r_sample - r0.astype(np.float32)
            wc = c_sample - c0.astype(np.float32)

            elev = (
                z[r0, c0] * (1 - wr) * (1 - wc)
                + z[r0, c1] * (1 - wr) * wc
                + z[r1, c0] * wr * (1 - wc)
                + z[r1, c1] * wr * wc
            )

            dist = np.float32(step * cell_size) * ray_len
            elev_angle = np.arctan2(elev - z, dist).astype(np.float32, copy=False)
            # NaN samples (centre or interpolated) come through as NaN
            # — substitute -π/2 so they leave the running max unchanged.
            elev_angle = np.where(np.isfinite(elev_angle), elev_angle, neg_half_pi)
            horizon = np.maximum(horizon, elev_angle)

    return horizon, False


def sky_view_factor(
    dem: np.ndarray,
    cell_size: float,
    n_directions: int = 16,
    max_radius: int = 50,
    nodata: float | None = None,
    _progress=None,
) -> np.ndarray:
    """
    Sky-View Factor.  SVF ∈ [0, 1]: 1 = fully open sky, 0 = completely enclosed.

    Parameters
    ----------
    n_directions : number of search azimuths (default 16; 8 is faster)
    max_radius   : maximum search radius in cells
    nodata       : sentinel value to treat as missing (NaN output for those cells)
    """
    z = to_nan_float64_cached(dem, nodata)
    centre_missing = ~np.isfinite(z)
    sin_sum = np.zeros(z.shape, dtype=np.float64)

    directions = np.linspace(0, 360, n_directions, endpoint=False)
    span = 100.0 / n_directions
    for i, azimuth in enumerate(directions):
        h, cancelled = _horizon_angles(
            z, cell_size, azimuth, max_radius,
            _progress=_progress,
            _progress_start=i * span,
            _progress_end=(i + 1) * span,
        )
        if cancelled:
            break
        sin_sum += np.sin(np.maximum(h, 0.0))

    svf = 1.0 - sin_sum / n_directions
    svf = np.clip(svf, 0.0, 1.0).astype(np.float32)
    # NaN centres would otherwise come out as SVF=1 (fully open sky); mark
    # them so consumers can distinguish "open sky" from "no data".
    svf[centre_missing] = np.nan
    return svf


def positive_openness(
    dem: np.ndarray,
    cell_size: float,
    n_directions: int = 8,
    max_radius: int = 50,
    nodata: float | None = None,
    _progress=None,
) -> np.ndarray:
    """
    Positive openness = mean horizon angle above horizontal (Yokoyama et al. 2002).
    Large → open ridgeline; small → enclosed valley.  Units: degrees.
    """
    z = to_nan_float64_cached(dem, nodata)
    centre_missing = ~np.isfinite(z)
    angle_sum = np.zeros(z.shape, dtype=np.float64)

    directions = np.linspace(0, 360, n_directions, endpoint=False)
    span = 100.0 / n_directions
    for i, azimuth in enumerate(directions):
        h, cancelled = _horizon_angles(
            z, cell_size, azimuth, max_radius,
            _progress=_progress,
            _progress_start=i * span,
            _progress_end=(i + 1) * span,
        )
        if cancelled:
            break
        angle_sum += np.maximum(h, 0.0)

    openness = np.degrees(angle_sum / n_directions).astype(np.float32)
    openness[centre_missing] = np.nan
    return openness


def negative_openness(
    dem: np.ndarray,
    cell_size: float,
    n_directions: int = 8,
    max_radius: int = 50,
    nodata: float | None = None,
    _progress=None,
) -> np.ndarray:
    """
    Negative openness = same as positive but looking downward (below horizontal).
    Large → deep enclosed depression; small → flat/convex surface.
    """
    # Flip the DEM sign and compute positive openness of the inverted surface.
    # Sentinel handling happens inside positive_openness; we still need to
    # convert here so the unary minus doesn't mangle the sentinel value
    # itself (e.g. -(-9999) = 9999 would look like a real elevation).
    z = to_nan_float64_cached(dem, nodata)
    return positive_openness(-z, cell_size, n_directions, max_radius,
                             nodata=None, _progress=_progress)
