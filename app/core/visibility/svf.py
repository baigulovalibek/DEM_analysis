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
import numpy as np

from app.core._dem_cache import to_nan_float64_cached


def _horizon_angles(
    dem: np.ndarray,
    cell_size: float,
    direction_deg: float,
    max_radius: int,
) -> np.ndarray:
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

    Returns a float32 array of horizon angles in [0, π/2].

    Arithmetic is done in float32 throughout — the bilinear sample and the
    ``np.maximum`` are memory-bandwidth-bound on big DEMs, so halving the
    element width (8 → 4 bytes) roughly halves the per-step cost.  The
    ~1 ULP loss never escapes the ``np.clip(svf, 0, 1)`` in
    :func:`sky_view_factor`.
    """
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

    with np.errstate(invalid="ignore"):
        for step in range(1, max_radius + 1):
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

    return horizon


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
    for i, azimuth in enumerate(directions):
        if _progress and _progress(int(i / n_directions * 100)):
            break
        h = _horizon_angles(z, cell_size, azimuth, max_radius)
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
    for i, azimuth in enumerate(directions):
        if _progress is not None and _progress(int(i / n_directions * 100)):
            break
        h = _horizon_angles(z, cell_size, azimuth, max_radius)
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
