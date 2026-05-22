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


def _horizon_angles(
    dem: np.ndarray,
    cell_size: float,
    direction_deg: float,
    max_radius: int,
) -> np.ndarray:
    """
    For each cell, find the maximum horizon elevation angle (radians) looking
    in direction `direction_deg` (geographic: 0=N clockwise) up to `max_radius`.

    Returns a float32 array of horizon angles in [0, π/2].
    """
    angle_rad = np.radians(direction_deg)
    dc = np.sin(angle_rad)    # eastward component
    dr = -np.cos(angle_rad)   # row-upward component (row 0 = north)

    rows, cols = dem.shape
    horizon = np.full((rows, cols), -np.pi / 2, dtype=np.float64)

    for step in range(1, max_radius + 1):
        # Fractional shift in row and column
        frac_r = dr * step
        frac_c = dc * step

        # Bilinear interpolated elevation at (r + frac_r, c + frac_c)
        ri = np.arange(rows)
        ci = np.arange(cols)
        r_sample = np.clip(ri[:, None] + frac_r, 0, rows - 1)
        c_sample = np.clip(ci[None, :] + frac_c, 0, cols - 1)

        # Bilinear weights
        r0 = np.floor(r_sample).astype(int)
        r1 = np.minimum(r0 + 1, rows - 1)
        c0 = np.floor(c_sample).astype(int)
        c1 = np.minimum(c0 + 1, cols - 1)

        wr = r_sample - r0
        wc = c_sample - c0

        elev = (
            dem[r0, c0] * (1 - wr) * (1 - wc)
            + dem[r0, c1] * (1 - wr) * wc
            + dem[r1, c0] * wr * (1 - wc)
            + dem[r1, c1] * wr * wc
        )

        dist = step * cell_size * np.sqrt(dr ** 2 + dc ** 2)
        elev_angle = np.arctan2(elev - dem, dist)
        horizon = np.maximum(horizon, elev_angle)

    return horizon.astype(np.float32)


def sky_view_factor(
    dem: np.ndarray,
    cell_size: float,
    n_directions: int = 16,
    max_radius: int = 50,
    _progress=None,
) -> np.ndarray:
    """
    Sky-View Factor.  SVF ∈ [0, 1]: 1 = fully open sky, 0 = completely enclosed.

    Parameters
    ----------
    n_directions : number of search azimuths (default 16; 8 is faster)
    max_radius   : maximum search radius in cells
    """
    z = dem.astype(np.float64)
    sin_sum = np.zeros(z.shape, dtype=np.float64)

    directions = np.linspace(0, 360, n_directions, endpoint=False)
    for i, azimuth in enumerate(directions):
        if _progress and _progress(int(i / n_directions * 100)):
            break
        h = _horizon_angles(z, cell_size, azimuth, max_radius)
        sin_sum += np.sin(np.maximum(h, 0.0))

    svf = 1.0 - sin_sum / n_directions
    return np.clip(svf, 0.0, 1.0).astype(np.float32)


def positive_openness(
    dem: np.ndarray,
    cell_size: float,
    n_directions: int = 8,
    max_radius: int = 50,
    _progress=None,
) -> np.ndarray:
    """
    Positive openness = mean horizon angle above horizontal (Yokoyama et al. 2002).
    Large → open ridgeline; small → enclosed valley.  Units: degrees.
    """
    z = dem.astype(np.float64)
    angle_sum = np.zeros(z.shape, dtype=np.float64)

    directions = np.linspace(0, 360, n_directions, endpoint=False)
    for i, azimuth in enumerate(directions):
        if _progress is not None and _progress(int(i / n_directions * 100)):
            break
        h = _horizon_angles(z, cell_size, azimuth, max_radius)
        angle_sum += np.maximum(h, 0.0)

    openness = np.degrees(angle_sum / n_directions)
    return openness.astype(np.float32)


def negative_openness(
    dem: np.ndarray,
    cell_size: float,
    n_directions: int = 8,
    max_radius: int = 50,
    _progress=None,
) -> np.ndarray:
    """
    Negative openness = same as positive but looking downward (below horizontal).
    Large → deep enclosed depression; small → flat/convex surface.
    """
    # Flip the DEM sign and compute positive openness of the inverted surface
    return positive_openness(-dem, cell_size, n_directions, max_radius, _progress)
