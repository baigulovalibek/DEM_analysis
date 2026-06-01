"""
Elevation profile sampling along a polyline.

Reference: Burrough & McDonnell (1998). Principles of GIS. Oxford Univ. Press.

Bilinear interpolation (default) matches QGIS/GDAL behaviour.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ElevationProfile:
    distances: np.ndarray           # cumulative distance along path (m)
    elevations: np.ndarray          # interpolated elevation at each sample
    lon_lat_samples: list = field(default_factory=list)   # (lon, lat) pairs

    @property
    def total_length(self) -> float:
        return float(self.distances[-1]) if len(self.distances) else 0.0

    @property
    def min_elev(self) -> float:
        return float(np.nanmin(self.elevations))

    @property
    def max_elev(self) -> float:
        return float(np.nanmax(self.elevations))

    @property
    def total_ascent(self) -> float:
        diff = np.diff(self.elevations)
        return float(np.nansum(diff[diff > 0]))

    @property
    def total_descent(self) -> float:
        diff = np.diff(self.elevations)
        return float(np.abs(np.nansum(diff[diff < 0])))

    def along_slope(self) -> np.ndarray:
        """Slope in degrees between consecutive sample points."""
        dz = np.diff(self.elevations)
        dx = np.diff(self.distances)
        dx = np.where(dx == 0, 1e-10, dx)
        return np.degrees(np.arctan(dz / dx))


def sample_profile(
    dem: np.ndarray,
    cell_size: float,
    row_col_points: List[Tuple[float, float]],
    sample_spacing: float = None,
) -> ElevationProfile:
    """
    Sample elevation along a polyline defined by (row, col) coordinates.

    Parameters
    ----------
    row_col_points : list of (row, col) floats defining the polyline vertices
    sample_spacing : ground distance between samples; defaults to cell_size/2

    Returns
    -------
    ElevationProfile
    """
    rows, cols = dem.shape
    if sample_spacing is None:
        sample_spacing = cell_size * 0.5

    if len(row_col_points) < 2:
        return ElevationProfile(np.array([0.0]), np.array([np.nan]))

    # Densify polyline to uniform spacing.  Each segment contributes its
    # END point; the FIRST segment additionally seeds the polyline start
    # so the result is monotonically increasing in cumulative distance
    # with no duplicate samples at internal vertices (the previous loop
    # produced one duplicate per internal vertex, plus a duplicated
    # final point).
    sample_rows, sample_cols, cumul_dist = [], [], []
    cum = 0.0

    # Seed with the polyline start so the first segment can append its
    # interior samples + endpoint without duplicating the vertex.
    r_start, c_start = row_col_points[0]
    sample_rows.append(r_start)
    sample_cols.append(c_start)
    cumul_dist.append(0.0)

    for i in range(len(row_col_points) - 1):
        r0, c0 = row_col_points[i]
        r1, c1 = row_col_points[i + 1]
        seg_len = np.sqrt((r1 - r0) ** 2 + (c1 - c0) ** 2) * cell_size
        n_steps = max(1, int(np.ceil(seg_len / sample_spacing)))

        # Walk frac = 1/n .. n/n, i.e. interior samples plus the segment
        # endpoint.  The segment START was already appended by the
        # previous iteration's endpoint (or by the seed above).
        for step in range(1, n_steps + 1):
            frac = step / n_steps
            sample_rows.append(r0 + (r1 - r0) * frac)
            sample_cols.append(c0 + (c1 - c0) * frac)
            cumul_dist.append(cum + seg_len * frac)
        cum += seg_len

    sample_rows = np.array(sample_rows)
    sample_cols = np.array(sample_cols)
    cumul_dist = np.array(cumul_dist)

    # Bilinear interpolation
    r0 = np.floor(sample_rows).astype(int).clip(0, rows - 2)
    c0 = np.floor(sample_cols).astype(int).clip(0, cols - 2)
    r1 = r0 + 1
    c1 = c0 + 1
    wr = sample_rows - r0
    wc = sample_cols - c0

    elevations = (
        dem[r0, c0] * (1 - wr) * (1 - wc)
        + dem[r0, c1] * (1 - wr) * wc
        + dem[r1, c0] * wr * (1 - wc)
        + dem[r1, c1] * wr * wc
    )

    return ElevationProfile(
        distances=cumul_dist.astype(np.float32),
        elevations=elevations.astype(np.float32),
    )
