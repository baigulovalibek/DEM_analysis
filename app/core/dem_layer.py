"""
DEM data model.  Holds raw elevation array + geospatial metadata.
"""
from __future__ import annotations
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_uid_counter = itertools.count(1)


@dataclass
class GeoBounds:
    """Geographic bounding box in WGS84 (EPSG:4326)."""
    west: float
    south: float
    east: float
    north: float

    @property
    def center_lon(self) -> float:
        return (self.west + self.east) / 2

    @property
    def center_lat(self) -> float:
        return (self.south + self.north) / 2

    def leaflet(self) -> list:
        """[[south, west], [north, east]] as Leaflet expects."""
        return [[self.south, self.west], [self.north, self.east]]


@dataclass
class DemLayer:
    """A single raster layer (DEM or analysis result)."""

    # Display
    name: str
    product: str            # e.g. "dem", "hillshade", "slope"
    visible: bool = True
    opacity: float = 0.75

    # Data
    data: np.ndarray = field(repr=False, default=None)
    nodata: Optional[float] = None

    # Geospatial
    bounds: Optional[GeoBounds] = None   # WGS84 extent
    crs_wkt: str = ""
    cell_size_m: float = 30.0            # approximate cell size in metres

    # Source
    source_path: Optional[Path] = None
    parent_name: Optional[str] = None   # DEM layer this was derived from

    # UI state — set by renderer
    render_min: Optional[float] = None
    render_max: Optional[float] = None
    colormap: str = "terrain"

    # Stable identity — survives renames; used to key intermediate caches.
    uid: int = field(default_factory=lambda: next(_uid_counter))

    # ── computed properties ────────────────────────────────────────────────

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape if self.data is not None else (0, 0)

    @property
    def valid_data(self) -> np.ndarray:
        """Data with nodata masked to NaN."""
        if self.data is None:
            return np.array([])
        d = self.data.astype(np.float64)
        if self.nodata is not None:
            d[d == self.nodata] = np.nan
        return d

    @property
    def stats(self) -> dict:
        d = self.valid_data.ravel()
        d = d[~np.isnan(d)]
        if d.size == 0:
            return {}
        return {
            "min": float(d.min()),
            "max": float(d.max()),
            "mean": float(d.mean()),
            "std": float(d.std()),
            "count": int(d.size),
        }

    def auto_range(self) -> Tuple[float, float]:
        """2nd–98th percentile for stretch, avoiding outlier spikes."""
        d = self.valid_data.ravel()
        d = d[~np.isnan(d)]
        if d.size == 0:
            return 0.0, 1.0
        lo, hi = np.percentile(d, [2, 98])
        if lo == hi:
            lo, hi = d.min(), d.max()
        return float(lo), float(hi)
