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

    def _cache_key(self) -> tuple:
        """Fingerprint that changes whenever the underlying buffer does.

        Uses the buffer's memory address *and* its shape/dtype/nbytes, so
        even if CPython recycles the same id() for a new array we still
        detect the change (id-alone would silently return stale data).
        """
        arr = self.data
        if arr is None:
            return (None, self.nodata)
        return (
            arr.ctypes.data,
            arr.shape,
            arr.dtype.str,
            arr.nbytes,
            self.nodata,
        )

    @property
    def valid_data(self) -> np.ndarray:
        """Data with nodata masked to NaN.

        The masked copy is memoised against a fingerprint of the underlying
        buffer so repeated accesses (stats, histogram, profile sampling, 3D
        view, …) don't rebuild a fresh float copy each time.
        """
        if self.data is None:
            return np.array([])
        cached = getattr(self, "_valid_cache", None)
        cache_key = self._cache_key()
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        if (
            self.nodata is None
            and np.issubdtype(self.data.dtype, np.floating)
            and not np.any(~np.isfinite(self.data))
        ):
            # No masking needed and no dtype upcast required — share the buffer.
            out = self.data
        else:
            out = self.data.astype(np.float32, copy=True)
            if self.nodata is not None:
                out[out == self.nodata] = np.nan
        self._valid_cache = (cache_key, out)
        return out

    @property
    def stats(self) -> dict:
        cached = getattr(self, "_stats_cache", None)
        cache_key = self._cache_key()
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        d = self.valid_data.ravel()
        d = d[np.isfinite(d)]
        if d.size == 0:
            stats = {}
        else:
            stats = {
                "min": float(d.min()),
                "max": float(d.max()),
                "mean": float(d.mean()),
                "std": float(d.std()),
                "count": int(d.size),
            }
        self._stats_cache = (cache_key, stats)
        return stats

    def auto_range(self) -> Tuple[float, float]:
        """2nd–98th percentile for stretch, avoiding outlier spikes."""
        cached = getattr(self, "_range_cache", None)
        cache_key = self._cache_key()
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        d = self.valid_data.ravel()
        d = d[np.isfinite(d)]
        if d.size == 0:
            out = (0.0, 1.0)
        else:
            lo, hi = np.percentile(d, [2, 98])
            if lo == hi:
                lo, hi = d.min(), d.max()
            out = (float(lo), float(hi))
        self._range_cache = (cache_key, out)
        return out
