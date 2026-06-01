"""
Earthquake catalog data model.

A catalog is a point dataset that lives parallel to the raster DemLayers in
the application: it's loaded from a spreadsheet, filtered by the user, and
rendered on both the 2D map and the 3D scene with depth as the headline
visual.  The model itself is Qt-free; the panels and renderers consume it.

Expected columns
----------------
The Excel/CSV reader is forgiving about column naming — common variants are
recognised case-insensitively:

* time     : ``origin``, ``time``, ``date``, ``datetime``
* latitude : ``lat``, ``lat n``, ``latitude``, ``y``
* longitude: ``lon``, ``long``, ``long e``, ``longitude``, ``x``
* depth km : ``dept``, ``depth``, ``depth_km``, ``z``
* magnitude: ``mag``, ``mag*``, ``magnitude``, ``ml``

All other columns are kept verbatim on each event under ``extras`` so popups
can show them without us hardcoding a schema.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


# Column-name lookups (lowercased / stripped).
_TIME_KEYS = {"origin", "time", "date", "datetime", "origin time", "origintime"}
_LAT_KEYS  = {"lat", "lat n", "latitude", "y", "latn"}
_LON_KEYS  = {"lon", "long", "long e", "longe", "longitude", "x"}
_DEP_KEYS  = {"dept", "depth", "depth_km", "depth km", "z"}
_MAG_KEYS  = {"mag", "mag*", "magnitude", "ml", "mw", "mb"}


def _find_col(df_cols: Iterable[str], targets: set[str]) -> Optional[str]:
    for c in df_cols:
        if str(c).strip().lower() in targets:
            return c
    return None


@dataclass
class EarthquakeEvent:
    """A single seismic event."""

    origin: Optional[datetime]
    lat: float
    lon: float
    depth_km: float          # positive = below surface
    magnitude: float
    extras: dict = field(default_factory=dict)   # remaining catalog columns

    def to_json(self, idx: int) -> dict:
        """JSON-friendly dict shipped to Leaflet."""
        return {
            "idx": idx,
            "lat": self.lat,
            "lon": self.lon,
            "depth": self.depth_km,
            "mag": self.magnitude,
            "time": self.origin.isoformat() if self.origin else None,
            "extras": {k: _json_safe(v) for k, v in self.extras.items()},
        }


def _json_safe(v):
    """Convert pandas / numpy / datetime scalars to JSON-serialisable form."""
    if v is None:
        return None
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if hasattr(v, "isoformat"):
        try:
            return v.isoformat()
        except Exception:
            return str(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (str, int, float, bool)):
        return v
    return str(v)


@dataclass
class EarthquakeStyle:
    """Symbology + filter state for the catalog.

    A single style instance drives both the 2D Leaflet markers and the 3D
    point sprites, so depth coloring stays consistent between views.
    """

    # Symbology — encoding is locked: depth → color, magnitude → size.  The
    # ``color_by`` field is kept for forward compatibility and dead paths in
    # the legacy rendering code but the UI no longer exposes it.
    color_by: str = "depth"
    colormap: str = "RdYlBu"         # USGS-style: shallow=red, deep=blue
    size_scale: float = 1.0          # 0.5..3.0 multiplier on magnitude->radius
    visible: bool = True

    # Color stretch (auto-filled from catalog if None)
    color_min: Optional[float] = None
    color_max: Optional[float] = None

    # Filters
    mag_min: Optional[float] = None
    mag_max: Optional[float] = None
    depth_min_km: Optional[float] = None
    depth_max_km: Optional[float] = None
    date_min: Optional[datetime] = None
    date_max: Optional[datetime] = None


@dataclass
class EarthquakeCatalog:
    """In-memory earthquake catalog with filter + style state."""

    events: list[EarthquakeEvent] = field(default_factory=list)
    source_path: Optional[Path] = None
    style: EarthquakeStyle = field(default_factory=EarthquakeStyle)

    # ── Loaders ───────────────────────────────────────────────────────────────

    @classmethod
    def from_file(cls, path: str | Path) -> "EarthquakeCatalog":
        """Load from .xlsx or .csv, detecting columns case-insensitively."""
        import pandas as pd

        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in (".xlsx", ".xls", ".xlsm"):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
        cat = cls.from_dataframe(df)
        cat.source_path = path
        return cat

    @classmethod
    def from_dataframe(cls, df) -> "EarthquakeCatalog":
        cols = list(df.columns)
        c_time = _find_col(cols, _TIME_KEYS)
        c_lat  = _find_col(cols, _LAT_KEYS)
        c_lon  = _find_col(cols, _LON_KEYS)
        c_dep  = _find_col(cols, _DEP_KEYS)
        c_mag  = _find_col(cols, _MAG_KEYS)

        missing = [
            name for name, col in [
                ("latitude", c_lat), ("longitude", c_lon),
                ("depth", c_dep), ("magnitude", c_mag),
            ] if col is None
        ]
        if missing:
            raise ValueError(
                "Earthquake catalog is missing required columns: "
                + ", ".join(missing)
                + f".  Found columns: {cols}"
            )

        events: list[EarthquakeEvent] = []
        used = {c_time, c_lat, c_lon, c_dep, c_mag} - {None}
        extra_cols = [c for c in cols if c not in used]

        # Pull each needed column as a numpy/object array — much faster than
        # itertuples and preserves the original column names (itertuples
        # mangles them to valid Python identifiers).
        lat_vals = df[c_lat].to_numpy()
        lon_vals = df[c_lon].to_numpy()
        dep_vals = df[c_dep].to_numpy()
        mag_vals = df[c_mag].to_numpy()
        time_vals = df[c_time].to_numpy() if c_time is not None else None
        extra_vals = {c: df[c].to_numpy() for c in extra_cols}

        n = len(df)
        for i in range(n):
            try:
                lat = float(lat_vals[i])
                lon = float(lon_vals[i])
                dep = float(dep_vals[i])
                mag = float(mag_vals[i])
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(lat) and math.isfinite(lon)
                    and math.isfinite(dep) and math.isfinite(mag)):
                continue

            origin = None
            if time_vals is not None:
                v = time_vals[i]
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    if isinstance(v, datetime):
                        origin = v
                    elif hasattr(v, "to_pydatetime"):
                        try:
                            origin = v.to_pydatetime()
                        except Exception:
                            origin = None
                    else:
                        try:
                            import pandas as pd
                            ts = pd.to_datetime(v, errors="coerce")
                            origin = ts.to_pydatetime() if ts is not None else None
                        except Exception:
                            origin = None

            extras = {c: extra_vals[c][i] for c in extra_cols}
            events.append(EarthquakeEvent(
                origin=origin, lat=lat, lon=lon,
                depth_km=dep, magnitude=mag, extras=extras,
            ))

        cat = cls(events=events)
        cat._auto_style()
        return cat

    # ── Derived stats ─────────────────────────────────────────────────────────

    def _auto_style(self) -> None:
        """Fill in style defaults from the catalog's actual ranges."""
        if not self.events:
            return
        s = self.style
        mags = self.magnitudes
        deps = self.depths_km
        dts  = [e.origin for e in self.events if e.origin is not None]

        if s.mag_min is None:    s.mag_min = float(np.min(mags))
        if s.mag_max is None:    s.mag_max = float(np.max(mags))
        if s.depth_min_km is None: s.depth_min_km = float(np.min(deps))
        if s.depth_max_km is None: s.depth_max_km = float(np.max(deps))
        if s.date_min is None and dts: s.date_min = min(dts)
        if s.date_max is None and dts: s.date_max = max(dts)
        if s.color_min is None or s.color_max is None:
            lo, hi = self._color_range()
            s.color_min = lo
            s.color_max = hi

    def _color_range(self) -> tuple[float, float]:
        """Min/max of the field the user is colouring by."""
        if not self.events:
            return 0.0, 1.0
        s = self.style
        if s.color_by == "magnitude":
            v = self.magnitudes
        elif s.color_by == "time":
            ts = [e.origin.timestamp() for e in self.events if e.origin is not None]
            v = np.asarray(ts) if ts else np.asarray([0.0, 1.0])
        else:
            v = self.depths_km
        lo, hi = float(np.min(v)), float(np.max(v))
        if lo == hi:
            hi = lo + 1.0
        return lo, hi

    def recompute_color_range(self) -> None:
        """Reset the color stretch to match the current ``color_by`` field."""
        lo, hi = self._color_range()
        self.style.color_min = lo
        self.style.color_max = hi

    @property
    def lats(self) -> np.ndarray:
        return np.asarray([e.lat for e in self.events], dtype=np.float64)

    @property
    def lons(self) -> np.ndarray:
        return np.asarray([e.lon for e in self.events], dtype=np.float64)

    @property
    def depths_km(self) -> np.ndarray:
        return np.asarray([e.depth_km for e in self.events], dtype=np.float64)

    @property
    def magnitudes(self) -> np.ndarray:
        return np.asarray([e.magnitude for e in self.events], dtype=np.float64)

    def bounds(self) -> Optional[tuple[float, float, float, float]]:
        """(south, west, north, east) in WGS84, or None if empty."""
        if not self.events:
            return None
        lats = self.lats
        lons = self.lons
        return (float(lats.min()), float(lons.min()),
                float(lats.max()), float(lons.max()))

    # ── Filtering ─────────────────────────────────────────────────────────────

    def visible_indices(self) -> np.ndarray:
        """Indices of events that pass the current style's filter."""
        if not self.events:
            return np.array([], dtype=np.int64)
        s = self.style
        mags = self.magnitudes
        deps = self.depths_km
        mask = np.ones(len(self.events), dtype=bool)
        if s.mag_min is not None:    mask &= mags >= s.mag_min
        if s.mag_max is not None:    mask &= mags <= s.mag_max
        if s.depth_min_km is not None: mask &= deps >= s.depth_min_km
        if s.depth_max_km is not None: mask &= deps <= s.depth_max_km
        if s.date_min is not None or s.date_max is not None:
            lo = s.date_min.timestamp() if s.date_min else -1e18
            hi = s.date_max.timestamp() if s.date_max else 1e18
            ts = np.array([
                e.origin.timestamp() if e.origin else math.nan
                for e in self.events
            ], dtype=np.float64)
            # Events without a timestamp pass-through (don't filter them out).
            time_mask = np.isnan(ts) | ((ts >= lo) & (ts <= hi))
            mask &= time_mask
        return np.where(mask)[0]

    # ── Color helpers ─────────────────────────────────────────────────────────

    def color_values(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Per-event scalar value driving the color ramp (depth/mag/time)."""
        s = self.style
        if s.color_by == "magnitude":
            v = self.magnitudes
        elif s.color_by == "time":
            v = np.asarray([
                e.origin.timestamp() if e.origin else math.nan
                for e in self.events
            ], dtype=np.float64)
        else:
            v = self.depths_km
        if indices is not None:
            v = v[indices]
        return v

    def colors_rgba(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Resolve color values through the current colormap → (N, 4) uint8."""
        import matplotlib
        from matplotlib import cm

        vals = self.color_values(indices)
        s = self.style
        lo = s.color_min if s.color_min is not None else float(np.nanmin(vals))
        hi = s.color_max if s.color_max is not None else float(np.nanmax(vals))
        if hi <= lo:
            hi = lo + 1.0
        t = np.clip((vals - lo) / (hi - lo), 0.0, 1.0)
        # Handle NaN (events without timestamps when coloring by time) → mid grey
        t = np.where(np.isfinite(t), t, 0.5)
        try:
            cmap = matplotlib.colormaps[s.colormap]
        except (KeyError, AttributeError):
            cmap = cm.get_cmap(s.colormap)
        rgba = (cmap(t) * 255.0).astype(np.uint8)
        return rgba

    def __len__(self) -> int:
        return len(self.events)
