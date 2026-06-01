"""
Earthquake depth cross-section dialog.

Given a polyline drawn on the 2D map plus the current earthquake catalog, plot
distance-along-section vs depth-below-surface for every event within a chosen
buffer width.  Magnitude → dot radius; depth → dot fill (same ramp as the 2D
map and the 3D viewport, so the views are mutually consistent).

A faint terrain elevation line is drawn above zero when a DEM is active, so
the user can see hypocenters' relationship to surface relief along the
section.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QDialogButtonBox, QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigCanvas
from matplotlib.figure import Figure

from app.core.dem_layer import DemLayer
from app.core.earthquakes import EarthquakeCatalog


# Equirectangular metres-per-degree approximation; accurate well past the
# scale of any single cross-section the user is likely to draw.
_METRES_PER_DEG_LAT = 111_320.0


def _latlon_to_local(lat: float, lon: float,
                     center_lat: float, center_lon: float) -> tuple[float, float]:
    mlat = _METRES_PER_DEG_LAT
    mlon = _METRES_PER_DEG_LAT * math.cos(math.radians(center_lat))
    return (lon - center_lon) * mlon, (lat - center_lat) * mlat


class EarthquakeSectionDialog(QDialog):
    """Distance-vs-depth scatter for events within a buffer of a polyline."""

    def __init__(
        self,
        catalog: EarthquakeCatalog,
        line_latlons: list[tuple[float, float]],
        dem: Optional[DemLayer] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Earthquake Cross-Section")
        self.resize(820, 460)
        self._cat = catalog
        self._line = list(line_latlons)
        self._dem = dem

        # Local frame anchored at the section midpoint so the equirectangular
        # approximation stays accurate near the line itself.
        center_lat = float(np.mean([p[0] for p in self._line]))
        center_lon = float(np.mean([p[1] for p in self._line]))
        self._center = (center_lat, center_lon)

        # Project the polyline into local metres up-front; the projection of
        # quake positions reuses the same basis.
        self._line_xy = np.array([
            _latlon_to_local(lat, lon, center_lat, center_lon)
            for lat, lon in self._line
        ], dtype=np.float64)
        self._segments = self._build_segments(self._line_xy)
        self._total_length = float(self._segments[-1, 1]) if self._segments.size else 0.0

        self._build_ui()
        self._replot()

    # ── Geometry ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_segments(line_xy: np.ndarray) -> np.ndarray:
        """For each segment, return (cum_start, cum_end, dx, dy, length)."""
        if line_xy.shape[0] < 2:
            return np.zeros((0, 5), dtype=np.float64)
        diffs = np.diff(line_xy, axis=0)
        lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        cum = np.concatenate([[0.0], np.cumsum(lengths)])
        out = np.zeros((len(lengths), 5), dtype=np.float64)
        out[:, 0] = cum[:-1]
        out[:, 1] = cum[1:]
        out[:, 2] = diffs[:, 0]
        out[:, 3] = diffs[:, 1]
        out[:, 4] = lengths
        return out

    def _project_point(self, x: float, y: float) -> tuple[float, float]:
        """Project (x, y) onto the closest segment; return (along, perp).

        ``along`` is the cumulative distance in metres along the polyline.
        ``perp`` is the perpendicular distance to the closest segment.
        """
        best_perp = float("inf")
        best_along = 0.0
        for i, (cum0, _cum1, dx, dy, length) in enumerate(self._segments):
            if length <= 0:
                continue
            sx, sy = self._line_xy[i]
            wx, wy = x - sx, y - sy
            u = max(0.0, min(1.0, (wx * dx + wy * dy) / (length * length)))
            px = sx + u * dx
            py = sy + u * dy
            off = math.hypot(x - px, y - py)
            if off < best_perp:
                best_perp = off
                best_along = cum0 + u * length
        return best_along, best_perp

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Header: buffer-width control + summary labels.
        head = QHBoxLayout()
        head.addWidget(QLabel("Buffer width:"))
        self._buffer_spin = QDoubleSpinBox()
        self._buffer_spin.setSuffix(" km")
        self._buffer_spin.setDecimals(2)
        self._buffer_spin.setRange(0.5, 200.0)
        # Default buffer ≈ 1/10th of the section length, but at least 5 km.
        default_km = max(5.0, self._total_length / 1000.0 / 10.0)
        self._buffer_spin.setValue(default_km)
        # Re-filter + re-plot only when the value is committed (Enter / Tab /
        # focus-out / step buttons), not on every keystroke during typing.
        self._buffer_spin.setKeyboardTracking(False)
        self._buffer_spin.valueChanged.connect(self._replot)
        head.addWidget(self._buffer_spin)
        head.addSpacing(20)
        self._lbl_summary = QLabel("—")
        head.addWidget(self._lbl_summary, 1)
        head.addStretch(0)
        layout.addLayout(head)

        # Chart
        self._fig = Figure(figsize=(7, 3.2), facecolor="#1e1e1e")
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigCanvas(self._fig)
        layout.addWidget(self._canvas, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.accept)
        layout.addWidget(buttons)

    # ── Plotting ──────────────────────────────────────────────────────────────

    def _replot(self):
        buffer_m = float(self._buffer_spin.value()) * 1000.0
        ax = self._ax
        ax.cla()
        ax.set_facecolor("#252525")

        if self._segments.size == 0:
            ax.text(0.5, 0.5, "Section too short", color="#888",
                    ha="center", va="center", transform=ax.transAxes)
            self._canvas.draw()
            return

        # Pick the events that pass the current filter, then project + clip.
        visible = self._cat.visible_indices()
        if visible.size == 0:
            ax.text(0.5, 0.5, "No events pass current filters", color="#888",
                    ha="center", va="center", transform=ax.transAxes)
            self._canvas.draw()
            self._lbl_summary.setText("0 events")
            return

        center_lat, center_lon = self._center
        distances, depths, mags, offsets = [], [], [], []
        for i in visible:
            ev = self._cat.events[i]
            x, y = _latlon_to_local(ev.lat, ev.lon, center_lat, center_lon)
            along, perp = self._project_point(x, y)
            if perp > buffer_m:
                continue
            distances.append(along / 1000.0)        # km
            depths.append(ev.depth_km)
            mags.append(ev.magnitude)
            offsets.append(perp / 1000.0)
        distances = np.asarray(distances)
        depths    = np.asarray(depths)
        mags      = np.asarray(mags)

        # Color = depth, same ramp the catalog uses elsewhere.
        s = self._cat.style
        lo = s.color_min if s.color_by == "depth" else float(np.min(self._cat.depths_km))
        hi = s.color_max if s.color_by == "depth" else float(np.max(self._cat.depths_km))
        if hi <= lo: hi = lo + 1.0
        try:
            import matplotlib
            cmap = matplotlib.colormaps[s.colormap]
        except (KeyError, AttributeError):
            from matplotlib import cm
            cmap = cm.get_cmap(s.colormap)
        # Marker area in pts² — area scales with magnitude so the visual
        # weight is proportional to the energy proxy people are used to.
        sizes = 8.0 + 16.0 * np.maximum(mags, 0.0) ** 1.4
        if distances.size:
            ax.scatter(distances, -depths,
                       c=depths, cmap=cmap, vmin=lo, vmax=hi,
                       s=sizes, edgecolors="#111", linewidths=0.7,
                       alpha=0.92, zorder=3)

        # Optional surface elevation line (when an active DEM is provided).
        self._draw_surface_line(ax)

        ax.axhline(0, color="#888", linewidth=0.8, alpha=0.6)
        ax.set_xlim(0, max(self._total_length / 1000.0, 1.0))
        # Depth scale: 0 at top → max depth at bottom, plus a hair of headroom.
        max_dep = float(np.max(self._cat.depths_km)) if len(self._cat) else 1.0
        ax.set_ylim(-max_dep * 1.05, max(2.0, max_dep * 0.10))
        ax.set_xlabel("Distance along section (km)", color="#aaa", fontsize=9)
        ax.set_ylabel("Elevation / depth (km)", color="#aaa", fontsize=9)
        ax.tick_params(colors="#aaa", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        self._fig.tight_layout()
        self._canvas.draw()

        self._lbl_summary.setText(
            f"{distances.size} event(s) within ±{buffer_m / 1000.0:.1f} km  "
            f"·  section length {self._total_length / 1000.0:.1f} km"
        )

    def _draw_surface_line(self, ax) -> None:
        """Plot the terrain elevation along the polyline above the axis.

        Skipped silently when there is no DEM or the line falls outside it —
        the cross-section is still useful without surface relief.
        """
        dem = self._dem
        if dem is None or dem.data is None or dem.bounds is None or len(self._line) < 2:
            return
        from app.core.visibility.profile import sample_profile

        b = dem.bounds
        rows, cols = dem.shape
        row_cols: list[tuple[int, int]] = []
        for lat, lon in self._line:
            if not (b.south <= lat <= b.north and b.west <= lon <= b.east):
                continue
            col = int(np.clip((lon - b.west) / (b.east - b.west) * cols, 0, cols - 1))
            row = int(np.clip((b.north - lat) / (b.north - b.south) * rows, 0, rows - 1))
            row_cols.append((row, col))
        if len(row_cols) < 2:
            return
        try:
            prof = sample_profile(dem.valid_data, dem.cell_size_m, row_cols)
        except Exception:
            return
        # Elevation plotted in km (above sea level → positive on the same axis
        # earthquakes use; depth-below-surface is negative).
        elev_km = prof.elevations / 1000.0
        dist_km = prof.distances / 1000.0
        ax.plot(dist_km, elev_km, color="#bbb", linewidth=1.0, alpha=0.7, zorder=2)
        ax.fill_between(dist_km, elev_km, 0, color="#bbb", alpha=0.10, zorder=1)
