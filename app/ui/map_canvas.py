"""
Map canvas — Qt WebEngine view embedding the Leaflet map.

Python ↔ JavaScript communication:
  Python → JS : page().runJavaScript(code)
  JS → Python : MapBridge (QObject exposed via QWebChannel)
"""
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import QObject, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEnginePage
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWidgets import QWidget, QVBoxLayout

from app.core.renderer import array_to_png_b64
from app.core.dem_layer import DemLayer, GeoBounds


_HTML_PATH = Path(__file__).parent.parent.parent / "resources" / "map.html"


class MapBridge(QObject):
    """Slots exposed to JavaScript via QWebChannel."""

    mouse_moved       = pyqtSignal(float, float)      # lat, lon
    map_clicked       = pyqtSignal(float, float)
    map_moved         = pyqtSignal(float, float, int) # lat, lon, zoom
    profile_point     = pyqtSignal(float, float)
    profile_complete  = pyqtSignal()
    viewshed_point    = pyqtSignal(float, float)

    @pyqtSlot(float, float)
    def onMouseMove(self, lat: float, lon: float):
        self.mouse_moved.emit(lat, lon)

    @pyqtSlot(float, float)
    def onMapClick(self, lat: float, lon: float):
        self.map_clicked.emit(lat, lon)

    @pyqtSlot(float, float, int)
    def onMapMoved(self, lat: float, lon: float, zoom: int):
        self.map_moved.emit(lat, lon, zoom)

    @pyqtSlot(float, float)
    def onProfilePoint(self, lat: float, lon: float):
        self.profile_point.emit(lat, lon)

    @pyqtSlot()
    def onProfileComplete(self):
        self.profile_complete.emit()

    @pyqtSlot(float, float)
    def onViewshedPoint(self, lat: float, lon: float):
        self.viewshed_point.emit(lat, lon)


class MapCanvas(QWidget):
    """
    Central map widget.

    Provides:
      - OSM / satellite / topo base layer switching
      - DEM overlay management (add/remove/opacity/visibility)
      - Profile line drawing tool
      - Viewshed observer picker tool
    """

    coordinate_changed = pyqtSignal(float, float)   # lat, lon from hover
    clicked            = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._setup_channel()
        self._load_map()

    # ── Setup ──────────────────────────────────────────────────────────────

    def _setup_ui(self):
        self._view = QWebEngineView(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

    def _setup_channel(self):
        self._bridge = MapBridge(self)
        self._channel = QWebChannel(self._view.page())
        self._channel.registerObject("bridge", self._bridge)
        self._view.page().setWebChannel(self._channel)

        self._bridge.mouse_moved.connect(self.coordinate_changed)
        self._bridge.map_clicked.connect(self.clicked)

    def _load_map(self):
        if _HTML_PATH.exists():
            url = QUrl.fromLocalFile(str(_HTML_PATH))
        else:
            url = QUrl("about:blank")
        self._view.load(url)

    # ── Public API ─────────────────────────────────────────────────────────

    def run_js(self, code: str):
        """Execute JavaScript in the map page."""
        self._view.page().runJavaScript(code)

    # Overlay management

    def add_layer(self, layer: DemLayer):
        """Render the layer's data array and push it to Leaflet."""
        if layer.data is None or layer.bounds is None:
            return
        lo = layer.render_min
        hi = layer.render_max
        if lo is None or hi is None:
            lo, hi = layer.auto_range()
        data_url = array_to_png_b64(
            layer.data, layer.colormap, lo, hi, layer.nodata
        )
        b = layer.bounds
        self.run_js(
            f'addDEMOverlay({json.dumps(layer.name)}, {json.dumps(data_url)}, '
            f'{b.south}, {b.west}, {b.north}, {b.east}, {layer.opacity});'
        )

    def remove_layer(self, name: str):
        self.run_js(f'removeDEMOverlay({json.dumps(name)});')

    def set_opacity(self, name: str, opacity: float):
        self.run_js(f'setLayerOpacity({json.dumps(name)}, {opacity});')

    def set_visible(self, name: str, visible: bool):
        v = "true" if visible else "false"
        self.run_js(f'setLayerVisible({json.dumps(name)}, {v});')

    def refresh_layer(self, layer: DemLayer):
        """Re-render and update an existing overlay."""
        self.add_layer(layer)

    def fit_bounds(self, bounds: GeoBounds):
        self.run_js(
            f'fitBounds({bounds.south}, {bounds.west}, {bounds.north}, {bounds.east});'
        )

    def set_base_layer(self, name: str):
        """name: 'osm' | 'satellite' | 'topo'"""
        self.run_js(f'setBaseLayer({json.dumps(name)});')

    # ── Drawing tools ──────────────────────────────────────────────────────

    def start_profile_draw(self):
        self.run_js('startProfileDraw();')

    def stop_profile_draw(self):
        self.run_js('stopProfileDraw();')

    def clear_profile(self):
        self.run_js('clearProfile();')

    def start_viewshed_draw(self):
        self.run_js('startViewshedDraw();')

    def stop_viewshed_draw(self):
        self.run_js('stopViewshedDraw();')

    # ── Bridge signal accessors ────────────────────────────────────────────

    @property
    def bridge(self) -> MapBridge:
        return self._bridge
