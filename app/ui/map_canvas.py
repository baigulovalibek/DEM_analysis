"""
Map canvas — Qt WebEngine view embedding the Leaflet map.

Python ↔ JavaScript communication:
  Python → JS : page().runJavaScript(code)
  JS → Python : MapBridge (QObject exposed via QWebChannel)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, Qt, QTimer, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

from app.core.renderer import array_to_png_b64_sized
from app.core.dem_layer import DemLayer, GeoBounds


def _resource_root() -> Path:
    """Project root in dev, ``sys._MEIPASS`` in a PyInstaller bundle.

    The spec ships ``resources/`` to the bundle's ``_internal/resources/``
    directory; in dev mode the same folder lives at the repo root. Going
    through ``_MEIPASS`` survives onefile/onedir flavour changes and the
    occasional PyInstaller release that rewrites entry-script ``__file__``
    semantics.
    """
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
    return Path(__file__).resolve().parent.parent.parent


_HTML_PATH = _resource_root() / "resources" / "map.html"


class MapBridge(QObject):
    """Slots exposed to JavaScript via QWebChannel."""

    mouse_moved       = pyqtSignal(float, float)      # lat, lon
    map_clicked       = pyqtSignal(float, float)
    map_moved         = pyqtSignal(float, float, int) # lat, lon, zoom
    profile_point     = pyqtSignal(float, float)
    profile_complete  = pyqtSignal()
    viewshed_point    = pyqtSignal(float, float)
    earthquake_clicked = pyqtSignal(int)              # event idx
    section_point     = pyqtSignal(float, float)
    section_complete  = pyqtSignal()
    # Fires once the JS side has finished initialising Leaflet. MapCanvas uses
    # this to distinguish "loadFinished but the WebEngine rendered a blank
    # canvas" (no callback) from "the page is alive" (callback within a few s).
    map_ready          = pyqtSignal()

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

    @pyqtSlot(int)
    def onEarthquakeClick(self, idx: int):
        self.earthquake_clicked.emit(idx)

    @pyqtSlot(float, float)
    def onSectionPoint(self, lat: float, lon: float):
        self.section_point.emit(lat, lon)

    @pyqtSlot()
    def onSectionComplete(self):
        self.section_complete.emit()

    @pyqtSlot()
    def onMapReady(self):
        self.map_ready.emit()


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
    # Fires when the watchdog detects a blank canvas. MainWindow uses this
    # to prompt the user with concrete recovery actions (restart in safe-gfx
    # mode, reload the map, …). Emitted at most once per launch.
    map_unresponsive   = pyqtSignal()

    # Time we give Leaflet to fire onMapReady after loadFinished(ok=True).
    # Past this we assume Chromium produced a blank canvas (typical on
    # machines where the GPU init failed and DEM_ANALYST_SAFE_GFX wasn't
    # set) and show a recovery banner.
    _READY_TIMEOUT_MS = 6000

    def __init__(self, parent=None):
        super().__init__(parent)
        self._page_ready = False
        self._map_alive = False
        # Try one silent reload before giving up — sometimes QWebChannel hands
        # off late and Leaflet initialises after a refresh. Only the second
        # failure surfaces a UI prompt.
        self._reload_attempts = 0
        self._unresponsive_emitted = False
        self._pending_js: list[str] = []
        self._setup_ui()
        self._setup_channel()
        self._load_map()

    # ── Setup ──────────────────────────────────────────────────────────────

    def _setup_ui(self):
        self._view = QWebEngineView(self)
        # Allow CDN scripts (Leaflet) to load when the page is served from file://
        self._view.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True
        )
        # Some QtWebEngine builds also require this for local file:// pages to
        # XHR-load adjacent files. Vendored Leaflet uses <script> tags so this
        # is belt-and-braces, but flipping it on costs nothing.
        self._view.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

        # Recovery banner shown when Chromium loads but produces a blank
        # canvas (the usual cause of "the map is blank on my coworker's PC").
        # Hidden by default; floats above the QWebEngineView.
        self._diagnostic = QLabel(self)
        self._diagnostic.setWordWrap(True)
        self._diagnostic.setTextFormat(Qt.TextFormat.RichText)
        self._diagnostic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._diagnostic.setStyleSheet(
            "QLabel { background: rgba(40,20,10,235); color: #ffd9a8; "
            "border: 1px solid #c8843a; padding: 16px; font-size: 12px; }"
        )
        self._diagnostic.setOpenExternalLinks(False)
        self._diagnostic.hide()

    def _setup_channel(self):
        self._bridge = MapBridge(self)
        self._channel = QWebChannel(self._view.page())
        self._channel.registerObject("bridge", self._bridge)
        self._view.page().setWebChannel(self._channel)

        self._bridge.mouse_moved.connect(self.coordinate_changed)
        self._bridge.map_clicked.connect(self.clicked)
        self._bridge.map_ready.connect(self._on_map_ready)

    def _load_map(self):
        if _HTML_PATH.exists():
            url = QUrl.fromLocalFile(str(_HTML_PATH))
        else:
            print(f"[MapCanvas] map.html not found at {_HTML_PATH}", file=sys.stderr)
            url = QUrl("about:blank")
        self._view.loadFinished.connect(self._on_load_finished)
        self._view.load(url)

    def _on_load_finished(self, ok: bool):
        self._page_ready = bool(ok)
        if not ok:
            # Surface the failure rather than silently disabling the map.
            print(
                f"[MapCanvas] Failed to load {_HTML_PATH} — the map will be blank.",
                file=sys.stderr,
            )
            self._show_diagnostic(
                "<b>Map page failed to load.</b><br/><br/>"
                f"Could not load <code>{_HTML_PATH.name}</code>. The "
                "<code>resources/</code> folder may be missing or unreadable. "
                "Reinstall or re-extract the bundle and try again."
            )
            return
        # Flush any JS that was requested before the page finished loading
        # (e.g. a DEM opened from the command line during start-up).
        pending, self._pending_js = self._pending_js, []
        for code in pending:
            self._view.page().runJavaScript(code)

        # If onMapReady doesn't fire within the watchdog window, we assume
        # Chromium initialised but rendered a blank canvas (the typical
        # "blank map" symptom on locked-down or driver-broken Windows
        # machines). Pop a banner explaining the workaround.
        QTimer.singleShot(self._READY_TIMEOUT_MS, self._check_map_alive)

    def _on_map_ready(self):
        self._map_alive = True
        self._diagnostic.hide()

    def _check_map_alive(self):
        if self._map_alive:
            return
        # First failure: silently retry once. QtWebEngine sometimes wins the
        # race only on a second pass (the QWebChannel handshake can land
        # after the page already finished loading).
        if self._reload_attempts == 0:
            self._reload_attempts = 1
            print("[MapCanvas] map silent after load — reloading once",
                  file=sys.stderr)
            self._page_ready = False
            self._view.reload()
            return
        # Second failure: surface to MainWindow so the user can pick a
        # recovery action with full context (which safe-gfx state they're
        # already in, etc.). Also show the inline banner as a fallback in
        # case the parent ignores the signal.
        if not self._unresponsive_emitted:
            self._unresponsive_emitted = True
            self.map_unresponsive.emit()
        self._show_diagnostic(
            "<b>Map is not rendering.</b><br/><br/>"
            "The WebEngine page loaded but Leaflet didn't initialise — this "
            "is almost always a graphics-driver / GPU-init failure on the "
            "current machine.<br/><br/>"
            "Use <b>Help → Safe Graphics Mode</b> to switch Chromium to "
            "software rendering, then restart DEM Analyst."
        )

    def _show_diagnostic(self, html: str):
        self._diagnostic.setText(html)
        # Re-position the banner over the centre of the view.
        margin = 24
        w = max(320, self.width() - 2 * margin)
        h = 220
        x = (self.width() - w) // 2
        y = (self.height() - h) // 2
        self._diagnostic.setGeometry(x, max(margin, y), w, h)
        self._diagnostic.raise_()
        self._diagnostic.show()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep the diagnostic banner centred when the user resizes the window.
        if self._diagnostic.isVisible():
            self._show_diagnostic(self._diagnostic.text())

    # ── Public API ─────────────────────────────────────────────────────────

    def run_js(self, code: str):
        """Execute JavaScript in the map page, buffering until it is ready."""
        if self._page_ready:
            self._view.page().runJavaScript(code)
        else:
            self._pending_js.append(code)

    # Overlay management

    def add_layer(self, layer: DemLayer):
        """Render the layer's data array and push it to Leaflet."""
        if layer.data is None or layer.bounds is None:
            return
        lo = layer.render_min
        hi = layer.render_max
        if lo is None or hi is None:
            lo, hi = layer.auto_range()
        # D8 flow direction grids are categorical (8 power-of-two codes), so
        # bypass the continuous stretch and use a fixed LUT instead.
        categorical = (
            layer.product == "flow_direction"
            and np.issubdtype(layer.data.dtype, np.integer)
        )
        data_url, px_w, px_h = array_to_png_b64_sized(
            layer.data, layer.colormap, lo, hi, layer.nodata,
            categorical=categorical,
        )
        b = layer.bounds
        # px_w / px_h are the overlay's native (post-decimation) resolution.
        # The map page uses them to render the overlay smoothly while it is
        # shown at or below native size, and crisply once it is upscaled past
        # native — so zooming in to inspect cells stays sharp instead of blurry.
        self.run_js(
            f'addDEMOverlay({json.dumps(layer.name)}, {json.dumps(data_url)}, '
            f'{b.south}, {b.west}, {b.north}, {b.east}, {layer.opacity}, '
            f'{px_w}, {px_h});'
        )
        if not layer.visible:
            self.set_visible(layer.name, False)

    def remove_layer(self, name: str):
        self.run_js(f'removeDEMOverlay({json.dumps(name)});')

    def rename_layer(self, old: str, new: str):
        self.run_js(f'renameDEMOverlay({json.dumps(old)}, {json.dumps(new)});')

    def set_opacity(self, name: str, opacity: float):
        self.run_js(f'setLayerOpacity({json.dumps(name)}, {opacity});')

    def set_visible(self, name: str, visible: bool):
        v = "true" if visible else "false"
        self.run_js(f'setLayerVisible({json.dumps(name)}, {v});')

    def set_z_index(self, name: str, z_index: int):
        self.run_js(f'setLayerZIndex({json.dumps(name)}, {int(z_index)});')

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

    # ── Value legend ─────────────────────────────────────────────────────────

    def set_legend(
        self,
        title: str,
        stops: list[str],
        vmin: float,
        vmax: float,
        units: str = "",
        categorical: list[dict] | None = None,
    ):
        """Show the colour→value legend for the selected raster overlay.

        ``stops`` is an evenly spaced list of ``"#rrggbb"`` colours (low→high);
        ``categorical`` (when given) is a list of ``{"label", "color"}`` dicts
        and takes precedence, rendering discrete swatches instead of a ramp.
        """
        payload = {
            "title": title,
            "stops": stops,
            "vmin": vmin,
            "vmax": vmax,
            "units": units,
            "categorical": categorical,
        }
        self.run_js(f'setLegend({json.dumps(payload)});')

    def clear_legend(self):
        self.run_js('clearLegend();')

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

    def start_section_draw(self):
        self.run_js('startSectionDraw();')

    def stop_section_draw(self):
        self.run_js('stopSectionDraw();')

    def clear_section(self):
        self.run_js('clearSection();')

    # ── Earthquake catalog ─────────────────────────────────────────────────

    def set_earthquakes(self, events: list[dict], style: dict):
        """Push the full catalog payload + style to the map."""
        self.run_js(
            f'setEarthquakes({json.dumps(events)}, {json.dumps(style)});'
        )

    def set_earthquake_style(self, style: dict):
        self.run_js(f'setEarthquakeStyle({json.dumps(style)});')

    def clear_earthquakes(self):
        self.run_js('clearEarthquakes();')

    def highlight_earthquake(self, idx: int | None, open_popup: bool = True):
        idx_js = "null" if idx is None else str(int(idx))
        flag = "true" if open_popup else "false"
        self.run_js(f'highlightEarthquake({idx_js}, {flag});')

    # ── 3D view sync ───────────────────────────────────────────────────────

    def set_camera_frustum(
        self,
        corners_latlon: list[tuple[float, float]] | None,
        camera_latlon: tuple[float, float] | None = None,
    ):
        """Show / hide the 3D camera's ground frustum on the 2D map.

        ``corners_latlon`` is a list of ``(lat, lon)`` tuples (typically four)
        tracing the camera's view extent on the ground plane.  Pass ``None``
        to clear the overlay.
        """
        if not corners_latlon:
            self.run_js('setCameraFrustum(null, null);')
            return
        corners_js = json.dumps([[lat, lon] for lat, lon in corners_latlon])
        cam_js = (
            json.dumps([camera_latlon[0], camera_latlon[1]])
            if camera_latlon is not None
            else "null"
        )
        self.run_js(f'setCameraFrustum({corners_js}, {cam_js});')

    # ── Bridge signal accessors ────────────────────────────────────────────

    @property
    def bridge(self) -> MapBridge:
        return self._bridge
