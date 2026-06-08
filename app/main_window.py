"""
Main application window.

Layout (QGIS-inspired):
  - Menu bar + toolbar (top)
  - Left dock: Layer panel
  - Right dock: Properties panel  (top) + Analysis panel (bottom)
  - Central widget: Map canvas (Leaflet/WebEngine)
  - Bottom dock: 3D view (closable)
  - Status bar (coordinate, elevation, CRS, scale)
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence
from PyQt6.QtWidgets import (
    QMainWindow, QMessageBox, QFileDialog, QDockWidget,
    QLabel, QProgressBar, QWidget, QToolBar, QStatusBar,
    QApplication,
)

import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

from app.config import COLORMAPS, DEFAULT_OPACITY, LEGEND_UNITS
from app.core.dem_layer import DemLayer, GeoBounds
from app.core.earthquakes import EarthquakeCatalog
from app.core.layer_manager import LayerManager
from app.core.worker import AnalysisWorker
from app.core.renderer import array_to_png_b64, colormap_stops, d8_legend_entries

from app.ui.map_canvas import MapCanvas
from app.ui.layer_panel import LayerPanel
from app.ui.analysis_panel import AnalysisPanel
from app.ui.properties_panel import PropertiesPanel
from app.ui.view3d import View3DWindow
from app.ui.profile_dialog import ProfileDialog
from app.ui.section_dialog import EarthquakeSectionDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DEM Analyst")
        self.resize(1400, 900)

        self._mgr = LayerManager(self)
        self._worker: Optional[AnalysisWorker] = None
        self._profile_latlons: list[tuple[float, float]] = []
        self._viewshed_latlon: Optional[tuple[float, float]] = None

        # Earthquake catalog state — None until the user loads one (or until
        # ``catalog.xlsx`` is auto-loaded from CWD at start-up).
        self._catalog: Optional[EarthquakeCatalog] = None
        self._section_latlons: list[tuple[float, float]] = []

        # Pre-computed intermediate results (keyed by DEM name)
        self._flow_dir_cache: dict = {}      # D8 int32 codes
        self._flow_angle_cache: dict = {}    # D-infinity float32 angles
        self._flow_accum_cache: dict = {}
        self._slope_rad_cache: dict = {}
        self._filled_cache: dict = {}

        self._build_ui()
        self._connect_signals()
        self._apply_stylesheet()
        # Pick up a catalog.xlsx sitting next to main.py without making the
        # user open it explicitly.  Safe no-op when the file isn't there.
        # self._autoload_catalog_if_present()

    # ── UI Construction ────────────────────────────────────────────────────

    def _build_ui(self):
        # Central map
        self._map = MapCanvas(self)
        self.setCentralWidget(self._map)

        # Docks
        self._layer_panel = LayerPanel(self._mgr, self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._layer_panel)

        self._analysis_panel = AnalysisPanel(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._analysis_panel)

        self._props_panel = PropertiesPanel(self._mgr, self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._props_panel)

        # 3D view is a separate top-level window (QGIS-style), created lazily
        # on first toggle of the toolbar action.
        self._3d_window: Optional[View3DWindow] = None

        # Toolbar
        self._build_toolbar()

        # Status bar
        self._build_statusbar()

        # Menu
        self._build_menu()

    def _build_toolbar(self):
        tb = QToolBar("Main", self)
        tb.setObjectName("main_toolbar")
        tb.setMovable(False)
        self.addToolBar(tb)

        self._act_open  = QAction("📂 Open DEM", self, shortcut=QKeySequence.StandardKey.Open)
        self._act_save  = QAction("💾 Export Layer", self)
        tb.addAction(self._act_open)
        tb.addAction(self._act_save)
        tb.addSeparator()

        self._act_3d    = QAction("3D View", self, checkable=True)
        self._act_base_osm  = QAction("OSM",       self, checkable=True); self._act_base_osm.setChecked(True)
        self._act_base_sat  = QAction("Satellite",  self, checkable=True)
        self._act_base_topo = QAction("Topo",       self, checkable=True)
        self._act_base_none = QAction("None",       self, checkable=True)

        # Group the base-layer actions exclusively so the user can't end up
        # with all four unchecked (which previously left the map blank with
        # no UI cue that "None" was implicit).
        self._base_group = QActionGroup(self)
        self._base_group.setExclusive(True)
        for act in (self._act_base_osm, self._act_base_sat,
                    self._act_base_topo, self._act_base_none):
            act.setActionGroup(self._base_group)

        tb.addAction(self._act_3d)
        tb.addSeparator()
        tb.addWidget(QLabel(" Base: "))
        for act in (self._act_base_osm, self._act_base_sat,
                    self._act_base_topo, self._act_base_none):
            tb.addAction(act)

        self._act_open.triggered.connect(self._open_dem)
        self._act_save.triggered.connect(self._export_layer)
        self._act_3d.toggled.connect(self._toggle_3d)
        self._act_base_osm.triggered.connect(lambda: self._set_base("osm"))
        self._act_base_sat.triggered.connect(lambda: self._set_base("satellite"))
        self._act_base_topo.triggered.connect(lambda: self._set_base("topo"))
        self._act_base_none.triggered.connect(lambda: self._set_base("none"))

    def _build_statusbar(self):
        sb = self.statusBar()
        self._sb_crs   = QLabel("CRS: —")
        self._sb_coord = QLabel("Lon: — Lat: —")
        self._sb_elev  = QLabel("Elev: —")
        self._sb_msg   = QLabel("")
        for lbl in (self._sb_crs, self._sb_coord, self._sb_elev, self._sb_msg):
            sb.addPermanentWidget(lbl)

    def _build_menu(self):
        mbar = self.menuBar()

        # File
        fm = mbar.addMenu("File")
        fm.addAction(self._act_open)
        fm.addAction(self._act_save)
        fm.addSeparator()
        self._act_open_eq = QAction("Open Earthquake Catalog…", self)
        self._act_open_eq.triggered.connect(self._open_earthquake_catalog)
        fm.addAction(self._act_open_eq)
        self._act_clear_eq = QAction("Close Earthquake Catalog", self)
        self._act_clear_eq.triggered.connect(self._clear_earthquake_catalog)
        fm.addAction(self._act_clear_eq)
        fm.addSeparator()
        fm.addAction("Exit", self.close)

        # View — create QActions explicitly with `self` as parent so their
        # lifetime isn't tied to the menu (which would invalidate the slot
        # connection if the menu is rebuilt).
        vm = mbar.addMenu("View")
        vm.addAction(self._act_3d)
        vm.addSeparator()
        act_show_layer = QAction("Layer Panel", self)
        act_show_layer.triggered.connect(self._layer_panel.show)
        vm.addAction(act_show_layer)
        act_show_anal = QAction("Analysis Panel", self)
        act_show_anal.triggered.connect(self._analysis_panel.show)
        vm.addAction(act_show_anal)
        act_show_props = QAction("Properties Panel", self)
        act_show_props.triggered.connect(self._props_panel.show)
        vm.addAction(act_show_props)

        # Analysis — quick launch
        am = mbar.addMenu("Analysis")
        for name, key in [
            ("Hillshade", "hillshade"),
            ("Slope", "slope"),
            ("Aspect", "aspect"),
            ("TWI", "twi"),
            ("TPI", "tpi"),
        ]:
            act = QAction(name, self)
            act.triggered.connect(lambda checked, k=key: self._quick_run(k))
            am.addAction(act)
        am.addSeparator()
        self._act_eq_section = QAction("Earthquake Cross-Section…", self)
        self._act_eq_section.triggered.connect(self._start_section_draw)
        am.addAction(self._act_eq_section)

        # Help
        hm = mbar.addMenu("Help")
        self._act_safe_gfx = QAction("Safe Graphics Mode (restart required)",
                                     self, checkable=True)
        self._act_safe_gfx.setToolTip(
            "Run Chromium without GPU acceleration. Turn this on if the map "
            "area is blank on this PC. Takes effect after restart."
        )
        # Reflect what the *current* process is actually doing, not just the
        # raw persistent flag. main.py exposes its decision via this env var
        # so the menu's tick matches reality even on first launch (when the
        # frozen bundle defaults safe-gfx ON before the user has chosen).
        import sys as _sys
        from PyQt6.QtCore import QSettings as _QSettings
        _persisted = _QSettings().value("safe_gfx", None)
        if _persisted is None:
            _active = bool(getattr(_sys, "frozen", False))
        else:
            _active = str(_persisted).strip().lower() in ("1", "true", "yes")
        self._act_safe_gfx.setChecked(_active)
        self._act_safe_gfx.toggled.connect(self._toggle_safe_gfx)
        hm.addAction(self._act_safe_gfx)
        act_open_log = QAction("Open Diagnostic Log Folder…", self)
        act_open_log.setToolTip(
            "Reveal the folder containing log.txt. Attach this file to bug "
            "reports — it captures every error since the last launch."
        )
        act_open_log.triggered.connect(self._open_log_folder)
        hm.addAction(act_open_log)
        hm.addSeparator()
        act_about = QAction("About", self)
        act_about.triggered.connect(self._about)
        hm.addAction(act_about)

    def _apply_stylesheet(self):
        qss_path = Path(__file__).parent / "ui" / "style.qss"
        if qss_path.exists():
            with open(qss_path, encoding="utf-8") as f:
                QApplication.instance().setStyleSheet(f.read())

    # ── Signal wiring ──────────────────────────────────────────────────────

    def _connect_signals(self):
        # Layer panel
        self._layer_panel.remove_requested.connect(self._remove_layer)
        self._layer_panel.zoom_requested.connect(self._zoom_to_layer)
        self._layer_panel.export_requested.connect(self._export_layer)
        self._layer_panel.visibility_toggled.connect(self._map.set_visible)
        self._layer_panel.active_dem_change.connect(self._mgr.set_active_dem)
        # Earthquake-row hooks
        self._layer_panel.earthquake_selected.connect(self._on_earthquake_selected)
        self._layer_panel.earthquake_visibility_toggled.connect(
            self._on_earthquake_visibility
        )
        self._layer_panel.earthquake_zoom_requested.connect(self._zoom_to_earthquakes)
        self._layer_panel.earthquake_remove_requested.connect(
            self._clear_earthquake_catalog
        )
        self._layer_panel.earthquake_open_requested.connect(
            self._open_earthquake_catalog
        )

        # Analysis panel
        self._analysis_panel.run_analysis.connect(self._dispatch_analysis)
        self._analysis_panel.start_tool.connect(self._activate_tool)
        self._analysis_panel.cancel_analysis.connect(self._on_cancel_analysis)

        # Properties panel
        self._props_panel.style_changed.connect(self._refresh_layer_on_map)
        self._props_panel.earthquake_style_changed.connect(
            self._on_earthquake_style_changed
        )

        # Layer manager
        self._mgr.layer_updated.connect(self._on_layer_updated)
        self._mgr.layer_renamed.connect(self._on_layer_renamed)
        self._mgr.layers_changed.connect(self._reapply_layer_order)
        # Keep the 3D scene in sync with the 2D layer stack.
        self._mgr.layer_updated.connect(lambda _name: self._sync_3d_drape())
        self._mgr.layers_changed.connect(self._sync_3d_drape)
        self._mgr.active_dem_changed.connect(
            lambda _name: self._sync_3d_active_dem(self._mgr.active_dem)
        )
        # Keep the map's colour→value legend tied to the selected layer.
        self._mgr.layer_selected.connect(self._update_legend)
        self._mgr.layers_changed.connect(self._refresh_legend)

        # Map canvas-level signals (watchdog, etc.) bypass the JS bridge.
        self._map.map_unresponsive.connect(self._on_map_unresponsive)

        # Map bridge
        self._map.bridge.mouse_moved.connect(self._on_mouse_moved)
        self._map.bridge.profile_point.connect(self._on_profile_point)
        self._map.bridge.profile_complete.connect(self._on_profile_complete)
        self._map.bridge.viewshed_point.connect(self._on_viewshed_point)
        self._map.bridge.earthquake_clicked.connect(self._on_earthquake_clicked)
        self._map.bridge.section_point.connect(self._on_section_point)
        self._map.bridge.section_complete.connect(self._on_section_complete)

    # ── File operations ────────────────────────────────────────────────────

    def _open_dem(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open DEM",
            "", "Raster files (*.tif *.tiff *.img *.dem *.adf *.nc *.hgt);;All files (*)"
        )
        if not path:
            return
        try:
            self._load_dem_file(Path(path))
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error loading DEM", str(e))

    def _load_dem_file(self, path: Path):
        with rasterio.open(path) as src:
            # Read band 1 directly as float32 — reading as float64 first and
            # casting down transiently doubles the memory footprint.
            data = src.read(1, out_dtype=np.float32)
            nodata = src.nodata
            crs = src.crs
            transform = src.transform
            bounds_native = src.bounds
            n_bands = src.count

        if n_bands > 1:
            self._status(
                f"Note: {path.name} has {n_bands} bands; only band 1 was loaded."
            )

        # Convert bounds to WGS84.  Surface reprojection failures to the user
        # instead of silently faking native coordinates as WGS84 (which puts
        # projected rasters thousands of degrees off the map).
        reproj_failed = False
        if crs and crs.to_epsg() != 4326:
            try:
                west, south, east, north = transform_bounds(
                    crs, CRS.from_epsg(4326), *bounds_native
                )
            except Exception as exc:
                reproj_failed = True
                QMessageBox.warning(
                    self, "Coordinate reprojection failed",
                    f"Could not reproject {path.name} bounds to WGS84:\n{exc}\n\n"
                    "The DEM may appear at the wrong location on the map."
                )
                west, south, east, north = (
                    bounds_native.left, bounds_native.bottom,
                    bounds_native.right, bounds_native.top
                )
        else:
            west, south, east, north = (
                bounds_native.left, bounds_native.bottom,
                bounds_native.right, bounds_native.top
            )

        # Approximate cell size in metres.  Honour rotation terms so rotated
        # affines (which set ``a`` to 0) don't yield a 0 cell size.
        lat_center = (south + north) / 2
        a, b, _, _, d, e = transform.a, transform.b, 0, 0, transform.d, transform.e
        # Pixel width/height = magnitudes of the affine column vectors.
        px_w_units = float(np.hypot(a, d))
        px_h_units = float(np.hypot(b, e))
        if crs and crs.is_geographic:
            lon_size_m = px_w_units * 111320 * np.cos(np.radians(lat_center))
            lat_size_m = px_h_units * 111320
            cell_size_m = (lon_size_m + lat_size_m) / 2
        else:
            cell_size_m = (px_w_units + px_h_units) / 2

        layer = DemLayer(
            name=path.stem,
            product="dem",
            data=data,
            nodata=nodata,
            bounds=GeoBounds(west=west, south=south, east=east, north=north),
            crs_wkt=crs.to_wkt() if crs else "",
            cell_size_m=cell_size_m,
            source_path=path,
            colormap=COLORMAPS["dem"],
            opacity=DEFAULT_OPACITY,
        )

        self._mgr.add(layer)
        self._map.add_layer(layer)
        self._reapply_layer_order()
        self._map.fit_bounds(layer.bounds)
        self._mgr.select(layer.name)
        self._sb_crs.setText(f"CRS: {crs.to_epsg() if crs else 'unknown'}")
        self._status(f"Loaded {path.name}  ({layer.shape[0]}×{layer.shape[1]})")

    def _export_layer(self, name: str = ""):
        if not name:
            layer = self._mgr.selected
            if not layer:
                return
            name = layer.name
        layer = self._mgr.get(name)
        if not layer or layer.data is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Layer", f"{layer.name}.tif", "GeoTIFF (*.tif)"
        )
        if not path:
            return

        # Use the layer's OWN bounds/CRS — for derived layers this comes from
        # the parent DEM at creation time and matches the result grid exactly.
        # Falling back to the active DEM (as the previous implementation did)
        # produced misaligned exports whenever the active DEM differed from
        # the layer being exported.
        src_layer = layer
        if src_layer.bounds is None:
            # Try parent if the derived layer lost its bounds for some reason.
            parent = self._mgr.get(layer.parent_name) if layer.parent_name else None
            src_layer = parent or layer

        if src_layer.bounds is not None:
            b = src_layer.bounds
            from rasterio.transform import from_bounds
            transform = from_bounds(b.west, b.south, b.east, b.north,
                                    layer.shape[1], layer.shape[0])
            crs = CRS.from_wkt(src_layer.crs_wkt) if src_layer.crs_wkt else CRS.from_epsg(4326)
        else:
            transform = None
            crs = None

        with rasterio.open(
            path, 'w',
            driver='GTiff',
            height=layer.shape[0],
            width=layer.shape[1],
            count=1,
            dtype=layer.data.dtype,
            crs=crs,
            transform=transform,
            nodata=layer.nodata,
        ) as dst:
            dst.write(layer.data, 1)
        self._status(f"Exported {name} → {path}")

    # ── Layer actions ──────────────────────────────────────────────────────

    def _remove_layer(self, name: str):
        layer = self._mgr.get(name)
        if layer is not None:
            # Drop any cached intermediates derived from this layer.
            for cache in (self._flow_dir_cache, self._flow_angle_cache,
                          self._flow_accum_cache, self._slope_rad_cache,
                          self._filled_cache):
                cache.pop(layer.uid, None)
        self._map.remove_layer(name)
        self._mgr.remove(name)

    def _zoom_to_layer(self, name: str):
        layer = self._mgr.get(name)
        if layer and layer.bounds:
            self._map.fit_bounds(layer.bounds)

    def _refresh_layer_on_map(self, name: str):
        layer = self._mgr.get(name)
        if layer:
            self._map.refresh_layer(layer)
            # Re-adding the overlay resets its stacking order.
            self._reapply_layer_order()
            # Colormap / render-range edits change what the legend should show.
            self._update_legend(name)

    # ── Value legend ─────────────────────────────────────────────────────────

    def _update_legend(self, name: str):
        """Push the colour→value legend for ``name`` to the map (or clear it)."""
        layer = self._mgr.get(name)
        if layer is None or layer.data is None:
            self._map.clear_legend()
            return

        # D8 flow-direction grids are categorical — a gradient would be
        # meaningless, so show the eight direction swatches instead.
        if (
            layer.product == "flow_direction"
            and np.issubdtype(layer.data.dtype, np.integer)
        ):
            cats = [{"label": lbl, "color": col} for lbl, col in d8_legend_entries()]
            self._map.set_legend(layer.name, [], 0.0, 0.0, categorical=cats)
            return

        lo = layer.render_min
        hi = layer.render_max
        if lo is None or hi is None:
            lo, hi = layer.auto_range()
        self._map.set_legend(
            layer.name,
            colormap_stops(layer.colormap),
            float(lo),
            float(hi),
            units=LEGEND_UNITS.get(layer.product, ""),
        )

    def _refresh_legend(self):
        """Re-evaluate the legend after the layer stack changes (add/remove).

        Falls back to clearing when the selected layer is gone.
        """
        sel = self._mgr.selected
        if sel is None:
            self._map.clear_legend()
        else:
            self._update_legend(sel.name)

    def _on_layer_updated(self, name: str):
        layer = self._mgr.get(name)
        if layer:
            self._map.set_opacity(name, layer.opacity)
            self._map.set_visible(name, layer.visible)

    def _on_layer_renamed(self, old: str, new: str):
        self._map.rename_layer(old, new)

    def _reapply_layer_order(self):
        """Push the layer stacking order to the map (top layer = highest z)."""
        layers = self._mgr.all()           # index 0 = top of the stack
        n = len(layers)
        for i, layer in enumerate(layers):
            self._map.set_z_index(layer.name, 200 + (n - i))

    # ── Analysis dispatch ──────────────────────────────────────────────────

    def _quick_run(self, product: str):
        # Check prerequisites that the dispatcher would silently bail on, so
        # the user gets a modal explaining what's missing instead of a
        # status-bar message that disappears after a few seconds.
        dem = self._mgr.active_dem
        if dem is None or dem.data is None:
            QMessageBox.information(
                self, "No DEM loaded",
                "Open a DEM first (File → Open DEM)."
            )
            return
        missing = self._missing_prerequisites(product, dem)
        if missing:
            QMessageBox.information(
                self, f"Run {product} — prerequisites missing",
                "Before this analysis can run, please compute first:\n\n• "
                + "\n• ".join(missing)
            )
            return
        self._dispatch_analysis(product, {})

    def _missing_prerequisites(self, product: str, dem: DemLayer) -> list[str]:
        """Return a human-readable list of missing cached intermediates."""
        uid = dem.uid
        deps: list[str] = []
        if product in ("twi", "spi"):
            if self._slope_rad_cache.get(uid) is None:
                deps.append("Slope")
            if self._flow_accum_cache.get(uid) is None:
                deps.append("Flow Accumulation")
        elif product == "flow_accumulation":
            if (self._flow_dir_cache.get(uid) is None
                    and self._flow_angle_cache.get(uid) is None):
                deps.append("Flow Direction")
        elif product == "streams":
            if self._flow_accum_cache.get(uid) is None:
                deps.append("Flow Accumulation")
        return deps

    def _dispatch_analysis(self, product: str, params: dict):
        dem = self._mgr.active_dem
        if dem is None or dem.data is None:
            # Modal feedback — the status bar message used to disappear in 8 s
            # and clients reported "the Compute button doesn't work" because
            # they never saw the hint.
            QMessageBox.information(
                self, "No DEM loaded",
                "Open a DEM first via File → Open DEM (or the toolbar)."
            )
            return
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(
                self, "Analysis in progress",
                "Another analysis is still running. Wait for it to finish "
                "or click Cancel before starting a new one."
            )
            return
        # Surface missing prerequisites the same way as the quick-run menu so
        # the user knows exactly which intermediate analysis to run first
        # (TWI/SPI need Slope+Flow Accumulation, Streams need Flow Accumulation,
        # Flow Accumulation needs Flow Direction).
        missing = self._missing_prerequisites(product, dem)
        if missing:
            QMessageBox.information(
                self, f"Run {product} — prerequisites missing",
                "Before this analysis can run, please compute first:\n\n• "
                + "\n• ".join(missing)
            )
            return

        # Hand the algorithms a NaN-masked copy of the elevation grid.
        # Neighbourhood operations (TPI, curvature, roughness, uniform_filter
        # for SD, …) would otherwise mix the raw nodata sentinel (-9999 etc.)
        # into their kernels and produce extreme spikes across many cells
        # near nodata — which then break the percentile-based render stretch.
        # NaNs propagate cleanly through arithmetic and are masked transparent
        # by the renderer, leaving a thin nodata border instead.
        data = dem.valid_data.astype(np.float64)
        cs = dem.cell_size_m

        # Resolve BEFORE entering the running state so a missing-dependency
        # bail-out cannot leave the progress bar stuck on screen.
        func, extra = self._resolve_analysis(product, params, data, cs, dem)
        if func is None:
            return  # _resolve_analysis already reported why

        self._status(f"Running {product}…")
        self._analysis_panel.set_running(True)

        self._worker = AnalysisWorker(func, extra, product=product,
                                      dem=dem, params=params)
        self._worker.progress.connect(self._analysis_panel.set_progress)
        self._worker.result.connect(self._on_worker_result)
        self._worker.error.connect(self._on_worker_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_cancel_analysis(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._status("Cancelling analysis…")

    def _resolve_analysis(self, product, params, data, cs, dem):
        """Map product name → (callable, kwargs_for_worker)."""
        from app.core.derivatives.hillshade import hillshade
        from app.core.derivatives.slope import slope
        from app.core.derivatives.aspect import aspect
        from app.core.derivatives.curvature import profile_curvature, plan_curvature, mean_curvature
        from app.core.visibility.multidirectional import multidirectional_hillshade
        from app.core.hydrology.fill_sinks import priority_flood, breach_and_fill
        from app.core.hydrology.flow_direction import d8_flow_direction, d_infinity_flow_direction
        from app.core.hydrology.flow_accumulation import d8_flow_accumulation, d_inf_flow_accumulation
        from app.core.hydrology.streams import extract_streams, auto_threshold
        from app.core.indices.twi import twi
        from app.core.indices.spi import spi
        from app.core.indices.tpi import tpi
        from app.core.indices.tri import tri
        from app.core.indices.roughness import roughness_range, roughness_sd, vrm, surface_area_ratio
        from app.core.visibility.svf import sky_view_factor, positive_openness, negative_openness

        nd = dem.nodata
        k = product

        # ── Primary derivatives ────────────────────────────────────────────
        if k == "hillshade":
            return hillshade, dict(dem=data, cell_size=cs, nodata=nd, **params)

        if k == "slope":
            return slope, dict(dem=data, cell_size=cs, nodata=nd, **params)

        if k == "aspect":
            return aspect, dict(dem=data, cell_size=cs, nodata=nd, **params)

        if k == "curvature":
            ct = params.pop("curv_type", "profile")
            fn = {"profile": profile_curvature, "plan": plan_curvature, "mean": mean_curvature}[ct]
            return fn, dict(dem=data, cell_size=cs, nodata=nd, **params)

        if k == "multidirectional":
            return multidirectional_hillshade, dict(dem=data, cell_size=cs, nodata=nd, **params)

        # ── Hydrology ──────────────────────────────────────────────────────
        if k == "fill_sinks":
            method = params.pop("method", "Priority-Flood")
            eps    = params.pop("epsilon", 1e-6)
            fn = priority_flood if "Priority" in method else breach_and_fill
            self._status("Filling sinks…")
            return fn, dict(dem=data, nodata=nd, epsilon=eps)

        if k == "flow_direction":
            algo = params.pop("algorithm", "D8")
            if algo == "D8":
                def _d8(dem, cell_size, nodata, _progress=None):
                    return d8_flow_direction(dem, cell_size, nodata)
                return _d8, dict(dem=data, cell_size=cs, nodata=nd)
            else:
                def _dinf(dem, cell_size, nodata, _progress=None):
                    angle, slope_mag = d_infinity_flow_direction(dem, cell_size, nodata)
                    return angle   # visualise the angle
                return _dinf, dict(dem=data, cell_size=cs, nodata=nd)

        if k == "flow_accumulation":
            uid = dem.uid
            fd       = self._flow_dir_cache.get(uid)
            fa_angle = self._flow_angle_cache.get(uid)
            if fd is None and fa_angle is None:
                QMessageBox.information(
                    self, "Flow Accumulation — prerequisite missing",
                    "Compute Flow Direction first (Hydrological Analysis → "
                    "Flow Direction)."
                )
                return None, None
            if fa_angle is not None:
                def _accum_dinf(dem, angle, nodata, _progress=None):
                    return d_inf_flow_accumulation(dem, angle, nodata=nodata,
                                                   _progress=_progress)
                return _accum_dinf, dict(dem=data, angle=fa_angle, nodata=nd)
            else:
                def _accum(dem, flow_dir, nodata, _progress=None):
                    return d8_flow_accumulation(dem, flow_dir, nodata=nodata,
                                                _progress=_progress)
                return _accum, dict(dem=data, flow_dir=fd, nodata=nd)

        if k == "streams":
            fa   = self._flow_accum_cache.get(dem.uid)
            if fa is None:
                QMessageBox.information(
                    self, "Streams — prerequisite missing",
                    "Compute Flow Accumulation first (Hydrological Analysis "
                    "→ Flow Direction → Flow Accumulation)."
                )
                return None, None
            auto_thr = params.pop("auto_threshold", True)
            user_thr = params.pop("threshold", 1000.0)
            def _streams(dem, fa, auto_thr, user_thr, _progress=None):
                thr = auto_threshold(fa) if auto_thr else user_thr
                return extract_streams(fa, thr).astype(np.float32)
            return _streams, dict(dem=data, fa=fa, auto_thr=auto_thr, user_thr=user_thr)

        # ── Indices ────────────────────────────────────────────────────────
        if k in ("twi", "spi"):
            fa   = self._flow_accum_cache.get(dem.uid)
            slp  = self._slope_rad_cache.get(dem.uid)
            if fa is None or slp is None:
                missing = []
                if slp is None:
                    missing.append("Slope (Primary Derivatives → Slope)")
                if fa is None:
                    missing.append(
                        "Flow Accumulation (Hydrological Analysis → "
                        "Flow Direction → Flow Accumulation)"
                    )
                QMessageBox.information(
                    self, f"{k.upper()} — prerequisites missing",
                    "Before this analysis can run, please compute first:\n\n• "
                    + "\n• ".join(missing)
                )
                return None, None
            if k == "twi":
                return twi, dict(flow_accum=fa, slope_rad=slp, cell_size=cs, **params)
            else:
                return spi, dict(flow_accum=fa, slope_rad=slp, cell_size=cs, **params)

        if k == "tpi":
            return tpi, dict(dem=data, **params)

        if k == "tri":
            return tri, dict(dem=data, nodata=nd)

        if k == "roughness":
            method = params.pop("method", "VRM")
            window = params.pop("window", 3)
            def _rough(dem, cell_size, method, window, _progress=None):
                if method == "VRM":
                    return vrm(dem, cell_size, window)
                elif method == "Range (max-min)":
                    return roughness_range(dem, window)
                elif method == "Std-Dev":
                    return roughness_sd(dem, window)
                else:
                    return surface_area_ratio(dem, cell_size)
            return _rough, dict(dem=data, cell_size=cs, method=method, window=window)

        # ── Visibility ────────────────────────────────────────────────────
        if k == "svf":
            out_type = params.pop("output_type", "SVF")
            if out_type == "SVF":
                return sky_view_factor, dict(dem=data, cell_size=cs, **params)
            elif out_type == "Positive Openness":
                def _po(dem, cell_size, n_directions, max_radius, _progress=None):
                    return positive_openness(dem, cell_size, n_directions,
                                             max_radius, _progress=_progress)
                return _po, dict(dem=data, cell_size=cs, **params)
            else:
                def _no(dem, cell_size, n_directions, max_radius, _progress=None):
                    return negative_openness(dem, cell_size, n_directions,
                                             max_radius, _progress=_progress)
                return _no, dict(dem=data, cell_size=cs, **params)

        if k == "viewshed":
            if self._viewshed_latlon is None:
                QMessageBox.information(
                    self, "Viewshed — observer required",
                    "Click 'Pick Observer Point', then click on the map to "
                    "place the observer, then press Compute."
                )
                return None, None
            lat, lon = self._viewshed_latlon
            r, c = self._latlon_to_rowcol(lat, lon, dem)
            if r is None:
                QMessageBox.information(
                    self, "Viewshed — observer outside DEM",
                    "The observer point you picked is outside the DEM extent. "
                    "Pick a point inside the loaded raster."
                )
                return None, None
            from app.core.visibility.viewshed import viewshed as vw
            def _vs(dem, cell_size, observer_row, observer_col,
                    observer_height, target_height, max_radius, correct_curvature,
                    _progress=None):
                return vw(dem, cell_size, observer_row, observer_col,
                          observer_height, target_height, max_radius,
                          correct_curvature, _progress=_progress).astype(np.float32)
            return _vs, dict(dem=data, cell_size=cs,
                             observer_row=r, observer_col=c, **params)

        if k == "profile":
            if len(self._profile_latlons) < 2:
                QMessageBox.information(
                    self, "Profile — line required",
                    "Click 'Draw Profile Line', then click on the map to add "
                    "at least 2 vertices and double-click to finish before "
                    "pressing Compute."
                )
                return None, None
            self._run_profile(dem)
            return None, None

        QMessageBox.warning(self, "Unknown analysis", f"Unknown analysis: {product}")
        return None, None

    def _on_worker_result(self, result: np.ndarray):
        worker = self.sender()
        if worker is None or getattr(worker, "cancelled", False):
            return
        product = worker.product
        dem = worker.dem
        params = worker.params
        if dem is None or result is None:
            return

        # Cache intermediate results for dependent analyses, keyed by the
        # DEM's stable uid so the caches survive a layer rename.
        if product == "slope":
            units = params.get("units", "degrees")
            if units == "radians":
                self._slope_rad_cache[dem.uid] = result.copy()
            elif units == "degrees":
                self._slope_rad_cache[dem.uid] = np.radians(result).astype(np.float32)
            else:  # percent
                self._slope_rad_cache[dem.uid] = np.arctan(result / 100.0).astype(np.float32)
        elif product == "fill_sinks":
            self._filled_cache[dem.uid] = result
        elif product == "flow_direction":
            if np.issubdtype(result.dtype, np.integer):
                self._flow_dir_cache[dem.uid] = result.astype(np.int32)
            else:
                # D-infinity: store float32 angles separately
                self._flow_angle_cache[dem.uid] = result.astype(np.float32)
        elif product == "flow_accumulation":
            self._flow_accum_cache[dem.uid] = result

        # Stream networks are a binary mask — render 0 as transparent so the
        # basemap shows through everywhere there isn't a channel.
        layer_nodata = 0 if product == "streams" else dem.nodata
        layer = DemLayer(
            name=f"{dem.name} · {product}",
            product=product,
            data=result,
            nodata=layer_nodata,
            bounds=dem.bounds,
            crs_wkt=dem.crs_wkt,
            cell_size_m=dem.cell_size_m,
            parent_name=dem.name,
            colormap=COLORMAPS.get(product, "terrain"),
            opacity=DEFAULT_OPACITY,
        )
        lo, hi = layer.auto_range()
        layer.render_min = lo
        layer.render_max = hi

        self._mgr.add(layer)               # may de-duplicate layer.name
        self._map.add_layer(layer)
        self._reapply_layer_order()
        self._mgr.select(layer.name)
        self._status(f"✓ {product} computed for {dem.name}")

        # The new derived layer enters the manager's stack via _mgr.add(), so
        # the LayerManager.layers_changed signal already triggered a drape
        # rebuild on the 3D window.  Nothing extra to do here.

    def _on_worker_error(self, msg: str):
        QMessageBox.critical(self, "Analysis Error", msg)
        self._status("Analysis failed.")

    def _on_worker_finished(self):
        self._analysis_panel.set_running(False)
        if self.sender() is self._worker:
            self._worker = None

    # ── Interactive tools ──────────────────────────────────────────────────

    def _activate_tool(self, tool: str):
        if tool == "profile":
            self._profile_latlons.clear()
            self._map.start_profile_draw()
            self._status("Click on map to add profile vertices. Double-click to finish.")
        elif tool == "viewshed":
            self._map.start_viewshed_draw()
            self._status("Click on map to place observer point.")

    def _on_profile_point(self, lat: float, lon: float):
        self._profile_latlons.append((lat, lon))

    def _on_profile_complete(self):
        self._map.stop_profile_draw()
        if len(self._profile_latlons) < 2:
            self._status("Profile needs at least 2 points.")
            return
        self._run_profile(self._mgr.active_dem)
        self._profile_latlons.clear()

    def _run_profile(self, dem: Optional[DemLayer]):
        if not dem or dem.data is None or len(self._profile_latlons) < 2:
            return
        from app.core.visibility.profile import sample_profile
        row_cols = [self._latlon_to_rowcol(lat, lon, dem, clip=True)
                    for lat, lon in self._profile_latlons]
        row_cols = [(r, c) for r, c in row_cols if r is not None]
        if len(row_cols) < 2:
            return
        profile = sample_profile(dem.valid_data, dem.cell_size_m, row_cols)
        dlg = ProfileDialog(profile, self)
        dlg.show()

    def _on_viewshed_point(self, lat: float, lon: float):
        self._viewshed_latlon = (lat, lon)
        self._status(f"Observer set at Lat:{lat:.4f} Lon:{lon:.4f} — click Compute.")

    # ── Coordinate utilities ───────────────────────────────────────────────

    def _latlon_to_rowcol(
        self, lat: float, lon: float, layer: DemLayer, clip: bool = False
    ) -> tuple[Optional[int], Optional[int]]:
        b = layer.bounds
        if b is None:
            return None, None
        if not (b.south <= lat <= b.north and b.west <= lon <= b.east):
            if clip:
                lat = np.clip(lat, b.south, b.north)
                lon = np.clip(lon, b.west, b.east)
            else:
                return None, None
        rows, cols = layer.shape
        col = int((lon - b.west) / (b.east - b.west) * cols)
        row = int((b.north - lat) / (b.north - b.south) * rows)
        col = np.clip(col, 0, cols - 1)
        row = np.clip(row, 0, rows - 1)
        return row, col

    def _on_mouse_moved(self, lat: float, lon: float):
        self._sb_coord.setText(f"Lat: {lat:.4f}°  Lon: {lon:.4f}°")
        dem = self._mgr.active_dem
        if dem and dem.data is not None:
            r, c = self._latlon_to_rowcol(lat, lon, dem)
            if r is not None:
                # Read through valid_data so nodata cells show "—" instead of
                # the raw sentinel (-9999, etc.).
                val = dem.valid_data[r, c]
                if np.isfinite(val):
                    self._sb_elev.setText(f"Elev: {float(val):,.1f} m")
                else:
                    self._sb_elev.setText("Elev: —")

    # ── View / base layer ──────────────────────────────────────────────────

    def _set_base(self, name: str):
        self._act_base_osm.setChecked(name == "osm")
        self._act_base_sat.setChecked(name == "satellite")
        self._act_base_topo.setChecked(name == "topo")
        self._act_base_none.setChecked(name == "none")
        self._map.set_base_layer(name)

    def _toggle_3d(self, checked: bool):
        if checked:
            if self._3d_window is None:
                self._3d_window = View3DWindow(self._mgr, self)
                # Reflect the window close button back into the toolbar toggle
                # so the two stay in sync.
                self._3d_window.closed.connect(
                    lambda: self._act_3d.setChecked(False)
                )
                # 2D ↔ 3D sync wiring.
                self._3d_window.frustum_changed.connect(
                    self._map.set_camera_frustum
                )
            self._3d_window.show()
            self._3d_window.raise_()
            self._3d_window.activateWindow()
            dem = self._mgr.active_dem
            if dem is not None:
                self._3d_window.set_active_dem(dem)
            # Push any already-loaded catalog so sticks appear on first open
            # without waiting for a filter/style nudge.
            if self._catalog is not None:
                self._3d_window.set_earthquakes(self._catalog)
        else:
            if self._3d_window is not None:
                self._3d_window.hide()

    def _sync_3d_active_dem(self, dem: Optional[DemLayer]):
        """Forward an active-DEM change to the 3D window if it is open.

        ``set_active_dem`` rebuilds the terrain mesh and — since stick XY
        coordinates are terrain-relative — also re-derives the earthquake
        renderable from the catalog reference it already holds.
        """
        if self._3d_window is not None and self._3d_window.isVisible() and dem is not None:
            self._3d_window.set_active_dem(dem)

    def _sync_3d_drape(self):
        """Forward a layer-stack or style change to the 3D window."""
        if self._3d_window is not None and self._3d_window.isVisible():
            self._3d_window.rebuild_drape()

    # ── Earthquake catalog ─────────────────────────────────────────────────

    def _open_earthquake_catalog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Earthquake Catalog",
            "", "Excel / CSV (*.xlsx *.xls *.xlsm *.csv);;All files (*)"
        )
        if not path:
            return
        try:
            cat = EarthquakeCatalog.from_file(path)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load catalog", str(exc))
            return
        self._adopt_catalog(cat)

    def _autoload_catalog_if_present(self):
        """Auto-load ``catalog.xlsx`` from the CWD on start-up if present.

        Saves the user a click when the project's reference catalog sits next
        to ``main.py`` (as it does for this project's demo data).
        """
        candidate = Path("catalog.xlsx")
        if not candidate.exists():
            return
        try:
            cat = EarthquakeCatalog.from_file(candidate)
        except Exception as exc:
            self._status(f"Auto-load of catalog.xlsx failed: {exc}")
            return
        self._adopt_catalog(cat)

    def _adopt_catalog(self, cat: EarthquakeCatalog):
        """Wire a freshly-loaded catalog into both views and the panels."""
        self._catalog = cat
        self._layer_panel.set_catalog(cat)
        self._props_panel.set_catalog(cat)
        self._push_earthquakes_to_views(full_resend=True)
        b = cat.bounds()
        if b is not None:
            south, west, north, east = b
            self._map.fit_bounds(GeoBounds(west=west, south=south,
                                           east=east, north=north))
        self._status(
            f"Loaded {len(cat)} earthquakes from "
            f"{cat.source_path.name if cat.source_path else 'memory'}"
        )

    def _clear_earthquake_catalog(self):
        self._catalog = None
        self._layer_panel.set_catalog(None)
        self._props_panel.set_catalog(None)
        self._map.clear_earthquakes()
        if self._3d_window is not None:
            self._3d_window.set_earthquakes(None)
        self._status("Earthquake catalog cleared.")

    def _zoom_to_earthquakes(self):
        if self._catalog is None:
            return
        b = self._catalog.bounds()
        if b is None:
            return
        south, west, north, east = b
        self._map.fit_bounds(GeoBounds(west=west, south=south,
                                       east=east, north=north))

    def _on_earthquake_selected(self):
        """Layer-panel earthquake row clicked → show its properties page."""
        self._props_panel.show_earthquakes()

    def _on_earthquake_visibility(self, visible: bool):
        if self._catalog is None:
            return
        self._catalog.style.visible = bool(visible)
        self._push_earthquakes_to_views(full_resend=False)

    def _on_earthquake_style_changed(self):
        """Properties panel fired a style/filter change — re-push to the 2D map.

        ``full_resend`` is True because filter changes alter which subset of
        events is visible.
        """
        if self._catalog is None:
            return
        self._layer_panel.refresh_catalog_counts()
        self._props_panel.refresh_earthquake_widgets()
        self._push_earthquakes_to_views(full_resend=True)

    def _earthquake_style_payload(self) -> dict:
        """Style dict shipped to map.html — matches the JS schema."""
        s = self._catalog.style
        return {
            "color_by":   s.color_by,
            "colormap":   s.colormap,
            "size_scale": s.size_scale,
            "visible":    s.visible,
            "color_min":  s.color_min,
            "color_max":  s.color_max,
        }

    def _push_earthquakes_to_views(self, full_resend: bool):
        """Apply the current catalog+style to the 2D map and the 3D view.

        ``full_resend=True`` means the event set (or its filters) changed,
        so we re-ship the whole array to Leaflet.  ``False`` means only the
        styling changed, so the 2D side does an in-place restyle.  The 3D
        view always rebuilds from the catalog because depth-to-z, surface
        anchoring, and width all derive from the per-event arrays.
        """
        if self._catalog is None:
            self._map.clear_earthquakes()
            if self._3d_window is not None:
                self._3d_window.set_earthquakes(None)
            return

        style = self._earthquake_style_payload()
        if full_resend:
            visible = self._catalog.visible_indices().tolist()
            events = [self._catalog.events[i].to_json(i) for i in visible]
            self._map.set_earthquakes(events, style)
        else:
            self._map.set_earthquake_style(style)

        if self._3d_window is not None and self._3d_window.isVisible():
            self._3d_window.set_earthquakes(self._catalog)

    def _on_earthquake_clicked(self, idx: int):
        """2D map → re-emphasise this event (and could later flash in 3D)."""
        if self._catalog is None or idx < 0 or idx >= len(self._catalog):
            return
        ev = self._catalog.events[idx]
        self._status(
            f"Earthquake #{idx}: M{ev.magnitude:.2f}, depth {ev.depth_km:.2f} km, "
            f"{ev.origin.isoformat() if ev.origin else '—'}"
        )

    # ── Cross-section tool ─────────────────────────────────────────────────

    def _start_section_draw(self):
        if self._catalog is None or len(self._catalog) == 0:
            QMessageBox.information(
                self, "No earthquake catalog",
                "Open an earthquake catalog first "
                "(File → Open Earthquake Catalog…)."
            )
            return
        self._section_latlons.clear()
        self._map.start_section_draw()
        self._status(
            "Click on map to add cross-section vertices. "
            "Double-click to finish."
        )

    def _on_section_point(self, lat: float, lon: float):
        self._section_latlons.append((lat, lon))

    def _on_section_complete(self):
        self._map.stop_section_draw()
        pts = list(self._section_latlons)
        self._section_latlons.clear()
        if len(pts) < 2 or self._catalog is None:
            self._status("Cross-section needs at least 2 points.")
            return
        # Use the active DEM (if any) for surface elevation along the section.
        dem = self._mgr.active_dem
        dlg = EarthquakeSectionDialog(
            catalog=self._catalog, line_latlons=pts, dem=dem, parent=self
        )
        dlg.show()

    # ── Status helper ─────────────────────────────────────────────────────

    def _status(self, msg: str):
        self._sb_msg.setText(msg)
        self.statusBar().showMessage(msg, 8000)

    # ── Shutdown ───────────────────────────────────────────────────────────

    def closeEvent(self, event):
        # Stop a running analysis cleanly so the QThread is not destroyed
        # while still executing.  Cancel only flips a flag; functions that
        # don't poll _progress (slope, aspect, hillshade, tpi, …) can't be
        # interrupted at all, so we may genuinely have to wait.
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._status("Waiting for analysis to finish before closing…")
            if not self._worker.wait(5000):
                resp = QMessageBox.question(
                    self, "Analysis still running",
                    "An analysis is still running and cannot be cancelled "
                    "(it's likely inside a non-interruptible NumPy/SciPy "
                    "call).\n\nClose anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if resp != QMessageBox.StandardButton.Yes:
                    event.ignore()
                    return
                # User chose to close — wait a final beat to give the thread
                # a chance to land; Qt will warn but won't crash.
                self._worker.wait(1000)
        # Close the 3D window if it's open so its GL context is destroyed
        # before the QApplication exits (avoids a Qt warning on shutdown).
        if self._3d_window is not None:
            self._3d_window.close()
        super().closeEvent(event)

    # ── Safe Graphics Mode ─────────────────────────────────────────────────

    def _on_map_unresponsive(self):
        """The map page loaded but Leaflet never initialised.

        Pop a recovery dialog with concrete actions. If safe-gfx isn't on
        yet, offer to enable it and quit so the user can relaunch. If it's
        already on and we're still blank, point at the log file.
        """
        from PyQt6.QtCore import QSettings
        already_safe = self._act_safe_gfx.isChecked()
        if not already_safe:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Map is not rendering")
            box.setText(
                "The map area is blank — Chromium loaded but didn't render. "
                "This is usually a GPU / driver issue on this PC."
            )
            box.setInformativeText(
                "Would you like to enable Safe Graphics Mode (software "
                "rendering) and quit so you can relaunch?\n\n"
                "This costs almost nothing for a Leaflet map and resolves "
                "the blank-canvas problem on locked-down or driver-broken "
                "Windows boxes."
            )
            enable_btn = box.addButton(
                "Enable Safe Mode && Quit", QMessageBox.ButtonRole.AcceptRole
            )
            box.addButton("Dismiss", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() is enable_btn:
                QSettings().setValue("safe_gfx", "true")
                QSettings().sync()
                QApplication.instance().quit()
            return

        # Already in safe-gfx and still blank — that's deeper. Tell the user
        # where to find the log file so they can send it for diagnosis.
        log_hint = self._log_path_hint()
        QMessageBox.critical(
            self, "Map is still not rendering",
            "Safe Graphics Mode is already on, but the map still didn't "
            "initialise. Please send the diagnostic log to support:\n\n"
            f"{log_hint}"
        )

    def _log_path_hint(self) -> str:
        try:
            from app.core.crash_log import log_path
            p = log_path()
            return str(p) if p else "<log file location unavailable>"
        except Exception:
            return "<log file location unavailable>"

    def _open_log_folder(self):
        """Open the diagnostic log directory in the system file browser."""
        from app.core.crash_log import log_path
        p = log_path()
        if not p:
            QMessageBox.information(
                self, "No log file",
                "Diagnostic logging was not initialised this session."
            )
            return
        folder = p.parent
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _toggle_safe_gfx(self, checked: bool):
        """Persist the user's choice and prompt for a restart.

        Reading this setting at startup is handled in ``main.py`` — we only
        write it here, because Chromium flags must be set BEFORE
        QApplication is constructed. We always write an explicit string
        ("true"/"false") so main.py can distinguish "user chose Off" from
        "user never touched the setting" (where bundle default applies).
        """
        from PyQt6.QtCore import QSettings
        QSettings().setValue("safe_gfx", "true" if checked else "false")
        QSettings().sync()
        QMessageBox.information(
            self, "Safe Graphics Mode",
            ("Safe Graphics Mode will be {state} after the next restart."
             "\n\nThis disables Chromium's GPU acceleration in the embedded "
             "browser. Turn it ON if the map area appears blank — common on "
             "Windows PCs with locked-down or out-of-date GPU drivers. "
             "Turn it OFF if you want hardware-accelerated rendering on a "
             "machine where the GPU works correctly.")
            .format(state="ENABLED" if checked else "DISABLED")
        )

    # ── About ──────────────────────────────────────────────────────────────

    def _about(self):
        QMessageBox.about(
            self, "DEM Analyst",
            "<b>DEM Analyst v1.0</b><br/><br/>"
            "Digital Elevation Model analysis toolkit.<br/>"
            "Implements Horn (1981), Zevenbergen & Thorne (1987), "
            "Barnes et al. (2014), Tarboton (1997), Zakšek et al. (2011), "
            "and many others.<br/><br/>"
            "Built with PyQt6, Leaflet, rasterio, NumPy."
        )
