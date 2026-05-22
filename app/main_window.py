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
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QMainWindow, QMessageBox, QFileDialog, QDockWidget,
    QLabel, QProgressBar, QWidget, QToolBar, QStatusBar,
    QApplication,
)

import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

from app.config import COLORMAPS, DEFAULT_OPACITY
from app.core.dem_layer import DemLayer, GeoBounds
from app.core.layer_manager import LayerManager
from app.core.worker import AnalysisWorker
from app.core.renderer import array_to_png_b64

from app.ui.map_canvas import MapCanvas
from app.ui.layer_panel import LayerPanel
from app.ui.analysis_panel import AnalysisPanel
from app.ui.properties_panel import PropertiesPanel
from app.ui.view3d import View3DDock
from app.ui.profile_dialog import ProfileDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DEM Analyst")
        self.resize(1400, 900)

        self._mgr = LayerManager(self)
        self._worker: Optional[AnalysisWorker] = None
        self._profile_latlons: list[tuple[float, float]] = []
        self._viewshed_latlon: Optional[tuple[float, float]] = None

        # Pre-computed intermediate results (keyed by DEM name)
        self._flow_dir_cache: dict = {}      # D8 int32 codes
        self._flow_angle_cache: dict = {}    # D-infinity float32 angles
        self._flow_accum_cache: dict = {}
        self._slope_rad_cache: dict = {}
        self._filled_cache: dict = {}

        self._build_ui()
        self._connect_signals()
        self._apply_stylesheet()

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

        self._3d_dock = View3DDock(self)
        self._3d_dock.setMinimumHeight(200)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._3d_dock)
        self._3d_dock.hide()

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
        tb.addAction(self._act_3d)
        tb.addSeparator()
        tb.addWidget(QLabel(" Base: "))
        for act in (self._act_base_osm, self._act_base_sat, self._act_base_topo):
            tb.addAction(act)

        self._act_open.triggered.connect(self._open_dem)
        self._act_save.triggered.connect(self._export_layer)
        self._act_3d.toggled.connect(self._toggle_3d)
        self._act_base_osm.triggered.connect(lambda: self._set_base("osm"))
        self._act_base_sat.triggered.connect(lambda: self._set_base("satellite"))
        self._act_base_topo.triggered.connect(lambda: self._set_base("topo"))

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
        fm.addAction("Exit", self.close)

        # View
        vm = mbar.addMenu("View")
        vm.addAction(self._act_3d)
        vm.addSeparator()
        vm.addAction("Layer Panel").triggered.connect(self._layer_panel.show)
        vm.addAction("Analysis Panel").triggered.connect(self._analysis_panel.show)
        vm.addAction("Properties Panel").triggered.connect(self._props_panel.show)

        # Analysis — quick launch
        am = mbar.addMenu("Analysis")
        for name, key in [
            ("Hillshade", "hillshade"),
            ("Slope", "slope"),
            ("Aspect", "aspect"),
            ("TWI", "twi"),
            ("TPI", "tpi"),
        ]:
            am.addAction(name).triggered.connect(
                lambda checked, k=key: self._quick_run(k)
            )

        # Help
        hm = mbar.addMenu("Help")
        hm.addAction("About").triggered.connect(self._about)

    def _apply_stylesheet(self):
        qss_path = Path(__file__).parent / "style.qss"
        if qss_path.exists():
            with open(qss_path) as f:
                QApplication.instance().setStyleSheet(f.read())

    # ── Signal wiring ──────────────────────────────────────────────────────

    def _connect_signals(self):
        # Layer panel
        self._layer_panel.remove_requested.connect(self._remove_layer)
        self._layer_panel.zoom_requested.connect(self._zoom_to_layer)
        self._layer_panel.export_requested.connect(self._export_layer)
        self._layer_panel.visibility_toggled.connect(self._map.set_visible)
        self._layer_panel.active_dem_change.connect(self._mgr.set_active_dem)

        # Analysis panel
        self._analysis_panel.run_analysis.connect(self._dispatch_analysis)
        self._analysis_panel.start_tool.connect(self._activate_tool)

        # Properties panel
        self._props_panel.style_changed.connect(self._refresh_layer_on_map)

        # Layer manager
        self._mgr.layer_updated.connect(self._on_layer_updated)

        # Map bridge
        self._map.bridge.mouse_moved.connect(self._on_mouse_moved)
        self._map.bridge.profile_point.connect(self._on_profile_point)
        self._map.bridge.profile_complete.connect(self._on_profile_complete)
        self._map.bridge.viewshed_point.connect(self._on_viewshed_point)

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
            QMessageBox.critical(self, "Error loading DEM", str(e))

    def _load_dem_file(self, path: Path):
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float64)
            nodata = src.nodata
            crs = src.crs
            transform = src.transform
            bounds_native = src.bounds

        # Convert bounds to WGS84
        if crs and crs.to_epsg() != 4326:
            try:
                west, south, east, north = transform_bounds(
                    crs, CRS.from_epsg(4326), *bounds_native
                )
            except Exception:
                west, south, east, north = (
                    bounds_native.left, bounds_native.bottom,
                    bounds_native.right, bounds_native.top
                )
        else:
            west, south, east, north = (
                bounds_native.left, bounds_native.bottom,
                bounds_native.right, bounds_native.top
            )

        # Approximate cell size in metres
        lat_center = (south + north) / 2
        if crs and crs.is_geographic:
            lon_size_m = abs(transform.a) * 111320 * np.cos(np.radians(lat_center))
            lat_size_m = abs(transform.e) * 111320
            cell_size_m = (lon_size_m + lat_size_m) / 2
        else:
            cell_size_m = abs(transform.a)

        layer = DemLayer(
            name=path.stem,
            product="dem",
            data=data.astype(np.float32),
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
        self._map.fit_bounds(layer.bounds)
        self._sb_crs.setText(f"CRS: {crs.to_epsg() if crs else 'unknown'}")
        self._status(f"Loaded {path.name}  ({layer.shape[0]}×{layer.shape[1]})")

        # Auto-run hillshade
        self._quick_run("hillshade")

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

        src_layer = self._mgr.active_dem
        if src_layer:
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

    def _on_layer_updated(self, name: str):
        layer = self._mgr.get(name)
        if layer:
            self._map.set_opacity(name, layer.opacity)
            self._map.set_visible(name, layer.visible)

    # ── Analysis dispatch ──────────────────────────────────────────────────

    def _quick_run(self, product: str):
        self._dispatch_analysis(product, {})

    def _dispatch_analysis(self, product: str, params: dict):
        dem = self._mgr.active_dem
        if dem is None or dem.data is None:
            self._status("No active DEM loaded.")
            return
        if self._worker and self._worker.isRunning():
            self._status("Analysis already running — please wait.")
            return

        self._status(f"Running {product}…")
        self._analysis_panel.set_progress(0)

        data = dem.valid_data.astype(np.float64)
        cs = dem.cell_size_m

        func, extra = self._resolve_analysis(product, params, data, cs, dem)
        if func is None:
            return

        self._worker = AnalysisWorker(func, **extra, progress_callback=True)
        self._worker.progress.connect(self._analysis_panel.set_progress)
        self._worker.result.connect(
            lambda result: self._on_analysis_done(result, product, dem, params)
        )
        self._worker.error.connect(self._on_analysis_error)
        self._worker.start()

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
            name = dem.name
            fd       = self._flow_dir_cache.get(name)
            fa_angle = self._flow_angle_cache.get(name)
            if fd is None and fa_angle is None:
                self._status("Run Flow Direction first.")
                return None, None
            if fa_angle is not None:
                def _accum_dinf(dem, angle, _progress=None):
                    return d_inf_flow_accumulation(dem, angle)
                return _accum_dinf, dict(dem=data, angle=fa_angle)
            else:
                def _accum(dem, flow_dir, nodata, _progress=None):
                    return d8_flow_accumulation(dem, flow_dir, nodata=nodata)
                return _accum, dict(dem=data, flow_dir=fd, nodata=nd)

        if k == "streams":
            name = dem.name
            fa   = self._flow_accum_cache.get(name)
            if fa is None:
                self._status("Run Flow Accumulation first.")
                return None, None
            auto_thr = params.pop("auto_threshold", True)
            user_thr = params.pop("threshold", 1000.0)
            def _streams(dem, fa, auto_thr, user_thr, _progress=None):
                thr = auto_threshold(fa) if auto_thr else user_thr
                return extract_streams(fa, thr).astype(np.float32)
            return _streams, dict(dem=data, fa=fa, auto_thr=auto_thr, user_thr=user_thr)

        # ── Indices ────────────────────────────────────────────────────────
        if k in ("twi", "spi"):
            name = dem.name
            fa   = self._flow_accum_cache.get(name)
            slp  = self._slope_rad_cache.get(name)
            if fa is None or slp is None:
                self._status("Run Slope + Flow Accumulation first.")
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
                    return positive_openness(dem, cell_size, n_directions, max_radius)
                return _po, dict(dem=data, cell_size=cs, **params)
            else:
                def _no(dem, cell_size, n_directions, max_radius, _progress=None):
                    return negative_openness(dem, cell_size, n_directions, max_radius)
                return _no, dict(dem=data, cell_size=cs, **params)

        if k == "viewshed":
            if self._viewshed_latlon is None:
                self._status("Pick an observer point on the map first.")
                return None, None
            lat, lon = self._viewshed_latlon
            r, c = self._latlon_to_rowcol(lat, lon, dem)
            if r is None:
                self._status("Observer point is outside DEM extent.")
                return None, None
            from app.core.visibility.viewshed import viewshed as vw
            def _vs(dem, cell_size, observer_row, observer_col,
                    observer_height, target_height, max_radius, correct_curvature,
                    _progress=None):
                return vw(dem, cell_size, observer_row, observer_col,
                          observer_height, target_height, max_radius, correct_curvature).astype(np.float32)
            return _vs, dict(dem=data, cell_size=cs,
                             observer_row=r, observer_col=c, **params)

        if k == "profile":
            self._run_profile(dem)
            return None, None

        self._status(f"Unknown analysis: {product}")
        return None, None

    def _on_analysis_done(self, result: np.ndarray, product: str, dem: DemLayer, params: dict):
        self._analysis_panel.reset_progress()

        # Cache intermediate results for dependent analyses
        if product == "slope":
            units = params.get("units", "degrees")
            if units == "radians":
                self._slope_rad_cache[dem.name] = result.copy()
            elif units == "degrees":
                self._slope_rad_cache[dem.name] = np.radians(result).astype(np.float32)
            else:  # percent
                self._slope_rad_cache[dem.name] = np.arctan(result / 100.0).astype(np.float32)
        if product == "fill_sinks":
            self._filled_cache[dem.name] = result
        if product == "flow_direction":
            if np.issubdtype(result.dtype, np.integer):
                self._flow_dir_cache[dem.name] = result.astype(np.int32)
            else:
                # D-infinity: store float32 angles separately
                self._flow_angle_cache[dem.name] = result.astype(np.float32)
        if product == "flow_accumulation":
            self._flow_accum_cache[dem.name] = result

        name = f"{dem.name} · {product}"
        layer = DemLayer(
            name=name,
            product=product,
            data=result,
            nodata=dem.nodata,
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

        self._mgr.add(layer)
        self._map.add_layer(layer)
        self._mgr.select(name)
        self._status(f"✓ {product} computed for {dem.name}")

        # Update 3D if visible
        if self._act_3d.isChecked():
            self._update_3d(dem)

    def _on_analysis_error(self, msg: str):
        self._analysis_panel.reset_progress()
        QMessageBox.critical(self, "Analysis Error", msg)
        self._status("Analysis failed.")

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
                val = dem.data[r, c]
                self._sb_elev.setText(f"Elev: {val:,.1f} m")

    # ── View / base layer ──────────────────────────────────────────────────

    def _set_base(self, name: str):
        self._act_base_osm.setChecked(name == "osm")
        self._act_base_sat.setChecked(name == "satellite")
        self._act_base_topo.setChecked(name == "topo")
        self._map.set_base_layer(name)

    def _toggle_3d(self, checked: bool):
        if checked:
            self._3d_dock.show()
            dem = self._mgr.active_dem
            if dem:
                self._update_3d(dem)
        else:
            self._3d_dock.hide()

    def _update_3d(self, dem: DemLayer):
        hs_layer = next(
            (l for l in self._mgr.all()
             if l.parent_name == dem.name and l.product in ("hillshade", "multidirectional")),
            None
        )
        texture = None
        if hs_layer is not None:
            from app.core.renderer import array_to_rgba
            texture = array_to_rgba(hs_layer.data, "Greys_r")
        self._3d_dock.load_dem(dem.valid_data, dem.cell_size_m, texture=texture)

    # ── Status helper ─────────────────────────────────────────────────────

    def _status(self, msg: str):
        self._sb_msg.setText(msg)
        self.statusBar().showMessage(msg, 8000)

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
