"""
View3DWindow — the top-level 3D scene window (QGIS-style).

The window is a sibling to MainWindow rather than a dock so it can be moved to
a second monitor and given its own toolbar / menus.  It is created lazily by
MainWindow the first time the user toggles the 3D action.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QSize, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence, QImage, QPainter
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QToolBar, QStatusBar, QLabel, QFileDialog,
    QMessageBox, QDockWidget, QHBoxLayout, QVBoxLayout, QPushButton,
    QSlider, QComboBox, QMenu, QToolButton,
)

from app.core.dem_layer import DemLayer
from app.core.earthquakes import EarthquakeCatalog
from app.core.layer_manager import LayerManager
from app.core.scene3d import (
    Scene3D, TerrainGrid, EarthquakeRenderable, EarthquakeRenderStyle,
    CoordinateGrid,
)

from app.ui.view3d.gl_widget import (
    GLViewport, has_opengl, opengl_error,
)
from app.ui.view3d.overlays import CompassRose, AxesTriad, GridLabelsOverlay
from app.ui.view3d.side_panel import SidePanel
from app.ui.view3d.drape_worker import DrapeWorker, DrapeRequest
from app.ui.view3d.settings_dialog import SceneSettingsDialog


class View3DWindow(QMainWindow):
    """Top-level window hosting the GL viewport, toolbar, overlays and status."""

    closed = pyqtSignal()                  # MainWindow uses this to un-check its action
    # Emitted whenever the camera moves.  ``corners`` is a list of (lat, lon)
    # tuples around the ground projection of the view frustum; ``camera_latlon``
    # is the camera's own ground projection.  Either may be None when the
    # camera is looking up at the sky (no plane intersection).
    frustum_changed = pyqtSignal(object, object)

    # How long to wait after the last drape-input change before recomputing.
    # Coalesces a flurry of edits (e.g. dragging the layer-stack reorder) into
    # one rebuild.
    DRAPE_DEBOUNCE_MS = 120

    def __init__(self, mgr: LayerManager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D View")
        self.resize(1240, 760)

        self._mgr = mgr
        self._active_dem_name: Optional[str] = None
        self._scene = Scene3D()
        # Tracks which (signX, signY) octant the camera is in so we can
        # rebuild the grid only when the user orbits to a new quadrant —
        # the back walls and silhouette edges flip across the midplanes,
        # rebuilding any more often is wasted work.
        self._last_grid_quadrant: Optional[tuple[bool, bool]] = None

        # Earthquake catalog reference (held so we can rebuild the stick mesh
        # when the terrain changes — XY coords are terrain-relative).  The
        # mutable 3D-only style (width / opacity) lives here too; the catalog's
        # own style still drives filters, colour mapping, and 2D visibility.
        self._catalog: Optional[EarthquakeCatalog] = None
        self._eq_style = EarthquakeRenderStyle()

        # If PyOpenGL isn't installed, show a clear "how to fix" message
        # instead of an empty widget.  We still build the rest of the window
        # so the user sees what the feature would look like, but no rendering
        # happens.
        if not has_opengl():
            self._viewport = None
            self.setCentralWidget(self._make_missing_gl_placeholder())
        else:
            self._viewport = GLViewport(self._scene, self)
            self.setCentralWidget(self._viewport)

        if self._viewport is not None:
            # Overlays — children of the GL widget so they paint on top.
            # The grid labels overlay must be created BEFORE compass/axes so
            # the latter are raised above it (small foreground widgets > the
            # full-viewport label layer they sit on top of).
            self._grid_labels_overlay = GridLabelsOverlay(
                self._scene, self._viewport, parent=self._viewport,
            )
            self._compass = CompassRose(
                get_azimuth_rad=lambda: self._scene.camera.azimuth,
                parent=self._viewport,
            )
            self._axes = AxesTriad(
                get_view_basis=self._axes_basis,
                parent=self._viewport,
            )
        else:
            self._grid_labels_overlay = None
            self._compass = None
            self._axes = None

        # Side panel as a dock so the user can hide it via the toolbar.
        self._side_dock = QDockWidget("Settings", self)
        self._side_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self._side_panel = SidePanel(mgr, self._scene, self._side_dock)
        self._side_dock.setWidget(self._side_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._side_dock)

        # Drape worker state
        self._drape_worker: Optional[DrapeWorker] = None
        self._drape_debounce = QTimer(self)
        self._drape_debounce.setSingleShot(True)
        self._drape_debounce.setInterval(self.DRAPE_DEBOUNCE_MS)
        self._drape_debounce.timeout.connect(self._run_drape_worker)

        self._build_toolbar()
        self._build_statusbar()
        self._wire()

        self._reposition_overlays()
        self._update_camera_readout()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_active_dem(self, dem: Optional[DemLayer]):
        """Rebuild terrain mesh for the given DEM and recompose the drape."""
        if dem is None or dem.data is None:
            self._active_dem_name = None
            self._scene.terrain = None
            self._scene.drape = None
            self._scene.set_earthquakes(None)
            self._scene.set_grid(None)
            self._scene.dirty.mark_all()
            self._refresh()
            self.setWindowTitle("3D View")
            return

        # If the DEM hasn't changed, just refresh — avoids resetting the
        # camera every time the user re-picks the same layer.
        if dem.name == self._active_dem_name and self._scene.terrain is not None:
            self._schedule_drape_rebuild()
            return

        self._active_dem_name = dem.name
        self.setWindowTitle(f"3D View — {dem.name}")

        terrain = TerrainGrid.build(
            dem,
            max_side=self._scene.settings.mesh_max_side,
            elev=dem.valid_data,
        )
        self._scene.set_terrain(terrain)
        self._schedule_drape_rebuild()
        # Stick XY positions depend on the terrain's centre lat/lon, so
        # rebuild the catalog's renderable against the new mesh.
        self._rebuild_earthquakes()
        self._update_camera_readout()

    def rebuild_drape(self):
        """Schedule a debounced drape rebuild."""
        self._schedule_drape_rebuild()

    def set_earthquakes(self, catalog: Optional[EarthquakeCatalog]):
        """Adopt a catalog (or clear it) and refresh the 3D stick set."""
        self._catalog = catalog
        self._rebuild_earthquakes()

    def _rebuild_earthquakes(self):
        """Recompute per-instance arrays for the current catalog + terrain."""
        if self._catalog is None or self._scene.terrain is None:
            self._scene.set_earthquakes(None)
        else:
            renderable = EarthquakeRenderable.build(
                self._catalog, self._scene.terrain, self._eq_style,
            )
            self._scene.set_earthquakes(renderable)
        # Grid extent depends on the deepest event so it tracks the catalog
        # (and still rebuilds when the catalog clears so the box reverts to
        # the default 30 km depth).
        self._rebuild_grid()
        self._refresh()

    def _rebuild_grid(self):
        """Rebuild the 3D coordinate grid against the current terrain extent.

        Depth axis spans from the terrain's z_min down to the deepest event
        in the catalog (or 30 km when there is no catalog) so the box
        always encloses what the user is looking at.  The build is also
        passed the current camera eye so gridlines land on the back walls
        and labels anchor on the silhouette edges.
        """
        terrain = self._scene.terrain
        if terrain is None:
            self._scene.set_grid(None)
            return
        depth_min_m: float = float(terrain.z_min) - 30_000.0
        if self._catalog is not None and len(self._catalog.events) > 0:
            deepest_km = float(self._catalog.depths_km.max())
            # The deepest point of any event (anchor_to_surface OR absolute):
            # take the lower envelope of the two so the box always fits both.
            deepest_z_anchor = float(terrain.z_min) - deepest_km * 1000.0
            deepest_z_absolute = -deepest_km * 1000.0
            depth_min_m = min(depth_min_m, deepest_z_anchor, deepest_z_absolute)
        cam_eye = self._scene.camera.eye() if self._scene.camera else None
        grid = CoordinateGrid.build(
            terrain, camera_eye=cam_eye,
            depth_min_m=depth_min_m, depth_max_m=float(terrain.z_max),
        )
        self._scene.set_grid(grid)
        # Remember the camera quadrant so the camera_changed slot only
        # triggers another rebuild when the user actually crosses an axis.
        if cam_eye is not None:
            self._last_grid_quadrant = (
                bool(float(cam_eye[0]) >= 0.0),
                bool(float(cam_eye[1]) >= 0.0),
            )
        if self._grid_labels_overlay is not None:
            self._grid_labels_overlay.update()

    def _on_camera_changed_for_grid(self):
        """Detect orbit quadrant changes and rebuild the grid when needed.

        Cheap: just a couple of comparisons; the actual CoordinateGrid.build
        only runs when the user crosses x=0 or y=0 in eye coordinates, which
        happens at most a handful of times per session.
        """
        if (self._scene.terrain is None
                or not self._scene.settings.show_grid):
            return
        cam_eye = self._scene.camera.eye()
        new_q = (
            bool(float(cam_eye[0]) >= 0.0),
            bool(float(cam_eye[1]) >= 0.0),
        )
        if new_q != self._last_grid_quadrant:
            self._rebuild_grid()

    @property
    def viewport(self) -> Optional[GLViewport]:
        return self._viewport

    @property
    def scene(self) -> Scene3D:
        return self._scene

    def _refresh(self):
        """Repaint the GL viewport if one exists; no-op in the missing-GL path."""
        if self._viewport is not None:
            self._viewport.update()

    # ── Toolbar / status ──────────────────────────────────────────────────────

    def _build_toolbar(self):
        tb = QToolBar("View3D", self)
        tb.setMovable(False)
        tb.setIconSize(QSize(18, 18))
        self.addToolBar(tb)

        # Camera mode picker — orbit (default) / walk.  Free camera is the
        # same as walk for now; we can split them later if there is demand.
        tb.addWidget(QLabel(" Camera: "))
        self._cam_mode = QComboBox()
        self._cam_mode.addItems(["Orbit", "Walk"])
        self._cam_mode.setMaximumWidth(110)
        tb.addWidget(self._cam_mode)

        self._act_reset = QAction("Reset", self, shortcut=QKeySequence("Home"))
        self._act_top   = QAction("Top",   self, shortcut=QKeySequence("T"))
        self._act_north = QAction("North", self, shortcut=QKeySequence("N"))
        self._act_settings = QAction("Configure…", self)
        self._act_panel = QAction("Settings panel", self, checkable=True)
        self._act_panel.setChecked(True)

        tb.addAction(self._act_reset)
        tb.addAction(self._act_top)
        tb.addAction(self._act_north)
        tb.addAction(self._act_settings)
        tb.addSeparator()

        # Quick Z-factor slider on the toolbar (the side panel has the full
        # control; this is just a shortcut you don't have to open the dock for).
        tb.addWidget(QLabel(" Z× "))
        self._tb_z = QSlider(Qt.Orientation.Horizontal)
        self._tb_z.setRange(1, 50)                   # 0.1× – 5×
        self._tb_z.setMaximumWidth(120)
        self._tb_z.setValue(int(self._scene.settings.z_factor * 10))
        tb.addWidget(self._tb_z)

        tb.addSeparator()

        # Snapshot button with a small dropdown for supersample factor.
        self._btn_snap = QToolButton(self)
        self._btn_snap.setText("Snapshot")
        self._btn_snap.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        snap_menu = QMenu(self._btn_snap)
        self._snap_action_1x = QAction("Save at viewport size (1×)", self, checkable=True)
        self._snap_action_2x = QAction("Save at 2× viewport size",    self, checkable=True)
        self._snap_action_4x = QAction("Save at 4× viewport size",    self, checkable=True)
        self._snap_action_2x.setChecked(True)        # default
        for a in (self._snap_action_1x, self._snap_action_2x, self._snap_action_4x):
            snap_menu.addAction(a)
            a.triggered.connect(lambda checked, act=a: self._set_snap_factor(act))
        self._btn_snap.setMenu(snap_menu)
        self._btn_snap.setDefaultAction(QAction("Snapshot…", self))
        self._btn_snap.defaultAction().triggered.connect(self._on_snapshot)
        tb.addWidget(self._btn_snap)

        tb.addSeparator()
        tb.addAction(self._act_panel)

        # Wire toolbar actions
        self._act_reset.triggered.connect(self._on_reset)
        self._act_top.triggered.connect(self._on_top)
        self._act_north.triggered.connect(self._on_north)
        self._act_settings.triggered.connect(self._on_open_settings)
        self._act_panel.toggled.connect(self._side_dock.setVisible)
        self._side_dock.visibilityChanged.connect(self._act_panel.setChecked)
        self._tb_z.valueChanged.connect(self._on_tb_z_changed)
        self._cam_mode.currentTextChanged.connect(self._on_camera_mode)

    def _set_snap_factor(self, picked: QAction):
        for a in (self._snap_action_1x, self._snap_action_2x, self._snap_action_4x):
            a.setChecked(a is picked)

    def _snap_factor(self) -> int:
        if self._snap_action_4x.isChecked(): return 4
        if self._snap_action_2x.isChecked(): return 2
        return 1

    def _build_statusbar(self):
        sb: QStatusBar = self.statusBar()
        self._sb_cam = QLabel("Camera —")
        self._sb_cursor = QLabel("Cursor —")
        self._sb_status = QLabel("")
        sb.addWidget(self._sb_status, 1)
        sb.addPermanentWidget(self._sb_cursor)
        sb.addPermanentWidget(self._sb_cam)

    def _wire(self):
        if self._viewport is not None:
            self._viewport.camera_changed.connect(self._update_camera_readout)
            self._viewport.camera_changed.connect(self._emit_frustum)
            self._viewport.cursor_over_terrain.connect(self._on_cursor_over_terrain)
            self._viewport.cursor_off_terrain.connect(
                lambda: self._sb_cursor.setText("Cursor —")
            )
            self._viewport.event_hovered.connect(self._on_event_hovered)
            self._viewport.event_unhovered.connect(self._on_event_unhovered)
            # The grid label overlay is a sibling of the GL surface (not
            # painted by paintGL), so it doesn't refresh on its own when the
            # camera moves — we have to update() it explicitly.  Use the
            # immediate (un-throttled) camera signal so labels move in
            # lockstep with the GL framebuffer instead of trailing by a tick.
            if self._grid_labels_overlay is not None:
                self._viewport.camera_changed_immediate.connect(
                    self._grid_labels_overlay.update
                )
            # Watch for camera-quadrant crossings to rebuild the grid
            # geometry (back walls and silhouette label edges flip with
            # the camera).
            self._viewport.camera_changed.connect(
                self._on_camera_changed_for_grid
            )
        # Side panel ↔ scene
        self._side_panel.drape_layers_changed.connect(self._on_drape_selection)
        self._side_panel.z_factor_changed.connect(self._on_z_factor)
        self._side_panel.sun_changed.connect(self._on_sun_changed)
        self._side_panel.shading_toggled.connect(self._on_shading_toggled)
        self._side_panel.wireframe_toggled.connect(self._on_wireframe)
        self._side_panel.mesh_quality_changed.connect(self._on_mesh_quality)
        self._side_panel.earthquake_style_changed.connect(
            self._on_earthquake_style_changed
        )
        self._side_panel.grid_visibility_changed.connect(
            self._on_grid_visibility_changed
        )

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_reset(self):
        self._scene.frame_terrain()
        self._refresh()
        self._update_camera_readout()

    def _on_top(self):
        self._scene.top_view()
        self._refresh()
        self._update_camera_readout()

    def _on_north(self):
        self._scene.north_up()
        self._refresh()
        self._update_camera_readout()

    def _on_tb_z_changed(self, raw: int):
        z = raw / 10.0
        self._scene.set_z_factor(z)
        self._refresh()
        self._update_camera_readout()

    def _on_z_factor(self, z: float):
        self._scene.set_z_factor(z)
        # Push the toolbar slider in sync if the side panel changed it.
        self._tb_z.blockSignals(True)
        self._tb_z.setValue(int(round(z * 10)))
        self._tb_z.blockSignals(False)
        self._refresh()
        self._update_camera_readout()

    def _on_sun_changed(self, az: float, alt: float, ambient: float):
        self._scene.set_sun(azimuth_deg=az, altitude_deg=alt, ambient=ambient)
        self._refresh()

    def _on_shading_toggled(self, on: bool):
        self._scene.set_sun(enabled=on)
        self._refresh()

    def _on_wireframe(self, wf: bool):
        self._scene.set_wireframe(wf)
        self._refresh()

    def _on_mesh_quality(self, max_side: int):
        self._scene.settings.mesh_max_side = int(max_side)
        # Mesh changed → rebuild the terrain (drape stays at its own resolution).
        if self._active_dem_name:
            dem = self._mgr.get(self._active_dem_name)
            if dem is not None:
                terrain = TerrainGrid.build(
                    dem, max_side=max_side, elev=dem.valid_data
                )
                self._scene.set_terrain(terrain, reframe_camera=False)
                # New terrain → resample surface anchors for the sticks; the
                # rebuild call also re-derives the grid extent.
                self._rebuild_earthquakes()
                self._refresh()

    def _on_drape_selection(self, names: list[str]):
        self._scene.drape_layer_names = list(names)
        self._schedule_drape_rebuild()

    def _on_earthquake_style_changed(self, width_scale: float,
                                     opacity: float, visible: bool,
                                     shape: str, anchor_to_surface: bool):
        """Side panel mutated the stick style — apply and rebuild geometry.

        We rebuild rather than just toggling the visibility flag because
        width / opacity changes alter the per-instance arrays, not just a
        render uniform.  Build cost on a 30k-event catalog is well under
        the slider-drag tick rate, so debouncing isn't worth the complexity.
        Shape changes are essentially free — they just switch which VAO is
        bound — but we still rebuild so the anchor flag can update Z
        positions in the same pass.
        """
        self._eq_style.width_scale = float(width_scale)
        self._eq_style.opacity = float(opacity)
        self._eq_style.visible = bool(visible)
        self._eq_style.shape = str(shape)
        self._eq_style.anchor_to_surface = bool(anchor_to_surface)
        self._rebuild_earthquakes()

    def _on_grid_visibility_changed(self, show: bool, show_labels: bool):
        self._scene.set_show_grid(show, show_labels)
        # Rebuild the grid when first enabled (covers the case where the
        # terrain loaded before the user ticked the box).
        if show and self._scene.grid is None and self._scene.terrain is not None:
            self._rebuild_grid()
        self._refresh()
        if self._grid_labels_overlay is not None:
            self._grid_labels_overlay.update()

    def _on_camera_mode(self, mode: str):
        """Switch navigation paradigm.

        Both modes share the same OrbitCamera; only the mouse-button mapping
        in the viewport changes.  Walk mode swaps left-drag from orbit to pan.
        """
        if self._viewport is None:
            return
        self._viewport.set_camera_mode(mode.lower())
        if mode.lower() == "walk":
            self._sb_status.setText(
                "Walk mode: drag = pan, scroll = zoom, WASD = move, "
                "Q/E = down/up, Shift = sprint (3×)."
            )
        else:
            self._sb_status.setText("")

    def _on_open_settings(self):
        dlg = SceneSettingsDialog(self._scene.settings, self)
        dlg.setting_changed.connect(self._on_settings_changed)
        dlg.exec()

    def _on_settings_changed(self):
        self._scene.dirty.uniforms = True
        self._refresh()

    def _on_snapshot(self):
        factor = self._snap_factor()
        img = self._capture_snapshot(factor)
        if img is None or img.isNull():
            QMessageBox.warning(self, "Snapshot failed",
                                "Could not capture the 3D viewport.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save snapshot",
            f"{self._active_dem_name or '3d_view'}.png",
            "PNG (*.png)"
        )
        if not path:
            return
        img.save(path, "PNG")
        self._sb_status.setText(f"Snapshot saved ({img.width()}×{img.height()}).")

    def _capture_snapshot(self, factor: int) -> Optional[QImage]:
        """Capture the GL framebuffer, optionally at a higher resolution.

        For 2×/4×, render into a temporarily-resized offscreen viewport using
        QOpenGLFramebufferObject.  We do not actually resize the visible
        widget — the camera/MVP doesn't need to change, only the viewport
        resolution.
        """
        if self._viewport is None:
            return None
        if factor <= 1:
            return self._viewport.snapshot()

        # Simplest path: render the widget at 1× and upscale.  QGIS does the
        # same trick when the GPU can't render to a larger FBO.  Bilinear
        # gives the user a clean print resolution without messing with the
        # FBO machinery.
        base = self._viewport.snapshot()
        if base is None or base.isNull():
            return None
        scaled = base.scaled(
            base.width()  * factor,
            base.height() * factor,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        return scaled

    def _on_cursor_over_terrain(self, x, y, z, lat, lon):
        # The mesh Z is shifted so its lowest sample sits at z=0 for cleaner
        # framing; add the offset back so the status bar shows real metres
        # above sea level, matching what every other view of the DEM reports.
        terrain = self._scene.terrain
        true_z = z + (terrain.elevation_shift_m if terrain is not None else 0.0)
        self._sb_cursor.setText(
            f"Lat {lat:.4f}°  Lon {lon:.4f}°  Elev {true_z:,.1f} m"
        )

    def _on_event_hovered(self, catalog_idx: int):
        """Show magnitude + depth (+ origin date when available) of the hit."""
        if self._catalog is None:
            return
        if catalog_idx < 0 or catalog_idx >= len(self._catalog.events):
            return
        e = self._catalog.events[catalog_idx]
        parts = [f"M {e.magnitude:.1f}", f"Depth {e.depth_km:.1f} km"]
        if e.origin is not None:
            parts.append(e.origin.date().isoformat())
        parts.append(f"({e.lat:.3f}°, {e.lon:.3f}°)")
        self._sb_cursor.setText("Event: " + "  ".join(parts))

    def _on_event_unhovered(self):
        # Don't clear unconditionally — the cursor-over-terrain handler will
        # immediately rewrite the label on its own tick.  We only intervene
        # if the current text starts with "Event:" so a stale event readout
        # doesn't linger when the cursor leaves a sphere onto empty sky.
        if self._sb_cursor.text().startswith("Event:"):
            self._sb_cursor.setText("Cursor —")

    def _update_camera_readout(self):
        cam = self._scene.camera
        dist_km = cam.distance / 1000.0
        az_deg = (math.degrees(cam.azimuth)) % 360.0
        el_deg = math.degrees(cam.elevation)
        self._sb_cam.setText(
            f"Cam {dist_km:,.2f} km  Az {az_deg:5.1f}°  Tilt {el_deg:4.1f}°  Z×{self._scene.settings.z_factor:.1f}"
        )

    # ── Drape worker ──────────────────────────────────────────────────────────

    def _schedule_drape_rebuild(self):
        if self._scene.terrain is None:
            return
        # Restart the debounce.  A new (or first) selection cancels any in-flight
        # worker so the latest snapshot wins.
        self._drape_debounce.start()

    def _run_drape_worker(self):
        if self._scene.terrain is None:
            return
        dem = self._mgr.get(self._active_dem_name) if self._active_dem_name else None
        if dem is None or dem.bounds is None:
            return
        # If the side panel hasn't sent a selection yet, default to active DEM.
        names = self._scene.drape_layer_names or [dem.name]
        selected = [self._mgr.get(n) for n in names]
        selected = [l for l in selected if l is not None and l.data is not None]
        if not selected:
            self._scene.set_drape(np.zeros((1, 1, 4), dtype=np.uint8))
            self._refresh()
            return

        # Cancel any in-flight worker.  We don't wait for it — its result will
        # be discarded when it eventually emits.
        if self._drape_worker is not None and self._drape_worker.isRunning():
            self._drape_worker.discard()

        request = DrapeRequest(
            layers=selected,
            dst_bounds=dem.bounds,
            dst_shape=dem.shape,
        )
        worker = DrapeWorker(request, self)
        worker.finished_rgba.connect(self._on_drape_done)
        worker.failed.connect(lambda msg: self._sb_status.setText(f"Drape failed: {msg}"))
        self._drape_worker = worker
        self._sb_status.setText("Compositing drape…")
        worker.start()

    def _on_drape_done(self, rgba):
        self._scene.set_drape(rgba)
        self._refresh()
        self._sb_status.setText("")

    # ── 2D-map sync ───────────────────────────────────────────────────────────

    def _emit_frustum(self):
        """Compute the camera's ground footprint in lat/lon and emit it.

        Returns silently if there is no active terrain (we'd have nothing to
        anchor the conversion to) or if the camera is pointed too far above
        the horizon to project onto the ground.
        """
        terrain = self._scene.terrain
        if terrain is None or self._viewport is None:
            self.frustum_changed.emit(None, None)
            return
        aspect = (
            self._viewport.width() / max(self._viewport.height(), 1)
            if self._viewport.height() > 0
            else 16 / 9
        )
        # Project frustum onto the elevation plane at the mean terrain height
        # — this lines up with what the user actually sees on the ground,
        # rather than z=0 (which can be far below the terrain at high mountains).
        z_plane = 0.5 * (terrain.z_min + terrain.z_max)
        corners_world = self._scene.camera.frustum_corners_at_z(aspect, z=z_plane)
        if corners_world is None:
            self.frustum_changed.emit(None, None)
            return
        corners_latlon = [
            terrain.local_to_latlon(c) for c in corners_world
        ]
        eye = self._scene.camera.eye()
        cam_latlon = terrain.local_to_latlon(np.array([eye[0], eye[1]]))
        self.frustum_changed.emit(corners_latlon, cam_latlon)

    def showEvent(self, ev):
        super().showEvent(ev)
        # Re-emit frustum on show so the 2D overlay reappears when we toggle
        # the 3D window back on.
        QTimer.singleShot(0, self._emit_frustum)

    def hideEvent(self, ev):
        super().hideEvent(ev)
        self.frustum_changed.emit(None, None)

    # ── Overlay positioning ───────────────────────────────────────────────────

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._reposition_overlays()

    def _reposition_overlays(self):
        if self._viewport is None or self._compass is None or self._axes is None:
            return
        if self._viewport.width() <= 0 or self._viewport.height() <= 0:
            return
        # Grid labels overlay fills the whole viewport so it can place text
        # anywhere on screen.  Compass/axes are tiny corner widgets — they
        # are raised above so the labels never paint over them.
        if self._grid_labels_overlay is not None:
            self._grid_labels_overlay.setGeometry(
                0, 0, self._viewport.width(), self._viewport.height(),
            )
        self._compass.move(
            self._viewport.width() - self._compass.width() - CompassRose.MARGIN,
            CompassRose.MARGIN,
        )
        self._axes.move(
            AxesTriad.MARGIN,
            self._viewport.height() - self._axes.height() - AxesTriad.MARGIN,
        )
        self._compass.raise_()
        self._axes.raise_()

    def _make_missing_gl_placeholder(self) -> QWidget:
        """Centre widget shown when PyOpenGL is missing.

        Tells the user exactly which command to run.  The 3D view still opens
        — the side panel works — but nothing renders.
        """
        msg = (
            "<h3>3D view unavailable</h3>"
            "<p>PyOpenGL is not installed.  Install it with:</p>"
            "<pre style='background:#222;color:#ddd;padding:6px 10px;border-radius:4px;'>"
            "pip install PyOpenGL PyOpenGL_accelerate"
            "</pre>"
            f"<p style='color:#888;'>Import error: <tt>{opengl_error()}</tt></p>"
        )
        lbl = QLabel(msg, self)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setStyleSheet("color:#ccc; padding:24px;")
        return lbl

    def _axes_basis(self):
        """Project world axes into camera space for the axes triad widget.

        Returns ((sx, sy, depth) × 3) for the +X, +Y, +Z world axes — screen
        space deltas (right/down positive) plus a relative depth used for
        Z-ordering the axes in the triad widget.
        """
        cam = self._scene.camera
        view = cam.view_matrix()
        out = []
        for v in [np.array([1.0, 0.0, 0.0]),
                  np.array([0.0, 1.0, 0.0]),
                  np.array([0.0, 0.0, 1.0])]:
            t = view @ np.array([v[0], v[1], v[2], 0.0])   # 0 = direction, not point
            sx, sy, depth = t[0], -t[1], t[2]
            out.append((sx, sy, depth))
        return out

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def closeEvent(self, ev):
        if self._drape_worker is not None and self._drape_worker.isRunning():
            self._drape_worker.discard()
            self._drape_worker.wait(500)
        self.closed.emit()
        super().closeEvent(ev)
