"""
Side configuration panel for the 3D window.

Four sections collapsed into a ``QToolBox``:

* **Layers** — check which 2D layers texture the terrain.  Reflects
  ``LayerManager`` state and is rebuilt on ``layers_changed``.
* **Terrain** — vertical exaggeration, mesh quality.
* **Lighting** — sun azimuth, altitude, ambient.
* **Earthquakes** — width / opacity / visibility for the stick renderer.
  The catalog itself, depth-to-colour mapping, and event filters all live
  in the 2D-side properties panel so the two views stay in lockstep.

Emits high-level signals the View3DWindow connects to scene mutators.
"""
from __future__ import annotations

import math
from typing import List

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QToolBox, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QCheckBox, QSlider, QDoubleSpinBox, QListWidget, QListWidgetItem,
    QPushButton, QFrame, QSizePolicy, QScrollArea, QComboBox,
)

from app.core.layer_manager import LayerManager
from app.core.scene3d import Scene3D


def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


class _Slider(QWidget):
    """A labelled slider + spinbox that share a single float value."""

    value_changed = pyqtSignal(float)

    def __init__(self, label: str, lo: float, hi: float, step: float,
                 initial: float, suffix: str = "", parent=None):
        super().__init__(parent)
        self._lo = lo
        self._hi = hi
        self._step = step
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._label = QLabel(label)
        self._label.setMinimumWidth(80)
        self._spin = QDoubleSpinBox()
        self._spin.setRange(lo, hi)
        self._spin.setSingleStep(step)
        self._spin.setDecimals(2 if step < 1 else 0)
        self._spin.setValue(initial)
        self._spin.setSuffix(suffix)
        self._spin.setMaximumWidth(80)
        # Drag the slider for live preview; the spinbox is for committed
        # numeric input — fire valueChanged only on Enter/Tab/focus-out so
        # typing "1500" doesn't rebuild the scene three times en route.
        self._spin.setKeyboardTracking(False)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.setValue(int((initial - lo) / (hi - lo) * 1000))

        layout.addWidget(self._label)
        layout.addWidget(self._slider, 1)
        layout.addWidget(self._spin)

        # Cross-wire slider ↔ spinbox without infinite loops.
        self._slider.valueChanged.connect(self._on_slider)
        self._spin.valueChanged.connect(self._on_spin)

    def _on_slider(self, raw: int):
        v = self._lo + (raw / 1000.0) * (self._hi - self._lo)
        self._spin.blockSignals(True)
        self._spin.setValue(v)
        self._spin.blockSignals(False)
        self.value_changed.emit(v)

    def _on_spin(self, v: float):
        raw = int((v - self._lo) / max(self._hi - self._lo, 1e-9) * 1000)
        self._slider.blockSignals(True)
        self._slider.setValue(raw)
        self._slider.blockSignals(False)
        self.value_changed.emit(v)

    def set_value(self, v: float):
        self._spin.setValue(v)


class SidePanel(QWidget):
    """Vertical config panel docked at the right of the 3D window."""

    # Drape selection — list of layer names (top of stack first)
    drape_layers_changed = pyqtSignal(list)
    # Live scene-property changes
    z_factor_changed = pyqtSignal(float)
    sun_changed = pyqtSignal(float, float, float)   # az_deg, alt_deg, ambient
    shading_toggled = pyqtSignal(bool)              # master lighting on/off
    wireframe_toggled = pyqtSignal(bool)
    mesh_quality_changed = pyqtSignal(int)          # mesh max_side cells
    # Earthquake style: (width_scale, opacity, visible, shape, anchor_to_surface)
    # ``shape`` is "cylinder" or "sphere"; ``anchor_to_surface`` is the
    # "Place at absolute depth" toggle inverted (True = anchor to terrain).
    earthquake_style_changed = pyqtSignal(float, float, bool, str, bool)
    # Coordinate grid: (show_grid, show_labels)
    grid_visibility_changed = pyqtSignal(bool, bool)

    def __init__(self, mgr: LayerManager, scene: Scene3D, parent=None):
        super().__init__(parent)
        self._mgr = mgr
        self._scene = scene
        self._suppress_layer_signal = False
        self._build_ui()
        self._mgr.layers_changed.connect(self._rebuild_layer_list)
        self._mgr.layer_renamed.connect(lambda *_: self._rebuild_layer_list())
        self._mgr.active_dem_changed.connect(lambda *_: self._rebuild_layer_list())

    # ── UI build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setMinimumWidth(260)
        self.setMaximumWidth(360)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self._toolbox = QToolBox(self)
        layout.addWidget(self._toolbox, 1)

        self._toolbox.addItem(self._build_layers_section(), "Layers")
        self._toolbox.addItem(self._build_terrain_section(), "Terrain")
        self._toolbox.addItem(self._build_lighting_section(), "Lighting")
        self._toolbox.addItem(self._build_grid_section(), "Coordinate grid")
        self._toolbox.addItem(self._build_earthquakes_section(), "Earthquakes")

    def _build_layers_section(self) -> QWidget:
        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        hint = QLabel("Pick which 2D layers drape over the terrain. "
                      "Top of list = top of stack.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#aaa; font-size:10px;")
        layout.addWidget(hint)

        self._layers_list = QListWidget()
        self._layers_list.setSelectionMode(
            QListWidget.SelectionMode.SingleSelection
        )
        layout.addWidget(self._layers_list, 1)

        btn_row = QHBoxLayout()
        self._btn_check_all = QPushButton("All")
        self._btn_check_none = QPushButton("None")
        self._btn_check_visible = QPushButton("Visible")
        for b in (self._btn_check_all, self._btn_check_none, self._btn_check_visible):
            b.setMaximumWidth(60)
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self._btn_check_all.clicked.connect(lambda: self._bulk_check(True))
        self._btn_check_none.clicked.connect(lambda: self._bulk_check(False))
        self._btn_check_visible.clicked.connect(self._check_visible_only)

        self._rebuild_layer_list()
        return wrap

    def _build_terrain_section(self) -> QWidget:
        wrap = QWidget()
        layout = QFormLayout(wrap)
        layout.setContentsMargins(4, 4, 4, 4)

        self._z_slider = _Slider("Z exaggeration",
                                 0.1, 5.0, 0.1,
                                 initial=self._scene.settings.z_factor,
                                 suffix="×")
        self._z_slider.value_changed.connect(self.z_factor_changed.emit)
        layout.addRow(self._z_slider)

        self._quality_combo = QComboBox()
        self._quality_combo.addItems([
            "Low (128×)",  "Medium (256×)", "High (512×)", "Very high (1024×)"
        ])
        self._quality_combo.setCurrentIndex(1)
        self._quality_combo.currentIndexChanged.connect(self._on_quality_change)
        layout.addRow("Mesh quality:", self._quality_combo)

        self._wf_check = QCheckBox("Wireframe")
        self._wf_check.toggled.connect(self.wireframe_toggled.emit)
        layout.addRow(self._wf_check)

        return wrap

    def _build_lighting_section(self) -> QWidget:
        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(4, 4, 4, 4)

        self._shading_check = QCheckBox("Enable shading")
        self._shading_check.setToolTip(
            "Apply directional (sun) lighting to the terrain.  When off, the "
            "drape colour is shown as-is with no Lambertian shading."
        )
        self._shading_check.setChecked(self._scene.sun.enabled)
        self._shading_check.toggled.connect(self._on_shading_toggled)
        layout.addWidget(self._shading_check)

        self._az = _Slider("Azimuth",  0.0, 360.0, 5.0,
                           initial=self._scene.sun.azimuth_deg, suffix="°")
        self._alt = _Slider("Altitude", 0.0,  90.0, 1.0,
                            initial=self._scene.sun.altitude_deg, suffix="°")
        self._amb = _Slider("Ambient", 0.0,   1.0, 0.05,
                            initial=self._scene.sun.ambient, suffix="")

        layout.addWidget(self._az)
        layout.addWidget(self._alt)
        layout.addWidget(self._amb)

        for s in (self._az, self._alt, self._amb):
            s.value_changed.connect(lambda _v: self._emit_sun())

        layout.addWidget(_hline())

        self._btn_match = QPushButton("Match active hillshade")
        self._btn_match.setToolTip(
            "Pull azimuth/altitude from the most recent hillshade layer of "
            "the active DEM."
        )
        layout.addWidget(self._btn_match)
        self._btn_match.clicked.connect(self._match_hillshade)

        # Reflect initial enabled state on the sliders.
        self._apply_shading_enabled(self._scene.sun.enabled)

        layout.addStretch(1)
        return wrap

    def _build_grid_section(self) -> QWidget:
        """Coordinate-grid toggles — gridlines and numeric tick labels."""
        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        hint = QLabel(
            "Draws a lat/lon/depth bounding box with tick labels.  Axes are "
            "coloured: longitude=red, latitude=green, depth=blue."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#aaa; font-size:10px;")
        layout.addWidget(hint)

        self._grid_show = QCheckBox("Show coordinate grid")
        self._grid_show.setChecked(self._scene.settings.show_grid)
        layout.addWidget(self._grid_show)

        self._grid_labels = QCheckBox("Show tick labels")
        self._grid_labels.setChecked(self._scene.settings.show_grid_labels)
        layout.addWidget(self._grid_labels)

        self._grid_show.toggled.connect(lambda _v: self._emit_grid())
        self._grid_labels.toggled.connect(lambda _v: self._emit_grid())

        layout.addStretch(1)
        return wrap

    def _emit_grid(self):
        self.grid_visibility_changed.emit(
            bool(self._grid_show.isChecked()),
            bool(self._grid_labels.isChecked()),
        )

    def _build_earthquakes_section(self) -> QWidget:
        """Section for the 3D-only stick styling (width / opacity / visibility).

        Filters (mag/depth/date) and the depth-to-colour ramp stay on the
        properties panel that drives the 2D markers — same catalog, same
        filter logic, one place to edit.  Here we only tune how the sticks
        themselves *render* in 3D.
        """
        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        hint = QLabel(
            "Earthquakes hang as cylinders below the terrain — top sits on "
            "the surface, stick length = depth, width ∝ magnitude^1.5.  "
            "Depth colour, filters and 2D markers are controlled from the "
            "Properties panel."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#aaa; font-size:10px;")
        layout.addWidget(hint)

        self._eq_visible = QCheckBox("Show in 3D")
        self._eq_visible.setChecked(True)
        layout.addWidget(self._eq_visible)

        # Shape picker — cylinder (default, hangs from surface to hypocenter)
        # or sphere (single marker at the hypocenter).
        shape_row = QHBoxLayout()
        shape_row.setSpacing(6)
        shape_row.addWidget(QLabel("Shape:"))
        self._eq_shape = QComboBox()
        self._eq_shape.addItems(["Cylinder", "Sphere"])
        self._eq_shape.setCurrentIndex(1)               # default to Sphere
        shape_row.addWidget(self._eq_shape, 1)
        layout.addLayout(shape_row)

        # Anchor toggle — when checked, points sit at absolute depth from
        # sea level (z = -depth) instead of hanging beneath the terrain.
        self._eq_absolute = QCheckBox("Place at absolute depth")
        self._eq_absolute.setToolTip(
            "When on, earthquakes sit at their true depth below sea level "
            "instead of below the local terrain surface.  Useful for "
            "reading a regional catalog against a cross-section."
        )
        self._eq_absolute.setChecked(True)              # absolute depth is the default
        layout.addWidget(self._eq_absolute)

        self._eq_width = _Slider(
            "Width ×", 0.2, 8.0, 0.1, initial=3.0, suffix="×",
        )
        self._eq_opacity = _Slider(
            "Opacity", 0.1, 1.0, 0.05, initial=0.85, suffix="",
        )
        layout.addWidget(self._eq_width)
        layout.addWidget(self._eq_opacity)

        self._eq_visible.toggled.connect(lambda _v: self._emit_eq_style())
        self._eq_shape.currentIndexChanged.connect(lambda _i: self._emit_eq_style())
        self._eq_absolute.toggled.connect(lambda _v: self._emit_eq_style())
        self._eq_width.value_changed.connect(lambda _v: self._emit_eq_style())
        self._eq_opacity.value_changed.connect(lambda _v: self._emit_eq_style())

        layout.addStretch(1)
        return wrap

    def _emit_eq_style(self):
        width = float(self._eq_width._spin.value())
        opacity = float(self._eq_opacity._spin.value())
        visible = bool(self._eq_visible.isChecked())
        shape = "sphere" if self._eq_shape.currentIndex() == 1 else "cylinder"
        # "Place at absolute depth" ↔ anchor_to_surface inverted.
        anchor = not bool(self._eq_absolute.isChecked())
        self.earthquake_style_changed.emit(width, opacity, visible, shape, anchor)

    # ── Slot handlers ─────────────────────────────────────────────────────────

    def _on_quality_change(self, idx: int):
        sizes = [128, 256, 512, 1024]
        self.mesh_quality_changed.emit(sizes[max(0, min(idx, len(sizes) - 1))])

    def _emit_sun(self):
        # Slider value getters: read the spinbox values directly.
        az = self._az._spin.value()
        alt = self._alt._spin.value()
        amb = self._amb._spin.value()
        self.sun_changed.emit(az, alt, amb)

    def _on_shading_toggled(self, on: bool):
        self._apply_shading_enabled(on)
        self.shading_toggled.emit(on)

    def _apply_shading_enabled(self, on: bool):
        # The sun controls have no effect when shading is off; grey them out
        # so the user can see why moving the sun isn't changing anything.
        for w in (self._az, self._alt, self._amb, self._btn_match):
            w.setEnabled(on)

    def _match_hillshade(self):
        """Pull azimuth/altitude from the latest hillshade of the active DEM."""
        active = self._mgr.active_dem
        if active is None:
            return
        latest = None
        for l in self._mgr.all():
            if l.parent_name == active.name and l.product == "hillshade":
                latest = l
                break  # all() returns newest first
        if latest is None:
            return
        # The hillshade kwargs aren't on the layer object, but the global
        # defaults are reasonable fallbacks.  Future versions could store the
        # generating parameters on the layer.
        from app.config import DEFAULT_AZIMUTH, DEFAULT_ALTITUDE
        self._az.set_value(DEFAULT_AZIMUTH)
        self._alt.set_value(DEFAULT_ALTITUDE)

    # ── Layer list ────────────────────────────────────────────────────────────

    def _rebuild_layer_list(self):
        """Rebuild the layer checklist from the LayerManager.

        Preserves the user's existing check state for layers that survived
        the rebuild; layers added since last rebuild default to checked.
        """
        prev_checked = self._currently_checked_names()
        had_prior = bool(self._scene.drape_layer_names)

        self._suppress_layer_signal = True
        self._layers_list.clear()
        active_name = (
            self._mgr.active_dem.name if self._mgr.active_dem else None
        )

        for layer in self._mgr.all():
            item = QListWidgetItem(layer.name)
            if active_name and layer.name == active_name:
                item.setText(f"{layer.name}  (active)")
                f = item.font(); f.setBold(True); item.setFont(f)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            if had_prior:
                checked = layer.name in prev_checked
            else:
                # Default selection: active DEM + visible derived layers.
                checked = layer.visible and (
                    layer.name == active_name
                    or (active_name and layer.parent_name == active_name)
                )
            item.setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )
            self._layers_list.addItem(item)
        self._suppress_layer_signal = False

        try:
            self._layers_list.itemChanged.disconnect(self._on_item_changed)
        except (TypeError, RuntimeError):
            pass
        self._layers_list.itemChanged.connect(self._on_item_changed)

        # Emit once so the scene picks up the rebuilt selection.
        self._emit_selection()

    def _on_item_changed(self, _item: QListWidgetItem):
        if self._suppress_layer_signal:
            return
        self._emit_selection()

    def _emit_selection(self):
        names: List[str] = self._currently_checked_names()
        self.drape_layers_changed.emit(names)

    def _currently_checked_names(self) -> List[str]:
        names = []
        for i in range(self._layers_list.count()):
            it = self._layers_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                # Strip "  (active)" suffix if present.
                text = it.text()
                if text.endswith("  (active)"):
                    text = text[: -len("  (active)")]
                names.append(text)
        return names

    def _bulk_check(self, check: bool):
        self._suppress_layer_signal = True
        state = Qt.CheckState.Checked if check else Qt.CheckState.Unchecked
        for i in range(self._layers_list.count()):
            self._layers_list.item(i).setCheckState(state)
        self._suppress_layer_signal = False
        self._emit_selection()

    def _check_visible_only(self):
        self._suppress_layer_signal = True
        layers = {l.name: l for l in self._mgr.all()}
        for i in range(self._layers_list.count()):
            it = self._layers_list.item(i)
            name = it.text().removesuffix("  (active)")
            layer = layers.get(name)
            it.setCheckState(
                Qt.CheckState.Checked if (layer and layer.visible)
                else Qt.CheckState.Unchecked
            )
        self._suppress_layer_signal = False
        self._emit_selection()
