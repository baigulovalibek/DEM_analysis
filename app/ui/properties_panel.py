"""
Properties panel — shows statistics, histogram, and color-ramp controls
for the currently selected layer.
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QDateTime, pyqtSignal
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
    QLabel, QComboBox, QDoubleSpinBox, QPushButton, QCheckBox,
    QGroupBox, QSizePolicy, QSlider, QStackedWidget, QDateTimeEdit,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigCanvas
from matplotlib.figure import Figure

from app.core.layer_manager import LayerManager
from app.core.dem_layer import DemLayer
from app.core.earthquakes import EarthquakeCatalog


_COLORMAPS = [
    "terrain", "Greys_r", "viridis", "plasma", "inferno",
    "RdBu_r", "RdYlGn", "YlOrRd", "Blues", "YlOrBr",
    "hsv", "twilight", "BrBG", "PuOr",
]

# Subset of colormaps the 2D map.html knows how to render in JS.  Keep the 3D
# view in sync by only offering these names in the earthquake dropdown.
# ``RdYlBu`` is first because it matches the USGS shallow=red / deep=blue
# convention every seismologist already has wired in.
_EQ_COLORMAPS = [
    "RdYlBu", "plasma", "viridis", "inferno", "magma", "cividis",
    "YlOrRd", "Greys",
]


class PropertiesPanel(QDockWidget):
    """Dockable properties & styling panel for the selected layer."""

    style_changed = pyqtSignal(str)   # layer name
    # Emitted whenever any earthquake style/filter widget changes.  MainWindow
    # listens and forwards to both the 2D and 3D renderers.
    earthquake_style_changed = pyqtSignal()

    def __init__(self, manager: LayerManager, parent=None):
        super().__init__("Properties", parent)
        self._mgr = manager
        self._current: Optional[str] = None
        self._catalog: Optional[EarthquakeCatalog] = None
        self._suppress_eq_signal = False

        self.setMinimumWidth(220)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self._build_ui()
        self._mgr.layer_selected.connect(self._load_layer)
        self._mgr.layer_updated.connect(self._on_layer_updated)
        self._mgr.layer_renamed.connect(self._on_layer_renamed)

    def _build_ui(self):
        # Outer stack: raster page (default) + earthquake page.  MainWindow
        # flips the page by calling ``show_earthquakes()`` / ``show_layer()``.
        self._stack = QStackedWidget(self)
        self.setWidget(self._stack)

        # Raster page is the original UI verbatim.
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── Layer info ──────────────────────────────────────────────────────
        info_box = QGroupBox("Layer Info")
        info_layout = QFormLayout(info_box)
        info_layout.setSpacing(3)
        self._lbl_name    = QLabel("—")
        self._lbl_product = QLabel("—")
        self._lbl_size    = QLabel("—")
        self._lbl_crs     = QLabel("—")
        info_layout.addRow("Name:",    self._lbl_name)
        info_layout.addRow("Product:", self._lbl_product)
        info_layout.addRow("Size:",    self._lbl_size)
        info_layout.addRow("CRS:",     self._lbl_crs)
        layout.addWidget(info_box)

        # ── Statistics ──────────────────────────────────────────────────────
        stats_box = QGroupBox("Statistics")
        stats_layout = QFormLayout(stats_box)
        stats_layout.setSpacing(3)
        self._lbl_min  = QLabel("—")
        self._lbl_max  = QLabel("—")
        self._lbl_mean = QLabel("—")
        self._lbl_std  = QLabel("—")
        for label, widget in [("Min:", self._lbl_min), ("Max:", self._lbl_max),
                               ("Mean:", self._lbl_mean), ("Std:", self._lbl_std)]:
            stats_layout.addRow(label, widget)
        layout.addWidget(stats_box)

        # ── Histogram ───────────────────────────────────────────────────────
        self._fig = Figure(figsize=(2.5, 1.2), facecolor="#252525")
        self._ax  = self._fig.add_subplot(111)
        self._ax.set_facecolor("#252525")
        self._ax.tick_params(colors="#aaa", labelsize=7)
        for spine in self._ax.spines.values():
            spine.set_edgecolor("#444")
        self._fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.25)
        self._canvas = FigCanvas(self._fig)
        self._canvas.setMinimumHeight(100)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self._canvas)

        # ── Styling ─────────────────────────────────────────────────────────
        style_box = QGroupBox("Render Style")
        style_layout = QFormLayout(style_box)
        style_layout.setSpacing(4)

        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(_COLORMAPS)
        self._cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        style_layout.addRow("Colormap:", self._cmap_combo)

        # Opacity / alpha control
        op_row = QWidget()
        op_layout = QHBoxLayout(op_row)
        op_layout.setContentsMargins(0, 0, 0, 0)
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(75)
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        self._opacity_label = QLabel("75%")
        self._opacity_label.setFixedWidth(36)
        op_layout.addWidget(self._opacity_slider)
        op_layout.addWidget(self._opacity_label)
        style_layout.addRow("Opacity:", op_row)

        self._spin_min = QDoubleSpinBox()
        self._spin_min.setRange(-1e9, 1e9)
        self._spin_min.setDecimals(2)
        # Defer the (expensive) PNG re-encode + map overlay swap until the
        # user commits a value — pressing Enter/Tab, leaving the field, or
        # using the step buttons. Without this, every keystroke (including
        # backspaces while editing) fires a full re-render.
        self._spin_min.setKeyboardTracking(False)
        self._spin_min.valueChanged.connect(self._on_range_changed)
        style_layout.addRow("Min value:", self._spin_min)

        self._spin_max = QDoubleSpinBox()
        self._spin_max.setRange(-1e9, 1e9)
        self._spin_max.setDecimals(2)
        self._spin_max.setKeyboardTracking(False)
        self._spin_max.valueChanged.connect(self._on_range_changed)
        style_layout.addRow("Max value:", self._spin_max)

        btn_auto = QPushButton("Auto stretch (2–98%)")
        btn_auto.clicked.connect(self._auto_stretch)
        style_layout.addRow(btn_auto)

        layout.addWidget(style_box)
        layout.addStretch()
        self._stack.addWidget(container)         # index 0 — raster
        self._stack.addWidget(self._build_eq_page())   # index 1 — earthquakes

    # ── Earthquake page ────────────────────────────────────────────────────

    def _build_eq_page(self) -> QWidget:
        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Info ─────────────────────────────────────────────
        info = QGroupBox("Earthquake Catalog")
        info_layout = QFormLayout(info)
        info_layout.setSpacing(3)
        self._eq_lbl_source = QLabel("—")
        self._eq_lbl_count  = QLabel("—")
        self._eq_lbl_mag    = QLabel("—")
        self._eq_lbl_depth  = QLabel("—")
        self._eq_lbl_date   = QLabel("—")
        info_layout.addRow("Source:", self._eq_lbl_source)
        info_layout.addRow("Events:", self._eq_lbl_count)
        info_layout.addRow("Mag range:", self._eq_lbl_mag)
        info_layout.addRow("Depth range:", self._eq_lbl_depth)
        info_layout.addRow("Date range:", self._eq_lbl_date)
        layout.addWidget(info)

        # Symbology ─────────────────────────────────────────
        sym = QGroupBox("Symbology")
        sym_layout = QFormLayout(sym)
        sym_layout.setSpacing(4)

        # Encoding is hard-locked: depth → color, magnitude → size.  The
        # cross-section dialog, 2D legend, and 3D legend all assume this; a
        # dropdown that lets the user override it just creates confusion.
        encoding_lbl = QLabel("Depth → color · Magnitude → size")
        encoding_lbl.setStyleSheet("color:#aaa; font-size:10px;")
        sym_layout.addRow(encoding_lbl)

        self._eq_cmap = QComboBox()
        self._eq_cmap.addItems(_EQ_COLORMAPS)
        self._eq_cmap.currentTextChanged.connect(self._on_eq_changed)
        sym_layout.addRow("Depth ramp:", self._eq_cmap)

        size_row = QWidget()
        size_layout = QHBoxLayout(size_row)
        size_layout.setContentsMargins(0, 0, 0, 0)
        self._eq_size = QSlider(Qt.Orientation.Horizontal)
        self._eq_size.setRange(5, 50)       # 0.5×..5.0× → 5..50 (÷10)
        self._eq_size.setValue(10)
        self._eq_size.valueChanged.connect(self._on_eq_changed)
        self._eq_size_label = QLabel("1.0×")
        self._eq_size_label.setFixedWidth(36)
        size_layout.addWidget(self._eq_size)
        size_layout.addWidget(self._eq_size_label)
        sym_layout.addRow("Size scale:", size_row)

        layout.addWidget(sym)

        # Filters ──────────────────────────────────────────
        flt = QGroupBox("Filters")
        flt_layout = QFormLayout(flt)
        flt_layout.setSpacing(4)

        mag_row = QWidget(); mag_layout = QHBoxLayout(mag_row)
        mag_layout.setContentsMargins(0, 0, 0, 0)
        self._eq_mag_min = QDoubleSpinBox()
        self._eq_mag_max = QDoubleSpinBox()
        for s in (self._eq_mag_min, self._eq_mag_max):
            s.setDecimals(2); s.setSingleStep(0.1); s.setRange(-2.0, 10.0)
            s.setKeyboardTracking(False)
            s.valueChanged.connect(self._on_eq_changed)
        mag_layout.addWidget(self._eq_mag_min); mag_layout.addWidget(QLabel("..")); mag_layout.addWidget(self._eq_mag_max)
        flt_layout.addRow("Magnitude:", mag_row)

        dep_row = QWidget(); dep_layout = QHBoxLayout(dep_row)
        dep_layout.setContentsMargins(0, 0, 0, 0)
        self._eq_dep_min = QDoubleSpinBox()
        self._eq_dep_max = QDoubleSpinBox()
        for s in (self._eq_dep_min, self._eq_dep_max):
            s.setDecimals(2); s.setSingleStep(1.0); s.setRange(0.0, 1000.0); s.setSuffix(" km")
            s.setKeyboardTracking(False)
            s.valueChanged.connect(self._on_eq_changed)
        dep_layout.addWidget(self._eq_dep_min); dep_layout.addWidget(QLabel("..")); dep_layout.addWidget(self._eq_dep_max)
        flt_layout.addRow("Depth:", dep_row)

        self._eq_date_min = QDateTimeEdit()
        self._eq_date_max = QDateTimeEdit()
        for w in (self._eq_date_min, self._eq_date_max):
            w.setCalendarPopup(True)
            w.setDisplayFormat("yyyy-MM-dd")
            w.setKeyboardTracking(False)
            w.dateTimeChanged.connect(self._on_eq_changed)
        flt_layout.addRow("From:", self._eq_date_min)
        flt_layout.addRow("To:",   self._eq_date_max)

        btn_reset = QPushButton("Reset filters")
        btn_reset.clicked.connect(self._reset_eq_filters)
        flt_layout.addRow(btn_reset)

        layout.addWidget(flt)

        self._eq_lbl_visible = QLabel("—")
        self._eq_lbl_visible.setStyleSheet("color:#aaa;")
        layout.addWidget(self._eq_lbl_visible)

        layout.addStretch()
        return wrap

    # ── Earthquake page wiring ─────────────────────────────────────────────

    def set_catalog(self, catalog: Optional[EarthquakeCatalog]):
        """MainWindow calls this whenever the loaded catalog changes."""
        self._catalog = catalog
        self._sync_eq_widgets()

    def show_earthquakes(self):
        """Switch to the earthquake-properties page."""
        self._stack.setCurrentIndex(1)
        self._sync_eq_widgets()

    def refresh_earthquake_widgets(self) -> None:
        """Re-sync the earthquake widgets from the catalog without changing
        which page is visible.  Used by MainWindow when another panel mutated
        the shared style and this panel needs to catch up.
        """
        self._sync_eq_widgets()

    def show_layer(self):
        """Switch back to the raster-properties page."""
        self._stack.setCurrentIndex(0)

    def _sync_eq_widgets(self):
        """Pull current catalog/style state into the widgets."""
        self._suppress_eq_signal = True
        try:
            cat = self._catalog
            if cat is None or len(cat) == 0:
                self._eq_lbl_source.setText("—")
                self._eq_lbl_count.setText("—")
                self._eq_lbl_mag.setText("—")
                self._eq_lbl_depth.setText("—")
                self._eq_lbl_date.setText("—")
                self._eq_lbl_visible.setText("Load a catalog from File → Open Earthquake Catalog…")
                return
            s = cat.style
            src = cat.source_path.name if cat.source_path else "in-memory"
            self._eq_lbl_source.setText(src)
            self._eq_lbl_count.setText(str(len(cat)))
            m_lo = float(np.min(cat.magnitudes)); m_hi = float(np.max(cat.magnitudes))
            d_lo = float(np.min(cat.depths_km)); d_hi = float(np.max(cat.depths_km))
            self._eq_lbl_mag.setText(f"M{m_lo:.2f} … M{m_hi:.2f}")
            self._eq_lbl_depth.setText(f"{d_lo:.2f} … {d_hi:.2f} km")
            dts = [e.origin for e in cat.events if e.origin is not None]
            if dts:
                self._eq_lbl_date.setText(
                    f"{min(dts).date().isoformat()} … {max(dts).date().isoformat()}"
                )
            else:
                self._eq_lbl_date.setText("(no timestamps)")

            idx = self._eq_cmap.findText(s.colormap)
            self._eq_cmap.setCurrentIndex(idx if idx >= 0 else 0)
            self._eq_size.setValue(int(round(s.size_scale * 10)))
            self._eq_size_label.setText(f"{s.size_scale:.1f}×")

            # Filter spinboxes — set range to actual data + a small buffer.
            self._eq_mag_min.setRange(m_lo - 0.5, m_hi)
            self._eq_mag_max.setRange(m_lo, m_hi + 0.5)
            self._eq_mag_min.setValue(s.mag_min if s.mag_min is not None else m_lo)
            self._eq_mag_max.setValue(s.mag_max if s.mag_max is not None else m_hi)

            self._eq_dep_min.setRange(0.0, max(d_hi, 1.0))
            self._eq_dep_max.setRange(0.0, max(d_hi, 1.0) + 1.0)
            self._eq_dep_min.setValue(s.depth_min_km if s.depth_min_km is not None else d_lo)
            self._eq_dep_max.setValue(s.depth_max_km if s.depth_max_km is not None else d_hi)

            if dts:
                dlo = s.date_min or min(dts)
                dhi = s.date_max or max(dts)
                self._eq_date_min.setEnabled(True)
                self._eq_date_max.setEnabled(True)
                self._eq_date_min.setDateTimeRange(
                    QDateTime(min(dts)), QDateTime(max(dts))
                )
                self._eq_date_max.setDateTimeRange(
                    QDateTime(min(dts)), QDateTime(max(dts))
                )
                self._eq_date_min.setDateTime(QDateTime(dlo))
                self._eq_date_max.setDateTime(QDateTime(dhi))
            else:
                self._eq_date_min.setEnabled(False)
                self._eq_date_max.setEnabled(False)

            n_vis = int(cat.visible_indices().size)
            self._eq_lbl_visible.setText(f"{n_vis} of {len(cat)} events pass current filters")
        finally:
            self._suppress_eq_signal = False

    def _on_eq_changed(self, *_args):
        if self._suppress_eq_signal or self._catalog is None:
            return
        s = self._catalog.style
        # color_by is hard-locked to "depth" at the model level; the UI does
        # not expose it.  Re-asserting here defends against an old style
        # surviving from a serialised state in the future.
        s.color_by = "depth"
        s.colormap = self._eq_cmap.currentText()
        s.size_scale = self._eq_size.value() / 10.0
        # Enforce min<=max on the spinboxes by nudging the opposing side.
        if self._eq_mag_min.value() > self._eq_mag_max.value():
            self._eq_mag_max.blockSignals(True)
            self._eq_mag_max.setValue(self._eq_mag_min.value())
            self._eq_mag_max.blockSignals(False)
        if self._eq_dep_min.value() > self._eq_dep_max.value():
            self._eq_dep_max.blockSignals(True)
            self._eq_dep_max.setValue(self._eq_dep_min.value())
            self._eq_dep_max.blockSignals(False)
        s.mag_min     = self._eq_mag_min.value()
        s.mag_max     = self._eq_mag_max.value()
        s.depth_min_km = self._eq_dep_min.value()
        s.depth_max_km = self._eq_dep_max.value()
        if self._eq_date_min.isEnabled():
            s.date_min = self._eq_date_min.dateTime().toPyDateTime()
            s.date_max = self._eq_date_max.dateTime().toPyDateTime()
        self._eq_size_label.setText(f"{s.size_scale:.1f}×")
        # Recompute the color stretch to span the new ``color_by`` field; users
        # can still override by setting color_min/max explicitly later.
        self._catalog.recompute_color_range()

        n_vis = int(self._catalog.visible_indices().size)
        self._eq_lbl_visible.setText(
            f"{n_vis} of {len(self._catalog)} events pass current filters"
        )
        self.earthquake_style_changed.emit()

    def _reset_eq_filters(self):
        if self._catalog is None:
            return
        s = self._catalog.style
        s.mag_min = s.mag_max = None
        s.depth_min_km = s.depth_max_km = None
        s.date_min = s.date_max = None
        # Re-run the catalog's auto-style to pull fresh full-range values.
        self._catalog._auto_style()
        self._sync_eq_widgets()
        self.earthquake_style_changed.emit()

    # ── Loading ────────────────────────────────────────────────────────────

    def _load_layer(self, name: str):
        self._current = name
        layer = self._mgr.get(name)
        if not layer:
            return
        # Picking a raster layer always swaps back to the raster page so the
        # user sees the histogram/style controls they expect.
        self.show_layer()

        self._lbl_name.setText(layer.name)
        self._lbl_product.setText(layer.product)
        h, w = layer.shape
        self._lbl_size.setText(f"{w} × {h} px")
        crs_short = layer.crs_wkt[:40] + "…" if len(layer.crs_wkt) > 40 else layer.crs_wkt
        self._lbl_crs.setText(crs_short or "unknown")

        stats = layer.stats
        if stats:
            self._lbl_min.setText(f"{stats['min']:.4g}")
            self._lbl_max.setText(f"{stats['max']:.4g}")
            self._lbl_mean.setText(f"{stats['mean']:.4g}")
            self._lbl_std.setText(f"{stats['std']:.4g}")

        self._draw_histogram(layer)
        self._sync_style(layer)

    def _sync_style(self, layer: DemLayer):
        """Refresh the lightweight style widgets (no histogram redraw)."""
        lo = layer.render_min
        hi = layer.render_max
        if lo is None or hi is None:
            # Only fall back to an expensive percentile pass when the layer
            # genuinely has no render range yet.
            lo, hi = layer.auto_range()

        self._spin_min.blockSignals(True)
        self._spin_max.blockSignals(True)
        self._spin_min.setValue(lo)
        self._spin_max.setValue(hi)
        self._spin_min.blockSignals(False)
        self._spin_max.blockSignals(False)

        idx = self._cmap_combo.findText(layer.colormap)
        self._cmap_combo.blockSignals(True)
        self._cmap_combo.setCurrentIndex(max(idx, 0))
        self._cmap_combo.blockSignals(False)

        pct = int(round(layer.opacity * 100))
        self._opacity_slider.blockSignals(True)
        self._opacity_slider.setValue(pct)
        self._opacity_slider.blockSignals(False)
        self._opacity_label.setText(f"{pct}%")

    def _on_layer_updated(self, name: str):
        # Style/opacity/visibility change — sync widgets only; the histogram
        # depends solely on the data array, so it never needs redrawing here.
        if name == self._current:
            layer = self._mgr.get(name)
            if layer:
                self._sync_style(layer)

    def _on_layer_renamed(self, old: str, new: str):
        if self._current == old:
            self._current = new
            self._lbl_name.setText(new)

    # ── Histogram ─────────────────────────────────────────────────────────

    _HIST_MAX_SAMPLES = 200_000

    def _draw_histogram(self, layer: DemLayer):
        self._ax.cla()
        self._ax.set_facecolor("#252525")
        d = layer.valid_data.ravel()
        # Subsample huge layers — a histogram doesn't need every cell, and
        # ravel→isfinite→hist on a 4000×4000 float array stalls the UI.
        n = d.size
        if n > self._HIST_MAX_SAMPLES:
            stride = int(np.ceil(n / self._HIST_MAX_SAMPLES))
            d = d[::stride]
        d = d[np.isfinite(d)]
        if d.size > 0:
            self._ax.hist(d, bins=80, color="#5599dd", alpha=0.85, linewidth=0)
        self._ax.tick_params(colors="#aaa", labelsize=7)
        for spine in self._ax.spines.values():
            spine.set_edgecolor("#444")
        self._canvas.draw()

    # ── Style callbacks ────────────────────────────────────────────────────

    def _on_cmap_changed(self, cmap: str):
        if self._current:
            self._mgr.set_colormap(self._current, cmap)
            self.style_changed.emit(self._current)

    def _on_opacity_changed(self, value: int):
        self._opacity_label.setText(f"{value}%")
        if self._current:
            self._mgr.set_opacity(self._current, value / 100.0)

    def _on_range_changed(self):
        if not self._current:
            return
        lo = self._spin_min.value()
        hi = self._spin_max.value()
        if lo >= hi:
            # Clamp the offending spinbox to keep min < max instead of silently
            # ignoring the change.  We nudge the other side by one increment.
            sender = self.sender()
            step = max(self._spin_min.singleStep(), 1e-6)
            self._spin_min.blockSignals(True)
            self._spin_max.blockSignals(True)
            if sender is self._spin_min:
                self._spin_min.setValue(hi - step)
                lo = hi - step
            else:
                self._spin_max.setValue(lo + step)
                hi = lo + step
            self._spin_min.blockSignals(False)
            self._spin_max.blockSignals(False)
        self._mgr.set_render_range(self._current, lo, hi)
        self.style_changed.emit(self._current)

    def _auto_stretch(self):
        if not self._current:
            return
        layer = self._mgr.get(self._current)
        if not layer:
            return
        lo, hi = layer.auto_range()
        self._spin_min.blockSignals(True)
        self._spin_max.blockSignals(True)
        self._spin_min.setValue(lo)
        self._spin_max.setValue(hi)
        self._spin_min.blockSignals(False)
        self._spin_max.blockSignals(False)
        self._mgr.set_render_range(self._current, lo, hi)
        self.style_changed.emit(self._current)
