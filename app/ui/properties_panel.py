"""
Properties panel — shows statistics, histogram, and color-ramp controls
for the currently selected layer.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QFormLayout,
    QLabel, QComboBox, QDoubleSpinBox, QPushButton,
    QGroupBox, QSizePolicy,
)

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigCanvas
from matplotlib.figure import Figure

from app.core.layer_manager import LayerManager
from app.core.dem_layer import DemLayer


_COLORMAPS = [
    "terrain", "Greys_r", "viridis", "plasma", "inferno",
    "RdBu_r", "RdYlGn", "YlOrRd", "Blues", "YlOrBr",
    "hsv", "twilight", "BrBG", "PuOr",
]


class PropertiesPanel(QDockWidget):
    """Dockable properties & styling panel for the selected layer."""

    style_changed = pyqtSignal(str)   # layer name

    def __init__(self, manager: LayerManager, parent=None):
        super().__init__("Properties", parent)
        self._mgr = manager
        self._current: Optional[str] = None

        self.setMinimumWidth(220)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self._build_ui()
        self._mgr.layer_selected.connect(self._load_layer)
        self._mgr.layer_updated.connect(self._on_layer_updated)

    def _build_ui(self):
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

        self._spin_min = QDoubleSpinBox()
        self._spin_min.setRange(-1e9, 1e9)
        self._spin_min.setDecimals(2)
        self._spin_min.valueChanged.connect(self._on_range_changed)
        style_layout.addRow("Min value:", self._spin_min)

        self._spin_max = QDoubleSpinBox()
        self._spin_max.setRange(-1e9, 1e9)
        self._spin_max.setDecimals(2)
        self._spin_max.valueChanged.connect(self._on_range_changed)
        style_layout.addRow("Max value:", self._spin_max)

        btn_auto = QPushButton("Auto stretch (2–98%)")
        btn_auto.clicked.connect(self._auto_stretch)
        style_layout.addRow(btn_auto)

        layout.addWidget(style_box)
        layout.addStretch()
        self.setWidget(container)

    # ── Loading ────────────────────────────────────────────────────────────

    def _load_layer(self, name: str):
        self._current = name
        layer = self._mgr.get(name)
        if not layer:
            return

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

        lo = layer.render_min
        hi = layer.render_max
        if lo is None or hi is None:
            lo, hi = layer.auto_range()

        self._spin_min.blockSignals(True)
        self._spin_max.blockSignals(True)
        self._spin_min.setValue(lo)
        self._spin_max.setValue(hi)
        self._spin_min.blockSignals(False)
        self._spin_max.blockSignals(False)

        cmap = layer.colormap
        idx = self._cmap_combo.findText(cmap)
        self._cmap_combo.blockSignals(True)
        self._cmap_combo.setCurrentIndex(max(idx, 0))
        self._cmap_combo.blockSignals(False)

    def _on_layer_updated(self, name: str):
        if name == self._current:
            self._load_layer(name)

    # ── Histogram ─────────────────────────────────────────────────────────

    def _draw_histogram(self, layer: DemLayer):
        self._ax.cla()
        self._ax.set_facecolor("#252525")
        d = layer.valid_data.ravel()
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

    def _on_range_changed(self):
        if self._current:
            lo = self._spin_min.value()
            hi = self._spin_max.value()
            if lo < hi:
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
