"""
Analysis panel — parameter forms for every analysis type, grouped in a tree.
Emits run_analysis(product, params_dict) when the user clicks Compute.
"""
from __future__ import annotations
from typing import Dict, Any

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QScrollArea,
    QTreeWidget, QTreeWidgetItem, QStackedWidget,
    QFormLayout, QDoubleSpinBox, QSpinBox, QComboBox,
    QCheckBox, QPushButton, QLabel, QGroupBox, QSizePolicy,
    QProgressBar, QHBoxLayout,
)

from app.config import (
    DEFAULT_AZIMUTH, DEFAULT_ALTITUDE, DEFAULT_Z_FACTOR,
    DEFAULT_TPI_RADIUS, DEFAULT_SVF_DIRECTIONS, DEFAULT_SVF_RADIUS,
    DEFAULT_ROUGHNESS_WINDOW,
)


# ── Per-product parameter forms ────────────────────────────────────────────

class _ParamForm(QWidget):
    """Base class for parameter panels."""
    def params(self) -> Dict[str, Any]:
        raise NotImplementedError


class HillshadeForm(_ParamForm):
    def __init__(self):
        super().__init__()
        form = QFormLayout(self)
        self._az  = QDoubleSpinBox(); self._az.setRange(0, 360); self._az.setValue(DEFAULT_AZIMUTH)
        self._alt = QDoubleSpinBox(); self._alt.setRange(0, 90);  self._alt.setValue(DEFAULT_ALTITUDE)
        self._zf  = QDoubleSpinBox(); self._zf.setRange(0.01, 100); self._zf.setValue(DEFAULT_Z_FACTOR)
        form.addRow("Azimuth (°):", self._az)
        form.addRow("Altitude (°):", self._alt)
        form.addRow("Z-factor:", self._zf)

    def params(self):
        return {"azimuth": self._az.value(), "altitude": self._alt.value(),
                "z_factor": self._zf.value()}


class SlopeForm(_ParamForm):
    def __init__(self):
        super().__init__()
        form = QFormLayout(self)
        self._zf   = QDoubleSpinBox(); self._zf.setRange(0.01, 100); self._zf.setValue(DEFAULT_Z_FACTOR)
        self._unit = QComboBox(); self._unit.addItems(["degrees", "percent", "radians"])
        form.addRow("Z-factor:", self._zf)
        form.addRow("Units:", self._unit)

    def params(self):
        return {"z_factor": self._zf.value(), "units": self._unit.currentText()}


class AspectForm(_ParamForm):
    def __init__(self):
        super().__init__()
        form = QFormLayout(self)
        self._zf = QDoubleSpinBox(); self._zf.setRange(0.01, 100); self._zf.setValue(DEFAULT_Z_FACTOR)
        form.addRow("Z-factor:", self._zf)
        lbl = QLabel("Output: 0–360° clockwise from North\n−1 = flat cell")
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color:#888;font-size:10px;")
        form.addRow(lbl)

    def params(self):
        return {"z_factor": self._zf.value()}


class CurvatureForm(_ParamForm):
    def __init__(self):
        super().__init__()
        form = QFormLayout(self)
        self._zf   = QDoubleSpinBox(); self._zf.setRange(0.01, 100); self._zf.setValue(DEFAULT_Z_FACTOR)
        self._type = QComboBox(); self._type.addItems(["profile", "plan", "mean"])
        form.addRow("Z-factor:", self._zf)
        form.addRow("Curvature type:", self._type)

    def params(self):
        return {"z_factor": self._zf.value(), "curv_type": self._type.currentText()}


class MultidirHillshadeForm(_ParamForm):
    def __init__(self):
        super().__init__()
        form = QFormLayout(self)
        self._alt = QDoubleSpinBox(); self._alt.setRange(0, 90); self._alt.setValue(DEFAULT_ALTITUDE)
        self._zf  = QDoubleSpinBox(); self._zf.setRange(0.01, 100); self._zf.setValue(DEFAULT_Z_FACTOR)
        form.addRow("Altitude (°):", self._alt)
        form.addRow("Z-factor:", self._zf)
        form.addRow(QLabel("Azimuths: 225°, 270°, 315°, 360° (Mark 1992)"))

    def params(self):
        return {"altitude": self._alt.value(), "z_factor": self._zf.value()}


class FillSinksForm(_ParamForm):
    def __init__(self):
        super().__init__()
        form = QFormLayout(self)
        self._method = QComboBox(); self._method.addItems(["Priority-Flood", "Breach+Fill"])
        self._eps = QDoubleSpinBox(); self._eps.setDecimals(6); self._eps.setValue(1e-6)
        self._eps.setRange(0, 1)
        form.addRow("Method:", self._method)
        form.addRow("Epsilon:", self._eps)

    def params(self):
        return {"method": self._method.currentText(), "epsilon": self._eps.value()}


class FlowDirForm(_ParamForm):
    def __init__(self):
        super().__init__()
        form = QFormLayout(self)
        self._algo = QComboBox(); self._algo.addItems(["D8", "D-infinity"])
        form.addRow("Algorithm:", self._algo)
        form.addRow(QLabel("D8: single-flow (streams/watersheds)\n"
                           "D∞: multi-flow (TWI/SPI/wetness)"))

    def params(self):
        return {"algorithm": self._algo.currentText()}


class FlowAccumForm(_ParamForm):
    def __init__(self):
        super().__init__()
        form = QFormLayout(self)
        form.addRow(QLabel("Uses the Flow Direction layer for the\nactive DEM (run Flow Direction first)."))

    def params(self):
        return {}


class StreamsForm(_ParamForm):
    def __init__(self):
        super().__init__()
        form = QFormLayout(self)
        self._thr  = QDoubleSpinBox(); self._thr.setRange(1, 1e9); self._thr.setValue(1000)
        self._auto = QCheckBox("Auto threshold (99th percentile)")
        self._auto.setChecked(True)
        form.addRow("Min contrib. area:", self._thr)
        form.addRow(self._auto)

    def params(self):
        return {"auto_threshold": self._auto.isChecked(), "threshold": self._thr.value()}


class TWIForm(_ParamForm):
    def __init__(self):
        super().__init__()
        form = QFormLayout(self)
        self._smin = QDoubleSpinBox(); self._smin.setDecimals(4); self._smin.setValue(0.001)
        self._smin.setRange(1e-6, 1)
        form.addRow("Min slope (rad):", self._smin)
        form.addRow(QLabel("Requires Flow Accumulation (D∞ recommended)."))

    def params(self):
        return {"slope_min": self._smin.value()}


class SPIForm(_ParamForm):
    def __init__(self):
        super().__init__()
        form = QFormLayout(self)
        self._log = QCheckBox("Log scale (ln(SPI+1))"); self._log.setChecked(True)
        form.addRow(self._log)

    def params(self):
        return {"log_scale": self._log.isChecked()}


class TPIForm(_ParamForm):
    def __init__(self):
        super().__init__()
        form = QFormLayout(self)
        self._r    = QSpinBox(); self._r.setRange(1, 200); self._r.setValue(DEFAULT_TPI_RADIUS)
        self._ann  = QCheckBox("Annular window"); self._ann.setChecked(True)
        self._ir   = QSpinBox(); self._ir.setRange(1, 50);  self._ir.setValue(1)
        form.addRow("Outer radius (cells):", self._r)
        form.addRow(self._ann)
        form.addRow("Inner radius (cells):", self._ir)

    def params(self):
        return {
            "radius": self._r.value(),
            "annular": self._ann.isChecked(),
            "inner_radius": self._ir.value(),
        }


class TRIForm(_ParamForm):
    def __init__(self):
        super().__init__()
        QFormLayout(self).addRow(QLabel("Riley et al. (1999)\n3×3 neighbourhood — no parameters."))

    def params(self):
        return {}


class RoughnessForm(_ParamForm):
    def __init__(self):
        super().__init__()
        form = QFormLayout(self)
        self._method = QComboBox()
        self._method.addItems(["VRM", "Range (max-min)", "Std-Dev", "Surface-area ratio"])
        self._win = QSpinBox(); self._win.setRange(3, 51); self._win.setSingleStep(2)
        self._win.setValue(DEFAULT_ROUGHNESS_WINDOW)
        form.addRow("Method:", self._method)
        form.addRow("Window size:", self._win)

    def params(self):
        return {"method": self._method.currentText(), "window": self._win.value()}


class ViewshedForm(_ParamForm):
    def __init__(self):
        super().__init__()
        form = QFormLayout(self)
        self._obs_h  = QDoubleSpinBox(); self._obs_h.setValue(1.8); self._obs_h.setRange(0, 1000)
        self._tgt_h  = QDoubleSpinBox(); self._tgt_h.setValue(0.0); self._tgt_h.setRange(0, 1000)
        self._radius = QSpinBox(); self._radius.setRange(1, 10000); self._radius.setValue(100)
        self._curv   = QCheckBox("Earth curvature + refraction"); self._curv.setChecked(True)
        form.addRow("Observer height (m):", self._obs_h)
        form.addRow("Target height (m):", self._tgt_h)
        form.addRow("Max radius (cells):", self._radius)
        form.addRow(self._curv)
        form.addRow(QLabel("Click on map to set observer point."))

    def params(self):
        return {
            "observer_height": self._obs_h.value(),
            "target_height": self._tgt_h.value(),
            "max_radius": self._radius.value(),
            "correct_curvature": self._curv.isChecked(),
        }


class SVFForm(_ParamForm):
    def __init__(self):
        super().__init__()
        form = QFormLayout(self)
        self._ndirs  = QSpinBox(); self._ndirs.setRange(4, 64); self._ndirs.setValue(DEFAULT_SVF_DIRECTIONS)
        self._radius = QSpinBox(); self._radius.setRange(5, 500); self._radius.setValue(DEFAULT_SVF_RADIUS)
        self._type   = QComboBox()
        self._type.addItems(["SVF", "Positive Openness", "Negative Openness"])
        form.addRow("Directions:", self._ndirs)
        form.addRow("Search radius (cells):", self._radius)
        form.addRow("Output type:", self._type)
        form.addRow(QLabel("Zakšek et al. (2011)"))

    def params(self):
        return {
            "n_directions": self._ndirs.value(),
            "max_radius": self._radius.value(),
            "output_type": self._type.currentText(),
        }


class ProfileForm(_ParamForm):
    def __init__(self):
        super().__init__()
        form = QFormLayout(self)
        self._spacing = QDoubleSpinBox(); self._spacing.setRange(0, 10000)
        self._spacing.setValue(0); self._spacing.setSpecialValueText("Auto (½ cell)")
        form.addRow("Sample spacing (m):", self._spacing)
        form.addRow(QLabel("Click map to add vertices.\nDouble-click to finish."))

    def params(self):
        sp = self._spacing.value()
        return {"sample_spacing": sp if sp > 0 else None}


# ── Registry ────────────────────────────────────────────────────────────────

_TREE_STRUCTURE = [
    ("Primary Derivatives", [
        ("hillshade",         "Hillshade",                HillshadeForm),
        ("slope",             "Slope",                    SlopeForm),
        ("aspect",            "Aspect",                   AspectForm),
        ("curvature",         "Curvature",                CurvatureForm),
        ("multidirectional",  "Multi-dir. Hillshade",     MultidirHillshadeForm),
    ]),
    ("Hydrological Analysis", [
        ("fill_sinks",        "Fill Sinks",               FillSinksForm),
        ("flow_direction",    "Flow Direction",           FlowDirForm),
        ("flow_accumulation", "Flow Accumulation",        FlowAccumForm),
        ("streams",           "Stream Network",           StreamsForm),
    ]),
    ("Terrain Indices", [
        ("twi",               "TWI",                      TWIForm),
        ("spi",               "SPI",                      SPIForm),
        ("tpi",               "TPI",                      TPIForm),
        ("tri",               "TRI",                      TRIForm),
        ("roughness",         "Roughness",                RoughnessForm),
    ]),
    ("Visibility & Illumination", [
        ("viewshed",          "Viewshed",                 ViewshedForm),
        ("svf",               "Sky-View Factor",          SVFForm),
        ("profile",           "Elevation Profile",        ProfileForm),
    ]),
]


class AnalysisPanel(QDockWidget):
    """Dockable analysis control panel."""

    run_analysis    = pyqtSignal(str, dict)   # (product_key, params)
    start_tool      = pyqtSignal(str)         # signal a drawing tool needs activating
    cancel_analysis = pyqtSignal()            # user asked to cancel the running job

    def __init__(self, parent=None):
        super().__init__("Analysis", parent)
        self._forms: Dict[str, _ParamForm] = {}
        self._current_product = None

        self.setMinimumWidth(240)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self._build_ui()

    def _build_ui(self):
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Tree
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setMaximumHeight(260)
        self._tree.itemClicked.connect(self._on_tree_item)

        for group_name, items in _TREE_STRUCTURE:
            group_item = QTreeWidgetItem([group_name])
            font = group_item.font(0)
            font.setBold(True)
            group_item.setFont(0, font)
            for key, label, form_cls in items:
                child = QTreeWidgetItem([f"  {label}"])
                child.setData(0, Qt.ItemDataRole.UserRole, key)
                group_item.addChild(child)
                self._forms[key] = form_cls()
            self._tree.addTopLevelItem(group_item)

        self._tree.expandAll()
        main_layout.addWidget(self._tree)

        # Param area (stacked)
        self._param_label = QLabel("Select an analysis above.")
        self._param_label.setWordWrap(True)
        self._param_label.setStyleSheet("color:#888;padding:8px;")
        main_layout.addWidget(self._param_label)

        self._stack = QStackedWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._stack)
        scroll.setMinimumHeight(160)
        for form in self._forms.values():
            self._stack.addWidget(form)
        main_layout.addWidget(scroll)

        # Action buttons
        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        self._btn_tool    = QPushButton("Activate Tool")
        self._btn_compute = QPushButton("Compute")
        self._btn_compute.setDefault(True)
        self._btn_cancel  = QPushButton("Cancel")
        self._btn_cancel.setVisible(False)
        btn_layout.addWidget(self._btn_tool)
        btn_layout.addWidget(self._btn_compute)
        btn_layout.addWidget(self._btn_cancel)
        main_layout.addWidget(btn_row)

        self._btn_tool.clicked.connect(self._activate_tool)
        self._btn_compute.clicked.connect(self._compute)
        self._btn_cancel.clicked.connect(self._on_cancel)

        # Progress
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        main_layout.addWidget(self._progress)

        # Tool-specific visibility
        self._btn_tool.setVisible(False)

        self.setWidget(container)

    def _on_tree_item(self, item: QTreeWidgetItem, col: int):
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if not key:
            return
        self._current_product = key
        form = self._forms[key]
        self._stack.setCurrentWidget(form)
        self._param_label.hide()

        # Show tool button only for interactive analyses
        needs_tool = key in ("viewshed", "profile")
        self._btn_tool.setVisible(needs_tool)
        self._btn_tool.setText({
            "viewshed": "Pick Observer Point",
            "profile": "Draw Profile Line",
        }.get(key, "Activate Tool"))

    def _activate_tool(self):
        if self._current_product:
            self.start_tool.emit(self._current_product)

    def _compute(self):
        if not self._current_product:
            return
        form = self._forms[self._current_product]
        self.run_analysis.emit(self._current_product, form.params())

    def _on_cancel(self):
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.setText("Cancelling…")
        self.cancel_analysis.emit()

    # ── Running state (called by main window) ──────────────────────────────

    def set_running(self, running: bool):
        """Toggle the panel between idle and 'analysis in progress'."""
        self._btn_compute.setEnabled(not running)
        self._btn_cancel.setVisible(running)
        self._btn_cancel.setEnabled(running)
        self._btn_cancel.setText("Cancel")
        if running:
            # Indeterminate (busy) bar until a real progress value arrives.
            self._progress.setRange(0, 0)
            self._progress.setVisible(True)
        else:
            self._progress.setVisible(False)
            self._progress.setRange(0, 100)
            self._progress.setValue(0)

    def set_progress(self, pct: int):
        """Switch the bar to determinate mode and show real progress."""
        self._progress.setRange(0, 100)
        self._progress.setValue(max(0, min(100, pct)))
        self._progress.setVisible(True)
