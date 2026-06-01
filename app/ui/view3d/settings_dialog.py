"""
Scene-level settings dialog.

Houses the bits that don't belong in the always-on side panel: background
colour, skirt height, eye-dome lighting toggle, near/far clip override.

Most users won't touch this; QGIS calls the equivalent its "Configure" panel.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QPushButton, QDoubleSpinBox, QCheckBox,
    QColorDialog, QHBoxLayout, QDialogButtonBox, QLabel, QWidget,
)

from app.core.scene3d import RenderSettings


class _ColorButton(QPushButton):
    """A small button whose face shows the currently-selected colour."""

    color_picked = pyqtSignal(tuple)        # (r, g, b) in 0..1

    def __init__(self, initial: tuple[float, float, float], parent=None):
        super().__init__(parent)
        self.setFixedSize(60, 22)
        self._color = initial
        self._update_face()
        self.clicked.connect(self._pick)

    def _update_face(self):
        r, g, b = (int(c * 255) for c in self._color)
        self.setStyleSheet(
            f"QPushButton {{ background:rgb({r},{g},{b}); border:1px solid #444; }}"
        )

    def _pick(self):
        r, g, b = self._color
        initial = QColor(int(r * 255), int(g * 255), int(b * 255))
        col = QColorDialog.getColor(initial, self, "Choose colour")
        if col.isValid():
            self._color = (col.red() / 255.0, col.green() / 255.0, col.blue() / 255.0)
            self._update_face()
            self.color_picked.emit(self._color)

    def value(self) -> tuple[float, float, float]:
        return self._color


class SceneSettingsDialog(QDialog):
    """Modal-ish dialog editing a ``RenderSettings`` in-place.

    Changes are applied live (no Apply button); a Close button just dismisses
    the dialog.  The owning ``View3DWindow`` is responsible for calling
    ``viewport.update()`` after each change via the emitted signal.
    """

    setting_changed = pyqtSignal()

    def __init__(self, settings: RenderSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D Scene Settings")
        self._settings = settings

        layout = QVBoxLayout(self)

        form = QFormLayout()
        layout.addLayout(form)

        # Background colour
        self._bg_btn = _ColorButton(settings.background, self)
        self._bg_btn.color_picked.connect(self._on_bg)
        form.addRow("Background:", self._bg_btn)

        # Skirt height — visual only; helps the boundary not float in space.
        self._skirt_spin = QDoubleSpinBox()
        self._skirt_spin.setRange(0.0, 5000.0)
        self._skirt_spin.setSingleStep(10.0)
        self._skirt_spin.setSuffix(" m")
        self._skirt_spin.setValue(settings.skirt_height_m)
        # Commit only on Enter/Tab/focus-out — typing "1000" otherwise rebuilds
        # the 3D scene three times.
        self._skirt_spin.setKeyboardTracking(False)
        self._skirt_spin.valueChanged.connect(self._on_skirt)
        form.addRow("Skirt height:", self._skirt_spin)

        # Eye-dome lighting
        self._edl_check = QCheckBox()
        self._edl_check.setChecked(settings.eye_dome)
        self._edl_check.toggled.connect(self._on_edl)
        form.addRow("Eye-dome lighting:", self._edl_check)

        self._edl_spin = QDoubleSpinBox()
        self._edl_spin.setRange(0.0, 2.0)
        self._edl_spin.setSingleStep(0.1)
        self._edl_spin.setValue(settings.eye_dome_strength)
        self._edl_spin.setKeyboardTracking(False)
        self._edl_spin.valueChanged.connect(self._on_edl_strength)
        form.addRow("EDL strength:", self._edl_spin)

        hint = QLabel(
            "Eye-dome lighting accentuates ridges and breaks by darkening "
            "edges (post-process). v1 stub — not yet active."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888; font-size:10px;")
        layout.addWidget(hint)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        layout.addWidget(btns)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_bg(self, rgb: tuple):
        self._settings.background = rgb
        self.setting_changed.emit()

    def _on_skirt(self, v: float):
        self._settings.skirt_height_m = float(v)
        self.setting_changed.emit()

    def _on_edl(self, on: bool):
        self._settings.eye_dome = bool(on)
        self.setting_changed.emit()

    def _on_edl_strength(self, v: float):
        self._settings.eye_dome_strength = float(v)
        self.setting_changed.emit()
