"""
Elevation profile chart dialog.
"""
from __future__ import annotations
import numpy as np
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDialogButtonBox

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigCanvas
from matplotlib.figure import Figure

from app.core.visibility.profile import ElevationProfile


class ProfileDialog(QDialog):
    def __init__(self, profile: ElevationProfile, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Elevation Profile")
        self.resize(700, 380)
        self._build(profile)

    def _build(self, p: ElevationProfile):
        layout = QVBoxLayout(self)

        # Stats row
        stats_row = QHBoxLayout()
        for label, value in [
            ("Length:", f"{p.total_length:,.0f} m"),
            ("Min elev:", f"{p.min_elev:,.1f} m"),
            ("Max elev:", f"{p.max_elev:,.1f} m"),
            ("Ascent:", f"+{p.total_ascent:,.0f} m"),
            ("Descent:", f"−{p.total_descent:,.0f} m"),
        ]:
            lbl = QLabel(f"<b>{label}</b> {value}")
            lbl.setStyleSheet("margin-right:16px;")
            stats_row.addWidget(lbl)
        stats_row.addStretch()
        layout.addLayout(stats_row)

        # Chart
        fig = Figure(figsize=(6, 2.5), facecolor="#1e1e1e")
        ax  = fig.add_subplot(111)
        ax.set_facecolor("#252525")

        dist_km = p.distances / 1000.0
        ax.fill_between(dist_km, p.elevations, alpha=0.35, color="#5599dd")
        ax.plot(dist_km, p.elevations, color="#80bbff", linewidth=1.2)

        ax.set_xlabel("Distance (km)", color="#aaa", fontsize=9)
        ax.set_ylabel("Elevation (m)", color="#aaa", fontsize=9)
        ax.tick_params(colors="#aaa", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        fig.tight_layout()

        canvas = FigCanvas(fig)
        layout.addWidget(canvas)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.accept)
        layout.addWidget(buttons)
