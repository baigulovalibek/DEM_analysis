"""
DEM Analyst — entry point.

Usage:
    python main.py [dem_file]

If a DEM path is passed as an argument it will be opened on startup.
"""
from __future__ import annotations
import sys
import os
from pathlib import Path

# Allow running from any working directory
os.chdir(Path(__file__).parent)

# Qt WebEngine must be initialised before QApplication on some platforms
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QCoreApplication

QCoreApplication.setApplicationName("DEM Analyst")
QCoreApplication.setOrganizationName("DEMAnalyst")
# Must be set BEFORE QApplication is constructed
QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

app = QApplication(sys.argv)

from app.main_window import MainWindow

window = MainWindow()
window.show()

# Optional: open a DEM passed on the command line
if len(sys.argv) > 1:
    path = Path(sys.argv[1])
    if path.exists():
        window._load_dem_file(path)

sys.exit(app.exec())
