# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for DEM Analyst (Windows onedir build).

Goals:
  - Bundle PyQt6 + QtWebEngine + Leaflet assets + rasterio (GDAL/PROJ data)
    so the result is fully self-contained and "portable".
  - Trim the on-disk footprint by excluding every Qt6 submodule and scipy
    subpackage the app does not import.

Build from project root:
    env\Scripts\pyinstaller packaging\DEM-Analyst.spec --noconfirm --clean

Notes:
  - UPX is intentionally disabled: it corrupts QtWebEngineProcess.exe and
    several Qt6 DLLs on Windows.
  - onedir (not onefile): a folder ships ~3x faster on every launch than the
    self-extracting onefile flavour, which matters a lot when QtWebEngine
    alone is ~250 MB.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SPEC_DIR = Path(os.path.abspath(SPECPATH))
PROJECT_ROOT = SPEC_DIR.parent
ENTRY = str(PROJECT_ROOT / "main.py")

# --------------------------------------------------------------------------- #
# Data files
# --------------------------------------------------------------------------- #
# Application resources (Leaflet, map.html, Qt stylesheet, GLSL shaders).
#
# IMPORTANT: PyInstaller only bundles ``.py`` files inside packages by
# default — every non-Python asset must be listed here explicitly or the
# packaged exe will hit FileNotFoundError when it tries to open it.
datas = [
    (str(PROJECT_ROOT / "resources"), "resources"),
    (str(PROJECT_ROOT / "app" / "ui" / "style.qss"), "app/ui"),
    # 3D viewport shaders — opened by gl_widget.py at GL init time.
    # Missing these is a guaranteed crash the moment the user opens 3D View.
    (str(PROJECT_ROOT / "app" / "ui" / "view3d" / "shaders"),
     "app/ui/view3d/shaders"),
]

# rasterio ships GDAL_DATA and PROJ_DATA as package data; the hook usually
# picks them up but we add them explicitly so a missing hook does not silently
# break CRS reprojection at runtime.
datas += collect_data_files("rasterio", subdir="gdal_data")
datas += collect_data_files("rasterio", subdir="proj_data")

# Hidden imports — modules PyInstaller's static analysis misses because
# they are imported lazily (inside functions) or behind ``try/except``.
hiddenimports = []
hiddenimports += collect_submodules("rasterio")
hiddenimports += ["rasterio._shim", "rasterio.vrt", "rasterio.sample",
                  "rasterio.crs", "rasterio.transform", "rasterio.warp"]
hiddenimports += ["scipy.ndimage"]
hiddenimports += ["pyqtgraph.opengl"]
# pandas is imported INSIDE methods in app/core/earthquakes.py — PyInstaller's
# scanner sometimes misses deferred imports, and openpyxl is pandas' xlsx
# backend (also lazy-imported when the user picks a .xlsx catalog).
hiddenimports += ["pandas", "openpyxl"]
# PyOpenGL is imported behind a try/except in app/ui/view3d/gl_widget.py.
# Without explicit hidden imports PyInstaller can skip it, and then the 3D
# viewport silently disables itself with a misleading "driver missing" hint.
# OpenGL.platform.* is OS-specific — the wrong submodule listed here is
# benign on the build host (still produces a working bundle for that OS)
# but adds dead weight, so pick the one that matches sys.platform.
hiddenimports += [
    "OpenGL",
    "OpenGL.GL",
    "OpenGL.GL.shaders",
    "OpenGL.arrays",
    "OpenGL.arrays.ctypesarrays",
    "OpenGL.arrays.ctypesparameters",
    "OpenGL.arrays.ctypespointers",
    "OpenGL.arrays.lists",
    "OpenGL.arrays.numbers",
    "OpenGL.arrays.numpymodule",
    "OpenGL.arrays.strings",
]
if sys.platform == "win32":
    hiddenimports += ["OpenGL.platform.win32"]
elif sys.platform == "darwin":
    hiddenimports += ["OpenGL.platform.darwin"]
else:
    hiddenimports += ["OpenGL.platform.glx", "OpenGL.platform.egl"]

# GDAL DLLs that rasterio loads at runtime.
binaries = []
binaries += collect_dynamic_libs("rasterio")

# --------------------------------------------------------------------------- #
# Exclusions — the main size win
# --------------------------------------------------------------------------- #
# Qt submodules the app does not import. Each one drops 5-30 MB of DLLs +
# resources from the bundle. Verified safe against the import map in CLAUDE.md
# (only QtCore, QtGui, QtWidgets, QtOpenGLWidgets, QtWebChannel,
# QtWebEngineCore, QtWebEngineWidgets are actually used).
QT_EXCLUDES = [
    "PyQt6.QtBluetooth",
    "PyQt6.QtDBus",
    "PyQt6.QtDesigner",
    "PyQt6.QtHelp",
    "PyQt6.QtMultimedia",
    "PyQt6.QtMultimediaWidgets",
    "PyQt6.QtNfc",
    "PyQt6.QtPdf",
    "PyQt6.QtPdfWidgets",
    "PyQt6.QtPositioning",
    "PyQt6.QtQml",
    "PyQt6.QtQuick",
    "PyQt6.QtQuick3D",
    "PyQt6.QtQuickWidgets",
    "PyQt6.QtRemoteObjects",
    "PyQt6.QtSensors",
    "PyQt6.QtSerialPort",
    "PyQt6.QtSpatialAudio",
    "PyQt6.QtSql",
    "PyQt6.QtSvg",
    "PyQt6.QtSvgWidgets",
    "PyQt6.QtTest",
    "PyQt6.QtTextToSpeech",
    "PyQt6.QtWebChannelQuick",
    "PyQt6.QtWebSockets",
    "PyQt6.QtXml",
]

# scipy: only ndimage is used.
SCIPY_EXCLUDES = [
    "scipy.cluster", "scipy.constants", "scipy.datasets", "scipy.fft",
    "scipy.fftpack", "scipy.integrate", "scipy.interpolate", "scipy.io",
    "scipy.linalg.cython_lapack", "scipy.misc", "scipy.optimize", "scipy.odr",
    "scipy.signal", "scipy.sparse.csgraph", "scipy.spatial",
    "scipy.special.cython_special", "scipy.stats",
]

# Generic dead weight.
# NB: do NOT exclude "unittest" — pyparsing.testing (pulled by matplotlib)
# imports it unconditionally and the bundle will crash on first import of
# matplotlib.
# NB: do NOT exclude "distutils" — PyInstaller's own pre-import hook aliases
# the setuptools-vendored copy and an exclusion makes the build crash.
# NB: do NOT exclude "pydoc"/"pydoc_data"/"doctest" — scipy._lib._docscrape
# (pulled in by `import scipy.ndimage`) imports pydoc unconditionally. Excluding
# it makes EVERY analysis silently no-op: _resolve_analysis imports scipy.ndimage
# on the GUI thread, the ModuleNotFoundError escapes the Qt slot, and the
# Compute button appears dead with no error dialog. These stdlib modules are
# tiny — not worth the breakage.
GENERIC_EXCLUDES = [
    "tkinter", "_tkinter", "tcl", "tk", "turtle", "turtledemo",
    "xmlrpc", "lib2to3",
    "matplotlib.tests", "numpy.tests", "scipy.tests",
    "IPython", "jupyter", "notebook", "sphinx",
    # nope on bare "test"/"tests" — those names appear as the stdlib top-level
    # package and several third-party libs reach into a test_xxx module.
]

excludes = QT_EXCLUDES + SCIPY_EXCLUDES + GENERIC_EXCLUDES

# --------------------------------------------------------------------------- #
# Build pipeline
# --------------------------------------------------------------------------- #
block_cipher = None

a = Analysis(
    [ENTRY],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# --------------------------------------------------------------------------- #
# Post-Analysis filtering — strip bytes the hooks pulled in but the app never
# touches at runtime. This is where the bulk of the size savings live.
# --------------------------------------------------------------------------- #
KEEP_LOCALES = {"en-us.pak", "en-gb.pak"}

def _drop(entry):
    dest = entry[0].replace("\\", "/").lower()
    # matplotlib/numpy/scipy test fixtures
    if "/tests/" in dest or dest.endswith("/tests"):
        return True
    if "sample_data" in dest:
        return True
    # Qt WebEngine devtools UI + every *.debug.pak/.bin variant (debug-only).
    if dest.endswith(".debug.pak") or dest.endswith(".debug.bin"):
        return True
    if dest.endswith("qtwebengine_devtools_resources.pak"):
        return True
    # Chromium locale paks: keep only English. Path looks like
    # 'PyQt6/Qt6/translations/qtwebengine_locales/de.pak'.
    if "/qtwebengine_locales/" in dest:
        name = dest.rsplit("/", 1)[-1]
        if name not in KEEP_LOCALES:
            return True
    # Qt UI translation .qm files: ditto.
    if "/translations/" in dest and dest.endswith(".qm"):
        name = dest.rsplit("/", 1)[-1]
        # qtbase_en.qm / qtwebengine_en.qm etc.
        if "_en" not in name:
            return True
    return False

before = len(a.datas)
a.datas = [d for d in a.datas if not _drop(d)]
print(f"[spec] filtered {before - len(a.datas)} data entries "
      f"(devtools, debug paks, non-English locales, test fixtures)")

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DEM-Analyst",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                      # UPX corrupts QtWebEngine — keep off.
    console=False,                  # GUI app, no console window.
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,                      # add a .ico here later if you have one.
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="DEM-Analyst",
)
