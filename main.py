"""
DEM Analyst — entry point.

Usage:
    python main.py [dem_file]

If a DEM path is passed as an argument it will be opened on startup.

Environment overrides (handy when shipping bundles to clients):

    DEM_ANALYST_SAFE_GFX=1
        Disable Chromium GPU acceleration and the software rasteriser.
        Use when QtWebEngine fails to initialise on a client machine
        (symptom: blank/grey map area). Pure CPU rendering — slower
        but works on locked-down or driver-broken Windows boxes.

    DEM_ANALYST_FORCE_MPL=1
        Force the matplotlib 3D backend (legacy — kept for parity with
        older builds; the OpenGL viewport is the default now).

Logs land in ``%LOCALAPPDATA%/DEM-Analyst/log.txt`` so a client running
a packaged bundle can attach the file to a bug report — see
``app/core/crash_log.py``.
"""
from __future__ import annotations
import sys
import os
from pathlib import Path

# Allow running from any working directory
os.chdir(Path(__file__).parent)

# Install the crash logger BEFORE anything else so import-time failures land
# in the log file. With console=False bundles this is the only diagnostic
# channel back to the developer.
from app.core.crash_log import install as _install_crash_log, install_qt_handler
_LOG_PATH = _install_crash_log()

# Qt WebEngine must be initialised before QApplication on some platforms.
# DEM_ANALYST_SAFE_GFX=1 forces a CPU rendering path — clients reporting a
# blank map almost always have a Chromium GPU init failure underneath, and
# this env var is the one-line workaround you can ship without rebuilding.
# Users can also flip the persistent QSettings flag from Help → Safe Graphics
# Mode; we read that here (before QApplication exists) so the next launch
# honours it without needing the env var on the client's machine.
_safe_gfx_env = os.environ.get("DEM_ANALYST_SAFE_GFX", "").strip().lower() in (
    "1", "true", "yes"
)
_safe_gfx_off_env = os.environ.get("DEM_ANALYST_SAFE_GFX", "").strip().lower() in (
    "0", "false", "no", "off"
)
_safe_gfx_persistent = False
try:
    from PyQt6.QtCore import QSettings, QCoreApplication
    QCoreApplication.setApplicationName("DEM Analyst")
    QCoreApplication.setOrganizationName("DEMAnalyst")
    # ``safe_gfx`` defaults to ``None`` so we can distinguish "never touched"
    # from "explicitly disabled". Frozen bundles default to safe-gfx ON
    # because the cost (CPU rendering for a Leaflet page) is negligible and
    # the win (no GPU init failures on locked-down or driver-broken client
    # PCs) is the single biggest source of "blank map" reports in the field.
    _persisted = QSettings().value("safe_gfx", None)
    if _persisted is not None:
        _safe_gfx_persistent = str(_persisted).strip().lower() in ("1", "true", "yes")
        _safe_gfx_choice_known = True
    else:
        _safe_gfx_choice_known = False
except Exception as exc:
    print(f"[main] could not read safe_gfx setting: {exc}", file=sys.stderr)
    _safe_gfx_choice_known = False

# Bundle default: if the user hasn't explicitly chosen, AND we're running
# from a PyInstaller bundle, default to safe-gfx ON. Dev runs from source
# keep the default OFF so developers see real GPU-accelerated rendering.
_is_frozen = bool(getattr(sys, "frozen", False))
_safe_gfx_default = _is_frozen and not _safe_gfx_choice_known

# Final decision: env var wins (both directions), else persisted, else default.
if _safe_gfx_off_env:
    _safe_gfx = False
elif _safe_gfx_env:
    _safe_gfx = True
elif _safe_gfx_choice_known:
    _safe_gfx = _safe_gfx_persistent
else:
    _safe_gfx = _safe_gfx_default

# --no-sandbox is needed on Linux when the host lacks the unprivileged-userns
# capability Chromium expects (common on hardened distros / containers) and on
# Windows where the Qt WebEngine sandbox sometimes refuses to start on locked-
# down corporate machines. On macOS the system sandbox does the right thing —
# adding --no-sandbox there yields a startup warning and reduces security.
_chromium_flags_parts: list[str] = []
if sys.platform != "darwin":
    _chromium_flags_parts.append("--no-sandbox")
if _safe_gfx:
    _chromium_flags_parts += [
        "--disable-gpu", "--disable-gpu-compositing", "--disable-software-rasterizer",
    ]
_chromium_flags = " ".join(_chromium_flags_parts)
print(
    f"[main] Safe Graphics Mode {'ON' if _safe_gfx else 'OFF'} "
    f"(env={_safe_gfx_env}, env_off={_safe_gfx_off_env}, "
    f"persistent={_safe_gfx_persistent if _safe_gfx_choice_known else '<unset>'}, "
    f"frozen={_is_frozen})",
    file=sys.stderr,
)
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", _chromium_flags)

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QCoreApplication

QCoreApplication.setApplicationName("DEM Analyst")
QCoreApplication.setOrganizationName("DEMAnalyst")
# Must be set BEFORE QApplication is constructed
QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

# Request a 3.3 core GL context for the 3D view.  This must happen before any
# QOpenGLWidget is created, which means before MainWindow is imported.
from app.ui.view3d import set_default_format
set_default_format()

app = QApplication(sys.argv)

# Route Qt's own warnings/criticals into the log file (requires QApplication
# to exist first).
install_qt_handler()
print(f"[main] log file: {_LOG_PATH}", file=sys.stderr)

# Verbose startup dump: when a client reports "doesn't work on my PC", we
# need enough context to diagnose without remote access. This block runs
# once per launch and lands in log.txt next to anything Qt complains about.
def _dump_startup_env():
    import platform
    try:
        from PyQt6.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
    except Exception:
        QT_VERSION_STR = PYQT_VERSION_STR = "<unknown>"
    print("[startup] ----- environment dump -----", file=sys.stderr)
    print(f"[startup] os:           {platform.platform()}", file=sys.stderr)
    print(f"[startup] machine:      {platform.machine()}", file=sys.stderr)
    print(f"[startup] python:       {sys.version.splitlines()[0]}", file=sys.stderr)
    print(f"[startup] Qt:           {QT_VERSION_STR}  PyQt: {PYQT_VERSION_STR}",
          file=sys.stderr)
    print(f"[startup] frozen:       {_is_frozen}", file=sys.stderr)
    print(f"[startup] _MEIPASS:     {getattr(sys, '_MEIPASS', '<unset>')}",
          file=sys.stderr)
    print(f"[startup] safe_gfx:     {_safe_gfx} "
          f"(env={_safe_gfx_env}, env_off={_safe_gfx_off_env}, "
          f"persistent_set={_safe_gfx_choice_known}, frozen_default={_safe_gfx_default})",
          file=sys.stderr)
    print(f"[startup] chromium:     {os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS')}",
          file=sys.stderr)
    print(f"[startup] cwd:          {os.getcwd()}", file=sys.stderr)
    # QtWebEngineProcess helper — if missing or quarantined by AV, the map
    # area will be permanently blank no matter what the user does. Binary
    # name differs by OS, so probe the right one.
    try:
        from PyQt6.QtCore import QLibraryInfo
        libpath = QLibraryInfo.path(QLibraryInfo.LibraryPath.LibraryExecutablesPath)
        if sys.platform == "win32":
            qtwep_candidates = [Path(libpath) / "QtWebEngineProcess.exe"]
        elif sys.platform == "darwin":
            qtwep_candidates = [
                Path(libpath) / "QtWebEngineProcess.app" /
                    "Contents" / "MacOS" / "QtWebEngineProcess",
                Path(libpath) / "QtWebEngineProcess",
            ]
        else:
            qtwep_candidates = [Path(libpath) / "QtWebEngineProcess"]
        print(f"[startup] WebEngine:    {libpath}", file=sys.stderr)
        for qtwep in qtwep_candidates:
            print(f"[startup] WebEngine bin: {qtwep}  exists={qtwep.exists()}",
                  file=sys.stderr)
    except Exception as exc:
        print(f"[startup] WebEngine probe failed: {exc}", file=sys.stderr)
    # Optional GPU stack probe — best-effort, never fatal.
    try:
        import ctypes
        is_admin = bool(ctypes.windll.shell32.IsUserAnAdmin()) if sys.platform == "win32" else False
        print(f"[startup] admin:        {is_admin}", file=sys.stderr)
    except Exception:
        pass
    # The user's QSettings backing file — useful when "the toggle doesn't
    # stick" reports come in (usually because of roaming-profile redirection).
    try:
        from PyQt6.QtCore import QSettings
        s = QSettings()
        print(f"[startup] settings file: {s.fileName()}", file=sys.stderr)
    except Exception:
        pass
    print("[startup] -----------------------------", file=sys.stderr)

_dump_startup_env()

from app.main_window import MainWindow

window = MainWindow()
window.show()

# Optional: open a DEM passed on the command line
if len(sys.argv) > 1:
    path = Path(sys.argv[1])
    if path.exists():
        window._load_dem_file(path)

sys.exit(app.exec())
