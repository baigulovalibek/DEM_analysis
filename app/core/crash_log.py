"""
Crash and diagnostic logging for production bundles.

The PyInstaller build ships with ``console=False``, so stdout/stderr are
silently discarded. Any unhandled Python exception, Qt warning, or
QtWebEngine init failure on a client machine therefore disappears — the
program "just closes" with no clue why.

This module tees stdout/stderr to ``%LOCALAPPDATA%/DEM-Analyst/log.txt``
and installs a :data:`sys.excepthook` plus a Qt message handler so every
diagnostic lands in the same file. Clients can attach the log to a bug
report and the cause is visible without having to re-build with a console.

Wire from ``main.py`` *before* importing Qt so import-time failures are
captured too.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, TextIO


_MAX_LOG_BYTES = 5 * 1024 * 1024     # rotate at 5 MB

_log_path: Optional[Path] = None
_log_fh: Optional[TextIO] = None


def log_path() -> Optional[Path]:
    """Location of the active log file, or None if logging wasn't installed."""
    return _log_path


def _resolve_log_dir() -> Path:
    """Pick a per-user log directory that follows each OS's convention.

    * Windows : ``%LOCALAPPDATA%\\DEM-Analyst``  (falls back to ``%APPDATA%``)
    * macOS   : ``~/Library/Logs/DEM-Analyst``
    * Linux   : ``$XDG_STATE_HOME/dem-analyst`` (else ``~/.local/state/dem-analyst``)
    * Other   : ``~/.dem-analyst`` (legacy fallback)
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        d = Path(base) / "DEM-Analyst" if base else Path.home() / ".dem-analyst"
    elif sys.platform == "darwin":
        d = Path.home() / "Library" / "Logs" / "DEM-Analyst"
    elif sys.platform.startswith("linux"):
        base = os.environ.get("XDG_STATE_HOME")
        d = Path(base) / "dem-analyst" if base else Path.home() / ".local" / "state" / "dem-analyst"
    else:
        d = Path.home() / ".dem-analyst"
    d.mkdir(parents=True, exist_ok=True)
    return d


class _Tee:
    """Write-through duplicator. Failures on either stream are swallowed
    — we never want diagnostics to themselves raise."""

    def __init__(self, *streams):
        self._streams = [s for s in streams if s is not None]

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass
        self.flush()
        return len(data) if isinstance(data, str) else 0

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return False

    # Some libraries probe these.
    def writable(self):
        return True

    def readable(self):
        return False

    def fileno(self):
        raise OSError("Tee stream has no underlying fileno")


def install() -> Path:
    """Tee stdout/stderr to log.txt and install :data:`sys.excepthook`.

    Idempotent — calling twice is a no-op. Returns the active log path.
    """
    global _log_path, _log_fh

    if _log_fh is not None and _log_path is not None:
        return _log_path

    log_dir = _resolve_log_dir()
    path = log_dir / "log.txt"

    # Cheap rotation so the log doesn't grow unbounded.
    try:
        if path.exists() and path.stat().st_size > _MAX_LOG_BYTES:
            path.replace(path.with_suffix(".prev.txt"))
    except OSError:
        pass

    try:
        fh = open(path, "a", encoding="utf-8", buffering=1)
    except OSError:
        # If we can't write the log, fall back silently to plain stdio.
        return path

    fh.write("\n" + "=" * 60 + "\n")
    fh.write(f"DEM Analyst session started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    fh.write(f"Python:    {sys.version.splitlines()[0]}\n")
    fh.write(f"Platform:  {sys.platform}\n")
    fh.write(f"Frozen:    {getattr(sys, 'frozen', False)}\n")
    fh.write(f"_MEIPASS:  {getattr(sys, '_MEIPASS', '<unset>')}\n")
    fh.write(f"Executable:{sys.executable}\n")
    fh.write(f"CWD:       {os.getcwd()}\n")
    fh.write("=" * 60 + "\n")
    fh.flush()

    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)

    def _excepthook(exc_type, exc_value, exc_tb):
        try:
            print("\n--- UNHANDLED EXCEPTION ---", file=sys.stderr)
            traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
            print("--- END EXCEPTION ---\n", file=sys.stderr)
        except Exception:
            pass

    sys.excepthook = _excepthook

    _log_path = path
    _log_fh = fh
    return path


def install_qt_handler() -> None:
    """Route Qt's own warnings/criticals into the log.

    Must be called after QApplication is constructed (Qt requires this).
    Failures are non-fatal — production logging keeps working without it.
    """
    try:
        from PyQt6.QtCore import qInstallMessageHandler, QtMsgType
    except Exception as exc:
        print(f"[crash_log] Qt message handler unavailable: {exc}",
              file=sys.stderr)
        return

    _LEVELS = {
        QtMsgType.QtDebugMsg:    "DEBUG",
        QtMsgType.QtInfoMsg:     "INFO",
        QtMsgType.QtWarningMsg:  "WARN",
        QtMsgType.QtCriticalMsg: "CRIT",
        QtMsgType.QtFatalMsg:    "FATAL",
    }

    def _handler(msg_type, context, message):
        label = _LEVELS.get(msg_type, "?")
        try:
            print(f"[Qt {label}] {message}", file=sys.stderr)
        except Exception:
            pass

    qInstallMessageHandler(_handler)
