"""
Background QThread worker for running analysis functions without blocking the UI.
"""
from __future__ import annotations
import inspect
from typing import Callable, Any

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


class AnalysisWorker(QThread):
    """
    Runs `func(*args, **kwargs)` in a background thread.
    The function must return a numpy array (the result) or raise an exception.
    """
    progress = pyqtSignal(int)          # 0–100
    result = pyqtSignal(object)         # np.ndarray or dict of arrays
    error = pyqtSignal(str)
    finished_ok = pyqtSignal()

    def __init__(
        self,
        func: Callable,
        *args,
        progress_callback: bool = False,
        **kwargs,
    ):
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self._progress_callback = progress_callback
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            if self._progress_callback:
                try:
                    sig = inspect.signature(self._func)
                    if "_progress" in sig.parameters:
                        self._kwargs["_progress"] = self._emit_progress
                except (ValueError, TypeError):
                    pass
            out = self._func(*self._args, **self._kwargs)
            if not self._cancelled:
                self.result.emit(out)
                self.finished_ok.emit()
        except Exception as exc:
            if not self._cancelled:
                self.error.emit(str(exc))

    def _emit_progress(self, pct: int) -> bool:
        """Called from inside analysis functions; returns True if cancelled."""
        self.progress.emit(pct)
        return self._cancelled
