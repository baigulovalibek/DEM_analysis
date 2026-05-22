"""
Background QThread worker for running analysis functions without blocking the UI.

The worker carries the originating request context (product / source DEM /
params) so the main window can connect plain *bound-method* slots to its
signals.  Connecting bound methods of QObjects guarantees the result is
delivered to the GUI thread via a queued connection — connecting lambdas
instead can run the slot in the worker thread, which corrupts Qt state.
"""
from __future__ import annotations
import inspect
import traceback
from typing import Callable, Any, Optional

from PyQt6.QtCore import QThread, pyqtSignal


class AnalysisWorker(QThread):
    """
    Runs ``func(**kwargs)`` in a background thread and emits the result.

    Signals
    -------
    progress(int)  : 0-100, only emitted by functions that accept ``_progress``
    result(object) : the function's return value (numpy array, tuple, ...)
    error(str)     : human-readable error message
    """
    progress = pyqtSignal(int)
    result = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        func: Callable,
        kwargs: dict,
        *,
        product: str = "",
        dem: Any = None,
        params: Optional[dict] = None,
    ):
        super().__init__()
        self._func = func
        self._kwargs = dict(kwargs)
        self._cancelled = False

        # Request context — read back by the result handler.
        self.product = product
        self.dem = dem
        self.params = params or {}

    # ── Cancellation ───────────────────────────────────────────────────────

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    # ── Thread body ────────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            try:
                sig = inspect.signature(self._func)
                if "_progress" in sig.parameters:
                    self._kwargs["_progress"] = self._emit_progress
            except (ValueError, TypeError):
                pass

            out = self._func(**self._kwargs)
        except Exception as exc:
            if not self._cancelled:
                traceback.print_exc()
                self.error.emit(f"{type(exc).__name__}: {exc}")
            return

        if not self._cancelled:
            self.result.emit(out)

    def _emit_progress(self, pct: int) -> bool:
        """Called from inside analysis functions; returns True if cancelled."""
        self.progress.emit(int(max(0, min(100, pct))))
        return self._cancelled
