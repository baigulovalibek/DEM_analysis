"""
Background QThread worker for running analysis functions without blocking the UI.

The worker carries the originating request context (product / source DEM /
params) so the main window can connect plain *bound-method* slots to its
signals.  Connecting bound methods of QObjects guarantees the result is
delivered to the GUI thread via a queued connection — connecting lambdas
instead can run the slot in the worker thread, which corrupts Qt state.

Progress signals are throttled to one emission per ``_MIN_EMIT_INTERVAL``
seconds (default 50 ms = 20 Hz).  A naive callback that emits on every
percent point can flood the GUI thread on routines that report progress
hundreds of times per analysis (viewshed, SVF), starving the event loop
for repaint and user input.  20 Hz is the cap mainstream Qt apps use for
progress bars — granular enough to feel live, infrequent enough to keep
the GUI thread breathing.
"""
from __future__ import annotations
import inspect
import time
import traceback
from typing import Callable, Any, Optional

from PyQt6.QtCore import QThread, pyqtSignal


_MIN_EMIT_INTERVAL = 0.05   # 50 ms — 20 Hz cap


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

        # Progress throttling state.
        self._last_emit_ts = 0.0
        self._last_emit_pct = -1

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
            # Force a final 100% emission so the bar lands on full when
            # the analysis completes — the throttle would otherwise
            # swallow the last tick if it fired inside the 50 ms window.
            self.progress.emit(100)
            self.result.emit(out)

    def _emit_progress(self, pct: int) -> bool:
        """Throttled progress relay.

        Suppresses emissions that arrive within :data:`_MIN_EMIT_INTERVAL`
        of the previous one *and* report no change in percentage.
        Returns True when the caller has cancelled — so analysis routines
        can short-circuit out of their inner loops.
        """
        pct = int(max(0, min(100, pct)))
        now = time.monotonic()
        # 0 % and 100 % are special — always emit so the bar visibly
        # starts and ends without waiting for the throttle window.
        if pct in (0, 100) or pct != self._last_emit_pct:
            if pct in (0, 100) or now - self._last_emit_ts >= _MIN_EMIT_INTERVAL:
                self.progress.emit(pct)
                self._last_emit_ts = now
                self._last_emit_pct = pct
        return self._cancelled
