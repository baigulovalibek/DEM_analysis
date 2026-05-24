"""
Numba JIT shim with a graceful pure-Python fallback.

Numba is an *optional* runtime dependency: install it (``pip install numba``)
for the 20-80x speedups on the hydrology kernels documented in
``OPTIMIZATION.md`` Phase B; without it the app still runs correctly,
just slower.  All hot-loop kernels in :mod:`app.core.hydrology` and
:mod:`app.core.visibility.viewshed` use the symbols re-exported here
(``njit``, ``prange``, ``JIT_ENABLED``) so they pick up the JIT
transparently when it's available.

A note on cold-compile cost
---------------------------
Numba JIT-compiles a kernel the first time it runs.  Each ``@njit``
specialisation costs ~1-3 s of compile time.  We pass ``cache=True``
on every decorator so the compiled artifact lands in the on-disk
PyCache and the second-and-later runs skip compilation entirely.

The first analysis after a fresh install is therefore visibly slower
than subsequent ones; this is normal.  Set
``NUMBA_DISABLE_JIT=1`` in the environment to force the Python
fallback at runtime (handy for debugging).
"""
from __future__ import annotations

import os
from typing import Any, Callable


# ``NUMBA_DISABLE_JIT=1`` is a Numba convention; we respect it here too so
# the same env var skips both the import and the cached kernels.
_FORCE_DISABLE = os.environ.get("NUMBA_DISABLE_JIT", "").strip() in ("1", "true", "yes")

JIT_ENABLED: bool = False

if not _FORCE_DISABLE:
    try:
        from numba import njit as _real_njit, prange as _real_prange  # type: ignore
        JIT_ENABLED = True
    except Exception:
        JIT_ENABLED = False


if JIT_ENABLED:
    # Re-export the real decorator and parallel-range.  All kernels use
    # ``@njit(cache=True)`` so first-run compile is paid once per machine.
    njit = _real_njit
    prange = _real_prange  # noqa: F401  (re-exported)
else:
    # Pure-Python fallback.  ``@njit(...)`` and ``@njit`` should both work
    # as decorators in a Numba-free environment, so the shim accepts
    # arbitrary arguments and returns the function unchanged.
    def _identity_decorator(fn: Callable) -> Callable:
        return fn

    def njit(*args: Any, **kwargs: Any):  # type: ignore[override]
        # Two call patterns:
        #   @njit            -> args=(func,), kwargs={}
        #   @njit(cache=True) -> args=(),     kwargs={'cache': True}
        if len(args) == 1 and not kwargs and callable(args[0]):
            return args[0]
        return _identity_decorator

    prange = range  # type: ignore[assignment]


__all__ = ["njit", "prange", "JIT_ENABLED"]
