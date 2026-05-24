"""
Process-lifetime cache of the sentinel→NaN conversion produced by
:func:`app.core._nodata.to_nan_float64`.

Background
----------
Every stencil-based routine (gradient, curvature, TRI, roughness, horizon
sweep, …) starts by upcasting the DEM to float64 and replacing the nodata
sentinel with NaN.  For a 4 096 × 4 096 float32 DEM the conversion allocates
~256 MB and copies the entire buffer.  Running hillshade → slope → aspect →
multidirectional → curvature against the same DEM today repeats that work
five times — half a gigabyte of redundant temporaries.

Strategy
--------
A small LRU keyed on ``(buffer_pointer, shape, dtype, nbytes, nodata)``.
The numeric fingerprint catches the case where CPython recycles the same
``ctypes.data`` address for a new array — without it, a freshly allocated
buffer at the recycled address would silently return stale data.

Entries are capped at :data:`CACHE_CAP` to bound memory.  The cap is the
only protection against pinning large arrays after the corresponding
``DemLayer`` has been removed from the manager; the layer manager also
calls :func:`invalidate` explicitly when a layer is dropped so the buffer
can be freed immediately instead of waiting for LRU eviction.

The returned array is shared across callers and marked read-only.  None
of the consumers in this codebase mutate the converted array (they all
follow the pattern ``z * z_factor`` or ``np.pad(z, ...)``, both of which
allocate fresh output buffers); the read-only flag enforces that contract
for future callers as well.
"""
from __future__ import annotations
from collections import OrderedDict
from typing import Optional

import numpy as np

from app.core._nodata import to_nan_float64 as _convert


CACHE_CAP = 4

_nan_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_gradient_cache: "OrderedDict[tuple, tuple[np.ndarray, np.ndarray]]" = OrderedDict()


def _buffer_key(dem: np.ndarray) -> tuple:
    """Fingerprint that uniquely identifies a buffer.

    ``ctypes.data`` alone isn't enough — CPython may recycle the address
    for a different array of a different shape, which would silently
    return stale results.  Pairing the address with shape/dtype/nbytes
    makes a collision essentially impossible without also changing one
    of those metadata fields.
    """
    return (
        int(dem.ctypes.data),
        tuple(dem.shape),
        dem.dtype.str,
        int(dem.nbytes),
    )


def _nan_key(dem: np.ndarray, nodata: Optional[float]) -> tuple:
    return (*_buffer_key(dem), None if nodata is None else float(nodata))


def _gradient_key(
    dem: np.ndarray,
    cell_size: float,
    z_factor: float,
    nodata: Optional[float],
) -> tuple:
    return (
        *_buffer_key(dem),
        float(cell_size),
        float(z_factor),
        None if nodata is None else float(nodata),
    )


def _lru_put(cache: OrderedDict, key, value) -> None:
    cache[key] = value
    while len(cache) > CACHE_CAP:
        cache.popitem(last=False)


def to_nan_float64_cached(
    dem: np.ndarray, nodata: Optional[float] = None
) -> np.ndarray:
    """Return the cached sentinel→NaN float64 view of ``dem``.

    The returned array is shared across callers and **read-only** —
    consumers must not write to it.  If you need a mutable copy, call
    :func:`numpy.ndarray.copy` on the result.
    """
    k = _nan_key(dem, nodata)
    cached = _nan_cache.get(k)
    if cached is not None:
        _nan_cache.move_to_end(k)
        return cached
    out = _convert(dem, nodata)
    out.setflags(write=False)
    _lru_put(_nan_cache, k, out)
    return out


def get_cached_gradient(
    dem: np.ndarray,
    cell_size: float,
    z_factor: float,
    nodata: Optional[float],
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return the cached ``(dz_dx, dz_dy)`` pair or ``None`` on miss."""
    k = _gradient_key(dem, cell_size, z_factor, nodata)
    pair = _gradient_cache.get(k)
    if pair is not None:
        _gradient_cache.move_to_end(k)
    return pair


def put_cached_gradient(
    dem: np.ndarray,
    cell_size: float,
    z_factor: float,
    nodata: Optional[float],
    dz_dx: np.ndarray,
    dz_dy: np.ndarray,
) -> None:
    """Store a freshly computed gradient pair under the buffer fingerprint."""
    k = _gradient_key(dem, cell_size, z_factor, nodata)
    # Mark read-only so a future consumer can't mutate the cached arrays.
    dz_dx.setflags(write=False)
    dz_dy.setflags(write=False)
    _lru_put(_gradient_cache, k, (dz_dx, dz_dy))


def invalidate(dem: Optional[np.ndarray] = None) -> None:
    """Drop cache entries for ``dem``'s underlying buffer.

    With ``dem=None`` clears every cache.  Called from
    :class:`app.core.layer_manager.LayerManager` when a layer is removed
    so the converted buffers can be freed immediately.
    """
    if dem is None:
        _nan_cache.clear()
        _gradient_cache.clear()
        return
    base_addr = int(dem.ctypes.data)
    for cache in (_nan_cache, _gradient_cache):
        stale = [k for k in cache if k[0] == base_addr]
        for k in stale:
            cache.pop(k, None)


def cache_size() -> tuple[int, int]:
    """``(nan_cache_size, gradient_cache_size)`` — used by tests."""
    return len(_nan_cache), len(_gradient_cache)
