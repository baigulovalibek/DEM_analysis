"""
Flow accumulation (upslope contributing area).

References:
  D8  — O'Callaghan & Mark (1984)
  D∞  — Tarboton (1997), split-flow variant

Both are computed by topological (descending-elevation) traversal — O(n log n)
due to the argsort step, then O(n) propagation.

The accumulation kernels are JIT-compiled via :mod:`app.core._jit` when
Numba is installed, giving the 20-60x speed-up documented in
OPTIMIZATION.md Phase B.  Without Numba the same kernels run as pure
Python and produce identical results, just slower.
"""
from __future__ import annotations
import numpy as np

from app.core._jit import njit
from app.core.hydrology.flow_direction import D8_OFFSETS


@njit(cache=True)
def _d8_accumulate_kernel(
    order: np.ndarray,         # int64[N], descending-elevation cell order
    target_flat: np.ndarray,   # int64[N], receiver flat index or -1
    valid_flat: np.ndarray,    # bool[N],  cell contributes its accum downstream
    accum_flat: np.ndarray,    # float64[N] (mutated in place)
    start: int,
    stop: int,
) -> None:
    """Propagate accumulation from ``order[start:stop]`` downstream.

    Chunking is what lets the Python caller drive a progress bar; the
    arithmetic itself is order-sensitive across the full grid, so the
    chunk boundaries are purely a callback-cadence concern — they do not
    affect numerical results.
    """
    for k in range(start, stop):
        idx = order[k]
        if not valid_flat[idx]:
            continue
        tgt = target_flat[idx]
        if tgt >= 0:
            accum_flat[tgt] += accum_flat[idx]


@njit(cache=True)
def _dinf_accumulate_kernel(
    order: np.ndarray,         # int64[N]
    target1_flat: np.ndarray,  # int64[N] (-1 if out-of-bounds / can't receive)
    target2_flat: np.ndarray,  # int64[N]
    prop_flat: np.ndarray,     # float64[N], proportion sent to target2
    valid_flat: np.ndarray,    # bool[N]
    accum_flat: np.ndarray,    # float64[N] (mutated)
    start: int,
    stop: int,
) -> None:
    """D-infinity (Tarboton 1997) accumulation, split between two receivers.

    Same chunking pattern as :func:`_d8_accumulate_kernel`.
    """
    for k in range(start, stop):
        idx = order[k]
        if not valid_flat[idx]:
            continue
        a = accum_flat[idx]
        p = prop_flat[idx]
        t1 = target1_flat[idx]
        t2 = target2_flat[idx]
        if t1 >= 0:
            accum_flat[t1] += a * (1.0 - p)
        if t2 >= 0:
            accum_flat[t2] += a * p


def d8_flow_accumulation(
    dem: np.ndarray,
    flow_dir: np.ndarray,
    weight: np.ndarray = None,
    nodata: float = None,
    _progress=None,
) -> np.ndarray:
    """
    D8 flow accumulation.

    Parameters
    ----------
    dem      : elevation array (used only for ordering)
    flow_dir : D8 direction grid from d8_flow_direction()
    weight   : optional per-cell weight (e.g. rainfall); defaults to cell area=1
    nodata   : nodata value

    Returns
    -------
    float32 accumulation grid (number of upstream cells × weight)
    """
    rows, cols = dem.shape
    if weight is None:
        accum = np.ones((rows, cols), dtype=np.float64)
    else:
        accum = weight.astype(np.float64).copy()

    # ``valid`` cells contribute their accumulation downstream.  Cells without
    # a flow direction (boundary outlets, flats, sinks) can still RECEIVE
    # flow as long as their elevation is known — they're terminal sinks.
    missing = ~np.isfinite(dem)
    if nodata is not None:
        missing |= (dem == nodata)
    valid = (flow_dir > 0) & ~missing
    can_receive = ~missing
    accum[missing] = 0.0

    # Reverse-direction lookup: which code points to (dr, dc)?
    code_dr = np.zeros(256, dtype=np.int32)
    code_dc = np.zeros(256, dtype=np.int32)
    for code, (dr, dc) in D8_OFFSETS.items():
        code_dr[code] = dr
        code_dc[code] = dc

    # Pre-resolve the receiver for every cell up front, vectorised:
    #   target_flat[i] = flattened index of the cell that i drains into,
    #                    or -1 if the receiver is out of bounds, missing,
    #                    or i itself has no flow direction.
    # This collapses the per-cell ``r + dr`` / ``c + dc`` arithmetic and
    # boundary-check that used to run inside the hot Python loop into a
    # single vectorised NumPy expression.
    rr_arange = np.arange(rows, dtype=np.int64)[:, None]
    cc_arange = np.arange(cols, dtype=np.int64)[None, :]
    dr_grid = code_dr[flow_dir].astype(np.int64)
    dc_grid = code_dc[flow_dir].astype(np.int64)
    tr = rr_arange + dr_grid
    tc = cc_arange + dc_grid
    in_bounds = (tr >= 0) & (tr < rows) & (tc >= 0) & (tc < cols)
    target_flat = np.where(in_bounds, tr * cols + tc, -1).ravel()
    # Cells whose receiver can't accept flow (missing) are also marked -1.
    can_receive_flat = can_receive.ravel()
    safe = target_flat >= 0
    drops = np.zeros_like(target_flat, dtype=bool)
    drops[safe] = ~can_receive_flat[target_flat[safe]]
    target_flat[drops] = -1

    # Process cells from highest to lowest elevation.  Sort NaNs to the
    # bottom by substituting -inf so they're processed (and skipped) last.
    elev = np.where(np.isfinite(dem), dem.astype(np.float64), -np.inf)
    order = np.argsort(elev.ravel())[::-1].astype(np.int64, copy=False)

    accum_flat = accum.ravel()
    valid_flat = valid.ravel()

    # Drive the JIT kernel in ~100 chunks so the Python side can update
    # the progress bar without re-entering Numba per cell (Numba can't
    # call Python callbacks cheaply).  The chunking is purely a callback
    # cadence — it does not change which cells are summed in what order.
    N = order.size
    total = max(N, 1)
    chunk = max(1, total // 100) if _progress is not None else total

    start = 0
    while start < N:
        stop = min(start + chunk, N)
        _d8_accumulate_kernel(order, target_flat, valid_flat, accum_flat, start, stop)
        start = stop
        if _progress is not None and _progress(int(start / total * 100)):
            return accum.astype(np.float32)

    return accum.astype(np.float32)


def d_inf_flow_accumulation(
    dem: np.ndarray,
    angle: np.ndarray,
    weight: np.ndarray | None = None,
    nodata: float | None = None,
    _progress=None,
) -> np.ndarray:
    """
    D-infinity flow accumulation (Tarboton 1997).

    Flow is proportioned between the two cells adjacent to the flow angle.

    ``nodata`` is the elevation sentinel in ``dem`` (e.g. -9999); those
    cells are treated as missing in the same way as NaN.  Pre-converting
    your DEM to NaN at the I/O boundary works too — both forms are
    accepted here for symmetry with :func:`d8_flow_accumulation`.
    """
    rows, cols = dem.shape
    if weight is None:
        accum = np.ones((rows, cols), dtype=np.float64)
    else:
        accum = weight.astype(np.float64).copy()

    # Cells with a valid flow direction contribute their accumulation.  Cells
    # without (NaN angle) can still RECEIVE flow — they're outlets/boundaries
    # where water pools — so we keep their initial weight (1 by default) and
    # only zero out cells with NaN elevation, which represent missing data.
    valid = np.isfinite(angle)
    missing = ~np.isfinite(dem)
    if nodata is not None:
        missing |= (dem == nodata)
    accum[missing] = 0.0
    # A cell can receive flow as long as its elevation is known.
    can_receive = ~missing
    valid = valid & can_receive

    # Tarboton facet geometry: 8 triangular facets, angle 0=east CCW.
    # Sector k spans [k·π/4, (k+1)·π/4); the two neighbours bracketing
    # that sector are at angles k·π/4 (sector start) and (k+1)·π/4
    # (sector end).  Receiving neighbours in CCW order from E:
    #   E, NE, N, NW, W, SW, S, SE.
    facets = np.array(
        [
            (0,  1),   # E
            (-1, 1),   # NE
            (-1, 0),   # N
            (-1, -1),  # NW
            (0, -1),   # W
            (1, -1),   # SW
            (1,  0),   # S
            (1,  1),   # SE
        ],
        dtype=np.int64,
    )

    # Vectorised pre-compute of per-cell (target1, target2, prop).  All
    # the trig + branchy modulo math that used to live inside the Python
    # hot loop happens once here against NumPy C; the JIT kernel that
    # follows is then a clean ~5-operation arithmetic loop over N cells.
    QUART = np.pi / 4.0
    ang_wrapped = np.where(valid, angle.astype(np.float64), 0.0) % (2.0 * np.pi)
    sector = (ang_wrapped / QUART).astype(np.int64) % 8
    prop = (ang_wrapped - sector.astype(np.float64) * QUART) / QUART
    prop = np.clip(prop, 0.0, 1.0)

    sector_next = (sector + 1) % 8
    dr1 = facets[sector, 0]
    dc1 = facets[sector, 1]
    dr2 = facets[sector_next, 0]
    dc2 = facets[sector_next, 1]

    rr = np.arange(rows, dtype=np.int64)[:, None]
    cc = np.arange(cols, dtype=np.int64)[None, :]
    tr1 = rr + dr1
    tc1 = cc + dc1
    tr2 = rr + dr2
    tc2 = cc + dc2

    in1 = (tr1 >= 0) & (tr1 < rows) & (tc1 >= 0) & (tc1 < cols)
    in2 = (tr2 >= 0) & (tr2 < rows) & (tc2 >= 0) & (tc2 < cols)
    target1_flat = np.where(in1, tr1 * cols + tc1, -1).ravel()
    target2_flat = np.where(in2, tr2 * cols + tc2, -1).ravel()

    can_receive_flat = can_receive.ravel()
    safe1 = target1_flat >= 0
    drop1 = np.zeros_like(safe1)
    drop1[safe1] = ~can_receive_flat[target1_flat[safe1]]
    target1_flat[drop1] = -1
    safe2 = target2_flat >= 0
    drop2 = np.zeros_like(safe2)
    drop2[safe2] = ~can_receive_flat[target2_flat[safe2]]
    target2_flat[drop2] = -1

    # Process highest cells first so contributing area accumulates downstream.
    # NaN angles sort to the end; we skip them inside the loop via `valid`.
    finite = np.where(np.isfinite(dem), dem, -np.inf)
    order = np.argsort(finite.ravel())[::-1].astype(np.int64, copy=False)

    accum_flat = accum.ravel()
    valid_flat = valid.ravel()
    prop_flat = prop.ravel().astype(np.float64, copy=False)

    N = order.size
    total = max(N, 1)
    chunk = max(1, total // 100) if _progress is not None else total

    start = 0
    while start < N:
        stop = min(start + chunk, N)
        _dinf_accumulate_kernel(
            order, target1_flat, target2_flat, prop_flat,
            valid_flat, accum_flat, start, stop,
        )
        start = stop
        if _progress is not None and _progress(int(start / total * 100)):
            return accum.astype(np.float32)

    return accum.astype(np.float32)


def specific_catchment_area(
    flow_accum: np.ndarray,
    cell_size: float,
) -> np.ndarray:
    """
    Specific catchment area a = A / w, where:
      A = number-of-upstream-cells × cell_area = flow_accum · cell_size²
      w = effective contour width ≈ cell_size

    Reducing the expression gives ``flow_accum · cell_size``, which is what
    we return here.  Units: m (when cell_size is in m).
    """
    return (flow_accum * cell_size).astype(np.float32)
