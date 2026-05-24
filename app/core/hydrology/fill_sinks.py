"""
Depression removal by the Priority-Flood algorithm.

Reference: Barnes, R., Lehman, C., Mulla, D. (2014). Priority-flood: an optimal
           depression-filling and watershed-labeling algorithm for digital elevation
           models. Computers & Geosciences 62, 117-127.

The implementation is O(n log n) using a min-heap and processes the grid by
flooding inward from the boundary, guaranteeing every cell drains to the edge.

Breaching (Lindsay 2016) is offered as an alternative that better preserves
valley morphology. The hybrid approach (breach then fill) is the default.

When Numba is available (``app.core._jit.JIT_ENABLED``) the inner heap
loop runs as a compiled ``@njit`` kernel with a manual binary heap over
two arrays — Numba inlines that much better than CPython ``heapq`` over
3-tuples and the kernel hits the 20-40x lift documented in
OPTIMIZATION.md Phase B.1.  Tie-breaking is done on the flat cell index
``r * cols + c``, which mirrors ``heapq``'s lexicographic
``(elev, r, c)`` ordering exactly — outputs are bit-identical between
the two paths.
"""
from __future__ import annotations
import heapq

import numpy as np
from scipy.ndimage import minimum_filter

from app.core._jit import JIT_ENABLED, njit


_EIGHT_DIRS = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1),  (1, 0),  (1, 1)]


# ── JIT-compiled inner heap kernel ──────────────────────────────────────

@njit(cache=True)
def _priority_flood_kernel(
    filled: np.ndarray,        # float64[H, W] (mutated)
    nodata_mask: np.ndarray,   # bool[H, W]
    processed: np.ndarray,     # bool[H, W] (mutated; border already True)
    heap_elev: np.ndarray,     # float64[capacity]
    heap_idx: np.ndarray,      # int64[capacity]
    sz: int,                   # current heap size (border seeds already in)
    epsilon: float,
) -> None:
    """Run the Barnes (2014) Priority-Flood inner loop in JIT.

    ``heap_elev`` / ``heap_idx`` form a binary min-heap ordered by
    ``(elev, idx)`` where ``idx = r * cols + c``.  That ordering is
    equivalent to ``heapq``'s ``(elev, r, c)`` tuple comparison, so the
    pop sequence — and therefore the filled output — is bit-identical
    to the pure-Python reference.
    """
    rows = filled.shape[0]
    cols = filled.shape[1]
    DR0 = -1; DR1 = -1; DR2 = -1; DR3 = 0; DR4 = 0; DR5 = 1; DR6 = 1; DR7 = 1
    DC0 = -1; DC1 = 0;  DC2 = 1;  DC3 = -1; DC4 = 1; DC5 = -1; DC6 = 0; DC7 = 1

    while sz > 0:
        # ── heap pop ────────────────────────────────────────────────
        elev = heap_elev[0]
        ix = heap_idx[0]
        sz -= 1
        if sz > 0:
            ne = heap_elev[sz]
            ni = heap_idx[sz]
            heap_elev[0] = ne
            heap_idx[0] = ni
            pos = 0
            # Sift-down preserving (elev, idx) lex order
            while True:
                lc = 2 * pos + 1
                rc = 2 * pos + 2
                small = pos
                if lc < sz:
                    el = heap_elev[lc]
                    il = heap_idx[lc]
                    es = heap_elev[small]
                    iss = heap_idx[small]
                    if el < es or (el == es and il < iss):
                        small = lc
                if rc < sz:
                    er = heap_elev[rc]
                    ir = heap_idx[rc]
                    es = heap_elev[small]
                    iss = heap_idx[small]
                    if er < es or (er == es and ir < iss):
                        small = rc
                if small == pos:
                    break
                t_e = heap_elev[pos]
                t_i = heap_idx[pos]
                heap_elev[pos] = heap_elev[small]
                heap_idx[pos] = heap_idx[small]
                heap_elev[small] = t_e
                heap_idx[small] = t_i
                pos = small

        # Decode flat index into (r, c)
        r = ix // cols
        c = ix - r * cols

        # ── 8-neighbour scan, sift-up for each newly enqueued cell ──
        for d in range(8):
            if d == 0:
                nr = r + DR0; nc = c + DC0
            elif d == 1:
                nr = r + DR1; nc = c + DC1
            elif d == 2:
                nr = r + DR2; nc = c + DC2
            elif d == 3:
                nr = r + DR3; nc = c + DC3
            elif d == 4:
                nr = r + DR4; nc = c + DC4
            elif d == 5:
                nr = r + DR5; nc = c + DC5
            elif d == 6:
                nr = r + DR6; nc = c + DC6
            else:
                nr = r + DR7; nc = c + DC7

            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if processed[nr, nc]:
                continue
            if nodata_mask[nr, nc]:
                processed[nr, nc] = True
                continue
            processed[nr, nc] = True

            cur = filled[nr, nc]
            test = elev + epsilon
            new_elev = cur if cur > test else test
            filled[nr, nc] = new_elev

            # Heap push
            n_idx = nr * cols + nc
            heap_elev[sz] = new_elev
            heap_idx[sz] = n_idx
            ppos = sz
            sz += 1
            while ppos > 0:
                par = (ppos - 1) >> 1
                ep = heap_elev[ppos]
                ip = heap_idx[ppos]
                ea = heap_elev[par]
                ia = heap_idx[par]
                if ea > ep or (ea == ep and ia > ip):
                    heap_elev[par] = ep
                    heap_idx[par] = ip
                    heap_elev[ppos] = ea
                    heap_idx[ppos] = ia
                    ppos = par
                else:
                    break


def _priority_flood_py(
    dem: np.ndarray,
    nodata: float | None,
    epsilon: float,
    _progress,
) -> np.ndarray:
    """Pure-Python Priority-Flood — the fallback when Numba isn't available.

    Output is bit-identical to :func:`priority_flood`; that's enforced by
    the shared regression fixture under ``tests/fixtures/``.
    """
    rows, cols = dem.shape
    filled = dem.astype(np.float64)
    processed = np.zeros((rows, cols), dtype=bool)

    nodata_mask = ~np.isfinite(filled)
    if nodata is not None:
        nodata_mask |= (filled == nodata)

    border_rr = np.concatenate([
        np.zeros(cols, dtype=np.int64),
        np.full(cols, rows - 1, dtype=np.int64),
        np.arange(1, rows - 1, dtype=np.int64),
        np.arange(1, rows - 1, dtype=np.int64),
    ])
    border_cc = np.concatenate([
        np.arange(cols, dtype=np.int64),
        np.arange(cols, dtype=np.int64),
        np.zeros(rows - 2, dtype=np.int64),
        np.full(rows - 2, cols - 1, dtype=np.int64),
    ])
    border_mask = nodata_mask[border_rr, border_cc]
    processed[border_rr, border_cc] = True
    keep = ~border_mask
    keep_rr = border_rr[keep]
    keep_cc = border_cc[keep]
    keep_elev = filled[keep_rr, keep_cc]
    heap = [
        (float(e), int(r), int(c))
        for e, r, c in zip(keep_elev, keep_rr, keep_cc)
    ]
    heapq.heapify(heap)

    total = max(rows * cols, 1)
    done = 0
    next_report = 0

    while heap:
        elev, r, c = heapq.heappop(heap)

        done += 1
        if _progress is not None and done >= next_report:
            if _progress(int(done / total * 100)):
                return filled
            next_report = done + max(1, total // 100)

        for dr, dc in _EIGHT_DIRS:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if processed[nr, nc]:
                continue
            if nodata_mask[nr, nc]:
                processed[nr, nc] = True
                continue
            processed[nr, nc] = True
            new_elev = max(filled[nr, nc], elev + epsilon)
            filled[nr, nc] = new_elev
            heapq.heappush(heap, (new_elev, nr, nc))

    return filled


def priority_flood(
    dem: np.ndarray,
    nodata: float | None = None,
    epsilon: float = 1e-6,
    _progress=None,
) -> np.ndarray:
    """
    Fill depressions using Priority-Flood (Barnes et al. 2014).

    Parameters
    ----------
    dem     : 2-D float elevation array
    nodata  : elevation value treated as missing (retained as-is in output)
    epsilon : tiny increment added to each flat-traversal step so water
              drains directionally (Garbrecht & Martz 1997 flat-resolution
              concept).  The increment accumulates linearly along a flat:
              after N steps the surface is raised by ``N · epsilon``.  For
              a 10 000-cell flat at the default 1e-6 that is 10 mm — fine
              for most DEMs, but for sub-metre LiDAR over long floodplains
              consider lowering ``epsilon`` (e.g. 1e-9) or pre-resolving
              flats explicitly.

    Returns
    -------
    filled DEM as float64, same shape
    """
    if not JIT_ENABLED:
        return _priority_flood_py(dem, nodata, epsilon, _progress)

    rows, cols = dem.shape
    filled = dem.astype(np.float64)
    processed = np.zeros((rows, cols), dtype=bool)

    nodata_mask = ~np.isfinite(filled)
    if nodata is not None:
        nodata_mask |= (filled == nodata)

    # Border seeding stays in Python: it's a single O(rows + cols) pass
    # whose cost is dwarfed by the inner kernel.  We pre-allocate the
    # heap to rows*cols cells — the worst-case frontier — so the JIT
    # kernel never needs to resize (it operates on raw arrays).
    capacity = rows * cols
    heap_elev = np.empty(capacity, dtype=np.float64)
    heap_idx = np.empty(capacity, dtype=np.int64)

    border_rr = np.concatenate([
        np.zeros(cols, dtype=np.int64),
        np.full(cols, rows - 1, dtype=np.int64),
        np.arange(1, rows - 1, dtype=np.int64),
        np.arange(1, rows - 1, dtype=np.int64),
    ])
    border_cc = np.concatenate([
        np.arange(cols, dtype=np.int64),
        np.arange(cols, dtype=np.int64),
        np.zeros(rows - 2, dtype=np.int64),
        np.full(rows - 2, cols - 1, dtype=np.int64),
    ])
    border_mask = nodata_mask[border_rr, border_cc]
    processed[border_rr, border_cc] = True
    keep = ~border_mask
    keep_rr = border_rr[keep]
    keep_cc = border_cc[keep]
    keep_elev = filled[keep_rr, keep_cc].astype(np.float64)
    keep_idx = (keep_rr * cols + keep_cc).astype(np.int64)

    # Manually heapify the border seeds so the JIT kernel can start
    # straight into its hot loop.  Equivalent to ``heapq.heapify`` plus
    # the (elev, idx) tie-break ordering above.
    sz = keep_elev.size
    heap_elev[:sz] = keep_elev
    heap_idx[:sz] = keep_idx
    if sz > 1:
        _heapify_inplace(heap_elev, heap_idx, sz)

    # Coarse 0% → 100% progress: the JIT kernel finishes priority-flood
    # at 4 k² in 1-3 seconds, so a granular progress bar buys no UX —
    # one tick at the top and one at the bottom is enough.
    if _progress is not None and _progress(0):
        return filled
    _priority_flood_kernel(filled, nodata_mask, processed,
                           heap_elev, heap_idx, sz, float(epsilon))
    if _progress is not None:
        _progress(100)

    return filled


@njit(cache=True)
def _heapify_inplace(heap_elev: np.ndarray, heap_idx: np.ndarray, sz: int) -> None:
    """Build the heap invariant over ``heap_*[:sz]`` in place.

    Equivalent to :func:`heapq.heapify` against the lexicographic
    ``(heap_elev[i], heap_idx[i])`` ordering, but JIT-compiled so the
    border seeding cost stays negligible even for 4 k² DEMs.
    """
    # Standard Floyd build-heap: sift down from the last non-leaf to root.
    start = (sz - 2) // 2
    while start >= 0:
        pos = start
        while True:
            lc = 2 * pos + 1
            rc = 2 * pos + 2
            small = pos
            if lc < sz:
                el = heap_elev[lc]; il = heap_idx[lc]
                es = heap_elev[small]; iss = heap_idx[small]
                if el < es or (el == es and il < iss):
                    small = lc
            if rc < sz:
                er = heap_elev[rc]; ir = heap_idx[rc]
                es = heap_elev[small]; iss = heap_idx[small]
                if er < es or (er == es and ir < iss):
                    small = rc
            if small == pos:
                break
            t_e = heap_elev[pos]; t_i = heap_idx[pos]
            heap_elev[pos] = heap_elev[small]; heap_idx[pos] = heap_idx[small]
            heap_elev[small] = t_e; heap_idx[small] = t_i
            pos = small
        start -= 1


def breach_and_fill(
    dem: np.ndarray,
    nodata: float | None = None,
    max_breach_depth: float = 10.0,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Hybrid breach-then-fill (Lindsay 2016).

    Carves a least-cost path from each pit cell outward — Dijkstra over the
    elevation surface — until the path reaches a cell lower than the pit.  If
    the carving depth stays within ``max_breach_depth`` the path is breached
    (elevations along the path are lowered to form a monotonically descending
    channel); otherwise the pit is filled instead.  A final Priority-Flood
    pass guarantees every remaining cell drains to the boundary.

    ``max_breach_depth`` is in the DEM's elevation units (typically metres):
    paths whose maximum-cost cell exceeds the outlet elevation by more than
    this value are abandoned and filled instead.  The default 10 m is sized
    for ~30 m DEMs with metre-scale relief; tune downward for high-resolution
    LiDAR over flat terrain.
    """
    rows, cols = dem.shape
    result = dem.astype(np.float64)

    # Treat NaN and explicit nodata as missing.
    nodata_mask = ~np.isfinite(result)
    if nodata is not None:
        nodata_mask |= (result == nodata)

    # Identify pits as TRUE local minima — cells strictly lower than all 8
    # neighbours that were also raised by Priority-Flood (i.e. sit inside
    # an actual depression rather than on the outflow side).  The previous
    # implementation processed every cell raised by the fill, which on a
    # large lake meant running an O(N) Dijkstra once per lake cell —
    # quadratic in lake area.
    work = np.where(nodata_mask, np.inf, result)
    local_min_elev = minimum_filter(work, size=3, mode="nearest")
    local_minima = (work <= local_min_elev) & ~nodata_mask

    filled = priority_flood(dem, nodata, epsilon=0.0)
    raised = (filled > result + 1e-12) & ~nodata_mask
    pits_mask = local_minima & raised
    pit_indices = np.argwhere(pits_mask)
    # Process the deepest pits first.
    pit_indices = sorted(
        pit_indices.tolist(),
        key=lambda rc: filled[rc[0], rc[1]] - result[rc[0], rc[1]],
        reverse=True,
    )

    # Pre-allocate the per-pit Dijkstra scratch arrays once and reset
    # between pits.  ``parent_r`` / ``parent_c`` replace the Python
    # ``dict[(r, c) -> (r, c)]`` used by the legacy reference — Numba
    # can't lower a tuple-keyed dict, and the two int32 grids halve the
    # per-pit allocation cost on every backend.
    cost = np.empty((rows, cols), dtype=np.float64)
    parent_r = np.empty((rows, cols), dtype=np.int32)
    parent_c = np.empty((rows, cols), dtype=np.int32)
    heap_capacity = rows * cols
    heap_cost = np.empty(heap_capacity, dtype=np.float64)
    heap_idx = np.empty(heap_capacity, dtype=np.int64)

    dijkstra = (
        _breach_dijkstra_kernel if JIT_ENABLED else _breach_dijkstra_py
    )

    for pr, pc in pit_indices:
        pit_elev = result[pr, pc]
        # Skip pits that have already been resolved by a prior breach pass.
        if result[pr, pc] >= filled[pr, pc] - 1e-12:
            continue

        cost.fill(np.inf)
        parent_r.fill(-1)
        parent_c.fill(-1)

        out_r, out_c, max_cost_along_path = dijkstra(
            result, nodata_mask, int(pr), int(pc), float(pit_elev),
            cost, parent_r, parent_c, heap_cost, heap_idx,
        )

        if out_r < 0:
            continue  # No lower outlet reachable — handled by final fill.

        breach_depth = max_cost_along_path - result[out_r, out_c]
        if breach_depth > max_breach_depth:
            continue  # Too deep — leave for the fill pass to handle.

        # Reconstruct the path from pit → outlet via the parent grids.
        path: list[tuple[int, int]] = [(int(out_r), int(out_c))]
        cur_r, cur_c = int(out_r), int(out_c)
        while True:
            pr_prev = int(parent_r[cur_r, cur_c])
            pc_prev = int(parent_c[cur_r, cur_c])
            if pr_prev < 0:
                break
            path.append((pr_prev, pc_prev))
            cur_r, cur_c = pr_prev, pc_prev
        path.reverse()   # pit → outlet

        out_elev = float(result[out_r, out_c])
        n = len(path)
        if n < 2:
            continue
        # Carve a monotonically descending channel from pit → outlet.  Each
        # interior cell is at most one ε above its downstream neighbour, and
        # cells are only lowered (never raised) so existing terrain wins.
        for i, (rr, cc) in enumerate(path[:-1]):
            target = out_elev + (n - 1 - i) * epsilon
            if target < result[rr, cc]:
                result[rr, cc] = target

    # Final fill pass to ensure full connectivity even for pits we skipped.
    return priority_flood(result, nodata, epsilon)


# ── Breach-Dijkstra kernel + Python fallback ──────────────────────────

@njit(cache=True)
def _breach_dijkstra_kernel(
    result: np.ndarray,        # float64[H, W]
    nodata_mask: np.ndarray,   # bool[H, W]
    pit_pr: int,
    pit_pc: int,
    pit_elev: float,
    cost: np.ndarray,          # float64[H, W] (caller filled with +inf)
    parent_r: np.ndarray,      # int32[H, W]   (caller filled with -1)
    parent_c: np.ndarray,      # int32[H, W]   (caller filled with -1)
    heap_cost: np.ndarray,     # float64[capacity] scratch
    heap_idx: np.ndarray,      # int64[capacity]   scratch
) -> tuple[int, int, float]:
    """JIT-compiled Lindsay (2016) breach Dijkstra.

    Returns ``(out_r, out_c, max_cost_along_path)`` for the first
    outlet (cell with ``result < pit_elev``) reached.  Returns
    ``(-1, -1, 0.0)`` when no outlet exists.  Tie-breaking uses
    ``(cost, idx)`` with ``idx = r * cols + c`` — the same lex order as
    the pure-Python heapq reference, so the parent chains match.
    """
    rows = result.shape[0]
    cols = result.shape[1]
    sz = 0

    cost[pit_pr, pit_pc] = pit_elev
    heap_cost[0] = pit_elev
    heap_idx[0] = pit_pr * cols + pit_pc
    sz = 1

    while sz > 0:
        # ── heap pop ────────────────────────────────────────────────
        c_cost = heap_cost[0]
        ix = heap_idx[0]
        sz -= 1
        if sz > 0:
            ne = heap_cost[sz]
            ni = heap_idx[sz]
            heap_cost[0] = ne
            heap_idx[0] = ni
            pos = 0
            while True:
                lc = 2 * pos + 1
                rc = 2 * pos + 2
                small = pos
                if lc < sz:
                    el = heap_cost[lc]
                    il = heap_idx[lc]
                    es = heap_cost[small]
                    iss = heap_idx[small]
                    if el < es or (el == es and il < iss):
                        small = lc
                if rc < sz:
                    er = heap_cost[rc]
                    ir = heap_idx[rc]
                    es = heap_cost[small]
                    iss = heap_idx[small]
                    if er < es or (er == es and ir < iss):
                        small = rc
                if small == pos:
                    break
                t_e = heap_cost[pos]
                t_i = heap_idx[pos]
                heap_cost[pos] = heap_cost[small]
                heap_idx[pos] = heap_idx[small]
                heap_cost[small] = t_e
                heap_idx[small] = t_i
                pos = small

        r = ix // cols
        c = ix - r * cols

        # Stale entry — superseded by a later push at the same cell.
        if c_cost > cost[r, c]:
            continue

        # Outlet: any cell other than the pit whose elevation is below
        # ``pit_elev``.  ``nodata_mask`` is already excluded from
        # neighbour expansion below, so we don't need to re-check it.
        if (r != pit_pr or c != pit_pc) and result[r, c] < pit_elev:
            return r, c, c_cost

        for d in range(8):
            if d == 0:
                nr = r - 1; nc = c - 1
            elif d == 1:
                nr = r - 1; nc = c
            elif d == 2:
                nr = r - 1; nc = c + 1
            elif d == 3:
                nr = r; nc = c - 1
            elif d == 4:
                nr = r; nc = c + 1
            elif d == 5:
                nr = r + 1; nc = c - 1
            elif d == 6:
                nr = r + 1; nc = c
            else:
                nr = r + 1; nc = c + 1

            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if nodata_mask[nr, nc]:
                continue
            other = result[nr, nc]
            new_cost = c_cost if c_cost > other else other
            if new_cost < cost[nr, nc]:
                cost[nr, nc] = new_cost
                parent_r[nr, nc] = r
                parent_c[nr, nc] = c
                # heap push
                heap_cost[sz] = new_cost
                heap_idx[sz] = nr * cols + nc
                ppos = sz
                sz += 1
                while ppos > 0:
                    par = (ppos - 1) >> 1
                    ep = heap_cost[ppos]
                    ip = heap_idx[ppos]
                    ea = heap_cost[par]
                    ia = heap_idx[par]
                    if ea > ep or (ea == ep and ia > ip):
                        heap_cost[par] = ep
                        heap_idx[par] = ip
                        heap_cost[ppos] = ea
                        heap_idx[ppos] = ia
                        ppos = par
                    else:
                        break

    return -1, -1, 0.0


def _breach_dijkstra_py(
    result: np.ndarray,
    nodata_mask: np.ndarray,
    pit_pr: int,
    pit_pc: int,
    pit_elev: float,
    cost: np.ndarray,
    parent_r: np.ndarray,
    parent_c: np.ndarray,
    heap_cost: np.ndarray,    # unused on this path — kept for signature parity
    heap_idx: np.ndarray,
) -> tuple[int, int, float]:
    """Pure-Python breach Dijkstra — signature-compatible with the JIT kernel."""
    rows, cols = result.shape
    cost[pit_pr, pit_pc] = pit_elev
    heap: list[tuple[float, int, int]] = [(pit_elev, pit_pr, pit_pc)]

    while heap:
        c_cost, r, c = heapq.heappop(heap)
        if c_cost > cost[r, c]:
            continue
        if (r, c) != (pit_pr, pit_pc) and result[r, c] < pit_elev:
            return r, c, c_cost
        for dr, dc in _EIGHT_DIRS:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if nodata_mask[nr, nc]:
                continue
            new_cost = max(c_cost, float(result[nr, nc]))
            if new_cost < cost[nr, nc]:
                cost[nr, nc] = new_cost
                parent_r[nr, nc] = r
                parent_c[nr, nc] = c
                heapq.heappush(heap, (new_cost, nr, nc))
    return -1, -1, 0.0
