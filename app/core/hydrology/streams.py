"""
Stream network extraction and Strahler order assignment.

References:
  Strahler (1957) — stream order
  Tarboton, Bras & Rodriguez-Iturbe (1991) — threshold selection
  Montgomery & Dietrich (1988) — slope-area channel initiation
"""
from __future__ import annotations
import numpy as np
from app.core.hydrology.flow_direction import D8_OFFSETS


def extract_streams(
    flow_accum: np.ndarray,
    threshold: float,
    slope: np.ndarray | None = None,
    slope_threshold: float | None = None,
) -> np.ndarray:
    """
    Binary stream grid: 1 where flow_accum >= threshold, 0 otherwise.

    If slope is provided and slope_threshold is set, a cell is also a channel
    only when slope <= slope_threshold (Montgomery & Dietrich 1988 concept:
    channels form where SCA is high AND slope is not too steep).
    """
    stream_bool = flow_accum >= threshold
    if slope is not None and slope_threshold is not None:
        stream_bool = stream_bool & (slope <= slope_threshold)
    return stream_bool.astype(np.uint8)


def strahler_order(
    stream: np.ndarray,
    flow_dir: np.ndarray,
) -> np.ndarray:
    """
    Strahler (1957) stream order on a D8 network.

    Rules:
      1. Headwater streams (no upstream tributaries) = order 1.
      2. Where two streams of the same order join → order + 1.
      3. Where streams of different orders join → max order.

    Returns int16 grid; 0 for non-channel cells.  Disconnected stream cells
    (e.g. a single sink-cell channel where flow_dir == 0) come out as
    order 1 stubs — these are valid headwater terminations rather than
    bugs.
    """
    rows, cols = stream.shape
    order = np.zeros((rows, cols), dtype=np.int16)
    in_degree = np.zeros((rows, cols), dtype=np.int32)

    # Iterate only over the stream cells — np.argwhere skips the
    # ~99 % of cells that aren't channels.  For a 4 k × 4 k DEM with
    # 1 % stream coverage this cuts the outer pass from 16 M iterations
    # to ~160 k.
    stream_bool = stream.astype(bool)
    stream_cells = np.argwhere(stream_bool)

    contributes_to: dict = {}
    for r, c in stream_cells:
        code = int(flow_dir[r, c])
        if code not in D8_OFFSETS:
            continue
        dr, dc = D8_OFFSETS[code]
        nr, nc = int(r + dr), int(c + dc)
        if 0 <= nr < rows and 0 <= nc < cols and stream_bool[nr, nc]:
            in_degree[nr, nc] += 1
            contributes_to[(int(r), int(c))] = (nr, nc)

    # BFS from headwaters (in_degree == 0 stream cells)
    from collections import deque, defaultdict
    queue: deque = deque()
    for r, c in stream_cells:
        rr, cc = int(r), int(c)
        if in_degree[rr, cc] == 0:
            order[rr, cc] = 1
            queue.append((rr, cc))

    # Track how many tributaries of each order drain into each junction
    incoming_orders: dict = defaultdict(list)

    while queue:
        r, c = queue.popleft()
        if (r, c) not in contributes_to:
            continue
        nr, nc = contributes_to[(r, c)]
        incoming_orders[(nr, nc)].append(int(order[r, c]))
        in_degree[nr, nc] -= 1
        if in_degree[nr, nc] == 0:
            # All upstream tributaries processed — assign order
            trib_orders = incoming_orders[(nr, nc)]
            max_ord = max(trib_orders)
            if trib_orders.count(max_ord) >= 2:
                order[nr, nc] = max_ord + 1
            else:
                order[nr, nc] = max_ord
            queue.append((nr, nc))

    return order


def auto_threshold(
    flow_accum: np.ndarray,
    percentile: float = 99.0,
) -> float:
    """
    "Top percentile" auto-threshold: returns the flow-accumulation value such
    that ``percentile`` % of positive-accumulation cells fall below it.
    Cells at or above the returned value become channels.

    This is *not* the constant-drop method of Tarboton et al. (1991) — that
    method picks the threshold at which the mean along-channel slope stops
    decreasing.  The percentile rule used here is a simpler heuristic that
    works well on DEMs without large flat regions; tune ``percentile`` to
    taste (95–99 typical).

    Returns 0.0 if the input has no positive-accumulation cells (e.g. a
    fully masked input), which is a safer fallback than letting
    ``np.percentile`` raise on an empty array.
    """
    positive = flow_accum[np.isfinite(flow_accum) & (flow_accum > 0)]
    if positive.size == 0:
        return 0.0
    return float(np.percentile(positive, percentile))
