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
    slope: np.ndarray = None,
    slope_threshold: float = None,
) -> np.ndarray:
    """
    Binary stream grid: 1 where flow_accum >= threshold, 0 otherwise.

    If slope is provided and slope_threshold is set, a cell is also a channel
    only when slope <= slope_threshold (Montgomery & Dietrich 1988 concept:
    channels form where SCA is high AND slope is not too steep).
    """
    stream = (flow_accum >= threshold).astype(np.uint8)
    if slope is not None and slope_threshold is not None:
        stream &= (slope <= slope_threshold)
    return stream


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

    Returns int16 grid; 0 for non-channel cells.
    """
    rows, cols = stream.shape
    order = np.zeros((rows, cols), dtype=np.int16)
    in_degree = np.zeros((rows, cols), dtype=np.int32)

    # Compute in-degree for each stream cell
    for code, (dr, dc) in D8_OFFSETS.items():
        # Cells whose downstream neighbour is (r+dr, c+dc)
        # We need to find source cells that drain INTO (r, c)
        pass

    # Build adjacency in topological order (upstream-to-downstream)
    # Reverse D8 map: for each cell, who flows INTO it?
    reverse = {(dr, dc): code for code, (dr, dc) in D8_OFFSETS.items()}

    # In-flow count for ordering
    contributes_to = {}   # (r, c) → (nr, nc)

    for r in range(rows):
        for c in range(cols):
            if not stream[r, c]:
                continue
            code = flow_dir[r, c]
            if code not in D8_OFFSETS:
                continue
            dr, dc = D8_OFFSETS[code]
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and stream[nr, nc]:
                in_degree[nr, nc] += 1
                contributes_to[(r, c)] = (nr, nc)

    # BFS from headwaters (in_degree == 0 stream cells)
    from collections import deque
    queue = deque()
    for r in range(rows):
        for c in range(cols):
            if stream[r, c] and in_degree[r, c] == 0:
                order[r, c] = 1
                queue.append((r, c))

    # Track how many tributaries of each order drain into each junction
    from collections import defaultdict
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
    Simple automatic threshold: cells in the top `percentile` of accumulation
    define channels — a quick proxy for Tarboton et al. (1991) constant-drop.
    """
    return float(np.percentile(flow_accum[flow_accum > 0], percentile))
