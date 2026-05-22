"""
Flow direction algorithms: D8 and D-infinity.

References:
  D8   — O'Callaghan & Mark (1984). CVGIP 28(3), 323-344.
  D-inf — Tarboton, D.G. (1997). Water Resources Research 33(2), 309-319.

D8 direction codes (ESRI / TauDEM convention):
    32  64  128
    16   0    1
     8   4    2

D-inf returns a continuous angle (radians, 0=east, CCW) and two direction grids
for the split-flow accumulation.
"""
from __future__ import annotations
import numpy as np

# D8 direction codes → (row_offset, col_offset)
D8_OFFSETS = {
    1:   (0,  1),   # E
    2:   (1,  1),   # SE
    4:   (1,  0),   # S
    8:   (1, -1),   # SW
    16:  (0, -1),   # W
    32:  (-1, -1),  # NW
    64:  (-1,  0),  # N
    128: (-1,  1),  # NE
}

# 8 ordered codes for the D8 search
_D8_CODES = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.int32)
# Corresponding (dr, dc) — row increases SOUTH
_D8_DR    = np.array([0,  1, 1,  1,  0, -1, -1, -1], dtype=np.float64)
_D8_DC    = np.array([1,  1, 0, -1, -1, -1,  0,  1], dtype=np.float64)
_D8_DIST  = np.where(
    (_D8_DR == 0) | (_D8_DC == 0), 1.0, np.sqrt(2.0)
)  # 1 or √2 cell-size units


def d8_flow_direction(
    dem: np.ndarray,
    cell_size: float = 1.0,
    nodata: float = None,
) -> np.ndarray:
    """
    Vectorised D8 flow direction.

    Returns an int32 grid where each cell contains one of the 8 direction codes
    above, or 0 for flat/undrained cells and nodata cells.
    """
    rows, cols = dem.shape
    z = dem.astype(np.float64)
    nodata_mask = np.zeros((rows, cols), dtype=bool)
    if nodata is not None:
        nodata_mask = z == nodata

    zp = np.pad(z, 1, mode="edge")

    # Stack drop = (center - neighbor) / distance for all 8 directions
    drops = np.stack([
        (z - zp[1 + int(dr): 1 + int(dr) + rows, 1 + int(dc): 1 + int(dc) + cols])
        / (dist * cell_size)
        for dr, dc, dist in zip(_D8_DR, _D8_DC, _D8_DIST)
    ], axis=0)   # (8, rows, cols)

    max_idx   = np.argmax(drops, axis=0)     # direction index with steepest drop
    max_drop  = drops[max_idx, np.arange(rows)[:, None], np.arange(cols)[None, :]]

    flow_dir = np.where(max_drop > 0, _D8_CODES[max_idx], 0).astype(np.int32)
    flow_dir[nodata_mask] = 0
    return flow_dir


# ── D-infinity (Tarboton 1997) ─────────────────────────────────────────────

# 8 triangular facets: each defined by two adjacent cell offsets (e1, e2)
# relative to centre.  Tarboton's numbering uses atan facets from 0=E, CCW.
_FACETS = [
    ((0,  1), (-1,  1)),   # facet 0: E–NE
    ((-1, 1), (-1,  0)),   # facet 1: NE–N
    ((-1, 0), (-1, -1)),   # facet 2: N–NW
    ((-1,-1), ( 0, -1)),   # facet 3: NW–W
    (( 0,-1), ( 1, -1)),   # facet 4: W–SW
    (( 1,-1), ( 1,  0)),   # facet 5: SW–S
    (( 1, 0), ( 1,  1)),   # facet 6: S–SE
    (( 1, 1), ( 0,  1)),   # facet 7: SE–E (wraps back)
]
# Start angle (radians, 0=east CCW) for facet i
_FACET_START = [i * np.pi / 4 for i in range(8)]


def d_infinity_flow_direction(
    dem: np.ndarray,
    cell_size: float = 1.0,
    nodata: float = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    D-infinity flow direction (Tarboton 1997).

    Returns
    -------
    angle  : float32 (H, W), flow angle in radians 0=east CCW; NaN for nodata/flat
    slope  : float32 (H, W), slope magnitude in the flow direction
    """
    rows, cols = dem.shape
    z = dem.astype(np.float64)
    nodata_mask = np.zeros((rows, cols), dtype=bool)
    if nodata is not None:
        nodata_mask = (z == nodata)

    zp = np.pad(z, 1, mode="edge")

    best_slope = np.full((rows, cols), -np.inf)
    best_angle = np.full((rows, cols), np.nan)

    for k, ((dr1, dc1), (dr2, dc2)) in enumerate(_FACETS):
        e1 = zp[1 + dr1: 1 + dr1 + rows, 1 + dc1: 1 + dc1 + cols]
        e2 = zp[1 + dr2: 1 + dr2 + rows, 1 + dc2: 1 + dc2 + cols]
        e0 = z

        # Slopes along the two facet edges
        s1 = (e0 - e1) / cell_size          # gradient toward e1
        s2 = (e1 - e2) / cell_size          # gradient toward e2 (diagonal to adjacent)

        # Steepest angle within this triangular facet
        r = np.arctan2(s2, s1)
        # Clamp to facet boundaries [0, π/4]
        r = np.clip(r, 0.0, np.pi / 4)

        s_r = np.sqrt(s1 ** 2 + s2 ** 2)   # combined slope

        # Geographic angle (0=east CCW) for this facet
        angle = _FACET_START[k] + r

        # Keep facet if it has the steepest downslope
        better = (s_r > best_slope) & (s_r > 0)
        best_slope = np.where(better, s_r, best_slope)
        best_angle = np.where(better, angle, best_angle)

    best_angle[nodata_mask] = np.nan
    best_slope[nodata_mask] = np.nan
    return best_angle.astype(np.float32), best_slope.astype(np.float32)
