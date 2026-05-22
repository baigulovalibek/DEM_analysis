"""
Flow accumulation (upslope contributing area).

References:
  D8  — O'Callaghan & Mark (1984)
  D∞  — Tarboton (1997), split-flow variant

Both are computed by topological (descending-elevation) traversal — O(n log n)
due to the argsort step, then O(n) propagation.
"""
from __future__ import annotations
import numpy as np
from app.core.hydrology.flow_direction import D8_OFFSETS


def d8_flow_accumulation(
    dem: np.ndarray,
    flow_dir: np.ndarray,
    weight: np.ndarray = None,
    nodata: float = None,
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

    valid = (flow_dir > 0)
    if nodata is not None:
        valid &= (dem != nodata)
    accum[~valid] = 0.0

    # Reverse-direction lookup: which code points to (dr, dc)?
    code_dr = np.zeros(256, dtype=np.int32)
    code_dc = np.zeros(256, dtype=np.int32)
    for code, (dr, dc) in D8_OFFSETS.items():
        code_dr[code] = dr
        code_dc[code] = dc

    # Process cells from highest to lowest elevation
    elev = dem.astype(np.float64)
    order = np.argsort(elev.ravel())[::-1]   # descending elevation indices

    rows_flat = order // cols
    cols_flat = order %  cols

    for r, c in zip(rows_flat, cols_flat):
        if not valid[r, c]:
            continue
        code = flow_dir[r, c]
        dr = code_dr[code]
        dc = code_dc[code]
        nr = r + dr
        nc = c + dc
        if 0 <= nr < rows and 0 <= nc < cols and valid[nr, nc]:
            accum[nr, nc] += accum[r, c]

    return accum.astype(np.float32)


def d_inf_flow_accumulation(
    dem: np.ndarray,
    angle: np.ndarray,
    weight: np.ndarray = None,
) -> np.ndarray:
    """
    D-infinity flow accumulation (Tarboton 1997).

    Flow is proportioned between the two cells adjacent to the flow angle.
    """
    rows, cols = dem.shape
    if weight is None:
        accum = np.ones((rows, cols), dtype=np.float64)
    else:
        accum = weight.astype(np.float64).copy()

    valid = np.isfinite(angle)
    accum[~valid] = 0.0

    # Map each cell's angle to its two receiving neighbours + proportions
    # Tarboton facet geometry: 8 triangular facets, angle 0=east CCW
    # Receiving cells: facet k → (dr1, dc1) and (dr2, dc2)
    _FACETS_DR = [
        (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)
    ]  # 8 cardinal+diagonal neighbours in CCW order starting from E

    order = np.argsort(dem.ravel())[::-1]
    for idx in order:
        r, c = divmod(int(idx), cols)
        if not valid[r, c]:
            continue
        ang = float(angle[r, c])
        # Determine which two neighbours receive flow
        sector = int(ang / (np.pi / 4)) % 8
        prop = (ang - sector * np.pi / 4) / (np.pi / 4)
        prop = np.clip(prop, 0.0, 1.0)

        dr1, dc1 = _FACETS_DR[sector]
        dr2, dc2 = _FACETS_DR[(sector + 1) % 8]

        nr1, nc1 = r + dr1, c + dc1
        nr2, nc2 = r + dr2, c + dc2

        if 0 <= nr1 < rows and 0 <= nc1 < cols and valid[nr1, nc1]:
            accum[nr1, nc1] += accum[r, c] * (1.0 - prop)
        if 0 <= nr2 < rows and 0 <= nc2 < cols and valid[nr2, nc2]:
            accum[nr2, nc2] += accum[r, c] * prop

    return accum.astype(np.float32)


def specific_catchment_area(
    flow_accum: np.ndarray,
    cell_size: float,
) -> np.ndarray:
    """
    a = A / w  where w = cell_size (contour width).  Units: m (if cell_size in m).
    """
    return (flow_accum * cell_size ** 2 / cell_size).astype(np.float32)
