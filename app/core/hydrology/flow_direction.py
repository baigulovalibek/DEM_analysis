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

    NaN values in ``dem`` (or cells equal to ``nodata``) are treated as missing
    and produce a 0 code in the output.  Drops involving NaN neighbours are
    ignored — never selected as steepest.
    """
    rows, cols = dem.shape
    z = dem.astype(np.float64)
    # Treat both NaN and explicit nodata sentinels as missing.
    nodata_mask = ~np.isfinite(z)
    if nodata is not None:
        nodata_mask |= (z == nodata)

    zp = np.pad(z, 1, mode="edge")

    # Stack drop = (center - neighbor) / distance for all 8 directions.
    # NaN neighbours produce NaN drops, which compare false in all >/>= checks.
    with np.errstate(invalid="ignore"):
        drops = np.stack([
            (z - zp[1 + int(dr): 1 + int(dr) + rows, 1 + int(dc): 1 + int(dc) + cols])
            / (dist * cell_size)
            for dr, dc, dist in zip(_D8_DR, _D8_DC, _D8_DIST)
        ], axis=0)   # (8, rows, cols)
    # Replace NaN drops with -inf so argmax never selects them.  Also force
    # cells whose centre is missing to -inf across the whole stack — without
    # this, NaN-centred cells would have NaN drops everywhere, argmax would
    # return 0 (East), and only the final nodata mask saves us.  Defending
    # the intermediate state keeps the function robust to refactors.
    drops = np.where(np.isfinite(drops), drops, -np.inf)
    drops[:, nodata_mask] = -np.inf

    # Break ties by perturbing each direction's drop with a tiny per-cell
    # value so equal-drop cells (planar slopes) don't all pick east.  The
    # perturbation is keyed by (direction, row, col) so it is deterministic
    # — given the same DEM you always get the same flow direction.
    # Magnitude is well below any realistic elevation difference per metre
    # so it cannot flip a genuine "this direction is steeper" decision.
    rng = np.random.default_rng(0xD8F10D1)
    jitter = rng.uniform(0.0, 1.0, size=drops.shape) * 1e-12
    drops = drops + jitter

    max_idx   = np.argmax(drops, axis=0)     # direction index with steepest drop
    max_drop  = drops[max_idx, np.arange(rows)[:, None], np.arange(cols)[None, :]]

    flow_dir = np.where(max_drop > 0, _D8_CODES[max_idx], 0).astype(np.int32)
    flow_dir[nodata_mask] = 0
    return flow_dir


# ── D-infinity (Tarboton 1997) ─────────────────────────────────────────────

# 8 triangular facets, numbered CCW starting from the E–NE facet.  Each facet
# is described by (cardinal_offset, diagonal_offset, base_angle, sign) per
# Tarboton (1997) Table 1.  The geographic flow angle in 0=E-CCW convention
# is:   angle = base_angle + sign · r,  with r ∈ [0, π/4]
#
# r = atan2(s2, s1) where
#   s1 = (e0 − e_cardinal) / cell_size
#   s2 = (e_cardinal − e_diagonal) / cell_size
#
# The cardinal is the closer (axis-aligned) neighbour of the facet; the
# diagonal is the corner (√2 away).  r=0 means pure-cardinal flow, r=π/4
# means pure-diagonal flow.
_FACETS_T = [
    # (cardinal_offset, diagonal_offset, base_angle, sign)
    (( 0,  1), (-1,  1),       0.0,        +1),  # E–NE
    ((-1,  0), (-1,  1),  np.pi / 2,        -1),  # N–NE
    ((-1,  0), (-1, -1),  np.pi / 2,        +1),  # N–NW
    (( 0, -1), (-1, -1),       np.pi,       -1),  # W–NW
    (( 0, -1), ( 1, -1),       np.pi,       +1),  # W–SW
    (( 1,  0), ( 1, -1),  3 * np.pi / 2,    -1),  # S–SW
    (( 1,  0), ( 1,  1),  3 * np.pi / 2,    +1),  # S–SE
    (( 0,  1), ( 1,  1),  2 * np.pi,        -1),  # E–SE
]


def d_infinity_flow_direction(
    dem: np.ndarray,
    cell_size: float = 1.0,
    nodata: float = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    D-infinity flow direction (Tarboton 1997).

    For each cell we evaluate eight triangular facets.  Inside each facet we
    find the steepest downslope direction.  Tarboton's prescription
    (eq. (1)–(2) of the 1997 paper):

      - r = atan2(s2, s1) gives the angle of steepest descent within the facet
      - if r < 0   → steepest descent is along the cardinal edge; r=0, s=s1
      - if r > π/4 → steepest descent is along the diagonal;     r=π/4,
                     s = (e0 − e2) / (cell_size·√2)
      - cardinal-edge slope s1 must be > 0 for the facet to be a valid
        downhill flow direction (uphill cardinal edges are rejected).

    Returns
    -------
    angle  : float32 (H, W), flow angle in radians 0=east CCW; NaN for nodata/flat
    slope  : float32 (H, W), slope magnitude in the flow direction
    """
    rows, cols = dem.shape
    z = dem.astype(np.float64)
    # Treat NaN and explicit nodata sentinels as missing.
    nodata_mask = ~np.isfinite(z)
    if nodata is not None:
        nodata_mask |= (z == nodata)

    zp = np.pad(z, 1, mode="edge")

    best_slope = np.full((rows, cols), -np.inf)
    best_angle = np.full((rows, cols), np.nan)

    diag_len = cell_size * np.sqrt(2.0)

    with np.errstate(invalid="ignore"):
        for (dr_c, dc_c), (dr_d, dc_d), base, sign in _FACETS_T:
            e_card = zp[1 + dr_c: 1 + dr_c + rows, 1 + dc_c: 1 + dc_c + cols]
            e_diag = zp[1 + dr_d: 1 + dr_d + rows, 1 + dc_d: 1 + dc_d + cols]
            e0 = z

            s1 = (e0 - e_card) / cell_size                 # toward cardinal
            s2 = (e_card - e_diag) / cell_size             # cardinal→diagonal

            r = np.arctan2(s2, s1)

            use_cardinal = r < 0.0
            use_diag     = r > (np.pi / 4)
            r_in         = np.clip(r, 0.0, np.pi / 4)

            # Slope magnitude consistent with the chosen r.
            s_card_only = s1
            s_diag_only = (e0 - e_diag) / diag_len
            s_interior  = np.sqrt(s1 ** 2 + s2 ** 2)
            s_r   = np.where(use_cardinal, s_card_only,
                    np.where(use_diag,     s_diag_only, s_interior))
            r_eff = np.where(use_cardinal, 0.0,
                    np.where(use_diag,     np.pi / 4,   r_in))

            angle = base + sign * r_eff

            # Facet is only valid when the cardinal edge is downhill from the
            # centre (Tarboton 1997 §2.1).  A facet may pick its diagonal
            # endpoint via use_diag, in which case we additionally require
            # the diagonal slope to be downhill.
            valid_facet = (
                np.isfinite(s_r)
                & (s_r > 0)
                & np.where(use_diag, s_diag_only > 0, s1 > 0)
            )

            better = valid_facet & (s_r > best_slope)
            best_slope = np.where(better, s_r, best_slope)
            best_angle = np.where(better, angle, best_angle)

    # Cells that never found a downhill facet stay as NaN with slope = 0.
    no_flow = ~np.isfinite(best_angle)
    best_slope = np.where(no_flow, 0.0, best_slope)

    best_angle[nodata_mask] = np.nan
    best_slope[nodata_mask] = 0.0
    # Normalise angle into [0, 2π)
    best_angle = np.where(np.isfinite(best_angle),
                          np.mod(best_angle, 2 * np.pi),
                          np.nan)
    return best_angle.astype(np.float32), best_slope.astype(np.float32)
