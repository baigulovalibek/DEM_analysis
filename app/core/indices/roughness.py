"""
Terrain roughness / surface irregularity measures.

Implementations:
  1. Range roughness  — Wilson's max−min in window (gdaldem roughness)
  2. SD roughness     — standard deviation of residual elevation in window
  3. VRM              — Vector Ruggedness Measure (Sappington et al. 2007)
  4. Surface-area ratio — Jenness (2004)

Reference:
  Sappington, J.M. et al. (2007). J. Wildlife Management 71(5), 1419-1426.
  Jenness, J.S. (2004). Wildlife Society Bulletin 32(3), 829-839.
  Grohmann et al. (2011). IEEE TGRS 49(4), 1200-1213.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter

from app.core._dem_cache import to_nan_float64_cached


def roughness_range(
    dem: np.ndarray,
    window: int = 3,
    nodata: float | None = None,
) -> np.ndarray:
    """Max − min elevation in a square window.  (gdaldem roughness method)"""
    z = to_nan_float64_cached(dem, nodata)
    # NaN-aware ordfilters: a window containing any NaN produces NaN, which
    # is the correct "uncertain" result at the rim of a nodata patch.
    with np.errstate(invalid="ignore"):
        rng = maximum_filter(z, size=window) - minimum_filter(z, size=window)
    return rng.astype(np.float32)


def roughness_sd(
    dem: np.ndarray,
    window: int = 5,
    nodata: float | None = None,
) -> np.ndarray:
    """
    Standard deviation of elevation in a square window.

    Computed via the centred Welford-style identity

        SD² = E[(z − μ)²]   where μ = E[z]

    rather than ``E[z²] − μ²``.  The two-pass form is numerically stable
    even when ``E[z²] ≈ μ²`` (homogeneous patches, mm-scale LiDAR), where
    the textbook ``E[z²] − μ²`` suffers catastrophic cancellation and
    reports SD = 0 for terrain with real micro-relief.
    """
    z = to_nan_float64_cached(dem, nodata)
    mean_z   = uniform_filter(z,                size=window, mode="nearest")
    residual = z - mean_z
    variance = uniform_filter(residual * residual, size=window, mode="nearest")
    variance = np.maximum(variance, 0.0)
    return np.sqrt(variance).astype(np.float32)


def vrm(
    dem: np.ndarray,
    cell_size: float,
    window: int = 3,
    z_factor: float = 1.0,
    nodata: float | None = None,
) -> np.ndarray:
    """
    Vector Ruggedness Measure (Sappington et al. 2007).

    Computes the resultant length of unit normal vectors summed over the window,
    normalised to [0, 1].  VRM ≈ 0 = smooth/flat; VRM ≈ 1 = extremely rough.
    This measure is slope-independent — it detects micro-relief on steep slopes
    just as well as on flat surfaces.
    """
    z = to_nan_float64_cached(dem, nodata) * z_factor
    zp = np.pad(z, 1, mode="edge")

    # Surface normal for each cell (un-normalised: (-dz/dx, -dz/dy, 1))
    # Using simple 2nd-order central differences for speed
    with np.errstate(invalid="ignore"):
        dz_dx = (zp[1:-1, 2:] - zp[1:-1, :-2]) / (2.0 * cell_size)
        dz_dy = (zp[:-2, 1:-1] - zp[2:, 1:-1]) / (2.0 * cell_size)  # northward

        mag = np.sqrt(dz_dx ** 2 + dz_dy ** 2 + 1.0)
        nx = -dz_dx / mag
        ny = -dz_dy / mag
        nz = 1.0 / mag

        # Sum unit normals over the window.  NaN-contaminated cells poison
        # their window — uniform_filter then propagates NaN to all cells the
        # bad sample touches, which is exactly the rim we want to mask.
        sx = uniform_filter(nx, size=window, mode="nearest") * window ** 2
        sy = uniform_filter(ny, size=window, mode="nearest") * window ** 2
        sz = uniform_filter(nz, size=window, mode="nearest") * window ** 2

        resultant = np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)
        n = window ** 2
        vrm_val = 1.0 - resultant / n

    # np.clip preserves NaN; the cast to float32 keeps it as NaN too.
    out = np.clip(vrm_val, 0.0, 1.0).astype(np.float32)
    return out


def surface_area_ratio(
    dem: np.ndarray,
    cell_size: float,
    z_factor: float = 1.0,
    nodata: float | None = None,
) -> np.ndarray:
    """
    3-D surface area ÷ planimetric area — four-triangle stencil approximation.

    Ratio ≥ 1; flat = 1.0; rough terrain >> 1.
    Each cell's 3-D area is summed from four triangles fanning out from the
    centre to its four cardinal neighbours (N, E, S, W); the plan area
    those triangles cover is 4 · (L²/2) = 2·L².

    Note this is *not* Jenness's (2004) original 8-triangle method — Jenness
    places vertices at neighbour midpoints and his triangles together cover
    exactly one cell area L².  The two formulations give similar values on
    gently rolling terrain but diverge for high-frequency roughness; the
    four-triangle variant here is faster and the value the rest of the code
    has historically reported.
    """
    if z_factor == 1.0:
        z = to_nan_float64_cached(dem, nodata)
    else:
        z = to_nan_float64_cached(dem, nodata) * z_factor
    zp = np.pad(z, 1, mode="edge")

    L = cell_size
    diag = L * np.sqrt(2.0)

    # Four triangles meeting at centre
    dz_n  = zp[:-2, 1:-1] - z
    dz_e  = zp[1:-1, 2:]  - z
    dz_s  = zp[2:,  1:-1] - z
    dz_w  = zp[1:-1, :-2] - z

    def tri_area(dz_a, len_a, dz_b, len_b, base):
        # Area of triangle with two sides of length len_a, len_b and shared dz
        # using Heron's formula on 3-D side lengths
        a = np.sqrt(len_a ** 2 + dz_a ** 2)
        b = np.sqrt(len_b ** 2 + dz_b ** 2)
        c = np.sqrt(base ** 2 + (dz_a - dz_b) ** 2)
        s = (a + b + c) / 2
        return np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0.0))

    with np.errstate(invalid="ignore"):
        area_3d = (
            tri_area(dz_n, L, dz_e, L, diag)
            + tri_area(dz_e, L, dz_s, L, diag)
            + tri_area(dz_s, L, dz_w, L, diag)
            + tri_area(dz_w, L, dz_n, L, diag)
        )
        plan_area = 2.0 * L ** 2
        result = (area_3d / plan_area).astype(np.float32)
    return result
