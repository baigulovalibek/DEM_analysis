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


def roughness_range(dem: np.ndarray, window: int = 3) -> np.ndarray:
    """Max − min elevation in a square window.  (gdaldem roughness method)"""
    z = dem.astype(np.float64)
    return (maximum_filter(z, size=window) - minimum_filter(z, size=window)).astype(np.float32)


def roughness_sd(dem: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Standard deviation of residual (detrended) elevation in a square window.
    Residual = z − local_mean, so this is the SD of local elevation variation.
    """
    z = dem.astype(np.float64)
    mean_z = uniform_filter(z, size=window, mode="nearest")
    resid = z - mean_z
    mean_sq = uniform_filter(resid ** 2, size=window, mode="nearest")
    return np.sqrt(np.maximum(mean_sq, 0.0)).astype(np.float32)


def vrm(
    dem: np.ndarray,
    cell_size: float,
    window: int = 3,
    z_factor: float = 1.0,
) -> np.ndarray:
    """
    Vector Ruggedness Measure (Sappington et al. 2007).

    Computes the resultant length of unit normal vectors summed over the window,
    normalised to [0, 1].  VRM ≈ 0 = smooth/flat; VRM ≈ 1 = extremely rough.
    This measure is slope-independent — it detects micro-relief on steep slopes
    just as well as on flat surfaces.
    """
    z = dem.astype(np.float64) * z_factor
    zp = np.pad(z, 1, mode="edge")

    # Surface normal for each cell (un-normalised: (-dz/dx, -dz/dy, 1))
    # Using simple 2nd-order central differences for speed
    dz_dx = (zp[1:-1, 2:] - zp[1:-1, :-2]) / (2.0 * cell_size)
    dz_dy = (zp[:-2, 1:-1] - zp[2:, 1:-1]) / (2.0 * cell_size)  # northward

    mag = np.sqrt(dz_dx ** 2 + dz_dy ** 2 + 1.0)
    nx = -dz_dx / mag
    ny = -dz_dy / mag
    nz = 1.0 / mag

    # Sum unit normals over the window
    sx = uniform_filter(nx, size=window, mode="nearest") * window ** 2
    sy = uniform_filter(ny, size=window, mode="nearest") * window ** 2
    sz = uniform_filter(nz, size=window, mode="nearest") * window ** 2

    resultant = np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)
    n = window ** 2
    vrm_val = 1.0 - resultant / n
    return np.clip(vrm_val, 0.0, 1.0).astype(np.float32)


def surface_area_ratio(
    dem: np.ndarray,
    cell_size: float,
    z_factor: float = 1.0,
) -> np.ndarray:
    """
    3-D surface area ÷ planimetric area (Jenness 2004).

    Ratio ≥ 1; flat = 1.0; rough terrain >> 1.
    Computed per cell using the 4-triangle decomposition of the 3×3 neighbourhood.
    """
    z = dem.astype(np.float64) * z_factor
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

    area_3d = (
        tri_area(dz_n, L, dz_e, L, diag)
        + tri_area(dz_e, L, dz_s, L, diag)
        + tri_area(dz_s, L, dz_w, L, diag)
        + tri_area(dz_w, L, dz_n, L, diag)
    )
    plan_area = L ** 2
    return (area_3d / plan_area).astype(np.float32)
