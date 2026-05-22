"""
Surface curvature — profile, plan, and mean.

Reference: Zevenbergen & Thorne (1987). Quantitative analysis of land surface
           topography. Earth Surf. Proc. Landforms 12(1), 47-56.

Sign convention (following Zevenbergen & Thorne):
  Profile curvature: negative = concave (flow accelerates downslope)
  Plan curvature:    positive = concave (flow converges)

The quadratic surface fit uses the same 3×3 window as the Horn gradient.
"""
from __future__ import annotations
import numpy as np


def _zt_coefficients(dem: np.ndarray, cell_size: float, z_factor: float):
    """
    Compute Zevenbergen & Thorne polynomial coefficients D, E, F, G, H
    for the entire grid using vectorised operations.
    """
    L = cell_size
    z = dem.astype(np.float64) * z_factor
    zp = np.pad(z, 1, mode="edge")

    z2 = zp[:-2, 1:-1]   # N
    z4 = zp[1:-1, :-2]   # W
    z5 = zp[1:-1, 1:-1]  # centre
    z6 = zp[1:-1, 2:]    # E
    z8 = zp[2:, 1:-1]    # S

    z1 = zp[:-2, :-2]    # NW
    z3 = zp[:-2, 2:]     # NE
    z7 = zp[2:, :-2]     # SW
    z9 = zp[2:, 2:]      # SE

    D = ((z4 + z6) / 2 - z5) / (L ** 2)      # ½(∂²z/∂x²)
    E = ((z2 + z8) / 2 - z5) / (L ** 2)      # ½(∂²z/∂y²)
    F = (-z1 + z3 + z7 - z9) / (4 * L ** 2)  # ∂²z/∂x∂y
    G = (-z4 + z6) / (2 * L)                  # ½(∂z/∂x)
    H = (z2 - z8) / (2 * L)                   # ½(∂z/∂y)  (northward)

    return D, E, F, G, H


def profile_curvature(
    dem: np.ndarray,
    cell_size: float,
    z_factor: float = 1.0,
    nodata: float = None,
) -> np.ndarray:
    """
    Profile (vertical) curvature — curvature in the slope direction.
    Negative = concave (flow accelerates). Units: 1/cell_size.
    """
    D, E, F, G, H = _zt_coefficients(dem, cell_size, z_factor)
    denom = G ** 2 + H ** 2
    safe = denom > 1e-10
    Kp = np.where(
        safe,
        -2 * (D * G ** 2 + E * H ** 2 + F * G * H) / denom,
        0.0,
    ).astype(np.float32)
    if nodata is not None:
        Kp[dem == nodata] = np.nan
    return Kp


def plan_curvature(
    dem: np.ndarray,
    cell_size: float,
    z_factor: float = 1.0,
    nodata: float = None,
) -> np.ndarray:
    """
    Plan (horizontal) curvature — curvature perpendicular to slope direction.
    Positive = concave (flow converges). Units: 1/cell_size.
    """
    D, E, F, G, H = _zt_coefficients(dem, cell_size, z_factor)
    denom = G ** 2 + H ** 2
    Kc = np.where(
        denom > 1e-10,
        2 * (D * H ** 2 + E * G ** 2 - F * G * H) / denom,
        0.0,
    ).astype(np.float32)
    if nodata is not None:
        Kc[dem == nodata] = np.nan
    return Kc


def mean_curvature(
    dem: np.ndarray,
    cell_size: float,
    z_factor: float = 1.0,
    nodata: float = None,
) -> np.ndarray:
    """Mean curvature = (profile + plan) / 2."""
    Kp = profile_curvature(dem, cell_size, z_factor, nodata)
    Kc = plan_curvature(dem, cell_size, z_factor, nodata)
    return ((Kp + Kc) / 2.0).astype(np.float32)
