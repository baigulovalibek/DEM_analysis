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

from app.core._dem_cache import to_nan_float64_cached


def _zt_coefficients(
    dem: np.ndarray,
    cell_size: float,
    z_factor: float,
    nodata: float | None = None,
):
    """
    Compute Zevenbergen & Thorne polynomial coefficients D, E, F, G, H
    for the entire grid using vectorised operations.

    Sentinel ``nodata`` is converted to NaN before the stencil runs so a
    -9999 neighbour cannot generate spurious curvature spikes at the
    border of voids — those cells come back as NaN coefficients, which
    the caller masks.
    """
    L = cell_size
    z = to_nan_float64_cached(dem, nodata) * z_factor
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
    nodata: float | None = None,
) -> np.ndarray:
    """
    Profile (vertical) curvature — curvature in the slope direction.
    Negative = concave (flow accelerates). Units: 1/cell_size.

    The implementation matches ESRI's ``Curvature`` tool, which drops the
    ``(1 + G² + H²)^1.5`` denominator from Zevenbergen & Thorne (1987).
    The simplification overestimates magnitude on terrain steeper than
    ~30° but matches what most GIS users expect to see.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        D, E, F, G, H = _zt_coefficients(dem, cell_size, z_factor, nodata=nodata)
        denom = G ** 2 + H ** 2
        safe = denom > 1e-10
        Kp = np.where(
            safe,
            -2 * (D * G ** 2 + E * H ** 2 + F * G * H) / denom,
            0.0,
        ).astype(np.float32)
    Kp[~np.isfinite(Kp)] = np.nan
    return Kp


def plan_curvature(
    dem: np.ndarray,
    cell_size: float,
    z_factor: float = 1.0,
    nodata: float | None = None,
) -> np.ndarray:
    """
    Plan (horizontal) curvature — curvature perpendicular to slope direction.
    Positive = concave (flow converges). Units: 1/cell_size.

    Same small-slope simplification as :func:`profile_curvature`.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        D, E, F, G, H = _zt_coefficients(dem, cell_size, z_factor, nodata=nodata)
        denom = G ** 2 + H ** 2
        Kc = np.where(
            denom > 1e-10,
            2 * (D * H ** 2 + E * G ** 2 - F * G * H) / denom,
            0.0,
        ).astype(np.float32)
    Kc[~np.isfinite(Kc)] = np.nan
    return Kc


def mean_curvature(
    dem: np.ndarray,
    cell_size: float,
    z_factor: float = 1.0,
    nodata: float | None = None,
) -> np.ndarray:
    """Mean curvature = (profile + plan) / 2."""
    Kp = profile_curvature(dem, cell_size, z_factor, nodata)
    Kc = plan_curvature(dem, cell_size, z_factor, nodata)
    return ((Kp + Kc) / 2.0).astype(np.float32)
