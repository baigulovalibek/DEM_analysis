"""
Deterministic synthetic DEM for regression tests and benchmarks.

A "sum of Gaussian bumps over a random seed=42 plane" — produces terrain
that exercises every analysis path: positive and negative relief, smooth
gradients suitable for slope/aspect/curvature, ridges and basins for
hydrology, and enough roughness for TRI/TPI/SVF to register non-trivial
values.

Two helpers:

* :func:`make_dem(size, seed=42)` — return a freshly-generated float32 DEM.
* :func:`make_dem_with_void(size, seed=42)` — same plus a rectangular
  ``-9999`` patch so the nodata-handling paths are also covered.

Both are deterministic given the same ``(size, seed)`` pair.
"""
from __future__ import annotations

import numpy as np


NODATA = -9999.0


def make_dem(size: int, seed: int = 42) -> np.ndarray:
    """Sum of 24 Gaussian bumps overlaid on a tilted plane + low-amplitude noise."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(
        np.linspace(0, 1, size, dtype=np.float64),
        np.linspace(0, 1, size, dtype=np.float64),
    )

    # Tilted base plane — 80 m drop across the diagonal, so hydrology has
    # somewhere to flow.
    dem = 100.0 + 40.0 * xs - 40.0 * ys

    # 24 Gaussian bumps; amplitudes balanced so total relief stays bounded.
    n_bumps = 24
    cx = rng.uniform(0.05, 0.95, n_bumps)
    cy = rng.uniform(0.05, 0.95, n_bumps)
    sigma = rng.uniform(0.04, 0.18, n_bumps)
    amp = rng.uniform(-60.0, 80.0, n_bumps)
    for i in range(n_bumps):
        dem += amp[i] * np.exp(
            -((xs - cx[i]) ** 2 + (ys - cy[i]) ** 2) / (2.0 * sigma[i] ** 2)
        )

    # Small Gaussian noise — gives TRI/TPI something to bite on.
    dem += rng.standard_normal((size, size)) * 0.5

    return dem.astype(np.float32)


def make_dem_with_void(size: int, seed: int = 42) -> np.ndarray:
    """Same DEM plus a 10 %-side rectangular nodata patch."""
    dem = make_dem(size, seed)
    h = max(1, size // 10)
    w = max(1, size // 6)
    r0 = size // 4
    c0 = size // 3
    dem[r0:r0 + h, c0:c0 + w] = NODATA
    return dem
