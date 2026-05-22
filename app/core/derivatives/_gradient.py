"""
Horn (1981) weighted finite-difference gradient, shared by all derivative modules.

Neighbourhood labelling (north-up raster, row 0 = north):

    z1  z2  z3      z[r-1, c-1]  z[r-1, c]  z[r-1, c+1]
    z4  z5  z6  =   z[r,   c-1]  z[r,   c]  z[r,   c+1]
    z7  z8  z9      z[r+1, c-1]  z[r+1, c]  z[r+1, c+1]

Eastward gradient  (positive = elevation increases eastward):
    dz/dx = [(z3+2z6+z9) - (z1+2z4+z7)] / (8L)

Northward gradient (positive = elevation increases northward):
    dz/dy = [(z1+2z2+z3) - (z7+2z8+z9)] / (8L)
    (rows increase southward in raster storage, so we negate the raw row-diff)
"""
from __future__ import annotations
import numpy as np


def horn_gradients(
    dem: np.ndarray,
    cell_size: float,
    z_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (dz_dx, dz_dy): eastward and northward gradients, same shape as dem.
    Edge cells are handled by replicating the boundary (edge-padding).
    """
    z = dem.astype(np.float64) * z_factor
    zp = np.pad(z, 1, mode="edge")

    z1 = zp[:-2, :-2]
    z2 = zp[:-2, 1:-1]
    z3 = zp[:-2, 2:]
    z4 = zp[1:-1, :-2]
    z6 = zp[1:-1, 2:]
    z7 = zp[2:, :-2]
    z8 = zp[2:, 1:-1]
    z9 = zp[2:, 2:]

    dz_dx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8.0 * cell_size)
    dz_dy = ((z1 + 2 * z2 + z3) - (z7 + 2 * z8 + z9)) / (8.0 * cell_size)

    return dz_dx, dz_dy
