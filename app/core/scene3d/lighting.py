"""
Sun lighting parameters.

Lambertian + ambient is enough to match QGIS's default "phong" terrain.
Azimuth/altitude use the same convention as the hillshade analysis so the
two stay visually consistent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SunLight:
    """Directional light parameters.

    Attributes
    ----------
    azimuth_deg : compass bearing of the sun, 0=N, CW.
    altitude_deg : sun angle above horizon (0=horizon, 90=zenith).
    ambient : ambient strength, 0–1.  Lit colour mixes ``albedo * (ambient +
              (1-ambient) * N.L)`` so ``ambient=1`` disables shading.
    enabled : master switch for directional lighting.  When ``False`` the
              renderer pushes ambient=1 to the shader, yielding pure-albedo
              terrain (no Lambertian shading).  Defaults to ``False`` so the
              drape colours read cleanly out of the box.
    """

    azimuth_deg: float = 315.0
    altitude_deg: float = 45.0
    ambient: float = 0.30
    enabled: bool = False

    def direction(self) -> np.ndarray:
        """Unit vector pointing *toward* the sun, in world coordinates.

        Same axes as ``OrbitCamera``: +X east, +Y north, +Z up.
        """
        az = math.radians(self.azimuth_deg)
        alt = math.radians(self.altitude_deg)
        cos_alt = math.cos(alt)
        return np.array(
            [math.sin(az) * cos_alt, math.cos(az) * cos_alt, math.sin(alt)],
            dtype=np.float32,
        )
