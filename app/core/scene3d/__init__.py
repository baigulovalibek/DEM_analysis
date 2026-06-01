"""
3D scene model.  Pure data, no Qt / no GL.

The renderer (``app/ui/view3d/gl_widget.py``) reads from these objects and
re-uploads to the GPU when the corresponding dirty bit is set.
"""
from __future__ import annotations

from app.core.scene3d.scene import Scene3D, DirtyBits, RenderSettings
from app.core.scene3d.camera import OrbitCamera
from app.core.scene3d.terrain import TerrainGrid
from app.core.scene3d.lighting import SunLight
from app.core.scene3d.drape import compose_drape
from app.core.scene3d.earthquakes import (
    EarthquakeRenderable, EarthquakeRenderStyle,
    build_unit_cylinder, build_unit_sphere,
)
from app.core.scene3d.grid import CoordinateGrid

__all__ = [
    "Scene3D",
    "DirtyBits",
    "RenderSettings",
    "OrbitCamera",
    "TerrainGrid",
    "SunLight",
    "compose_drape",
    "EarthquakeRenderable",
    "EarthquakeRenderStyle",
    "build_unit_cylinder",
    "build_unit_sphere",
    "CoordinateGrid",
]
