"""
Scene3D — the top-level container the GL widget reads from.

Everything mutable in the 3D view lives here.  Changes set dirty bits; the
widget re-uploads only what changed on the next ``paintGL``.

This module is intentionally free of Qt and GL imports so unit tests and the
drape worker can use it without spinning up a GUI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.core.scene3d.camera import OrbitCamera
from app.core.scene3d.earthquakes import EarthquakeRenderable
from app.core.scene3d.grid import CoordinateGrid
from app.core.scene3d.lighting import SunLight
from app.core.scene3d.terrain import TerrainGrid


@dataclass
class DirtyBits:
    """Per-frame upload tracking.

    The widget consumes these in ``paintGL`` and resets them via ``clear()``
    after a successful re-upload.
    """
    mesh: bool = True
    texture: bool = True
    uniforms: bool = True
    earthquakes: bool = True
    grid: bool = True

    def mark_all(self) -> None:
        self.mesh = True
        self.texture = True
        self.uniforms = True
        self.earthquakes = True
        self.grid = True

    def clear(self) -> None:
        self.mesh = False
        self.texture = False
        self.uniforms = False
        self.earthquakes = False
        self.grid = False


@dataclass
class RenderSettings:
    """User-tunable render parameters.

    Setters route through ``Scene3D`` so the dirty bit is set in one place.
    """
    z_factor: float = 2.0          # vertical exaggeration
    skirt_height_m: float = 50.0   # how far the boundary drops below z_min
    wireframe: bool = False
    eye_dome: bool = False
    eye_dome_strength: float = 0.4
    background: tuple[float, float, float] = (0.10, 0.10, 0.12)   # dark grey
    mesh_max_side: int = 256       # downsample cap when building TerrainGrid
    show_grid: bool = False        # 3D coordinate grid
    show_grid_labels: bool = True  # numeric tick labels next to gridlines


@dataclass
class Scene3D:
    """Top-level 3D scene state."""

    terrain: Optional[TerrainGrid] = None
    drape: Optional[np.ndarray] = None              # (H, W, 4) uint8 RGBA
    earthquakes: Optional[EarthquakeRenderable] = None
    grid: Optional[CoordinateGrid] = None
    camera: OrbitCamera = field(default_factory=OrbitCamera)
    sun: SunLight = field(default_factory=SunLight)
    settings: RenderSettings = field(default_factory=RenderSettings)
    dirty: DirtyBits = field(default_factory=DirtyBits)

    # Layer-selection state for the drape compositor.  Names of layers the
    # user has ticked in the side panel (top of stack first); kept here so
    # that side-panel state survives the scene rebuild that follows an
    # "active DEM changed" event.
    drape_layer_names: list[str] = field(default_factory=list)

    # ── Mutators (single point that owns dirty bits) ──────────────────────────

    def set_terrain(self, terrain: TerrainGrid, reframe_camera: bool = True) -> None:
        """Replace the terrain mesh.

        ``reframe_camera=False`` keeps the current camera state — useful when
        the mesh is being rebuilt at a different quality but the user is
        already pointing at something they care about.
        """
        self.terrain = terrain
        self.dirty.mesh = True
        self.dirty.uniforms = True       # camera framing depends on extent
        if reframe_camera:
            self.frame_terrain()

    def set_drape(self, rgba: np.ndarray) -> None:
        self.drape = rgba
        self.dirty.texture = True

    def set_earthquakes(self, renderable: Optional[EarthquakeRenderable]) -> None:
        """Replace the earthquake stick set; ``None`` clears the catalog."""
        self.earthquakes = renderable
        self.dirty.earthquakes = True

    def set_grid(self, grid: Optional[CoordinateGrid]) -> None:
        """Replace the coordinate grid; ``None`` clears it."""
        self.grid = grid
        self.dirty.grid = True

    def set_show_grid(self, show: bool, show_labels: Optional[bool] = None) -> None:
        """Toggle grid visibility (and optionally tick label visibility)."""
        changed = False
        if bool(show) != self.settings.show_grid:
            self.settings.show_grid = bool(show)
            changed = True
        if show_labels is not None and bool(show_labels) != self.settings.show_grid_labels:
            self.settings.show_grid_labels = bool(show_labels)
            changed = True
        if changed:
            self.dirty.uniforms = True

    def set_z_factor(self, z: float) -> None:
        z = max(0.05, float(z))
        if abs(z - self.settings.z_factor) > 1e-6:
            self.settings.z_factor = z
            self.dirty.uniforms = True

    def set_sun(self, azimuth_deg: Optional[float] = None,
                altitude_deg: Optional[float] = None,
                ambient: Optional[float] = None,
                enabled: Optional[bool] = None) -> None:
        changed = False
        if azimuth_deg is not None and abs(azimuth_deg - self.sun.azimuth_deg) > 1e-6:
            self.sun.azimuth_deg = float(azimuth_deg) % 360.0
            changed = True
        if altitude_deg is not None and abs(altitude_deg - self.sun.altitude_deg) > 1e-6:
            self.sun.altitude_deg = max(0.0, min(90.0, float(altitude_deg)))
            changed = True
        if ambient is not None and abs(ambient - self.sun.ambient) > 1e-6:
            self.sun.ambient = max(0.0, min(1.0, float(ambient)))
            changed = True
        if enabled is not None and bool(enabled) != self.sun.enabled:
            self.sun.enabled = bool(enabled)
            changed = True
        if changed:
            self.dirty.uniforms = True

    def set_wireframe(self, wf: bool) -> None:
        if wf != self.settings.wireframe:
            self.settings.wireframe = wf
            self.dirty.uniforms = True

    def set_background(self, rgb: tuple[float, float, float]) -> None:
        self.settings.background = (
            float(rgb[0]), float(rgb[1]), float(rgb[2]),
        )
        self.dirty.uniforms = True

    # Camera mutators forward through the scene so the widget only watches one
    # dirty bit instead of subscribing to the camera object directly.

    def camera_changed(self) -> None:
        """Call after any direct mutation of ``self.camera``."""
        self.dirty.uniforms = True

    # ── Framing ───────────────────────────────────────────────────────────────

    def frame_terrain(self) -> None:
        """Reset the camera to view the whole terrain at a comfortable angle."""
        if self.terrain is None:
            return
        wx, wy = self.terrain.extent_m
        extent = max(wx, wy, 1.0)
        # Distance that frames the DEM with a small margin.
        margin = 1.2
        fov_y = max(self.camera.fov_y_deg, 1.0)
        dist = (extent / 2.0) / math.tan(math.radians(fov_y) / 2.0) * margin

        # Target the centre of the DEM at mid-elevation so the horizon line
        # sits near the middle of the screen.
        z_top = self.terrain.z_max
        z_bot = self.terrain.z_min
        z_mid = 0.5 * (z_top + z_bot)
        self.camera.target = np.array([0.0, 0.0, z_mid], dtype=np.float64)
        self.camera.distance = float(dist)
        self.camera.azimuth = math.radians(45.0)
        self.camera.elevation = math.radians(35.0)
        # Near/far that comfortably bracket the DEM at any view angle.
        self.camera.near = max(dist * 0.01, 1.0)
        self.camera.far = max(dist * 50.0, 1e5)
        self.dirty.uniforms = True

    def top_view(self) -> None:
        """Look straight down (matches the 2D map orientation)."""
        if self.terrain is None:
            return
        self.frame_terrain()
        self.camera.elevation = math.radians(89.0)
        self.camera.azimuth = 0.0
        self.dirty.uniforms = True

    def north_up(self) -> None:
        """Snap the camera azimuth to 0 (looking north) without changing tilt."""
        self.camera.azimuth = 0.0
        self.dirty.uniforms = True

    def look_at_xy(self, x: float, y: float) -> None:
        """Re-target the camera at a ground point, keeping distance and angles."""
        z = 0.0
        if self.terrain is not None:
            z = self.terrain.elevation_at(x, y) or 0.0
        self.camera.target = np.array([x, y, z], dtype=np.float64)
        self.dirty.uniforms = True
