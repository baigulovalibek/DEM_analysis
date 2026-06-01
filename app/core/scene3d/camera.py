"""
Cameras used by the 3D scene.

World convention
----------------
- Right-handed coordinate system.
- +X east, +Y north, +Z up.
- Distances are in metres (on a local tangent plane centred on the DEM).
- Angles stored as radians internally; UI converts at the edge.

The orbit camera is the QGIS-style default: the user thinks in terms of a
target point on the ground, distance to it, compass bearing the camera is
looking toward (azimuth, 0=N, CW), and the angle above the horizon
(elevation, 0=horizontal, 90=looking straight down).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# ── Matrix helpers ────────────────────────────────────────────────────────────
#
# We keep these here (instead of pulling in pyrr / glm) so the scene model has
# no third-party dependencies.  Matrices are column-major float32 so they can
# be uploaded straight to GL with ``glUniformMatrix4fv(..., transpose=GL_FALSE)``
# once we move past Phase 1.


def _normalise(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Right-handed view matrix.  Mirrors gluLookAt."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    f = _normalise(target - eye)                 # forward
    s = _normalise(np.cross(f, up))              # right
    u = np.cross(s, f)                           # true up

    m = np.eye(4, dtype=np.float64)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -float(np.dot(s, eye))
    m[1, 3] = -float(np.dot(u, eye))
    m[2, 3] = float(np.dot(f, eye))
    return m


def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Right-handed perspective projection (depth maps to [-1, 1])."""
    f = 1.0 / math.tan(math.radians(fov_y_deg) / 2.0)
    aspect = max(aspect, 1e-6)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


# ── Orbit camera ──────────────────────────────────────────────────────────────


@dataclass
class OrbitCamera:
    """Camera that orbits a target point.

    Attributes
    ----------
    target : world-space pivot point (metres).
    distance : eye-to-target distance.
    azimuth : compass bearing the camera is looking *toward*, radians, 0=N, CW.
    elevation : angle above horizon, radians.  0 = horizontal, π/2 = straight down.
    fov_y_deg : vertical field of view, degrees.
    near, far : clipping planes, metres.
    """

    target: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    distance: float = 5000.0
    azimuth: float = math.radians(45.0)
    elevation: float = math.radians(35.0)
    fov_y_deg: float = 45.0
    near: float = 1.0
    far: float = 1.0e7

    # ── Position ──────────────────────────────────────────────────────────────

    def eye(self) -> np.ndarray:
        """Camera position in world coordinates."""
        # View direction (unit vector toward what the camera looks at):
        #   azimuth 0 → +Y (north),  azimuth +π/2 → +X (east)
        #   elevation tilts the view downward
        cos_el = math.cos(self.elevation)
        sin_el = math.sin(self.elevation)
        sin_az = math.sin(self.azimuth)
        cos_az = math.cos(self.azimuth)
        view_dir = np.array(
            [sin_az * cos_el, cos_az * cos_el, -sin_el],
            dtype=np.float64,
        )
        # Eye is opposite of view direction, at `distance` from target.
        return self.target - view_dir * self.distance

    def view_direction(self) -> np.ndarray:
        return _normalise(self.target - self.eye())

    # ── Matrices ──────────────────────────────────────────────────────────────

    def view_matrix(self) -> np.ndarray:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        # Avoid gimbal lock when the camera looks straight down.
        if abs(self.elevation - math.pi / 2.0) < 1e-3:
            # Use north as up so the view stays oriented consistently with
            # the 2D map at top-down view.
            up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        return look_at(self.eye(), self.target, up)

    def projection_matrix(self, aspect: float) -> np.ndarray:
        return perspective(self.fov_y_deg, aspect, self.near, self.far)

    def mvp(self, aspect: float) -> np.ndarray:
        return self.projection_matrix(aspect) @ self.view_matrix()

    # ── Interactions ──────────────────────────────────────────────────────────

    def orbit(self, d_azimuth: float, d_elevation: float):
        """Rotate around the target.  Inputs in radians."""
        self.azimuth = (self.azimuth + d_azimuth) % (2.0 * math.pi)
        # Clamp elevation to (-89°, +89°).  Going past straight-up flips the
        # camera and would require renegotiating the up vector mid-drag.
        lim = math.radians(89.0)
        self.elevation = max(-lim, min(lim, self.elevation + d_elevation))

    def zoom(self, factor: float):
        """Multiply distance by `factor`.  factor < 1 → zoom in."""
        self.distance = max(1.0, self.distance * float(factor))

    def pan_screen(self, dx_px: float, dy_px: float, viewport_height: int):
        """Pan the target across the ground plane in screen-relative units.

        ``dx_px`` / ``dy_px`` are screen-space pixel deltas (right/down positive).
        The motion is scaled so dragging one viewport-height covers the same
        ground footprint regardless of zoom level.
        """
        # Metres per pixel at the target plane (small-angle approximation).
        metres_per_px = (
            2.0 * self.distance * math.tan(math.radians(self.fov_y_deg) / 2.0)
            / max(viewport_height, 1)
        )

        # Build the right + forward (ground-projected) basis at the current
        # azimuth.  At azimuth=0 the camera looks +Y so right=+X, forward=+Y.
        sin_az = math.sin(self.azimuth)
        cos_az = math.cos(self.azimuth)
        right = np.array([cos_az, -sin_az, 0.0], dtype=np.float64)
        # "Forward" projected onto the ground plane; do not move along Z.
        fwd = np.array([sin_az, cos_az, 0.0], dtype=np.float64)

        # Right-drag should drag the ground toward the cursor → the target
        # moves the *opposite* way of the cursor delta.
        delta = (-dx_px * right + dy_px * fwd) * metres_per_px
        self.target = self.target + delta

    # ── Frustum corner projection (for 2D map sync) ───────────────────────────

    def frustum_corners_at_z(
        self,
        aspect: float,
        z: float = 0.0,
        max_distance: Optional[float] = None,
    ) -> Optional[list[np.ndarray]]:
        """Return the four corners of the camera's view frustum projected onto
        the ``z = const`` plane (world coordinates).

        Returns
        -------
        list of 4 ``np.ndarray`` (world XY at the requested Z), in
        ``[bottom-left, bottom-right, top-right, top-left]`` screen order; or
        ``None`` if any of the four frustum rays does not strike the plane
        below the camera (e.g. the camera is looking up or away from the
        ground).

        ``max_distance`` (metres) caps how far rays may travel before being
        considered "no hit" — useful when the camera looks near-horizontal,
        where intersection points race off to infinity.
        """
        try:
            inv_proj = np.linalg.inv(self.projection_matrix(aspect))
            inv_view = np.linalg.inv(self.view_matrix())
        except np.linalg.LinAlgError:
            return None

        eye = self.eye()
        if max_distance is None:
            max_distance = float(self.far) * 0.95

        # NDC corners at the near plane (z = -1).  Order chosen so the
        # returned polygon traces a simple quad on the ground.
        ndc_corners = [
            (-1.0, -1.0),   # bottom-left
            ( 1.0, -1.0),   # bottom-right
            ( 1.0,  1.0),   # top-right
            (-1.0,  1.0),   # top-left
        ]

        out: list[np.ndarray] = []
        for nx, ny in ndc_corners:
            ndc = np.array([nx, ny, -1.0, 1.0], dtype=np.float64)
            view_pt = inv_proj @ ndc
            view_pt = view_pt / view_pt[3]
            world_pt = inv_view @ view_pt
            world_pt = (world_pt / world_pt[3])[:3]

            direction = world_pt - eye
            n = float(np.linalg.norm(direction))
            if n < 1e-9:
                return None
            direction = direction / n

            # Ray-plane intersection with horizontal plane at z = ``z``.
            denom = direction[2]
            if abs(denom) < 1e-9:
                return None  # ray is parallel to the plane
            t = (z - eye[2]) / denom
            if t <= 0.0 or t > max_distance:
                return None  # plane is behind the camera or too far away
            hit = eye + direction * t
            out.append(hit[:2].copy())  # XY only
        return out
