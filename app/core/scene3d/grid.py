"""
3D coordinate grid for the scene.

Builds line geometry for a matplotlib/QGIS-style bounding-box cage around the
terrain + earthquake volume.  The build is **camera-aware**: it picks which
two walls face *away* from the camera and draws gridlines on those (so the
terrain renders in front of, not behind, the gridlines), and it anchors the
tick labels on the *silhouette* edges of the visible faces — the bottom-front
edges and one of the silhouette vertical edges — so labels sit on the
outline of the box, not its front-corner interior.

Each :class:`TickLabel` also carries an ``outward`` unit vector pointing out
of the bbox at that edge; the GL widget's label overlay uses it to nudge
each label a few screen pixels outside the box silhouette, into the clear
space beyond the geometry.

Coordinate frame matches :class:`TerrainGrid`:

    +X  → east (longitude)
    +Y  → north (latitude)
    +Z  → up (depth is negative, mesh-shifted so the surface floor sits at z=0)

Tick steps follow the 1-2-5 progression used by every plotting library —
that is, sequences like (0.05°, 0.1°, 0.2°, 0.5°, 1°, …) — so a regional
catalog gets human-readable tick labels without manual tweaking.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from app.core.scene3d.terrain import TerrainGrid


# Target number of ticks along each axis.  6 is the QGIS sweet spot — fewer
# leaves the box visually empty; more crowds the axis labels.
_TARGET_TICKS = 6


@dataclass
class TickLabel:
    """One numeric label to draw next to a tick.

    ``outward`` is a unit vector pointing away from the bounding box at the
    label's anchor edge — used by the overlay to nudge the label off the
    box silhouette into the clear space outside.
    """

    world: np.ndarray            # (3,) float32 — world-space anchor point
    text: str                    # rendered label
    axis: str                    # "lon" | "lat" | "depth"
    outward: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -1.0], dtype=np.float32)
    )


@dataclass
class CoordinateGrid:
    """CPU-side line geometry + tick labels for the 3D coordinate grid.

    ``vertices`` / ``colors`` are flat (N, 3) and (N, 4) float32 arrays uploaded
    as ``GL_LINES`` (consecutive pairs form one segment).
    """

    vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    labels: List[TickLabel] = field(default_factory=list)

    # Extents so the GL widget knows when the camera is inside the grid (so
    # the back face becomes the near face — currently we just always draw
    # all six faces' worth of lines and let depth occlude as appropriate).
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
    z_min: float = 0.0           # bottom (most negative in mesh-shifted Z)
    z_max: float = 0.0           # top (terrain surface high point)

    @classmethod
    def build(
        cls,
        terrain: TerrainGrid,
        camera_eye: Optional[np.ndarray] = None,
        depth_min_m: Optional[float] = None,
        depth_max_m: Optional[float] = None,
    ) -> "CoordinateGrid":
        """Build grid lines + tick labels around the terrain + depth volume.

        ``camera_eye`` is the camera position in world coords.  It picks the
        "back" walls (the two NS/EW walls *facing away* from the camera) so
        gridlines render *behind* the terrain instead of on top of it, and it
        chooses the silhouette edges where tick labels are anchored.  Without
        an eye, the build assumes a south-west camera (the default framing).

        ``depth_min_m`` / ``depth_max_m`` are mesh-shifted Z values defining
        the vertical extent.  When omitted, defaults span from a slab below
        the terrain (z_min - 30 km) up to the terrain's top (z_max).
        """
        if terrain is None:
            return cls()

        wx, wy = terrain.extent_m
        x_min = -0.5 * wx
        x_max = +0.5 * wx
        y_min = -0.5 * wy
        y_max = +0.5 * wy

        z_top = float(terrain.z_max)
        if depth_min_m is None:
            depth_min_m = float(terrain.z_min) - 30_000.0
        if depth_max_m is None:
            depth_max_m = z_top
        z_bot = float(depth_min_m)
        z_top = float(depth_max_m)
        if z_bot >= z_top:
            z_bot = z_top - 1000.0

        # Tick steps in their native units (degrees / metres).
        lon_step_deg = _nice_step(
            (x_max - x_min) / max(terrain.metres_per_deg_lon, 1.0), _TARGET_TICKS,
        )
        lat_step_deg = _nice_step(
            (y_max - y_min) / max(terrain.metres_per_deg_lat, 1.0), _TARGET_TICKS,
        )
        depth_step_m = _nice_step((z_top - z_bot) / 1000.0, _TARGET_TICKS) * 1000.0

        lon_ticks_world, lon_tick_vals = _ticks_along_axis_geo(
            center_value=terrain.center_lon,
            metres_per_unit=terrain.metres_per_deg_lon,
            local_min=x_min, local_max=x_max,
            step=lon_step_deg,
        )
        lat_ticks_world, lat_tick_vals = _ticks_along_axis_geo(
            center_value=terrain.center_lat,
            metres_per_unit=terrain.metres_per_deg_lat,
            local_min=y_min, local_max=y_max,
            step=lat_step_deg,
        )
        depth_ticks_world, depth_tick_vals = _ticks_along_axis_z(
            z_min=z_bot, z_max=z_top, step=depth_step_m,
            elevation_shift=float(terrain.elevation_shift_m),
        )

        # ── Pick visible vs hidden walls from the camera ──────────────────
        # The bbox is centred on the origin, so a negative camera coordinate
        # on an axis means the camera looks toward +X / +Y across that face,
        # which makes the corresponding "_min" face the visible (front) face.
        if camera_eye is None:
            cam_x, cam_y = -1.0, -1.0           # default SW view
        else:
            cam_x = float(camera_eye[0])
            cam_y = float(camera_eye[1])
        vis_x_min = cam_x < 0.5 * (x_min + x_max)
        vis_y_min = cam_y < 0.5 * (y_min + y_max)
        vis_x_val = x_min if vis_x_min else x_max
        vis_y_val = y_min if vis_y_min else y_max
        hid_x_val = x_max if vis_x_min else x_min
        hid_y_val = y_max if vis_y_min else y_min
        vis_x_sign = -1.0 if vis_x_min else +1.0
        vis_y_sign = -1.0 if vis_y_min else +1.0
        hid_y_sign = -vis_y_sign

        # ── Bounding-box edges ────────────────────────────────────────────
        verts: list[np.ndarray] = []
        cols: list[np.ndarray] = []

        # Coloured principal axes along the visible corner edge — that's the
        # corner closest to the camera, where the user can clearly see all
        # three coloured axes meet.
        AXIS_COLOR_X = (0.85, 0.35, 0.35, 0.95)      # red — longitude
        AXIS_COLOR_Y = (0.35, 0.80, 0.40, 0.95)      # green — latitude
        AXIS_COLOR_Z = (0.40, 0.55, 0.95, 0.95)      # blue — depth

        # X axis: bottom edge of the visible NS face (runs the full X span).
        _add_seg(verts, cols,
                 (x_min, vis_y_val, z_bot), (x_max, vis_y_val, z_bot),
                 AXIS_COLOR_X)
        # Y axis: bottom edge of the visible EW face.
        _add_seg(verts, cols,
                 (vis_x_val, y_min, z_bot), (vis_x_val, y_max, z_bot),
                 AXIS_COLOR_Y)
        # Z axis: vertical edge at the corner where the two visible walls meet.
        _add_seg(verts, cols,
                 (vis_x_val, vis_y_val, z_bot), (vis_x_val, vis_y_val, z_top),
                 AXIS_COLOR_Z)

        # Remaining 9 bbox edges in a muted grey so the box reads as a
        # box rather than three coloured rulers floating in space.
        EDGE = (0.55, 0.55, 0.60, 0.55)
        # Top loop (z_top, all 4 edges).
        _add_seg(verts, cols, (x_min, y_min, z_top), (x_max, y_min, z_top), EDGE)
        _add_seg(verts, cols, (x_min, y_max, z_top), (x_max, y_max, z_top), EDGE)
        _add_seg(verts, cols, (x_min, y_min, z_top), (x_min, y_max, z_top), EDGE)
        _add_seg(verts, cols, (x_max, y_min, z_top), (x_max, y_max, z_top), EDGE)
        # Bottom loop — only the two edges we didn't already paint as axes.
        _add_seg(verts, cols, (x_min, hid_y_val, z_bot), (x_max, hid_y_val, z_bot), EDGE)
        _add_seg(verts, cols, (hid_x_val, y_min, z_bot), (hid_x_val, y_max, z_bot), EDGE)
        # Three remaining vertical edges (the visible-corner one is the Z axis).
        _add_seg(verts, cols, (vis_x_val, hid_y_val, z_bot), (vis_x_val, hid_y_val, z_top), EDGE)
        _add_seg(verts, cols, (hid_x_val, vis_y_val, z_bot), (hid_x_val, vis_y_val, z_top), EDGE)
        _add_seg(verts, cols, (hid_x_val, hid_y_val, z_bot), (hid_x_val, hid_y_val, z_top), EDGE)

        # ── Gridlines on the back walls + floor ───────────────────────────
        # The two back walls (hid_x_val, hid_y_val) and the floor get the
        # gridlines; the terrain renders in front of them naturally.  The
        # two front walls deliberately get no gridlines so they don't paint
        # on top of the terrain.
        FAINT = (0.50, 0.50, 0.55, 0.30)

        # Back NS wall (y = hid_y_val).
        for x in lon_ticks_world:
            _add_seg(verts, cols,
                     (x, hid_y_val, z_bot), (x, hid_y_val, z_top), FAINT)
        for z in depth_ticks_world:
            _add_seg(verts, cols,
                     (x_min, hid_y_val, z), (x_max, hid_y_val, z), FAINT)

        # Back EW wall (x = hid_x_val).
        for y in lat_ticks_world:
            _add_seg(verts, cols,
                     (hid_x_val, y, z_bot), (hid_x_val, y, z_top), FAINT)
        for z in depth_ticks_world:
            _add_seg(verts, cols,
                     (hid_x_val, y_min, z), (hid_x_val, y_max, z), FAINT)

        # Floor (z_bot) — both directions of ticks.
        for x in lon_ticks_world:
            _add_seg(verts, cols,
                     (x, y_min, z_bot), (x, y_max, z_bot), FAINT)
        for y in lat_ticks_world:
            _add_seg(verts, cols,
                     (x_min, y, z_bot), (x_max, y, z_bot), FAINT)

        vertices = np.array(verts, dtype=np.float32)
        colors = np.array(cols, dtype=np.float32)

        # ── Tick labels at silhouette edges ───────────────────────────────
        # Each label has an outward unit vector (sum of the two face normals
        # meeting at its edge) so the overlay can offset the label off the
        # box silhouette into clear space.
        sqrt2 = math.sqrt(2.0)
        out_lon = np.array([0.0, vis_y_sign, -1.0], dtype=np.float32) / sqrt2
        out_lat = np.array([vis_x_sign, 0.0, -1.0], dtype=np.float32) / sqrt2
        # Depth labels live on the silhouette vertical edge between the
        # visible NS face and the hidden EW face — that edge is on the
        # outline of the box from the camera's angle, never inside it.
        # Outward = sum of the two adjacent face normals: the *hidden* EW
        # face (pointing -vis_x_sign) + the visible NS face (pointing
        # +vis_y_sign), so labels push horizontally outside the silhouette.
        out_depth = np.array(
            [-vis_x_sign, vis_y_sign, 0.0], dtype=np.float32,
        ) / sqrt2

        labels: List[TickLabel] = []
        # Lon labels — bottom edge of visible NS face.
        for x, val in zip(lon_ticks_world, lon_tick_vals):
            labels.append(TickLabel(
                world=np.array([x, vis_y_val, z_bot], dtype=np.float32),
                text=_fmt_lon(val), axis="lon", outward=out_lon,
            ))
        # Lat labels — bottom edge of visible EW face.
        for y, val in zip(lat_ticks_world, lat_tick_vals):
            labels.append(TickLabel(
                world=np.array([vis_x_val, y, z_bot], dtype=np.float32),
                text=_fmt_lat(val), axis="lat", outward=out_lat,
            ))
        # Depth labels — silhouette vertical edge (visible NS face × hidden
        # EW face), sitting on the outline of the projected box.
        for z, val in zip(depth_ticks_world, depth_tick_vals):
            labels.append(TickLabel(
                world=np.array([hid_x_val, vis_y_val, z], dtype=np.float32),
                text=_fmt_depth(val), axis="depth", outward=out_depth,
            ))

        return cls(
            vertices=vertices, colors=colors, labels=labels,
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            z_min=z_bot, z_max=z_top,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _nice_step(span: float, target_ticks: int) -> float:
    """Round-number tick step covering ``span`` with roughly ``target_ticks``.

    Returns a value from the 1-2-5 × 10^k family — same routine matplotlib
    uses under the hood.  Always > 0.
    """
    if span <= 0 or target_ticks <= 0:
        return 1.0
    raw = span / target_ticks
    if raw <= 0:
        return 1.0
    exp = np.floor(np.log10(raw))
    frac = raw / (10.0 ** exp)
    if frac < 1.5:
        nice = 1.0
    elif frac < 3.5:
        nice = 2.0
    elif frac < 7.5:
        nice = 5.0
    else:
        nice = 10.0
    return float(nice * (10.0 ** exp))


def _ticks_along_axis_geo(
    center_value: float, metres_per_unit: float,
    local_min: float, local_max: float, step: float,
) -> Tuple[List[float], List[float]]:
    """Tick positions on a geographic axis (lat or lon).

    Returns ``(local_metres, degree_value)`` lists matched element-wise.
    """
    if metres_per_unit <= 0 or step <= 0:
        return [], []
    # Convert local extent to degrees relative to the centre.
    deg_min = center_value + local_min / metres_per_unit
    deg_max = center_value + local_max / metres_per_unit
    if deg_min > deg_max:
        deg_min, deg_max = deg_max, deg_min

    # Round to the next step below deg_min.
    start = np.ceil(deg_min / step) * step
    ticks_deg = []
    v = start
    # Guard against float overshoot at the end of the range.
    while v <= deg_max + step * 1e-6:
        ticks_deg.append(float(v))
        v += step

    ticks_local = [
        float((deg - center_value) * metres_per_unit) for deg in ticks_deg
    ]
    return ticks_local, ticks_deg


def _ticks_along_axis_z(
    z_min: float, z_max: float, step: float, elevation_shift: float,
) -> Tuple[List[float], List[float]]:
    """Tick positions on the depth axis.

    Ticks are placed at round absolute elevations (i.e. real metres above
    sea level), but returned in the mesh-shifted Z frame the renderer uses.
    The corresponding ``label_values`` list is the absolute elevation in
    *kilometres* so labels read e.g. "−15 km".
    """
    if step <= 0:
        return [], []
    # Convert mesh-shifted Z back to absolute elevation (so ticks land on
    # round numbers in the user's mental model: 0 m sea level, etc.).
    abs_min = z_min + elevation_shift
    abs_max = z_max + elevation_shift
    if abs_min > abs_max:
        abs_min, abs_max = abs_max, abs_min

    start = np.ceil(abs_min / step) * step
    ticks_abs = []
    v = start
    while v <= abs_max + step * 1e-6:
        ticks_abs.append(float(v))
        v += step

    ticks_local = [float(t - elevation_shift) for t in ticks_abs]
    ticks_km = [t / 1000.0 for t in ticks_abs]
    return ticks_local, ticks_km


def _add_seg(
    verts: list, cols: list,
    a: Tuple[float, float, float], b: Tuple[float, float, float],
    rgba: Tuple[float, float, float, float],
):
    verts.append(a)
    verts.append(b)
    cols.append(rgba)
    cols.append(rgba)


def _fmt_lon(v: float) -> str:
    hemi = "E" if v >= 0 else "W"
    return f"{abs(v):.2f}°{hemi}"


def _fmt_lat(v: float) -> str:
    hemi = "N" if v >= 0 else "S"
    return f"{abs(v):.2f}°{hemi}"


def _fmt_depth(km: float) -> str:
    # Round to one decimal if step < 1 km; otherwise integers read cleaner.
    if abs(km) >= 10 or float(km).is_integer():
        return f"{km:+.0f} km"
    return f"{km:+.1f} km"
