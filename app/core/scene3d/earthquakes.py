"""
Stick visualisation for an :class:`EarthquakeCatalog` inside the 3D scene.

Each event becomes a vertical cylinder that hangs *below* the terrain:

* ``z_top = terrain elevation at the event's (lat, lon)``  — anchored at surface
* ``z_bot = z_top - depth_km * 1000``                      — drops by the depth

so a deeper event hangs farther down into the scene's lower half.  The
terrain mesh is itself shifted at build time so its lowest sample sits at
z=0 — sticks therefore live in the negative-Z region beneath the DEM
"floor", which reads naturally as "depth below the ground".

Width follows ``magnitude ** 1.5`` so a M2 → M7 catalog spans roughly a 6×
visual range — enough that strong events read as distinctly chunky against
small ones, but not so steep that energy-proportional (10^(1.5·M)) scaling
makes the high end explode.  Width is clamped to a configurable
``[min, max]`` band in world metres so micro-events stay visible and great
events don't eclipse the terrain.

Because the cylinders sit beneath the terrain mesh, the renderer clears
the depth buffer between the terrain pass and the stick pass — otherwise
every stick fragment would fail the depth test against the surface above
it and the catalog would render invisibly.  Sticks still depth-test
against each other so closer cylinders correctly cover farther ones.

The module is Qt-free / GL-free: it builds flat NumPy arrays ready for
upload as per-instance attributes by the GL widget's stick renderer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.core.earthquakes import EarthquakeCatalog
from app.core.scene3d.terrain import TerrainGrid


@dataclass
class EarthquakeRenderStyle:
    """3D-specific styling for the stick renderer.

    Kept separate from :class:`app.core.earthquakes.EarthquakeStyle` (which
    drives the 2D markers) so the two views can be tuned independently.
    Depth-to-colour, filters, and visibility still come from the catalog's
    own style — only width / opacity / capping live here.
    """

    visible: bool = True
    width_scale: float = 3.0          # multiplier on the base mag**1.5 width
    opacity: float = 0.9
    surface_cap: bool = True          # render top/bottom disc caps on each stick
    # "cylinder" hangs a stick from the surface to the hypocenter; "sphere"
    # renders just the hypocenter as a point at its lat/lon/depth.  Spheres
    # are the default — single markers at true (lat, lon, depth) read more
    # cleanly than sticks for most regional catalogs.
    shape: str = "sphere"
    # When True, the cylinder top / sphere centre is anchored to the terrain
    # elevation at the event's lat/lon.  When False (default), the anchor is
    # sea level (z=0 in mesh-shifted coordinates) so the event sits at its
    # true depth below datum regardless of the mountain above it — a
    # tomography-style cross-section is the natural read for a 3D catalog,
    # so absolute depth is the out-of-the-box behaviour.
    anchor_to_surface: bool = False
    # Width at M=1 (so M=1 → base width, M=4 → 8×, M=7 → ~18.5× before clamp).
    base_width_m: float = 80.0
    min_width_m: float = 20.0
    max_width_m: float = 8000.0
    # Exponent on magnitude.  1.5 gives ~6× width ratio between M2 and M7 —
    # roughly the contrast a reader expects when comparing a moderate quake
    # to a strong one without making microseismicity disappear.
    width_exponent: float = 1.5


@dataclass
class EarthquakeRenderable:
    """CPU-side per-instance arrays for the stick renderer.

    One row per visible event.  The GL widget uploads these straight into
    per-instance vertex buffers and renders all events in a single
    ``glDrawElementsInstanced`` call.
    """

    bottom: np.ndarray   # (N, 3) float32 — hypocenter world (x, y, z)
    top:    np.ndarray   # (N, 3) float32 — surface anchor world (x, y, z)
    radius: np.ndarray   # (N,)   float32 — stick half-width, metres
    rgba:   np.ndarray   # (N, 4) float32 — 0..1 with style opacity baked into alpha
    idx:    np.ndarray   # (N,)   int32   — catalog event index (reserved for picking)
    style:  EarthquakeRenderStyle = field(default_factory=EarthquakeRenderStyle)

    @property
    def n_instances(self) -> int:
        return int(self.bottom.shape[0])

    @classmethod
    def empty(cls, style: Optional[EarthquakeRenderStyle] = None) -> "EarthquakeRenderable":
        return cls(
            bottom=np.zeros((0, 3), dtype=np.float32),
            top=np.zeros((0, 3), dtype=np.float32),
            radius=np.zeros((0,), dtype=np.float32),
            rgba=np.zeros((0, 4), dtype=np.float32),
            idx=np.zeros((0,), dtype=np.int32),
            style=style or EarthquakeRenderStyle(),
        )

    @classmethod
    def build(
        cls,
        catalog: EarthquakeCatalog,
        terrain: TerrainGrid,
        style: Optional[EarthquakeRenderStyle] = None,
    ) -> "EarthquakeRenderable":
        """Build per-instance arrays for the catalog's currently-visible events.

        Events outside the terrain's footprint still render — they get hung
        from ``terrain.z_max`` instead of being dropped, so a regional
        catalog stays visible even when only part of it overlaps the DEM.
        """
        style = style or EarthquakeRenderStyle()

        visible = catalog.visible_indices()
        if visible.size == 0:
            return cls.empty(style)

        lats = catalog.lats[visible]
        lons = catalog.lons[visible]
        deps = catalog.depths_km[visible]
        mags = catalog.magnitudes[visible]

        # Lat/lon → terrain local frame (vectorised; mirrors
        # TerrainGrid.latlon_to_local but on whole arrays at once).
        xs = (lons - terrain.center_lon) * terrain.metres_per_deg_lon
        ys = (lats - terrain.center_lat) * terrain.metres_per_deg_lat

        # Top anchored at the (shifted) terrain surface; bottom drops by the
        # event's depth.  Stick *length* equals depth in metres, hanging
        # down into the scene's lower half.
        #
        # With ``anchor_to_surface=False`` the top sits at z=0 (sea level in
        # mesh-shifted coordinates) instead, so events display their true
        # depth below datum even when the terrain above is far from sea
        # level.  Useful when reading a regional catalog against tomography
        # cross-sections rather than as topographic decoration.
        if style.anchor_to_surface:
            z_top = _sample_terrain(terrain, xs, ys)
        else:
            z_top = np.zeros_like(xs, dtype=np.float32)
        z_bot = z_top - np.clip(deps, 0.0, None).astype(np.float32) * 1000.0

        bottom = np.stack(
            [xs.astype(np.float32), ys.astype(np.float32),
             z_bot.astype(np.float32)],
            axis=1,
        )
        top = np.stack(
            [xs.astype(np.float32), ys.astype(np.float32),
             z_top.astype(np.float32)],
            axis=1,
        )

        # Guarantee a minimum stick height so a 0-km-depth event still
        # renders as a small marker rather than a flat disc.
        too_thin = z_top - z_bot < 1.0
        if np.any(too_thin):
            bottom[too_thin, 2] = top[too_thin, 2] - 1.0

        # Width = base * mag**exponent * scale, clamped to the world-space
        # band.  mag**1.5 gives clearly differentiated stick thicknesses
        # across the typical M2–M7 catalog range without the runaway scaling
        # an energy-proportional law would produce.
        widths = (
            style.base_width_m
            * np.power(np.clip(mags, 0.0, None), float(style.width_exponent))
            * float(style.width_scale)
        )
        widths = np.clip(widths, style.min_width_m, style.max_width_m)
        radius = (0.5 * widths).astype(np.float32)

        # Colour: use the catalog's depth→colormap so the 2D markers and
        # the 3D sticks share a key.  Alpha overridden by the 3D opacity
        # slider; we don't multiply premultiplied into rgb because the
        # fragment shader will composite using straight alpha blending.
        rgba_u8 = catalog.colors_rgba(visible)
        rgba = rgba_u8.astype(np.float32) / 255.0
        rgba[:, 3] = float(style.opacity)

        return cls(
            bottom=np.ascontiguousarray(bottom, dtype=np.float32),
            top=np.ascontiguousarray(top, dtype=np.float32),
            radius=np.ascontiguousarray(radius, dtype=np.float32),
            rgba=np.ascontiguousarray(rgba, dtype=np.float32),
            idx=visible.astype(np.int32),
            style=style,
        )


def _sample_terrain(
    terrain: TerrainGrid, xs: np.ndarray, ys: np.ndarray,
) -> np.ndarray:
    """Vectorised bilinear elevation lookup for a batch of XY positions.

    Mirrors :py:meth:`TerrainGrid.elevation_at` but works on arrays without
    the per-event Python overhead.  Events outside the grid fall back to
    ``z_min`` (the scene's floor in shifted coordinates) — they hang from
    the floor going further down, which keeps them visually grouped at
    the bottom of the scene rather than implausibly anchored to a
    mountaintop.
    """
    cs = terrain.cell_size_m
    rows = terrain.rows
    cols = terrain.cols
    if cs <= 0 or rows < 2 or cols < 2:
        return np.full(xs.shape, float(terrain.z_min), dtype=np.float32)

    col = (xs / cs) + (cols - 1) / 2.0
    row = (rows - 1) / 2.0 - (ys / cs)

    inside = (row >= 0) & (row <= rows - 1) & (col >= 0) & (col <= cols - 1)

    r0 = np.clip(np.floor(row).astype(np.int64), 0, rows - 1)
    c0 = np.clip(np.floor(col).astype(np.int64), 0, cols - 1)
    r1 = np.clip(r0 + 1, 0, rows - 1)
    c1 = np.clip(c0 + 1, 0, cols - 1)
    dr = (row - r0).astype(np.float64)
    dc = (col - c0).astype(np.float64)

    z = terrain.vertices[:, 2]
    z00 = z[r0 * cols + c0]
    z01 = z[r0 * cols + c1]
    z10 = z[r1 * cols + c0]
    z11 = z[r1 * cols + c1]

    elev = (
        z00 * (1 - dr) * (1 - dc)
        + z01 * (1 - dr) * dc
        + z10 * dr * (1 - dc)
        + z11 * dr * dc
    )
    fallback = float(terrain.z_min)
    return np.where(inside, elev, fallback).astype(np.float32)


def build_unit_cylinder(segments: int = 16, with_caps: bool = True):
    """Build a unit cylinder mesh (axis +Z, radius 1, base at z=0, top at z=1).

    Returns ``(vertices, normals, indices)`` ready for GL upload.  Side
    normals are radial; cap normals are ±Z so the top reads as a flat disc
    when viewed from above (essential for the top-down camera view).
    """
    segments = max(3, int(segments))
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False, dtype=np.float32)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    zeros = np.zeros_like(cos_a)
    ones = np.ones_like(cos_a)

    # Side ring vertices (duplicated for bottom and top so we can attach the
    # right per-vertex normal even though the geometry is the same point).
    side_bot = np.stack([cos_a, sin_a, zeros], axis=1)
    side_top = np.stack([cos_a, sin_a, ones], axis=1)
    side_nrm = np.stack([cos_a, sin_a, zeros], axis=1)

    vertices = [side_bot, side_top]
    normals = [side_nrm, side_nrm]

    # Side triangles: stitch each adjacent pair of segments into two tris.
    # Vertex layout: bottom ring = [0..segments-1], top ring = [segments..2s-1].
    indices = []
    for i in range(segments):
        j = (i + 1) % segments
        b0 = i
        b1 = j
        t0 = segments + i
        t1 = segments + j
        # CCW when viewed from outside (radial-outward normal direction).
        indices.append((b0, b1, t1))
        indices.append((b0, t1, t0))

    offset = 2 * segments

    if with_caps:
        # Top cap: centre + ring with normal +Z.
        top_centre = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        top_ring = np.stack([cos_a, sin_a, ones], axis=1)
        top_nrm = np.tile([0.0, 0.0, 1.0], (segments + 1, 1)).astype(np.float32)
        vertices += [top_centre, top_ring]
        normals += [top_nrm[:1], top_nrm[1:]]
        centre_idx = offset
        ring_start = offset + 1
        for i in range(segments):
            j = (i + 1) % segments
            indices.append((centre_idx, ring_start + i, ring_start + j))
        offset += 1 + segments

        # Bottom cap: centre + ring with normal -Z.  Wind reversed so the
        # outward face points -Z.
        bot_centre = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        bot_ring = np.stack([cos_a, sin_a, zeros], axis=1)
        bot_nrm = np.tile([0.0, 0.0, -1.0], (segments + 1, 1)).astype(np.float32)
        vertices += [bot_centre, bot_ring]
        normals += [bot_nrm[:1], bot_nrm[1:]]
        centre_idx = offset
        ring_start = offset + 1
        for i in range(segments):
            j = (i + 1) % segments
            indices.append((centre_idx, ring_start + j, ring_start + i))
        offset += 1 + segments

    verts_arr = np.concatenate(vertices, axis=0).astype(np.float32)
    norms_arr = np.concatenate(normals, axis=0).astype(np.float32)
    idx_arr = np.asarray(indices, dtype=np.uint32)
    return verts_arr, norms_arr, idx_arr


def build_unit_sphere(rings: int = 12, segments: int = 18):
    """Build a UV-sphere mesh (radius 1, centred at origin).

    Returns ``(vertices, normals, indices)`` ready for GL upload.  Normals
    equal vertex positions (radial outward) which gives a clean Lambert
    shading and matches the unit-sphere case where ``|p| == 1``.

    ``rings`` × ``segments`` ≈ 200 triangles at the defaults — light enough
    for thousands of instanced spheres to stay interactive on integrated
    graphics.  Top and bottom poles use a single shared vertex each to
    avoid a degenerate strip seam.
    """
    rings = max(2, int(rings))
    segments = max(3, int(segments))

    verts: list[np.ndarray] = []
    norms: list[np.ndarray] = []

    # Top pole (a single vertex; normal is +Z).
    verts.append(np.array([[0.0, 0.0, 1.0]], dtype=np.float32))
    norms.append(np.array([[0.0, 0.0, 1.0]], dtype=np.float32))

    # Intermediate rings.  ``i`` runs 1..rings-1 so we get ``rings - 1``
    # latitude bands of vertices, capped above and below by the poles.
    for i in range(1, rings):
        phi = math.pi * i / rings              # 0 (top) → π (bottom)
        z = math.cos(phi)
        r = math.sin(phi)
        ang = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False,
                          dtype=np.float32)
        x = r * np.cos(ang)
        y = r * np.sin(ang)
        ring = np.stack([x, y, np.full_like(x, z, dtype=np.float32)], axis=1)
        verts.append(ring.astype(np.float32))
        norms.append(ring.astype(np.float32))

    # Bottom pole.
    verts.append(np.array([[0.0, 0.0, -1.0]], dtype=np.float32))
    norms.append(np.array([[0.0, 0.0, -1.0]], dtype=np.float32))

    vertices = np.concatenate(verts, axis=0).astype(np.float32)
    normals = np.concatenate(norms, axis=0).astype(np.float32)

    top_idx = 0
    bot_idx = vertices.shape[0] - 1
    # First ring starts right after the top pole.
    ring0_start = 1

    indices: list[tuple[int, int, int]] = []

    # Top cap fan.
    for s in range(segments):
        a = ring0_start + s
        b = ring0_start + (s + 1) % segments
        indices.append((top_idx, a, b))

    # Quad bands.  Each band has ``segments`` vertices; we stitch adjacent
    # rings into two triangles per segment.
    for i in range(rings - 2):
        row_a = ring0_start + i * segments
        row_b = row_a + segments
        for s in range(segments):
            sn = (s + 1) % segments
            a0 = row_a + s
            a1 = row_a + sn
            b0 = row_b + s
            b1 = row_b + sn
            indices.append((a0, b0, a1))
            indices.append((a1, b0, b1))

    # Bottom cap fan.  Last ring is at ring0_start + (rings - 2) * segments.
    last_ring = ring0_start + (rings - 2) * segments
    for s in range(segments):
        a = last_ring + s
        b = last_ring + (s + 1) % segments
        indices.append((bot_idx, b, a))

    idx_arr = np.asarray(indices, dtype=np.uint32)
    return vertices, normals, idx_arr
