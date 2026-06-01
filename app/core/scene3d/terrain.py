"""
Terrain mesh builder.

Takes a ``DemLayer`` and produces flat NumPy arrays ready for upload to a
single GL VAO: position, normal, uv, and triangle indices.  Optionally
downsamples large DEMs so the GPU mesh stays interactive.

Coordinate frame
----------------
The mesh is centred on the DEM and laid out in metres on a local tangent
plane:

    x  = (c - (cols - 1)/2) * cell_size_m       # east positive
    y  = ((rows - 1)/2 - r) * cell_size_m       # north positive (row 0 is north)
    z  = elevation                              # metres, NaN → z_min

Normals use a Sobel-weighted central-difference gradient (same Horn-1981
recipe as the hillshade analysis), then are divided by ``z_factor`` in the
vertex shader so changing the vertical exaggeration tilts the normals
correctly without a CPU recompute.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from app.core.dem_layer import DemLayer, GeoBounds


# Cap the number of cells per side in the GPU mesh.  256 cells² ≈ 65k verts
# ≈ 130k triangles — comfortable on integrated graphics, and the high-res
# drape texture means visual detail is preserved even with a coarse mesh.
DEFAULT_MAX_SIDE = 256


@dataclass
class TerrainGrid:
    """CPU-side mesh data.  Members are flat float32/int32 arrays ready for GL."""

    vertices: np.ndarray              # (N, 3) float32   x, y, z (metres, post-shift)
    normals: np.ndarray               # (N, 3) float32   unit, +z up
    uvs: np.ndarray                   # (N, 2) float32   (u, v)  v=0 at top row
    indices: np.ndarray               # (M, 3) uint32    triangle vertex indices
    rows: int                         # mesh rows (post-downsample)
    cols: int                         # mesh cols
    cell_size_m: float                # mesh cell size in metres (post-downsample)
    z_min: float                      # finite z range in MESH frame (always 0 after shift)
    z_max: float                      # post-shift: equals real elevation range
    extent_m: Tuple[float, float]     # (width_x, width_y) of the centred mesh
    # Mesh Z is shifted so the lowest sample sits at z=0; this is the offset
    # that turns shifted-mesh Z back into true metres-above-sea-level
    # (true_z = mesh_z + elevation_shift_m).  Used by the cursor readout
    # so the status bar still shows real elevations.
    elevation_shift_m: float = 0.0

    # Geographic anchor so we can map world XY back to lat/lon for 2D sync.
    bounds: Optional[GeoBounds] = None
    center_lat: float = 0.0
    center_lon: float = 0.0
    metres_per_deg_lon: float = 111320.0
    metres_per_deg_lat: float = 111320.0

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        dem: DemLayer,
        max_side: int = DEFAULT_MAX_SIDE,
        elev: Optional[np.ndarray] = None,
    ) -> "TerrainGrid":
        """Build mesh from a DEM layer.

        ``elev`` lets the caller pass a pre-masked elevation array (e.g.
        ``dem.valid_data``); when omitted the layer's masked accessor is used.
        """
        if elev is None:
            elev = dem.valid_data
        if elev.size == 0:
            raise ValueError("Cannot build terrain mesh from empty DEM")

        z = np.asarray(elev, dtype=np.float32)
        rows0, cols0 = z.shape

        # Downsample.  Use a fixed integer stride (block-averaging would smear
        # ridges and undercut the lighting; nearest stride keeps the relief
        # honest at the cost of a tiny bit of aliasing).
        step = 1
        if max(rows0, cols0) > max_side:
            step = int(np.ceil(max(rows0, cols0) / max_side))
        z_ds = z[::step, ::step]
        rows, cols = z_ds.shape

        # Replace NaN with the true min so the mesh stays watertight, then
        # shift the whole mesh so the lowest sample sits at z=0.  This
        # gives the rendered scene a clean "DEM rests on the floor" look
        # regardless of whether the DEM's absolute elevations live near sea
        # level, kilometres below, or kilometres above.  The shift is
        # remembered as ``elevation_shift_m`` so the cursor readout can
        # restore true elevations for display.  Drape sampling, normals,
        # and UVs are unaffected — they don't depend on absolute Z.
        finite = z_ds[np.isfinite(z_ds)]
        if finite.size:
            z_min_true = float(finite.min())
            z_max_true = float(finite.max())
        else:
            z_min_true = 0.0
            z_max_true = 1.0
        if not np.isfinite(z_min_true) or not np.isfinite(z_max_true):
            z_min_true, z_max_true = 0.0, 1.0
        elevation_shift_m = z_min_true
        z_filled = (
            np.where(np.isfinite(z_ds), z_ds, z_min_true).astype(np.float32)
            - z_min_true
        )
        z_min = 0.0
        z_max = z_max_true - z_min_true

        # Effective cell size after downsampling.  Use the DEM's reported
        # metres-per-cell × stride; that already accounts for geographic vs
        # projected CRS (computed in main_window when the DEM was loaded).
        cs = float(dem.cell_size_m) * float(step)

        width_x = cols * cs
        width_y = rows * cs

        # Vertex positions, centred on the origin.
        xs = (np.arange(cols, dtype=np.float32) - (cols - 1) / 2.0) * cs
        ys = ((rows - 1) / 2.0 - np.arange(rows, dtype=np.float32)) * cs
        X, Y = np.meshgrid(xs, ys)                        # both (rows, cols)
        verts = np.empty((rows, cols, 3), dtype=np.float32)
        verts[..., 0] = X
        verts[..., 1] = Y
        verts[..., 2] = z_filled

        # UVs.  v=0 at row 0 (north edge); the GL widget orients the texture
        # to match by uploading rows top-to-bottom (no flip).
        u = np.arange(cols, dtype=np.float32) / max(cols - 1, 1)
        v = np.arange(rows, dtype=np.float32) / max(rows - 1, 1)
        U, V = np.meshgrid(u, v)
        uvs = np.stack([U, V], axis=-1).astype(np.float32)

        # Normals.  np.gradient handles the boundary rows/cols by forward/backward
        # differences, so we don't need to special-case them.
        # dz/dy uses spacing = cs along axis 0 of the array (rows).  Note that
        # row 0 is north, so increasing row index → decreasing world Y.
        dz_dr, dz_dc = np.gradient(z_filled, cs)
        dz_dx = dz_dc                                # column index ↔ +X
        dz_dy = -dz_dr                               # row index ↔ -Y
        normals = np.empty((rows, cols, 3), dtype=np.float32)
        normals[..., 0] = -dz_dx
        normals[..., 1] = -dz_dy
        normals[..., 2] = 1.0
        nrm = np.linalg.norm(normals, axis=-1, keepdims=True)
        nrm[nrm < 1e-12] = 1.0
        normals /= nrm

        # Triangle indices.  Two triangles per quad, wound CCW when viewed from
        # above (camera at +Z looking down) so the surface normal points up
        # under OpenGL's default GL_CCW front-face convention.
        #
        # Vertex layout in world space:
        #   tl (row r,   col c)   = NW (low row → high Y, low col → low X)
        #   tr (row r,   col c+1) = NE
        #   bl (row r+1, col c)   = SW
        #   br (row r+1, col c+1) = SE
        #
        # CCW from above means: tl → bl → tr (NW → SW → NE) and tr → bl → br
        # (NE → SW → SE).  The previous (tl, tr, bl) order produced downward
        # normals, which combined with GL_BACK culling discarded the visible
        # face and left the terrain looking broken.
        rr, cc = np.meshgrid(
            np.arange(rows - 1, dtype=np.int64),
            np.arange(cols - 1, dtype=np.int64),
            indexing="ij",
        )
        tl = (rr * cols + cc).ravel()
        tr = tl + 1
        bl = tl + cols
        br = bl + 1
        faces = np.empty((tl.size * 2, 3), dtype=np.uint32)
        faces[0::2] = np.stack([tl, bl, tr], axis=1)
        faces[1::2] = np.stack([tr, bl, br], axis=1)

        # Geographic anchor (needed by 2D-sync code that converts world XY back
        # to lat/lon).
        if dem.bounds is not None:
            b = dem.bounds
            center_lat = b.center_lat
            center_lon = b.center_lon
            metres_per_deg_lat = 111320.0
            metres_per_deg_lon = 111320.0 * np.cos(np.radians(center_lat))
        else:
            center_lat = 0.0
            center_lon = 0.0
            metres_per_deg_lat = 111320.0
            metres_per_deg_lon = 111320.0

        return cls(
            vertices=verts.reshape(-1, 3),
            normals=normals.reshape(-1, 3),
            uvs=uvs.reshape(-1, 2),
            indices=faces,
            rows=rows,
            cols=cols,
            cell_size_m=cs,
            z_min=z_min,
            z_max=z_max,
            extent_m=(width_x, width_y),
            elevation_shift_m=float(elevation_shift_m),
            bounds=dem.bounds,
            center_lat=center_lat,
            center_lon=center_lon,
            metres_per_deg_lon=float(metres_per_deg_lon),
            metres_per_deg_lat=float(metres_per_deg_lat),
        )

    # ── Coordinate conversion ─────────────────────────────────────────────────

    def local_to_latlon(self, xy: np.ndarray) -> Tuple[float, float]:
        """Convert local-frame (x, y) metres back to (lat, lon).

        Uses the simple equirectangular approximation around the DEM centre.
        Accurate to better than a metre over typical DEM extents (< ~50 km
        on a side).
        """
        x, y = float(xy[0]), float(xy[1])
        lon = self.center_lon + x / self.metres_per_deg_lon
        lat = self.center_lat + y / self.metres_per_deg_lat
        return lat, lon

    def latlon_to_local(self, lat: float, lon: float) -> Tuple[float, float]:
        x = (lon - self.center_lon) * self.metres_per_deg_lon
        y = (lat - self.center_lat) * self.metres_per_deg_lat
        return x, y

    # ── Sampling ──────────────────────────────────────────────────────────────

    def elevation_at(self, x: float, y: float) -> Optional[float]:
        """Bilinear elevation lookup at local (x, y); ``None`` if outside grid."""
        cs = self.cell_size_m
        if cs <= 0:
            return None
        # Convert world XY back to fractional (row, col).
        col = (x / cs) + (self.cols - 1) / 2.0
        row = (self.rows - 1) / 2.0 - (y / cs)
        if row < 0 or row > self.rows - 1 or col < 0 or col > self.cols - 1:
            return None
        r0 = int(np.floor(row)); r1 = min(r0 + 1, self.rows - 1)
        c0 = int(np.floor(col)); c1 = min(c0 + 1, self.cols - 1)
        dr = row - r0
        dc = col - c0
        # Flat index lookup into self.vertices Z column.
        def z(rr, cc): return float(self.vertices[rr * self.cols + cc, 2])
        z00 = z(r0, c0); z01 = z(r0, c1)
        z10 = z(r1, c0); z11 = z(r1, c1)
        return (
            z00 * (1 - dr) * (1 - dc) +
            z01 * (1 - dr) * dc +
            z10 * dr * (1 - dc) +
            z11 * dr * dc
        )
