"""
3D terrain view using pyqtgraph OpenGL.

If OpenGL is unavailable (e.g. no GPU/display), the dock shows a placeholder.
"""
from __future__ import annotations
import numpy as np
from PyQt6.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

try:
    import pyqtgraph.opengl as gl
    _HAS_GL = True
except Exception:
    _HAS_GL = False


class View3DDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("3D View", parent)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable  |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self._gl_widget = None
        self._build_ui()

    def _build_ui(self):
        if _HAS_GL:
            try:
                self._gl_widget = gl.GLViewWidget()
                self._gl_widget.setMinimumSize(300, 200)
                self._gl_widget.setCameraPosition(distance=200, elevation=30, azimuth=45)
                self.setWidget(self._gl_widget)
                return
            except Exception:
                pass

        # Fallback placeholder
        placeholder = QWidget()
        lbl = QLabel("3D view requires OpenGL.\nNo compatible display detected.")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color:#666;")
        layout = QVBoxLayout(placeholder)
        layout.addWidget(lbl)
        self.setWidget(placeholder)

    def load_dem(
        self,
        dem: np.ndarray,
        cell_size: float = 1.0,
        texture: np.ndarray = None,
        z_exaggeration: float = 1.0,
    ):
        """
        Build a 3D mesh from the DEM and optionally texture it.

        Parameters
        ----------
        dem          : elevation array
        cell_size    : horizontal cell size (used to scale x/y)
        texture      : (H, W, 4) uint8 RGBA array to drape over the mesh
        z_exaggeration : vertical scale multiplier
        """
        if not _HAS_GL or self._gl_widget is None:
            return

        # Clear old items
        self._gl_widget.clear()
        self._gl_widget.setCameraPosition(distance=200, elevation=30, azimuth=45)

        rows, cols = dem.shape

        # Downsample large DEMs for interactive performance
        max_dim = 256
        if rows > max_dim or cols > max_dim:
            step_r = max(1, rows // max_dim)
            step_c = max(1, cols // max_dim)
            dem = dem[::step_r, ::step_c]
            if texture is not None:
                texture = texture[::step_r, ::step_c]
            rows, cols = dem.shape

        # Normalise elevation to fit nicely in view
        z = dem.astype(np.float32) * z_exaggeration
        z_range = z.max() - z.min()
        if z_range < 1:
            z_range = 1.0
        scale = 100.0 / max(rows, cols)

        # Build vertex grid
        xi = np.arange(cols) * scale * cell_size
        yi = np.arange(rows) * scale * cell_size
        xs, ys = np.meshgrid(xi, yi)
        zs = (z - z.min()) / z_range * 50.0   # normalise height to 0-50

        # Build face index array
        verts = np.zeros((rows, cols, 3), dtype=np.float32)
        verts[..., 0] = xs
        verts[..., 1] = ys
        verts[..., 2] = zs

        # Two triangles per cell — built vectorised (a Python double loop here
        # noticeably stalls the UI every time the 3D view is refreshed).
        rr, cc = np.meshgrid(
            np.arange(rows - 1), np.arange(cols - 1), indexing="ij"
        )
        tl = (rr * cols + cc).ravel()
        tr = tl + 1
        bl = tl + cols
        br = bl + 1
        faces_arr = np.empty((tl.size * 2, 3), dtype=np.int32)
        faces_arr[0::2] = np.stack([tl, tr, bl], axis=1)
        faces_arr[1::2] = np.stack([tr, br, bl], axis=1)

        verts_flat = verts.reshape(-1, 3)

        # Colour from texture or slope-based gradient
        if texture is not None:
            colors = texture.reshape(-1, 4).astype(np.float32) / 255.0
            face_colors = (colors[faces_arr[:, 0]] + colors[faces_arr[:, 1]] + colors[faces_arr[:, 2]]) / 3
        else:
            # Colour by normalised height
            heights = verts_flat[:, 2]
            h_norm  = (heights - heights.min()) / (heights.max() - heights.min() + 1e-6)
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap("terrain")
            vertex_colors = np.array([cmap(float(h)) for h in h_norm], dtype=np.float32)
            face_colors = (
                vertex_colors[faces_arr[:, 0]]
                + vertex_colors[faces_arr[:, 1]]
                + vertex_colors[faces_arr[:, 2]]
            ) / 3

        mesh = gl.GLMeshItem(
            vertexes=verts_flat,
            faces=faces_arr,
            faceColors=face_colors,
            smooth=True,
            drawEdges=False,
        )
        self._gl_widget.addItem(mesh)

        # Add ground grid
        grid = gl.GLGridItem()
        grid.scale(scale * cols / 10, scale * rows / 10, 1)
        grid.translate(scale * cols / 2, scale * rows / 2, -1)
        self._gl_widget.addItem(grid)

        # Centre camera
        self._gl_widget.setCameraPosition(
            distance=max(cols, rows) * scale * 1.5,
            elevation=30,
            azimuth=45,
        )


def pg_vector(x, y, z):
    """Return a pyqtgraph Vector if available."""
    try:
        from pyqtgraph import Vector
        return Vector(x, y, z)
    except Exception:
        return None
