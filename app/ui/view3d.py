"""
3D terrain view.

Two rendering backends are supported:

  * pyqtgraph's OpenGL view  — fast, GPU-accelerated; used when an OpenGL
    context is available.
  * a matplotlib 3-D surface — fully software-rendered; used as a fallback
    on machines without a GPU / OpenGL so the 3-D view still works.

The backend is chosen automatically at construction time.
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

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigCanvas
# Importing this registers the '3d' projection with matplotlib.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# A software-rendered surface must stay small to remain interactive.
_GL_MAX_DIM = 256
_MPL_MAX_DIM = 80


def _decimate(dem: np.ndarray, texture, max_dim: int):
    """Stride-subsample the DEM (and matching texture) to at most max_dim."""
    rows, cols = dem.shape
    if rows > max_dim or cols > max_dim:
        step = int(np.ceil(max(rows, cols) / max_dim))
        dem = dem[::step, ::step]
        if texture is not None:
            texture = texture[::step, ::step]
    return dem, texture


def _fill_nan(dem: np.ndarray) -> np.ndarray:
    """Replace non-finite cells with the minimum finite elevation."""
    finite = np.isfinite(dem)
    if finite.all():
        return dem
    fill = float(dem[finite].min()) if finite.any() else 0.0
    return np.where(finite, dem, fill)


class View3DDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("3D View", parent)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable  |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self._backend = None          # 'gl' | 'mpl' | None
        self._gl_widget = None
        self._mpl_canvas = None
        self._mpl_ax = None
        self._build_ui()

    # ── Backend setup ──────────────────────────────────────────────────────

    def _build_ui(self):
        if _HAS_GL:
            try:
                self._gl_widget = gl.GLViewWidget()
                self._gl_widget.setMinimumSize(300, 200)
                self._gl_widget.setCameraPosition(distance=200, elevation=30, azimuth=45)
                self.setWidget(self._gl_widget)
                self._backend = "gl"
                return
            except Exception:
                self._gl_widget = None

        # Software fallback — matplotlib 3-D surface (no OpenGL required).
        try:
            self._fig = Figure(figsize=(4, 3), facecolor="#1e1e1e")
            self._mpl_ax = self._fig.add_subplot(111, projection="3d")
            self._mpl_canvas = FigCanvas(self._fig)
            self._mpl_canvas.setMinimumSize(300, 200)
            self._style_mpl_axes()
            self.setWidget(self._mpl_canvas)
            self._backend = "mpl"
            return
        except Exception:
            self._mpl_canvas = None

        placeholder = QWidget()
        lbl = QLabel("3D view unavailable on this system.")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color:#666;")
        layout = QVBoxLayout(placeholder)
        layout.addWidget(lbl)
        self.setWidget(placeholder)
        self._backend = None

    def _style_mpl_axes(self):
        ax = self._mpl_ax
        ax.set_facecolor("#1e1e1e")
        try:
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_pane_color((0.14, 0.14, 0.14, 1.0))
            ax.tick_params(colors="#888", labelsize=6)
        except Exception:
            pass
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    # ── Public API ─────────────────────────────────────────────────────────

    def load_dem(
        self,
        dem: np.ndarray,
        cell_size: float = 1.0,
        texture: np.ndarray = None,
        z_exaggeration: float = 1.0,
    ):
        """Build a 3-D mesh from the DEM and optionally drape a texture."""
        if dem is None or dem.size == 0:
            return
        if self._backend == "gl":
            self._load_gl(dem, cell_size, texture, z_exaggeration)
        elif self._backend == "mpl":
            self._load_mpl(dem, cell_size, texture, z_exaggeration)

    # ── matplotlib (software) backend ──────────────────────────────────────

    def _load_mpl(self, dem, cell_size, texture, z_exaggeration):
        dem = _fill_nan(np.asarray(dem, dtype=np.float64))
        dem, texture = _decimate(dem, texture, _MPL_MAX_DIM)
        rows, cols = dem.shape
        if rows < 2 or cols < 2:
            return

        ax = self._mpl_ax
        ax.clear()
        self._style_mpl_axes()

        xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
        zs = dem

        if texture is not None and texture.shape[:2] == dem.shape:
            facecolors = np.clip(texture[..., :4].astype(np.float64) / 255.0, 0.0, 1.0)
            ax.plot_surface(xs, ys, zs, facecolors=facecolors,
                            rstride=1, cstride=1, linewidth=0,
                            antialiased=False, shade=False)
        else:
            ax.plot_surface(xs, ys, zs, cmap="terrain",
                            rstride=1, cstride=1, linewidth=0, antialiased=False)

        # Stretch the bounding box vertically so relief is visible.
        span = max(rows, cols)
        try:
            ax.set_box_aspect((cols, rows, 0.45 * span * max(z_exaggeration, 0.01)))
        except Exception:
            pass
        ax.view_init(elev=35, azim=-60)
        self._mpl_canvas.draw_idle()

    # ── pyqtgraph OpenGL backend ───────────────────────────────────────────

    def _load_gl(self, dem, cell_size, texture, z_exaggeration):
        if self._gl_widget is None:
            return

        self._gl_widget.clear()
        self._gl_widget.setCameraPosition(distance=200, elevation=30, azimuth=45)

        dem = _fill_nan(np.asarray(dem, dtype=np.float64))
        dem, texture = _decimate(dem, texture, _GL_MAX_DIM)
        rows, cols = dem.shape
        if rows < 2 or cols < 2:
            return

        z = dem.astype(np.float32) * z_exaggeration
        z_range = float(z.max() - z.min())
        if z_range < 1:
            z_range = 1.0
        scale = 100.0 / max(rows, cols)

        xi = np.arange(cols) * scale * cell_size
        yi = np.arange(rows) * scale * cell_size
        xs, ys = np.meshgrid(xi, yi)
        zs = (z - z.min()) / z_range * 50.0

        verts = np.zeros((rows, cols, 3), dtype=np.float32)
        verts[..., 0] = xs
        verts[..., 1] = ys
        verts[..., 2] = zs

        # Two triangles per cell — built vectorised.
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

        if texture is not None and texture.shape[:2] == dem.shape:
            colors = texture.reshape(-1, 4).astype(np.float32) / 255.0
            face_colors = (colors[faces_arr[:, 0]]
                           + colors[faces_arr[:, 1]]
                           + colors[faces_arr[:, 2]]) / 3
        else:
            heights = verts_flat[:, 2]
            h_norm = (heights - heights.min()) / (heights.max() - heights.min() + 1e-6)
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap("terrain")
            vertex_colors = cmap(h_norm).astype(np.float32)
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

        grid = gl.GLGridItem()
        grid.scale(scale * cols / 10, scale * rows / 10, 1)
        grid.translate(scale * cols / 2, scale * rows / 2, -1)
        self._gl_widget.addItem(grid)

        self._gl_widget.setCameraPosition(
            distance=max(cols, rows) * scale * 1.5,
            elevation=30,
            azimuth=45,
        )
