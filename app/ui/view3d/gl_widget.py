"""
OpenGL viewport for the 3D scene.

A QOpenGLWidget subclass that owns the GL resources (VAO, VBOs, IBO, drape
texture, shader program).  It does not own the scene data — it reads from a
``Scene3D`` and re-uploads only what the dirty bits say has changed.

Requires OpenGL 3.3 core (released 2010 — present on essentially all
graphics drivers from the last 12 years).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QTimer
from PyQt6.QtGui import QSurfaceFormat, QMouseEvent, QWheelEvent, QKeyEvent
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

# All low-level GL calls go through PyOpenGL — far less verbose than the
# context().functions() route and the only sane way to write shader plumbing.
try:
    from OpenGL import GL as gl
    from OpenGL.GL import shaders as gl_shaders
    HAS_GL = True
    GL_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover — driver/wheel issue surfaces at runtime
    print(f"[view3d] PyOpenGL import failed: {exc}", file=sys.stderr)
    HAS_GL = False
    GL_IMPORT_ERROR = str(exc)


def has_opengl() -> bool:
    """True iff PyOpenGL was imported successfully."""
    return HAS_GL


def opengl_error() -> str:
    """Human-readable explanation of why PyOpenGL is unavailable, or ''."""
    return GL_IMPORT_ERROR

from app.core.scene3d import Scene3D, build_unit_cylinder, build_unit_sphere


def _shader_root() -> Path:
    """Locate the ``shaders/`` directory in dev and in a PyInstaller bundle.

    The spec mirrors the source tree under ``_MEIPASS`` so the same
    ``app/ui/view3d/shaders/`` path works once we anchor to ``_MEIPASS``
    when frozen instead of the entry-script-relative ``__file__``.
    """
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass) / "app" / "ui" / "view3d" / "shaders"
    return Path(__file__).resolve().parent / "shaders"


_SHADER_DIR = _shader_root()

# Side-count for the unit cylinder used by each earthquake stick.  16 is the
# sweet spot — at typical zooms the side facets disappear into the apparent
# diameter, but the geometry stays light (<100 triangles per stick incl. caps).
_STICK_SEGMENTS = 16

# Sphere resolution.  12 rings × 18 segments ≈ 400 triangles — a sweet spot
# between visual smoothness at typical zooms and instancing thousands of
# events without a frame-rate hit on integrated graphics.
_SPHERE_RINGS = 12
_SPHERE_SEGMENTS = 18


# ── Default surface format ────────────────────────────────────────────────────
#
# Must be set BEFORE the first QOpenGLWidget exists.  ``set_default_format`` is
# called from main.py (and harmlessly idempotent if called again).


def set_default_format():
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSamples(4)                              # MSAA for cleaner ridges
    QSurfaceFormat.setDefaultFormat(fmt)


# ── Viewport ──────────────────────────────────────────────────────────────────


class GLViewport(QOpenGLWidget):
    """OpenGL renderer for the terrain scene."""

    # Emitted whenever the camera changes — wired up to drive the 2D map
    # frustum overlay.  Throttled to ~30 Hz so the map repaint doesn't melt.
    camera_changed = pyqtSignal()
    # Same trigger, but fires on EVERY nudge (no throttle).  In-viewport
    # overlays (grid labels) hook this so they paint in lockstep with the
    # GL framebuffer; visible lag was the "draggy labels" symptom.
    camera_changed_immediate = pyqtSignal()
    # Emitted on mouse hover when the cursor sits over the terrain.
    # (world_x_m, world_y_m, world_z_m, lat, lon)
    cursor_over_terrain = pyqtSignal(float, float, float, float, float)
    cursor_off_terrain = pyqtSignal()
    # Emitted on mouse hover when the cursor lands on an earthquake instance.
    # Payload is the catalog index (i.e. ``earthquakes.idx[hit]``) so the
    # window can look up magnitude / depth / time straight from the catalog.
    event_hovered = pyqtSignal(int)
    event_unhovered = pyqtSignal()

    def __init__(self, scene: Scene3D, parent=None):
        super().__init__(parent)
        self._scene = scene
        self._program = 0
        self._vao = 0
        self._vbo_pos = 0
        self._vbo_norm = 0
        self._vbo_uv = 0
        self._ibo = 0
        self._n_indices = 0
        self._tex = 0
        self._tex_w = 0
        self._tex_h = 0
        self._uniform_locs: dict[str, int] = {}
        self._gl_ready = False
        self._camera_mode = "orbit"           # "orbit" | "walk"

        # Earthquake-stick renderer state.  All zero until initializeGL runs.
        self._stick_program = 0
        self._stick_uniform_locs: dict[str, int] = {}
        self._stick_vao = 0
        self._stick_vbo_pos = 0       # unit cylinder vertex positions
        self._stick_vbo_norm = 0      # unit cylinder vertex normals
        self._stick_ibo = 0           # unit cylinder triangle indices
        self._stick_n_indices = 0
        self._stick_vbo_bottom = 0    # per-instance (x, y, z) hypocenter
        self._stick_vbo_top = 0       # per-instance (x, y, z) surface anchor
        self._stick_vbo_radius = 0    # per-instance radius
        self._stick_vbo_rgba = 0      # per-instance colour
        self._stick_n_instances = 0   # last-uploaded count

        # Earthquake-sphere renderer state.  Shares the per-instance VBOs with
        # the stick pipeline (sphere shader reads ``i_bottom`` as the centre
        # and ignores ``i_top``) but uses its own program, VAO, and unit-sphere
        # base mesh.
        self._sphere_program = 0
        self._sphere_uniform_locs: dict[str, int] = {}
        self._sphere_vao = 0
        self._sphere_vbo_pos = 0
        self._sphere_vbo_norm = 0
        self._sphere_ibo = 0
        self._sphere_n_indices = 0

        # Coordinate grid renderer state.  Lines + per-vertex colour with no
        # lighting; the GL widget only re-uploads when ``dirty.grid`` is set.
        self._grid_program = 0
        self._grid_uniform_locs: dict[str, int] = {}
        self._grid_vao = 0
        self._grid_vbo_pos = 0
        self._grid_vbo_color = 0
        self._grid_n_vertices = 0


        # Input state
        self._last_mouse_pos: Optional[QPoint] = None
        self._mouse_button: Optional[Qt.MouseButton] = None
        self._keys_down: set[int] = set()
        self._walk_timer = QTimer(self)
        self._walk_timer.setInterval(16)            # ~60 Hz
        self._walk_timer.timeout.connect(self._tick_walk)

        # Throttle camera_changed emissions so the 2D map overlay doesn't
        # repaint at GL framerate.
        self._cam_emit_timer = QTimer(self)
        self._cam_emit_timer.setSingleShot(True)
        self._cam_emit_timer.setInterval(33)        # ~30 Hz
        self._cam_emit_timer.timeout.connect(self.camera_changed.emit)

        # Throttled cursor-on-terrain picker.  Ray-marching ~2000 sample
        # iterations on every mousemove blows the CPU; gate it to ~20 Hz and
        # only when no mouse button is down (we don't need the cursor readout
        # mid-drag, when the camera is moving).
        self._cursor_pick_timer = QTimer(self)
        self._cursor_pick_timer.setSingleShot(True)
        self._cursor_pick_timer.setInterval(50)     # ~20 Hz
        self._cursor_pick_timer.timeout.connect(self._do_cursor_pick)
        self._cursor_last_pos: Optional[QPoint] = None

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ── Scene wiring ──────────────────────────────────────────────────────────

    @property
    def scene(self) -> Scene3D:
        return self._scene

    def set_scene(self, scene: Scene3D):
        self._scene = scene
        self._scene.dirty.mark_all()
        self.update()

    def request_redraw(self):
        """Public hook external code uses after mutating the scene."""
        self.update()

    def set_camera_mode(self, mode: str):
        """Switch between orbit and walk mouse mappings.

        Both modes use the same OrbitCamera under the hood; only the meaning
        of the mouse drag changes.  Keyboard (WASD/QE) behaves identically.
        """
        m = mode.lower()
        if m not in ("orbit", "walk"):
            m = "orbit"
        self._camera_mode = m

    # ── GL lifecycle ──────────────────────────────────────────────────────────

    def initializeGL(self):
        if not HAS_GL:
            self._gl_ready = False
            return
        self._program = self._build_program()
        if self._program == 0:
            self._gl_ready = False
            return
        self._init_uniforms()
        self._init_buffers()
        # Sticks share the same context.  Failure to compile the stick shader
        # is non-fatal — terrain still renders, sticks just don't.
        self._stick_program = self._build_stick_program()
        if self._stick_program:
            self._init_stick_uniforms()
            self._init_stick_buffers()
        # Spheres reuse the per-instance VBOs created by the stick pipeline,
        # so they must come after _init_stick_buffers.
        self._sphere_program = self._build_sphere_program()
        if self._sphere_program and self._stick_program:
            self._init_sphere_uniforms()
            self._init_sphere_buffers()
        # Coordinate grid: simple unlit lines; non-fatal if it fails to build.
        self._grid_program = self._build_grid_program()
        if self._grid_program:
            self._init_grid_uniforms()
            self._init_grid_buffers()
        # Point sprites require PROGRAM_POINT_SIZE so the vertex shader can
        # write gl_PointSize.  Cheap on every driver from the last decade.
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glEnable(gl.GL_DEPTH_TEST)
        # Terrain is an open surface, not closed geometry — back-face culling
        # buys nothing here and turning it on punishes any winding mistake by
        # making the mesh look broken.  Off is the safe default for terrain.
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glClearColor(*self._scene.settings.background, 1.0)
        self._gl_ready = True

    def resizeGL(self, w: int, h: int):
        if not self._gl_ready:
            return
        gl.glViewport(0, 0, max(w, 1), max(h, 1))
        # Aspect changed → MVP changes → re-emit for the 2D frustum.
        self._cam_emit_timer.start()

    def paintGL(self):
        if not self._gl_ready:
            return
        s = self._scene
        bg = s.settings.background
        gl.glClearColor(bg[0], bg[1], bg[2], 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        if s.terrain is None:
            return

        if s.dirty.mesh:
            self._upload_mesh()
        if s.dirty.texture:
            self._upload_texture()
        # Stick upload must happen BEFORE dirty.clear() — otherwise the bit
        # is wiped and the per-instance VBOs never receive their first
        # glBufferData call, so the instanced draw reads from empty buffers
        # and renders nothing.
        if self._stick_program and s.dirty.earthquakes and s.earthquakes is not None:
            self._upload_earthquakes()
        if self._grid_program and s.dirty.grid:
            self._upload_grid()
        # Uniforms are cheap; always set them.
        self._set_uniforms()
        s.dirty.clear()

        if s.settings.wireframe:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

        gl.glUseProgram(self._program)
        gl.glBindVertexArray(self._vao)
        gl.glDrawElements(gl.GL_TRIANGLES, self._n_indices,
                          gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)

        if s.settings.wireframe:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        # Coordinate grid: drawn before earthquakes so the grid is occluded by
        # earthquake geometry it sits behind.  Drawn AFTER terrain so the
        # lines drape correctly on the visible side and disappear behind the
        # terrain on the far side.
        if self._grid_program and s.grid is not None and s.settings.show_grid:
            self._draw_grid()

        # Earthquakes: draw after terrain so the depth buffer already contains
        # the surface — partially-buried sticks behind ridges occlude
        # correctly.  Dispatch to either the cylinder or sphere pipeline based
        # on the catalog's current shape style.
        if s.earthquakes is not None and s.earthquakes.style.visible:
            shape = getattr(s.earthquakes.style, "shape", "cylinder")
            if shape == "sphere" and self._sphere_program:
                self._draw_spheres()
            elif self._stick_program:
                self._draw_sticks()

        # Tick labels are NOT drawn here — they live on a separate non-MSAA
        # overlay widget (see overlays.GridLabelsOverlay) so the standard Qt
        # text pipeline can render them crisply.  Drawing them straight onto
        # the MSAA GL framebuffer blurred them during MSAA resolve.

    # ── Program / shaders ─────────────────────────────────────────────────────

    def _build_program(self) -> int:
        try:
            with open(_SHADER_DIR / "terrain.vert", encoding="utf-8") as f:
                vsrc = f.read()
            with open(_SHADER_DIR / "terrain.frag", encoding="utf-8") as f:
                fsrc = f.read()
            vs = gl_shaders.compileShader(vsrc, gl.GL_VERTEX_SHADER)
            fs = gl_shaders.compileShader(fsrc, gl.GL_FRAGMENT_SHADER)
            prog = gl_shaders.compileProgram(vs, fs, validate=False)
            return int(prog)
        except Exception as exc:
            print(f"[view3d] Shader compile/link failed: {exc}", file=sys.stderr)
            return 0

    def _init_uniforms(self):
        names = [
            "u_mvp", "u_model", "u_zFactor",
            "u_drape", "u_sunDir", "u_ambient", "u_background",
            "u_hasTexture", "u_zMin", "u_zMax",
        ]
        self._uniform_locs = {
            n: gl.glGetUniformLocation(self._program, n) for n in names
        }

    def _init_buffers(self):
        # Allocate but don't fill.  Real upload happens on first paint when
        # the scene has a terrain mesh.
        self._vao = gl.glGenVertexArrays(1)
        self._vbo_pos = gl.glGenBuffers(1)
        self._vbo_norm = gl.glGenBuffers(1)
        self._vbo_uv = gl.glGenBuffers(1)
        self._ibo = gl.glGenBuffers(1)
        self._tex = gl.glGenTextures(1)

        gl.glBindTexture(gl.GL_TEXTURE_2D, self._tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    # ── Upload ────────────────────────────────────────────────────────────────

    def _upload_mesh(self):
        t = self._scene.terrain
        if t is None:
            return
        gl.glBindVertexArray(self._vao)

        # Position
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_pos)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, t.vertices.nbytes, t.vertices,
                        gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        # Normal
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_norm)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, t.normals.nbytes, t.normals,
                        gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        # UV
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_uv)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, t.uvs.nbytes, t.uvs,
                        gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        # Indices
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._ibo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, t.indices.nbytes,
                        t.indices, gl.GL_STATIC_DRAW)
        self._n_indices = t.indices.size

        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

    def _upload_texture(self):
        rgba = self._scene.drape
        if rgba is None:
            # Mark "no texture" via the uniform; the procedural ramp kicks in.
            return
        h, w = rgba.shape[:2]
        # Ensure contiguous bytes for glTexImage2D
        rgba = np.ascontiguousarray(rgba)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._tex)
        # Row alignment: numpy gives tightly-packed RGBA so 1-byte alignment is safe.
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, w, h, 0,
                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, rgba.tobytes())
        self._tex_w = w
        self._tex_h = h
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def _set_uniforms(self):
        s = self._scene
        gl.glUseProgram(self._program)

        aspect = self.width() / max(self.height(), 1)
        mvp = s.camera.mvp(aspect).astype(np.float32)
        # GL is column-major; numpy is row-major.  We pass GL_TRUE to transpose
        # at upload so the maths above can stay in textbook row-major form.
        gl.glUniformMatrix4fv(self._uniform_locs["u_mvp"], 1, gl.GL_TRUE, mvp)

        ident = np.eye(4, dtype=np.float32)
        gl.glUniformMatrix4fv(self._uniform_locs["u_model"], 1, gl.GL_TRUE, ident)

        gl.glUniform1f(self._uniform_locs["u_zFactor"], float(s.settings.z_factor))
        # ambient=1.0 collapses the Lambert term to pure albedo — that's how
        # the "Shading off" toggle is implemented without touching the shader.
        ambient = 1.0 if not s.sun.enabled else float(s.sun.ambient)
        gl.glUniform1f(self._uniform_locs["u_ambient"], ambient)

        bg = s.settings.background
        gl.glUniform3f(self._uniform_locs["u_background"], bg[0], bg[1], bg[2])

        sun = s.sun.direction()
        gl.glUniform3f(self._uniform_locs["u_sunDir"], float(sun[0]), float(sun[1]), float(sun[2]))

        if s.terrain is not None:
            gl.glUniform1f(self._uniform_locs["u_zMin"], float(s.terrain.z_min))
            gl.glUniform1f(self._uniform_locs["u_zMax"], float(s.terrain.z_max))
        else:
            gl.glUniform1f(self._uniform_locs["u_zMin"], 0.0)
            gl.glUniform1f(self._uniform_locs["u_zMax"], 1.0)

        has_tex = 1 if s.drape is not None else 0
        gl.glUniform1i(self._uniform_locs["u_hasTexture"], has_tex)

        if has_tex:
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._tex)
            gl.glUniform1i(self._uniform_locs["u_drape"], 0)

    # ── Earthquake sticks ─────────────────────────────────────────────────────

    def _build_stick_program(self) -> int:
        """Compile + link the instanced-cylinder shader pair."""
        try:
            with open(_SHADER_DIR / "sticks.vert", encoding="utf-8") as f:
                vsrc = f.read()
            with open(_SHADER_DIR / "sticks.frag", encoding="utf-8") as f:
                fsrc = f.read()
            vs = gl_shaders.compileShader(vsrc, gl.GL_VERTEX_SHADER)
            fs = gl_shaders.compileShader(fsrc, gl.GL_FRAGMENT_SHADER)
            prog = gl_shaders.compileProgram(vs, fs, validate=False)
            return int(prog)
        except Exception as exc:
            print(f"[view3d] Stick shader compile/link failed: {exc}", file=sys.stderr)
            return 0

    def _init_stick_uniforms(self):
        names = ["u_mvp", "u_zFactor", "u_sunDir", "u_ambient"]
        self._stick_uniform_locs = {
            n: gl.glGetUniformLocation(self._stick_program, n) for n in names
        }

    def _init_stick_buffers(self):
        """Allocate VAO + VBOs and upload the unit-cylinder base mesh once.

        Per-instance VBOs are created here but left empty; the first
        ``_upload_earthquakes`` call fills them with real data.
        """
        verts, norms, indices = build_unit_cylinder(segments=_STICK_SEGMENTS)
        self._stick_n_indices = int(indices.size)

        self._stick_vao = gl.glGenVertexArrays(1)
        self._stick_vbo_pos = gl.glGenBuffers(1)
        self._stick_vbo_norm = gl.glGenBuffers(1)
        self._stick_ibo = gl.glGenBuffers(1)
        self._stick_vbo_bottom = gl.glGenBuffers(1)
        self._stick_vbo_top = gl.glGenBuffers(1)
        self._stick_vbo_radius = gl.glGenBuffers(1)
        self._stick_vbo_rgba = gl.glGenBuffers(1)

        gl.glBindVertexArray(self._stick_vao)

        # Per-vertex unit cylinder geometry (slots 0, 1).
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._stick_vbo_pos)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._stick_vbo_norm)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, norms.nbytes, norms, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._stick_ibo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes,
                        indices, gl.GL_STATIC_DRAW)

        # Per-instance attributes (slots 2..5).  Buffers stay unallocated until
        # the first upload; we just enable the attribute slots + advance rate.
        for loc, vbo, components in (
            (2, self._stick_vbo_bottom, 3),
            (3, self._stick_vbo_top,    3),
            (4, self._stick_vbo_radius, 1),
            (5, self._stick_vbo_rgba,   4),
        ):
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, components, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            gl.glVertexAttribDivisor(loc, 1)        # advance once per instance

        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

    def _upload_earthquakes(self):
        """Re-upload per-instance VBOs from the current Scene3D earthquakes."""
        eq = self._scene.earthquakes
        if eq is None:
            self._stick_n_instances = 0
            return
        n = eq.n_instances
        self._stick_n_instances = n
        if n == 0:
            return

        # GL_DYNAMIC_DRAW because the user can mutate the catalog (filter
        # sliders) and the buffer gets re-uploaded each time.
        for vbo, data in (
            (self._stick_vbo_bottom, eq.bottom),
            (self._stick_vbo_top,    eq.top),
            (self._stick_vbo_radius, eq.radius),
            (self._stick_vbo_rgba,   eq.rgba),
        ):
            arr = np.ascontiguousarray(data, dtype=np.float32)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, arr.nbytes, arr, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def _draw_sticks(self):
        """Issue the instanced draw for all earthquake cylinders."""
        eq = self._scene.earthquakes
        if eq is None or self._stick_n_instances <= 0:
            return

        s = self._scene
        aspect = self.width() / max(self.height(), 1)
        mvp = s.camera.mvp(aspect).astype(np.float32)

        gl.glUseProgram(self._stick_program)
        gl.glUniformMatrix4fv(self._stick_uniform_locs["u_mvp"],
                              1, gl.GL_TRUE, mvp)
        gl.glUniform1f(self._stick_uniform_locs["u_zFactor"],
                       float(s.settings.z_factor))
        ambient = 1.0 if not s.sun.enabled else float(s.sun.ambient)
        gl.glUniform1f(self._stick_uniform_locs["u_ambient"], ambient)
        sun = s.sun.direction()
        gl.glUniform3f(self._stick_uniform_locs["u_sunDir"],
                       float(sun[0]), float(sun[1]), float(sun[2]))

        # Sticks hang *below* the terrain surface (top = surface, bottom =
        # surface − depth).  With the depth buffer already populated by the
        # terrain pass, every stick fragment would fail the depth test
        # against the surface above it and render invisibly.  Clearing
        # only the depth buffer lets sticks emerge into view; keeping
        # depth test enabled within the stick pass preserves mutual
        # ordering so a closer stick still covers a farther one.
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)

        # Alpha blending so semi-transparent sticks composite cleanly.
        # We deliberately leave depth-write enabled — at the default opacity
        # (~0.9) sticks read as nearly opaque and the simpler depth path
        # avoids the per-frame back-to-front sort a proper transparent
        # pipeline would need.
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        gl.glBindVertexArray(self._stick_vao)
        gl.glDrawElementsInstanced(
            gl.GL_TRIANGLES, self._stick_n_indices,
            gl.GL_UNSIGNED_INT, None, self._stick_n_instances,
        )
        gl.glBindVertexArray(0)

        gl.glDisable(gl.GL_BLEND)
        gl.glUseProgram(0)

    # ── Earthquake spheres ────────────────────────────────────────────────────

    def _build_sphere_program(self) -> int:
        try:
            with open(_SHADER_DIR / "spheres.vert", encoding="utf-8") as f:
                vsrc = f.read()
            with open(_SHADER_DIR / "spheres.frag", encoding="utf-8") as f:
                fsrc = f.read()
            vs = gl_shaders.compileShader(vsrc, gl.GL_VERTEX_SHADER)
            fs = gl_shaders.compileShader(fsrc, gl.GL_FRAGMENT_SHADER)
            prog = gl_shaders.compileProgram(vs, fs, validate=False)
            return int(prog)
        except Exception as exc:
            print(f"[view3d] Sphere shader compile/link failed: {exc}", file=sys.stderr)
            return 0

    def _init_sphere_uniforms(self):
        names = ["u_mvp", "u_zFactor", "u_sunDir", "u_ambient"]
        self._sphere_uniform_locs = {
            n: gl.glGetUniformLocation(self._sphere_program, n) for n in names
        }

    def _init_sphere_buffers(self):
        """Allocate VAO + base-mesh VBOs and re-bind the per-instance VBOs.

        The per-instance attributes (bottom, top, radius, rgba) are uploaded by
        the stick pipeline; the sphere VAO simply points at the same buffers
        so toggling shape on a 30k-event catalog is a state change, not a
        re-upload.
        """
        verts, norms, indices = build_unit_sphere(
            rings=_SPHERE_RINGS, segments=_SPHERE_SEGMENTS,
        )
        self._sphere_n_indices = int(indices.size)

        self._sphere_vao = gl.glGenVertexArrays(1)
        self._sphere_vbo_pos = gl.glGenBuffers(1)
        self._sphere_vbo_norm = gl.glGenBuffers(1)
        self._sphere_ibo = gl.glGenBuffers(1)

        gl.glBindVertexArray(self._sphere_vao)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._sphere_vbo_pos)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._sphere_vbo_norm)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, norms.nbytes, norms, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._sphere_ibo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes,
                        indices, gl.GL_STATIC_DRAW)

        # Reuse the stick pipeline's per-instance VBOs by binding them into
        # this VAO at the same attribute slots and the same divisor.
        for loc, vbo, components in (
            (2, self._stick_vbo_bottom, 3),
            (3, self._stick_vbo_top,    3),
            (4, self._stick_vbo_radius, 1),
            (5, self._stick_vbo_rgba,   4),
        ):
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, components, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            gl.glVertexAttribDivisor(loc, 1)

        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

    def _draw_spheres(self):
        """Issue the instanced draw for all earthquake spheres."""
        eq = self._scene.earthquakes
        if eq is None or self._stick_n_instances <= 0:
            return

        s = self._scene
        aspect = self.width() / max(self.height(), 1)
        mvp = s.camera.mvp(aspect).astype(np.float32)

        gl.glUseProgram(self._sphere_program)
        gl.glUniformMatrix4fv(self._sphere_uniform_locs["u_mvp"],
                              1, gl.GL_TRUE, mvp)
        gl.glUniform1f(self._sphere_uniform_locs["u_zFactor"],
                       float(s.settings.z_factor))
        ambient = 1.0 if not s.sun.enabled else float(s.sun.ambient)
        gl.glUniform1f(self._sphere_uniform_locs["u_ambient"], ambient)
        sun = s.sun.direction()
        gl.glUniform3f(self._sphere_uniform_locs["u_sunDir"],
                       float(sun[0]), float(sun[1]), float(sun[2]))

        # See _draw_sticks: clear the depth buffer between terrain and the
        # earthquake pass so spheres beneath the surface still render.  Mutual
        # depth-test within the sphere pass keeps near-occludes-far correct.
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        gl.glBindVertexArray(self._sphere_vao)
        gl.glDrawElementsInstanced(
            gl.GL_TRIANGLES, self._sphere_n_indices,
            gl.GL_UNSIGNED_INT, None, self._stick_n_instances,
        )
        gl.glBindVertexArray(0)

        gl.glDisable(gl.GL_BLEND)
        gl.glUseProgram(0)

    # ── Coordinate grid ───────────────────────────────────────────────────────

    def _build_grid_program(self) -> int:
        try:
            with open(_SHADER_DIR / "grid.vert", encoding="utf-8") as f:
                vsrc = f.read()
            with open(_SHADER_DIR / "grid.frag", encoding="utf-8") as f:
                fsrc = f.read()
            vs = gl_shaders.compileShader(vsrc, gl.GL_VERTEX_SHADER)
            fs = gl_shaders.compileShader(fsrc, gl.GL_FRAGMENT_SHADER)
            prog = gl_shaders.compileProgram(vs, fs, validate=False)
            return int(prog)
        except Exception as exc:
            print(f"[view3d] Grid shader compile/link failed: {exc}", file=sys.stderr)
            return 0

    def _init_grid_uniforms(self):
        self._grid_uniform_locs = {
            n: gl.glGetUniformLocation(self._grid_program, n)
            for n in ("u_mvp", "u_zFactor")
        }

    def _init_grid_buffers(self):
        self._grid_vao = gl.glGenVertexArrays(1)
        self._grid_vbo_pos = gl.glGenBuffers(1)
        self._grid_vbo_color = gl.glGenBuffers(1)

        gl.glBindVertexArray(self._grid_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._grid_vbo_pos)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._grid_vbo_color)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glBindVertexArray(0)

    def _upload_grid(self):
        g = self._scene.grid
        if g is None or g.vertices.size == 0:
            self._grid_n_vertices = 0
            return
        verts = np.ascontiguousarray(g.vertices, dtype=np.float32)
        colors = np.ascontiguousarray(g.colors, dtype=np.float32)
        gl.glBindVertexArray(self._grid_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._grid_vbo_pos)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._grid_vbo_color)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, colors.nbytes, colors, gl.GL_STATIC_DRAW)
        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        self._grid_n_vertices = int(verts.shape[0])

    def _draw_grid(self):
        if self._grid_n_vertices <= 0:
            return
        s = self._scene
        aspect = self.width() / max(self.height(), 1)
        mvp = s.camera.mvp(aspect).astype(np.float32)

        gl.glUseProgram(self._grid_program)
        gl.glUniformMatrix4fv(self._grid_uniform_locs["u_mvp"],
                              1, gl.GL_TRUE, mvp)
        gl.glUniform1f(self._grid_uniform_locs["u_zFactor"],
                       float(s.settings.z_factor))

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        # Translucent grid reads as background structure rather than competing
        # with the terrain.  Core-profile OpenGL 3.3 only guarantees a line
        # width of 1.0 — calling glLineWidth with anything else throws on most
        # drivers, so we live with the default thickness.
        gl.glBindVertexArray(self._grid_vao)
        gl.glDrawArrays(gl.GL_LINES, 0, self._grid_n_vertices)
        gl.glBindVertexArray(0)
        gl.glDisable(gl.GL_BLEND)
        gl.glUseProgram(0)

    # ── Camera state nudging helper ───────────────────────────────────────────

    def _notify_camera(self):
        self._scene.camera_changed()
        self.update()
        # Immediate signal so in-viewport overlays repaint in lockstep with
        # the GL framebuffer.  Throttling this caused labels to lag behind
        # the box by one tick during continuous drags.
        self.camera_changed_immediate.emit()
        if not self._cam_emit_timer.isActive():
            self._cam_emit_timer.start()

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def mousePressEvent(self, ev: QMouseEvent):
        self._last_mouse_pos = ev.position().toPoint()
        self._mouse_button = ev.button()
        self.setFocus()

    def mouseReleaseEvent(self, ev: QMouseEvent):
        self._mouse_button = None
        self._last_mouse_pos = None

    def mouseMoveEvent(self, ev: QMouseEvent):
        pos = ev.position().toPoint()
        if self._last_mouse_pos is None:
            self._last_mouse_pos = pos
            # First sample — schedule a hover pick if no button is down.
            if self._mouse_button is None:
                self._cursor_last_pos = pos
                if not self._cursor_pick_timer.isActive():
                    self._cursor_pick_timer.start()
            return
        dx = pos.x() - self._last_mouse_pos.x()
        dy = pos.y() - self._last_mouse_pos.y()
        self._last_mouse_pos = pos

        # Hover-pick when no drag is happening.
        if self._mouse_button is None:
            self._cursor_last_pos = pos
            if not self._cursor_pick_timer.isActive():
                self._cursor_pick_timer.start()

        if self._mouse_button == Qt.MouseButton.LeftButton:
            if self._camera_mode == "walk":
                # Walk mode: left-drag pans the ground.
                self._scene.camera.pan_screen(dx, dy, self.height())
            else:
                # Orbit: horizontal drag → azimuth, vertical drag → elevation
                self._scene.camera.orbit(
                    d_azimuth=math.radians(-dx * 0.3),
                    d_elevation=math.radians(dy * 0.3),
                )
            self._notify_camera()
        elif self._mouse_button == Qt.MouseButton.RightButton:
            if self._camera_mode == "walk":
                # In walk mode, right-drag rotates the view.
                self._scene.camera.orbit(
                    d_azimuth=math.radians(-dx * 0.3),
                    d_elevation=math.radians(dy * 0.3),
                )
            else:
                self._scene.camera.pan_screen(dx, dy, self.height())
            self._notify_camera()
        elif self._mouse_button == Qt.MouseButton.MiddleButton:
            # "Look around" — pitch + yaw without moving the camera.  Approximate
            # by orbiting in reverse: the eye stays put if we move the target
            # along an inverted-distance sphere.  Cheap enough for v1.
            self._scene.camera.orbit(
                d_azimuth=math.radians(-dx * 0.3),
                d_elevation=math.radians(-dy * 0.3),
            )
            self._notify_camera()

    def wheelEvent(self, ev: QWheelEvent):
        steps = ev.angleDelta().y() / 120.0          # one notch = ±1
        if not steps:
            return
        mods = ev.modifiers()
        if mods & Qt.KeyboardModifier.ShiftModifier:
            # FOV zoom
            new_fov = self._scene.camera.fov_y_deg * (0.9 ** steps)
            self._scene.camera.fov_y_deg = max(5.0, min(120.0, new_fov))
        elif mods & Qt.KeyboardModifier.ControlModifier:
            # Z-exaggeration
            self._scene.set_z_factor(
                self._scene.settings.z_factor * (1.15 ** steps)
            )
        else:
            self._scene.camera.zoom(0.85 ** steps)
        self._notify_camera()

    def mouseDoubleClickEvent(self, ev: QMouseEvent):
        # Recenter target on the clicked point (if it hits terrain).
        world = self._pick_terrain(ev.position().toPoint())
        if world is not None:
            self._scene.look_at_xy(float(world[0]), float(world[1]))
            self._notify_camera()

    # ── Keyboard ──────────────────────────────────────────────────────────────

    def keyPressEvent(self, ev: QKeyEvent):
        # Skip the press half of Qt's autoRepeat — the key is already in the
        # set, so adding it again is harmless, but ignoring the autoRepeat
        # event matches how we handle the release half (below) and keeps the
        # code symmetric.
        if ev.isAutoRepeat():
            return
        self._keys_down.add(ev.key())
        # Single-shot keys
        if ev.key() == Qt.Key.Key_Home:
            self._scene.frame_terrain()
            self._notify_camera()
        elif ev.key() == Qt.Key.Key_T:
            self._scene.top_view()
            self._notify_camera()
        elif ev.key() == Qt.Key.Key_N:
            self._scene.north_up()
            self._notify_camera()
        # Walk keys: keep the timer running while at least one is held.
        if self._keys_down & {Qt.Key.Key_W, Qt.Key.Key_A, Qt.Key.Key_S,
                              Qt.Key.Key_D, Qt.Key.Key_Q, Qt.Key.Key_E}:
            if not self._walk_timer.isActive():
                self._walk_timer.start()

    def keyReleaseEvent(self, ev: QKeyEvent):
        # Qt fires synthetic release+press pairs while a key auto-repeats on
        # some platforms.  Discarding the key on those releases would yank
        # the walk-key out of ``_keys_down`` mid-hold and stop the timer one
        # tick later when the next autoRepeat press arrives — visually, the
        # camera stutters or "ghosts" forward after the user lets go.
        # Ignore autoRepeat releases so only the real release clears the key.
        if ev.isAutoRepeat():
            return
        self._keys_down.discard(ev.key())
        if not (self._keys_down & {Qt.Key.Key_W, Qt.Key.Key_A, Qt.Key.Key_S,
                                   Qt.Key.Key_D, Qt.Key.Key_Q, Qt.Key.Key_E}):
            self._walk_timer.stop()

    def focusOutEvent(self, ev):
        """Clear key state when the GL widget loses keyboard focus.

        Without this, holding W while clicking the side panel never delivers
        a keyReleaseEvent (the focus moved away from us before the user let
        go), so W stays in ``_keys_down`` forever and the walk timer keeps
        panning the camera until the user comes back, presses W, then
        releases it.
        """
        self._keys_down.clear()
        self._walk_timer.stop()
        super().focusOutEvent(ev)

    def _tick_walk(self):
        """Per-frame keyboard-driven pan around the ground plane.

        Step size scales with the current zoom (~0.8% of camera distance per
        tick) with a small minimum so zoomed-in walks don't crawl.  Hold
        Shift for a 3× sprint when traversing wide regional catalogs.  We
        poll the live keyboard modifiers each tick because Shift isn't part
        of the WASD ``_keys_down`` set — it's a modifier, not a movement key.
        """
        from PyQt6.QtWidgets import QApplication
        cam = self._scene.camera
        # 0.8% per tick (~0.5×/sec at 60 Hz) gives a comfortable strolling
        # pace — 2.5% felt jumpy on regional DEMs where one tick of W would
        # cover a kilometre or more.  Shift sprints at 3× for traversal.
        step = max(cam.distance * 0.008, 75.0)
        mods = QApplication.keyboardModifiers()
        if mods & Qt.KeyboardModifier.ShiftModifier:
            step *= 3.0
        sin_az = math.sin(cam.azimuth)
        cos_az = math.cos(cam.azimuth)
        right = np.array([cos_az, -sin_az, 0.0], dtype=np.float64)
        fwd   = np.array([sin_az,  cos_az, 0.0], dtype=np.float64)
        delta = np.zeros(3, dtype=np.float64)
        if Qt.Key.Key_W in self._keys_down: delta += fwd
        if Qt.Key.Key_S in self._keys_down: delta -= fwd
        if Qt.Key.Key_D in self._keys_down: delta += right
        if Qt.Key.Key_A in self._keys_down: delta -= right
        if Qt.Key.Key_E in self._keys_down: delta[2] += 1.0
        if Qt.Key.Key_Q in self._keys_down: delta[2] -= 1.0
        if np.allclose(delta, 0):
            return
        cam.target = cam.target + delta * step
        self._notify_camera()

    # ── Cursor hover pick ─────────────────────────────────────────────────────

    def _do_cursor_pick(self):
        """Ray-march under the current cursor; emit lat/lon/elev when it hits.

        Earthquakes take priority over terrain so hovering a sphere right on
        top of the surface still shows the event readout instead of swapping
        between the two.
        """
        if self._cursor_last_pos is None or self._scene.terrain is None:
            self.cursor_off_terrain.emit()
            self.event_unhovered.emit()
            return

        # 1) Try earthquakes first.  Cheap (vectorised) and the user expects
        #    a hover on a stick/sphere to show its magnitude and depth.
        #    Crucially, we do NOT also emit cursor_off_terrain here — its
        #    slot resets the same status label we just wrote to, so the
        #    event readout would flash and disappear on every tick.
        eq_hit = self._pick_earthquake(self._cursor_last_pos)
        if eq_hit is not None:
            self.event_hovered.emit(int(eq_hit))
            return
        self.event_unhovered.emit()

        # 2) Terrain ray-march.
        world = self._pick_terrain(self._cursor_last_pos)
        if world is None:
            self.cursor_off_terrain.emit()
            return
        x, y, z = float(world[0]), float(world[1]), float(world[2])
        # World Z is the camera-space elevation × z_factor; report the
        # un-exaggerated elevation by sampling the mesh directly.
        terrain_z = self._scene.terrain.elevation_at(x, y)
        if terrain_z is None:
            self.cursor_off_terrain.emit()
            return
        lat, lon = self._scene.terrain.local_to_latlon(np.array([x, y]))
        self.cursor_over_terrain.emit(x, y, terrain_z, lat, lon)

    def _pick_earthquake(self, screen_pos: QPoint) -> Optional[int]:
        """Return the catalog index of the nearest event under ``screen_pos``.

        Vectorised ray-vs-instance test across every visible event:

        * Spheres — ray–sphere intersection at ``(centre, radius)``.
        * Cylinders — ray–side-surface intersection on a +Z axis cylinder
          with the Z range clamped to ``[z_bot, z_top]``.  End caps are not
          tested separately; for typical earthquake sticks (long, thin) the
          side surface is overwhelmingly what the user lands on.

        Z exaggeration is applied to the per-instance centres so picking
        matches what is actually drawn (the GL pipeline scales Z by
        ``u_zFactor`` in the vertex shader).
        """
        eq = self._scene.earthquakes
        if eq is None or self._stick_n_instances <= 0 or not eq.style.visible:
            return None

        # Build the ray through ``screen_pos`` exactly the way _pick_terrain
        # does so picking and the visible image stay in lockstep.
        aspect = self.width() / max(self.height(), 1)
        x_ndc = 2.0 * screen_pos.x() / max(self.width(), 1) - 1.0
        y_ndc = -2.0 * screen_pos.y() / max(self.height(), 1) + 1.0
        try:
            inv_proj = np.linalg.inv(self._scene.camera.projection_matrix(aspect))
            inv_view = np.linalg.inv(self._scene.camera.view_matrix())
        except np.linalg.LinAlgError:
            return None
        ndc = np.array([x_ndc, y_ndc, -1.0, 1.0], dtype=np.float64)
        view_pt = inv_proj @ ndc
        view_pt /= view_pt[3]
        world_pt = inv_view @ view_pt
        world_pt = (world_pt / world_pt[3])[:3]
        O = self._scene.camera.eye()
        D = world_pt - O
        n = float(np.linalg.norm(D))
        if n < 1e-9:
            return None
        D /= n

        z_factor = float(self._scene.settings.z_factor)
        # Per-instance positions, exaggerated to match the drawn geometry.
        bot = eq.bottom.astype(np.float64).copy()
        top = eq.top.astype(np.float64).copy()
        bot[:, 2] *= z_factor
        top[:, 2] *= z_factor
        r = eq.radius.astype(np.float64)

        shape = getattr(eq.style, "shape", "cylinder")
        if shape == "sphere":
            # Sphere centred at bot (== top by construction for spheres).
            centres = bot
            diff = O[None, :] - centres                        # (N, 3)
            b = (diff * D[None, :]).sum(axis=1)                # (N,)
            c = (diff * diff).sum(axis=1) - r * r              # (N,)
            disc = b * b - c
            hit = disc >= 0.0
            if not np.any(hit):
                return None
            sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
            t = -b - sqrt_disc                                  # entry parameter
            valid = hit & (t > 0.0)
            if not np.any(valid):
                # Camera inside a sphere — fall back to the exit point.
                t = -b + sqrt_disc
                valid = hit & (t > 0.0)
                if not np.any(valid):
                    return None
            t_arr = np.where(valid, t, np.inf)
            i = int(np.argmin(t_arr))
            if not np.isfinite(t_arr[i]):
                return None
            return int(eq.idx[i])

        # Cylinder side-surface intersection.  Vertical axis at (bot.xy),
        # radius r, Z bounds [z_bot, z_top] (which may be inverted).
        cx = bot[:, 0]
        cy = bot[:, 1]
        Dx, Dy = D[0], D[1]
        Ox, Oy = O[0], O[1]
        a_coef = Dx * Dx + Dy * Dy
        if a_coef < 1e-12:
            # Ray is vertical — only end-cap hits possible.  Not worth a
            # second code path; report no hit and let the user nudge the
            # camera.
            return None
        h_b = Dx * (Ox - cx) + Dy * (Oy - cy)                  # (N,) ½·b
        c_coef = (Ox - cx) ** 2 + (Oy - cy) ** 2 - r * r       # (N,)
        disc = h_b * h_b - a_coef * c_coef
        hit = disc >= 0.0
        if not np.any(hit):
            return None
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        t_entry = (-h_b - sqrt_disc) / a_coef
        z_bot_a = bot[:, 2]
        z_top_a = top[:, 2]
        z_lo = np.minimum(z_bot_a, z_top_a)
        z_hi = np.maximum(z_bot_a, z_top_a)
        z_hit = O[2] + t_entry * D[2]
        in_z = (z_hit >= z_lo) & (z_hit <= z_hi)
        valid = hit & (t_entry > 0.0) & in_z
        if not np.any(valid):
            # Try the back surface (camera inside the cylinder shell).
            t_exit = (-h_b + sqrt_disc) / a_coef
            z_hit2 = O[2] + t_exit * D[2]
            in_z2 = (z_hit2 >= z_lo) & (z_hit2 <= z_hi)
            valid = hit & (t_exit > 0.0) & in_z2
            if not np.any(valid):
                return None
            t_entry = t_exit
        t_arr = np.where(valid, t_entry, np.inf)
        i = int(np.argmin(t_arr))
        if not np.isfinite(t_arr[i]):
            return None
        return int(eq.idx[i])

    def leaveEvent(self, ev):
        self.cursor_off_terrain.emit()
        super().leaveEvent(ev)

    # ── Picking ───────────────────────────────────────────────────────────────

    def _pick_terrain(self, screen_pos: QPoint) -> Optional[np.ndarray]:
        """Reverse-project a screen point onto the terrain by stepping rays.

        Not the cleanest method, but it avoids glReadPixels (which forces a
        GL flush and adds frame latency).  Iterates along the ray, sampling
        terrain elevation until the ray drops below the surface.
        """
        if self._scene.terrain is None:
            return None
        aspect = self.width() / max(self.height(), 1)

        # NDC of the click position (origin top-left in Qt, bottom-left in NDC).
        x_ndc =  2.0 * screen_pos.x() / max(self.width(), 1) - 1.0
        y_ndc = -2.0 * screen_pos.y() / max(self.height(), 1) + 1.0

        try:
            inv_proj = np.linalg.inv(self._scene.camera.projection_matrix(aspect))
            inv_view = np.linalg.inv(self._scene.camera.view_matrix())
        except np.linalg.LinAlgError:
            return None

        ndc = np.array([x_ndc, y_ndc, -1.0, 1.0], dtype=np.float64)
        view_pt = inv_proj @ ndc
        view_pt /= view_pt[3]
        world_pt = inv_view @ view_pt
        world_pt = (world_pt / world_pt[3])[:3]
        eye = self._scene.camera.eye()
        direction = world_pt - eye
        n = float(np.linalg.norm(direction))
        if n < 1e-9:
            return None
        direction /= n

        # March the ray.  Step size starts large (1% of camera distance) and
        # gets refined once we cross the surface.
        t = 0.0
        max_t = float(self._scene.camera.far) * 0.95
        step = max(self._scene.camera.distance * 0.01, 1.0)
        prev_above = None
        z_factor = float(self._scene.settings.z_factor)
        for _ in range(2000):
            if t > max_t:
                return None
            p = eye + direction * t
            terrain_z = self._scene.terrain.elevation_at(float(p[0]), float(p[1]))
            if terrain_z is None:
                # We've left the DEM footprint without hitting the surface.
                prev_above = None
                t += step
                continue
            terrain_z *= z_factor
            above = p[2] > terrain_z
            if prev_above is None:
                prev_above = above
            elif above != prev_above:
                # Crossed the surface — bisect once for a cleaner hit point.
                lo, hi = t - step, t
                for _ in range(8):
                    mid = 0.5 * (lo + hi)
                    pm = eye + direction * mid
                    tz = self._scene.terrain.elevation_at(float(pm[0]), float(pm[1]))
                    if tz is None:
                        break
                    if (pm[2] > tz * z_factor) == prev_above:
                        lo = mid
                    else:
                        hi = mid
                hit = eye + direction * (0.5 * (lo + hi))
                return hit
            t += step
        return None

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> Optional["object"]:
        """Return a QImage of the current viewport contents."""
        return self.grabFramebuffer()
