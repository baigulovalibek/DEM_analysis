"""
2D overlays painted on top of the GL viewport: compass rose, axes triad,
and the coordinate-grid tick labels.

These widgets are transparent and ignore mouse events (so the GL widget below
still receives them).  They watch a ``Scene3D`` and repaint when the camera
azimuth changes.

Why the labels live here instead of inside ``GLViewport.paintGL``: text
rendered straight into a multisampled GL framebuffer (the viewport asks for
``samples=4`` MSAA) gets blurred during the MSAA resolve — glyph fragments
don't benefit from box-filtering the way edges do.  Painting labels into a
separate, non-MSAA child widget gives Qt's normal sub-pixel text pipeline
and keeps the text crisp.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from PyQt6.QtCore import Qt, QPoint, QPointF, QRectF
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QFontMetrics
from PyQt6.QtWidgets import QWidget


def _make_overlay(parent: QWidget) -> QWidget:
    """Common widget setup for click-through overlays."""
    w = QWidget(parent)
    w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
    w.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
    w.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
    return w


class CompassRose(QWidget):
    """Top-right compass.  The needle rotates so 'N' always points to world north."""

    SIZE = 60
    MARGIN = 12

    def __init__(self, get_azimuth_rad: Callable[[], float], parent: QWidget):
        super().__init__(parent)
        self._get_az = get_azimuth_rad
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedSize(self.SIZE, self.SIZE)

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        r = self.SIZE / 2 - 4
        cx = self.SIZE / 2
        cy = self.SIZE / 2

        # Bezel
        p.setPen(QPen(QColor(220, 220, 220, 180), 1.3))
        p.setBrush(QBrush(QColor(20, 20, 22, 170)))
        p.drawEllipse(QPointF(cx, cy), r, r)

        # Needle.  Azimuth is what direction the camera looks toward.  N on the
        # compass should point opposite to the camera's view direction, so we
        # rotate the rose by -azimuth.
        p.save()
        p.translate(cx, cy)
        p.rotate(-math.degrees(self._get_az()))

        # North half — red
        p.setBrush(QBrush(QColor(220, 60, 60)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(QPointF(0, -r * 0.82), QPointF(-r * 0.18, 0),
                      QPointF(r * 0.18, 0))
        # South half — light grey
        p.setBrush(QBrush(QColor(200, 200, 200)))
        p.drawPolygon(QPointF(0, r * 0.82), QPointF(-r * 0.18, 0),
                      QPointF(r * 0.18, 0))

        # Cardinal letters
        p.setPen(QColor(230, 230, 230))
        f = QFont(self.font())                # inherit the app font (valid pt size)
        f.setBold(True)
        f.setPointSize(8)
        p.setFont(f)
        fm = QFontMetrics(f)
        for letter, dx, dy in [
            ("N", 0, -r + 8),
            ("S", 0,  r - 2),
            ("E",  r - 6, 4),
            ("W", -r + 4, 4),
        ]:
            w = fm.horizontalAdvance(letter)
            p.drawText(QPointF(dx - w / 2, dy), letter)
        p.restore()


class AxesTriad(QWidget):
    """Bottom-left 3D axes (X red / Y green / Z blue)."""

    SIZE = 70
    MARGIN = 12

    def __init__(self, get_view_basis: Callable[[], tuple],
                 parent: QWidget):
        """``get_view_basis`` returns (right_vec, up_vec) of the camera, each a
        2-tuple of screen-space (dx, dy) for the world +X, +Y, +Z axes.  Easier
        for the parent window to compute from the OrbitCamera than duplicating
        the math here.
        """
        super().__init__(parent)
        self._get_basis = get_view_basis
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedSize(self.SIZE, self.SIZE)

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        cx, cy = self.SIZE / 2, self.SIZE / 2
        L = self.SIZE * 0.40

        try:
            axes = self._get_basis()  # ((x,y), (x,y), (x,y)) for X,Y,Z
        except Exception:
            return

        colors = [QColor(220, 80, 80),  QColor(120, 200, 80), QColor(90, 140, 230)]
        labels = ["E", "N", "Z"]

        # Draw furthest-first so labels of nearer axes overlap correctly.
        depths = [(i, axes[i][2] if len(axes[i]) > 2 else 0.0) for i in range(3)]
        depths.sort(key=lambda x: -x[1])

        for i, _z in depths:
            dx, dy = axes[i][0], axes[i][1]
            ex = cx + dx * L
            ey = cy + dy * L
            p.setPen(QPen(colors[i], 2.0))
            p.drawLine(QPointF(cx, cy), QPointF(ex, ey))

            p.setPen(colors[i])
            f = QFont(self.font())
            f.setBold(True)
            f.setPointSize(8)
            p.setFont(f)
            p.drawText(QRectF(ex - 8, ey - 14, 16, 12),
                       Qt.AlignmentFlag.AlignCenter, labels[i])

        # Origin dot
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(220, 220, 220)))
        p.drawEllipse(QPointF(cx, cy), 2.0, 2.0)


class GridLabelsOverlay(QWidget):
    """Transparent full-viewport overlay that paints coordinate-grid tick
    labels with Qt's normal text pipeline (sharp), not on the MSAA GL
    framebuffer (blurry).

    The widget projects each ``TickLabel`` through the current camera MVP,
    skips labels behind the camera or off-screen, and renders the survivors
    with a small dark backdrop for legibility against any terrain colour.
    """

    def __init__(self, scene, viewport: QWidget, parent: QWidget):
        super().__init__(parent)
        self._scene = scene
        self._viewport = viewport
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

    def paintEvent(self, _ev):
        s = self._scene
        if (s.grid is None or s.terrain is None
                or not s.settings.show_grid
                or not s.settings.show_grid_labels
                or not s.grid.labels):
            return
        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            return
        aspect = w / max(h, 1)
        mvp = s.camera.mvp(aspect)
        z_factor = float(s.settings.z_factor)

        # 4% of the longest bbox extent as the outward sampling distance —
        # comfortably outside the box without being so far the projection
        # collapses into the back of the scene.
        bbox_extent = max(
            s.grid.x_max - s.grid.x_min,
            s.grid.y_max - s.grid.y_min,
            s.grid.z_max - s.grid.z_min,
            1.0,
        )
        outward_m = 0.04 * bbox_extent
        # Final on-screen push so labels sit fully clear of the silhouette.
        screen_push_px = 18.0

        # Pre-compute camera eye for occlusion ray-tests.  Labels whose
        # camera→anchor segment crosses the terrain mesh or an earthquake
        # marker are hidden so they don't paint over geometry that ought to
        # be in front of them.  Both tests use the same z_factor scaling
        # the GL pipeline applies in its vertex shader, so picking matches
        # the visible image.
        eye = s.camera.eye().astype(np.float64)
        margin = 4
        screen: list[tuple[float, float, str, str, np.ndarray]] = []
        for lbl in s.grid.labels:
            wx = float(lbl.world[0])
            wy = float(lbl.world[1])
            wz = float(lbl.world[2]) * z_factor
            clip = mvp @ np.array([wx, wy, wz, 1.0], dtype=np.float64)
            if clip[3] <= 1e-6:
                continue
            ndc = clip[:3] / clip[3]
            sx = (ndc[0] * 0.5 + 0.5) * w
            sy = (1.0 - (ndc[1] * 0.5 + 0.5)) * h

            # Project a point offset along the outward direction in world
            # space; the screen delta tells us which way "outside the box"
            # points on the projected image.  Normalise to a fixed pixel
            # push so labels sit at a constant distance off the silhouette
            # regardless of zoom.
            outward = lbl.outward
            wx2 = wx + float(outward[0]) * outward_m
            wy2 = wy + float(outward[1]) * outward_m
            wz2 = wz + float(outward[2]) * outward_m * z_factor
            clip2 = mvp @ np.array([wx2, wy2, wz2, 1.0], dtype=np.float64)
            if clip2[3] > 1e-6:
                ndc2 = clip2[:3] / clip2[3]
                sx2 = (ndc2[0] * 0.5 + 0.5) * w
                sy2 = (1.0 - (ndc2[1] * 0.5 + 0.5)) * h
                dx = sx2 - sx
                dy = sy2 - sy
                n = math.sqrt(dx * dx + dy * dy)
                if n > 1e-3:
                    sx += (dx / n) * screen_push_px
                    sy += (dy / n) * screen_push_px

            if sx < -margin or sx > w + margin or sy < -margin or sy > h + margin:
                continue
            anchor = np.array([wx, wy, wz], dtype=np.float64)
            screen.append((sx, sy, lbl.text, lbl.axis, anchor))

        if not screen:
            return

        # Hide labels whose camera→anchor ray is blocked by terrain or by
        # an earthquake marker — the bbox edges themselves get depth-tested
        # in GL, and the labels should follow the same occlusion rule.
        if screen:
            anchors_arr = np.stack([a for *_, a in screen], axis=0)
            occluded = self._compute_occlusion(eye, anchors_arr, z_factor)
            screen = [item for item, occ in zip(screen, occluded) if not occ]
        if not screen:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        font = QFont("Segoe UI")
        font.setStyleHint(QFont.StyleHint.SansSerif)
        font.setPixelSize(11)
        font.setWeight(QFont.Weight.Normal)
        painter.setFont(font)
        fm = QFontMetrics(font)

        # Slimmer styling than the previous attempt — the goal is a sub-
        # ordinate readout next to the gridline, not a banner.  Backdrop
        # alpha down to ~160 so it's there for legibility but doesn't
        # block the scene.
        backdrop = QBrush(QColor(20, 22, 28, 160))
        text_default = QColor(230, 234, 244)
        axis_tints = {
            "lon":   QColor(230, 175, 175),
            "lat":   QColor(175, 220, 185),
            "depth": QColor(185, 200, 240),
        }

        for sx, sy, text, axis, _anchor in screen:
            tw = fm.horizontalAdvance(text)
            th = fm.height()
            # Centre the label on the offset point rather than anchoring to
            # the bottom-left — combined with the outward push, the label
            # sits cleanly off the silhouette in all eight orbit quadrants.
            rect = QRectF(sx - tw / 2.0 - 3, sy - th / 2.0,
                          tw + 6, th)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(backdrop)
            painter.drawRoundedRect(rect, 2, 2)
            painter.setPen(axis_tints.get(axis, text_default))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.end()

    # ── Occlusion ─────────────────────────────────────────────────────────────

    def _compute_occlusion(
        self, eye: np.ndarray, anchors: np.ndarray, z_factor: float,
    ) -> np.ndarray:
        """Return a (L,) bool array — True where a label is hidden by scene.

        Two checks per label:

        * **Terrain** — march the camera→anchor segment in a small number of
          steps; if the terrain surface at the sample's (x, y) is above the
          segment's z at that point, the label sits behind the mountain.
        * **Earthquakes** — vectorised ray-vs-sphere (or ray-vs-cylinder)
          test against every visible instance.  A hit with parameter ``t``
          strictly less than the distance to the anchor counts as an
          occluder in front of the label.

        Both tests apply ``z_factor`` to terrain elevations and earthquake
        centres so the picking matches the visible image (the GL vertex
        shader scales Z by the same factor).
        """
        s = self._scene
        L = anchors.shape[0]
        if L == 0:
            return np.zeros(0, dtype=bool)
        occluded = np.zeros(L, dtype=bool)

        # ── Terrain occlusion ─────────────────────────────────────────────
        terrain = s.terrain
        if terrain is not None:
            n_steps = 32
            # Skip both endpoints — anchor sits exactly on the bbox edge and
            # eye is outside the DEM; ratios 1..n_steps-1 exclusive both ends.
            ts = (np.arange(1, n_steps, dtype=np.float64) / n_steps)
            for i in range(L):
                if occluded[i]:
                    continue
                d = anchors[i] - eye
                hit = False
                for t in ts:
                    p = eye + d * t
                    tz = terrain.elevation_at(float(p[0]), float(p[1]))
                    if tz is None:
                        continue
                    if tz * z_factor > p[2]:
                        hit = True
                        break
                if hit:
                    occluded[i] = True

        # ── Earthquake occlusion ──────────────────────────────────────────
        eq = s.earthquakes
        if (eq is not None and eq.n_instances > 0
                and getattr(eq.style, "visible", True)):
            D = anchors - eye[None, :]                     # (L, 3)
            t_max = np.linalg.norm(D, axis=1)              # (L,)
            valid_ray = t_max > 1e-9
            if np.any(valid_ray):
                D_unit = D / np.maximum(t_max[:, None], 1e-9)
                centres = eq.bottom.astype(np.float64).copy()
                centres[:, 2] *= z_factor
                r = eq.radius.astype(np.float64)
                # Don't count a hit inside the last metre — labels sit on the
                # silhouette edge, often touching a nearby sphere.
                t_limit = np.maximum(t_max - 1.0, 0.0)
                shape = getattr(eq.style, "shape", "cylinder")
                if shape == "sphere":
                    diff = eye[None, :] - centres          # (N, 3)
                    b = D_unit @ diff.T                    # (L, N)
                    c = (diff * diff).sum(axis=1) - r * r  # (N,)
                    disc = b * b - c[None, :]              # (L, N)
                    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
                    t_entry = -b - sqrt_disc               # (L, N)
                    hit = (disc >= 0.0) & (t_entry > 0.0) & (t_entry < t_limit[:, None])
                else:
                    cx = centres[:, 0]
                    cy = centres[:, 1]
                    top_z = eq.top.astype(np.float64)[:, 2] * z_factor
                    z_lo = np.minimum(centres[:, 2], top_z)
                    z_hi = np.maximum(centres[:, 2], top_z)
                    Dx = D_unit[:, 0]; Dy = D_unit[:, 1]; Dz = D_unit[:, 2]
                    a_coef = Dx * Dx + Dy * Dy             # (L,)
                    nonzero = a_coef > 1e-12
                    ex_cx = eye[0] - cx                    # (N,)
                    ey_cy = eye[1] - cy                    # (N,)
                    h_b = Dx[:, None] * ex_cx[None, :] + Dy[:, None] * ey_cy[None, :]
                    c_coef = ex_cx * ex_cx + ey_cy * ey_cy - r * r
                    disc = h_b * h_b - a_coef[:, None] * c_coef[None, :]
                    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
                    t_entry = np.where(
                        nonzero[:, None],
                        (-h_b - sqrt_disc) / np.maximum(a_coef[:, None], 1e-12),
                        -1.0,
                    )
                    z_hit = eye[2] + t_entry * Dz[:, None]
                    in_z = (z_hit >= z_lo[None, :]) & (z_hit <= z_hi[None, :])
                    hit = (disc >= 0.0) & (t_entry > 0.0) & (t_entry < t_limit[:, None]) & in_z
                eq_occluded = hit.any(axis=1) & valid_ray
                occluded |= eq_occluded

        return occluded
