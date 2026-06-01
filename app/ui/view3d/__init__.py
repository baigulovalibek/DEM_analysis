"""
3D view package.

Public entry points:
- ``View3DWindow``: the top-level Qt window
- ``set_default_format``: call once before QApplication is created so the
  default ``QSurfaceFormat`` requests a 3.3 core context.
"""
from __future__ import annotations

from app.ui.view3d.window import View3DWindow
from app.ui.view3d.gl_widget import set_default_format

__all__ = ["View3DWindow", "set_default_format"]
