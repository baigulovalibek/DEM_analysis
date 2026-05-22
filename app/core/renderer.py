"""
Converts a NumPy array to a QImage / base64 PNG string for Leaflet overlays.
"""
from __future__ import annotations
import base64
import io
from typing import Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

from PyQt6.QtGui import QImage


def array_to_rgba(
    data: np.ndarray,
    cmap: str = "terrain",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    nodata: Optional[float] = None,
    alpha: float = 1.0,
) -> np.ndarray:
    """Return (H, W, 4) uint8 RGBA array using the given matplotlib colormap."""
    arr = data.astype(np.float64)
    mask = np.zeros(arr.shape, dtype=bool)

    if nodata is not None:
        mask |= arr == nodata
    mask |= ~np.isfinite(arr)

    if vmin is None:
        valid = arr[~mask]
        vmin = float(np.percentile(valid, 2)) if valid.size else 0.0
    if vmax is None:
        valid = arr[~mask]
        vmax = float(np.percentile(valid, 98)) if valid.size else 1.0
    if vmax == vmin:
        vmax = vmin + 1.0

    norm = (arr - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)

    cm = plt.get_cmap(cmap)
    rgba = (cm(norm) * 255).astype(np.uint8)
    rgba[mask, 3] = 0                      # transparent where nodata
    rgba[~mask, 3] = int(alpha * 255)
    return rgba


def rgba_to_qimage(rgba: np.ndarray) -> QImage:
    h, w, _ = rgba.shape
    # QImage.Format.Format_RGBA8888 expects RGBA byte order
    img = QImage(rgba.tobytes(), w, h, QImage.Format.Format_RGBA8888)
    return img.copy()


def array_to_png_b64(
    data: np.ndarray,
    cmap: str = "terrain",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    nodata: Optional[float] = None,
) -> str:
    """Encode array as base64 PNG data-URL suitable for Leaflet imageOverlay."""
    rgba = array_to_rgba(data, cmap, vmin, vmax, nodata, alpha=1.0)
    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def hillshade_rgba(
    hs: np.ndarray,
    overlay: Optional[np.ndarray] = None,
    overlay_cmap: str = "terrain",
    overlay_alpha: float = 0.5,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Blend a hillshade (uint8) with an optional overlay for a QGIS-style look."""
    h, w = hs.shape
    hs_rgb = np.stack([hs, hs, hs], axis=-1)

    if overlay is None:
        rgba = np.concatenate([hs_rgb, np.full((h, w, 1), 255, np.uint8)], axis=-1)
        return rgba

    overlay_rgba = array_to_rgba(overlay, overlay_cmap, vmin, vmax)
    ov_alpha = overlay_rgba[..., 3:4] / 255.0 * overlay_alpha
    blended = (hs_rgb * (1 - ov_alpha) + overlay_rgba[..., :3] * ov_alpha).astype(np.uint8)
    alpha_ch = np.full((h, w, 1), 255, dtype=np.uint8)
    return np.concatenate([blended, alpha_ch], axis=-1)
