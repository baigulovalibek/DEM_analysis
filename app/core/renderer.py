"""
Converts a NumPy array to a QImage / base64 PNG string for Leaflet overlays.
"""
from __future__ import annotations
import base64
import io
import warnings
from typing import Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

from PyQt6.QtGui import QImage


# Overlays larger than this (in either dimension) are downsampled before
# being encoded as a PNG — a huge base64 image makes the embedded browser
# sluggish when several layers are stacked.  The overlay stays georeferenced
# to the same bounds, so downsampling only lowers its display resolution.
_MAX_OVERLAY_DIM = 2048


def _decimate(
    arr: np.ndarray,
    max_dim: int = _MAX_OVERLAY_DIM,
    categorical: bool = False,
) -> np.ndarray:
    """Downsample a 2-D array so neither dimension exceeds ``max_dim``.

    Continuous data is block-averaged (NaN-aware): averaging preserves
    high-frequency detail as smooth gradients, where stride subsampling
    would throw away 1−1/step² of the data and produce a blocky result
    once the browser upscales the PNG.

    Categorical data (D8 codes, label grids) must use nearest-neighbour
    instead — averaging would invent codes that aren't in the LUT.
    """
    h, w = arr.shape[:2]
    if h <= max_dim and w <= max_dim:
        return arr
    step = int(np.ceil(max(h, w) / max_dim))

    if categorical:
        return arr[::step, ::step]

    # Crop to a multiple of `step` so the reshape works without leftovers.
    new_h = (h // step) * step
    new_w = (w // step) * step
    out_h = new_h // step
    out_w = new_w // step

    block = arr[:new_h, :new_w].astype(np.float32, copy=False)
    block = block.reshape(out_h, step, out_w, step)
    # nanmean preserves NaN only where every cell in the block is NaN.
    # The "Mean of empty slice" RuntimeWarning fires for those all-NaN blocks
    # and is exactly what we want — silence it to avoid console spam.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        downsampled = np.nanmean(block, axis=(1, 3))
    return downsampled


# Fixed colour LUTs for products whose values are categorical, not continuous.
# Each row is RGBA in 0–255.
_D8_CATEGORICAL_COLORS = {
    0:   (  0,   0,   0,   0),   # undrained / flat → transparent
    1:   (228,  26,  28, 255),   # E   — red
    2:   (255, 127,   0, 255),   # SE  — orange
    4:   (255, 255,  51, 255),   # S   — yellow
    8:   ( 77, 175,  74, 255),   # SW  — green
    16:  ( 55, 126, 184, 255),   # W   — blue
    32:  (152,  78, 163, 255),   # NW  — purple
    64:  (166,  86,  40, 255),   # N   — brown
    128: (247, 129, 191, 255),   # NE  — pink
}


def _categorical_d8_rgba(arr: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Direct code→colour mapping for D8 direction grids (no stretching)."""
    rgba = np.zeros(arr.shape + (4,), dtype=np.uint8)
    codes = arr.astype(np.int64)
    for code, color in _D8_CATEGORICAL_COLORS.items():
        m = codes == code
        if not m.any():
            continue
        rgba[m, 0] = color[0]
        rgba[m, 1] = color[1]
        rgba[m, 2] = color[2]
        rgba[m, 3] = int(color[3] * alpha) if code != 0 else 0
    return rgba


def array_to_rgba(
    data: np.ndarray,
    cmap: str = "terrain",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    nodata: Optional[float] = None,
    alpha: float = 1.0,
    categorical: bool = False,
) -> np.ndarray:
    """Return (H, W, 4) uint8 RGBA array using the given matplotlib colormap.

    If ``categorical`` is true and the data values look like ESRI D8 codes
    (powers of two ≤ 128), a fixed code-to-colour LUT is used so each of the
    eight directions remains visually distinct.
    """
    if categorical:
        return _categorical_d8_rgba(data, alpha=alpha)

    arr = data.astype(np.float32)
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
        # Bump just enough that the normalisation denominator is
        # representable at the current magnitude — adding a literal 1.0 to
        # vmin=1e10 disappears at float32 precision and (vmax-vmin) becomes
        # 0, yielding all-NaN normalisation.
        vmax = vmin + max(1.0, abs(vmin) * 1e-6)

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
    categorical: bool = False,
) -> str:
    """Encode array as base64 PNG data-URL suitable for Leaflet imageOverlay."""
    data = _decimate(data, categorical=categorical)
    rgba = array_to_rgba(data, cmap, vmin, vmax, nodata, alpha=1.0,
                         categorical=categorical)
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
