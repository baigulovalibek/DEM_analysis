"""
Drape texture composition.

Takes the selected layers (in z-order, top-of-stack first) and composites them
into a single RGBA byte array sized to the DEM's grid.  The output is the
texture that gets uploaded as the terrain's albedo.

The composition mirrors what Leaflet does on the 2D map: each layer's RGBA
(produced by ``renderer.array_to_rgba``) is multiplied by the layer's opacity
and ``over``-composited with what's below.  Layers that don't share the DEM's
bounds are resampled via PIL bilinear (categorical layers nearest).
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
from PIL import Image

from app.core.dem_layer import DemLayer, GeoBounds
from app.core.renderer import array_to_rgba


# Texture cap.  Big drapes blow VRAM budgets (16k² × 4 B = 1 GB) without
# adding visible detail past the screen's pixel count.  4096² is a good ceiling
# for 4K monitors with one DEM on screen.
MAX_TEXTURE_SIDE = 4096


def _resample_rgba(
    rgba: np.ndarray,
    src_bounds: GeoBounds,
    dst_bounds: GeoBounds,
    dst_shape: Tuple[int, int],
    categorical: bool = False,
) -> np.ndarray:
    """Resample a layer's RGBA into the DEM's grid extent.

    If the source bounds equal the destination bounds and the shapes already
    match, the input is returned untouched.  Otherwise PIL handles the resize
    + crop; we paint the source onto a transparent canvas spanning the
    destination bounds, so areas outside the source contribute zero alpha.
    """
    dst_rows, dst_cols = dst_shape

    bounds_match = (
        abs(src_bounds.west  - dst_bounds.west)  < 1e-12 and
        abs(src_bounds.east  - dst_bounds.east)  < 1e-12 and
        abs(src_bounds.south - dst_bounds.south) < 1e-12 and
        abs(src_bounds.north - dst_bounds.north) < 1e-12
    )
    if bounds_match and rgba.shape[:2] == dst_shape:
        return rgba

    resampling = Image.Resampling.NEAREST if categorical else Image.Resampling.BILINEAR

    if bounds_match:
        img = Image.fromarray(rgba, "RGBA").resize(
            (dst_cols, dst_rows), resampling
        )
        return np.asarray(img)

    # Bounds differ — map source extent onto destination grid.  Compute the
    # pixel coordinates (in the destination grid) where the source's
    # west/east/north/south fall, then paste a resized source patch there.
    dst_w = dst_bounds.east  - dst_bounds.west
    dst_h = dst_bounds.north - dst_bounds.south

    if dst_w <= 0 or dst_h <= 0:
        return np.zeros((dst_rows, dst_cols, 4), dtype=np.uint8)

    # Source rectangle in destination pixel space.
    px_x0 = (src_bounds.west - dst_bounds.west) / dst_w * dst_cols
    px_x1 = (src_bounds.east - dst_bounds.west) / dst_w * dst_cols
    # row 0 is north in the destination grid
    px_y0 = (dst_bounds.north - src_bounds.north) / dst_h * dst_rows
    px_y1 = (dst_bounds.north - src_bounds.south) / dst_h * dst_rows

    target_w = max(int(round(px_x1 - px_x0)), 1)
    target_h = max(int(round(px_y1 - px_y0)), 1)

    src_img = Image.fromarray(rgba, "RGBA").resize((target_w, target_h), resampling)
    canvas = Image.new("RGBA", (dst_cols, dst_rows), (0, 0, 0, 0))
    canvas.paste(src_img, (int(round(px_x0)), int(round(px_y0))))
    return np.asarray(canvas)


def _layer_to_rgba(layer: DemLayer, categorical_hint: bool) -> np.ndarray:
    """Produce an (H, W, 4) uint8 RGBA for a layer, matching the 2D render.

    Uses the layer's own colormap and render range so the drape looks like
    what the user sees on the 2D map.
    """
    lo, hi = layer.render_min, layer.render_max
    if lo is None or hi is None:
        lo, hi = layer.auto_range()
    return array_to_rgba(
        layer.data,
        cmap=layer.colormap,
        vmin=lo, vmax=hi,
        nodata=layer.nodata,
        alpha=1.0,
        categorical=categorical_hint,
    )


def compose_drape(
    layers_top_first: Iterable[DemLayer],
    dst_bounds: GeoBounds,
    dst_shape: Tuple[int, int],
    max_side: int = MAX_TEXTURE_SIDE,
) -> np.ndarray:
    """Composite ``layers_top_first`` into a single RGBA texture.

    Parameters
    ----------
    layers_top_first : iterable of layers in stacking order, top first
        (i.e. the order returned by ``LayerManager.all()``).
    dst_bounds : geographic extent the texture covers (the active DEM's bounds).
    dst_shape : ``(rows, cols)`` of the active DEM's grid.
    max_side : cap on either dimension of the output texture.  Larger DEMs are
        composited at full resolution and the final result is downscaled once
        with bilinear filtering — this is much faster than resampling every
        contributing layer down individually.

    Returns
    -------
    (H, W, 4) uint8 RGBA array.  Areas not covered by any visible draped
    layer are fully transparent.
    """
    # The caller already filtered by "what to drape" (the 3D side-panel
    # checklist).  Do NOT also filter by ``l.visible`` here — 2D-map visibility
    # is an independent concern; gating on it silently drops layers the user
    # explicitly checked for the 3D drape (e.g. TPI hidden on the 2D map but
    # ticked in the drape list).
    layers = [l for l in layers_top_first if l.data is not None]
    dst_rows, dst_cols = dst_shape

    if not layers or dst_rows == 0 or dst_cols == 0:
        return np.zeros((max(dst_rows, 1), max(dst_cols, 1), 4), dtype=np.uint8)

    # Accumulator: premultiplied float RGBA.  Bottom-up composition with the
    # "over" operator → iterate from bottom layer to top so each step writes
    # over what's already there.
    acc_rgb = np.zeros((dst_rows, dst_cols, 3), dtype=np.float32)
    acc_a   = np.zeros((dst_rows, dst_cols, 1), dtype=np.float32)

    for layer in reversed(list(layers)):
        if layer.bounds is None:
            continue
        categorical = (
            layer.product == "flow_direction"
            and np.issubdtype(layer.data.dtype, np.integer)
        )
        rgba = _layer_to_rgba(layer, categorical_hint=categorical)
        rgba = _resample_rgba(rgba, layer.bounds, dst_bounds, dst_shape,
                              categorical=categorical)
        lyr_rgb = rgba[..., :3].astype(np.float32) / 255.0
        lyr_a   = (rgba[..., 3:4].astype(np.float32) / 255.0) * float(layer.opacity)

        # Standard non-premultiplied "over" operator.
        inv_a = 1.0 - lyr_a
        acc_rgb = lyr_rgb * lyr_a + acc_rgb * inv_a
        acc_a   = lyr_a + acc_a * inv_a

    out = np.empty((dst_rows, dst_cols, 4), dtype=np.uint8)
    out[..., :3] = np.clip(acc_rgb * 255.0, 0, 255).astype(np.uint8)
    out[...,  3] = np.clip(acc_a[..., 0] * 255.0, 0, 255).astype(np.uint8)

    # Final downscale if the DEM is larger than our texture cap.
    if dst_rows > max_side or dst_cols > max_side:
        scale = max_side / max(dst_rows, dst_cols)
        new_w = max(int(dst_cols * scale), 1)
        new_h = max(int(dst_rows * scale), 1)
        out = np.asarray(
            Image.fromarray(out, "RGBA").resize((new_w, new_h), Image.Resampling.BILINEAR)
        )
    return out
