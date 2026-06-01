"""
Background worker for drape texture composition.

The drape is rebuilt whenever the layer selection, layer styling, or active
DEM changes.  Compositing a stack of 4k² rasters via ``compose_drape`` takes
100–300 ms — too long to block the main thread on every slider drag.

The worker takes a snapshot of the inputs (so the main thread can keep
editing while it runs) and emits the resulting RGBA array.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from app.core.dem_layer import DemLayer, GeoBounds
from app.core.scene3d.drape import compose_drape


@dataclass
class DrapeRequest:
    """Snapshot of the inputs needed to recomposite the drape.

    Lives on its own so the worker thread has no references to anything the
    main thread might mutate underneath it.  The layer arrays are passed by
    reference (NumPy arrays are immutable in practice once they've been added
    to the layer manager), but layer style fields are copied.
    """
    layers: List[DemLayer]
    dst_bounds: GeoBounds
    dst_shape: Tuple[int, int]


class DrapeWorker(QThread):
    """Runs ``compose_drape`` on a snapshot of the inputs."""

    finished_rgba = pyqtSignal(object)        # np.ndarray
    failed        = pyqtSignal(str)

    def __init__(self, request: DrapeRequest, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._req = request
        # If a new request comes in while this one is running, the requester
        # can set this flag and we'll discard the result instead of emitting.
        self._discarded = False

    def discard(self):
        self._discarded = True

    def run(self):
        try:
            rgba = compose_drape(
                layers_top_first=self._req.layers,
                dst_bounds=self._req.dst_bounds,
                dst_shape=self._req.dst_shape,
            )
        except Exception as exc:
            if not self._discarded:
                self.failed.emit(str(exc))
            return
        if not self._discarded:
            self.finished_rgba.emit(rgba)
