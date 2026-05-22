"""
Layer registry.  Single source of truth for all layers; emits Qt signals on changes.
"""
from __future__ import annotations
from typing import List, Optional

from PyQt6.QtCore import QObject, pyqtSignal

from app.core.dem_layer import DemLayer


class LayerManager(QObject):
    layers_changed = pyqtSignal()           # any structural change
    layer_updated = pyqtSignal(str)         # opacity/visibility/style change
    layer_selected = pyqtSignal(str)        # user clicked a layer
    active_dem_changed = pyqtSignal(str)    # which DEM drives analyses

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layers: List[DemLayer] = []
        self._active_dem: Optional[str] = None
        self._selected: Optional[str] = None

    # ── CRUD ───────────────────────────────────────────────────────────────

    def add(self, layer: DemLayer) -> None:
        # Ensure unique name
        base = layer.name
        existing = {l.name for l in self._layers}
        i = 1
        while layer.name in existing:
            layer.name = f"{base} ({i})"
            i += 1
        self._layers.insert(0, layer)
        if layer.product == "dem" and self._active_dem is None:
            self._active_dem = layer.name
            self.active_dem_changed.emit(layer.name)
        self.layers_changed.emit()

    def remove(self, name: str) -> None:
        self._layers = [l for l in self._layers if l.name != name]
        if self._active_dem == name:
            dems = [l for l in self._layers if l.product == "dem"]
            self._active_dem = dems[0].name if dems else None
            if self._active_dem:
                self.active_dem_changed.emit(self._active_dem)
        self.layers_changed.emit()

    def get(self, name: str) -> Optional[DemLayer]:
        for l in self._layers:
            if l.name == name:
                return l
        return None

    def all(self) -> List[DemLayer]:
        return list(self._layers)

    # ── Active DEM ─────────────────────────────────────────────────────────

    @property
    def active_dem(self) -> Optional[DemLayer]:
        return self.get(self._active_dem) if self._active_dem else None

    def set_active_dem(self, name: str) -> None:
        if self.get(name) and self._active_dem != name:
            self._active_dem = name
            self.active_dem_changed.emit(name)

    # ── Selection ──────────────────────────────────────────────────────────

    def select(self, name: str) -> None:
        self._selected = name
        self.layer_selected.emit(name)

    @property
    def selected(self) -> Optional[DemLayer]:
        return self.get(self._selected) if self._selected else None

    # ── Mutations ──────────────────────────────────────────────────────────

    def set_visible(self, name: str, visible: bool) -> None:
        layer = self.get(name)
        if layer and layer.visible != visible:
            layer.visible = visible
            self.layer_updated.emit(name)

    def set_opacity(self, name: str, opacity: float) -> None:
        layer = self.get(name)
        if layer:
            layer.opacity = max(0.0, min(1.0, opacity))
            self.layer_updated.emit(name)

    def set_colormap(self, name: str, cmap: str) -> None:
        layer = self.get(name)
        if layer:
            layer.colormap = cmap
            self.layer_updated.emit(name)

    def set_render_range(self, name: str, lo: float, hi: float) -> None:
        layer = self.get(name)
        if layer:
            layer.render_min = lo
            layer.render_max = hi
            self.layer_updated.emit(name)

    def move_up(self, name: str) -> None:
        idx = next((i for i, l in enumerate(self._layers) if l.name == name), -1)
        if idx > 0:
            self._layers[idx], self._layers[idx - 1] = (
                self._layers[idx - 1], self._layers[idx]
            )
            self.layers_changed.emit()

    def move_down(self, name: str) -> None:
        idx = next((i for i, l in enumerate(self._layers) if l.name == name), -1)
        if 0 <= idx < len(self._layers) - 1:
            self._layers[idx], self._layers[idx + 1] = (
                self._layers[idx + 1], self._layers[idx]
            )
            self.layers_changed.emit()
