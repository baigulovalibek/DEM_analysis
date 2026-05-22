"""
Layer panel dock — tree widget showing all loaded layers.
"""
from __future__ import annotations
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QTreeWidget, QTreeWidgetItem, QToolBar, QSlider,
    QLabel, QMenu, QSizePolicy, QAbstractItemView, QInputDialog,
)

from app.core.layer_manager import LayerManager
from app.core.dem_layer import DemLayer


class LayerItem(QTreeWidgetItem):
    def __init__(self, layer: DemLayer):
        super().__init__()
        self.layer_name = layer.name
        self._refresh(layer)

    def _refresh(self, layer: DemLayer):
        self.setText(0, layer.name)
        self.setCheckState(0, Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked)
        self.setToolTip(0, f"{layer.product}  ·  {layer.shape[0]}×{layer.shape[1]}")


class LayerPanel(QDockWidget):
    """Dockable layer tree panel."""

    layer_selected    = pyqtSignal(str)
    visibility_toggled = pyqtSignal(str, bool)
    remove_requested  = pyqtSignal(str)
    zoom_requested    = pyqtSignal(str)
    export_requested  = pyqtSignal(str)
    active_dem_change = pyqtSignal(str)

    def __init__(self, manager: LayerManager, parent=None):
        super().__init__("Layers", parent)
        self._mgr = manager
        self._items: dict[str, LayerItem] = {}
        self._building = False

        self.setMinimumWidth(200)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        self._build_ui()
        self._connect_manager()

    def _build_ui(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Toolbar
        tb = QToolBar()
        tb.setIconSize(tb.iconSize().__class__(16, 16))

        self._act_up   = QAction("▲", self, toolTip="Move layer up")
        self._act_down = QAction("▼", self, toolTip="Move layer down")
        self._act_del  = QAction("✕", self, toolTip="Remove layer")
        for act in (self._act_up, self._act_down, self._act_del):
            tb.addAction(act)

        self._act_up.triggered.connect(self._move_up)
        self._act_down.triggered.connect(self._move_down)
        self._act_del.triggered.connect(self._remove_selected)
        layout.addWidget(tb)

        # Tree
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._context_menu)
        self._tree.itemClicked.connect(self._on_item_clicked)
        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self._tree)

        # Opacity slider
        op_row = QWidget()
        op_layout = QHBoxLayout(op_row)
        op_layout.setContentsMargins(0, 0, 0, 0)
        op_layout.addWidget(QLabel("Opacity:"))
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(75)
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        op_layout.addWidget(self._opacity_slider)
        self._opacity_label = QLabel("75%")
        self._opacity_label.setFixedWidth(36)
        op_layout.addWidget(self._opacity_label)
        layout.addWidget(op_row)

        self.setWidget(container)

    def _connect_manager(self):
        self._mgr.layers_changed.connect(self._rebuild)
        self._mgr.layer_updated.connect(self._update_item)
        self._mgr.layer_selected.connect(self._highlight)

    # ── Internal rebuild ───────────────────────────────────────────────────

    def _rebuild(self):
        self._building = True
        self._tree.clear()
        self._items.clear()
        for layer in self._mgr.all():
            item = LayerItem(layer)
            self._tree.addTopLevelItem(item)
            self._items[layer.name] = item
        self._building = False

    def _update_item(self, name: str):
        item = self._items.get(name)
        if item:
            layer = self._mgr.get(name)
            if layer:
                self._building = True
                item._refresh(layer)
                self._building = False
        sel = self._mgr.selected
        if sel is not None and sel.name == name:
            self._sync_opacity_slider(sel)

    def _highlight(self, name: str):
        item = self._items.get(name)
        if item:
            self._tree.setCurrentItem(item)
            layer = self._mgr.get(name)
            if layer:
                self._sync_opacity_slider(layer)

    def _sync_opacity_slider(self, layer: DemLayer):
        pct = int(round(layer.opacity * 100))
        self._opacity_slider.blockSignals(True)
        self._opacity_slider.setValue(pct)
        self._opacity_slider.blockSignals(False)
        self._opacity_label.setText(f"{pct}%")

    # ── Slot handlers ──────────────────────────────────────────────────────

    def _on_item_clicked(self, item: QTreeWidgetItem, _col: int):
        if isinstance(item, LayerItem):
            self._mgr.select(item.layer_name)
            self.layer_selected.emit(item.layer_name)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, _col: int):
        if isinstance(item, LayerItem):
            self._rename_layer(item.layer_name)

    def _rename_layer(self, name: str):
        new, ok = QInputDialog.getText(self, "Rename Layer", "New name:", text=name)
        if ok and new.strip():
            self._mgr.rename(name, new.strip())

    def _on_item_changed(self, item: QTreeWidgetItem, col: int):
        if self._building or not isinstance(item, LayerItem):
            return
        visible = item.checkState(0) == Qt.CheckState.Checked
        self._mgr.set_visible(item.layer_name, visible)
        self.visibility_toggled.emit(item.layer_name, visible)

    def _on_opacity_changed(self, value: int):
        self._opacity_label.setText(f"{value}%")
        layer = self._mgr.selected
        if layer:
            self._mgr.set_opacity(layer.name, value / 100.0)

    def _move_up(self):
        layer = self._mgr.selected
        if layer:
            self._mgr.move_up(layer.name)

    def _move_down(self):
        layer = self._mgr.selected
        if layer:
            self._mgr.move_down(layer.name)

    def _remove_selected(self):
        layer = self._mgr.selected
        if layer:
            self.remove_requested.emit(layer.name)

    def _context_menu(self, pos):
        item = self._tree.itemAt(pos)
        if not isinstance(item, LayerItem):
            return
        menu = QMenu(self)
        menu.addAction("Zoom to Layer").triggered.connect(
            lambda: self.zoom_requested.emit(item.layer_name)
        )
        menu.addAction("Rename Layer…").triggered.connect(
            lambda: self._rename_layer(item.layer_name)
        )
        menu.addAction("Export as GeoTIFF…").triggered.connect(
            lambda: self.export_requested.emit(item.layer_name)
        )
        menu.addSeparator()

        act_dem = menu.addAction("Set as Active DEM")
        layer = self._mgr.get(item.layer_name)
        act_dem.setEnabled(layer is not None and layer.product == "dem")
        act_dem.triggered.connect(lambda: self.active_dem_change.emit(item.layer_name))

        menu.addSeparator()
        menu.addAction("Remove").triggered.connect(
            lambda: self.remove_requested.emit(item.layer_name)
        )
        menu.exec(self._tree.mapToGlobal(pos))
