# DEM Analyst — Project Map

A desktop GIS-style application for analysing Digital Elevation Models (DEMs).
QGIS-inspired layout: a Leaflet map canvas in the centre, dockable panels for
layers, properties, and analyses around it, plus an optional 3D OpenGL view.
All terrain algorithms are implemented from scratch on NumPy/SciPy with
references to the original papers (Horn 1981, Zevenbergen & Thorne 1987,
Barnes 2014, Tarboton 1997, Beven & Kirkby 1979, Zakšek 2011, …).

## Stack

- **Language:** Python 3 (PEP 563 `from __future__ import annotations` throughout)
- **GUI:** PyQt6 + PyQt6-WebEngine (Qt WebEngine hosts Leaflet)
- **Map / web layer:** Leaflet.js (vendored locally under `resources/leaflet/`) inside `QWebEngineView`, wired to Python via `QWebChannel`
- **Numerics:** NumPy, SciPy (`ndimage` filters), Pillow (PNG encode)
- **Raster I/O:** rasterio (GeoTIFF, IMG, ADF, NetCDF, HGT, …) + CRS reprojection of bounds to WGS84
- **2D charts:** matplotlib (`Agg` backend, embedded `FigureCanvasQTAgg`)
- **3D view:** pyqtgraph OpenGL (`GLViewWidget` + `GLMeshItem`) when available; transparently falls back to a CPU matplotlib `mpl_toolkits.mplot3d` surface (embedded `FigureCanvasQTAgg`) when OpenGL is missing — common on lab PCs without GPU drivers. Set `DEM_ANALYST_FORCE_MPL=1` to force the matplotlib backend.
- **Threading:** custom `QThread` worker — analyses run off the GUI thread with progress / cancel support
- **Styling:** Qt stylesheet `app/ui/style.qss` applied app-wide; dark theme also baked into `map.html`

## Entry point

`main.py` (project root) — the only entry point:

1. Switches CWD to the project directory so relative paths resolve.
2. Sets `QTWEBENGINE_CHROMIUM_FLAGS=--no-sandbox` and `AA_ShareOpenGLContexts` **before** `QApplication`.
3. Constructs `QApplication`, then imports `MainWindow` (the deferred import matters — Qt attributes must be set first).
4. Optionally opens a DEM passed as `python main.py path/to/dem.tif`.

## Top-level layout

```
DEM_analysis/
├── main.py                 # entry point
├── requirements.txt        # PyQt6, rasterio, numpy, scipy, matplotlib, pyqtgraph, Pillow, requests
├── app/
│   ├── config.py           # constants: colormaps, defaults, tile URL, parallel threshold
│   ├── main_window.py      # MainWindow — orchestrator; wires panels↔manager↔worker↔map
│   ├── core/               # data model + analysis algorithms (no Qt UI here, except Qt signals on the manager)
│   └── ui/                 # Qt widgets and the Leaflet bridge
└── resources/
    ├── map.html            # Leaflet page + JS bridge (loaded into QWebEngineView)
    └── leaflet/            # vendored leaflet.js + leaflet.css + marker images (offline-capable)
```

## Module map

### `app/` — orchestration layer

| File | Responsibility |
|---|---|
| `config.py` | App name/version, default azimuth/altitude/Z, tile URL/cache size, default analysis parameters, **`COLORMAPS`** map (`product → matplotlib cmap`), `DEFAULT_OPACITY`, `PARALLEL_THRESHOLD`. |
| `main_window.py` | `MainWindow(QMainWindow)`. Builds the UI shell, wires every signal, opens/exports rasters via rasterio, **dispatches analyses** (`_resolve_analysis` maps a product key → callable + kwargs), runs them on an `AnalysisWorker`, ingests the result into a new `DemLayer`, and keeps **per-DEM intermediate caches** (`_flow_dir_cache`, `_flow_angle_cache`, `_flow_accum_cache`, `_slope_rad_cache`, `_filled_cache`) keyed by `DemLayer.uid` so dependent analyses (Streams needs Flow Accumulation, TWI needs both Slope+FA, …) can be chained without recomputing. Also owns the lat/lon → row/col coordinate utility and the 3D update path. |

### `app/core/` — data model + raster engine (Qt-free, except `LayerManager` which is a `QObject`)

| File | Responsibility |
|---|---|
| `dem_layer.py` | `GeoBounds` (WGS84 bbox) + `DemLayer` dataclass — the universal raster object. Holds `data`, `nodata`, `bounds`, `crs_wkt`, `cell_size_m`, `colormap`, `opacity`, `render_min/max`, a stable `uid` (survives renames; used as cache key), and `product` tag (`"dem"`, `"hillshade"`, …). Computes `valid_data` (NaN-masked), `stats`, and `auto_range` (2–98 pct stretch). |
| `layer_manager.py` | `LayerManager(QObject)` — single source of truth for the layer stack. CRUD + uniqueness, selection, active-DEM tracking, opacity/visibility/cmap/range mutators, reordering. Emits `layers_changed`, `layer_updated`, `layer_selected`, `active_dem_changed`, `layer_renamed`. |
| `renderer.py` | NumPy → PNG/QImage. `array_to_rgba` (matplotlib cmap + nodata transparency + percentile auto-range), `array_to_png_b64` (decimated → base64 data-URL for Leaflet `imageOverlay`, capped at 2048 px), `hillshade_rgba` (Greys blended with a colourised overlay). Forces matplotlib `Agg` backend. |
| `worker.py` | `AnalysisWorker(QThread)`. Calls `func(**kwargs)` off-thread. Inspects the callable's signature — if it accepts `_progress`, injects a callback that emits `progress(int)` and returns the cancel flag. Emits `result(object)` / `error(str)`. Carries request context (`product`, `dem`, `params`) so the result slot can route correctly. |

#### Analysis sub-packages under `app/core/`

All modules are pure NumPy/SciPy. Long-running ones accept a `_progress(pct)` callback that doubles as a cancel probe.

**`derivatives/`** — primary first/second-order derivatives
- `_gradient.py` — shared Horn-1981 Sobel-weighted 3×3 finite-difference gradient (foundation for hillshade/slope/aspect).
- `hillshade.py` — Lambertian shaded relief (Horn 1981; matches gdaldem/QGIS).
- `slope.py` — degrees / percent / radians.
- `aspect.py` — 0–360° clockwise from North; flat = −1.
- `curvature.py` — Zevenbergen & Thorne (1987) profile / plan / mean curvature.

**`hydrology/`** — flow modelling pipeline
- `fill_sinks.py` — Priority-Flood (Barnes 2014, O(n log n) via min-heap) and Breach+Fill (Lindsay 2016).
- `flow_direction.py` — D8 (O'Callaghan & Mark 1984, ESRI/TauDEM codes) and D-infinity (Tarboton 1997, continuous angle).
- `flow_accumulation.py` — topological-traversal accumulation for both D8 and D∞ (with `specific_catchment_area` helper, used by TWI/SPI).
- `streams.py` — channel extraction by contributing-area threshold (auto = 99th-pct or user value) and Strahler ordering.

**`indices/`** — terrain indices
- `twi.py` — Topographic Wetness Index `ln(a/tan β)` (Beven & Kirkby 1979).
- `spi.py` — Stream Power Index `a · tan β` (Moore 1991), optional log scale.
- `tpi.py` — Topographic Position Index with annular window (Weiss 2001).
- `tri.py` — Terrain Ruggedness Index (Riley 1999).
- `roughness.py` — Range, Std-Dev, VRM (Sappington 2007), Surface-area ratio (Jenness 2004).

**`visibility/`** — illumination, viewshed, sky-view, line-of-sight
- `multidirectional.py` — Mark (1992) 4-azimuth weighted hillshade (gdaldem `-multidirectional`).
- `viewshed.py` — R3 line-of-sight (Wang 2000), with optional Earth-curvature + atmospheric refraction.
- `svf.py` — Sky-View Factor (Zakšek 2011) + positive/negative openness (Yokoyama 2002).
- `profile.py` — `ElevationProfile` dataclass + `sample_profile` bilinear sampling along a polyline (used by the profile dialog).

### `app/ui/` — Qt widgets

| File | Responsibility |
|---|---|
| `map_canvas.py` | `MapCanvas(QWidget)` wraps `QWebEngineView` loading `resources/map.html`. **`MapBridge(QObject)`** is registered on a `QWebChannel` as `"bridge"` and exposes `pyqtSlot`s called from JS (`onMouseMove`, `onMapClick`, `onMapMoved`, `onProfilePoint`, `onProfileComplete`, `onViewshedPoint`) that re-emit as Qt signals. Python → JS goes through `run_js()` which buffers calls until `loadFinished` (so a CLI-supplied DEM doesn't race the page load). High-level API: `add_layer / remove_layer / rename_layer / set_opacity / set_visible / set_z_index / fit_bounds / set_base_layer / start_profile_draw / start_viewshed_draw`. |
| `layer_panel.py` | `LayerPanel(QDockWidget)` — tree of all layers with checkboxes for visibility, context menu for rename/remove/zoom/export, opacity slider. Emits `visibility_toggled`, `remove_requested`, `zoom_requested`, `export_requested`, `active_dem_change`. |
| `analysis_panel.py` | `AnalysisPanel(QDockWidget)` — categorized tree (Primary Derivatives / Hydrological / Indices / Visibility) over a `QStackedWidget` of per-product `_ParamForm` subclasses. Emits **`run_analysis(product_key, params)`**, **`start_tool(tool)`** (for profile/viewshed map drawing), `cancel_analysis`. Shows the progress bar fed by the worker. |
| `properties_panel.py` | `PropertiesPanel(QDockWidget)` — info, stats, histogram (`FigureCanvasQTAgg`), colormap picker, render-range stretch, opacity slider for the **selected** layer. Emits `style_changed(name)` which `MainWindow` routes to `MapCanvas.refresh_layer`. |
| `view3d.py` | `View3DDock(QDockWidget)` — terrain mesh with two interchangeable backends. **GL backend** (`"gl"`): pyqtgraph `GLViewWidget` + `GLMeshItem`, optional hillshade texture drape, downsampling to ≤256 cells per side. **Matplotlib backend** (`"mpl"`): `mpl_toolkits.mplot3d` `plot_surface` embedded in `FigureCanvasQTAgg`, downsampled to ≤100 cells per side (CPU-bound), mouse drag rotates / scroll zooms. Backend is auto-selected; `DEM_ANALYST_FORCE_MPL=1` forces matplotlib. Falls back to a placeholder label only if both backends fail. |
| `profile_dialog.py` | `ProfileDialog(QDialog)` — renders an `ElevationProfile`: a matplotlib line/fill chart plus a stats row (length, min/max elev, ascent, descent). |
| `style.qss` | Application-wide Qt stylesheet (dark theme, dock titles, button states). Loaded by `MainWindow._apply_stylesheet` (note: the code looks for `style.qss` next to `main_window.py`; the file actually lives in `app/ui/`). |

### `resources/`

- `map.html` — Leaflet page rendered inside `QWebEngineView`. Sets up 3 base layers (OSM, Esri Satellite, OpenTopoMap, plus "none"), tracks tile errors to show an **offline banner** after 3 failures, exposes the JS API used from Python (`addDEMOverlay`, `removeDEMOverlay`, `renameDEMOverlay`, `setLayerOpacity / Visible / ZIndex`, `fitBounds`, `setBaseLayer`, `start/stopProfileDraw`, `start/stopViewshedDraw`), and calls back into Python via `qt.webChannelTransport` → `bridge`.
- `leaflet/` — vendored Leaflet 1.x JS + CSS + default marker images so the app's basemap controls still work when there is no internet (DEM overlays keep rendering even when no tiles are available).

## How it all connects

```
main.py
   │ builds QApplication, then
   ▼
MainWindow ─── owns ──► LayerManager (QObject)
   │  │
   │  ├─► LayerPanel ◄── shows ── LayerManager.layers
   │  ├─► AnalysisPanel  (emits run_analysis → MainWindow._dispatch_analysis)
   │  ├─► PropertiesPanel (reads selected layer; emits style_changed)
   │  ├─► View3DDock     (rebuilt from active DEM + its hillshade)
   │  └─► MapCanvas ──── QWebEngineView ── resources/map.html (Leaflet)
   │                         ▲   │
   │                   run_js │   │ QWebChannel
   │                         │   ▼
   │                       MapBridge ── pyqtSignals (mouse_moved, profile_point, viewshed_point, …)
   │
   ├─► AnalysisWorker (QThread, on demand)
   │       │  func picked by MainWindow._resolve_analysis
   │       │     ├─ app/core/derivatives/*
   │       │     ├─ app/core/hydrology/*
   │       │     ├─ app/core/indices/*
   │       │     └─ app/core/visibility/*
   │       └─ emits progress / result / error → MainWindow slots
   │
   └─► rasterio (open DEM, export GeoTIFF) ── bounds reprojected to WGS84 for Leaflet
```

**Typical flow for an analysis:**

1. User clicks a node in `AnalysisPanel`, adjusts a `_ParamForm`, clicks **Compute**.
2. `AnalysisPanel.run_analysis(product, params)` → `MainWindow._dispatch_analysis`.
3. `MainWindow._resolve_analysis` returns `(callable, kwargs)`, optionally pulling cached intermediates (e.g. flow direction) from per-DEM dicts keyed by `DemLayer.uid`.
4. An `AnalysisWorker(QThread)` runs the callable. If the function accepts `_progress`, the worker injects a progress/cancel callback.
5. On success, `MainWindow._on_worker_result` (a) caches intermediates for downstream analyses, (b) wraps the array in a new `DemLayer`, (c) hands it to `LayerManager.add` and `MapCanvas.add_layer`, which `renderer.array_to_png_b64`-encodes it and adds an `L.imageOverlay` over the DEM's geographic bounds.

**Map interaction flow:** Leaflet event → JS calls `bridge.onXxx(...)` → `MapBridge` pyqtSignal → `MainWindow` slot (status bar update, profile vertex collection, viewshed observer placement, …).
