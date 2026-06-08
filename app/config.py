"""
Global constants and default settings.
"""
from __future__ import annotations

APP_NAME = "DEM Analyst"
APP_VERSION = "1.0.0"

# Default illumination parameters (Horn 1981 / GDAL defaults)
DEFAULT_AZIMUTH = 315.0     # degrees, geographic (0=N, CW)
DEFAULT_ALTITUDE = 45.0     # degrees above horizon
DEFAULT_Z_FACTOR = 1.0      # vertical exaggeration

# Tile settings
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256
MAX_TILE_CACHE = 512        # tiles kept in memory

# Analysis defaults
DEFAULT_TPI_RADIUS = 5      # cells
DEFAULT_SVF_DIRECTIONS = 16
DEFAULT_SVF_RADIUS = 100    # cells
DEFAULT_ROUGHNESS_WINDOW = 3

# Color maps per product (matplotlib names)
COLORMAPS = {
    "hillshade":             "Greys_r",
    "slope":                 "YlOrRd",
    "aspect":                "hsv",
    "curvature":             "RdBu_r",
    "curvature_profile":     "RdBu_r",
    "curvature_plan":        "RdBu",
    "fill_sinks":            "terrain",
    "flow_direction":        "Set1",
    "flow_accumulation":     "Blues",
    "streams":               "Blues",
    "twi":                   "RdYlBu",
    "spi":                   "YlOrRd",
    "tpi":                   "RdYlGn",
    "tri":                   "YlOrBr",
    "roughness":             "YlOrBr",
    "viewshed":              "RdYlGn",
    "svf":                   "Greys_r",
    "multidirectional":      "Greys_r",
    "dem":                   "terrain",
}

# Units shown on the map legend's min/max labels, per product.  Only entries
# whose units are unambiguous are listed; everything else (indices, ratios,
# direction-configurable slope, …) renders without a unit suffix.
LEGEND_UNITS = {
    "dem":        "m",
    "fill_sinks": "m",
    "aspect":     "°",
    "twi":        "",
    "spi":        "",
}

# Default opacity for new overlay layers
DEFAULT_OPACITY = 0.75

# Maximum pixel dimension (either axis) of a DEM overlay PNG pushed to Leaflet.
# Larger rasters are NaN-aware block-averaged down to this size before encoding,
# trading display resolution for a smaller base64 payload.  4096 keeps a full
# 1°×1° SRTM tile (3601²) at native resolution; raise it for sharper overlays
# at the cost of a larger payload / slower style refreshes, lower it if the
# embedded browser feels sluggish with several layers stacked.
MAX_OVERLAY_DIM = 4096
