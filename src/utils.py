"""Shared utility functions for the Casablanca Tramway Simulation."""

import os
import sys
from pathlib import Path
import json
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, box

# ── Project paths ──
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# ── Casablanca constants ──
CASA_CENTER = (33.5731, -7.5898)  # lat, lon
CASA_BBOX = (-7.7, 33.48, -7.45, 33.65)  # (west, south, east, north)
CASA_PLACE = "Casablanca, Morocco"

# ── Aliases used across modules ──
CASABLANCA_CENTER = CASA_CENTER
CASABLANCA_BBOX = CASA_BBOX

# ── CRS constants ──
# Use proj4 strings instead of EPSG codes — proj4 works purely from in-memory
# math and does NOT require pyproj to find its proj.db database file.
# This is the only reliable fix for Windows + conda + Streamlit thread contexts.
DEFAULT_CRS    = "+proj=longlat +datum=WGS84 +no_defs"          # WGS-84
PROJECTED_CRS  = "+proj=utm +zone=29 +datum=WGS84 +units=m +no_defs"  # UTM 29N

# ── Simulation constants ──
TRAM_SPEED_KMH = 25      # average commercial tram speed
BUFFER_DISTANCE_M = 500  # catchment buffer in metres

# Road type importance weights
ROAD_IMPORTANCE = {
    "primary": 1.0,
    "secondary": 0.7,
    "tertiary": 0.4,
    "residential": 0.2,
    "unclassified": 0.15,
    "living_street": 0.1,
}

# Excluded road types (not suitable for tram)
EXCLUDED_ROADS = ["motorway", "motorway_link", "trunk", "trunk_link"]

# POI weights for demand scoring
POI_WEIGHTS = {
    "hotel": 5.0,
    "school": 3.0,
    "university": 3.5,
    "hospital": 4.0,
    "mall": 4.0,
    "marketplace": 3.0,
    "bus_station": 2.5,
    "attraction": 3.0,
    "supermarket": 2.0,
}

# Traffic color scale (TomTom-style)
TRAFFIC_COLORS = {
    "free_flow": "#00C853",      # Green
    "moderate": "#FFD600",       # Yellow
    "heavy": "#FF9100",          # Orange
    "congestion": "#FF1744",     # Red
    "severe": "#B71C1C",         # Dark red
}

def traffic_color(speed_ratio: float) -> str:
    """Return traffic color based on speed ratio (current/free-flow)."""
    if speed_ratio > 0.8:
        return TRAFFIC_COLORS["free_flow"]
    elif speed_ratio > 0.6:
        return TRAFFIC_COLORS["moderate"]
    elif speed_ratio > 0.4:
        return TRAFFIC_COLORS["heavy"]
    elif speed_ratio > 0.2:
        return TRAFFIC_COLORS["congestion"]
    else:
        return TRAFFIC_COLORS["severe"]


# ── Alias used by webapp and traffic_simulation ──
get_traffic_color = traffic_color


def get_traffic_color_rgba(speed_ratio: float) -> list:
    """Return [R, G, B, A] for PyDeck-style traffic colouring."""
    hex_color = get_traffic_color(speed_ratio).lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return [r, g, b, 200]


def to_projected(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject GeoDataFrame to UTM 29N (proj4) for metric operations."""
    if gdf is None or gdf.empty:
        return gdf
    if gdf.crs is None:
        gdf = gdf.set_crs(DEFAULT_CRS)
    return gdf.to_crs(PROJECTED_CRS)


def to_geographic(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject GeoDataFrame back to WGS-84 (proj4) geographic coordinates."""
    if gdf is None or gdf.empty:
        return gdf
    return gdf.to_crs(DEFAULT_CRS)


def ensure_dirs():
    """Create data directories if they don't exist."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def load_geojson(filename: str) -> gpd.GeoDataFrame:
    """Load a GeoJSON file from the processed data directory."""
    path = DATA_PROCESSED / filename
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}. Run 'python src/data_prep.py --sample' first.")
    return gpd.read_file(path)


def save_geojson(gdf: gpd.GeoDataFrame, filename: str):
    """Save a GeoDataFrame to the processed data directory."""
    ensure_dirs()
    path = DATA_PROCESSED / filename
    gdf.to_file(path, driver="GeoJSON")
    print(f"  ✓ Saved {filename} ({len(gdf)} features)")


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two lat/lon points."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def random_point_in_bbox(bbox=CASA_BBOX, rng=None):
    """Generate a random point within bounding box."""
    if rng is None:
        rng = np.random.default_rng()
    west, south, east, north = bbox
    lon = rng.uniform(west, east)
    lat = rng.uniform(south, north)
    return Point(lon, lat)



# Alias — traffic_simulation.py imports this name
def get_traffic_color(speed_ratio: float) -> str:
    return traffic_color(speed_ratio)

