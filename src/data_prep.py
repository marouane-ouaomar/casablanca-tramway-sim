"""
Data Collection & Preparation — Phase 1
ETL helpers: fetch roads, tram lines, POIs, population grids.
Supports both real OSM data (requires network) and sample data generation.
"""

import argparse
import sys
import os

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString, box, Polygon
from shapely.ops import unary_union

try:
    from src.utils import (
        CASA_CENTER, CASA_BBOX, CASA_PLACE,
        EXCLUDED_ROADS, ROAD_IMPORTANCE, POI_WEIGHTS,
        ensure_dirs, save_geojson, DATA_PROCESSED, DATA_RAW,
    )
except ImportError:
    from utils import (
        CASA_CENTER, CASA_BBOX, CASA_PLACE,
        EXCLUDED_ROADS, ROAD_IMPORTANCE, POI_WEIGHTS,
        ensure_dirs, save_geojson, DATA_PROCESSED, DATA_RAW,
    )


# ═══════════════════════════════════════════════════════════
# REAL DATA FETCHING (requires network + OSMnx)
# ═══════════════════════════════════════════════════════════

def fetch_roads_osm():
    """Fetch drivable roads from OpenStreetMap for Casablanca."""
    import osmnx as ox
    print("Fetching roads from OpenStreetMap...")
    G = ox.graph_from_place(CASA_PLACE, network_type="drive")
    nodes, edges = ox.graph_to_gdfs(G)

    # Filter out highways and restricted roads
    if "highway" in edges.columns:
        mask = edges["highway"].apply(
            lambda x: not any(h in EXCLUDED_ROADS for h in (x if isinstance(x, list) else [x]))
        )
        edges = edges[mask]

    save_geojson(edges.reset_index(), "roads.geojson")
    return G, edges


def fetch_tram_lines_osm():
    """Extract existing tram lines from OSM."""
    import osmnx as ox
    print("Fetching tram lines from OpenStreetMap...")
    tram = ox.features_from_place(CASA_PLACE, tags={"railway": "tram"})
    save_geojson(tram.reset_index(), "tram_lines.geojson")
    return tram


def fetch_pois_osm():
    """Extract POIs (hotels, malls, schools, hospitals, etc.) from OSM."""
    import osmnx as ox
    print("Fetching POIs from OpenStreetMap...")
    tags = {
        "amenity": ["school", "hospital", "marketplace", "university", "bus_station", "clinic"],
        "tourism": ["hotel", "attraction", "museum"],
        "shop": ["mall", "supermarket"],
    }
    pois = ox.features_from_place(CASA_PLACE, tags=tags)

    # Normalize POI types
    def classify_poi(row):
        if pd.notna(row.get("tourism")):
            return row["tourism"]
        if pd.notna(row.get("shop")):
            return row["shop"]
        if pd.notna(row.get("amenity")):
            return row["amenity"]
        return "other"

    pois["poi_type"] = pois.apply(classify_poi, axis=1)
    # Convert polygons to centroids for point-based analysis
    pois["geometry"] = pois.geometry.centroid
    save_geojson(pois[["geometry", "poi_type", "name"]].reset_index(), "pois.geojson")
    return pois


# ═══════════════════════════════════════════════════════════
# SAMPLE DATA GENERATION (no network needed)
# ═══════════════════════════════════════════════════════════

def generate_sample_roads(rng):
    """
    Generate a fully-connected road network for Casablanca.

    Key design: every intersection uses EXACT shared coordinates so that
    NetworkX sees edges that share endpoints and can route across the whole
    city.  The old jitter approach created thousands of isolated 2-node
    components — this replaces it with a proper topology.
    """
    print("Generating sample road network...")
    west, south, east, north = CASA_BBOX

    SPEED = {"primary": 50, "secondary": 40, "tertiary": 30,
             "residential": 20, "unclassified": 20, "living_street": 15}

    # Named boulevards / avenues for flavour
    H_NAMES = [
        "Boulevard Mohammed V", "Boulevard Zerktouni", "Avenue Hassan II",
        "Avenue des FAR", "Boulevard Bir Anzarane", "Boulevard Ghandi",
        "Avenue Al Massira", "Avenue Mers Sultan",
    ]
    V_NAMES = [
        "Rue de Fès", "Rue Allal Ben Abdellah", "Rue Ibnou Majid",
        "Avenue Lalla Yacout", "Rue Tahar Sebti", "Rue Abou Bakr Saddiq",
        "Boulevard Rachidi", "Avenue de la Résistance",
    ]

    # ── Build a precise grid of intersection nodes ──────────────────────────
    # 18 columns × 14 rows → 17×13 = 221 horizontal + 221 vertical segments
    NCOLS = 18
    NROWS = 14
    lons = np.linspace(west + 0.02, east - 0.02, NCOLS)
    lats = np.linspace(south + 0.015, north - 0.015, NROWS)

    # Assign road types to each row/column (consistent per street)
    h_types = rng.choice(
        ["primary", "secondary", "tertiary", "residential", "unclassified"],
        size=NROWS,
        p=[0.10, 0.15, 0.30, 0.30, 0.15],
    )
    v_types = rng.choice(
        ["primary", "secondary", "tertiary", "residential", "unclassified"],
        size=NCOLS,
        p=[0.10, 0.15, 0.30, 0.30, 0.15],
    )

    roads = []
    seg_id = 0

    # Horizontal streets — row i connects (lons[j], lats[i]) → (lons[j+1], lats[i])
    for i, lat in enumerate(lats):
        rtype = h_types[i]
        speed = SPEED[rtype]
        name = H_NAMES[i % len(H_NAMES)]
        for j in range(NCOLS - 1):
            p1 = (lons[j],   lat)
            p2 = (lons[j+1], lat)
            line = LineString([p1, p2])
            roads.append({
                "segment_id":   f"seg_{seg_id:05d}",
                "road_name":    name,
                "road_type":    rtype,
                "length":       haversine_m(p1[1], p1[0], p2[1], p2[0]),
                "baseline_speed": speed,
                "speed_kph":    speed,
                "importance":   ROAD_IMPORTANCE.get(rtype, 0.1),
                "capacity":     int(speed * rng.uniform(1.5, 3.0)),
                "geometry":     line,
            })
            seg_id += 1

    # Vertical streets — col j connects (lons[j], lats[i]) → (lons[j], lats[i+1])
    for j, lon in enumerate(lons):
        rtype = v_types[j]
        speed = SPEED[rtype]
        name = V_NAMES[j % len(V_NAMES)]
        for i in range(NROWS - 1):
            p1 = (lon, lats[i])
            p2 = (lon, lats[i+1])
            line = LineString([p1, p2])
            roads.append({
                "segment_id":   f"seg_{seg_id:05d}",
                "road_name":    name,
                "road_type":    rtype,
                "length":       haversine_m(p1[1], p1[0], p2[1], p2[0]),
                "baseline_speed": speed,
                "speed_kph":    speed,
                "importance":   ROAD_IMPORTANCE.get(rtype, 0.1),
                "capacity":     int(speed * rng.uniform(1.5, 3.0)),
                "geometry":     line,
            })
            seg_id += 1

    # ── Diagonal connector roads (start AND end snapped to grid nodes) ──────
    # Pick random pairs of grid nodes and connect them; this simulates organic
    # boulevards.  Because endpoints are exact grid nodes they stay connected.
    diag_pairs = [
        # (row_a, col_a, row_b, col_b)
        (0, 0, 3, 4), (1, 2, 5, 7), (2, 1, 6, 9), (3, 5, 8, 12),
        (4, 0, 7, 6), (5, 3, 10, 8), (6, 2, 11, 13), (7, 6, 12, 10),
        (0, 8, 4, 14), (2, 10, 6, 16), (8, 1, 13, 5), (9, 7, 12, 15),
    ]
    diag_names = [
        "Avenue Royale", "Rue du Port", "Boulevard Côtier",
        "Avenue Panoramique", "Rue des Orangers", "Boulevard de Paris",
    ]
    for k, (ri, ci, rj, cj) in enumerate(diag_pairs):
        ri, rj = min(ri, NROWS-1), min(rj, NROWS-1)
        ci, cj = min(ci, NCOLS-1), min(cj, NCOLS-1)
        p1 = (lons[ci], lats[ri])
        p2 = (lons[cj], lats[rj])
        rtype = "secondary"
        speed = SPEED[rtype]
        roads.append({
            "segment_id":   f"seg_{seg_id:05d}",
            "road_name":    diag_names[k % len(diag_names)],
            "road_type":    rtype,
            "length":       haversine_m(p1[1], p1[0], p2[1], p2[0]),
            "baseline_speed": speed,
            "speed_kph":    speed,
            "importance":   ROAD_IMPORTANCE.get(rtype, 0.1),
            "capacity":     int(speed * rng.uniform(1.5, 3.0)),
            "geometry":     LineString([p1, p2]),
        })
        seg_id += 1

    gdf = gpd.GeoDataFrame(roads, crs="+proj=longlat +datum=WGS84 +no_defs")
    save_geojson(gdf, "roads.geojson")
    print(f"  ✓ Road network: {len(gdf)} segments, fully connected grid")
    return gdf


def haversine_m(lat1, lon1, lat2, lon2):
    """Quick haversine distance in metres (used only in data_prep)."""
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def generate_sample_tram_lines(rng):
    """
    Generate sample tram lines that follow road grid nodes so they integrate
    into the road graph and appear on top of roads on the map.
    """
    print("Generating sample tram lines...")
    west, south, east, north = CASA_BBOX

    # Use the same grid parameters as generate_sample_roads
    NCOLS, NROWS = 18, 14
    lons = np.linspace(west + 0.02, east - 0.02, NCOLS)
    lats = np.linspace(south + 0.015, north - 0.015, NROWS)

    # T1 — east-west through city centre (row ~7), across most columns
    t1_row = 7
    t1_cols = [2, 4, 6, 8, 10, 12, 14, 16]
    t1_coords = [(lons[c], lats[t1_row]) for c in t1_cols]

    # T2 — north-south (col ~8), across most rows
    t2_col = 8
    t2_rows = [2, 4, 6, 7, 8, 10, 12]
    t2_coords = [(lons[t2_col], lats[r]) for r in t2_rows]

    tram_lines = gpd.GeoDataFrame([
        {
            "line_id": "T1",
            "line_name": "Tramway Line T1 — Sidi Moumen / Ain Diab",
            "color": "#0057A8",
            "geometry": LineString(t1_coords),
        },
        {
            "line_id": "T2",
            "line_name": "Tramway Line T2 — Ain Sebaa / Hay Hassani",
            "color": "#E8000B",
            "geometry": LineString(t2_coords),
        },
    ], crs="+proj=longlat +datum=WGS84 +no_defs")

    save_geojson(tram_lines, "tram_lines.geojson")
    print("  ✓ Tram lines snapped to road grid nodes")
    return tram_lines


def generate_sample_pois(rng, n_pois=150):
    """Generate sample POIs across Casablanca."""
    print("Generating sample POIs...")
    west, south, east, north = CASA_BBOX

    poi_types = list(POI_WEIGHTS.keys())
    poi_probs = [0.15, 0.2, 0.05, 0.1, 0.08, 0.12, 0.1, 0.08, 0.12]

    pois = []
    # Cluster POIs more densely near center
    for i in range(n_pois):
        # 60% near center, 40% spread out
        if rng.random() < 0.6:
            lat = CASA_CENTER[0] + rng.normal(0, 0.02)
            lon = CASA_CENTER[1] + rng.normal(0, 0.025)
        else:
            lat = rng.uniform(south, north)
            lon = rng.uniform(west, east)

        ptype = rng.choice(poi_types, p=poi_probs)
        names_by_type = {
            "hotel": ["Hotel Hyatt", "Riad Casablanca", "Hotel Kenzi", "Sofitel Casa", "Ibis Budget"],
            "school": ["École Primaire", "Lycée Mohammed V", "Collège Al Khawarizmi", "École Internationale"],
            "university": ["Université Hassan II", "ISCAE", "EMI Casablanca"],
            "hospital": ["CHU Ibn Rochd", "Clinique Badr", "Hôpital Moulay Youssef"],
            "mall": ["Morocco Mall", "Anfa Place", "Marina Shopping"],
            "marketplace": ["Marché Central", "Souk Habous", "Derb Omar"],
            "bus_station": ["Gare Routière Ouled Ziane", "Station CTM"],
            "attraction": ["Hassan II Mosque", "Corniche Ain Diab", "Parc de la Ligue Arabe"],
            "supermarket": ["Marjane", "Carrefour Market", "Acima", "Label Vie"],
        }
        pois.append({
            "poi_id": f"poi_{i:04d}",
            "poi_type": ptype,
            "name": rng.choice(names_by_type.get(ptype, ["Unknown"])),
            "geometry": Point(lon, lat),
        })

    gdf = gpd.GeoDataFrame(pois, crs="+proj=longlat +datum=WGS84 +no_defs")
    save_geojson(gdf, "pois.geojson")
    return gdf


def generate_sample_population(rng, resolution=0.005):
    """Generate population density grid (~500m resolution)."""
    print("Generating sample population grid...")
    west, south, east, north = CASA_BBOX

    lats = np.arange(south, north, resolution)
    lons = np.arange(west, east, resolution)

    cells = []
    grid_id = 0
    for lat in lats:
        for lon in lons:
            # Higher population near center, decreasing outward
            dist_to_center = np.sqrt(
                (lat - CASA_CENTER[0]) ** 2 + (lon - CASA_CENTER[1]) ** 2
            )
            # Base population with distance decay + noise
            base_pop = max(0, 5000 * np.exp(-dist_to_center / 0.05))
            population = int(base_pop * rng.uniform(0.3, 1.7))

            cell_geom = box(lon, lat, lon + resolution, lat + resolution)
            cells.append({
                "grid_id": f"grid_{grid_id:04d}",
                "population": population,
                "geometry": cell_geom,
            })
            grid_id += 1

    gdf = gpd.GeoDataFrame(cells, crs="+proj=longlat +datum=WGS84 +no_defs")
    gdf = gdf[gdf["population"] > 0]
    save_geojson(gdf, "pop_grid.geojson")
    return gdf


def generate_all_sample_data():
    """Generate all sample datasets for development."""
    print("\n" + "=" * 60)
    print("  GENERATING SAMPLE DATA FOR CASABLANCA")
    print("=" * 60 + "\n")
    ensure_dirs()
    rng = np.random.default_rng(42)

    roads = generate_sample_roads(rng)
    tram = generate_sample_tram_lines(rng)
    pois = generate_sample_pois(rng)
    pop = generate_sample_population(rng)

    print(f"\n{'=' * 60}")
    print(f"  DONE! Generated datasets in data/processed/")
    print(f"    Roads:      {len(roads)} segments")
    print(f"    Tram lines: {len(tram)} lines")
    print(f"    POIs:       {len(pois)} points")
    print(f"    Pop grid:   {len(pop)} cells")
    print(f"{'=' * 60}\n")
    return roads, tram, pois, pop


def fetch_all_osm_data():
    """Fetch all real data from OpenStreetMap (requires network)."""
    print("\n" + "=" * 60)
    print("  FETCHING REAL DATA FROM OPENSTREETMAP")
    print("=" * 60 + "\n")
    ensure_dirs()

    try:
        G, roads = fetch_roads_osm()
        tram = fetch_tram_lines_osm()
        pois = fetch_pois_osm()
        print("\n⚠ Population data: Download WorldPop raster manually from worldpop.org")
        print("  Then run: python src/data_prep.py --process-worldpop <path_to_tif>")
        return G, roads, tram, pois
    except Exception as e:
        print(f"\n✗ Error fetching OSM data: {e}")
        print("  Try using --sample flag for development data")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════
# PUBLIC API — function names expected by webapp & tests
# ═══════════════════════════════════════════════════════════

def _get_rng():
    return np.random.default_rng(42)


def fetch_roads_offline():
    """Public alias: generate sample road network (no network required)."""
    ensure_dirs()
    return generate_sample_roads(_get_rng())


def generate_synthetic_tram_lines():
    """Public alias: generate synthetic tram lines."""
    ensure_dirs()
    return generate_sample_tram_lines(_get_rng())


def generate_synthetic_pois():
    """Public alias: generate synthetic POIs."""
    ensure_dirs()
    return generate_sample_pois(_get_rng())


def generate_population_grid():
    """Public alias: generate synthetic population grid."""
    ensure_dirs()
    return generate_sample_population(_get_rng())


def prepare_all_data():
    """Public alias: generate all datasets and return (roads, tram, pois, pop_grid)."""
    ensure_dirs()
    return generate_all_sample_data()


# ═══════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Casablanca Tramway Data Preparation")
    parser.add_argument("--sample", action="store_true",
                        help="Generate sample data (no network needed)")
    parser.add_argument("--osm", action="store_true",
                        help="Fetch real data from OpenStreetMap")
    args = parser.parse_args()

    if args.sample or (not args.osm):
        generate_all_sample_data()
    elif args.osm:
        fetch_all_osm_data()
