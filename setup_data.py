"""
setup_data.py — One-time real OSM data download for Casablanca.

Run this ONCE before launching the app:
    python setup_data.py

Requires internet. Downloads real roads, tram lines, and POIs
from OpenStreetMap using osmnx, saves to data/processed/.

If you have no internet, the app will auto-generate synthetic data,
but routes will NOT follow real streets.
"""

import os, sys, pickle
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 60)
    print("  Casablanca OSM Data Setup")
    print("=" * 60)

    try:
        import osmnx as ox
        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Point
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Run: conda install -c conda-forge osmnx")
        sys.exit(1)

    ox.settings.log_console = False
    ox.settings.use_cache   = True
    PLACE = "Casablanca, Morocco"

    # ── 1. Road network ────────────────────────────────────────
    print("\n[1/4] Downloading road network from OSM …")
    G = ox.graph_from_place(PLACE, network_type="drive", simplify=True)

    # Save the raw osmnx graph (used by CityNetwork for routing)
    with open(PROCESSED / "osm_graph.pkl", "wb") as f:
        pickle.dump(G, f)
    print(f"      Saved osm_graph.pkl  ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")

    # Also save edges as GeoJSON for display
    nodes, edges = ox.graph_to_gdfs(G)
    edges = edges.reset_index()
    keep = [c for c in ["u","v","key","highway","name","length","geometry"] if c in edges.columns]
    edges[keep].to_file(PROCESSED / "roads.geojson", driver="GeoJSON")
    print(f"      Saved roads.geojson  ({len(edges)} segments)")

    # ── 2. Tram lines ──────────────────────────────────────────
    print("\n[2/4] Downloading tram lines from OSM …")
    try:
        tram = ox.features_from_place(PLACE, tags={"railway": "tram"})
        tram = tram[tram.geometry.geom_type.isin(["LineString","MultiLineString"])]
        tram = tram.reset_index()[["name","geometry"]].copy()
        tram["line_id"] = ["T" + str(i+1) for i in range(len(tram))]
        tram["color"]   = ["#0057A8","#E8000B","#00897B","#F4511E","#8E24AA"][:len(tram)]
        tram["line_name"] = tram["name"].fillna(tram["line_id"])
        tram.to_file(PROCESSED / "tram_lines.geojson", driver="GeoJSON")
        print(f"      Saved tram_lines.geojson  ({len(tram)} lines)")
    except Exception as e:
        print(f"      ⚠ Tram fetch failed ({e}) — using fallback coords")
        _save_fallback_tram()

    # ── 3. POIs ────────────────────────────────────────────────
    print("\n[3/4] Downloading POIs from OSM …")
    try:
        tags = {
            "amenity": ["school","hospital","university","bus_station","clinic","marketplace"],
            "tourism": ["hotel","attraction","museum"],
            "shop":    ["mall","supermarket"],
        }
        pois = ox.features_from_place(PLACE, tags=tags)
        pois["geometry"] = pois.geometry.centroid
        pois = pois[pois.geometry.geom_type == "Point"].copy()

        def _ptype(row):
            for col in ["tourism","shop","amenity"]:
                v = row.get(col)
                if v and str(v) not in ["nan","None"]:
                    return str(v)
            return "other"

        pois["poi_type"] = pois.apply(_ptype, axis=1)
        out = pois.reset_index()[["name","poi_type","geometry"]].copy()
        out["name"] = out["name"].fillna(out["poi_type"].str.title())
        out.to_file(PROCESSED / "pois.geojson", driver="GeoJSON")
        print(f"      Saved pois.geojson  ({len(out)} POIs)")
    except Exception as e:
        print(f"      ⚠ POI fetch failed ({e}) — generating synthetic POIs")
        _save_fallback_pois()

    # ── 4. Population grid (synthetic — no free API) ───────────
    print("\n[4/4] Generating population density grid (synthetic) …")
    _save_population_grid()

    print("\n" + "=" * 60)
    print("  ✅  All data saved to data/processed/")
    print("  Now run:  streamlit run webapp/streamlit_app.py")
    print("=" * 60)


def _save_fallback_tram():
    import geopandas as gpd
    from shapely.geometry import LineString
    PROCESSED = Path(__file__).parent / "data" / "processed"
    tram = gpd.GeoDataFrame([
        {"line_id":"T1","line_name":"Tramway T1 — Sidi Moumen / Ain Diab",
         "color":"#0057A8",
         "geometry": LineString([(-7.510,33.582),(-7.540,33.580),(-7.565,33.575),
                                  (-7.590,33.573),(-7.615,33.570),(-7.640,33.565)])},
        {"line_id":"T2","line_name":"Tramway T2 — Ain Sebaa / Hay Hassani",
         "color":"#E8000B",
         "geometry": LineString([(-7.553,33.618),(-7.560,33.600),(-7.568,33.582),
                                  (-7.578,33.565),(-7.590,33.548),(-7.605,33.525)])},
    ], crs="+proj=longlat +datum=WGS84 +no_defs")
    tram.to_file(PROCESSED / "tram_lines.geojson", driver="GeoJSON")


def _save_fallback_pois():
    import numpy as np, geopandas as gpd
    from shapely.geometry import Point
    PROCESSED = Path(__file__).parent / "data" / "processed"
    from src.utils import CASA_CENTER, CASA_BBOX, POI_WEIGHTS
    rng = np.random.default_rng(42)
    west, south, east, north = CASA_BBOX
    types = list(POI_WEIGHTS.keys())
    rows = []
    for i in range(200):
        lat = CASA_CENTER[0] + rng.normal(0, 0.025) if rng.random() < 0.6 else rng.uniform(south, north)
        lon = CASA_CENTER[1] + rng.normal(0, 0.030) if rng.random() < 0.6 else rng.uniform(west, east)
        pt  = rng.choice(types)
        rows.append({"poi_type": pt, "name": pt.title(), "geometry": Point(lon, lat)})
    gpd.GeoDataFrame(rows, crs="+proj=longlat +datum=WGS84 +no_defs").to_file(PROCESSED / "pois.geojson", driver="GeoJSON")


def _save_population_grid():
    import numpy as np, geopandas as gpd
    from shapely.geometry import box
    PROCESSED = Path(__file__).parent / "data" / "processed"
    from src.utils import CASA_CENTER, CASA_BBOX
    rng  = np.random.default_rng(42)
    west, south, east, north = CASA_BBOX
    res  = 0.005
    rows = []
    gid  = 0
    for lat in np.arange(south, north, res):
        for lon in np.arange(west, east, res):
            d   = ((lat - CASA_CENTER[0])**2 + (lon - CASA_CENTER[1])**2) ** 0.5
            pop = int(max(0, 6000 * np.exp(-d / 0.05)) * rng.uniform(0.4, 1.6))
            if pop > 0:
                rows.append({"grid_id": f"g{gid:05d}", "population": pop,
                              "geometry": box(lon, lat, lon+res, lat+res)})
            gid += 1
    gpd.GeoDataFrame(rows, crs="+proj=longlat +datum=WGS84 +no_defs").to_file(PROCESSED / "pop_grid.geojson", driver="GeoJSON")


if __name__ == "__main__":
    main()
