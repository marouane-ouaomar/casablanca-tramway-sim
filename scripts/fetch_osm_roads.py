"""
Script: Fetch OSM road data for Casablanca.
Usage: python scripts/fetch_osm_roads.py [--offline]
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_prep import fetch_roads, fetch_roads_offline
from src.utils import save_geojson, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    ensure_dirs()
    offline = "--offline" in sys.argv

    if offline:
        roads = fetch_roads_offline()
    else:
        try:
            edges, nodes, G = fetch_roads()
            roads = edges
        except Exception as e:
            logging.warning(f"OSM fetch failed: {e}. Using offline mode.")
            roads = fetch_roads_offline()

    path = save_geojson(roads, "roads.geojson")
    logging.info(f"Saved {len(roads)} road segments to {path}")


if __name__ == "__main__":
    main()
