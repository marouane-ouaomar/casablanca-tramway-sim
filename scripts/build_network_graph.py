"""
Script: Build the NetworkX graph from road + tram data.
Usage: python scripts/build_network_graph.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.route_network import CityNetwork
from src.utils import load_geojson

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    roads = load_geojson("roads.geojson")
    tram = load_geojson("tram_lines.geojson")

    network = CityNetwork()
    network.build_from_geodataframe(roads)
    network.add_tram_lines(tram)

    summary = network.summary()
    logging.info(f"Network summary: {summary}")

    network.save()
    logging.info("Network graph saved.")


if __name__ == "__main__":
    main()
