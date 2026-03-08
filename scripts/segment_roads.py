"""
Script: Segment roads into smaller pieces for traffic simulation.
Usage: python scripts/segment_roads.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.traffic_simulation import segment_roads
from src.utils import load_geojson, save_geojson

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    roads = load_geojson("roads.geojson")
    segments = segment_roads(roads, max_segment_length_m=200)
    path = save_geojson(segments, "road_segments.geojson")
    logging.info(f"Saved {len(segments)} segments to {path}")


if __name__ == "__main__":
    main()
