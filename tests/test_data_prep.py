"""Tests for data preparation module."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import geopandas as gpd
from shapely.geometry import Point, LineString

from src.data_prep import (
    fetch_roads_offline,
    generate_synthetic_tram_lines,
    generate_synthetic_pois,
    generate_population_grid,
)


class TestRoads:
    def test_offline_roads_returns_geodataframe(self):
        roads = fetch_roads_offline()
        assert isinstance(roads, gpd.GeoDataFrame)
        assert len(roads) > 0

    def test_offline_roads_have_geometry(self):
        roads = fetch_roads_offline()
        assert "geometry" in roads.columns
        assert roads.geometry.is_valid.all()

    def test_offline_roads_have_required_columns(self):
        roads = fetch_roads_offline()
        for col in ["road_type", "length", "speed_kph", "importance"]:
            assert col in roads.columns, f"Missing column: {col}"

    def test_offline_roads_exclude_motorways(self):
        roads = fetch_roads_offline()
        if "road_type" in roads.columns:
            assert "motorway" not in roads["road_type"].values

    def test_road_types_are_valid(self):
        roads = fetch_roads_offline()
        valid_types = {"primary", "secondary", "tertiary", "residential",
                       "unclassified", "living_street"}
        assert set(roads["road_type"].unique()).issubset(valid_types)


class TestTramLines:
    def test_tram_lines_returns_geodataframe(self):
        tram = generate_synthetic_tram_lines()
        assert isinstance(tram, gpd.GeoDataFrame)
        assert len(tram) >= 2

    def test_tram_lines_have_linestring_geometry(self):
        tram = generate_synthetic_tram_lines()
        for _, row in tram.iterrows():
            assert row.geometry.geom_type in ("LineString", "MultiLineString")

    def test_tram_lines_have_ids(self):
        tram = generate_synthetic_tram_lines()
        assert "line_id" in tram.columns
        assert tram["line_id"].nunique() == len(tram)


class TestPOIs:
    def test_pois_returns_geodataframe(self):
        pois = generate_synthetic_pois()
        assert isinstance(pois, gpd.GeoDataFrame)
        assert len(pois) > 50

    def test_pois_have_point_geometry(self):
        pois = generate_synthetic_pois()
        assert all(pois.geometry.geom_type == "Point")

    def test_pois_have_valid_coords(self):
        pois = generate_synthetic_pois()
        bounds = pois.total_bounds  # minx, miny, maxx, maxy
        assert bounds[0] > -8.0  # west
        assert bounds[1] > 33.0  # south
        assert bounds[2] < -7.0  # east
        assert bounds[3] < 34.0  # north

    def test_pois_have_types(self):
        pois = generate_synthetic_pois()
        assert "poi_type" in pois.columns
        assert pois["poi_type"].nunique() >= 5

    def test_pois_include_key_types(self):
        pois = generate_synthetic_pois()
        types = set(pois["poi_type"].unique())
        for expected in ["hotel", "school", "hospital"]:
            assert expected in types, f"Missing POI type: {expected}"


class TestPopulationGrid:
    def test_grid_returns_geodataframe(self):
        grid = generate_population_grid()
        assert isinstance(grid, gpd.GeoDataFrame)
        assert len(grid) > 10

    def test_grid_has_population(self):
        grid = generate_population_grid()
        assert "population" in grid.columns
        assert grid["population"].sum() > 0
        assert grid["population"].min() >= 0

    def test_grid_has_ids(self):
        grid = generate_population_grid()
        assert "grid_id" in grid.columns
        assert grid["grid_id"].nunique() == len(grid)

    def test_grid_covers_casablanca(self):
        grid = generate_population_grid()
        bounds = grid.total_bounds
        # Should roughly cover Casablanca bbox
        assert bounds[0] < -7.5  # west of center
        assert bounds[2] > -7.5  # east of center
