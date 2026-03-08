"""Tests for demand model module."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from shapely.geometry import LineString

from src.data_prep import generate_synthetic_pois, generate_population_grid
from src.demand_model import DemandModel


@pytest.fixture
def demand_model():
    pois = generate_synthetic_pois()
    pop_grid = generate_population_grid()
    return DemandModel(pop_grid=pop_grid, pois=pois)


@pytest.fixture
def sample_route():
    """A route near Casablanca city center."""
    return LineString([
        (-7.62, 33.58), (-7.60, 33.57), (-7.58, 33.56),
        (-7.56, 33.55), (-7.54, 33.55),
    ])


class TestDemandScoring:
    def test_score_route_returns_dict(self, demand_model, sample_route):
        result = demand_model.score_route(sample_route)
        assert isinstance(result, dict)
        assert "total_score" in result
        assert "covered_population" in result
        assert "total_pois" in result

    def test_score_is_positive_near_center(self, demand_model, sample_route):
        result = demand_model.score_route(sample_route)
        assert result["total_score"] > 0
        assert result["covered_population"] > 0

    def test_score_includes_route_length(self, demand_model, sample_route):
        result = demand_model.score_route(sample_route)
        assert result["route_length_km"] > 0
        assert result["score_per_km"] > 0

    def test_score_includes_poi_breakdown(self, demand_model, sample_route):
        result = demand_model.score_route(sample_route)
        assert "poi_breakdown" in result
        assert isinstance(result["poi_breakdown"], dict)

    def test_remote_route_has_lower_score(self, demand_model):
        """A route far from center should score lower."""
        center_route = LineString([(-7.60, 33.58), (-7.55, 33.56)])
        remote_route = LineString([(-7.70, 33.64), (-7.68, 33.63)])

        center_score = demand_model.score_route(center_route)
        remote_score = demand_model.score_route(remote_route)

        assert center_score["total_score"] >= remote_score["total_score"]


class TestMultipleRoutes:
    def test_score_multiple_routes(self, demand_model):
        routes = [
            {"route_id": "r1", "geometry": LineString([(-7.60, 33.58), (-7.55, 33.56)])},
            {"route_id": "r2", "geometry": LineString([(-7.62, 33.57), (-7.58, 33.55)])},
        ]
        df = demand_model.score_multiple_routes(routes)
        assert len(df) == 2
        assert "rank" in df.columns
        assert df["rank"].min() == 1


class TestCoverageMap:
    def test_coverage_map_returns_geodataframes(self, demand_model, sample_route):
        result = demand_model.generate_coverage_map(sample_route)
        assert "catchment" in result
        assert len(result["catchment"]) > 0


class TestHeatmapData:
    def test_heatmap_data_returns_list(self, demand_model):
        data = demand_model.get_demand_heatmap_data()
        assert isinstance(data, list)
        assert len(data) > 0
        assert len(data[0]) == 3  # [lat, lon, intensity]
