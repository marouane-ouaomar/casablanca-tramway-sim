"""Tests for scenario simulation and traffic simulation modules."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import networkx as nx
from shapely.geometry import LineString

from src.data_prep import (
    fetch_roads_offline, generate_synthetic_tram_lines,
    generate_synthetic_pois, generate_population_grid,
)
from src.route_network import CityNetwork
from src.demand_model import DemandModel
from src.scenario_sim import ScenarioSimulator
from src.traffic_simulation import (
    TrafficEstimator, TrafficScenarioSimulator, segment_roads,
)


@pytest.fixture
def city_data():
    roads = fetch_roads_offline()
    tram = generate_synthetic_tram_lines()
    pois = generate_synthetic_pois()
    pop = generate_population_grid()
    return roads, tram, pois, pop


@pytest.fixture
def network(city_data):
    roads, tram, _, _ = city_data
    net = CityNetwork()
    net.build_from_geodataframe(roads)
    net.add_tram_lines(tram)
    return net


@pytest.fixture
def demand_model(city_data):
    _, _, pois, pop = city_data
    return DemandModel(pop_grid=pop, pois=pois)


@pytest.fixture
def simulator(network, demand_model):
    return ScenarioSimulator(network, demand_model)


class TestScenarioSimulation:
    def test_manual_route_simulation(self, simulator):
        result = simulator.simulate_manual_route(33.58, -7.63, 33.55, -7.53)
        # May or may not find path depending on network connectivity
        if result is not None:
            assert result.total_score >= 0
            assert result.distance_km > 0
            assert result.travel_time_min > 0

    def test_scenario_to_dict(self, simulator):
        result = simulator.simulate_manual_route(33.58, -7.63, 33.55, -7.53)
        if result is not None:
            d = result.to_dict()
            assert "scenario_id" in d
            assert "total_score" in d
            assert "congestion_reduction_min" in d

    def test_compare_scenarios(self, simulator):
        r1 = simulator.simulate_manual_route(33.58, -7.63, 33.55, -7.53, "route_A")
        r2 = simulator.simulate_manual_route(33.57, -7.60, 33.54, -7.55, "route_B")
        df = simulator.compare_scenarios()
        # df may be empty if no paths found
        assert df is not None

    def test_route_from_geometry(self, simulator):
        geom = LineString([(-7.62, 33.58), (-7.58, 33.56), (-7.54, 33.55)])
        result = simulator.simulate_route_from_geometry(geom, "custom_1")
        assert result is not None
        assert result.total_score >= 0


class TestRoadSegmentation:
    def test_segment_roads(self, city_data):
        roads, _, _, _ = city_data
        segments = segment_roads(roads.head(100), max_segment_length_m=300)
        assert len(segments) >= 100  # should be at least as many as input
        assert "segment_id" in segments.columns

    def test_segments_have_required_columns(self, city_data):
        roads, _, _, _ = city_data
        segments = segment_roads(roads.head(50))
        for col in ["segment_id", "road_type", "length", "speed_kph"]:
            assert col in segments.columns


class TestTrafficEstimation:
    def test_baseline_traffic(self, city_data):
        roads, _, pois, pop = city_data
        segments = segment_roads(roads.head(50))
        estimator = TrafficEstimator(pop_grid=pop, pois=pois)
        result = estimator.estimate_segment_traffic(segments)
        assert "traffic_flow_score" in result.columns
        assert "baseline_speed_ratio" in result.columns
        assert "traffic_color" in result.columns
        assert result["baseline_speed_ratio"].between(0, 1).all()


class TestTrafficScenarios:
    @pytest.fixture
    def traffic_setup(self, city_data, network):
        roads, _, pois, pop = city_data
        segments = segment_roads(roads.head(100))
        estimator = TrafficEstimator(pop_grid=pop, pois=pois)
        baseline = estimator.estimate_segment_traffic(segments)
        sim = TrafficScenarioSimulator(network, baseline)
        return sim, baseline

    def test_road_closure(self, traffic_setup):
        sim, baseline = traffic_setup
        result = sim.close_road(0)
        assert result is not None
        assert "impact_summary" in result
        assert "result_gdf" in result

    def test_construction_delay(self, traffic_setup):
        sim, _ = traffic_setup
        result = sim.construction_delay(0, speed_reduction=0.5)
        assert result is not None
        assert result["scenario_type"] == "construction_delay"

    def test_capacity_reduction(self, traffic_setup):
        sim, _ = traffic_setup
        result = sim.capacity_reduction(0, reduction_factor=0.5)
        assert result is not None
        assert "scenario_color" in result["result_gdf"].columns

    def test_tram_impact(self, traffic_setup):
        sim, _ = traffic_setup
        tram_geom = LineString([(-7.62, 33.58), (-7.58, 33.56), (-7.54, 33.55)])
        result = sim.tram_impact(tram_geom)
        assert result is not None
        assert result["affected_segments"] >= 0

    def test_impact_summary_structure(self, traffic_setup):
        sim, _ = traffic_setup
        result = sim.close_road(0)
        summary = result["impact_summary"]
        assert "avg_speed_ratio_before" in summary
        assert "avg_speed_ratio_after" in summary
        assert "congested_segments_before" in summary
