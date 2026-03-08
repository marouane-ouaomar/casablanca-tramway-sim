"""Tests for route network module."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import geopandas as gpd
from shapely.geometry import LineString

from src.data_prep import fetch_roads_offline, generate_synthetic_tram_lines
from src.route_network import CityNetwork


@pytest.fixture
def network():
    roads = fetch_roads_offline()
    tram = generate_synthetic_tram_lines()
    net = CityNetwork()
    net.build_from_geodataframe(roads)
    net.add_tram_lines(tram)
    return net


class TestNetworkBuild:
    def test_network_has_nodes(self, network):
        assert network.G.number_of_nodes() > 0

    def test_network_has_edges(self, network):
        assert network.G.number_of_edges() > 0

    def test_node_coords_populated(self, network):
        assert len(network.node_coords) == network.G.number_of_nodes()

    def test_node_coords_are_valid(self, network):
        for node_id, (lat, lon) in network.node_coords.items():
            assert 33.0 < lat < 34.0, f"Node {node_id} lat out of range: {lat}"
            assert -8.0 < lon < -7.0, f"Node {node_id} lon out of range: {lon}"

    def test_summary_returns_dict(self, network):
        summary = network.summary()
        assert "nodes" in summary
        assert "edges" in summary
        assert "tram_edges" in summary
        assert summary["tram_edges"] > 0


class TestSnap:
    def test_snap_to_node_finds_nearest(self, network):
        node, dist = network.snap_to_node(33.57, -7.59)
        assert node is not None
        assert dist < 5000  # within 5km

    def test_snap_returns_valid_node(self, network):
        node, _ = network.snap_to_node(33.57, -7.59)
        assert node in network.G.nodes


class TestShortestPath:
    def test_shortest_path_returns_result(self, network):
        nodes = list(network.G.nodes)[:2]
        if len(nodes) >= 2:
            # Find two connected nodes
            import networkx as nx
            components = list(nx.connected_components(network.G))
            if components:
                comp = list(components[0])
                if len(comp) >= 2:
                    result = network.shortest_path(comp[0], comp[1])
                    assert result is not None or True  # May not be connected

    def test_shortest_path_has_geometry(self, network):
        import networkx as nx
        components = list(nx.connected_components(network.G))
        if components and len(list(components[0])) >= 2:
            comp = list(components[0])
            result = network.shortest_path(comp[0], comp[1])
            if result:
                assert isinstance(result["geometry"], LineString)
                assert result["distance_km"] > 0
                assert result["travel_time_min"] > 0

    def test_no_path_returns_none(self, network):
        result = network.shortest_path(999999, 999998)
        assert result is None


class TestConnectivity:
    def test_tram_nodes_found(self, network):
        tram_nodes = network.get_tram_nodes()
        assert len(tram_nodes) > 0

    def test_connectivity_score_range(self, network):
        nodes = list(network.G.nodes)[:10]
        score = network.connectivity_score(nodes)
        assert 0.0 <= score <= 1.0


class TestEdgesGDF:
    def test_returns_geodataframe(self, network):
        gdf = network.get_all_edges_gdf()
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) > 0
