"""
Phase 3: Scenario Simulation — pyproj-free rewrite.
Uses haversine distance instead of projection for length calculations.
"""
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString

from src.route_network import CityNetwork
from src.demand_model import DemandModel
from src.utils import TRAM_SPEED_KMH

logger = logging.getLogger(__name__)
_WGS84 = "+proj=longlat +datum=WGS84 +no_defs"


def _haversine_line_length_km(geometry):
    """Compute LineString length in km using haversine — no pyproj needed."""
    if geometry is None: return 0.0
    R = 6_371.0
    coords = list(geometry.coords)
    total = 0.0
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i];  lon2, lat2 = coords[i+1]
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dp = np.radians(lat2-lat1); dl = np.radians(lon2-lon1)
        a = np.sin(dp/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2)**2
        total += R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return total


class ScenarioResult:
    def __init__(self, scenario_id, scenario_type, route_result, demand_score, connectivity):
        self.scenario_id   = scenario_id
        self.scenario_type = scenario_type
        self.route_result  = route_result
        self.demand_score  = demand_score
        self.connectivity  = connectivity

    @property
    def geometry(self):
        return self.route_result.get("geometry") if self.route_result else None

    @property
    def distance_km(self):
        return self.route_result.get("distance_km", 0) if self.route_result else 0

    @property
    def travel_time_min(self):
        return self.route_result.get("travel_time_min", 0) if self.route_result else 0

    @property
    def total_score(self):
        return self.demand_score.get("total_score", 0) if self.demand_score else 0

    @property
    def score_per_km(self):
        d = self.distance_km
        return round(self.total_score / d, 1) if d > 0 else 0

    def to_dict(self):
        return {
            "scenario_id":            self.scenario_id,
            "type":                   self.scenario_type,
            "total_score":            self.total_score,
            "covered_population":     self.demand_score.get("covered_population", 0),
            "total_pois":             self.demand_score.get("total_pois", 0),
            "distance_km":            round(self.distance_km, 2),
            "travel_time_min":        round(self.travel_time_min, 1),
            "connectivity":           round(self.connectivity, 3),
            "score_per_km":           self.score_per_km,
            "congestion_reduction_min": self._congestion_reduction(),
        }

    def _congestion_reduction(self):
        pop = self.demand_score.get("covered_population", 0) if self.demand_score else 0
        return round(pop * 0.15 * 5, 0)


class ScenarioSimulator:
    def __init__(self, network: CityNetwork, demand_model: DemandModel):
        self.network      = network
        self.demand_model = demand_model
        self.scenarios    = {}

    def simulate_manual_route(self, start_lat, start_lon, end_lat, end_lon, scenario_id=None):
        if scenario_id is None:
            scenario_id = f"manual_{len(self.scenarios)+1}"

        start_node, start_dist = self.network.snap_to_node(start_lat, start_lon)
        end_node,   end_dist   = self.network.snap_to_node(end_lat,   end_lon)

        if start_node is None or end_node is None:
            logger.error("Could not snap points to network.")
            return None

        route_result = self.network.shortest_path(start_node, end_node)
        if route_result is None:
            logger.error("No path found.")
            return None

        route_result["travel_time_min"] = (route_result["distance_km"] / TRAM_SPEED_KMH) * 60

        try:
            demand_score = self.demand_model.score_route(route_result["geometry"], route_id=scenario_id)
        except Exception as e:
            logger.warning(f"Demand scoring failed: {e}")
            demand_score = {"total_score": 0, "covered_population": 0, "total_pois": 0}

        connectivity = self.network.connectivity_score(route_result["path"])
        result = ScenarioResult(scenario_id, "manual", route_result, demand_score, connectivity)
        self.scenarios[scenario_id] = result
        return result

    def simulate_route_from_geometry(self, geometry, scenario_id, scenario_type="custom"):
        geom = geometry

        # Use haversine — NO pyproj
        length_km    = _haversine_line_length_km(geom)
        travel_time  = (length_km / TRAM_SPEED_KMH) * 60

        route_result = {
            "geometry":        geom,
            "distance_km":     length_km,
            "travel_time_min": travel_time,
            "path":            [],
            "num_segments":    0,
        }

        try:
            demand_score = self.demand_model.score_route(geom, route_id=scenario_id)
        except Exception as e:
            logger.warning(f"Demand scoring failed: {e}")
            demand_score = {"total_score": 0, "covered_population": 0, "total_pois": 0}

        coords    = list(geom.coords)
        s_node, _ = self.network.snap_to_node(coords[0][1],  coords[0][0])
        e_node, _ = self.network.snap_to_node(coords[-1][1], coords[-1][0])
        path      = [s_node, e_node] if s_node and e_node else []
        connectivity = self.network.connectivity_score(path)

        result = ScenarioResult(scenario_id, scenario_type, route_result, demand_score, connectivity)
        self.scenarios[scenario_id] = result
        return result

    def compare_scenarios(self, scenario_ids=None):
        ids  = scenario_ids or list(self.scenarios.keys())
        rows = [self.scenarios[sid].to_dict() for sid in ids if sid in self.scenarios]
        df   = pd.DataFrame(rows)
        if not df.empty:
            df["rank"] = df["total_score"].rank(ascending=False).astype(int)
            df = df.sort_values("rank")
        return df

    def clear_scenarios(self):
        self.scenarios.clear()
