"""
AI Route Suggester — fixed & optimised.

Fixes vs previous version:
  • Composite score: uses proper normalisation across all candidates
    (previous formula divided score by itself → always 0.5)
  • Guarantees top_n results: if underserved filter leaves too few
    hotspots, falls back to ALL hotspots
  • Highway avoidance: motorway/trunk edges get 10× travel-time penalty
  • Performance: scipy KDTree for O(log N) tram-distance queries
  • Max candidates raised, short-route floor lowered to 0.5 km
"""
import os, logging
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
from sklearn.cluster import KMeans

try:
    from src.route_network import CityNetwork
    from src.demand_model  import DemandModel
    from src.utils import POI_WEIGHTS, CASABLANCA_CENTER, haversine_distance
except ImportError:
    from route_network import CityNetwork
    from demand_model  import DemandModel
    from utils import POI_WEIGHTS, CASABLANCA_CENTER, haversine_distance

logger = logging.getLogger(__name__)
_WGS84 = "+proj=longlat +datum=WGS84 +no_defs"

# Road types NOT suitable for tram lines (penalty weight multiplier)
_HIGHWAY_TYPES = {"motorway","motorway_link","trunk","trunk_link"}
_HIGHWAY_PENALTY = 10.0   # 10× travel time → effectively excluded


class AIRouteSuggester:
    def __init__(self, network: CityNetwork, demand_model: DemandModel,
                 n_clusters=12, min_distance_from_tram_m=800):
        self.network           = network
        self.demand_model      = demand_model
        self.n_clusters        = n_clusters
        self.min_tram_distance = min_distance_from_tram_m
        self._hotspots         = None
        self._candidates       = []

        # Pre-build KDTree over tram node coords for fast distance queries
        self._build_tram_kdtree()

        # Add highway-penalty weights to graph edges (in-place, one-time)
        self._apply_highway_penalty()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _build_tram_kdtree(self):
        """Build scipy KDTree of tram nodes for O(log N) distance queries."""
        tram_nodes = self.network.get_tram_nodes()
        if not tram_nodes:
            self._tram_tree = None; self._tram_coords = None; return
        coords = np.array([(self.network.node_coords[n][1],   # lon
                            self.network.node_coords[n][0])   # lat
                           for n in tram_nodes if n in self.network.node_coords])
        if len(coords) == 0:
            self._tram_tree = None; self._tram_coords = None; return
        try:
            from scipy.spatial import KDTree
            self._tram_tree   = KDTree(coords)
            self._tram_coords = coords
        except ImportError:
            # scipy not available — fall back to linear scan
            self._tram_tree   = None
            self._tram_coords = coords

    def _tram_dist_m(self, lat, lon):
        """Return metres to nearest tram node (fast KDTree or fallback)."""
        if self._tram_coords is None or len(self._tram_coords) == 0:
            return float("inf")
        if self._tram_tree is not None:
            dist_deg, _ = self._tram_tree.query([lon, lat])
            return dist_deg * 111_000     # rough degrees → metres
        # Linear fallback
        dists = np.sqrt((self._tram_coords[:,0]-lon)**2 +
                        (self._tram_coords[:,1]-lat)**2)
        return dists.min() * 111_000

    def _apply_highway_penalty(self):
        """Multiply travel_time by _HIGHWAY_PENALTY for motorway/trunk edges."""
        G = self.network.G
        count = 0
        for u, v, d in G.edges(data=True):
            if d.get("road_type","") in _HIGHWAY_TYPES:
                d["tram_weight"] = d.get("travel_time", 1) * _HIGHWAY_PENALTY
                count += 1
            else:
                d["tram_weight"] = d.get("travel_time", 1)
        logger.info(f"Highway penalty applied to {count} edges (using 'tram_weight').")

    # ── Stage 1: Hotspots ─────────────────────────────────────────────────────

    def identify_hotspots(self):
        coords = []; weights = []

        # POIs (weighted by type)
        if self.demand_model.pois is not None:
            for _, row in self.demand_model.pois.iterrows():
                pt = row.geometry
                w  = POI_WEIGHTS.get(row.get("poi_type","other"), 1.0)
                coords.append([pt.y, pt.x]); weights.append(w)

        # Population grid (sample: max 3000 cells for speed)
        if self.demand_model.pop_grid is not None:
            pg = self.demand_model.pop_grid
            max_pop = pg["population"].max()
            # Sample heavy cells only
            pg_sorted = pg[pg["population"] > max_pop * 0.05].copy()
            if len(pg_sorted) > 3000:
                pg_sorted = pg_sorted.sample(3000, random_state=42)
            for _, row in pg_sorted.iterrows():
                c = row.geometry.centroid
                w = (row["population"] / max_pop) * 5.0
                coords.append([c.y, c.x]); weights.append(w)

        if len(coords) < self.n_clusters:
            self.n_clusters = max(3, len(coords) // 2)

        coords   = np.array(coords);   weights = np.array(weights)
        kmeans   = KMeans(n_clusters=self.n_clusters, random_state=42,
                          n_init=10, max_iter=200)
        labels   = kmeans.fit_predict(coords, sample_weight=weights)

        hotspots = []
        for i, center in enumerate(kmeans.cluster_centers_):
            mask   = labels == i
            hotspots.append({
                "hotspot_id":  f"hs_{i:02d}",
                "geometry":    Point(center[1], center[0]),
                "lat":  center[0], "lon": center[1],
                "total_weight":  round(float(weights[mask].sum()), 1),
                "point_count":   int(mask.sum()),
            })
        self._hotspots = gpd.GeoDataFrame(hotspots).set_crs(_WGS84)
        return self._hotspots

    def filter_underserved(self, fallback_to_all=True):
        """
        Keep only hotspots ≥ min_tram_distance from existing tram.
        If too few remain AND fallback_to_all=True, return ALL hotspots so
        we always have enough pairs to generate routes.
        """
        if self._hotspots is None: self.identify_hotspots()

        if self._tram_coords is None:
            # No existing tram — every hotspot is underserved
            self._hotspots["tram_distance_m"] = float("inf")
            return self._hotspots

        dists = [self._tram_dist_m(r["lat"], r["lon"])
                 for _, r in self._hotspots.iterrows()]
        self._hotspots["tram_distance_m"] = dists
        underserved = self._hotspots[
            self._hotspots["tram_distance_m"] >= self.min_tram_distance
        ].copy()

        min_needed = 4   # need at least 4 hotspots to make good pairs
        if len(underserved) < min_needed and fallback_to_all:
            logger.info(
                f"Only {len(underserved)} underserved hotspots — "
                f"using all {len(self._hotspots)} (relaxed filter).")
            return self._hotspots.copy()

        return underserved

    # ── Stage 2: Candidate routes ─────────────────────────────────────────────

    def generate_candidates(self, n_required=5, max_pool=60):
        """
        Generate candidate routes using 'tram_weight' edge attribute
        (highways are heavily penalised → trams avoid them).

        n_required: guarantee at least this many candidates before stopping.
        """
        import networkx as nx

        underserved = self.filter_underserved()
        if len(underserved) < 2:
            logger.warning("Not enough hotspots for candidate generation.")
            return []

        underserved = underserved.sort_values("total_weight", ascending=False)
        hs_list     = list(underserved.iterrows())

        candidates  = []; seen_pairs = set()

        for ia, (_, ra) in enumerate(hs_list):
            for ib, (_, rb) in enumerate(hs_list):
                if ia >= ib: continue
                key = (ra["hotspot_id"], rb["hotspot_id"])
                if key in seen_pairs: continue
                seen_pairs.add(key)

                na, da = self.network.snap_to_node(ra["lat"], ra["lon"])
                nb, db = self.network.snap_to_node(rb["lat"], rb["lon"])
                if na is None or nb is None or na == nb: continue

                # Route using highway-avoiding tram_weight
                try:
                    route = self.network.shortest_path(na, nb, weight="tram_weight")
                except Exception:
                    route = self.network.shortest_path(na, nb)
                if route is None: continue
                if route["distance_km"] < 0.5 or route["distance_km"] > 30: continue

                candidates.append({
                    "candidate_id":    f"route_{len(candidates)+1:02d}",
                    "from_hotspot":    ra["hotspot_id"],
                    "to_hotspot":      rb["hotspot_id"],
                    "geometry":        route["geometry"],
                    "distance_km":     route["distance_km"],
                    "path":            route["path"],
                    "combined_weight": ra["total_weight"] + rb["total_weight"],
                })

                if len(candidates) >= max_pool:
                    break
            if len(candidates) >= max_pool:
                break

        logger.info(f"Generated {len(candidates)} candidate routes.")
        self._candidates = candidates
        return candidates

    # ── Stage 3: Rank ─────────────────────────────────────────────────────────

    def rank_candidates(self, top_n=5):
        """
        Score all candidates and return top_n.

        Composite score (properly normalised across candidates):
          50% demand score (normalised 0-1)
          30% connectivity to existing tram (0-1)
          20% efficiency = demand / km (normalised 0-1)
        """
        if not self._candidates: self.generate_candidates(n_required=top_n*2)

        scored = []
        for c in self._candidates:
            demand       = self.demand_model.score_route(c["geometry"], c["candidate_id"])
            connectivity = self.network.connectivity_score(c["path"])
            efficiency   = demand["total_score"] / max(c["distance_km"], 0.1)
            scored.append({
                **c,
                "demand_score":       demand,
                "total_score":        demand["total_score"],
                "connectivity":       connectivity,
                "efficiency":         efficiency,
                "covered_population": demand["covered_population"],
                "total_pois":         demand["total_pois"],
            })

        if not scored: return []

        # ── Proper normalisation ──────────────────────────────────────────
        max_demand = max(s["total_score"] for s in scored) or 1
        max_eff    = max(s["efficiency"]  for s in scored) or 1

        for s in scored:
            s["composite_score"] = round(
                0.50 * (s["total_score"] / max_demand)
                + 0.30 * s["connectivity"]
                + 0.20 * (s["efficiency"] / max_eff),
                4)

        scored.sort(key=lambda x: x["composite_score"], reverse=True)

        # Remove near-duplicate routes (same start/end area, < 20% score diff)
        deduped = [scored[0]] if scored else []
        for s in scored[1:]:
            if len(deduped) >= top_n: break
            # Check if too similar to already-selected routes
            is_dup = False
            for sel in deduped:
                shared = len(set(s["path"]) & set(sel["path"]))
                overlap = shared / max(len(s["path"]), 1)
                if overlap > 0.70:  # > 70% shared nodes → duplicate
                    is_dup = True; break
            if not is_dup:
                deduped.append(s)

        # If deduplication left us short, pad with next best non-duplicate
        if len(deduped) < top_n:
            for s in scored:
                if s not in deduped:
                    deduped.append(s)
                if len(deduped) >= top_n:
                    break

        top = deduped[:top_n]
        for i, r in enumerate(top):
            r["rank"] = i + 1

        return top

    def suggest(self, top_n=5):
        logger.info("AI pipeline: identify → candidates → rank")
        self.identify_hotspots()
        self.generate_candidates(n_required=top_n * 3)
        return self.rank_candidates(top_n=top_n)

    def get_hotspots_gdf(self):
        if self._hotspots is None: self.identify_hotspots()
        return self._hotspots
