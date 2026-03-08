"""
Route Network — wraps a plain NetworkX graph for routing.

Accepts either:
  CityNetwork(osm_graph=G, tram_lines_raw=[...])   ← from Overpass download
  CityNetwork(roads_gdf=gdf, tram_gdf=gdf)         ← legacy GDF path

No pyproj / CRS calls anywhere in this file.
"""

import logging
import numpy as np
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString

try:
    from src.utils import ROAD_IMPORTANCE, haversine_distance, DEFAULT_CRS
except ImportError:
    from utils import ROAD_IMPORTANCE, haversine_distance, DEFAULT_CRS

logger = logging.getLogger(__name__)

_WGS84 = "+proj=longlat +datum=WGS84 +no_defs"


class CityNetwork:
    def __init__(self, roads_gdf=None, tram_gdf=None,
                 osm_graph=None, tram_lines_raw=None):
        """
        osm_graph      : pre-built NetworkX graph (nodes have lat/lon attrs,
                         edges have length, travel_time, coords, is_tram)
        tram_lines_raw : list of {line_id, line_name, color, coords:[(lon,lat),...]}
        roads_gdf      : fallback GeoDataFrame (legacy)
        tram_gdf       : fallback tram GeoDataFrame (legacy)
        """
        self.G            = nx.Graph()
        self._id_to_coords = {}   # node_id → (lat, lon)
        self._coord_to_id  = {}   # (round_lon, round_lat) → node_id
        self._osmnx_mode   = False

        if osm_graph is not None:
            self._load_plain_graph(osm_graph)
            self._osmnx_mode = True
            logger.info(f"Loaded plain graph: {self.G.number_of_nodes()} nodes, "
                        f"{self.G.number_of_edges()} edges")
        elif roads_gdf is not None:
            self._build_from_geodataframe(roads_gdf)

        if tram_lines_raw is not None:
            self._add_tram_lines_raw(tram_lines_raw)
        elif tram_gdf is not None:
            self._add_tram_lines_gdf(tram_gdf)

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_plain_graph(self, G_src):
        """Copy a plain NetworkX graph (from Overpass or fallback) into self.G."""
        for nid, d in G_src.nodes(data=True):
            lat = float(d.get("lat", d.get("y", 0)))
            lon = float(d.get("lon", d.get("x", 0)))
            self.G.add_node(nid, lat=lat, lon=lon)
            self._id_to_coords[nid] = (lat, lon)
            self._coord_to_id[(round(lon,5), round(lat,5))] = nid

        for u, v, d in G_src.edges(data=True):
            if self.G.has_edge(u, v): continue
            coords = d.get("coords")
            if coords is None:
                # Build from node positions
                ud = self.G.nodes.get(u, {}); vd = self.G.nodes.get(v, {})
                coords = [(ud.get("lon",0), ud.get("lat",0)),
                          (vd.get("lon",0), vd.get("lat",0))]
            geom = LineString(coords)
            self.G.add_edge(u, v,
                            length      = float(d.get("length", 1)),
                            speed       = float(d.get("speed", 25)),
                            travel_time = float(d.get("travel_time", 0.1)),
                            road_type   = d.get("road_type", "unclassified"),
                            name        = d.get("name", ""),
                            geometry    = geom,
                            is_tram     = bool(d.get("is_tram", False)))

    def _add_tram_lines_raw(self, tram_lines_raw):
        """Add tram lines from [{coords:[(lon,lat),...], ...}] list."""
        for t in tram_lines_raw:
            coords   = t.get("coords", [])
            line_id  = t.get("line_id", "T?")
            line_name= t.get("line_name", line_id)
            prev_id  = None
            for lon, lat in coords:
                if self._id_to_coords:
                    snap_id, snap_dist = self.snap_to_nearest_node(lat, lon)
                    node_id = snap_id if snap_dist < 400 else self._register_node(lon, lat)
                else:
                    node_id = self._register_node(lon, lat)
                if prev_id is not None and prev_id != node_id:
                    p = self.G.nodes[prev_id]; c = self.G.nodes[node_id]
                    dist = haversine_distance(p["lat"], p["lon"], c["lat"], c["lon"])
                    self.G.add_edge(prev_id, node_id,
                                    length=dist, speed=25,
                                    travel_time=(dist/1000)/25*60,
                                    road_type="tram", name=line_name,
                                    geometry=LineString([(p["lon"],p["lat"]),(c["lon"],c["lat"])]),
                                    is_tram=True, line_id=line_id)
                prev_id = node_id

    def _build_from_geodataframe(self, gdf):
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty: continue
            coords = list(geom.coords)
            if len(coords) < 2: continue
            for i in range(len(coords)-1):
                u, v = coords[i], coords[i+1]
                uid = self._register_node(u[0], u[1])
                vid = self._register_node(v[0], v[1])
                dist = haversine_distance(u[1],u[0],v[1],v[0])
                rtype = row.get("road_type","unclassified")
                speed = float(row.get("speed_kph", row.get("baseline_speed", 25)))
                self.G.add_edge(uid, vid, length=dist, speed=speed,
                                travel_time=(dist/1000)/speed*60,
                                road_type=rtype, name=row.get("road_name",""),
                                geometry=LineString([u,v]), is_tram=False)

    def _add_tram_lines_gdf(self, tram_gdf):
        for _, row in tram_gdf.iterrows():
            geom = row.geometry
            if geom is None: continue
            raw = [{"line_id":row.get("line_id","T?"),
                    "line_name":row.get("line_name","Tram"),
                    "coords":[(c[0],c[1]) for c in geom.coords]}]
            self._add_tram_lines_raw(raw)

    def _register_node(self, lon, lat, precision=5):
        key = (round(lon, precision), round(lat, precision))
        if key not in self._coord_to_id:
            nid = len(self._coord_to_id)
            self._coord_to_id[key] = nid
            self._id_to_coords[nid] = (key[1], key[0])
            self.G.add_node(nid, lon=key[0], lat=key[1])
        return self._coord_to_id[key]

    @property
    def node_coords(self): return self._id_to_coords

    # ── Public API ────────────────────────────────────────────────────────────

    def _build_node_array(self):
        """Cache node positions as numpy arrays for vectorised nearest lookup."""
        if hasattr(self,"_node_ids") and len(self._node_ids)==self.G.number_of_nodes():
            return
        nids = list(self.G.nodes())
        self._node_ids  = nids
        self._node_lats = np.array([self.G.nodes[n]["lat"] for n in nids], dtype=np.float64)
        self._node_lons = np.array([self.G.nodes[n]["lon"] for n in nids], dtype=np.float64)

    def snap_to_nearest_node(self, lat, lon):
        """Vectorised O(N) haversine — ~100x faster than the loop version."""
        self._build_node_array()
        if not self._node_ids:
            return None, float("inf")
        dlat = np.radians(self._node_lats - lat)
        dlon = np.radians(self._node_lons - lon)
        a    = np.sin(dlat/2)**2 + np.cos(np.radians(lat))*np.cos(np.radians(self._node_lats))*np.sin(dlon/2)**2
        dist = 6_371_000 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        idx  = int(np.argmin(dist))
        return self._node_ids[idx], float(dist[idx])

    def snap_to_node(self, lat, lon):
        return self.snap_to_nearest_node(lat, lon)

    def shortest_path(self, start_node, end_node, weight="travel_time"):
        try:
            path = nx.shortest_path(self.G, start_node, end_node, weight=weight)
        except Exception:
            return None
        if len(path) < 2: return None

        all_coords = []; total_length = 0.0; total_time = 0.0
        for i in range(len(path)-1):
            e = self.G[path[i]][path[i+1]]
            total_length += e.get("length", 0)
            total_time   += e.get("travel_time", 0)
            geom = e.get("geometry")
            if geom:
                seg = list(geom.coords)
                all_coords += seg[1:] if all_coords else seg
            else:
                nd  = self.G.nodes[path[i]]; nd2 = self.G.nodes[path[i+1]]
                if not all_coords: all_coords.append((nd["lon"], nd["lat"]))
                all_coords.append((nd2["lon"], nd2["lat"]))

        if len(all_coords) < 2: return None
        return {"path":path, "geometry":LineString(all_coords),
                "distance_km":round(total_length/1000,2),
                "travel_time_min":round(total_time,1),
                "num_segments":len(path)-1}

    def path_to_linestring(self, path):
        coords = [(self.G.nodes[n]["lon"], self.G.nodes[n]["lat"]) for n in path]
        return LineString(coords) if len(coords)>=2 else None

    def path_metrics(self, path):
        if not path or len(path)<2:
            return {"length_km":0,"travel_time_min":0,"segments":0}
        tl=tt=0
        for i in range(len(path)-1):
            e=self.G[path[i]][path[i+1]]
            tl+=e.get("length",0); tt+=e.get("travel_time",0)
        return {"length_km":round(tl/1000,2),"travel_time_min":round(tt,1),"segments":len(path)-1}

    def tram_connectivity_score(self, path):
        if not path: return 0.0
        tns = {u for u,v,d in self.G.edges(data=True) if d.get("is_tram")} | \
              {v for u,v,d in self.G.edges(data=True) if d.get("is_tram")}
        if not tns: return 0.0
        min_d = float("inf")
        for pn in path:
            pd_ = self.G.nodes[pn]
            for tn in tns:
                td = self.G.nodes[tn]
                min_d = min(min_d, haversine_distance(pd_["lat"],pd_["lon"],td["lat"],td["lon"]))
        return round(max(0.0, 1.0-min_d/2000), 3)

    def connectivity_score(self, path): return self.tram_connectivity_score(path)

    def get_tram_nodes(self):
        ns = set()
        for u,v,d in self.G.edges(data=True):
            if d.get("is_tram"): ns.add(u); ns.add(v)
        return list(ns)

    def get_tram_stops(self):
        stops, seen = [], set()
        for u,v,d in self.G.edges(data=True):
            if d.get("is_tram"):
                for nid in [u,v]:
                    if nid not in seen:
                        seen.add(nid); nd=self.G.nodes[nid]
                        stops.append({"node_id":nid,"lat":nd["lat"],"lon":nd["lon"]})
        return stops

    def build_from_geodataframe(self, gdf): self._build_from_geodataframe(gdf)
    def add_tram_lines(self, tram_gdf):     self._add_tram_lines_gdf(tram_gdf)

    def summary(self):
        te = sum(1 for _,_,d in self.G.edges(data=True) if d.get("is_tram"))
        return {"nodes":self.G.number_of_nodes(),"edges":self.G.number_of_edges(),
                "tram_edges":te,"osmnx_mode":self._osmnx_mode,
                "connected_components":nx.number_connected_components(self.G)}

    def get_all_edges_gdf(self):
        rows = [{"u":u,"v":v,"road_type":d.get("road_type",""),
                 "is_tram":d.get("is_tram",False),"geometry":d["geometry"]}
                for u,v,d in self.G.edges(data=True) if d.get("geometry")]
        if not rows:
            return gpd.GeoDataFrame(columns=["u","v","road_type","geometry"],
                                    geometry="geometry").set_crs(_WGS84)
        return gpd.GeoDataFrame(rows, geometry="geometry").set_crs(_WGS84)

    def remove_edge(self, u, v):
        if self.G.has_edge(u,v): self.G.remove_edge(u,v); return True
        return False

    def modify_edge_speed(self, u, v, factor):
        if self.G.has_edge(u,v):
            self.G[u][v]["speed"]*=factor; self.G[u][v]["travel_time"]/=factor; return True
        return False
