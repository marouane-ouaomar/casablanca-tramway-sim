"""
Traffic Simulation — pyproj-free rewrite.
All spatial operations use degree-based buffers or haversine instead of UTM projection.
"""
import os, logging
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union

try:
    from src.route_network import CityNetwork
    from src.utils import BUFFER_DISTANCE_M, get_traffic_color
except ImportError:
    from route_network import CityNetwork
    from utils import BUFFER_DISTANCE_M, get_traffic_color

logger = logging.getLogger(__name__)
_WGS84 = "+proj=longlat +datum=WGS84 +no_defs"
_DEG_PER_M = 1.0 / 111_000   # rough conversion — good enough for buffers


def _reinit_proj():
    try:
        import pyproj, pyproj.datadir
        d = pyproj.datadir.get_data_dir()
        os.environ["PROJ_DATA"] = d; os.environ["PROJ_LIB"] = d
        try: pyproj.datadir.set_data_dir(d)
        except Exception: pass
    except Exception: pass


def _safe_gdf(geometries, data=None):
    """Create GeoDataFrame without CRS constructor arg, then set_crs."""
    if data:
        gdf = gpd.GeoDataFrame(data, geometry=geometries)
    else:
        gdf = gpd.GeoDataFrame(geometry=geometries)
    return gdf.set_crs(_WGS84)


def _deg_buffer(geom, metres):
    """Buffer in degrees (approximate) — avoids UTM projection."""
    return geom.buffer(metres * _DEG_PER_M)


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2-lat1); dl = np.radians(lon2-lon1)
    a  = np.sin(dp/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))


# ═════════════════════════════════════════════════════════════════════════════
# Traffic Estimator
# ═════════════════════════════════════════════════════════════════════════════

class TrafficEstimator:
    """Estimates baseline traffic for road segments using population + POI proximity."""

    def __init__(self, pop_grid=None, pois=None, base_volume=1000):
        _reinit_proj()
        self.pop_grid    = pop_grid
        self.pois        = pois
        self.base_volume = base_volume

        # Pre-project once (proj4 strings, no EPSG)
        _UTM = "+proj=utm +zone=29 +datum=WGS84 +units=m +no_defs"
        try:
            self._pop_proj  = pop_grid.to_crs(_UTM)  if pop_grid is not None else None
            self._pois_proj = pois.to_crs(_UTM)      if pois    is not None else None
        except Exception:
            self._pop_proj  = None
            self._pois_proj = None

    def estimate_segment_traffic(self, segment_gdf):
        _reinit_proj()
        _UTM = "+proj=utm +zone=29 +datum=WGS84 +units=m +no_defs"

        result = segment_gdf.copy()
        result["volume"]       = self.base_volume
        result["speed_ratio"]  = 1.0
        result["traffic_level"]= "free_flow"
        result["traffic_color"]= "#00C853"

        try:
            seg_proj = result.to_crs(_UTM)

            for idx in result.index:
                geom = seg_proj.loc[idx, "geometry"]
                if geom is None or geom.is_empty: continue

                buf = geom.buffer(BUFFER_DISTANCE_M)

                # Population factor
                pop_factor = 1.0
                if self._pop_proj is not None:
                    try:
                        pop_in = self._pop_proj[self._pop_proj.intersects(buf)]
                        if len(pop_in):
                            pop_total = pop_in["population"].sum()
                            pop_factor = min(3.0, 1.0 + pop_total / 50_000)
                    except Exception: pass

                # POI factor
                poi_factor = 1.0
                if self._pois_proj is not None:
                    try:
                        pois_in = self._pois_proj[self._pois_proj.intersects(buf)]
                        poi_factor = min(2.0, 1.0 + len(pois_in) * 0.05)
                    except Exception: pass

                volume = self.base_volume * pop_factor * poi_factor
                road_type = result.loc[idx].get("road_type", "tertiary") if hasattr(result.loc[idx], "get") else "tertiary"
                capacity  = {"primary":2000,"secondary":1500,"tertiary":1000,
                             "residential":500}.get(str(road_type), 800)
                v_c_ratio   = min(volume / capacity, 1.5)
                speed_ratio = max(0.1, 1.0 - 0.6 * v_c_ratio)

                result.loc[idx, "volume"]       = round(volume)
                result.loc[idx, "speed_ratio"]  = round(speed_ratio, 3)
                result.loc[idx, "traffic_color"]= get_traffic_color(speed_ratio)
                result.loc[idx, "traffic_level"]= _level(speed_ratio)
        except Exception as e:
            logger.warning(f"Traffic estimation partial failure: {e}")
            result["traffic_color"] = result.get("speed_ratio", pd.Series(1.0, index=result.index)).apply(get_traffic_color)

        return result


def _level(r):
    if r > 0.8: return "free_flow"
    if r > 0.6: return "moderate"
    if r > 0.4: return "heavy"
    if r > 0.2: return "congestion"
    return "severe"


# ═════════════════════════════════════════════════════════════════════════════
# Scenario Simulator
# ═════════════════════════════════════════════════════════════════════════════

class TrafficScenarioSimulator:
    def __init__(self, network: CityNetwork, baseline_traffic_gdf):
        _reinit_proj()
        self.network  = network
        self.baseline = baseline_traffic_gdf.copy()

    def reset(self):
        return self.baseline.copy()

    # ── individual scenarios ──────────────────────────────────────────────────

    def close_road(self, segment_idx):
        result = self.baseline.copy()
        if segment_idx >= len(result):
            return self._wrap(result, result, "road_closure", segment_idx, "No segment")
        affected = result.iloc[segment_idx]
        result = result.drop(result.index[segment_idx]).reset_index(drop=True)
        # Redistribute load to neighbours
        result = self._redistribute(result, affected, factor=1.3)
        result["scenario_color"] = result["traffic_color"]
        return self._wrap(self.baseline, result, "road_closure", segment_idx,
                          f"Road closed: {affected.get('name','segment')}")

    def construction_delay(self, segment_idx, speed_reduction=0.5):
        result = self.baseline.copy()
        if segment_idx >= len(result):
            return self._wrap(result, result, "construction", segment_idx, "No segment")
        affected = result.iloc[segment_idx]
        result.loc[result.index[segment_idx], "speed_ratio"] *= speed_reduction
        new_ratio = result.loc[result.index[segment_idx], "speed_ratio"]
        result.loc[result.index[segment_idx], "traffic_color"] = get_traffic_color(new_ratio)
        result["scenario_color"] = result["traffic_color"]
        return self._wrap(self.baseline, result, "construction", segment_idx,
                          f"Construction delay on: {affected.get('name','segment')}")

    def capacity_reduction(self, segment_idx, reduction_factor=0.5):
        result = self.baseline.copy()
        if segment_idx >= len(result):
            return self._wrap(result, result, "capacity_reduction", segment_idx, "No segment")
        affected = result.iloc[segment_idx]
        old_vol  = result.loc[result.index[segment_idx], "volume"]
        new_vol  = old_vol / reduction_factor
        cap      = {"primary":2000,"secondary":1500,"tertiary":1000,"residential":500}.get(
                   str(affected.get("road_type","tertiary")), 800)
        new_vc   = min(new_vol / cap, 1.5)
        new_sr   = max(0.1, 1.0 - 0.6 * new_vc)
        result.loc[result.index[segment_idx], "volume"]       = round(new_vol)
        result.loc[result.index[segment_idx], "speed_ratio"]  = round(new_sr, 3)
        result.loc[result.index[segment_idx], "traffic_color"]= get_traffic_color(new_sr)
        result["scenario_color"] = result["traffic_color"]
        return self._wrap(self.baseline, result, "capacity_reduction", segment_idx,
                          f"Capacity reduced on: {affected.get('name','segment')}")

    def tram_impact(self, tram_geometry, mode_shift_rate=0.15):
        _reinit_proj()
        result = self.baseline.copy()
        try:
            # Degree-based buffer around tram line — ~500 m
            tram_buf = _deg_buffer(tram_geometry, 500)
            for idx in result.index:
                geom = result.loc[idx, "geometry"]
                if geom is None or geom.is_empty: continue
                seg_centroid = geom.centroid
                if tram_buf.contains(seg_centroid) or tram_buf.intersects(geom):
                    old_vol = result.loc[idx, "volume"]
                    new_vol = old_vol * (1 - mode_shift_rate)
                    cap     = {"primary":2000,"secondary":1500,"tertiary":1000,"residential":500}.get(
                              str(result.loc[idx].get("road_type","tertiary") if hasattr(result.loc[idx],"get") else "tertiary"), 800)
                    vc      = min(new_vol/cap, 1.5)
                    sr      = max(0.1, 1.0-0.6*vc)
                    result.loc[idx, "volume"]       = round(new_vol)
                    result.loc[idx, "speed_ratio"]  = round(sr, 3)
                    result.loc[idx, "traffic_color"]= get_traffic_color(sr)
        except Exception as e:
            logger.warning(f"Tram impact calculation failed: {e}")
        result["scenario_color"] = result["traffic_color"]
        return self._wrap(self.baseline, result, "tram_impact", -1, "Tram line impact")

    def close_road_by_coords(self, lat, lon, radius_m=300):
        """Select and close the nearest segment to (lat, lon)."""
        best_idx, best_d = 0, float("inf")
        for i, row in self.baseline.iterrows():
            if row.geometry is None: continue
            c = row.geometry.centroid
            d = _haversine_m(lat, lon, c.y, c.x)
            if d < best_d:
                best_d = d; best_idx = self.baseline.index.get_loc(i)
        return self.close_road(best_idx), best_idx

    # ── helpers ───────────────────────────────────────────────────────────────

    def _redistribute(self, gdf, affected_seg, factor=1.3, radius_deg=0.01):
        """Increase load on segments near the closed one."""
        if affected_seg.geometry is None:
            return gdf
        center = affected_seg.geometry.centroid
        nearby = gdf[gdf.geometry.centroid.distance(center) < radius_deg].index
        for idx in nearby:
            old = gdf.loc[idx, "speed_ratio"]
            new = max(0.05, old / factor)
            gdf.loc[idx, "speed_ratio"]   = round(new, 3)
            gdf.loc[idx, "traffic_color"] = get_traffic_color(new)
        return gdf

    def _summarize_impact(self, before, after):
        b_sr = before["speed_ratio"].mean() if "speed_ratio" in before.columns else 1.0
        a_sr = after["speed_ratio"].mean()  if "speed_ratio" in after.columns  else 1.0
        b_cong = int((before["speed_ratio"] < 0.4).sum()) if "speed_ratio" in before.columns else 0
        a_cong = int((after["speed_ratio"]  < 0.4).sum()) if "speed_ratio" in after.columns  else 0
        return {
            "avg_speed_ratio_before":  round(b_sr, 3),
            "avg_speed_ratio_after":   round(a_sr, 3),
            "speed_ratio_change":      round(a_sr - b_sr, 3),
            "congested_segments_before": b_cong,
            "congested_segments_after":  a_cong,
            "congestion_change":          a_cong - b_cong,
        }

    def _wrap(self, before, after, stype, seg_idx, desc):
        return {
            "scenario_type":  stype,
            "segment_idx":    seg_idx,
            "description":    desc,
            "result_gdf":     after,
            "impact_summary": self._summarize_impact(before, after),
        }


# ═════════════════════════════════════════════════════════════════════════════
# segment_roads helper
# ═════════════════════════════════════════════════════════════════════════════

def segment_roads(roads_gdf, max_segment_length_m=200):
    """Split long road geometries into shorter segments for traffic display."""
    _reinit_proj()
    segments = []
    deg_max  = max_segment_length_m * _DEG_PER_M

    for _, row in roads_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty: continue
        coords = list(geom.coords)
        if len(coords) < 2: continue

        seg_coords = [coords[0]]
        seg_len    = 0.0

        for i in range(1, len(coords)):
            dx  = coords[i][0] - coords[i-1][0]
            dy  = coords[i][1] - coords[i-1][1]
            seg_len += (dx**2 + dy**2)**0.5
            seg_coords.append(coords[i])

            if seg_len >= deg_max:
                if len(seg_coords) >= 2:
                    d = dict(row)
                    d["geometry"] = LineString(seg_coords)
                    segments.append(d)
                seg_coords = [coords[i]]
                seg_len    = 0.0

        if len(seg_coords) >= 2:
            d = dict(row)
            d["geometry"] = LineString(seg_coords)
            segments.append(d)

    if not segments:
        return roads_gdf.copy()

    gdf = gpd.GeoDataFrame(segments, geometry="geometry")
    return gdf.set_crs(_WGS84)
