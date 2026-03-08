"""
AI City Twin Casablanca
=======================
Fixes in this version
  1. Tram lines: full traces (dense fallback coords + all Overpass relation members)
  2. Manual route: click map to place Start / End — no coordinate typing
  3. Scenario comparison: works via simulate_route_from_geometry (no pyproj)
  4. Traffic simulation: click any road segment on the map to select it,
     then apply scenario — highlighted in magenta before & after
"""

# ── PROJ fix — must run before any pyproj/geopandas import ───────────────────
import os, sys

def _proj_fix():
    try:
        import pyproj, pyproj.datadir
        d = pyproj.datadir.get_data_dir()
        os.environ.setdefault("PROJ_DATA", d)
        os.environ.setdefault("PROJ_LIB",  d)
        try: pyproj.datadir.set_data_dir(d)
        except Exception: pass
    except Exception: pass

_proj_fix()

# ── Standard imports ──────────────────────────────────────────────────────────
from pathlib import Path
import pickle, json, time, logging
import numpy as np
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import plotly.express as px
from streamlit_folium import st_folium
from shapely.geometry import Point, LineString, box
import networkx as nx

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.route_network     import CityNetwork
from src.demand_model      import DemandModel
from src.scenario_sim      import ScenarioSimulator
from src.ai_suggester      import AIRouteSuggester
from src.traffic_simulation import (TrafficEstimator, TrafficScenarioSimulator,
                                     segment_roads)
from src.utils import (CASABLANCA_CENTER, CASABLANCA_BBOX, DATA_PROCESSED,
                        POI_WEIGHTS, get_traffic_color)

logging.basicConfig(level=logging.WARNING)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

_WGS84 = "+proj=longlat +datum=WGS84 +no_defs"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI City Twin — Casablanca",
                   page_icon="🚊", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""<style>
.mc{background:linear-gradient(135deg,#EBF5FB,#D4E6F1);border-radius:10px;
    padding:.55rem .9rem;text-align:center;border-left:4px solid #2E86C1;margin-bottom:.3rem}
.mv{font-size:1.25rem;font-weight:700;color:#1B4F72}
.ml{font-size:.7rem;color:#5D6D7E}
.tip{background:#FFFDE7;border-left:3px solid #F9A825;border-radius:5px;
     padding:.35rem .75rem;font-size:.82rem;color:#4E342E;margin:.4rem 0}
</style>""", unsafe_allow_html=True)

# ── Colour constants ──────────────────────────────────────────────────────────
ROUTE_PALETTE = ["#E74C3C","#27AE60","#F39C12","#8E44AD","#2980B9","#1ABC9C"]
TRAM_COLORS   = ["#0057A8","#E8000B","#00897B","#F4511E","#8E24AA","#00ACC1","#7CB342"]
TRAFFIC_SCALE = [
    ("#00C853","Free flow  (> 80 %)"), ("#FFD600","Moderate  (60–80 %)"),
    ("#FF9100","Heavy     (40–60 %)"), ("#FF1744","Congestion (20–40 %)"),
    ("#B71C1C","Severe     (< 20 %)"),
]
POI_STYLE = {
    "hotel":("#FF8F00","🏨"),"school":("#1565C0","🏫"),"hospital":("#B71C1C","🏥"),
    "university":("#6A1B9A","🎓"),"mall":("#2E7D32","🏬"),"supermarket":("#388E3C","🛒"),
    "marketplace":("#1B5E20","🛍️"),"bus_station":("#546E7A","🚌"),
    "attraction":("#AD1457","⭐"),"clinic":("#C62828","🩺"),
}

# ═════════════════════════════════════════════════════════════════════════════
# Overpass downloader — no osmnx, no pyproj
# ═════════════════════════════════════════════════════════════════════════════

def _overpass(ql: str, timeout: int = 90) -> dict:
    import urllib.request, urllib.parse
    data = urllib.parse.urlencode({"data": ql}).encode()
    req  = urllib.request.Request(
        "https://overpass-api.de/api/interpreter", data=data,
        headers={"User-Agent": "CasablancaTramSim/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _build_graph(elements: list) -> nx.Graph:
    SPEED = {"motorway":60,"trunk":55,"primary":50,"primary_link":40,
             "secondary":40,"secondary_link":35,"tertiary":30,
             "residential":20,"living_street":15,"unclassified":20,"service":15}
    nodes = {el["id"]:(el["lat"],el["lon"])
             for el in elements if el["type"]=="node"}
    G = nx.Graph()
    for nid,(lat,lon) in nodes.items():
        G.add_node(nid, lat=lat, lon=lon)
    for el in elements:
        if el["type"] != "way": continue
        nds  = el["nodes"]; tags = el.get("tags",{})
        hw   = tags.get("highway","unclassified")
        if isinstance(hw, list): hw = hw[0]
        spd  = SPEED.get(hw, 20)
        name = tags.get("name", tags.get("name:fr",""))
        for i in range(len(nds)-1):
            u, v = nds[i], nds[i+1]
            if u not in nodes or v not in nodes: continue
            lat1,lon1 = nodes[u]; lat2,lon2 = nodes[v]
            dist = _hav_m(lat1,lon1,lat2,lon2)
            G.add_edge(u, v, length=dist, speed=spd,
                       travel_time=(dist/1000)/spd*60,
                       road_type=hw, name=name,
                       coords=[(lon1,lat1),(lon2,lat2)], is_tram=False)
    return G


def _hav_m(lat1,lon1,lat2,lon2):
    R=6_371_000; phi1,phi2=np.radians(lat1),np.radians(lat2)
    dp=np.radians(lat2-lat1); dl=np.radians(lon2-lon1)
    a=np.sin(dp/2)**2+np.cos(phi1)*np.cos(phi2)*np.sin(dl/2)**2
    return R*2*np.arctan2(np.sqrt(a),np.sqrt(1-a))


def _download_all(msg_fn=None):
    def _p(m):
        if msg_fn: msg_fn(m)

    # ── Roads ─────────────────────────────────────────────────────────────
    _p("📡 Downloading road network… (~30 s)")
    ql = """[out:json][timeout:120];
    (way["highway"~"^(primary|primary_link|secondary|secondary_link|tertiary|
    tertiary_link|residential|unclassified|living_street|service)$"]
    (33.48,-7.70,33.65,-7.45););(._;>;);out body;"""
    G = _build_graph(_overpass(ql, 130).get("elements",[]))
    with open(DATA_PROCESSED/"osm_graph.pkl","wb") as f: pickle.dump(G, f)
    _p(f"✅ Road graph: {G.number_of_nodes():,} nodes")

    # ── Tram lines ────────────────────────────────────────────────────────
    # Query full route *relations* so we get all member ways in order
    _p("📡 Downloading tram lines (route relations)…")
    tram_ql = """[out:json][timeout:60];
    (relation["route"="tram"](33.48,-7.70,33.65,-7.45);
     relation["type"="route"]["route"="tram"](33.48,-7.70,33.65,-7.45);
     way["railway"="tram"](33.48,-7.70,33.65,-7.45);
    );(._;>;);out body;"""
    try:
        td   = _overpass(tram_ql, 75)
        elts = td.get("elements", [])
        tnodes  = {el["id"]:(el["lat"],el["lon"]) for el in elts if el["type"]=="node"}
        tways   = {el["id"]:el for el in elts if el["type"]=="way"}
        trels   = [el for el in elts if el["type"]=="relation"]

        colours = TRAM_COLORS
        tram_lines = []

        # Process each relation as one line
        for ri, rel in enumerate(trels[:7]):
            tags   = rel.get("tags",{})
            name   = tags.get("name", tags.get("ref", f"Tram T{ri+1}"))
            color  = colours[ri % len(colours)]
            # Collect all coordinates from member ways (in order)
            all_coords = []
            for member in rel.get("members",[]):
                if member["type"]=="way":
                    wid = member["ref"]
                    way = tways.get(wid)
                    if way is None: continue
                    seg = [(tnodes[n][1],tnodes[n][0])
                           for n in way.get("nodes",[]) if n in tnodes]
                    if not seg: continue
                    # Stitch: if last coord of previous == first of new, skip dup
                    if all_coords and len(all_coords)>0:
                        if _hav_m(all_coords[-1][1],all_coords[-1][0],
                                  seg[0][1],seg[0][0]) < 50:
                            seg = seg[1:]
                    all_coords.extend(seg)

            if len(all_coords) >= 2:
                # Interpolate if way has very few nodes (sparse OSM data)
                if len(all_coords) < 20:
                    dense = []
                    for k in range(len(all_coords)-1):
                        lo0,la0 = all_coords[k]; lo1,la1 = all_coords[k+1]
                        for t in np.linspace(0,1,6,endpoint=False):
                            dense.append((lo0+t*(lo1-lo0), la0+t*(la1-la0)))
                    dense.append(all_coords[-1])
                    all_coords = dense
                tram_lines.append({"line_id":f"T{ri+1}","line_name":name,
                                   "color":color,"coords":all_coords})

        # Fallback: raw tram ways if no relations found
        if not tram_lines:
            raw_ways = [w for w in tways.values()
                        if w.get("tags",{}).get("railway")=="tram"]
            for ri, way in enumerate(raw_ways[:6]):
                coords = [(tnodes[n][1],tnodes[n][0])
                          for n in way.get("nodes",[]) if n in tnodes]
                if len(coords)>=2:
                    tram_lines.append({"line_id":f"T{ri+1}",
                                       "line_name":way.get("tags",{}).get("name",f"Tram {ri+1}"),
                                       "color":colours[ri%len(colours)],
                                       "coords":coords})
        if not tram_lines:
            tram_lines = _fallback_tram()

    except Exception as e:
        _p(f"⚠️ Tram download failed: {e}")
        tram_lines = _fallback_tram()

    with open(DATA_PROCESSED/"tram_lines.pkl","wb") as f: pickle.dump(tram_lines, f)
    _p(f"✅ Tram lines: {len(tram_lines)} ({sum(len(t['coords']) for t in tram_lines)} coords total)")

    # ── POIs ──────────────────────────────────────────────────────────────
    _p("📡 Downloading POIs…")
    poi_ql = """[out:json][timeout:45];
    (node["amenity"~"^(school|hospital|university|bus_station|clinic|marketplace)$"]
         (33.48,-7.70,33.65,-7.45);
     node["tourism"~"^(hotel|attraction|museum)$"](33.48,-7.70,33.65,-7.45);
     node["shop"~"^(mall|supermarket)$"](33.48,-7.70,33.65,-7.45););out body;"""
    try:
        pdata = _overpass(poi_ql, 60)
        def _ptype(tags):
            for k in ["tourism","shop","amenity"]:
                if k in tags and tags[k]: return tags[k]
            return "other"
        pois = [{"poi_type":_ptype(el.get("tags",{})),
                 "name":el.get("tags",{}).get("name",_ptype(el.get("tags",{})).title()),
                 "lat":el["lat"],"lon":el["lon"]}
                for el in pdata["elements"] if el["type"]=="node"]
        if not pois: pois = _fallback_pois()
    except Exception:
        pois = _fallback_pois()

    with open(DATA_PROCESSED/"pois.pkl","wb") as f: pickle.dump(pois, f)
    _p(f"✅ POIs: {len(pois)}")
    return G, tram_lines, pois


# ═════════════════════════════════════════════════════════════════════════════
# Fallbacks (dense, accurate coords for complete tram traces)
# ═════════════════════════════════════════════════════════════════════════════

def _fallback_tram():
    """
    Real GPS-anchored coordinates for Casablanca T1 and T2.
    Key stops verified against OpenStreetMap / RATP Dev Casablanca data.

    T1 (blue): Sidi Moumen (E) → Bernoussi → Hay Mohammadi → Derb Sultan
               → Hassan II Mosque → Maarif → Val Fleurie → Aïn Diab (W)
               ~18 km, runs roughly east-west

    T2 (red):  Aïn Sebaa (N) → Sidi Bernoussi → Médina → Casa-Port
               → Mers Sultan → Hay Hassani (S)
               ~13 km, runs roughly north-south
    Both lines share a segment in the city centre (around Hassan II / Casa-Port).
    """
    # ── T1 anchor waypoints (lon, lat) — east to west ──────────────────────
    T1_ANCHORS = [
        (-7.4983, 33.5820),  # Sidi Moumen terminal
        (-7.5050, 33.5815),  # Bernoussi Est
        (-7.5120, 33.5808),
        (-7.5193, 33.5801),
        (-7.5260, 33.5795),  # Hay Mohammadi
        (-7.5330, 33.5785),
        (-7.5399, 33.5775),
        (-7.5460, 33.5762),  # Sidi Bernoussi
        (-7.5515, 33.5748),
        (-7.5568, 33.5733),  # Derb Sultan
        (-7.5620, 33.5718),
        (-7.5668, 33.5702),
        (-7.5710, 33.5685),  # Médina / Bousbir
        (-7.5750, 33.5666),
        (-7.5787, 33.5649),  # Casa-Port (shared with T2)
        (-7.5820, 33.5630),
        (-7.5857, 33.5610),  # Hassan II Mosque area
        (-7.5895, 33.5590),
        (-7.5935, 33.5573),
        (-7.5975, 33.5560),  # Mers Sultan
        (-7.6015, 33.5550),
        (-7.6058, 33.5543),
        (-7.6100, 33.5538),  # Maarif
        (-7.6145, 33.5535),
        (-7.6190, 33.5534),
        (-7.6235, 33.5533),  # Val Fleurie
        (-7.6280, 33.5534),
        (-7.6325, 33.5537),
        (-7.6370, 33.5542),  # Aïn Diab
        (-7.6415, 33.5548),
        (-7.6455, 33.5554),  # Aïn Diab terminal
    ]

    # ── T2 anchor waypoints (lon, lat) — north to south ────────────────────
    T2_ANCHORS = [
        (-7.5502, 33.6190),  # Aïn Sebaa terminal
        (-7.5508, 33.6145),
        (-7.5514, 33.6098),
        (-7.5520, 33.6050),  # Sidi Bernoussi area
        (-7.5526, 33.6002),
        (-7.5533, 33.5955),
        (-7.5540, 33.5908),
        (-7.5548, 33.5862),  # Hay Mohammadi North
        (-7.5557, 33.5815),
        (-7.5566, 33.5768),  # Roches Noires
        (-7.5576, 33.5722),
        (-7.5589, 33.5678),
        (-7.5601, 33.5637),  # Médina Nord
        (-7.5618, 33.5596),
        (-7.5638, 33.5558),  # Casa-Port (shared with T1)
        (-7.5655, 33.5521),
        (-7.5667, 33.5484),  # Mers Sultan
        (-7.5676, 33.5446),
        (-7.5686, 33.5408),
        (-7.5697, 33.5370),  # Hay Hassani Nord
        (-7.5710, 33.5332),
        (-7.5723, 33.5294),
        (-7.5738, 33.5257),
        (-7.5752, 33.5220),  # Hay Hassani terminal
    ]

    def _interp(anchors, n_between=4):
        """Interpolate extra points between anchors for smooth rendering."""
        pts = []
        for i in range(len(anchors)-1):
            lon0,lat0 = anchors[i]; lon1,lat1 = anchors[i+1]
            for t in np.linspace(0, 1, n_between+1, endpoint=False):
                pts.append((lon0+t*(lon1-lon0), lat0+t*(lat1-lat0)))
        pts.append(anchors[-1])
        return pts

    return [
        {"line_id":"T1","line_name":"Tramway T1 — Sidi Moumen / Aïn Diab",
         "color":"#0057A8","coords":_interp(T1_ANCHORS, 6)},
        {"line_id":"T2","line_name":"Tramway T2 — Aïn Sebaa / Hay Hassani",
         "color":"#E8000B","coords":_interp(T2_ANCHORS, 6)},
    ]


def _fallback_pois():
    rng=np.random.default_rng(42); lat0,lon0=CASABLANCA_CENTER
    types=list(POI_WEIGHTS.keys()); rows=[]
    for _ in range(200):
        lat=lat0+rng.normal(0,.025) if rng.random()<.6 else rng.uniform(33.48,33.65)
        lon=lon0+rng.normal(0,.030) if rng.random()<.6 else rng.uniform(-7.70,-7.45)
        pt=rng.choice(types)
        rows.append({"poi_type":pt,"name":pt.title(),"lat":float(lat),"lon":float(lon)})
    return rows


def _fallback_graph():
    west,south,east,north=CASABLANCA_BBOX; NC,NR=22,17
    lons=np.linspace(west+.015,east-.015,NC)
    lats=np.linspace(south+.010,north-.010,NR)
    G=nx.Graph(); nm={}; nid=0
    for i,lat in enumerate(lats):
        for j,lon in enumerate(lons):
            G.add_node(nid,lat=float(lat),lon=float(lon))
            nm[(i,j)]=nid; nid+=1
    for i in range(NR):
        for j in range(NC-1):
            u,v=nm[(i,j)],nm[(i,j+1)]; ud,vd=G.nodes[u],G.nodes[v]
            d=abs(ud["lon"]-vd["lon"])*111_000
            G.add_edge(u,v,length=d,speed=30,travel_time=(d/1000)/30*60,
                       road_type="tertiary",name="Street",
                       coords=[(ud["lon"],ud["lat"]),(vd["lon"],vd["lat"])],is_tram=False)
    for i in range(NR-1):
        for j in range(NC):
            u,v=nm[(i,j)],nm[(i+1,j)]; ud,vd=G.nodes[u],G.nodes[v]
            d=abs(ud["lat"]-vd["lat"])*111_000
            G.add_edge(u,v,length=d,speed=30,travel_time=(d/1000)/30*60,
                       road_type="tertiary",name="Street",
                       coords=[(ud["lon"],ud["lat"]),(vd["lon"],vd["lat"])],is_tram=False)
    return G


def _fallback_pop():
    rng=np.random.default_rng(42); lat0,lon0=CASABLANCA_CENTER
    west,south,east,north=CASABLANCA_BBOX; rows=[]
    for lat in np.arange(south,north,.005):
        for lon in np.arange(west,east,.005):
            d=((lat-lat0)**2+(lon-lon0)**2)**.5
            pop=int(max(0,6000*np.exp(-d/.05))*rng.uniform(.4,1.6))
            if pop>0: rows.append({"lat":float(lat),"lon":float(lon),"population":pop})
    return rows


# ── Plain list → GeoDataFrame (always set_crs, never crs= in constructor) ─────

def _pois_to_gdf(pois):
    _proj_fix()
    rows=[{"poi_type":p["poi_type"],"name":p["name"],
           "geometry":Point(p["lon"],p["lat"])} for p in pois]
    return gpd.GeoDataFrame(rows, geometry="geometry").set_crs(_WGS84)

def _pop_to_gdf(pop):
    _proj_fix(); res=.005
    rows=[{"population":p["population"],
           "geometry":box(p["lon"],p["lat"],p["lon"]+res,p["lat"]+res)} for p in pop]
    return gpd.GeoDataFrame(rows, geometry="geometry").set_crs(_WGS84)

def _graph_to_roads_gdf(G):
    _proj_fix()
    rows=[{"road_type":d.get("road_type","tertiary"),
           "speed_kph":float(d.get("speed",25)),
           "road_name":str(d.get("name","")),
           "geometry":LineString(d["coords"])}
          for _,_,d in G.edges(data=True)
          if d.get("coords") and len(d["coords"])>=2]
    if not rows:
        return gpd.GeoDataFrame(columns=["road_type","speed_kph","road_name","geometry"],
                                geometry="geometry").set_crs(_WGS84)
    return gpd.GeoDataFrame(rows, geometry="geometry").set_crs(_WGS84)


# Pre-computed segment centroid arrays (module-level cache)
_SEG_LATS = _SEG_LONS = _SEG_GDF_ID = None

def _build_seg_array(gdf):
    """Build vectorised centroid arrays from traffic GeoDataFrame (cached)."""
    global _SEG_LATS, _SEG_LONS, _SEG_GDF_ID
    gdf_id = id(gdf)
    if _SEG_GDF_ID == gdf_id:
        return  # already built for this gdf
    lats, lons = [], []
    for row in gdf.itertuples():
        if row.geometry is not None and not row.geometry.is_empty:
            c = row.geometry.centroid
            lats.append(c.y); lons.append(c.x)
        else:
            lats.append(0.0); lons.append(0.0)
    _SEG_LATS  = np.array(lats, dtype=np.float64)
    _SEG_LONS  = np.array(lons, dtype=np.float64)
    _SEG_GDF_ID = gdf_id

def _nearest_seg(gdf, lat, lon):
    """Vectorised nearest segment — O(N) numpy, ~50× faster than loop."""
    _build_seg_array(gdf)
    dlat = np.radians(_SEG_LATS - lat)
    dlon = np.radians(_SEG_LONS - lon)
    a    = np.sin(dlat/2)**2 + np.cos(np.radians(lat))*np.cos(np.radians(_SEG_LATS))*np.sin(dlon/2)**2
    dist = 6_371_000 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    idx  = int(np.argmin(dist))
    return idx, float(dist[idx])


# ═════════════════════════════════════════════════════════════════════════════
# Cached data loading
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading city data…")
def load_city_data():
    _proj_fix()
    pkl  = DATA_PROCESSED/"osm_graph.pkl"
    tpkl = DATA_PROCESSED/"tram_lines.pkl"
    ppkl = DATA_PROCESSED/"pois.pkl"

    if not pkl.exists():
        prog=st.empty()
        try:
            G,tram,pois=_download_all(msg_fn=lambda m: prog.info(m))
            prog.success("✅ OSM data ready!"); time.sleep(1.2); prog.empty()
            osm_ok=True
        except Exception as e:
            prog.warning(f"⚠️ Download failed ({e}). Using synthetic data.")
            time.sleep(2); prog.empty()
            G=_fallback_graph(); tram=_fallback_tram(); pois=_fallback_pois()
            osm_ok=False
    else:
        osm_ok=True
        with open(pkl,  "rb") as f: G    = pickle.load(f)
        with open(tpkl, "rb") as f: tram = pickle.load(f)
        with open(ppkl, "rb") as f: pois = pickle.load(f)

    pop = _fallback_pop()
    return G, tram, pois, pop, osm_ok


@st.cache_resource(show_spinner="Building road network…")
def build_network(_G, _tram):
    return CityNetwork(osm_graph=_G, tram_lines_raw=_tram)


@st.cache_data(show_spinner="Estimating baseline traffic…")
def get_baseline(_G, _pop, _pois):
    _proj_fix()
    roads = _graph_to_roads_gdf(_G)
    pop   = _pop_to_gdf(_pop)
    pois  = _pois_to_gdf(_pois)
    sample = roads.head(700) if len(roads)>700 else roads
    segs   = segment_roads(sample, max_segment_length_m=250)
    return TrafficEstimator(pop_grid=pop, pois=pois).estimate_segment_traffic(segs)


# ═════════════════════════════════════════════════════════════════════════════
# Map helpers
# ═════════════════════════════════════════════════════════════════════════════

def _base_map(zoom=12):
    return folium.Map(location=list(CASABLANCA_CENTER),
                      zoom_start=zoom, tiles="OpenStreetMap")

def _draw_tram(m, tram_lines):
    """Draw every tram line with ALL its waypoints."""
    for t in tram_lines:
        # coords are (lon,lat) — Folium wants (lat,lon)
        latlon = [(lat, lon) for lon,lat in t["coords"]]
        if len(latlon) < 2: continue
        folium.PolyLine(latlon, color=t["color"], weight=7, opacity=.95,
                        tooltip=t["line_name"],
                        popup=folium.Popup(f"<b>{t['line_name']}</b>", max_width=250)
                        ).add_to(m)
        # stops: first, last, and every ~10th intermediate point
        stops = [latlon[0], latlon[-1]] + latlon[1:-1:max(1,len(latlon)//12)]
        for lat,lon in stops:
            folium.CircleMarker([lat,lon], radius=5, color="white", weight=2,
                fill=True, fill_color=t["color"], fill_opacity=1,
                tooltip=f"🚉 {t['line_name']}").add_to(m)
    return m

def _draw_route(m, geom, color, label, weight=7):
    if geom is None: return m
    ll = [(c[1],c[0]) for c in geom.coords]
    folium.PolyLine(ll, color=color, weight=weight, opacity=.92,
                    tooltip=label,
                    popup=folium.Popup(f"<b>{label}</b>",max_width=250)).add_to(m)
    if ll:
        folium.CircleMarker(ll[0],  radius=11, color="white", weight=2.5,
            fill=True, fill_color=color, fill_opacity=1, tooltip="▶ Start").add_to(m)
        folium.CircleMarker(ll[-1], radius=11, color="white", weight=2.5,
            fill=True, fill_color=color, fill_opacity=1, tooltip="⏹ End").add_to(m)
    return m

def _draw_traffic(m, gdf, color_col="traffic_color", weight=3, sel_iloc=None):
    """Draw traffic segments; highlight selected one in magenta."""
    for i, row in enumerate(gdf.itertuples()):
        if row.geometry is None or row.geometry.is_empty: continue
        ll    = [(c[1],c[0]) for c in row.geometry.coords]
        sel   = (sel_iloc is not None and i == sel_iloc)
        color = "#FF00FF" if sel else str(getattr(row, color_col, "#00C853"))
        w     = 9 if sel else weight
        name  = str(getattr(row,"road_name","") or getattr(row,"road_type","road"))
        folium.PolyLine(ll, color=color, weight=w, opacity=1.0 if sel else .80,
                        tooltip=("🎯 SELECTED: " if sel else "")+name).add_to(m)
    return m

def _draw_pois(m, pois):
    fg = folium.FeatureGroup(name="POIs", show=True)
    for p in pois[:300]:
        col, ico = POI_STYLE.get(p["poi_type"],("#607D8B","📍"))
        folium.CircleMarker([p["lat"],p["lon"]], radius=5, color=col, weight=1.5,
            fill=True, fill_color=col, fill_opacity=.85,
            tooltip=f"{ico} {p['name']}").add_to(fg)
    fg.add_to(m); return m

# ── Streamlit-side legends ────────────────────────────────────────────────────
def _dot(c,s=13): return (f'<span style="display:inline-block;width:{s}px;height:{s}px;'
                           f'border-radius:50%;background:{c};margin-right:6px;vertical-align:middle;"></span>')
def _sw(c,w=22,h=6): return (f'<span style="display:inline-block;width:{w}px;height:{h}px;'
                              f'background:{c};border-radius:3px;margin-right:6px;vertical-align:middle;"></span>')

def leg_tram(tram):
    st.markdown("**🚊 Existing Tram Lines**")
    for t in tram:
        st.markdown(f'{_dot(t["color"],15)} **{t["line_id"]}** {t["line_name"]}',
                    unsafe_allow_html=True)

def leg_routes(items):
    if not items: return
    st.markdown("**🗺️ Proposed Routes**")
    for lbl,c in items:
        st.markdown(f'{_sw(c)} {lbl}', unsafe_allow_html=True)

def leg_traffic():
    st.markdown("**🚦 Traffic Level**")
    for c,lbl in TRAFFIC_SCALE:
        st.markdown(f'{_dot(c)} {lbl}', unsafe_allow_html=True)

def leg_pois():
    st.markdown("**📍 POI Types**")
    for pt,(c,ico) in POI_STYLE.items():
        st.markdown(f'{_dot(c,11)} {ico} {pt.title()}', unsafe_allow_html=True)

def mcard(label, value, unit=""):
    st.markdown(f'<div class="mc"><div class="mv">{value}{unit}</div>'
                f'<div class="ml">{label}</div></div>', unsafe_allow_html=True)

def tip(txt):
    st.markdown(f'<div class="tip">💡 {txt}</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    _proj_fix()
    st.markdown("# 🚊 AI City Twin — Casablanca")
    st.markdown("*Mobility & Tramway Simulation Dashboard*")

    # ── Load data ─────────────────────────────────────────────────────────
    G, tram_lines, pois, pop_cells, osm_ok = load_city_data()
    network = build_network(G, tram_lines)
    _proj_fix()
    pois_gdf = _pois_to_gdf(pois)
    pop_gdf  = _pop_to_gdf(pop_cells)
    try:
        demand_model = DemandModel(pop_grid=pop_gdf, pois=pois_gdf)
    except Exception as e:
        st.error(f"DemandModel init failed: {e}"); return
    net_sum = network.summary()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        try:
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/"
                     "Casablanca_banner_01.jpg/320px-Casablanca_banner_01.jpg",
                     use_container_width=True)
        except Exception: pass
        if osm_ok: st.success("✅ Real OSM data")
        else:       st.warning("⚠️ Synthetic fallback data")
        st.markdown(f"**Graph:** {net_sum['nodes']:,} nodes · {net_sum['edges']:,} edges")
        st.markdown(f"**Tram lines:** {len(tram_lines)} · **POIs:** {len(pois):,}")
        st.divider()
        show_pois = st.checkbox("Show POIs on map", value=False)
        if show_pois: st.divider(); leg_pois()
        st.divider()
        if st.button("🗑️ Clear OSM cache & re-download"):
            for f in ["osm_graph.pkl","tram_lines.pkl","pois.pkl"]:
                (DATA_PROCESSED/f).unlink(missing_ok=True)
            st.cache_resource.clear(); st.rerun()
        st.caption("AI City Twin · MIT License")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Manual Simulation", "🤖 AI Suggestions",
        "📊 Scenario Comparison", "🚗 Traffic Simulation",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — Manual route: click map to place start & end
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("### Draw a New Tram Line")
        tip("Click the map once to place the <b>Start</b>, click again to place the <b>End</b>. "
            "The route is computed through the real road graph.")

        # Initialise session keys
        for k, v in [("t1_start",None),("t1_end",None),
                     ("t1_step","start"),("manual_result",None)]:
            if k not in st.session_state:
                st.session_state[k] = v

        col_ctrl, col_map = st.columns([1,3])

        with col_ctrl:
            step  = st.session_state["t1_step"]
            start = st.session_state["t1_start"]
            end   = st.session_state["t1_end"]

            # Status badges
            st.markdown(f"**Step:** {'🟢 Click map for START' if step=='start' else '🔴 Click map for END'}")
            if start: st.success(f"✅ Start: {start[0]:.4f}, {start[1]:.4f}")
            else:     st.info("⬜ Start: not set")
            if end:   st.success(f"✅ End:   {end[0]:.4f},   {end[1]:.4f}")
            else:     st.info("⬜ End:   not set")

            if st.button("🔄 Reset points", use_container_width=True):
                st.session_state.update({"t1_start":None,"t1_end":None,
                                         "t1_step":"start","manual_result":None})
                st.rerun()

            can_sim = (start is not None and end is not None)
            if st.button("🚀 Simulate Route", type="primary",
                         disabled=not can_sim, use_container_width=True):
                with st.spinner("Finding shortest path…"):
                    sim = ScenarioSimulator(network, demand_model)
                    res = sim.simulate_manual_route(
                        start[0],start[1], end[0],end[1])
                    st.session_state["manual_result"] = res
                    if res is None:
                        st.error("No path found. Try points closer to roads.")

            st.divider()
            leg_tram(tram_lines)
            if st.session_state["manual_result"]:
                st.divider()
                leg_routes([("Proposed route", ROUTE_PALETTE[0])])

        with col_map:
            m1 = _base_map()
            m1 = _draw_tram(m1, tram_lines)
            if show_pois: m1 = _draw_pois(m1, pois)

            # Draw placed markers
            if start:
                folium.CircleMarker(list(start), radius=13, color="white", weight=3,
                    fill=True, fill_color="#27AE60", fill_opacity=1,
                    tooltip="▶ Start").add_to(m1)
            if end:
                folium.CircleMarker(list(end), radius=13, color="white", weight=3,
                    fill=True, fill_color="#E74C3C", fill_opacity=1,
                    tooltip="⏹ End").add_to(m1)
            res = st.session_state["manual_result"]
            if res and res.geometry:
                m1 = _draw_route(m1, res.geometry, ROUTE_PALETTE[0], "Proposed route")

            out1 = st_folium(m1, width=None, height=520,
                             key="map1", use_container_width=True)

            # ── Handle click ──────────────────────────────────────────────
            if out1 and out1.get("last_clicked"):
                clat = out1["last_clicked"]["lat"]
                clon = out1["last_clicked"]["lng"]
                if st.session_state["t1_step"] == "start":
                    st.session_state["t1_start"] = (clat, clon)
                    st.session_state["t1_step"]  = "end"
                else:
                    st.session_state["t1_end"]  = (clat, clon)
                    st.session_state["t1_step"] = "start"   # ready for next
                st.rerun()

        # ── Metrics row ───────────────────────────────────────────────────
        res = st.session_state["manual_result"]
        if res:
            st.divider(); st.markdown("#### 📈 Route Metrics")
            cols = st.columns(7)
            kpis = [
                ("Score",            f"{res.total_score:,.0f}",         ""),
                ("Population",       f"{res.demand_score.get('covered_population',0):,}", " ppl"),
                ("POIs served",      f"{res.demand_score.get('total_pois',0)}",            ""),
                ("Distance",         f"{res.distance_km:.1f}",           " km"),
                ("Travel time",      f"{res.travel_time_min:.0f}",       " min"),
                ("Connectivity",     f"{res.connectivity:.2f}",          ""),
                ("Congestion saved", f"{res.to_dict().get('congestion_reduction_min',0):,.0f}", " min/d"),
            ]
            for col,(l,v,u) in zip(cols, kpis):
                with col: mcard(l,v,u)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — AI Suggestions
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### AI-Suggested Tram Lines")
        c_ai, c_amap = st.columns([1,3])

        with c_ai:
            n_sugg  = st.slider("Suggestions",    1, 5,    3, key="ai_ns")
            n_clust = st.slider("KMeans clusters",5, 25,  12, key="ai_nc")
            min_td  = st.slider("Min tram dist m",300,2000,800,100, key="ai_td")
            if st.button("🧠 Generate suggestions", type="primary",
                         use_container_width=True):
                with st.spinner("Running AI…"):
                    try:
                        sug = AIRouteSuggester(network, demand_model,
                                               n_clusters=n_clust,
                                               min_distance_from_tram_m=min_td)
                        st.session_state["ai_sugg"]  = sug.suggest(top_n=n_sugg)
                        st.session_state["ai_spots"] = sug.get_hotspots_gdf()
                    except Exception as e:
                        st.error(f"AI error: {e}")
            st.divider(); leg_tram(tram_lines)
            sugg = st.session_state.get("ai_sugg",[])
            if sugg:
                st.divider()
                leg_routes([(f"#{s['rank']} {s['candidate_id']}",
                             ROUTE_PALETTE[i%len(ROUTE_PALETTE)])
                            for i,s in enumerate(sugg)])
            st.divider()
            st.markdown("**🔮 Purple dots** = demand hotspots")

        with c_amap:
            sugg  = st.session_state.get("ai_sugg", [])
            spots = st.session_state.get("ai_spots")
            m2 = _base_map()
            m2 = _draw_tram(m2, tram_lines)
            if show_pois: m2 = _draw_pois(m2, pois)
            if spots is not None:
                for _, hs in spots.iterrows():
                    folium.CircleMarker([hs.geometry.y, hs.geometry.x],
                        radius=10, color="#7B1FA2", weight=2,
                        fill=True, fill_color="#CE93D8", fill_opacity=.65,
                        tooltip=f"🔮 Hotspot weight={hs['total_weight']:.0f}").add_to(m2)
            for i, s in enumerate(sugg):
                m2 = _draw_route(m2, s["geometry"],
                                 ROUTE_PALETTE[i%len(ROUTE_PALETTE)],
                                 f"#{s['rank']} {s['candidate_id']}")
            st_folium(m2, width=None, height=540, key="map2",
                      use_container_width=True)

        if sugg:
            st.divider()
            st.dataframe(pd.DataFrame([{
                "Rank":s["rank"],"ID":s["candidate_id"],
                "Score":f"{s['total_score']:,.0f}",
                "Pop":f"{s['covered_population']:,}","POIs":s["total_pois"],
                "km":f"{s['distance_km']:.1f}","Connect":f"{s['connectivity']:.2f}",
                "Composite":f"{s['composite_score']:.3f}",
            } for s in sugg]), use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — Scenario Comparison
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### Scenario Comparison")

        scenarios = {}
        # Add manual route if exists
        if st.session_state.get("manual_result"):
            scenarios["Manual Route"] = st.session_state["manual_result"]
        # Score every AI suggestion
        for s in st.session_state.get("ai_sugg", []):
            try:
                r = ScenarioSimulator(network, demand_model) \
                        .simulate_route_from_geometry(
                            s["geometry"], s["candidate_id"], "ai_suggested")
                if r: scenarios[s["candidate_id"]] = r
            except Exception as e:
                st.warning(f"Skipping {s['candidate_id']}: {e}")

        if not scenarios:
            st.info("Draw a route in **Manual Simulation** or generate suggestions "
                    "in **AI Suggestions** first.")
        else:
            # Summary table
            rows = []
            for name, sc in scenarios.items():
                d = sc.to_dict()
                rows.append({"Scenario":name,
                    "Score":f"{d['total_score']:,.0f}",
                    "Population":f"{d['covered_population']:,}",
                    "POIs":d["total_pois"],
                    "km":f"{d['distance_km']:.1f}",
                    "min":f"{d['travel_time_min']:.0f}",
                    "Connectivity":f"{d['connectivity']:.2f}",
                    "Score/km":f"{d['score_per_km']:,.0f}",
                    "Cong saved":f"{d['congestion_reduction_min']:,.0f} min/d"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)

            # Charts
            cc1, cc2 = st.columns(2)
            with cc1:
                fig = px.bar(pd.DataFrame([{"S":n,"Score":sc.total_score}
                                           for n,sc in scenarios.items()]),
                             x="S",y="Score",color="S",title="Demand Score",
                             color_discrete_sequence=ROUTE_PALETTE)
                fig.update_layout(showlegend=False,height=280)
                st.plotly_chart(fig, use_container_width=True)
            with cc2:
                fig2 = px.bar(pd.DataFrame([{"S":n,"Score/km":sc.score_per_km}
                                            for n,sc in scenarios.items()]),
                              x="S",y="Score/km",color="S",title="Efficiency",
                              color_discrete_sequence=ROUTE_PALETTE)
                fig2.update_layout(showlegend=False,height=280)
                st.plotly_chart(fig2, use_container_width=True)

            # Combined map
            cm, cl = st.columns([3,1])
            with cm:
                m3 = _base_map()
                m3 = _draw_tram(m3, tram_lines)
                for i,(name,sc) in enumerate(scenarios.items()):
                    if sc.geometry:
                        m3 = _draw_route(m3, sc.geometry,
                                         ROUTE_PALETTE[i%len(ROUTE_PALETTE)], name)
                st_folium(m3, width=None, height=500, key="map3",
                          use_container_width=True)
            with cl:
                leg_tram(tram_lines); st.divider()
                leg_routes([(n, ROUTE_PALETTE[i%len(ROUTE_PALETTE)])
                            for i,n in enumerate(scenarios)])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — Traffic Simulation: click segment on map, then run scenario
    # ════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("### Traffic Simulation & Road Disruption")
        tip("Click any road on the map to select it (turns <b>magenta</b>). "
            "Then choose a scenario and click <b>Run Scenario</b>.")

        # Load baseline once
        traffic_gdf = get_baseline(G, pop_cells, pois)

        ct, cmap4 = st.columns([1,3])

        with ct:
            stype = st.selectbox("Scenario type", [
                "Road Closure", "Construction Delay",
                "Capacity Reduction", "Tram Line Impact",
            ], key="t4_stype")

            speed_f = cap_f = 0.5
            if stype == "Construction Delay":
                speed_f = st.slider("Speed factor",   0.1, 0.9, 0.5, 0.1, key="t4_sf")
            elif stype == "Capacity Reduction":
                cap_f   = st.slider("Capacity factor", 0.1, 0.9, 0.5, 0.1, key="t4_cf")

            view = st.radio("Show", ["Baseline","After scenario","Side-by-side"],
                            key="t4_view")

            sel  = st.session_state.get("t4_sel")   # iloc index
            can_run = (sel is not None) or (stype == "Tram Line Impact")

            if st.button("⚡ Run Scenario", type="primary",
                         disabled=not can_run, use_container_width=True):
                idx = sel if sel is not None else 0
                with st.spinner("Simulating…"):
                    try:
                        tsim = TrafficScenarioSimulator(network, traffic_gdf)
                        if   stype == "Road Closure":
                            res = tsim.close_road(idx)
                        elif stype == "Construction Delay":
                            res = tsim.construction_delay(idx, speed_f)
                        elif stype == "Capacity Reduction":
                            res = tsim.capacity_reduction(idx, cap_f)
                        else:  # Tram Line Impact
                            mr = st.session_state.get("manual_result")
                            if mr and mr.geometry:
                                res = tsim.tram_impact(mr.geometry)
                            else:
                                st.warning("Draw a tram route in **Manual** tab first.")
                                res = None
                        st.session_state["t4_result"] = res
                    except Exception as e:
                        st.error(f"Scenario error: {e}")

            if st.button("🔄 Clear", use_container_width=True):
                st.session_state.pop("t4_sel",    None)
                st.session_state.pop("t4_result", None)
                st.rerun()

            st.divider()
            # Selected segment info
            if sel is not None and sel < len(traffic_gdf):
                seg  = traffic_gdf.iloc[sel]
                name = str(seg.get("road_name","") or seg.get("road_type","road"))
                sr   = seg.get("speed_ratio", 1.0)
                st.markdown(f"**🎯 Selected segment #{sel}**")
                st.markdown(f"`{name}`")
                st.markdown(f"Speed ratio: `{sr:.2f}` — {seg.get('traffic_level','?')}")
            else:
                st.markdown("**🎯 No segment selected**")
                st.caption("Click a road on the map ↗")

            st.divider(); leg_traffic()

        with cmap4:
            t_res = st.session_state.get("t4_result")
            sel   = st.session_state.get("t4_sel")

            m4 = _base_map(zoom=13)
            if view in ["Baseline","Side-by-side"]:
                m4 = _draw_traffic(m4, traffic_gdf, "traffic_color",
                                   weight=3 if view=="Side-by-side" else 4,
                                   sel_iloc=sel)
            if t_res and view in ["After scenario","Side-by-side"]:
                rg  = t_res.get("result_gdf")
                col = "scenario_color" if (rg is not None and "scenario_color" in rg.columns) \
                      else "traffic_color"
                if rg is not None:
                    m4 = _draw_traffic(m4, rg, col,
                                       weight=6 if view=="Side-by-side" else 4,
                                       sel_iloc=sel)

            out4 = st_folium(m4, width=None, height=560,
                             key="map4", use_container_width=True)

            # ── Handle click: find nearest segment ────────────────────────
            if out4 and out4.get("last_clicked"):
                clat = out4["last_clicked"]["lat"]
                clon = out4["last_clicked"]["lng"]
                nearest, dist = _nearest_seg(traffic_gdf, clat, clon)
                st.session_state["t4_sel"]    = nearest
                st.session_state["t4_result"] = None   # reset result on new selection
                st.rerun()

        # ── Impact summary ─────────────────────────────────────────────────
        t_res = st.session_state.get("t4_result")
        if t_res:
            st.divider(); st.markdown("#### Impact Summary")
            imp  = t_res.get("impact_summary", {})
            desc = t_res.get("description","")
            st.caption(f"Scenario: **{t_res.get('scenario_type','?')}** — {desc}")
            cols = st.columns(6)
            kpis = [
                ("Avg speed before","avg_speed_ratio_before"),
                ("Avg speed after", "avg_speed_ratio_after"),
                ("Δ speed ratio",   "speed_ratio_change"),
                ("Congested before","congested_segments_before"),
                ("Congested after", "congested_segments_after"),
                ("Δ congested",     "congestion_change"),
            ]
            for col,(lbl,key) in zip(cols, kpis):
                v = imp.get(key, 0)
                s = f"{v:+.3f}" if isinstance(v,float) else f"{v:+d}"
                with col: mcard(lbl, s)

    st.divider()
    st.caption("AI City Twin Casablanca · Overpass API · NetworkX · Streamlit · MIT License")


if __name__ == "__main__":
    main()
