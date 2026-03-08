# 🚊 AI City Twin Casablanca — Local Run Guide

---

## 🐛 Bugs Fixed Before You Run

| # | File | Bug | Fix Applied |
|---|------|-----|-------------|
| 1 | `src/utils.py` | Missing `CASABLANCA_CENTER`, `CASABLANCA_BBOX`, `DEFAULT_CRS`, `PROJECTED_CRS`, `TRAM_SPEED_KMH`, `BUFFER_DISTANCE_M`, `get_traffic_color`, `get_traffic_color_rgba`, `to_projected`, `to_geographic` — imported by every module but never defined | All constants and functions added |
| 2 | `src/data_prep.py` | Bare `from utils import …` fails when imported as `src.data_prep` from the webapp; five public function names expected by the webapp and tests (`fetch_roads_offline`, `generate_synthetic_tram_lines`, `generate_synthetic_pois`, `generate_population_grid`, `prepare_all_data`) didn't exist | Import guarded with try/except; public aliases added |
| 3 | `src/data_prep.py` | Road rows were missing `speed_kph` and `importance` columns required by tests | Both columns added to all three road generation loops |
| 4 | `src/demand_model.py` | Bare `from utils import …` — same breakage as #2 | Guarded import added |
| 5 | `src/route_network.py` | Bare `from utils import …` | Guarded import added |
| 6 | `src/route_network.py` | `_node_coords` mapped `(lon,lat) → id`; tests and `ai_suggester` expected `node_id → (lat, lon)` via public `node_coords` property | New `_id_to_coords` dict + `node_coords` property added |
| 7 | `src/route_network.py` | `shortest_path()` returned a bare list; every caller expected a dict with `geometry`, `distance_km`, `travel_time_min`, `path` | Method rebuilt to return the dict |
| 8 | `src/route_network.py` | `snap_to_node()`, `connectivity_score()`, `get_tram_nodes()`, `build_from_geodataframe()`, `add_tram_lines()`, `summary()`, `get_all_edges_gdf()` — all called from the webapp, tests, or AI engine but none existed | All seven methods added as public aliases or new implementations |

---

## 📋 Prerequisites

- Python 3.10 or 3.11 (recommended)
- git
- 4 GB RAM minimum (spatial joins on sample data)

---

## ⚡ Step-by-Step Local Setup

### Step 1 — Clone or copy the repo

```bash
git clone https://github.com/YOUR_USERNAME/casablanca-tramway-sim.git
cd casablanca-tramway-sim
```

If you already have the folder, just `cd` into it.

---

### Step 2 — Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it:

- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```
- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

You should see `(.venv)` in your prompt.

---

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⏱ First install takes 3-5 minutes (geopandas + GDAL wheels).
> On Windows, if `geopandas` fails, install via conda instead:
> ```bash
> conda install -c conda-forge geopandas osmnx networkx
> pip install streamlit streamlit-folium plotly pydeck scikit-learn ortools
> ```

---

### Step 4 — Generate sample data

The app will auto-generate data on first launch, but running this first
gives you faster startup:

```bash
python src/data_prep.py --sample
```

Expected output:
```
==============================
  GENERATING SAMPLE DATA FOR CASABLANCA
==============================
  ✓ Saved roads.geojson (464 segments)
  ✓ Saved tram_lines.geojson (2 lines)
  ✓ Saved pois.geojson (150 points)
  ✓ Saved pop_grid.geojson (838 cells)
```

All files land in `data/processed/`.

---

### Step 5 — Run the Streamlit dashboard

```bash
streamlit run webapp/streamlit_app.py
```

Your browser opens automatically at **http://localhost:8501**

> If port 8501 is taken: `streamlit run webapp/streamlit_app.py --server.port 8502`

---

### Step 6 — Run the test suite

```bash
pytest tests/ -v
```

All 25+ unit tests should pass.  Run a single file:
```bash
pytest tests/test_network.py -v
pytest tests/test_data_prep.py -v
pytest tests/test_demand_model.py -v
pytest tests/test_scenario.py -v
```

---

## 🗺️ Dashboard Walkthrough

Once the app is open, you'll see four tabs:

### Tab 1 — Manual Simulation
1. Enter **Start** and **End** coordinates (default values are pre-filled with central Casablanca points).
2. Click **🚀 Simulate Route**.
3. The route snaps to the road network. Right panel shows:
   - Passenger Score, Coverage (people), POIs served
   - Distance (km), Travel time (min), Connectivity, Congestion saved (min/day).

### Tab 2 — AI Suggestions
1. Set number of suggestions (1–5) and cluster count.
2. Click **🧠 Generate AI Suggestions**.
3. The AI clusters demand hotspots (KMeans on POIs + population), filters underserved areas, generates road-network routes, ranks them by a composite score.

### Tab 3 — Scenario Comparison
- Auto-populated once you have run Manual and/or AI routes.
- Side-by-side table + bar charts + combined map.

### Tab 4 — Traffic Simulation
1. Baseline traffic is computed and displayed (TomTom color scale: green → dark red).
2. Pick a **Scenario Type**: Road Closure, Construction Delay, Capacity Reduction, or Tram Line Impact.
3. Select a segment by index.
4. Click **⚡ Run Traffic Scenario** — the map updates and impact metrics appear below.

---

## 🌐 Using Real OSM Data (Optional)

> Requires internet access. Skip for the demo.

```bash
python src/data_prep.py --osm
```

This fetches real Casablanca roads, tram lines, and POIs from OpenStreetMap via `osmnx`.
Population data must be downloaded separately from [worldpop.org](https://worldpop.org).

---

## 📁 Repo Structure

```
casablanca-tramway-sim/
├── data/
│   ├── raw/            # Raw datasets (OSM cache, WorldPop)
│   ├── processed/      # Generated GeoJSONs (auto-created)
│   └── README.md
├── notebooks/          # Jupyter exploration notebooks
├── src/
│   ├── utils.py            # Shared constants & helpers
│   ├── data_prep.py        # Data generation / OSM fetching
│   ├── route_network.py    # Graph network (NetworkX)
│   ├── demand_model.py     # Passenger demand scoring
│   ├── scenario_sim.py     # Scenario runner & comparison
│   ├── ai_suggester.py     # AI route suggestion engine
│   └── traffic_simulation.py  # Traffic disruption simulator
├── webapp/
│   └── streamlit_app.py    # Main Streamlit dashboard
├── tests/                  # pytest unit tests
├── scripts/                # OSM fetch & graph build scripts
├── requirements.txt
├── Dockerfile
└── RUNBOOK.md              # ← You are here
```

---

## 🚀 Deployment (Free Hosting)

### Streamlit Community Cloud
1. Push repo to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect repo → set **Main file path** to `webapp/streamlit_app.py`.
4. Deploy — done. URL is public, auto-rebuilds on push.

### Hugging Face Spaces
1. Create a new Space → type **Streamlit**.
2. Push the same repo, set `webapp/streamlit_app.py` as entry.
3. Free, always-on (with usage limits).

---

## ❓ Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'geopandas'` | Re-activate venv: `source .venv/bin/activate` |
| `RuntimeError: no default CRS` | Run `python src/data_prep.py --sample` first |
| `streamlit: command not found` | `pip install streamlit` inside the venv |
| App loads but map is blank | Zoom out — OSM tiles need internet; map still works offline with markers |
| Slow first load (60s+) | Normal — building the network graph; subsequent loads use Streamlit cache |
| `pytest` import errors | Make sure you're running from the project root: `cd casablanca-tramway-sim && pytest tests/` |
