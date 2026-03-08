# 🚊 AI City Twin Casablanca — Mobility & Tramway Simulation

An AI-powered digital twin for Casablanca's urban mobility network. Simulate new tram lines, estimate passenger demand, compare route scenarios, and model traffic disruptions — all with free, open-source tools.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

## Features

- **Manual Tram Line Simulation** — Draw custom routes on the map, snapped to real roads
- **AI-Suggested Routes** — ML-powered route suggestions optimizing coverage & connectivity
- **Scenario Comparison** — Side-by-side metrics for multiple route proposals
- **Traffic Simulation** — Model road closures, construction, and capacity changes
- **Demand Estimation** — Score routes by population coverage and nearby POIs
- **Interactive Dashboard** — Professional Streamlit UI with Folium maps

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USER/casablanca-tramway-sim.git
cd casablanca-tramway-sim

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Generate sample data (no network needed)
python src/data_prep.py --sample

# Launch the dashboard
streamlit run webapp/streamlit_app.py
```

## Tech Stack

| Component | Tool |
|-----------|------|
| Dashboard | Streamlit Community Cloud |
| Maps | OpenStreetMap + Folium |
| Road Network | OSMnx + NetworkX |
| Geospatial | GeoPandas + Shapely |
| AI/ML | scikit-learn + OR-Tools |
| Visualization | Plotly + PyDeck |

## Data Sources

All data is free and open-licensed:
- **Roads & Tram Lines**: OpenStreetMap via OSMnx
- **Points of Interest**: OpenStreetMap amenity/tourism tags
- **Population Density**: WorldPop (worldpop.org)

## Project Structure

```
casablanca-tramway-sim/
├─ data/              # Raw & processed datasets
├─ notebooks/         # Exploration & analysis notebooks
├─ src/               # Core Python modules
├─ scripts/           # Data fetching & processing scripts
├─ webapp/            # Streamlit dashboard
├─ tests/             # Unit tests
└─ requirements.txt   # Python dependencies
```

## Deployment

### Streamlit Community Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select repo, branch `main`, file `webapp/streamlit_app.py`
4. Deploy

### Hugging Face Spaces
1. Create a new Space with Streamlit SDK
2. Copy `webapp/` and `src/` contents
3. Add `requirements.txt`

## License

MIT License — see [LICENSE](LICENSE)
