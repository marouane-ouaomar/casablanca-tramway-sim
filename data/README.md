# Data Sources

## Free Datasets Used

| Dataset | Source | License | Link |
|---------|--------|---------|------|
| Roads & Tram Lines | OpenStreetMap (via OSMnx) | ODbL | https://www.openstreetmap.org |
| Points of Interest | OpenStreetMap | ODbL | https://www.openstreetmap.org |
| Population Density | WorldPop | CC BY 4.0 | https://www.worldpop.org |
| Admin Boundaries | OpenStreetMap | ODbL | https://www.openstreetmap.org |

## Directory Structure

- `raw/` — Raw downloaded datasets (not committed to git if large)
- `processed/` — Cleaned GeoJSON/CSV files ready for the app

## Generating Data

Run the data preparation pipeline:

```bash
# Fetch and process all data
python src/data_prep.py

# Or generate sample data for development
python src/data_prep.py --sample
```

## Notes

- All data is freely available and open-licensed
- OSM data is fetched via OSMnx (no API key needed)
- WorldPop rasters should be downloaded manually from worldpop.org
- Sample/synthetic data is provided for development without network access
