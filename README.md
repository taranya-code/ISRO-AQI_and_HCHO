# 🛰️ Surface AQI Prediction & HCHO Hotspot Detection

**ISRO Bharatiya Antariksh Hackathon 2026 — Problem Statement 3**
Team: Cambridge Institute of Technology, Bengaluru

Predicts surface Air Quality Index (AQI) across India and detects HCHO
(formaldehyde) hotspots linked to crop-residue burning and forest fires,
using real satellite and ground-truth data.

## Data sources

| Source | Variables | Provider |
|---|---|---|
| TROPOMI | HCHO, NO2, SO2, CO, O3, AOD | DLR Geoservice (L3) |
| FIRMS VIIRS | Active fire counts | NASA |
| ERA5 | 2m temperature, wind, boundary-layer height, precipitation | Copernicus CDS |
| CPCB | Ground-truth AQI, 10 cities, 2021–2022 | CPCB CCR portal |

## Pipeline

```
TROPOMI (HCHO/NO2/SO2/CO/O3/AOD) ──┐
FIRMS VIIRS Fire Counts             ├──→ Real Satellite Data → Feature Grid
ERA5 (T2m/Wind/BLH/Precip)         ┘              │
CPCB Ground AQI (10 cities) ───────────────────────┘
                                          ▼
                             CNN → Attention-BiLSTM
                             (Gaussian NLL Uncertainty)
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
          Objective 1                            Objective 2
  Surface AQI Maps + Uncertainty       DBSCAN Hotspots + HCHO GIF
  Folium Interactive Map               Seasonal Stratification
  Spatial Cross-Validation             Fire-HCHO Correlation
```

Model: a Spatial CNN feature extractor feeding a BiLSTM with temporal
attention over a monthly sequence, trained with a Gaussian NLL loss so
every prediction carries a calibrated uncertainty estimate. Grid: India,
0.1° resolution. Hotspots are extracted from the seasonal HCHO field with
DBSCAN (vs. a simple mean+1.5σ threshold baseline).

## Repo structure

```
notebooks/
  isro-bah.ipynb        # full pipeline: data loading → model → all outputs
outputs/                # generated artifacts land here when the notebook is run
docs/                   # supporting write-ups / diagrams (optional)
requirements.txt
```

## Outputs produced by the notebook

| File | Objective | Description |
|---|---|---|
| `training_curve.png` | Obj 1 | Loss convergence |
| `evaluation.png` | Obj 1 | Scatter + residuals + uncertainty |
| `spatial_cv.csv` | Obj 1 | Spatial cross-validation results |
| `seasonal_aqi.png` | Obj 1 | 4-season AQI maps on an India basemap |
| `interactive_aqi_map.html` | Obj 1 | Folium interactive map |
| `seasonal_stratification.png` | Obj 2 | HCHO / Fire seasonal maps, India basemap |
| `dbscan_comparison.png` | Obj 2 | DBSCAN vs. threshold hotspot detection |
| `correlation_transport.png` | Obj 2 | Fire–HCHO correlation, BLH regression, wind transport |
| `hcho_animation.gif` | Obj 2 | Animated HCHO evolution, India basemap |
| `timeseries_attention.png` | Both | Regional HCHO time series + attention weights |
| `best_model.pt` | Model | Saved CNN + Attention-BiLSTM weights |
| `real_data_cache.npy` | Data | Cached real satellite data grids |

All spatial maps render on a proper India basemap (coastline, country and
state borders via Cartopy where available, with an India-outline fallback
when Cartopy/Natural-Earth data can't be downloaded).

## Running it

This notebook was built for a Kaggle GPU notebook environment with two
datasets attached:

- `real-satellite-data` (TROPOMI / ERA5 / FIRMS NetCDF + CSV files)
- `cpcb-real-data` (CPCB ground station AQI files)

1. Open `notebooks/isro-bah.ipynb` in Kaggle (or any Jupyter environment
   with a GPU), with the two datasets above attached and internet enabled.
2. Run all cells top to bottom. Cell 1 installs dependencies
   (`folium`, `cartopy`, `netCDF4`, etc.) automatically.
3. Outputs are written to `/kaggle/working/` (or `outputs/` if you adapt
   the paths for a local run — see `requirements.txt` for local setup).

## Requirements

See [`requirements.txt`](requirements.txt). Core stack: PyTorch, NumPy,
Pandas, scikit-learn, SciPy, Matplotlib, Cartopy, Folium, netCDF4, imageio.

## License

MIT — see [`LICENSE`](LICENSE).
