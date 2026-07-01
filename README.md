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

## Results

<!--
  HOW TO FILL THIS IN:
  1. Run the notebook, download each PNG/GIF from /kaggle/working/.
  2. Open this file in the GitHub web editor (or open a new Issue —
     either works), then drag-and-drop each image straight into the
     text box. GitHub uploads it and auto-inserts a line like:
       ![training_curve](https://github.com/user-attachments/assets/xxxx)
  3. Cut that auto-inserted line and paste it in place of the matching
     placeholder line below, then commit.
  This is exactly how the training_curve.png example was generated.
-->

### Training convergence

![training_curve](https://github.com/user-attachments/assets/e9a8f75b-3c11-464f-969b-7263e2ac2e11)

### Model evaluation — scatter, residuals, uncertainty

![evaluation](PASTE_evaluation_URL_HERE)

### Seasonal AQI maps (India basemap)

![seasonal_aqi](PASTE_seasonal_aqi_URL_HERE)

### Seasonal HCHO / Fire stratification (India basemap)

![seasonal_stratification](PASTE_seasonal_stratification_URL_HERE)

### DBSCAN vs. threshold hotspot detection

![dbscan_comparison](PASTE_dbscan_comparison_URL_HERE)

### Fire–HCHO correlation, BLH regression & wind transport

![correlation_transport](PASTE_correlation_transport_URL_HERE)

### Animated HCHO evolution (Kharif burning season)

![hcho_animation](PASTE_hcho_animation_URL_HERE)

### Regional time series & temporal attention weights

![timeseries_attention](PASTE_timeseries_attention_URL_HERE)

> `interactive_aqi_map.html` is a live Folium map and can't be embedded as
> a static image — link to it directly, or add a screenshot here too if
> you'd like a preview.

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
