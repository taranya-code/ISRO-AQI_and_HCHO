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

<img width="1350" height="600" alt="training_curve" src="https://github.com/user-attachments/assets/f9116a16-edf6-499e-bae4-5f0fa28dca9d" />


### Model evaluation — scatter, residuals, uncertainty

<img width="2550" height="750" alt="evaluation" src="https://github.com/user-attachments/assets/1db47940-7dd6-4c90-af8d-bcc7f928e00c" />


### Seasonal AQI maps (India basemap)

<img width="3562" height="1144" alt="seasonal_aqi" src="https://github.com/user-attachments/assets/6768b748-f6c2-496a-900a-cf71fa2748f6" />


### Seasonal HCHO / Fire stratification (India basemap)

<img width="3576" height="1947" alt="seasonal_stratification" src="https://github.com/user-attachments/assets/73a80339-3915-461a-a685-e22af0df0e7e" />


### DBSCAN vs. threshold hotspot detection

<img width="2949" height="904" alt="dbscan_comparison" src="https://github.com/user-attachments/assets/e63a2fec-8add-40c2-8068-792359b34335" />

### Fire–HCHO correlation, BLH regression & wind transport

<img width="2958" height="881" alt="correlation_transport" src="https://github.com/user-attachments/assets/45bf1fe5-fa8f-4742-b52e-9bc22d681073" />

### Animated HCHO evolution (Kharif burning season)

<img width="656" height="545" alt="hcho_animation" src="https://github.com/user-attachments/assets/f1f5234a-cce0-4890-ac6c-ae808448769b" />


### Regional time series & temporal attention weights

<img width="2079" height="1481" alt="timeseries_attention" src="https://github.com/user-attachments/assets/b5d12255-c5a8-486e-9847-29c7c5a36481" />



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
