# PyFlood: Coastal Flood Mapping using High-Resolution DEM and Land Cover Data

A modular, reproducible framework for simulating and validating coastal flooding using open-source tools and high-resolution datasets.

---

## Overview

PyFlood is a Python-based system for simulating coastal flood extents and water depths. It combines high-resolution Digital Elevation Models (DEM), land cover, and water level data. Core features include:

- Coastal flood mapping using Universal Kriging with External Drift.
- Post-calibration through Bayesian Optimization based on high water marks observations.
- Comparison against observed HWMs and hydrodynamic model outputs (SFINCS).

PyFlood is organized into modular functions and reproducible Jupyter notebooks.

---

## Project Structure

```
PyFlood/
├── config.py                     # Centralized paths, thresholds, constants
├── functions/                    # Modularized core functions
│   ├── preprocessing_functions.py
│   ├── preprocessing_attenuation_functions.py
│   ├── pyflood_functions.py
│   ├── pyflood_calibration_attenuation_functions.py
│   ├── validation_functions.py
├── input_data/                    # Raw input datasets
│   ├── dep_subgrid.tif
│   ├── resampled_landcover.tif
│   ├── filtered_hwm_study_area.shp
│   ├── SFINCS_Flood_Water_Depth_over_1m.tif
│   ├── SFINCS_Flood_Water_Level.tif
├── output_data/                   # Processed outputs
│   ├── (seed point, masks, coastline, flood maps, validation shapefiles, etc.)
├── notebooks/                     # Main Jupyter notebooks
│   ├── preprocessing.ipynb
│   ├── PyFlood.ipynb
│   ├── validation.ipynb
│   ├── visualization_example.ipynb
├── README.md
├── LICENSE
├── header.png
```

---

## Input Files

Example input files can be downloaded from:
[PyFlood Input Data Zenodo](https://zenodo.org/doi/10.5281/zenodo.15330868)

| File | Description | Source |
|:---|:---|:---|
| dep_subgrid.tif | High-resolution DEM | NOAA CUDEM |
| resampled_landcover.tif | Merged low-resolution land cover data resampled | NOAA Coastal C-CAP |
| filtered_hwm_study_area.shp | Filtered High Water Marks | FEMA Reports |
| SFINCS_Flood_Water_Depth_over_1m.tif | SFINCS flood depth output | Hydrodynamic model |
| SFINCS_Flood_Water_Level.tif | SFINCS water level predictions | Hydrodynamic model |

---

## Main Components

| Notebook | Purpose |
|:---|:---|
| Preprocessing.ipynb | Loads DEM and land cover, extracts seed point, generates coastline, aligns HWMs. |
| PyFlood_Modeling.ipynb | Performs kriging interpolation, flood mapping, Bayesian calibration, attenuation |
| Validation.ipynb | Compares CFM and CFM+A against SFINCS and observed HWMs, computes validation metrics. |

---

## Configuration Settings (config.py)

- Input and output paths.
- DEM and Land Cover thresholds (e.g., DEM_NULL_VALUE, DEM_HIGH_THRESHOLD).
- CRS settings (default EPSG:26915).
- Flood threshold depth (default 1.0 m).
- Bayesian Optimization parameters (INIT_POINTS = 22, N_ITER = 10).
- Manual reduction factors for land cover classes.

---

## Workflow Summary

### 1. Preprocessing
- Clean DEM and Land Cover datasets.
- Identify and validate the seed point (ocean seed).
- Extract coastline from DEM.
- Generate Distance-to-Coastline raster.
- Align observed HWMs to DEM grid.

### 2. Flood Mapping (PyFlood.ipynb)
- Perform Universal Kriging with External Drift.
- Predict Water Levels.
- Generate Coastal Flood Map (CFM).
- Compare CFM against HWMs.
- Optimize reduction factors using Bayesian Optimization.
- Generate Coastal Flood Map with Attenuation (CFM+A).

### 3. Validation
- Compare CFM and CFM+A against SFINCS flood maps.
- Compare against observed HWMs.
- Calculate confusion matrices and metrics (CSI, Recall, F1-Score, etc.).
- Produce observed vs predicted HWM scatter plots.

---

## Example Usage

### Preprocessing

```python
from functions.preprocessing_functions import load_raster_data, extract_raster_coordinates

z_dem, transform, crs = load_raster_data(config.DEM_FILE)
dem_lon, dem_lat, _ = extract_raster_coordinates(z_dem, transform, crs)
```

### Kriging-based Flood Mapping

```python
from functions.pyflood_functions import krige_water_levels

wl_pred, wl_var = krige_water_levels(
    lon_stations, lat_stations, wl_stations, dem_stations,
    dem_lon, dem_lat, z_dem
)
```

### Validation

```python
from functions.validation_functions import compare_hwm_with_model

validation_results = compare_hwm_with_model(
    wse_raster_path=config.SFINCS_WL_FILE,
    hwm_shapefile_path=config.ALIGNED_HWM_SHP,
    output_shapefile_path=config.HWM_VS_SFINCS_SHP
)
```

---

## Requirements

| Package | Version |
|:---|:---|
| numpy | >=1.21 |
| rasterio | >=1.2 |
| geopandas | >=0.10 |
| pykrige | >=1.6 |
| matplotlib | >=3.5 |
| shapely | >=2.0 |
| scikit-learn | >=1.0 |
| pandas | >=1.3 |
| joblib | >=1.1 |
| pyproj | >=3.2 |

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Future Enhancements

- Extensions to include sea-level rise and land cover changes.
---

## Author

**Armando Ulises Santos Cruz**  
Email: armando.ulises.santos@utexas.edu

Distributed under the Creative Commons CC0 1.0 Universal License. See LICENSE for more information.
