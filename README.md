
# PyFlood: Coastal Flood Mapping with High-Resolution Digital Elevation Model and Land Cover Data
> A comprehensive tool for simulating coastal flooding using high-resolution DEM and land cover data, incorporating advanced processing and validation techniques.

[![NPM Version][npm-image]][npm-url]
[![Build Status][travis-image]][travis-url]
[![Downloads Stats][npm-downloads]][npm-url]

PyFlood is a Python-based tool designed for simulating and analyzing coastal flooding. It leverages high-resolution Digital Elevation Models (DEM) and land cover data to provide accurate flood maps. The tool includes preprocessing, simulation, and validation notebooks to streamline the workflow.

![](header.png)

# Necessary Inputs Files
## Digital Elevation Model (Merged_DEM_Galv_Calib.tif)
Source of DEM Data
NOAA NCEI: CUDEM 1/3 arc-second resolution
Steps to Obtain and Merge DEM Tiles
1. Download Files:
Obtain the DEM files for Galveston County from the NOAA NCEI CUDEM dataset.
2. Merge Tiles in ArcGIS:
Use Mosaic To New Raster under Data Management Tools > Raster > Raster Dataset to combine the DEM files.
Maintain original parameters and save the output as Merged_DEM_Galv_Calib.tif.

## Land Cover Data (LandCover_Merged_Final.tif)
Source of Land Cover Data
NOAA NCEI: High-Resolution Coastal Land Cover Data (1-meter resolution) and 2016 C-CAP Regional Land Cover Dataset (30-meter resolution).
NOAA Land Cover Data

Steps to Obtain and Merge Land Cover Tiles
1. Download Files:
Obtain both high-resolution and regional land cover datasets for the area of interest.
2. Merge High-Resolution Tiles in ArcGIS:
Use Mosaic To New Raster to combine high-resolution land cover tiles by land cover class ID.
3. Merge High and Low-Resolution Data:
Overlay the merged high-resolution dataset onto the regional dataset, ensuring high-resolution tiles are prioritized and combining datasets by matching land cover class IDs.
4. Clip to DEM Extent:
Use the Clip tool in ArcGIS Pro to match the extent of the merged land cover raster with the DEM.
5. Save Combined Raster:
Save the combined and clipped raster as LandCover_Merged_Final.tif.

## High Water Marks (HWM_IKE_ORG.shp)
Source of High Water Marks Data
FEMA Report: Federal Emergency Management Agency (2009) provided flood depth measurements above NAVD88.
Steps to Obtain and Process High Water Marks
1. Retrieve Data:
Extract high water marks from the FEMA report, noting coordinates and height.
2. Create CSV:
Create a CSV file with columns for longitude, latitude, and z value (flood depth).
3. Import CSV and Create Points in ArcGIS:
Use the Table to Point tool to import the CSV and create point features.
4. Filter Data in ArcGIS:
Use ArcGIS to filter the high water marks to include only those within the delimited study area.
5. Save as Shapefile:
Save the filtered point features as HWM_IKE_ORG.shp.

# Configuration Parameters

## Preprocessing.ipynb Parameters
### Load Raster Data

raster_file_path: Path to the DEM file (Described in the Necessary Inputs Files section). Default: Merged_DEM_Galv_Calib.tif
null_value: Value to represent null data. Default: -9999
high_threshold: Maximum elevation threshold. Default: 8849
low_threshold: Minimum elevation threshold. Default: -500

### Process Land Cover Data
land_cover_raster_file: Path to the land cover raster file. Default: LandCover_Merged_Final.tif
land_cover_null_value: Value to represent null data in land cover. Default: -9999
land_cover_high_threshold: Maximum land cover class value. Default: 23
land_cover_low_threshold: Minimum land cover class value. Default: 2

### Slice Parameters
start_row: Starting row for slicing data. Default: 3250
end_row: Ending row for slicing data. Default: 18500
start_col: Starting column for slicing data. Default: 3250
end_col: Ending column for slicing data. Default: 17250

### Coordinate Reference Systems
crs_to: Coordinate Reference System for transformation. Default: epsg:4326

### File Paths for Saving Processed Data
dem_file_path: Path to save processed DEM data. Default: DEM.mat
land_cover_file_path: Path to save processed land cover data. Default: land_cover.mat


## PyFlood.ipynb Parameters
### Create Raster from DEM
output_crs: Coordinate Reference System of the output raster file. Default: epsg:4326

### Reduction Factors Grid and Optimal L
gamma: Observational error parameter. Default: 0.01
L_test_range: Range of L values to test. Default: (0.5, 100.5, 0.5)

### Paths to Data Files
dem_data_path: Path to DEM data file (Obtained from Preprocessing). Default: DEM.mat
land_cover_data_path: Path to land cover data file (Obtained from Preprocessing). Default: land_cover.mat
hwm_data_path: Path to high water marks shapefile (Described in the Necessary Inputs Files section). Default: HWM_IKE_ORG.shp

### Coastal Flood Mapping Parameters
wb: Coordinates of a known water body point. Default: [-94.85, 29.45]
wl_s1: Uniform water level. Default: [4.696968]
wl_s2: Water levels from multiple monitoring stations. Default:
[
    [-95.117222, 29.086111, 2.883408],  # Station 1
    [-95.04, 29.355833, 3.176016],      # Station 2
    [-94.944722, 29.220833, 3.770376],  # Station 3
    [-95.047778, 29.456667, 3.21564],   # Station 4
    [-94.877778, 29.238056, 4.416552],  # Station 5
    [-94.905278, 29.303889, 3.834384],  # Station 6
    [-94.877778, 29.238056, 4.696968],  # Station 7
    [-94.7933, 29.31, 3.206496],        # Station 8
    [-94.7247, 29.3575, 2.810256],      # Station 9
    [-94.9202016, 29.44745403, 3.898392] # Station 10
]


## Calibration Parameters
### Bounds for Calibration Process
value_mapping_min: Minimum bounds for land cover class IDs. Default:
{
    2.0: 6e-06,
    3.0: 4e-06,
    4.0: 1.8e-06,
    5.0: 5.4e-07,
    6.0: 0.00016,
    7.0: 0.000125,
    8.0: 1.25e-06,
    9.0: 5e-06,
    10.0: 4e-06,
    11.0: 0.0002,
    12.0: 3.5e-06,
    13.0: 1.5e-06,
    14.0: 1.25e-06,
    15.0: 0.00025,
    16.0: 0.00015,
    17.0: 1.25e-06,
    18.0: 2.5e-06,
    19.0: 1.15e-06,
    20.0: 1.15e-06,
    21.0: 0.000125,
    22.0: 0.000125,
    23.0: 0.000125
}

value_mapping_max: Maximum bounds for land cover class IDs. Default:
{
    2.0: 0.01,
    3.0: 0.0006,
    4.0: 0.00036,
    5.0: 9e-05,
    6.0: 0.0004,
    7.0: 0.0025,
    8.0: 0.00025,
    9.0: 0.001,
    10.0: 0.002426,
    11.0: 0.0005,
    12.0: 0.002413,
    13.0: 0.0005,
    14.0: 0.00025,
    15.0: 0.00425,
    16.0: 0.0005,
    17.0: 0.00025,
    18.0: 0.000425,
    19.0: 0.0015,
    20.0: 0.00015,
    21.0: 0.00025,
    22.0: 0.00025,
    23.0: 0.00025
}

## Post-Calibration Parameters
### Threshold and Output Files
threshold_value: Threshold filter value. Default: -1 (Equivalent to 1 meter)
pre_calibration_coastal_flood_map: Filename for the Pre-Calibration Coastal Flood Map. Default: PreCCFM_thr_1m.tif
post_calibration_coastal_flood_map: Filename for the Post-Calibration Coastal Flood Map. Default: PostCCFM_thr_1m.tif

### Comparison against high water marks and Output Files
pre_calibration_coastal_flood_map_shp: Filename for the Pre-Calibration Coastal Flood Map Difference Shapefile against high water marks. Default: Diff_PreCCFM.shp
post_calibration_coastal_flood_map_shp: Filename for the Post-Calibration Coastal Flood Map Shapefile against high water marks. Default: Diff_PostCCFM.shp



## Validation Parameters
### Validation Files
SFINCS: SFINCS model output file. Default: SFINCS_thr_1m.tif
PreCCFM: Pre-Calibration Coastal Flood Map file. Default: PreCCFM_thr_1m.tif
PostCCFM: Post-Calibration Coastal Flood Map file. Default: PostCCFM_thr_1m.tif

### Additional Rasters for Comparison Against HWMs
sfincc_thr_or: Path to SFINCS model threshold raster. Default: SFINCS_0_3048_thr_or.tif


## Usage example

PyFlood can be used to preprocess DEM and land cover data, simulate coastal flooding, and validate the results against high water marks (HWMs). Below are some example usages:

### Preprocessing DEM and Land Cover Data

```python
# Preprocess DEM data
elevation, transform, crs = load_raster_data(config.raster_file_path)
longitude, latitude = transform_coordinates(elevation, transform, crs)
save_data(elevation[config.start_row:config.end_row, config.start_col:config.end_col], 
          longitude[config.start_row:config.end_row, config.start_col:config.end_col], 
          latitude[config.start_row:config.end_row, config.start_col:config.end_col])
```

### Simulating Flooding

```python
# Simulate flooding with a uniform water level
wb = config.wb
wl_s1 = config.wl_s1
SF, fwl, L = PyFlood_StaticFlooding(lon, lat, z, wl_s1, wb)
```

### Validation Against SFINCS

```python
# Validate SFINCS against HWMs
raster_files = [config.sfincc_thr_or]
shapefile_path = config.hwm_data_path
shapefile_gdf = read_shapefile(shapefile_path)
extracted_data = {file: extract_raster_values(file, shapefile_gdf) for file in raster_files}
```

_For more examples and usage, please refer to the Jupyter Notebooks Files

## Development setup

To set up the development environment, install the dependencies and run the test suite.

```sh
pip install -r requirements.txt
pytest
```

## Release History
* 0.1.0
    * The first proper release
* 0.0.1
    * Work in progress

## Meta

Your Name – [@YourTwitter](https://twitter.com/dbader_org) – YourEmail@example.com

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/yourname/github-link](https://github.com/dbader/)

## Contributing

1. Fork it (<https://github.com/yourname/yourproject/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

<!-- Markdown link & img dfn's -->
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki
