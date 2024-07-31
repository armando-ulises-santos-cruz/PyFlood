import numpy as np

"""
Configuration File for PyFlood

This file contains all the input parameters for the PyFlood project.
Users can modify these parameters according to their study area and data.
"""

# Parameters configuration of PyFlood

# Preprocessing.ipynb
# Parameters for load_raster_data
raster_file_path = 'Merged_DEM_Galv_Calib.tif'
null_value = -9999
high_threshold = 8849
low_threshold = -500

# Parameters for process_land_cover
land_cover_raster_file = 'LandCover_Merged_Final.tif'
land_cover_null_value = -9999
land_cover_high_threshold = 23
land_cover_low_threshold = 2

# Slice parameters for both DEM and land cover data
start_row = 3250
end_row = 18500
start_col = 3250
end_col = 17250

# Coordinate reference systems
crs_to = 'epsg:4326'

# Parameters for plotting and saving DEM
dem_file_path = 'DEM.mat'

# Parameters for plotting and saving Land Cover Data
land_cover_file_path = 'land_cover.mat'


# PyFlood.ipynb
# Parameters for create_raster_from_dem
# Coordinate Reference System of the Output Raster File is the same as in crs_to since both must be in the same CRS.
output_crs = crs_to  # Coordinate Reference System of the output raster file

# Parameters for red_fac_grid_FOL and find_opt_L
gamma = 0.01  # representing observational error
L_test_range = (0.5, 100.5, 0.5)  # L test range

# Path to DEM.mat data
# It is the same as dem_file_path ('DEM.mat')
dem_data_path = dem_file_path

# Parameters for Coastal Flood Mapping
# Define the coordinates of a known water body point (e.g., sea) in decimal degrees.
# The format is [longitude, latitude].
# For example, [-94.85, 29.45] corresponds to a point with:
#   - Longitude: -94.85 degrees
#   - Latitude: 29.45 degrees
wb = [-94.85, 29.45]

# Define a uniform water level.
# This is a single value representing the water level to be used uniformly across the DEM.
# Example: 4.696968 meters above a reference level.
wl_s1 = np.array([4.696968])

# Define water levels from multiple monitoring stations.
# Each row in the array represents a monitoring station with the following format:
# [longitude, latitude, water level], where:
# - longitude: Longitude of the monitoring station in decimal degrees.
# - latitude: Latitude of the monitoring station in decimal degrees.
# - water level: Water level at the monitoring station in meters above a reference level.
wl_s2 = np.array([
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
])


# Parameters for Calibration
# Path to land_cover.mat data
# It is the same as land_cover_file_path ('land_cover.mat')
land_cover_data_path = land_cover_file_path

# Path to high water marks (hwm) shapefile for calibration
hwm_data_path = 'HWM_IKE_ORG.shp'

# Define bounds to use in the calibration process according to the IDs thresholds for the land cover class for the function process_land_cover
value_mapping_min = {
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

value_mapping_max = {
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

# Parameters Post-Calibration
# Generation of Pre-Calibration Coastal Flood Map and Post-Calibration Coastal Flood Map
# Apply the threshold filter
threshold_value = -1  # Equivalent to 1 meter

# Assign the name file to the Pre-Calibration Coastal Flood Map
pre_calibration_coastal_flood_map = 'PreCCFM_thr_1m.tif'

# Assign the name file to the Post-Calibration Coastal Flood Map
post_calibration_coastal_flood_map = 'PostCCFM_thr_1m.tif'

# Generation of Differences of high water marks vs Pre-Calibration Coastal Flood Map and Post-Calibration Coastal Flood Map shapefile
# Assign the name file to the Pre-Calibration Coastal Flood Map high water marks difference shapefile
pre_calibration_coastal_flood_map_shp = 'Diff_PreCCFM.shp'

# Assign the name file to the Post-Calibration Coastal Flood Map high water marks difference shapefile
post_calibration_coastal_flood_map_shp = 'Diff_PostCCFM.shp'



# Validation.ipynb
# Name of the files to be used in the validation process
# SFINCS: The SFINCS model output with a threshold of 1 meter
SFINCS = 'SFINCS_thr_1m.tif'

# PreCCFM: The Pre-Calibration Coastal Flood Map with a threshold of 1 meter
PreCCFM = 'PreCCFM_thr_1m.tif'

# PostCCFM: The Post-Calibration Coastal Flood Map with a threshold of 1 meter
PostCCFM = 'PostCCFM_thr_1m.tif'

# File paths for additional rasters used in comparison against HWMs
sfincc_thr_or = 'SFINCS_0_3048_thr_or.tif'