# config.py

# ----------------------------------------------------------
# Import Modules
import os

# ----------------------------------------------------------
# Folder Paths
INPUT_FOLDER = "../input_data"
OUTPUT_FOLDER = "../output_data"

# ----------------------------------------------------------
# Input Data Paths
DEM_FILE = os.path.join(INPUT_FOLDER, "dep_subgrid.tif")
LANDCOVER_FILE = os.path.join(INPUT_FOLDER, "resampled_landcover.tif")
HWM_INPUT_FILE = os.path.join(INPUT_FOLDER, "filtered_hwm_study_area.shp")
SFINCS_WD_FILE = os.path.join(INPUT_FOLDER, "SFINCS_Flood_Water_Depth_over_1m.tif")
SFINCS_WL_FILE = os.path.join(INPUT_FOLDER, "SFINCS_Flood_Water_Level.tif")

# ----------------------------------------------------------
# Preprocessing Outputs
SEED_CSV = os.path.join(OUTPUT_FOLDER, "seed_ocean.csv")
SEA_CORE_TIF = os.path.join(OUTPUT_FOLDER, "01_dem_sea_core.tif")
LAND_CORE_TIF = os.path.join(OUTPUT_FOLDER, "01_dem_land_core.tif")
WATER_BODY_SHP = os.path.join(OUTPUT_FOLDER, "water_body.shp")
DISTANCE_TO_WATER_BODY = os.path.join(OUTPUT_FOLDER, "Distance_to_Water_Body.tif")
ALIGNED_WATER_STATIONS_SHP = os.path.join(OUTPUT_FOLDER, "Aligned_Water_Stations.shp")
ALIGNED_HWM_SHP = os.path.join(OUTPUT_FOLDER, "Aligned_HWM_with_indices.shp")
# NEW – Water Body Perimeter
WATER_BODY_PERIMETER_TIF = os.path.join(OUTPUT_FOLDER, "02_water_body_perimeter.tif")

# ----------------------------------------------------------
# PyFlood Outputs
CFM_WD_TIF = os.path.join(OUTPUT_FOLDER, "CFM_Flood_Water_Depth.tif")
CFM_WL_TIF = os.path.join(OUTPUT_FOLDER, "CFM_Flood_Water_Level.tif")
CFM_A_WD_TIF = os.path.join(OUTPUT_FOLDER, "CFM_A_Flood_Water_Depth.tif")
CFM_A_WL_TIF = os.path.join(OUTPUT_FOLDER, "CFM_A_Flood_Water_Level.tif")
CFM_A_WD_FINAL_TIF = os.path.join(OUTPUT_FOLDER, "CFM_A_Flood_Water_Depth_FINAL.tif")
CFM_A_WL_FINAL_TIF = os.path.join(OUTPUT_FOLDER, "CFM_A_Flood_Water_Level_FINAL.tif")

# ----------------------------------------------------------
# Validation Outputs
HWM_VS_SFINCS_SHP = os.path.join(OUTPUT_FOLDER, "hwm_vs_sfincs_comparison.shp")
HWM_VS_CFM_SHP = os.path.join(OUTPUT_FOLDER, "hwm_vs_cfm_comparison.shp")
HWM_VS_CFM_A_SHP = os.path.join(OUTPUT_FOLDER, "hwm_vs_cfm_a_comparison.shp")
CONFUSION_MATRIX_CFM_TIF = os.path.join(OUTPUT_FOLDER, "Confusion_Matrix_CFM.tif")
CONFUSION_MATRIX_CFM_A_TIF = os.path.join(OUTPUT_FOLDER, "Confusion_Matrix_CFM_A.tif")

# ----------------------------------------------------------
# Optimization Outputs
BAYESIAN_OPTIMIZATION_CSV = os.path.join(OUTPUT_FOLDER, "bayesian_optimization_results.csv")

# ----------------------------------------------------------
# CRS and Spatial Settings
DEM_EPSG = 26915  # EPSG code for UTM Zone 15N

# ----------------------------------------------------------
# DEM Cleaning Thresholds
DEM_NULL_VALUE = -9999
DEM_HIGH_THRESHOLD = 8849  # Maximum allowed elevation (Mount Everest)
DEM_LOW_THRESHOLD = -500   # Minimum allowed elevation

# ----------------------------------------------------------
# Land Cover Settings
WATER_CLASS_ID = 21  # Land cover ID representing water

# ----------------------------------------------------------
# Distance Calculation Settings
CHUNK_SIZE_DISTANCE = 800000  # Number of DEM points per parallel processing chunk

# ----------------------------------------------------------
# Distance Parallel Interpolation Settings
CHUNK_SIZE = 500000  # Number of DEM points per parallel processing chunk

# ----------------------------------------------------------
# Flood Threshold Settings
THRESHOLD_DEPTH = 1.0  # Minimum flood depth threshold in meters

# ----------------------------------------------------------
# Manual Reduction Factors (Table 2)
MANUAL_REDUCTION_FACTORS = {
2: 0.00025,
3: 0.000667,
4: 0.0005,
5: 0.00025,
6: 0.0004,
7: 5.2e-05,
8: 0.00025,
9: 0.0005,
10: 0.000125,
11: 0.0004,
12: 0.00035,
13: 0.000225,
14: 0.00025,
15: 0.00025,
16: 0.000225,
17: 8.8e-05,
18: 8.8e-05,
19: 0.000115,
20: 0.000115,
21: 0.000125,
22: 3.8e-05,
23: 0.000125,
}

# ----------------------------------------------------------
# Bayesian Optimization Settings
INIT_POINTS = 22  # Number of random initial points
N_ITER = 10       # Number of optimization iterations
