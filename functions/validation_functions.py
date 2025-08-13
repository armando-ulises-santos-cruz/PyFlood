import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

def load_flood_maps(flood_map_path, inland_mask_path):
    """
    Load a flood raster and apply an inland mask to remove ocean regions.

    Parameters
    ----------
    flood_map_path : str
        Path to the input flood raster file (e.g., predicted depths).
    inland_mask_path : str
        Path to the inland mask raster (binary: 1 for inland, 0 for ocean).

    Returns
    -------
    flood : ndarray
        2D array of flood values with ocean areas masked as NaN.
    profile : dict
        Raster profile (metadata) from the flood raster.
    """
    with rasterio.open(inland_mask_path) as src:
        inland_mask = src.read(1).astype(np.uint8)

    with rasterio.open(flood_map_path) as src:
        flood = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata

    flood = np.where(inland_mask == 1, flood, np.nan)
    flood = np.where(flood == nodata, np.nan, flood)

    return flood, profile

def compute_confusion_matrix(sfincs_binary, model_binary, inland_mask):
    """
    Compute a confusion raster comparing SFINCS and model binary flood extents.

    Parameters
    ----------
    sfincs_binary : ndarray
        2D binary array (1: flood, 0: dry) from SFINCS model.
    model_binary : ndarray
        2D binary array (1: flood, 0: dry) from the tested model (e.g., PyFlood).
    inland_mask : ndarray
        2D binary array indicating inland areas (1: valid, 0: ocean).

    Returns
    -------
    confusion_raster : ndarray
        2D array with confusion classes:
            - 0: True Negative
            - 1: False Positive
            - 2: False Negative
            - 3: True Positive
            - 255: No Data
    """
    confusion_raster = np.full(sfincs_binary.shape, 255, dtype=np.uint8)
    valid_mask = (inland_mask == 1)

    confusion_raster[(valid_mask) & (sfincs_binary == 1) & (model_binary == 1)] = 3
    confusion_raster[(valid_mask) & (sfincs_binary == 0) & (model_binary == 0)] = 0
    confusion_raster[(valid_mask) & (sfincs_binary == 0) & (model_binary == 1)] = 1
    confusion_raster[(valid_mask) & (sfincs_binary == 1) & (model_binary == 0)] = 2

    return confusion_raster

def compute_extent_metrics(confusion_raster):
    """
    Calculate flood extent validation metrics from a confusion raster.

    Parameters
    ----------
    confusion_raster : ndarray
        2D array from compute_confusion_matrix().

    Returns
    -------
    metrics : dict
        Dictionary containing CSI, Precision, Recall, Specificity, F1 Score, FDR, Proportion Correct, and Bias Ratio.
    """
    TP = np.sum(confusion_raster == 3)
    TN = np.sum(confusion_raster == 0)
    FP = np.sum(confusion_raster == 1)
    FN = np.sum(confusion_raster == 2)

    CSI = TP / (TP + FN + FP) if (TP + FN + FP) > 0 else 0
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    Specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    F1_Score = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    FDR = FP / (TP + FP) if (TP + FP) > 0 else 0
    Proportion_Correct = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    Bias_Ratio = (TP + FP) / (TP + FN) if (TP + FN) > 0 else 0

    return {
        "CSI": CSI, "Precision": Precision, "Recall": Recall, "Specificity": Specificity,
        "F1_Score": F1_Score, "FDR": FDR, "Proportion_Correct": Proportion_Correct, "Bias_Ratio": Bias_Ratio
    }

def compare_hwm_with_model(wse_raster_path, hwm_shapefile_path, output_shapefile_path):
    """
    Compare observed HWMs to modeled Water Surface Elevation (WSE) values and save results as a shapefile.

    Parameters
    ----------
    wse_raster_path : str
        Path to the model WSE raster (predicted flood surface elevation).
    hwm_shapefile_path : str
        Path to the shapefile containing observed High-Water Marks (HWMs).
    output_shapefile_path : str
        Path to save the output comparison shapefile.

    Returns
    -------
    metrics : dict
        Dictionary containing RMSE, MAE, Bias, Scatter Index, and Number of Points.
    """
    hwm_gdf = gpd.read_file(hwm_shapefile_path)

    with rasterio.open(wse_raster_path) as src:
        wse_data = src.read(1)
        crs = src.crs
        nodata = src.nodata

    row = hwm_gdf["row"].astype(int).values
    col = hwm_gdf["col"].astype(int).values
    observed = hwm_gdf["hwm"].values
    predicted = wse_data[row, col]
    predicted = np.where(predicted == nodata, np.nan, predicted)

    errors = predicted - observed

    comparison_gdf = gpd.GeoDataFrame({
        "lon_dem": hwm_gdf["lon_dem"],
        "lat_dem": hwm_gdf["lat_dem"],
        "hwm": observed,
        "wse": predicted,
        "error": errors,
        "row": row,
        "col": col,
        "geometry": hwm_gdf.geometry
    }, crs=crs)

    comparison_gdf.to_file(output_shapefile_path)

    valid = ~np.isnan(predicted) & ~np.isnan(observed)
    n = np.sum(valid)
    rmse = np.sqrt(np.mean((predicted[valid] - observed[valid]) ** 2))
    mae = np.mean(np.abs(predicted[valid] - observed[valid]))
    bias = np.mean(predicted[valid] - observed[valid])
    si = rmse / np.mean(observed[valid])

    return {
        "Num Points": n,
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "Scatter Index": si
    }