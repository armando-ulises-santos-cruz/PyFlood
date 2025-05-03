import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from sklearn.metrics import mean_squared_error
import os

def load_land_cover_data(raster_file, null_value=None):
    """
    Load a land cover raster and handle optional null values.

    Parameters
    ----------
    raster_file : str
        Path to the input land cover raster file.
    null_value : int or float, optional
        Value in the raster to treat as NaN (default is None).

    Returns
    -------
    land_cover : ndarray
        2D array of land cover class IDs.
    transform : Affine
        Affine transformation matrix for the raster.
    crs : CRS
        Coordinate Reference System of the raster.
    """
    with rasterio.open(raster_file) as dataset:
        land_cover = dataset.read(1)
        transform = dataset.transform
        crs = dataset.crs

    if null_value is not None:
        land_cover = land_cover.astype(float)
        land_cover[land_cover == null_value] = np.nan

    return land_cover, transform, crs

def compare_hwms_to_model(gdf_aligned, wl_pred, crs, output_shapefile="hwm_vs_wl_comparison.shp"):
    """
    Compare observed High-Water Marks (HWMs) to model water level predictions and save comparison as a shapefile.

    Parameters
    ----------
    gdf_aligned : GeoDataFrame
        Aligned HWM points containing observed elevations and grid indices.
    wl_pred : ndarray
        2D array of predicted water levels from the model.
    crs : CRS
        Coordinate Reference System to assign to the output shapefile.
    output_shapefile : str, optional
        Path to save the comparison shapefile (default is "hwm_vs_wl_comparison.shp").

    Returns
    -------
    gdf_comparison : GeoDataFrame
        GeoDataFrame containing observed vs predicted water levels, errors, and pixel indices.
    """

    observed_hwm = gdf_aligned["hwm"].values
    row_indices = gdf_aligned["row"].values.astype(int)
    col_indices = gdf_aligned["col"].values.astype(int)

    predicted_wl = wl_pred[row_indices, col_indices]
    errors = predicted_wl - observed_hwm

    comparison_data = np.column_stack([
        gdf_aligned["lon_dem"].values,
        gdf_aligned["lat_dem"].values,
        observed_hwm,
        predicted_wl,
        errors,
        row_indices,
        col_indices
    ])

    gdf_comparison = gpd.GeoDataFrame(
        comparison_data,
        columns=["lon_dem", "lat_dem", "hwm", "z_wl", "error", "row", "col"],
        geometry=[Point(xy) for xy in zip(comparison_data[:, 0], comparison_data[:, 1])]
    )
    gdf_comparison.set_crs(crs, inplace=True)
    gdf_comparison.to_file(output_shapefile)

    print(f"Comparison results saved to: {output_shapefile}")

    return gdf_comparison


def compute_error_metrics(observed_hwm, predicted_wl):
    """
    Compute flood prediction error metrics between observed HWMs and modeled water levels.

    Parameters
    ----------
    observed_hwm : ndarray
        1D array of observed HWM elevations.
    predicted_wl : ndarray
        1D array of predicted water levels.

    Returns
    -------
    rmse : float
        Root Mean Squared Error between observed and predicted water levels.
    mae : float
        Mean Absolute Error.
    scatter_index : float
        Scatter index (normalized RMSE).
    bias : float
        Mean bias (predicted minus observed).
    n_points : int
        Number of valid points used in the calculations.
    """
    valid_mask = ~np.isnan(observed_hwm) & ~np.isnan(predicted_wl)
    observed = observed_hwm[valid_mask]
    predicted = predicted_wl[valid_mask]

    rmse = np.sqrt(mean_squared_error(observed, predicted))
    mae = np.mean(np.abs(predicted - observed))
    bias = np.mean(predicted - observed)
    scatter_index = rmse / np.mean(observed)

    return rmse, mae, scatter_index, bias, len(observed)

def load_distance_raster(distance_file):
    """
    Load a distance-to-coastline raster and replace NoData values with NaN.

    Parameters
    ----------
    distance_file : str
        Path to the input distance raster file.

    Returns
    -------
    distances : ndarray
        2D array of distance values with NaNs for invalid cells.
    """
    with rasterio.open(distance_file) as src:
        distances = src.read(1).astype(np.float32)

    distances = np.where(distances == -9999, np.nan, distances)
    return distances

def define_parameter_bounds(value_mapping_min, value_mapping_max):
    """
    Create parameter bounds for Bayesian Optimization from reduction factor mappings.

    Parameters
    ----------
    value_mapping_min : dict
        Dictionary mapping land cover class IDs to minimum reduction factors.
    value_mapping_max : dict
        Dictionary mapping land cover class IDs to maximum reduction factors.

    Returns
    -------
    pbounds : dict
        Dictionary of parameter bounds formatted for Bayesian Optimization.
    """
    pbounds = {f'rf_{int(k)}': (value_mapping_min[k], value_mapping_max[k]) for k in value_mapping_min.keys()}
    return pbounds

def objective_function_bayesopt(rf_values, rf_test, distances, wl_pred, z_dem, gdf_aligned, initial_flood_extent_points, csv_filename):
    """
    Objective function to minimize during Bayesian Optimization.

    Parameters
    ----------
    rf_values : dict
        Reduction factors proposed by the optimizer.
    rf_test : ndarray
        2D array of land cover class IDs.
    distances : ndarray
        2D array of distances to coastline.
    wl_pred : ndarray
        2D array of predicted water levels before calibration.
    z_dem : ndarray
        2D array of DEM elevations.
    gdf_aligned : GeoDataFrame
        Aligned High-Water Mark points.
    initial_flood_extent_points : int
        Number of HWM points within the original flood extent.
    csv_filename : str
        Path to save optimization progress to a CSV file.

    Returns
    -------
    modified_mse : float
        Negative modified Mean Squared Error for optimizer maximization.
    """
    rf_vector = {int(k.split('_')[1]): v for k, v in rf_values.items()}

    rf_assigned = np.full_like(rf_test, np.nan, dtype=np.float32)
    for lc_id, rf in rf_vector.items():
        rf_assigned[rf_test == lc_id] = rf

    reduced_height = distances * rf_assigned
    reduced_wl = wl_pred - reduced_height
    reduced_wd = reduced_wl - z_dem
    reduced_wd = np.where(reduced_wd < 0, np.nan, reduced_wd)
    actual_reduced_wl = reduced_wd + z_dem

    row_indices = gdf_aligned["row"].values.astype(int)
    col_indices = gdf_aligned["col"].values.astype(int)
    observed_hwm = gdf_aligned["hwm"].values
    predicted_wl_opt = actual_reduced_wl[row_indices, col_indices]

    valid_mask = ~np.isnan(observed_hwm) & ~np.isnan(predicted_wl_opt)
    observed_hwm = observed_hwm[valid_mask]
    predicted_wl_opt = predicted_wl_opt[valid_mask]

    num_points = np.sum(valid_mask)
    mse = mean_squared_error(observed_hwm, predicted_wl_opt)

    penalty_exponent = 3
    penalty_ratio = initial_flood_extent_points / num_points
    modified_mse = mse * (penalty_ratio ** penalty_exponent)

    # Save results to CSV
    iteration = len(pd.read_csv(csv_filename)) if os.path.exists(csv_filename) else 0
    result_data = {'Iteration': iteration, 'MSE': modified_mse, 'Num_Valid_Points': num_points, **rf_vector}
    pd.DataFrame([result_data]).to_csv(csv_filename, mode='a', header=not os.path.exists(csv_filename), index=False)

    print(f"Iteration {iteration}: Modified MSE = {modified_mse:.6f}, Num_Valid_Points = {num_points}")

    return -modified_mse

def apply_optimized_reduction_factors(rf_test, distances, optimized_rf, wl_pred, z_dem):
    """
    Apply optimized reduction factors to adjust predicted flood maps.

    Parameters
    ----------
    rf_test : ndarray
        2D array of land cover class IDs.
    distances : ndarray
        2D array of distances to coastline.
    optimized_rf : dict
        Dictionary of optimized reduction factors per land cover class.
    wl_pred : ndarray
        2D array of initial predicted water levels.
    z_dem : ndarray
        2D array of DEM elevations.

    Returns
    -------
    reduced_wl : ndarray
        2D array of recalculated water levels.
    reduced_wd : ndarray
        2D array of recalculated water depths.
    """
    rf_assigned = np.full_like(rf_test, np.nan, dtype=np.float32)
    for lc_id, rf in optimized_rf.items():
        rf_assigned[rf_test == lc_id] = rf

    reduced_height = distances * rf_assigned
    reduced_wl = wl_pred - reduced_height
    reduced_wd = reduced_wl - z_dem
    reduced_wd = np.where(reduced_wd < 0, np.nan, reduced_wd)

    return reduced_wl, reduced_wd


def mask_attenuated_results(array, sea_core_file):
    """
    Mask attenuated flood maps to exclude sea core regions.

    Parameters
    ----------
    array : ndarray
        2D array of flood depth or water level predictions.
    sea_core_file : str
        Path to the sea core raster file.

    Returns
    -------
    masked_array : ndarray
        2D array with sea regions masked out (set as NaN).
    """
    with rasterio.open(sea_core_file) as src:
        sea_core = src.read(1)

    return np.where(sea_core == 0, array, np.nan)


def save_attenuated_raster(array, reference_file, output_file, nodata_val=-9999):
    """
    Save an attenuated flood raster using a reference raster's profile.

    Parameters
    ----------
    array : ndarray
        2D array to save.
    reference_file : str
        Path to the reference raster to copy metadata.
    output_file : str
        Path to save the output raster.
    nodata_val : float, optional
        NoData value for the output raster (default is -9999).

    Returns
    -------
    None
    """
    with rasterio.open(reference_file) as src:
        profile = src.profile.copy()

    profile.update(dtype=rasterio.float32, count=1, nodata=nodata_val, compress='lzw')

    array_clean = np.where(np.isnan(array), nodata_val, array).astype(np.float32)

    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(array_clean, 1)

    print(f"Saved attenuated raster to: {output_file}")
