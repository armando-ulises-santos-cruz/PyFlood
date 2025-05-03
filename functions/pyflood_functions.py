import numpy as np
import rasterio
import os
import time
import psutil
from pyproj import Transformer
from pykrige.uk import UniversalKriging
from concurrent.futures import ThreadPoolExecutor
import geopandas as gpd

# Import constants from config
from config import (
    DEM_NULL_VALUE,
    DEM_HIGH_THRESHOLD,
    DEM_LOW_THRESHOLD,
    CHUNK_SIZE,
    THRESHOLD_DEPTH
)

def load_raster_data(raster_file, null_value=DEM_NULL_VALUE, high_threshold=DEM_HIGH_THRESHOLD, low_threshold=DEM_LOW_THRESHOLD):
    """
    Load a DEM raster, applying null values and elevation thresholds.

    Parameters
    ----------
    raster_file : str
        Path to the input DEM raster.
    null_value : float, optional
        Value to be treated as NoData.
    high_threshold : float, optional
        Maximum elevation to keep.
    low_threshold : float, optional
        Minimum elevation to keep.

    Returns
    -------
    z_dem : ndarray
        2D array of DEM elevations after cleaning.
    transform : Affine
        Rasterio affine transform.
    crs : CRS
        Coordinate reference system.
    """
    with rasterio.open(raster_file) as dataset:
        z_dem = dataset.read(1)
        transform = dataset.transform
        crs = dataset.crs

    z_dem = z_dem.astype(float)
    z_dem[z_dem == null_value] = np.nan
    z_dem[z_dem > high_threshold] = np.nan
    z_dem[z_dem < low_threshold] = np.nan

    return z_dem, transform, crs

def extract_raster_coordinates(elevation, transform, crs):
    """
    Extract X and Y coordinates from raster grid.

    Parameters
    ----------
    elevation : ndarray
        2D array of DEM elevations.
    transform : Affine
        Rasterio affine transform.
    crs : CRS
        Raster CRS.

    Returns
    -------
    x : ndarray
        2D array of X coordinates.
    y : ndarray
        2D array of Y coordinates.
    crs : CRS
        Possibly updated CRS (if reprojected).
    """
    cols, rows = np.meshgrid(np.arange(elevation.shape[1]), np.arange(elevation.shape[0]))
    x, y = rasterio.transform.xy(transform, rows, cols, offset='center')
    x = np.array(x)
    y = np.array(y)

    if not crs.is_projected:
        transformer = Transformer.from_crs(crs, crs.to_utm(), always_xy=True)
        x, y = transformer.transform(x, y)

    print("Coordinates extracted in UTM.")
    return np.array(x), np.array(y), crs

def load_aligned_water_stations(shapefile_path):
    """
    Load aligned water station points from a shapefile.

    Parameters
    ----------
    shapefile_path : str
        Path to the aligned stations shapefile.

    Returns
    -------
    aligned_stations : ndarray
        Array with columns [X, Y, Water Level, DEM Elevation].
    """
    gdf = gpd.read_file(shapefile_path)
    print(f"Shapefile fields available: {gdf.columns.tolist()}")

    aligned_stations = np.column_stack((
        gdf["X_UTM"].values,
        gdf["Y_UTM"].values,
        gdf["Water_Leve"].values,
        gdf["DEM_Elevat"].values
    ))

    print(f"Loaded {aligned_stations.shape[0]} aligned water stations.")
    return aligned_stations

def log_system_usage(message=""):
    """
    Log memory and CPU usage at a checkpoint.

    Parameters
    ----------
    message : str, optional
        Description for the checkpoint.

    Returns
    -------
    None
    """
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 ** 3)  # GB
    cpu_usage = psutil.cpu_percent(interval=1)
    print(f"[{message}] Memory Usage: {mem_usage:.2f} GB | CPU Usage: {cpu_usage:.1f}%")

def krige_water_levels(
    lon_stations, lat_stations, wl_stations, dem_stations, 
    dem_lon, dem_lat, dem_elev, chunk_size=CHUNK_SIZE
):
    """
    Perform Universal Kriging with external drift (DEM) using multi-threading.

    Parameters
    ----------
    lon_stations : ndarray
        Longitude of station points.
    lat_stations : ndarray
        Latitude of station points.
    wl_stations : ndarray
        Water levels at stations.
    dem_stations : ndarray
        DEM elevations at stations.
    dem_lon : ndarray
        DEM longitude grid.
    dem_lat : ndarray
        DEM latitude grid.
    dem_elev : ndarray
        DEM elevation grid.
    chunk_size : int, optional
        Number of prediction points per thread (default from config).

    Returns
    -------
    wl_pred : ndarray
        Interpolated water levels over the DEM grid.
    wl_var : ndarray
        Kriging prediction variance over the DEM grid.
    """
    n_total = dem_lon.size
    print("Total number of prediction points:", n_total)

    UK = UniversalKriging(
        lon_stations, lat_stations, wl_stations,
        variogram_model='spherical',
        drift_terms=['specified'],
        specified_drift=[dem_stations],
        verbose=False,
        enable_plotting=False
    )

    wl_pred_flat = np.empty(n_total)
    wl_var_flat = np.empty(n_total)

    def process_chunk(start, end):
        sub_lon = dem_lon.ravel()[start:end]
        sub_lat = dem_lat.ravel()[start:end]
        sub_z = dem_elev.ravel()[start:end]
        pred, var = UK.execute('points', sub_lon, sub_lat, specified_drift_arrays=[sub_z])
        return start, end, pred, var

    log_system_usage("Before kriging")
    t0 = time.time()

    chunk_indices = range(0, n_total, chunk_size)
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_chunk, i, min(i + chunk_size, n_total)) for i in chunk_indices]
        for future in futures:
            start, end, pred, var = future.result()
            wl_pred_flat[start:end] = pred
            wl_var_flat[start:end] = var

    t1 = time.time()
    print(f"Parallel Kriging completed in {t1 - t0:.2f} seconds.")
    log_system_usage("After kriging")

    wl_pred = wl_pred_flat.reshape(dem_lon.shape)
    wl_var = wl_var_flat.reshape(dem_lon.shape)

    return wl_pred, wl_var

def save_raster(array, reference_raster, output_filename, nodata_val=DEM_NULL_VALUE):
    """
    Save a numpy array as a GeoTIFF raster.

    Parameters
    ----------
    array : ndarray
        2D array to save.
    reference_raster : str
        Path to reference raster for metadata.
    output_filename : str
        Output raster file path.
    nodata_val : float, optional
        NoData value for the output (default from config).

    Returns
    -------
    None
    """
    with rasterio.open(reference_raster) as src:
        profile = src.profile.copy()

    profile.update(dtype=rasterio.float32, count=1, nodata=nodata_val)

    array_clean = np.where(np.isnan(array), nodata_val, array).astype(np.float32)

    with rasterio.open(output_filename, 'w', **profile) as dst:
        dst.write(array_clean, 1)

    print(f"Saved raster to: {output_filename}")

def compute_water_depth(wl_pred, z_dem):
    """
    Compute water depth from water level predictions and DEM.

    Parameters
    ----------
    wl_pred : ndarray
        Predicted water levels.
    z_dem : ndarray
        DEM elevations.

    Returns
    -------
    wd_pred : ndarray
        Computed water depth.
    """
    wd_pred = np.where(wl_pred > z_dem, wl_pred - z_dem, np.nan)
    return wd_pred

def mask_with_sea_core(pred_array, sea_core_file):
    """
    Apply a sea core mask to a prediction array.

    Parameters
    ----------
    pred_array : ndarray
        Array to mask.
    sea_core_file : str
        Path to sea core mask raster.

    Returns
    -------
    pred_masked : ndarray
        Masked prediction array.
    """
    with rasterio.open(sea_core_file) as src:
        sea_core = src.read(1)

    pred_masked = np.where(sea_core == 0, pred_array, np.nan)
    return pred_masked

def apply_threshold_and_export(input_file, threshold_value=THRESHOLD_DEPTH, output_suffix="_over_threshold", nodata_val=DEM_NULL_VALUE):
    """
    Apply a threshold to a raster and export the masked version.

    Parameters
    ----------
    input_file : str
        Input raster file path.
    threshold_value : float, optional
        Threshold value to apply (default from config).
    output_suffix : str, optional
        Suffix for the output file name.
    nodata_val : float, optional
        NoData value (default from config).

    Returns
    -------
    None
    """
    with rasterio.open(input_file) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    data_thresholded = np.where(data >= threshold_value, data, np.nan)
    profile.update(dtype=rasterio.float32, count=1, nodata=nodata_val)
    data_thresholded_clean = np.where(np.isnan(data_thresholded), nodata_val, data_thresholded)

    base, ext = os.path.splitext(input_file)
    output_file = f"{base}{output_suffix}.tif"

    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(data_thresholded_clean, 1)

    print(f"Thresholded raster exported to: {output_file}")


