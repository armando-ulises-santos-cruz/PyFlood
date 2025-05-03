import numpy as np
import pandas as pd
import rasterio
import time
from skimage import measure
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import Point

# Import constants from config.py
from config import (
    DEM_NULL_VALUE,
    DEM_HIGH_THRESHOLD,
    DEM_LOW_THRESHOLD,
    WATER_CLASS_ID
)


def load_raster_data(raster_file, null_value=DEM_NULL_VALUE, high_threshold=DEM_HIGH_THRESHOLD, low_threshold=DEM_LOW_THRESHOLD):
    """
    Load raster data and clean invalid elevation values.

    Parameters
    ----------
    raster_file : str
        Path to the input raster file (GeoTIFF format).
    null_value : int or float, optional
        Value in the raster to treat as 'no data'. Default is imported from config.
    high_threshold : int or float, optional
        Maximum allowed elevation value. Default is imported from config.
    low_threshold : int or float, optional
        Minimum allowed elevation value. Default is imported from config.

    Returns
    -------
    z_dem : ndarray
        2D array of cleaned elevation values.
    transform : Affine
        Affine transformation for pixel-to-world coordinates.
    crs : CRS
        Coordinate reference system of the raster.
    """
    with rasterio.open(raster_file) as dataset:
        z_dem = dataset.read(1).astype(float)
        transform = dataset.transform
        crs = dataset.crs

    z_dem[z_dem == null_value] = np.nan
    z_dem[z_dem > high_threshold] = np.nan
    z_dem[z_dem < low_threshold] = np.nan

    return z_dem, transform, crs


def generate_and_export_mask(name, compute_fn, reference_tif, nodata_val=0):
    """
    Generate a binary mask from a computation function and export it as a GeoTIFF.

    Parameters
    ----------
    name : str
        Base name for the output file (without extension).
    compute_fn : callable
        Function that returns a binary mask (2D ndarray).
    reference_tif : str
        Path to the reference raster file for geospatial metadata.
    nodata_val : int, optional
        Value to use for NoData pixels (default is 0).

    Returns
    -------
    None
    """
    start_total = time.time()

    mask = compute_fn()

    with rasterio.open(reference_tif) as src:
        profile = src.profile.copy()

    profile.update(dtype=rasterio.uint8, count=1, nodata=nodata_val)

    with rasterio.open(f"{name}.tif", 'w', **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)

    total_time = time.time() - start_total
    print(f"{name}.tif generated and saved | total time: {total_time:.2f} seconds")


def find_and_validate_seed(z_dem, land_cover, transform, crs, output_csv, water_class=WATER_CLASS_ID):
    """
    Find the lowest elevation seed point and validate its land cover as water.

    Parameters
    ----------
    z_dem : ndarray
        2D array of elevation values (DEM).
    land_cover : ndarray
        2D array of land cover class IDs.
    transform : Affine
        Affine transformation for pixel-to-world coordinates.
    crs : CRS
        Coordinate reference system of the raster.
    output_csv : str
        Path to save the seed point coordinates as a CSV file.
    water_class : int, optional
        Land cover class ID representing water. Default is imported from config.

    Returns
    -------
    row_seed : int
        Row index of the seed point.
    col_seed : int
        Column index of the seed point.
    """
    min_idx = np.unravel_index(np.nanargmin(z_dem), z_dem.shape)
    row_seed, col_seed = min_idx

    seed_elevation = z_dem[row_seed, col_seed]
    seed_land_cover = land_cover[row_seed, col_seed]

    if seed_land_cover != water_class:
        raise ValueError(f"Seed point is not in water class (ID={water_class}). Found land cover ID = {seed_land_cover}")
    else:
        print(f"Seed point verified: Water class (ID={water_class}). Elevation = {seed_elevation:.2f} m")

    x_seed, y_seed = rasterio.transform.xy(transform, row_seed, col_seed, offset='center')

    df = pd.DataFrame({
        "X_UTM": [x_seed],
        "Y_UTM": [y_seed],
        "Elevation_m": [seed_elevation]
    })
    df.to_csv(output_csv, index=False)
    print(f"Seed point saved to {output_csv}")

    return row_seed, col_seed


def sea_core_from_seed(z, seed_row, seed_col):
    """
    Identify the connected sea area starting from a seed point based on elevation.

    Parameters
    ----------
    z : ndarray
        2D array of elevation values.
    seed_row : int
        Row index of the seed point.
    seed_col : int
        Column index of the seed point.

    Returns
    -------
    sea_core : ndarray
        Binary mask (2D array) of the connected sea area.
    """
    sea_mask = (z < 0).astype(int)
    labeled = measure.label(sea_mask, connectivity=2)

    seed_label = labeled[seed_row, seed_col]

    if seed_label == 0:
        raise ValueError("Seed point is not connected to any sea core (label=0)")

    sea_core = (labeled == seed_label).astype(int)
    return sea_core


def land_core_from_sea(z, sea_core_mask):
    """
    Identify land areas adjacent to the connected sea core.

    Parameters
    ----------
    z : ndarray
        2D array of elevation values.
    sea_core_mask : ndarray
        Binary mask (2D array) of the connected sea area.

    Returns
    -------
    land_core : ndarray
        Binary mask (2D array) of the connected land areas.
    """
    land_core = ((z > 0) & (sea_core_mask == 0)).astype(int)
    return land_core


def extract_raster_coordinates(elevation, transform, crs):
    """
    Extract X, Y coordinates from raster grid cells.

    Parameters
    ----------
    elevation : ndarray
        2D array of elevation values.
    transform : Affine
        Affine transformation for pixel-to-world coordinates.
    crs : CRS
        Coordinate reference system of the raster.

    Returns
    -------
    x : ndarray
        Array of X coordinates in projected CRS.
    y : ndarray
        Array of Y coordinates in projected CRS.
    crs : CRS
        Coordinate reference system (possibly updated if projection applied).
    """
    cols, rows = np.meshgrid(np.arange(elevation.shape[1]), np.arange(elevation.shape[0]))
    x, y = rasterio.transform.xy(transform, rows, cols, offset='center')
    x = np.array(x)
    y = np.array(y)

    if not crs.is_projected:
        transformer = Transformer.from_crs(crs, crs.to_utm(), always_xy=True)
        x, y = transformer.transform(x, y)

    print("Coordinates extracted in UTM")
    return np.array(x), np.array(y), crs


def convert_wl_coordinates(wl_array, crs_raster):
    """
    Convert water level observation coordinates from geographic (lat/lon) to raster CRS.

    Parameters
    ----------
    wl_array : ndarray
        Array of shape (n, 3) where each row is (longitude, latitude, value).
    crs_raster : CRS
        Target raster coordinate reference system.

    Returns
    -------
    wl_x : ndarray
        Array of transformed X coordinates.
    wl_y : ndarray
        Array of transformed Y coordinates.
    wl_values : ndarray
        Original water level values.
    """
    wl_lon = wl_array[:, 0]
    wl_lat = wl_array[:, 1]
    wl_values = wl_array[:, 2]

    transformer = Transformer.from_crs("EPSG:4326", crs_raster, always_xy=True)
    wl_x, wl_y = transformer.transform(wl_lon, wl_lat)

    print(f"Transformed {len(wl_x)} water level coordinates to raster CRS.")
    return np.array(wl_x), np.array(wl_y), wl_values


def save_aligned_stations_to_shapefile(aligned_stations, output_shapefile, crs_epsg):
    """
    Save aligned station data to a shapefile.

    Parameters
    ----------
    aligned_stations : ndarray
        Array of shape (n, 4) with columns [X, Y, Water Level, DEM Elevation].
    output_shapefile : str
        Path to save the output shapefile.
    crs_epsg : int
        EPSG code for the output coordinate system.

    Returns
    -------
    None
    """
    gdf = gpd.GeoDataFrame({
        'X_UTM': aligned_stations[:, 0],
        'Y_UTM': aligned_stations[:, 1],
        'Water_Level_m': aligned_stations[:, 2],
        'DEM_Elevation_m': aligned_stations[:, 3],
        'geometry': [Point(xy) for xy in zip(aligned_stations[:, 0], aligned_stations[:, 1])]
    })

    gdf.set_crs(epsg=crs_epsg, inplace=True)
    gdf.to_file(output_shapefile)
    print(f"Saved aligned water stations to {output_shapefile}")





