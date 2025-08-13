import numpy as np
import pandas as pd
import time
import rasterio
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
from skimage.feature import canny
from skimage.morphology import binary_closing
from skimage.measure import label
from joblib import Parallel, delayed

# Import constants from config.py
from config import DEM_NULL_VALUE, CHUNK_SIZE_DISTANCE

# ----------------------------------------------------------
# NEW: Load perimeter-raster pixels as XYZ points
# ----------------------------------------------------------
def perimeter_raster_to_xyz(perim_tif, dem_lon, dem_lat, z_dem, nodata_val=0):
    """
    Convert a binary perimeter raster to XYZ arrays.

    Parameters
    ----------
    perim_tif : str
        Path to 02_water_body_perimeter.tif (binary, 1 = perimeter).
    dem_lon, dem_lat, z_dem : ndarray
        DEM longitude, latitude, and elevation grids (same shape as perimeter raster).
    nodata_val : int, optional
        Value representing NoData in the perimeter raster (default 0).

    Returns
    -------
    water_body_perimeter_xyz : ndarray
        Array [[X, Y, Z], …] for every perimeter pixel.
    water_body_perimeter_pixel_indices : ndarray
        Array [[row, col], …] of pixel indices.
    """
    with rasterio.open(perim_tif) as src:
        perim = src.read(1)

    rows, cols = np.where(perim != nodata_val)
    water_body_perimeter_xyz = np.column_stack((
        dem_lon[rows, cols],
        dem_lat[rows, cols],
        z_dem[rows, cols]
    ))
    water_body_perimeter_pixel_indices = np.column_stack((rows, cols))
    return water_body_perimeter_xyz, water_body_perimeter_pixel_indices


def load_seed_point(csv_file):
    """
    Load the seed point coordinates and elevation from a CSV file.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing seed point information.

    Returns
    -------
    seed_coords : tuple
        Tuple (x_seed, y_seed) in UTM coordinates.
    elevation_seed : float
        Elevation value at the seed point (meters).
    """
    df = pd.read_csv(csv_file)

    x_seed = df["X_UTM"].iloc[0]
    y_seed = df["Y_UTM"].iloc[0]
    elevation_seed = df["Elevation_m"].iloc[0]

    print(f"Loaded seed point: ({x_seed:.2f}, {y_seed:.2f}), Elevation: {elevation_seed:.2f} m")

    return (x_seed, y_seed), elevation_seed


def save_water_body_perimeter_as_shapefile(water_body_perimeter_xyz, output_shapefile, epsg_code):
    """
    Save extracted water body perimeter points to a Shapefile.

    Parameters
    ----------
    water_body_perimeter_xyz : ndarray
        Array of water body perimeter points (X, Y, Elevation).
    output_shapefile : str
        Path to the output Shapefile.
    epsg_code : int
        EPSG code for the coordinate system.

    Returns
    -------
    None
    """
    df = pd.DataFrame({
        "UTM_X": water_body_perimeter_xyz[:, 0],
        "UTM_Y": water_body_perimeter_xyz[:, 1],
        "Elevation_m": water_body_perimeter_xyz[:, 2]
    })

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.UTM_X, df.UTM_Y))
    gdf.set_crs(f"EPSG:{epsg_code}", inplace=True)
    gdf.to_file(output_shapefile)

    print(f"Water body perimeter shapefile saved as {output_shapefile}")


def compute_distances_chunk(start, end, dem_points, tree):
    """
    Compute distances for a subset of DEM points (helper for parallelization).

    Parameters
    ----------
    start : int
        Start index of the chunk.
    end : int
        End index of the chunk.
    dem_points : ndarray
        Array of DEM points (X, Y).
    tree : cKDTree
        KDTree built from water body perimeter points.

    Returns
    -------
    distances : ndarray
        Array of distances for the chunk.
    """
    return tree.query(dem_points[start:end], workers=-1)[0]


def calc_distance_to_water_body_parallel(dem_lon, dem_lat, z_dem, water_body_points, chunk_size=CHUNK_SIZE_DISTANCE):
    """
    Compute shortest distances from DEM points to nearest water body points using parallel processing.

    Parameters
    ----------
    dem_lon : ndarray
        2D array of DEM longitude coordinates.
    dem_lat : ndarray
        2D array of DEM latitude coordinates.
    z_dem : ndarray
        2D array of DEM elevation values.
    water_body_points : ndarray
        Array of water body points (X, Y).
    chunk_size : int, optional
        Number of DEM points per processing chunk. Default is imported from config.

    Returns
    -------
    distances : ndarray
        1D array of distances, same length as flattened DEM.
    """
    start_time = time.time()

    lon_flat = dem_lon.ravel()
    lat_flat = dem_lat.ravel()
    z_flat = z_dem.ravel()

    valid_mask = ~np.isnan(z_flat) & (z_flat > -0.01)
    valid_lon = lon_flat[valid_mask]
    valid_lat = lat_flat[valid_mask]

    print(f"Processing {len(valid_lon):,} valid land points in parallel...")

    dem_points = np.column_stack((valid_lon, valid_lat))
    tree = cKDTree(water_body_points)

    num_chunks = (len(valid_lon) + chunk_size - 1) // chunk_size
    chunk_indices = [(i * chunk_size, min((i + 1) * chunk_size, len(valid_lon))) for i in range(num_chunks)]

    distances_valid_chunks = Parallel(n_jobs=-1, backend="loky")(
        delayed(compute_distances_chunk)(start, end, dem_points, tree) for start, end in chunk_indices
    )
    distances_valid = np.concatenate(distances_valid_chunks)

    distances = np.full_like(z_flat, np.nan, dtype=np.float32)
    distances[valid_mask] = distances_valid

    end_time = time.time()
    print(f"Distance computation completed in {end_time - start_time:.2f} seconds.")

    return distances


def save_distance_raster(distances, dem_file, output_filename):
    """
    Save computed distances to a GeoTIFF raster file.

    Parameters
    ----------
    distances : ndarray
        1D array of distances (meters).
    dem_file : str
        Path to the reference DEM raster.
    output_filename : str
        Path to save the output distance raster.

    Returns
    -------
    None
    """
    with rasterio.open(dem_file) as src:
        raster_profile = src.profile.copy()

    raster_profile.update(
        dtype=rasterio.float32,
        count=1,
        nodata=DEM_NULL_VALUE
    )

    distances_clean = np.where(np.isnan(distances), DEM_NULL_VALUE, distances).astype(np.float32)

    with rasterio.open(output_filename, 'w', **raster_profile) as dst:
        dst.write(distances_clean, 1)

    print(f"Distance raster saved at: {output_filename}")


def align_hwm_to_dem(shapefile_path, dem_lon, dem_lat, z_dem, crs, output_shapefile):
    """
    Align High-Water Mark (HWM) points to nearest DEM locations and save the aligned points.

    Parameters
    ----------
    shapefile_path : str
        Path to the input HWM shapefile.
    dem_lon : ndarray
        2D array of DEM longitude coordinates.
    dem_lat : ndarray
        2D array of DEM latitude coordinates.
    z_dem : ndarray
        2D array of DEM elevation values.
    crs : CRS
        Coordinate reference system of the DEM.
    output_shapefile : str
        Path to save the aligned HWM shapefile.

    Returns
    -------
    gdf_aligned : GeoDataFrame
        GeoDataFrame of aligned HWM points with DEM elevations and pixel indices.
    hwm_indices : list of tuple
        List of (row, column) indices for aligned HWM points.
    """
    gdf = gpd.read_file(shapefile_path)

    # Reproject if necessary
    if gdf.crs != crs:
        print("Reprojecting HWM shapefile to match DEM CRS...")
        gdf = gdf.to_crs(crs)

    hwm_x = gdf.geometry.x
    hwm_y = gdf.geometry.y
    hwm_values = gdf["Elevatio_1"]  # Adjust if field name changes

    dem_shape = z_dem.shape
    points = np.column_stack((dem_lon.flatten(), dem_lat.flatten()))
    tree = cKDTree(points)

    aligned_hwm = []
    hwm_indices = []

    for i in range(len(hwm_x)):
        orig_x, orig_y, hwm_value = hwm_x[i], hwm_y[i], hwm_values[i]
        _, idx = tree.query([orig_x, orig_y])
        row, col = np.unravel_index(idx, dem_shape)
        snapped_x = dem_lon[row, col]
        snapped_y = dem_lat[row, col]
        aux_value = z_dem[row, col]
        aligned_hwm.append([snapped_x, snapped_y, hwm_value, aux_value, row, col])
        hwm_indices.append((row, col))

    gdf_aligned = gpd.GeoDataFrame(
        aligned_hwm,
        columns=["lon_dem", "lat_dem", "hwm", "z_dem", "row", "col"],
        geometry=[Point(lon, lat) for lon, lat, *_ in aligned_hwm]
    )

    gdf_aligned.set_crs(crs, inplace=True)
    gdf_aligned.to_file(output_shapefile)

    print(f"HWM aligned shapefile saved to: {output_shapefile}")

    return gdf_aligned, hwm_indices


