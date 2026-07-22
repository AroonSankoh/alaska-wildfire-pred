import os
import requests
import numpy as np 
import rasterio
import xml.etree.ElementTree as ET
from utils.geo_utils import vectorize, orthorectify, meters_per_degree, dem_crs
from rasterio.merge import merge 
from rasterio.control import GroundControlPoint
from scipy.interpolate import griddata, make_interp_spline
from tqdm import tqdm

def load_sar_band(sar_file_path, calibration_path, downsample_factor = 10):
    """
    Load GeoTIFF bands and convert raw digital numbers to decibels for normalization.
    """
    with rasterio.open(sar_file_path) as dataset:
        if dataset.count > 1:
             raise ValueError("The band must be single channel, but the dataset has multiple bands.")
        
        native_height, native_width = dataset.height, dataset.width 
        out_height = native_height // downsample_factor
        out_width = native_width // downsample_factor

        # read a downsampled view of the array (called a decimated array)
        data = dataset.read(
            1, out_shape=(out_height, out_width), resampling=rasterio.enums.Resampling.average
        ).astype(np.float64)
        gcps, gcp_crs = dataset.gcps

        # Calibrate to sigma-naught before orthorectification
        data = calibrate_to_sigma(data, calibration_path, native_height, native_width, downsample_factor)

        if gcps: 
            # rescale row/col to match the decimated array's coord space before orthorectification
            scaled_gcps = [
                GroundControlPoint(row = g.row / downsample_factor, col = g.col / downsample_factor, 
                                   x = g.x, y = g.y, z = g.z, id = g.id)
                for g in gcps
            ]
            data, transform, crs = orthorectify(data, scaled_gcps, gcp_crs)
        else: 
            transform = dataset.transform * dataset.transform.scale(downsample_factor, downsample_factor)
            crs = dataset.crs
        
    return data, transform, crs

def parse_calibration_lut(calibration_path, field="sigmaNought"):
    """
    Parse the sparse (line, pixel) calibration from a Sentinel-1 calibration xml annotation file.
    """
    tree = ET.parse(calibration_path)
    root = tree.getroot()
    vectors = root.findall(".//calibrationVector")

    # check that the calibration file has at least two vectors for bilinear interpolation
    if len(vectors) < 2:
        raise ValueError(
            f"Calibration file {calibration_path} has only {len(vectors)} calibration vectors."
        )

    lines = np.array([int(v.find("line").text) for v in vectors])
    pixel_lists = [v.find("pixel").text.split() for v in vectors]
    pixels = np.array([int(p) for p in pixel_lists[0]])

    if not all(pl == pixel_lists[0] for pl in pixel_lists):
        raise ValueError(f"Calibration pixel grid is not unfiform across lines in {calibration_path}")
    
    values = np.array([
        [float(x) for x in v.find(field).text.split()]
        for v in vectors
    ])
    return lines, pixels, values 

def interpolate_calibration_lut(lines, pixels, values, native_shape, downsample_factor):
    """
    Interpolate the sparse calibration LUT onto the downsampled pixel grid using bilinear interpolation.
    """
    native_n_rows, native_n_cols = native_shape
    out_n_rows = native_n_rows // downsample_factor
    out_n_cols = native_n_cols // downsample_factor

    # interpolate along the pixel (column) axis, once per sparse
    col_targets = np.arange(out_n_cols) * downsample_factor
    dense_cols = np.empty((len(lines), out_n_cols), dtype=np.float64)
    for i in range(len(lines)):
        dense_cols[i] = np.interp(col_targets, pixels, values[i])
        
    # interpolate along the line (row) axis, vectorized across
    row_targets = np.arange(out_n_rows) * downsample_factor
    row_interpolator = make_interp_spline(lines, dense_cols, k=1, axis=0)
    return row_interpolator(row_targets)

def calibrate_to_sigma(dn_data, calibration_path, native_height, native_width, downsample_factor):
    """
    Converts raw digital numbers to calibrated sigma-naught using the Sentinel-1 calibration LUT (DN^2 / A^2),
    sampled at the same downsampled grid as dn_data.
    """
    lines, pixels, sigma_naught_lut = parse_calibration_lut(calibration_path, field="sigmaNought")
    a = interpolate_calibration_lut(lines, pixels, sigma_naught_lut, (native_height, native_width), downsample_factor)
    if np.any(a <= 0):
        n_malformed = int((a <= 0).sum())
        print(f"WARNING: {n_malformed}/{a.size} calibration values are less than 0 in {calibration_path}. "
              "The resulting sigma-naught will contain NaN at those pixels.", flush=True)
    return (dn_data.astype(np.float64) ** 2) / (a ** 2)

def apply_rtc(vh_band, vv_band, transform, target_crs, output_dir, xml_annotation_path, threshold=0.05):
     """
     Applies RTC (Radiometric Terrain Correction) to align SAR data with a Digital Elevation Model (DEM).
     """

     # prepare dem tiles for normalization
     min_long, min_lat, max_long, max_lat = get_scene_bounds_lat_long(transform, vh_band.shape, target_crs)
     queries = get_dem_tile_coords(min_long, min_lat, max_long, max_lat)
     tile_file_paths = download_dem_tiles(queries, output_dir)
     dem = prepare_dem(tile_file_paths, transform, vh_band.shape, target_crs)

     # calculate the slope and aspect
     dy, dx = np.gradient(dem)
     center_lat = transform.f
     meters_per_deg_lat, meters_per_deg_long = meters_per_degree(center_lat)
     dx_real = dx / (transform.a * meters_per_deg_long) 
     dy_real = dy / (abs(transform.e) * meters_per_deg_lat)
     nx = -dx_real
     ny = -dy_real
     nz = np.ones_like(dem)  

     # calculate the surface normal vector
     magnitude = np.sqrt(nx**2 + ny**2 + nz**2)
     normal_nx = nx / magnitude 
     # normal_ny = ny / magnitude
     normal_nz = nz / magnitude

     # retrieve incidence angles and interpolate to a full image grid 
     lines, pixels, angles = parse_incidence_angles(xml_annotation_path)
     interpolated_angles = interpolate_incidence_angles(lines, pixels, angles, vh_band.shape)
     
     # retrieve radar look vector from incidence angle
     incidence_angle_radians = np.deg2rad(interpolated_angles) 
     look_x = np.sin(incidence_angle_radians)
     look_z = np.cos(incidence_angle_radians)

     # calculate cos of theta-local
     cos_local = normal_nx * look_x + normal_nz * look_z

     # mask layover and shadow pixels before division
     valid = cos_local > threshold
     vh_rtc = np.where(valid, vh_band / cos_local, np.nan)
     vv_rtc = np.where(valid, vv_band / cos_local, np.nan)


     print("RTC complete.")
     print(f"Masked pixels: {(~valid).sum()} / {valid.size}")

     return {"vh_band": vh_rtc, "vv_band": vv_rtc}

def get_scene_bounds_lat_long(transform, shape, source_crs):
     """
     Given a rasterio transform, shape, and CRS, return the bounding box in EPSG:4326 format.
     """
     longs, lats = vectorize(transform, shape, source_crs)
     return np.min(longs), np.min(lats), np.max(longs), np.max(lats)

def get_dem_tile_coords(min_long, min_lat, max_long, max_lat):
     """
     Returns list (lat, long) integer tile corners needed to cover the bounding box.
     """

     # calculate all lat, long combinations
     lats = np.arange(np.floor(min_lat), np.ceil(max_lat))
     longs = np.arange(np.floor(min_long), np.ceil(max_long))

     # list of queries to make to the Copernicus GLO-30 on AWS
     aws_queries = []
     for lat in lats:
         if lat >= 0:
             lat_str = f"N{int(lat)}"
         else:
             lat_str = f"S{int(np.abs(lat))}"
         for long in longs:  
             if long >= 0:
                 long_str = f"E{int(long)}"
             else:
                 long_str = f"W{int(np.abs(long))}"
             tile_name = f"Copernicus_DSM_COG_10_{lat_str}_00_{long_str}_00_DEM"
             aws_queries.append((
                  tile_name, 
                  f"https://copernicus-dem-30m.s3.amazonaws.com/{tile_name}/{tile_name}.tif"
             ))

     print(f"DEM tile grid: {len(lats)} lat rows x {len(longs)} long cols = {len(aws_queries)} tiles requested", flush=True)

     print("DEM tile coordinates retrieved.")

     return aws_queries
            
def download_dem_tiles(aws_queries, output_dir):
     """
     Download GLO-30 DEM tiles from the public S3 bucket using requests.
     """
     file_paths = []
     os.makedirs(output_dir, exist_ok=True)
     for query in tqdm(aws_queries, desc="Downloading DEM tiles."):
         filename = query[0]
         save_path = os.path.join(output_dir, filename + ".tiff")

         if os.path.isfile(save_path):
              print(f"The file {filename} already exists.")
         else: 
            # stream file downloads for memory efficiency
            with requests.get(query[1], stream=True, timeout=30) as response:
                response.raise_for_status()
                # open the local file for binary writing
                with open(save_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
         file_paths.append(save_path)

     return file_paths

def prepare_dem(tile_paths, target_transform, target_shape, target_crs):
     """
     Mosaic, reproject, and clip DEM tiles to match the S1 scene grid.
     """
     print("Preparing DEM for RTC computation.")

     # stitch each individual tile into a continuous raster
     sources = [rasterio.open(p) for p in tile_paths]
     mosaic, src_transform = merge(sources)
     for src in sources:
         src.close()
     
     # reproject the tiles into the source CRS and scale the resolution to match the S1 scene
     dest_array = np.empty(target_shape, dtype=np.float32)
     rasterio.warp.reproject(
          source=mosaic[0],
          destination=dest_array, 
          src_transform=src_transform, 
          src_crs=dem_crs,
          dst_transform=target_transform,
          dst_crs=target_crs,
          resampling=rasterio.warp.Resampling.bilinear
     )

     return dest_array

def parse_incidence_angles(xml_annotation_path):
     """
     Parse per-pixel incidence angles from the Sentinel-1 annotation XML files. 
     """
     tree = ET.parse(xml_annotation_path)
     root = tree.getroot()
    
     lines = []
     pixels = []
     angles = []

     # find all incidence angles and their line and pixels within the grid
     for point in root.findall('.//geolocationGridPoint'):
         lines.append(float(point.find('line').text))
         pixels.append(float(point.find('pixel').text))
         angles.append(float(point.find('incidenceAngle').text))

     return np.array(lines), np.array(pixels), np.array(angles)

def interpolate_incidence_angles(lines, pixels, angles, shape):
    """
    Interpolate sparse incidence angles to full image grid.
    """
    points = np.column_stack([lines, pixels])

    # retrieve every point in the image using a meshgrid
    rows = np.arange(shape[0])
    cols = np.arange(shape[1])
    row_grid, col_grid = np.meshgrid(rows, cols)

    # reshape to (N, 2) dimension
    target_points = np.column_stack([row_grid.ravel(), col_grid.ravel()])
    interpolated_angles = griddata(points, angles, target_points)

    return interpolated_angles.reshape(shape)

     
def load_sentinel1_bands(vh_path, vv_path, vh_cal_path, vv_cal_path, xml_annotation_path, output_dir, downsample_factor=10):
     """
     Loads all bands useful for fire prediction and analysis from a Sentinel-1 scene.
     """
     vh_band, vh_transform, src_crs = load_sar_band(vh_path, vh_cal_path, downsample_factor)
     vv_band, _, _= load_sar_band(vv_path, vv_cal_path, downsample_factor)

     # confirm vh and vv bands are the same shape
     if vh_band.shape != vv_band.shape:
         raise ValueError(
             f"VH band shape: {vh_band.shape} does not match VV band shape: {vv_band.shape}, these must align for RTC."
         )

     bands = apply_rtc(vh_band, vv_band, vh_transform, src_crs, output_dir, xml_annotation_path)

     # decibel conversion *post RTC
     vh_data = bands["vh_band"]
     vh_db_conv = np.full(vh_data.shape, np.nan)
     np.multiply(10, np.log10(vh_data, out=vh_db_conv, where=vh_data>0), out=vh_db_conv)
     
     vv_data = bands["vv_band"]
     vv_db_conv = np.full(vv_data.shape, np.nan)
     np.multiply(10, np.log10(vv_data, out=vv_db_conv, where=vv_data>0), out=vv_db_conv)

     bands = {"vh_band": vh_db_conv, "vv_band": vv_db_conv}

     return bands, vh_transform, vh_db_conv.shape, src_crs


