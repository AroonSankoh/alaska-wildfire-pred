import os
import requests
import numpy as np 
import rasterio
import xml.etree.ElementTree as ET
from utils.geo_utils import vectorize, orthorectify, dem_crs
from rasterio.merge import merge 
from scipy.interpolate import griddata
from tqdm import tqdm

def load_sar_band(file_path):
    """
    Load GeoTIFF bands and convert raw digital numbers to decibels for normalization.
    """
    with rasterio.open(file_path) as dataset:
        if dataset.count > 1:
             raise ValueError("The band must be single channel, but the dataset has multiple bands.")
        data = dataset.read(1).astype(np.float64)
        gcps, gcp_crs = dataset.gcps

        if gcps: 
            data, transform, crs = orthorectify(data, gcps, gcp_crs)
        else: 
            transform = dataset.transform
            crs = dataset.crs
        
        # downsample for testing
        data = data[::10, ::10]
        transform = transform * transform.scale(10, 10)
        
    return data, transform, crs


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
     dx_real = dx / transform.a
     dy_real = dy / abs(transform.e)
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
     valid = np.abs(cos_local) > threshold
     vh_rtc = np.where(valid, vh_band / cos_local, np.nan)
     vv_rtc = np.where(valid, vv_band / cos_local, np.nan)

     print(f"masked pixels: {(~valid).sum()} / {valid.size}")

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
     print(f"Bounds: {min_long}, {min_lat}, {max_long}, {max_lat}")

     print("grabbing dem tile coords")

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

     print("tile coords retrieved")

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
              print(f"File {filename} already exists.")
         else: 
            # stream file downloads for memory efficiency
            with requests.get(query[1], stream=True) as response:
                # raise an exception for 4xx or 5xx errors
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

     print("preparing dem...")

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

     print("dem prepared.")
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

     
def load_sentinel1_bands(vh_path, vv_path, xml_annotation_path, output_dir):
     """
     Loads all bands useful for fire prediction and analysis from a Sentinel-1 scene.
     """
     vh_band, vh_transform, src_crs = load_sar_band(vh_path)
     vv_band, _, _= load_sar_band(vv_path)

     bands = apply_rtc(vh_band, vv_band, vh_transform, src_crs, output_dir, xml_annotation_path)

     # decibel conversion *post RTC
     vh_data = bands["vh_band"]
     vh_db_conv = np.full(vh_data.shape, np.nan)
     np.multiply(10, np.log10(vh_data, out=vh_db_conv, where=vh_data>0), out=vh_db_conv)
     
     vv_data = bands["vv_band"]
     vv_db_conv = np.full(vv_data.shape, np.nan)
     np.multiply(10, np.log10(vv_data, out=vv_db_conv, where=vv_data>0), out=vv_db_conv)

     bands = {"vh_band": vh_db_conv, "vv_band": vv_db_conv}

     return bands, vh_transform, vh_band.shape


