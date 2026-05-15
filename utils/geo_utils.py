import numpy as np
from rasterio.transform import xy as xy_transform
from rasterio.transform import from_gcps
from rasterio.warp import transform as warp_transform
from rasterio.warp import calculate_default_transform, reproject, Resampling

dem_crs = "EPSG:4326" # GLO-30 Copernicus DEM is always delivered in WGS84

def vectorize(transform, shape, source_crs):
    """
    Produce longtitude and latitute arrays by vectorizing the given transformation.
    """
    lat_long_coord_format = "EPSG:4326"

    # retrieve row and col indices from band shape
    rows = np.arange(shape[0])
    cols = np.arange(shape[1])

    # create a meshgrid of all pixel row/col indices
    row_grid, col_grid = np.meshgrid(rows, cols)

    # transform CRS data by extracting x, y from each col, row then mapping to lat/long  
    x, y = xy_transform(transform, row_grid.ravel(), col_grid.ravel())
    x = np.array(x).reshape(shape)
    y = np.array(y).reshape(shape)

    longitudes, latitudes = warp_transform(source_crs, lat_long_coord_format, x.ravel(), y.ravel())
    longitudes = np.array(longitudes).reshape(shape)
    latitudes = np.array(latitudes).reshape(shape)

    return np.array(longitudes), np.array(latitudes)

def orthorectify(data, gcps, gcp_crs):
    """
    Convert GCP-based georeferencing to a proper affine transform for a north-up raster.
    """

    # calculate the north-up affine transform from GCP geometry
    transform, width, height = calculate_default_transform (
        gcp_crs, gcp_crs, data.shape[1], data.shape[0], gcps=gcps
    )

    # resamples mappings between each pixel in the destination grid to corresponding pixel in the source grid
    destination = np.zeros((height, width), dtype=data.dtype)
    reproject(
        source=data, 
        destination=destination,
        src_transform=from_gcps(gcps),
        src_crs=gcp_crs,
        dst_transform=transform,
        dst_crs=gcp_crs,
        resampling=Resampling.bilinear
    )
    
    return destination, transform, gcp_crs

def meters_per_degree(lat):
    """
    Returns meters per degree latitude and meters per degree longitude at a given latitude.
    """
    lat_rad = np.deg2rad(lat)

    # there are approximately 111,320 meters per degree latitude
    return 111320, 111320 * np.cos(lat_rad)



