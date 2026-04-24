import xarray as xr 
from rasterio.transform import xy as xy_transform
from rasterio.warp import transform as warp_transform

import numpy as np


def vectorize(transform, shape, source_crs):
    """
    Produce longtitude and latitute arrays by vectorizing the given transformation.
    """

    print("vectorizing...")
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

    latitudes, longitudes = warp_transform(source_crs, lat_long_coord_format, x.ravel(), y.ravel())
    latitudes = np.array(latitudes).reshape(shape)
    longitudes = np.array(longitudes).reshape(shape)

    print("vectorize complete")

    return np.array(latitudes), np.array(longitudes)

def bin_data(lats, lons, era5_lats, era5_lons):
    """
    Place each pixel (latitude + longitude) within their respective bins, as defined by the ERA5 data.
    """

    print("binning...")
    x_inds = np.digitize(lats, era5_lats)
    y_inds = np.digitize(lons, era5_lons)

    print("binning completed...")

    return x_inds, y_inds


def aggregate(sentinel1_data, sentinel2_data, era5_data, source_crs):
    """
    Aggregate Sentinel-1 data, Sentinel-2 data, and ERA5 data using a tiling strategy. 
    """

    # retrieve bands and other useful data from sentinel scenes
    sentinel1_bands = sentinel1_data[0]
    vh_transform = sentinel1_data[1]
    vh_shape = sentinel1_data[2]
    print(vh_shape)
    
    sentinel2_bands = sentinel2_data[0]
    nir_transform = sentinel2_data[1]
    nir_shape = sentinel2_data[2]
    print(nir_shape)

    era5_latitudes = era5_data['u10']['latitude']
    era5_longitudes = era5_data['u10']['longitude']

    # vectorize and bin each pixel according to the shape of each separate scene 
    sentinel1_lats, sentinel1_lons = vectorize(vh_transform, vh_shape, source_crs)
    sentinel2_lats, sentinel2_lons = vectorize(nir_transform, nir_shape, source_crs)
    print("both vectorizes completed.")
    x_indices_s1, y_indices_s1 = bin_data(sentinel1_lats, sentinel1_lons, era5_latitudes, era5_longitudes)
    x_indices_s2, y_indices_s2 = bin_data(sentinel2_lats, sentinel2_lons, era5_latitudes, era5_longitudes)

    print("tiling...")
    # create each tile with aggregated sentinel data 
    tiles = {}

    # combine unique indices from both scenes
    all_i = np.unique(np.concatenate([x_indices_s2.ravel(), x_indices_s1.ravel()]))
    all_j = np.unique(np.concatenate([y_indices_s2.ravel(), y_indices_s1.ravel()]))

    for i in all_i:
        for j in all_j:
            # isolate the pixels within each bin using a boolean mask
            mask_s1 = (x_indices_s1 == i) & (y_indices_s1 == j)

            # aggregate s1 data
            s1_statistics = {}
            for band_name, band_array in sentinel1_bands.items():
                s1_statistics[band_name] = {
                    "mean": np.mean(band_array[mask_s1]),
                    "std": np.std(band_array[mask_s1])
                } 
            
            mask_s2 = (x_indices_s2 == i) & (y_indices_s2 == j)
            # aggregate s2 data 
            s2_statistics = {}
            for band_name, band_array in sentinel2_bands["raw_bands"].items():
                s2_statistics[band_name] = {
                    "mean": np.mean(band_array[mask_s2]),
                    "std": np.std(band_array[mask_s2])
                } 
            for index_name, index_array in sentinel2_bands["indices"].items():
                s2_statistics[index_name] = {
                    "mean": np.mean(index_array[mask_s2]),
                    "std": np.std(index_array[mask_s2])
                } 

            # clamp the indices before the ERA5 lookup
            lat_idx = min(i, len(era5_latitudes) - 1)
            lon_idx = min(j, len(era5_longitudes) - 1)

            # extract era5 data at specific latitudes and longitudes
            lat = era5_latitudes[lat_idx]
            lon = era5_longitudes[lon_idx]

            era5_statistics = {
                "u10": era5_data["u10"].sel(latitude=lat, longitude=lon, method="nearest").values,
                "v10": era5_data["v10"].sel(latitude=lat, longitude=lon, method="nearest").values,
                "d2m": era5_data["d2m"].sel(latitude=lat, longitude=lon, method="nearest").values,
                "t2m": era5_data["t2m"].sel(latitude=lat, longitude=lon, method="nearest").values,
                "tp": era5_data["tp"].sel(latitude=lat, longitude=lon, method="nearest").values
            }

            # combine all the data into each tile
            tiles[(i, j)] = {
                "s1_stats": s1_statistics, 
                "s2_stats": s2_statistics,
                "era5_stats": era5_statistics
            }

    print("tiling completed")

    return tiles


