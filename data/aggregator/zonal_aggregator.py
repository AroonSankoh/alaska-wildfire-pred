from utils.geo_utils import vectorize
import numpy as np
import pandas as pd
from tqdm import tqdm

def bin_data(lats, longs, era5_lats, era5_lons):
    """
    Place each pixel (latitude + longitude) within their respective bins, as defined by the ERA5 data.
    """
    x_inds = np.digitize(lats, era5_lats) - 1
    y_inds = np.digitize(longs, era5_lons) - 1

    return x_inds, y_inds

def nullify_nan(stats):
     """
     Check if pixels for Sentinel-1 or Sentinel-2 were masked, if so set their dict to None.
     """
     if stats is None or all(np.isnan(v) for v in stats.values()):
         return None
     else: 
         return stats
                
def aggregate(sentinel1_data, sentinel2_data, era5_data):
    """
    Aggregate Sentinel-1 data, Sentinel-2 data, and ERA5 data using a tiling strategy. 
    """

    # retrieve bands and other useful data from sentinel scenes
    sentinel1_bands = sentinel1_data[0]
    vh_transform = sentinel1_data[1]
    vh_shape = sentinel1_data[2]
    s1_crs = sentinel1_data[3]
    
    sentinel2_bands = sentinel2_data[0]
    nir_transform = sentinel2_data[1]
    nir_shape = sentinel2_data[2]
    s2_crs = sentinel2_data[3]

    # ERA-5 lats/longs are sorted descending by default
    era5_sorted = {k: v.sortby('latitude') for k, v in era5_data.items()}
    era5_lats = np.sort(era5_data['u10']['latitude'])
    era5_longs = np.sort(era5_data['u10']['longitude'])
    
    # vectorize and bin each pixel according to the shape of each separate scene 
    sentinel1_longs, sentinel1_lats = vectorize(vh_transform, vh_shape, s1_crs)
    sentinel2_longs, sentinel2_lats = vectorize(nir_transform, nir_shape, s2_crs)

    # assert the bounding boxes of Sentinel-1 and Sentinel-2 fall within ERA5s
    if (np.min(era5_lats) > np.min(sentinel1_lats) or np.max(era5_lats) < np.max(sentinel1_lats)
        or np.min(era5_longs) > np.min(sentinel1_longs) or np.max(era5_longs) < np.max(sentinel1_longs)):
            raise ValueError("Sentinel-1 coordinate range falls outside ERA5 coordinate range.")
    if (np.min(era5_lats) > np.min(sentinel2_lats) or np.max(era5_lats) < np.max(sentinel2_lats)
        or np.min(era5_longs) > np.min(sentinel2_longs) or np.max(era5_longs) < np.max(sentinel2_longs)):
            raise ValueError("Sentinel-2 coordinate range falls outside ERA5 coordinate range.")
    
    x_indices_s1, y_indices_s1 = bin_data(sentinel1_lats, sentinel1_longs, era5_lats, era5_longs)
    x_indices_s2, y_indices_s2 = bin_data(sentinel2_lats, sentinel2_longs, era5_lats, era5_longs)

    print("Sentinel-1 and Sentinel-2 pixels vectorized and binned.")

    # construct respective dataframes for Sentinel-1 and Sentinel-2 with the vectorized pixels
    s1_df = pd.DataFrame({'x': x_indices_s1.ravel(), 'y': y_indices_s1.ravel(), 
                          **{k: v.ravel() for k, v in sentinel1_bands.items()}})
    s1_stats = s1_df.groupby(['x', 'y']).agg(['mean', 'std'])

    s2_combined = {**sentinel2_bands["filtered_bands"], **sentinel2_bands["indices"]}
    s2_df = pd.DataFrame({'x': x_indices_s2.ravel(), 'y': y_indices_s2.ravel(), 
                          **{k: v.ravel() for k, v in s2_combined.items()}})
    s2_stats = s2_df.groupby(['x', 'y']).agg(['mean', 'std'])

    # aggregate into tiles
    tiles = {}
    all_unique_indices = set(s1_stats.index).union(set(s2_stats.index))
    
    s1_stats.columns = ['_'.join(col) for col in s1_stats.columns]
    s2_stats.columns = ['_'.join(col) for col in s2_stats.columns]

    for (i, j) in tqdm(all_unique_indices, desc="Tiling statistics from each source according to ERA5 grid boundaries."):
        if 0 <= i < len(era5_lats) and 0 <= j < len(era5_longs):
            tiles[(i, j)] = {
                "s1_stats": s1_stats.loc[(i, j)].to_dict() if (i, j) in s1_stats.index else None,
                "s2_stats": s2_stats.loc[(i, j)].to_dict() if (i, j) in s2_stats.index else None,
                "era5_stats": {
                    'u10': float(era5_sorted['u10'].isel(latitude=i, longitude=j).mean().item()),
                    'v10': float(era5_sorted['v10'].isel(latitude=i, longitude=j).mean().item()),
                    'd2m': float(era5_sorted['d2m'].isel(latitude=i, longitude=j).mean().item()),
                    't2m': float(era5_sorted['t2m'].isel(latitude=i, longitude=j).mean().item()),
                    'tp':  float(era5_sorted['tp'].isel(latitude=i, longitude=j).mean().item()),
                }
            }
            tiles[(i, j)]["s1_stats"] = nullify_nan(tiles[(i, j)]["s1_stats"])
            tiles[(i, j)]["s2_stats"] = nullify_nan(tiles[(i, j)]["s2_stats"])
    return tiles
