import numpy as np
import rasterio 
from rasterio.enums import Resampling

def load_band(file_path, resample_continuous=True, target_shape=None):
    """
    Load a band from a raster file and resample it to the target shape using bilinear interpolation.
    """
    with rasterio.open(file_path) as dataset:
        if dataset.count > 1:
                raise ValueError("The band must be single channel, but the dataset has multiple bands.")
        data = dataset.read(
            out_shape=(
                dataset.count,
                target_shape[0] if target_shape is not None else dataset.height,
                target_shape[1] if target_shape is not None else dataset.width,
            ),
            resampling=(Resampling.bilinear if resample_continuous is True else Resampling.nearest)
        )

        # scale image transform to match the new shape
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )
        band_array = data[0] # Error if the band is not single channel
        
    return band_array, transform

def calculate_nbr(nir_path, swir_path, scl_path, target_shape=None):
     """ 
     Calculate Normalized Burn Ratio (NBR) from NIR and SWIR images.
     Returns NIR and SWIR bands as well to minimize any ineffiency from loading bands multiple times. 
     """

     # retrieve bands with compatible resolutions
     nir_band, nir_transform = load_band(nir_path, target_shape=target_shape)
     swir_band, _ = load_band(swir_path, target_shape=nir_band.shape)
     nir_band = nir_band.astype(np.float64)
     swir_band = swir_band.astype(np.float64)
     filtered_nir = apply_cloud_mask(nir_band, scl_path, nir_band.shape)
     filtered_swir = apply_cloud_mask(swir_band, scl_path, swir_band.shape)

     # calculate nbr with safe division
     num = filtered_nir - filtered_swir
     den = filtered_nir + filtered_swir
     nbr = np.full(num.shape, np.nan)
     np.divide(num, den, out=nbr, where= den!=0)
     
     return nbr, filtered_nir, filtered_swir, nir_transform

def calculate_ndvi(nir_path, red_path, scl_path, target_shape=None):
    """
    Calculate Normalized Difference Vegetation Index from Red and NIR images. 
    Returns NIR and Red bands as well to minimize any ineffiency from loading bands multiple times.
    """

    # retrieve bands with compatible resolutions 
    nir_band, nir_transform = load_band(nir_path, target_shape=target_shape)
    red_band, _ = load_band(red_path, target_shape=nir_band.shape)
    nir_band = nir_band.astype(np.float64)
    red_band = red_band.astype(np.float64)
    filtered_nir = apply_cloud_mask(nir_band, scl_path, nir_band.shape)
    filtered_red = apply_cloud_mask(red_band, scl_path, red_band.shape)

    # calculate ndvi with safe division 
    num = filtered_nir - filtered_red
    den = filtered_nir + filtered_red 
    ndvi = np.full(num.shape, np.nan)
    np.divide(num, den, out=ndvi, where= den!=0)

    return ndvi, filtered_nir, filtered_red, nir_transform

def apply_cloud_mask(band, scl_path, shape):
     """
     Masks clouds and cloud shadows of an NBR file using the scene classification layer (SCL) file.
     """
     # unnecessary file analysis categories to mask out
     risk_categories = {
          "no_data": 0,
          "cloud_shadows": 3, 
          "med_prob_cloud": 8, 
          "high_prob_cloud": 9,
          "thin_cirrus":10
     }
     # load the scl_band to retrieve its shape
     scl, _ = load_band(scl_path, resample_continuous=False, target_shape=shape)

     mask = ((scl != risk_categories["no_data"])
             & (scl != risk_categories["cloud_shadows"])
             & (scl != risk_categories["med_prob_cloud"])
             & (scl != risk_categories["high_prob_cloud"])
             & (scl != risk_categories["thin_cirrus"]))

     filtered_band = np.where(mask, band, np.nan)

     return filtered_band

def load_sentinel2_bands(red_path, green_path, nir_path, swir_path, scl_path, target_shape=None):
    """ 
    Loads all bands useful for fire prediction and analysis from a Sentinel-2 scene.
    """
    # retrieve bands by indices, raw bands, and any transformations 
    nbr, nir_band, swir_band, nir_transform = calculate_nbr(nir_path, swir_path, scl_path, target_shape=target_shape)
    ndvi, _, red_band, _ = calculate_ndvi(nir_path, red_path, scl_path, target_shape=target_shape)
    green_band, _ = load_band(green_path, target_shape=nir_band.shape)
    green_band = apply_cloud_mask(green_band, scl_path, green_band.shape)
    bands = {
         "indices": {"ndvi": ndvi, "nbr": nbr}, 
         "filtered_bands": {"red": red_band, "green": green_band, "nir": nir_band, "swir": swir_band} 
    }

    # retrieve coordinate reference system from the NIR band 
    with rasterio.open(nir_path) as dataset: 
             source_crs = dataset.crs

    return bands, nir_transform, nir_band.shape, source_crs



















          
     