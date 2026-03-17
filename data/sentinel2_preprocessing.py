import numpy as np
import rasterio 
from rasterio.enums import Resampling

nir_shape = None

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

def calculate_nbr(nir_path, swir_path):
     """ 
     Calculate Normalized Burn Ratio (NBR) from NIR and SWIR images.
     """
     # retrieve bands with compatible resolutions
     nir_band, nir_transform = load_band(nir_path)
     swir_band, _ = load_band(swir_path, target_shape=nir_band.shape)
     nir_band = nir_band.astype(np.float64)
     swir_band = swir_band.astype(np.float64)

     # calculate nbr with safe division
     num = nir_band - swir_band
     den = nir_band + swir_band
     nbr = np.zeros(num.shape)
     np.divide(num, den, out=nbr, where= den!=0)
     
     return nbr, nir_transform, nir_shape  

def apply_cloud_mask(nir_path, swir_path, scl_path, shape):
     """
     Masks clouds and cloud shadows using the scene classification layer (SCL) file.
     """
     # Unnecessary file analysis categories to mask out
     risk_categories = {
          "cloud_shadows": 3, 
          "med_prob_cloud": 8, 
          "high_prob_cloud": 9,
          "thin_cirrus":10
     }
     # load the nir_band to retrieve its shape
     scl, _ = load_band(scl_path, resample_continuous=False, target_shape=shape)

     mask = ((scl != risk_categories["cloud_shadows"])
             & (scl != risk_categories["med_prob_cloud"])
             & (scl != risk_categories["high_prob_cloud"])
             & (scl != risk_categories["thin_cirrus"]))

     nbr, nir_transform, _ = calculate_nbr(nir_path, swir_path)
     filtered_nbr = np.where(mask, nbr, np.nan)

     return filtered_nbr, nir_transform












          
     