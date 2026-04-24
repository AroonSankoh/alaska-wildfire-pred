import numpy as np 
import rasterio
import matplotlib.pyplot as plt

def load_sar_band(file_path):
    """
    Load GeoTIFF bands and convert raw digital numbers to decibels for normalization.
    """
    with rasterio.open(file_path) as dataset:
        if dataset.count > 1:
                raise ValueError("The band must be single channel, but the dataset has multiple bands.")
        data = dataset.read(1).astype(np.float64)

        # for testing
        # data = data[::10, ::10]

        transform = dataset.transform 
        

    db_conv = np.full(data.shape, np.nan)
    np.multiply(10, np.log10(data, out=db_conv, where=data>0), out=db_conv)


    return db_conv, transform

def apply_rtc(vh_band, vv_band, transform, dem_path):
     """
     Applies RTC (Radiometric Terrain Correction) to align SAR data with a Digital Elevation Model (DEM).
     TODO: Implement the full pipeline for this with Copernicus DEM.
     """
     raise NotImplementedError("RTC not yet implemented.")


def load_sentinel1_bands(vh_path, vv_path):
     """
     Loads all bands useful for fire prediction and analysis from a Sentinel-1 scene.
     """
     vh_band, vh_transform = load_sar_band(vh_path)
     vv_band, _ = load_sar_band(vv_path)
     bands = {"vh_band": vh_band, "vv_band": vv_band}

     return bands, vh_transform, vh_band.shape
