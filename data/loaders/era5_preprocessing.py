import xarray as xr
import numpy as np
import cfgrib

def load_era5_vars(grib_path):
    """
    Loads and returns variables (wind speed/direction, humidity, temperature and precipiation), useful for wildfire pred.
    """
    datasets = cfgrib.open_datasets(grib_path)

    # total precipitation is organized with forecast initializations and lead times so must be flatted to match the dimensions
    total_precipitation = datasets[1]['tp'].stack(valid_time=('time', 'step')).dropna('valid_time')
    # note this leaves the total precipitation with 204 timesteps
    temp_and_pressure = datasets[0]
    u_wind_10m = temp_and_pressure['u10']
    v_wind_10m = temp_and_pressure['v10']
    dewpoint_2m_temp = temp_and_pressure['d2m']
    temp_2m = temp_and_pressure['t2m']

    variables = {'u10': u_wind_10m, 'v10': v_wind_10m, 'd2m': dewpoint_2m_temp, 't2m': temp_2m, 'tp': total_precipitation}

    return variables 

def calculate_fwi(t2m, d2m, u10, v10, tp):
    """
    Calculate the Fire Weather Index (FWI) from ERA5 variables.
    TODO: Implement full FWI calculation, index to index (FFMC, DMC, DC, ISI, BUI -> FWI)
    """
    raise NotImplementedError("FWI calculation not yet implemented.")
     



