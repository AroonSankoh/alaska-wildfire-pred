import xarray as xr
import numpy as np
import cfgrib

def _find_variable(datasets, var_name):
    """
    Search every hypercube by name instead of trusting a fixed dataset index.
    """
    for ds in datasets:
        if var_name in ds.data_vars:
            return ds[var_name]
    raise ValueError(f"Could not find variable '{var_name}' in any hypercube of this grib.")


def _flatten_time_step(data_array):
    """
    Computes the actual physically-valid datetime (base time plus
    forecast lead time) and swaps that in as the 'valid_time' coordinate instead.
    """
    data_array = data_array.assign_coords(valid_datetime=data_array['time'] + data_array['step'])
    stacked = data_array.stack(valid_time=('time', 'step')).dropna('valid_time')
    stacked = stacked.reset_index('valid_time', drop=True).rename(valid_datetime='valid_time')
    return stacked


def load_era5_vars(grib_path, cutoff_datetime=None):
    """
    Loads and returns variables (wind speed/direction, humidity, temperature and precipiation), useful for wildfire pred.
    The cutoff datetime, if given, drops every timestep strictly after this datetime.
    """
    datasets = cfgrib.open_datasets(grib_path)

    variables = {}
    for var_name in ('u10', 'v10', 'd2m', 't2m', 'tp'):
        data_array = _find_variable(datasets, var_name)
        # some variables (always tp, occasionally others depending on how CDS bundled this
        # particular request) carry an unflattened time+step structure that needs flattening
        # onto a single real-valued time axis before it can be used
        if 'step' in data_array.dims and 'time' in data_array.dims:
            data_array = _flatten_time_step(data_array)
        variables[var_name] = data_array

    if cutoff_datetime is not None:
        cutoff = np.datetime64(cutoff_datetime)
        for key, data_array in variables.items():
            time_dim = 'valid_time' if 'valid_time' in data_array.dims else 'time'
            filtered = data_array.where(data_array[time_dim] < cutoff, drop=True)
            if filtered.sizes[time_dim] == 0:
                raise ValueError(
                    f"No ERA5 timesteps remain for '{key}' strictly before cutoff {cutoff_datetime}, meaning "
                    f"the downloaded grib may not actually cover the required antecedent window."
                )
            variables[key] = filtered

    return variables

def calculate_fwi(t2m, d2m, u10, v10, tp):
    """
    Calculate the Fire Weather Index (FWI) from ERA5 variables.
    TODO: Implement full FWI calculation, index to index (FFMC, DMC, DC, ISI, BUI -> FWI)
    """
    raise NotImplementedError("FWI calculation not yet implemented.")
     



