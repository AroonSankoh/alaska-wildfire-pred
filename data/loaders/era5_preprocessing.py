import xarray as xr
import numpy as np
import pandas as pd
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

def _time_dim_of(data_array):
    """
    Different ERA5 variables can end up with different time-dimension names after
    load_era5_vars, so resolve the variable representing per-array, not once globally. 
    """
    return 'valid_time' if 'valid_time' in data_array.dims else 'time'

def _approximate_noon_utc_hour(lon_center):
    """
    Offsets ERA5 timestamps from UTC to Local Standard Time (LST), the standard for FWI system inputs.
    """
    utc_offset_hours = round(lon_center / 15.0)
    return (12 - utc_offset_hours) % 24

def _grid_centroid(data_array):
    """
    Find the midpoint of the grib's own latitiude/longitude extent.
    """
    lat_center = float((data_array['latitude'].max() + data_array['latitude'].min()) / 2)
    lon_center = float((data_array['longitude'].max() + data_array['longitude'].min()) / 2)
    return lat_center, lon_center


def load_era5_vars(grib_path, cutoff_datetime=None):
    """
    Loads and returns variables (wind speed/direction, humidity, temperature and precipiation), useful for wildfire pred.
    The cutoff datetime, if given, drops every timestep strictly after this datetime.
    """
    datasets = cfgrib.open_datasets(grib_path, indexpath='')

    variables = {}
    for var_name in ('u10', 'v10', 'd2m', 't2m', 'tp'):
        data_array = _find_variable(datasets, var_name)
        # some variables (always tp) carry an unflattened time+step structure that needs flattening onto a single real-valued time axis 
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

def calculate_fwi(variables):
    """
    Recursively step FFMC/DMC/DC/ISI/BUI/FWI forward one day at a time across an entire ERA5
    hourly time series (as returned by load_era5_vars), starting from the standard startup
    codes and chaining each day's own codes into the next day's "yesterday" inputs.
    
    Equation numbers below are from Van Wagner & Pickett (1985), Technical Report 33.

    Parameters:
    `variables` is the dict returned by load_era5_vars: {'t2m': DataArray, 'd2m': DataArray,
    'u10': DataArray, 'v10': DataArray, 'tp': DataArray}, each indexed along an hourly
    'valid_time' (or 'time') dimension.

    Returns a dict with the last successfully computed day's FFMC, DMC, DC, ISI, BUI, and FWI,
    plus the date that final day corresponds to.
    """
    t2m_arr = variables['t2m']
    d2m_arr = variables['d2m']
    u10_arr = variables['u10']
    v10_arr = variables['v10']
    tp_arr = variables['tp']

    n_timesteps = t2m_arr.sizes[_time_dim_of(t2m_arr)]
    lat_center, lon_center = _grid_centroid(t2m_arr)
    noon_utc_hour = _approximate_noon_utc_hour(lon_center)

    # standard spring startup conditions (Van Wagner & Pickett 1985) that represent typical Canada/Alaska spring conditions
    ffmc_nought, dmc_nought, dc_nought = 85.0, 6.0, 15.0

    result = None
    # index 0 is midnight of day 1, so index 12 is noon, which the FWI calculator expects 
    noon_index = noon_utc_hour
    while noon_index < n_timesteps:
        t2m = float(t2m_arr.isel({_time_dim_of(t2m_arr): noon_index}).sel(
            latitude=lat_center, longitude=lon_center, method='nearest').values)
        d2m = float(d2m_arr.isel({_time_dim_of(d2m_arr): noon_index}).sel(
            latitude=lat_center, longitude=lon_center, method='nearest').values)
        u10 = float(u10_arr.isel({_time_dim_of(u10_arr): noon_index}).sel(
            latitude=lat_center, longitude=lon_center, method='nearest').values)
        v10 = float(v10_arr.isel({_time_dim_of(v10_arr): noon_index}).sel(
            latitude=lat_center, longitude=lon_center, method='nearest').values)
        tp = float(tp_arr.isel({_time_dim_of(tp_arr): noon_index}).sel(
            latitude=lat_center, longitude=lon_center, method='nearest').values)

        # month is extracted from the specific day's timestamp
        timestamp = pd.Timestamp(t2m_arr[_time_dim_of(t2m_arr)].isel({_time_dim_of(t2m_arr): noon_index}).values)
        month = timestamp.month

        # calculate today's fine fuel moisture content and initial spread index
        ffmc_today = calculate_ffmc(ffmc_nought, t2m, d2m, u10, v10, tp)
        isi_today = calculate_isi(ffmc_today, u10, v10)

        # calculate today's duff moisture code, drought code, and buildup index
        dmc_today = calculate_dmc(dmc_nought, t2m, d2m, tp, month)
        dc_today = calculate_dc(dc_nought, t2m, tp, month)
        bui_today = calculate_bui(dmc_today, dc_today)

        # Eq. 28a/28b: f(D), the buildup index's effect on the duff/heavier fuels' contribution to fire intensity
        if bui_today <= 80:
            f_d = 0.626 * np.power(bui_today, 0.809) + 2
        else:
            f_d = np.divide(1000, 25 + 108.64 * np.exp(-0.023 * bui_today))

        # Eq. 29: B is just an unnamed intermediate value 
        b = 0.1 * isi_today * f_d

        # Eq. 30a/30b: log-transform B onto the final FWI scale or leave when B <= 1, since the 
        # log transform would otherwise be negative or undefined there. 2.72, 0.434, 0.647 are fitted constants
        if b > 1:
            fwi_today = np.exp(2.72 * np.power(0.434 * np.log(b), 0.647))
        else:
            fwi_today = b

        result = {
            "date": timestamp,
            "ffmc": ffmc_today,
            "dmc": dmc_today,
            "dc": dc_today,
            "isi": isi_today,
            "bui": bui_today,
            "fwi": fwi_today,
        }

        # today's codes become tomorrow's "yesterday" inputs for the next recursive step
        ffmc_nought, dmc_nought, dc_nought = ffmc_today, dmc_today, dc_today
        noon_index += 24

    if result is None:
        raise ValueError(
            "Not enough hourly timesteps to compute a single day of FWI"
            f"(need at least 13 hours, got {n_timesteps})."
        )

    return result

def calculate_ffmc(f_nought, t2m, d2m, u10, v10, tp):
    """
    Recursively calculate today's Fine Fuel Moisture Code (F) from yesterday's (F0).
    Equation numbers below are from Van Wagner & Pickett (1985), Technical Report 33.
    """
    t_c = t2m - 273.15
    td_c = d2m - 273.15

    def calculate_m(f_nought):
        """
        Calculate m0 (moisture content) from yesterday's FFMC code -- Eq. 1.
        """
        # 250 * 59.5 / 101 = 147.277.., 101, 59.5 are the fitted constants converting the FF-scale code back to % moisture
        return np.divide((250 * 59.5 / 101) * (101 - f_nought), 59.5 + f_nought)

    m_nought = calculate_m(f_nought)

    # rainfall routine (Eq. 2, 3a/3b) must be skipped entirely when ro <= 0.5mm (p.12 restriction)
    r_nought = tp * 1000 # convert from meters to mm
    if r_nought > 0.5:
        r_final = r_nought - 0.5  # Eq. 2: rf = ro - 0.5
    else:
        r_final = 0
        m_r = m_nought

    if r_final > 0:
        # fine fuel moisture content after rain
        if m_nought <= 150:
            # Eq. 3a
            m_r = m_nought + 42.5 * r_final * (np.exp(np.divide(-100, 251 - m_nought))) * (1 - np.exp(np.divide(-6.93, r_final)))
        else:
            # Eq. 3b -- extra term (0.0015, 150 exponent) only applies above 150% moisture content
            m_r = m_nought + 42.5 * r_final * (np.exp(np.divide(-100, 251 - m_nought))) * (1 - np.exp(np.divide(-6.93, r_final))) \
            + 0.0015 * (m_nought - 150)**2 * np.power(r_final, 0.5)

    # m has an upper limit of 250 (p.12 restriction)
    m_r = np.minimum(m_r, 250)

    def calculate_relative_humidity(t_c, td_c):
        """
        Calculate relative humidity percentage from temperature and dewpoint.
        Derived from dewpoint and temp ERA5 variables.
        """
        # magnus-tetens constants for water vapour pressure
        a = 17.625
        b = 243.04

        # calculate actual vapour pressure (e) from dewpoint
        e = 6.11 * np.exp((a * td_c) / (b + td_c))

        # calculate saturation vapour pressure (es) from air temperature
        es = 6.11 * np.exp((a * t_c) / (b + t_c))

        # divide actual vapour pressure by saturation vapour pressure to get humidity percentage
        rh = (e / es) * 100

        # FFMC strictly caps relative humidity at 100%
        rh = np.clip(rh, 0.0, 100.0)
        return rh

    rh = calculate_relative_humidity(t_c, td_c)
        
    def calculate_dry_emc(rh):
        """
        Calculate the fine fuel equilibrium moisture content for drying (Eq. 4).
        """
        # 0.942, 0.679, 11, 0.18, 21.1, 0.115 are empirical curve-fit coefficients from Van Wagner's original regression
        return 0.942 * np.power(rh, 0.679) + 11 * np.exp(np.divide(rh - 100, 10)) + 0.18 * (21.1 - t_c) * (1 - np.exp(-0.115 * rh))

    def calculate_wet_emc(rh):
            """
            Calculate the fine fuel equilibrium moisture content for wetting (Eq. 5).
            """
            # same as calculate_dry_emc: 0.618, 0.753, 10, 0.18, 21.1, 0.115 are fitted constants
            return 0.618 * np.power(rh, 0.753) + 10 * np.exp( np.divide(rh - 100, 10)) + 0.18 * (21.1 - t_c) * (1 - np.exp(-0.115 * rh))

    dry_emc = calculate_dry_emc(rh)

    # calculate noon wind speed by converting era5 m/s vector components to scalar km/h
    wind_speed = np.sqrt(u10**2 + v10**2) * 3.6

    # branches mirror the paper's own procedure (p.12, steps 5-8): drying, wetting, or unchanged
    if m_r > dry_emc:
        # Eq. 6a: intermediate drying-rate term. 0.424, 1.7, 0.0694, 8 are fitted constants
        k_a = 0.424 * (1 - np.power(np.divide(rh, 100), 1.7)) + 0.0694 * np.power(wind_speed, 0.5) * (1 - np.power(np.divide(rh, 100), 8))
        # Eq. 6b: log drying rate. 0.581, 0.0365 are fitted constants
        k_dry = k_a * 0.581 * np.exp(0.0365 * t_c)

        # Eq. 8: final fine fuel moisture content after drying
        m = dry_emc + (m_r - dry_emc) * np.power(10, -k_dry)
    else:
        wet_emc = calculate_wet_emc(rh)
        if m_r < wet_emc:
            # Eq. 7a: intermediate wetting-rate term (same fitted constants as Eq. 6a)
            k_b = 0.424 * (1 - np.power(np.divide(100 - rh, 100), 1.7)) + 0.0694 * np.power(wind_speed, 0.5) * (1 - np.power(np.divide(100 - rh, 100), 8))
            # Eq. 7b: log wetting rate (same fitted constants as Eq. 6b)
            k_wet = k_b * 0.581 * np.exp(0.0365 * t_c)

            # Eq. 9: final fine fuel moisture content after wetting
            m = wet_emc - (wet_emc - m_r) * np.power(10, -k_wet)
        else:
            m = m_r # Ed >= m_r >= Ew: no change (p.12, step 8)

    # Eq. 10: convert moisture content back to the FFMC code. 59.5, 250, and 250 * 59.5 / 101 = 147.277.. are more fitted constants
    return 59.5 * np.divide(250 - m, (250 * 59.5 / 101) + m)


def calculate_dmc(p_nought, t2m, d2m, tp, month):
    """
    Recursively calculate today's Duff Moisture Code (P) from yesterday's (P0).
    Equation numbers below are from Van Wagner & Pickett (1985), Technical Report 33.
    """
    t_c = t2m - 273.15
    td_c = d2m - 273.15

    # DMC's low-temperature floor: T < -1.1 is treated as T = -1.1 in Eq. 16 (p.13 restriction)
    t_c_dmc = np.maximum(t_c, -1.1)

    def calculate_relative_humidity(t_c, td_c):
        """
        Same Magnus-Tetens derivation used in calculate_ffmc since ERA5 doesn't give RH directly.
        """
        a = 17.625
        b = 243.04
        e = 6.11 * np.exp((a * td_c) / (b + td_c))
        es = 6.11 * np.exp((a * t_c) / (b + t_c))
        rh = (e / es) * 100
        return np.clip(rh, 0.0, 100.0)

    rh = calculate_relative_humidity(t_c, td_c)

    # Table 1: effective day-length (Le) by month, for latitudes >= 30N, which covers Alaska.
    day_length_le = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
    le = day_length_le[month - 1]

    r_nought = tp * 1000  # convert from meters to mm

    # rainfall routine (Eq. 11, 12, 13a/b/c, 14, 15) is skipped entirely when ro <= 1.5mm
    if r_nought > 1.5:
        # Eq. 11: net rain amount that actually reaches the duff layer
        r_effective = 0.92 * r_nought - 1.27

        # Eq. 12: moisture content equivalent of yesterday's DMC code.
        m_nought = 20 + np.exp(5.6348 - np.divide(p_nought, 43.43))

        # Eq. 13a, 13b, 13c: b is a slope term whose form changes with how wet the duff already is
        if p_nought <= 33:
            b = np.divide(100, 0.5 + 0.3 * p_nought)
        elif p_nought <= 65:
            b = 14 - 1.3 * np.log(p_nought)
        else:
            b = 6.2 * np.log(p_nought) - 17.2

        # Eq. 14: moisture content after rain
        m_r = m_nought + np.divide(1000 * r_effective, 48.77 + b * r_effective)

        # Eq. 15: convert moisture content back onto the DMC scale
        p_r = 244.72 - 43.43 * np.log(m_r - 20)

        # Pr cannot be negative (p.13 restriction)
        p_r = np.maximum(p_r, 0)
    else:
        # no rainfall adjustment -- yesterday's code carries through unchanged
        p_r = p_nought

    # Eq. 16: log drying rate. 1.894, 1.1, 100 (as 1e-6, folded into Eq. 17 below) are fitted constants
    k = 1.894 * (t_c_dmc + 1.1) * (100 - rh) * le * 1e-6

    # Eq. 17: today's DMC. The paper's "100*K" term is applied here directly
    p = p_r + 100 * k
    # DMC has no upper bound but cannot be negative
    return np.maximum(p, 0)


def calculate_dc(d_nought, t2m, tp, month):
    """
    Recursively calculate today's Drought Code (D) from yesterday's (D0).
    Equation numbers below are from Van Wagner & Pickett (1985), Technical Report 33.
    """
    t_c = t2m - 273.15

    # DC's low-temperature floor: T < -2.8 is treated as T = -2.8 in Eq. 22 (p.14 restriction)
    t_c_dc = np.maximum(t_c, -2.8)

    # Table 2: day-length factor (Lf) by month, for latitudes >= 30N, which covers Alaska.
    day_length_lf = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]
    lf = day_length_lf[month - 1]

    r_nought = tp * 1000  # convert from meters to mm

    # rainfall routine (Eq. 18-21) is skipped entirely when ro <= 2.8mm (p.14 restriction)
    if r_nought > 2.8:
        # Eq. 18: effective rainfall reaching the deep duff/soil layer
        r_effective = 0.83 * r_nought - 1.27

        # Eq. 19: moisture-equivalent of yesterday's DC code. 800, 400 are fitted constants
        q_nought = 800 * np.exp(np.divide(-d_nought, 400))

        # Eq. 20: moisture equivalent after rain. 3.937 converts rd (mm) onto Q's scale
        q_r = q_nought + 3.937 * r_effective

        # Eq. 21: convert moisture equivalent back onto the DC scale
        d_r = 400 * np.log(np.divide(800, q_r))

        # Dr cannot be negative (p.14 restriction)
        d_r = np.maximum(d_r, 0)
    else:
        # no rainfall adjustment -- yesterday's code carries through unchanged
        d_r = d_nought

    # Eq. 22: potential evapotranspiration term. 0.36, 2.8 are fitted constants; Lf is the
    # Table 2 lookup for this month. V cannot be negative (p.14 restriction)
    v = np.maximum(0.36 * (t_c_dc + 2.8) + lf, 0)

    # Eq. 23: today's DC. 0.5 here is literally Eq. 23's own coefficient on V
    d = d_r + 0.5 * v
    return np.maximum(d, 0)


def calculate_isi(f_today, u10, v10):
    """
    Calculate today's Initial Spread Index (R) from today's FFMC and wind speed.
    Equation numbers below are from Van Wagner & Pickett (1985), Technical Report 33.
    """
    # Eq. 1 (reused): today's fine fuel moisture content, derived from TODAY's FFMC code
    m = np.divide((250 * 59.5 / 101) * (101 - f_today), 59.5 + f_today)
    wind_speed = np.sqrt(u10**2 + v10**2) * 3.6  # ERA5 m/s components -> km/h, as Eq. 24 expects

    # Eq. 24: wind effect on spread rate. 0.05039 is a fitted constant
    f_w = np.exp(0.05039 * wind_speed)

    # Eq. 25: fine fuel moisture's effect on spread rate. 91.9, 0.1386, 5.31, 4.93*10^7 are more fitted constants
    f_f = 91.9 * np.exp(-0.1386 * m) * (1 + np.divide(np.power(m, 5.31), 4.93e7))

    # Eq. 26: combine wind and moisture effects into the Initial Spread Index, with 0.208 as a fitted scaling constant
    return 0.208 * f_w * f_f


def calculate_bui(dmc_today, dc_today):
    """
    Calculate today's Buildup Index (U) from today's DMC and DC.
    Equation numbers below are from Van Wagner & Pickett (1985), Technical Report 33.
    """
    # Eq. 27a: BUI when the duff and deep layers are in their typical proportion to each other.
    # 0.8, 0.4 are fitted constants. 
    if dmc_today == 0 and dc_today == 0:
        u_a = 0.0
    else:
        # safe division to guard against division by 0
        u_a = np.divide(0.8 * dc_today * dmc_today, dmc_today + 0.4 * dc_today)

    # Eq. 27a's value is only used when it doesn't exceed the DMC it was derived from;
    # otherwise Eq. 27b's correction applies 
    if u_a < dmc_today:
        # Eq. 27b: correction that blends in more of the DC signal when the duff layer is
        # comparatively dry relative to the deep layer. 0.92, 0.0114, 1.7 are fitted constants
        p = np.divide(dmc_today - u_a, dmc_today) if dmc_today != 0 else 0.0
        cc = 0.92 + np.power(0.0114 * dmc_today, 1.7)
        u_b = dmc_today - cc * p
        return np.maximum(u_b, 0)
    else:
        return u_a

