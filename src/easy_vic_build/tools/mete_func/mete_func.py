# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: mete_func

This module contains functions for meteorological calculations. Specifically, it provides a method to calculate
the vapor pressure (VP) based on temperature (T) and relative humidity (RH). The function is useful for
various atmospheric and hydrological models that require vapor pressure as an input parameter.

Functions:
----------
    - cal_VP: Calculates the vapor pressure (e) based on temperature (T) and relative humidity (RH).
      The calculation is done using the Clausius-Clapeyron equation.

Dependencies:
-------------
    - numpy: Used for numerical operations and handling arrays, particularly for exponential functions.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com
"""

import math
import numpy as np
from datetime import datetime, timedelta, time


def cal_es_Tetens_eq(Ta_C, es_ref=0.6112):
    """ 
    Calculate the saturation vapor pressure (es) based on temperature (T) using Tetens equation.
    
    The saturation vapor pressure is calculated using the formula:
    es = es_ref * exp((17.67 * T) / (T + 243.5)), kPa
    
    where:
        - es_ref is the reference saturation vapor pressure constant: 0.6112 kPa or 0.61078 kPa.
        - es is the saturation vapor pressure in kPa.
        - T is the temperature in degrees Celsius.
    """
    # water
    if Ta_C >= 0:
        es = es_ref * np.exp((17.67 * Ta_C) / (Ta_C + 243.5))
    
    # ice
    else:
        es = es_ref * np.exp((21.875 * Ta_C) / (Ta_C + 265.5))
        
    return es
    
    
def cal_VP_from_RH_es(RH_100, es_kPa):
    """
    Calculate the vapor pressure (e) based on relative humidity (RH) and saturation vapor pressure (es).
    """
    e_kPa = (RH_100 / 100) * es_kPa
    return e_kPa


def cal_VP_from_prs_sh(prs_kPa, sh_kg_per_kg):
    """ 
    Calculate the vapor pressure (VP) from pressure (prs) and specific humidity (sh).
    """
    # Calculate vapor pressure using the formula: VP = sh * prs / (0.622 + sh)
    e_kPa = sh_kg_per_kg * prs_kPa / (0.622 + sh_kg_per_kg)
    
    return e_kPa

def cal_SWDOWN_Angstrom_Prescott_eq(ssd_h, lat, date, a=0.25, b=0.50, clearsky=False):
    """
    Calculate the downward shortwave radiation (SWDOWN) using Angstrom-Prescott equation.
    
    Rs = (a+b*n/N) * Ra, W m-2
    where:
        - Rs is the downward shortwave radiation (SWDOWN).
        - Ra is the extraterrestrial radiation.
        - n is the actual sunshine hours.
        - N is the maximum possible sunshine hours.
        - a and b are Angstrom coefficients.
    """
    # lat to radian
    lat_rad = math.radians(lat)
    
    # doy
    doy = date.timetuple().tm_yday
    
    # solar constant
    G_sc = 0.0820 # MJ m-2 min-1, per FAO-56
    
    # dr
    dr = 1 + 0.033 * math.cos(2 * math.pi * doy / 365)
    
    # delta
    delta = 0.409 * math.sin(2 * math.pi / 365 * doy - 1.39)
    
    # omega_s
    ws = math.acos(-math.tan(lat_rad) * math.tan(delta))
    
    # Ra, MJ m-2 day-1
    Ra = (24 * 60 / math.pi) * G_sc * dr * (ws * math.sin(lat_rad) * math.sin(delta) + 
        math.cos(lat_rad) * math.cos(delta) * math.sin(ws))
    
    # N
    N = (24 / math.pi) * ws
    
    # Rs
    if N > 0:
        Rs = (a + b * (ssd_h / N)) * Ra
    else:
        Rs = 0
    
    # clearsky condition
    Rs_clearsky = (a + b) * Ra
            
    # SWDOWN
    if clearsky:
        SWDOWN = Rs_clearsky * 1e6 / (24 * 3600)  # Convert from MJ m-2 day-1 to W m-2
    else:
        SWDOWN = Rs * 1e6 / (24 * 3600)  # Convert from MJ m-2 day-1 to W m-2
    
    return SWDOWN


def cal_clearsky_SWDOWN_Dudhia89_eq(date, lat, elevation=0, time_UTC=12, ESRA=False):
    """
    Calculate clearsky shortwave radiation (SWDOWN) at the surface using Dudhia (1989) equation.
    
    Args:
        date (datetime.date): Date of calculation.
        lat (float): Latitude (degrees).
        elevation (float): Elevation (m), default=0.
        time_UTC (int): Hour of the day (0-23), default=12 (solar noon).
    
    Returns:
        float: Clearsky SWDOWN (W/m²).
    """
    # Solar constant (W/m²)
    S0 = 1361.0
    
    # Day of year (1-365)
    n = date.timetuple().tm_yday
    
    # Solar declination (radians)
    decl_rad = math.radians(23.45 * math.sin(math.radians(360 * (284 + n) / 365)))
    
    # Convert latitude to radians
    lat_rad = math.radians(lat)
    
    # 15° per hour, 0 at solar noon
    h = math.radians(15 * (time_UTC - 12))
    
    # Solar zenith angle (θ_z)
    cos_theta_z = (math.sin(lat_rad) * math.sin(decl_rad) + 
                  math.cos(lat_rad) * math.cos(decl_rad) * math.cos(h))
    cos_theta_z = max(0, cos_theta_z)  # avoid negative values (night)
    
    if ESRA:
        # ESRA clear-sky transmissivity model
        # ESRA Model: τ = 0.664 + 0.163/cos(θ_z) - European Solar Radiation Atlas (ESRA)
        if cos_theta_z > 1e-3:
            transmissivity = 0.664 + (0.163 / cos_theta_z) if cos_theta_z > 1e-10 else 0
            transmissivity = min(transmissivity, 1.0)
        else:
            transmissivity = 0.0
        
        # Thin-air effect: ~10% increase per 1000m (empirical
        transmissivity *= min(1.2, 1.0 + 0.0001 * elevation)
        
    else:
        # Clear-sky transmissivity (τ_clearsky)
        transmissivity = max(0.6, 0.75 - 0.00002 * elevation)  # Dudhia (1989) approximation
    
    # Clearsky SWDOWN (W/m²)
    swdown_clearsky = S0 * cos_theta_z * transmissivity
    
    return swdown_clearsky


def cal_LWDOWN_Brutsaert_eq(Ta_K, VP_kPa):
    """
    Calculate the downward longwave radiation (LWDOWN) using Brutsaert's equation.
    
    LWDOWN = eps_a * sigma * Ta_K^4, W m-2
    
    where:
        - eps_a is the emissivity of the atmosphere, calculated as 1.24 * (VP_kPa / Ta_K)^(1/7).
        - sigma is the Stefan-Boltzmann constant: 5.670374419e-8 W m-2 K-4.
        - Ta_K is the air temperature in Kelvin.
        - VP_kPa is the vapor pressure in kilopascals (kPa).
    """
    
    sigma = 5.670374419e-8  # Stefan-Boltzmann constant (W m-2 K-4)
    VP_kPa = np.maximum(VP_kPa, 0.05)  # avoid emissivity becoming zero
    eps_a = 1.24 * (VP_kPa / Ta_K) ** (1 / 7)
    LWDOWN = eps_a * sigma * Ta_K ** 4  # W m-2
    
    return LWDOWN


def cal_LWDOWN_CD99_eq(Ta_K, cloud_cover=None, c_cloud=0.22):
    """ 
    Calculate the downward longwave radiation (LWDOWN) using the CD99 equation.
    
    Crawford, T. M., & Duchon, C. E. (1999). An Improved Parameterization for Estimating Effective Atmospheric Emissivity for Use in 
    Calculating Daytime Downwelling Longwave Radiation. Journal of Applied Meteorology, 38(4), 474-480. (DOI: 10.1175/1520-0450(1999)038<0474:AIPFEE>2.0.CO;2)
    """
    sigma = 5.670374419e-8  # Stefan-Boltzmann constant (W m-2 K-4)
    eps_a = 1 - 0.261 * np.exp(-7.77e-4 * (273.15 - Ta_K)**2)  # CD99
    LWDOWN_clear = eps_a * sigma * Ta_K ** 4  # W m-2
    
    if cloud_cover is not None:
        assert 0 <= cloud_cover <= 1, "Cloud cover must be between 0 and 1."
        LWDOWN_cloudy = LWDOWN_clear * (1 + c_cloud * cloud_cover**2)
        return LWDOWN_cloudy
    else:
        return LWDOWN_clear


def cal_max_ssd(date, lat):
    """ 
    Calculate the maximum sunshine duration (ssd) for a given date and location.
    """
    n = date.timetuple().tm_yday
    
    # δ ≈ 23.45° × sin(360° × (284 + n) / 365)
    decl = 23.45 * math.sin(math.radians(360 * (284 + n) / 365))
    
    # hour angle at sunrise/sunset
    lat_rad = math.radians(lat)
    decl_rad = math.radians(decl)
    cos_omega = -math.tan(lat_rad) * math.tan(decl_rad)

    if cos_omega >= 1:
        return 0.0
    elif cos_omega <= -1:
        return 24.0

    omega = math.acos(cos_omega)  # degree
    max_ssd = (2 * math.degrees(omega)) / 15  # 15 degree = 1h

    return max_ssd


def cal_cloud_fraction_from_ssd(ssd_h, date, lat):
    """ 
    Calculate the cloud fraction from sunshine duration (ssd).
    """
    max_ssd = cal_max_ssd(date, lat)
    cloud_cover = 1 - min(ssd_h / max_ssd, 1.0)
    return np.clip(cloud_cover, 0, 1)
    
    
def cal_cloud_fraction_from_swdown(sw_measure, sw_clearsky):
    """
    Calculate the cloud fraction from downward shortwave radiation (SWDOWN).
    """
    ratio = np.clip(np.array(sw_measure) / np.array(sw_clearsky), 0, 1)
    cloud_fraction = 1 - ratio
    return cloud_fraction
    
    
    