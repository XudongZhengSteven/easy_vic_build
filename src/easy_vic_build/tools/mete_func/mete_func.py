# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Meteorological utility functions for VIC forcing preparation."""

import math
import numpy as np
from datetime import datetime, timedelta, time


def cal_es_Tetens_eq(Ta_C, es_ref=0.6112):
    """Calculate saturation vapor pressure from air temperature.

    Parameters
    ----------
    Ta_C : float or numpy.ndarray
        Air temperature in degrees Celsius.
    es_ref : float, optional
        Reference saturation vapor pressure constant in kPa.

    Returns
    -------
    float or numpy.ndarray
        Saturation vapor pressure in kPa.

    Notes
    -----
    Piecewise Tetens form is used:

    - For ``Ta_C >= 0``:
      ``es = es_ref * exp(17.67 * Ta_C / (Ta_C + 243.5))``
    - For ``Ta_C < 0``:
      ``es = es_ref * exp(21.875 * Ta_C / (Ta_C + 265.5))``

    Unit of ``es`` is kPa.
    """
    # water
    if Ta_C >= 0:
        es = es_ref * np.exp((17.67 * Ta_C) / (Ta_C + 243.5))
    
    # ice
    else:
        es = es_ref * np.exp((21.875 * Ta_C) / (Ta_C + 265.5))
        
    return es
    
    
def cal_VP_from_RH_es(RH_100, es_kPa):
    """Calculate vapor pressure from relative humidity and saturation pressure.

    Parameters
    ----------
    RH_100 : float or numpy.ndarray
        Relative humidity in percent (0-100).
    es_kPa : float or numpy.ndarray
        Saturation vapor pressure in kPa.

    Returns
    -------
    float or numpy.ndarray
        Vapor pressure in kPa.

    Notes
    -----
    Formula:
    ``e = (RH / 100) * es``.
    """
    e_kPa = (RH_100 / 100) * es_kPa
    return e_kPa


def cal_VP_from_prs_sh(prs_kPa, sh_kg_per_kg):
    """Calculate vapor pressure from air pressure and specific humidity.

    Parameters
    ----------
    prs_kPa : float or numpy.ndarray
        Air pressure in kPa.
    sh_kg_per_kg : float or numpy.ndarray
        Specific humidity in kg/kg.

    Returns
    -------
    float or numpy.ndarray
        Vapor pressure in kPa.

    Notes
    -----
    Formula:
    ``e = q * p / (0.622 + q)``,
    where ``q`` is specific humidity (kg kg-1) and ``p`` is pressure (kPa).
    """
    # Calculate vapor pressure using the formula: VP = sh * prs / (0.622 + sh)
    e_kPa = sh_kg_per_kg * prs_kPa / (0.622 + sh_kg_per_kg)
    
    return e_kPa

def cal_SWDOWN_Angstrom_Prescott_eq(ssd_h, lat, date, a=0.25, b=0.50, clearsky=False):
    """Estimate downward shortwave radiation using Angstrom-Prescott equation.

    Parameters
    ----------
    ssd_h : float
        Observed sunshine duration in hours.
    lat : float
        Latitude in degrees.
    date : datetime.date or datetime.datetime
        Date used to compute solar geometry.
    a : float, optional
        Angstrom coefficient ``a``.
    b : float, optional
        Angstrom coefficient ``b``.
    clearsky : bool, optional
        If ``True``, return clear-sky shortwave radiation.

    Returns
    -------
    float
        Downward shortwave radiation in W/m2.

    Notes
    -----
    Angstrom-Prescott relation:
    ``Rs = (a + b * n / N) * Ra``.

    - ``n`` is observed sunshine duration (hours).
    - ``N`` is astronomical maximum sunshine duration (hours).
    - ``Ra`` is extraterrestrial radiation (MJ m-2 day-1, FAO-56 form).

    Output is converted from MJ m-2 day-1 to W m-2.
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
    """Calculate clear-sky shortwave radiation at the surface.

    Parameters
    ----------
    date : datetime.date or datetime.datetime
        Date of calculation.
    lat : float
        Latitude in degrees.
    elevation : float, optional
        Elevation in meters.
    time_UTC : int or float, optional
        UTC hour of day.
    ESRA : bool, optional
        If ``True``, use ESRA transmissivity; otherwise use Dudhia (1989).

    Returns
    -------
    float
        Clear-sky shortwave radiation in W/m2.

    Notes
    -----
    Surface clear-sky shortwave radiation is computed as:
    ``SWDOWN_clear = S0 * cos(theta_z) * tau``.

    - ``S0 = 1361`` W m-2.
    - ``theta_z`` is solar zenith angle at ``time_UTC``.
    - ``tau`` is atmospheric transmissivity.

    If ``ESRA=True``, this implementation uses
    ``tau = 0.664 + 0.163 / cos(theta_z)`` (clipped to [0, 1]),
    then applies a simple elevation correction.
    Otherwise, Dudhia (1989) approximation is used:
    ``tau = max(0.6, 0.75 - 0.00002 * elevation)``.
    """
    # Solar constant (W m-2)
    S0 = 1361.0
    
    # Day of year (1-365)
    n = date.timetuple().tm_yday
    
    # Solar declination (radians)
    decl_rad = math.radians(23.45 * math.sin(math.radians(360 * (284 + n) / 365)))
    
    # Convert latitude to radians
    lat_rad = math.radians(lat)
    
    # 15 degrees per hour, 0 at solar noon
    h = math.radians(15 * (time_UTC - 12))
    
    # Solar zenith angle (theta_z)
    cos_theta_z = (math.sin(lat_rad) * math.sin(decl_rad) + 
                  math.cos(lat_rad) * math.cos(decl_rad) * math.cos(h))
    cos_theta_z = max(0, cos_theta_z)  # avoid negative values (night)
    
    if ESRA:
        # ESRA clear-sky transmissivity model
        # ESRA model: tau = 0.664 + 0.163 / cos(theta_z)
        if cos_theta_z > 1e-3:
            transmissivity = 0.664 + (0.163 / cos_theta_z) if cos_theta_z > 1e-10 else 0
            transmissivity = min(transmissivity, 1.0)
        else:
            transmissivity = 0.0
        
        # Thin-air effect: ~10% increase per 1000m (empirical
        transmissivity *= min(1.2, 1.0 + 0.0001 * elevation)
        
    else:
        # Dudhia-like clear-sky transmissivity (tau)
        transmissivity = max(0.6, 0.75 - 0.00002 * elevation)  # Dudhia (1989) approximation
    
    # Clearsky SWDOWN (W m-2)
    swdown_clearsky = S0 * cos_theta_z * transmissivity
    
    return swdown_clearsky


def cal_LWDOWN_Brutsaert_eq(Ta_K, VP_kPa):
    """Calculate downward longwave radiation using Brutsaert equation.

    Parameters
    ----------
    Ta_K : float or numpy.ndarray
        Air temperature in Kelvin.
    VP_kPa : float or numpy.ndarray
        Vapor pressure in kPa.

    Returns
    -------
    float or numpy.ndarray
        Downward longwave radiation in W/m2.

    Notes
    -----
    Brutsaert clear-sky form:

    - ``eps_a = 1.24 * (e / Ta_K) ** (1/7)``
    - ``LWDOWN = eps_a * sigma * Ta_K ** 4``

    where ``sigma = 5.670374419e-8`` W m-2 K-4 and ``e`` is vapor pressure in kPa.
    """
    
    sigma = 5.670374419e-8  # Stefan-Boltzmann constant (W m-2 K-4)
    VP_kPa = np.maximum(VP_kPa, 0.05)  # avoid emissivity becoming zero
    eps_a = 1.24 * (VP_kPa / Ta_K) ** (1 / 7)
    LWDOWN = eps_a * sigma * Ta_K ** 4  # W m-2
    
    return LWDOWN


def cal_LWDOWN_CD99_eq(Ta_K, cloud_cover=None, c_cloud=0.22):
    """Calculate downward longwave radiation using CD99 emissivity equation.

    Parameters
    ----------
    Ta_K : float or numpy.ndarray
        Air temperature in Kelvin.
    cloud_cover : float, optional
        Cloud fraction in [0, 1]. If provided, cloudy-sky correction is applied.
    c_cloud : float, optional
        Cloud correction coefficient.

    Returns
    -------
    float or numpy.ndarray
        Downward longwave radiation in W/m2.

    Raises
    ------
    AssertionError
        If ``cloud_cover`` is outside [0, 1].

    Notes
    -----
    CD99 emissivity:
    ``eps_a = 1 - 0.261 * exp(-7.77e-4 * (273.15 - Ta_K)^2)``.

    Clear-sky longwave:
    ``LWDOWN_clear = eps_a * sigma * Ta_K^4``.

    Cloud correction (if ``cloud_cover`` is provided):
    ``LWDOWN = LWDOWN_clear * (1 + c_cloud * cloud_cover^2)``.

    Reference: Crawford and Duchon (1999),
    Journal of Applied Meteorology, 38(4), 474-480.
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
    """Calculate astronomical maximum sunshine duration for a given day.

    Parameters
    ----------
    date : datetime.date or datetime.datetime
        Date of calculation.
    lat : float
        Latitude in degrees.

    Returns
    -------
    float
        Maximum sunshine duration in hours.

    Notes
    -----
    Uses declination approximation:
    ``decl = 23.45 * sin(2*pi*(284 + n)/365)`` (degrees),
    then sunrise/sunset hour-angle ``omega`` to compute:
    ``N = 2 * degrees(omega) / 15``.
    """
    n = date.timetuple().tm_yday
    
    # declination approximation in degrees
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
    """Estimate cloud fraction from sunshine duration.

    Parameters
    ----------
    ssd_h : float
        Observed sunshine duration in hours.
    date : datetime.date or datetime.datetime
        Date of calculation.
    lat : float
        Latitude in degrees.

    Returns
    -------
    float
        Cloud fraction in [0, 1].

    Notes
    -----
    Formula:
    ``cloud_fraction = 1 - min(ssd_h / max_ssd, 1.0)``,
    then clipped to [0, 1].
    """
    max_ssd = cal_max_ssd(date, lat)
    cloud_cover = 1 - min(ssd_h / max_ssd, 1.0)
    return np.clip(cloud_cover, 0, 1)
    
    
def cal_cloud_fraction_from_swdown(sw_measure, sw_clearsky):
    """Estimate cloud fraction from measured and clear-sky shortwave radiation.

    Parameters
    ----------
    sw_measure : float or numpy.ndarray
        Measured downward shortwave radiation.
    sw_clearsky : float or numpy.ndarray
        Clear-sky downward shortwave radiation.

    Returns
    -------
    float or numpy.ndarray
        Cloud fraction in [0, 1].

    Notes
    -----
    Formula:
    ``cloud_fraction = 1 - clip(sw_measure / sw_clearsky, 0, 1)``.
    """
    ratio = np.clip(np.array(sw_measure) / np.array(sw_clearsky), 0, 1)
    cloud_fraction = 1 - ratio
    return cloud_fraction
    
    
    
