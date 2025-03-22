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


import numpy as np


def cal_VP(T, RH):
    """
    Calculate the vapor pressure (e) based on temperature (T) and relative humidity (RH).

    The vapor pressure is calculated using the Clausius-Clapeyron equation:
    e = RH * es / 100, where:
        - es is the saturation vapor pressure, calculated as 6.112 * exp(17.67*T / (T + 243.5)).
        - T is the temperature in degrees Celsius.
        - RH is the relative humidity in percentage (0 to 100).

    Parameters
    ----------
    T : float or array-like
        The temperature in degrees Celsius (°C).
    RH : float or array-like
        The relative humidity as a percentage (0 to 100).

    Returns
    -------
    float or ndarray
        The vapor pressure (e) in hPa.

    Notes
    -----
    The function assumes that temperature (T) is in degrees Celsius and relative humidity (RH) is a percentage.
    """
    # T in C
    es = 6.112 * np.exp((17.67 * T) / (T + 243.5))
    e = RH * es / 100
    return e
