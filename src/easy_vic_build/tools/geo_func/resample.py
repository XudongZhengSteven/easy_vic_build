# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: resample

This module provides various resampling methods for spatial grid data. It includes functions
for interpolating or aggregating values from nearby grid points to estimate values at a given
destination location. These methods are useful for spatial analysis in hydrology, meteorology,
and geographic data processing.

Functions:
----------
    - removeMissData: Removes missing values from the input grid data.
    - resampleMethod_SimpleAverage: Computes the simple average of the searched grid data for resampling.
    - resampleMethod_IDW: Performs Inverse Distance Weighted (IDW) interpolation for resampling.
    - resampleMethod_bilinear: Performs bilinear interpolation for resampling.
    - resampleMethod_GeneralFunction: Applies a general aggregation function (e.g., mean, max, min) for resampling.
    - resampleMethod_Majority: Finds the most frequently occurring value (majority vote) in the searched grid data.

Dependencies:
-------------
    - numpy: Provides support for numerical operations.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com
"""

import numpy as np
from collections import Counter
from ..decoractors import resample_time_series_wrapper, resample_missing_wrapper

@resample_time_series_wrapper
@resample_missing_wrapper
def resampleMethod_SimpleAverage(
    searched_grids_data,
    searched_grids_lat,
    searched_grids_lon,
    dst_lat=None,
    dst_lon=None,
    missing_value=None,
):
    """
    Resamples the input grid data using a simple average method.

    Parameters
    ----------
    searched_grids_data : array-like
        The data values of the searched grids.
    searched_grids_lat : array-like
        The latitudes corresponding to the searched grids.
    searched_grids_lon : array-like
        The longitudes corresponding to the searched grids.
    dst_lat : float, optional
        The latitude of the destination grid (not used in computation).
    dst_lon : float, optional
        The longitude of the destination grid (not used in computation).
    missing_value : float or None, optional
        The value representing missing data. If provided, missing data will be removed before averaging.

    Returns
    -------
    float or None
        The resampled data value obtained by simple averaging. If no valid data remains after
        removing missing values, returns `missing_value` or None.
    """
    if len(searched_grids_data) == 0:
        return np.nan if missing_value is None else missing_value
    
    return float(np.nanmean(searched_grids_data))


@resample_time_series_wrapper
@resample_missing_wrapper
def resampleMethod_IDW(
    searched_grids_data,
    searched_grids_lat,
    searched_grids_lon,
    dst_lat,
    dst_lon,
    missing_value=None,
    p=2,
):
    """
    Resamples the input grid data using Inverse Distance Weighting (IDW) interpolation.

    Parameters
    ----------
    searched_grids_data : array-like
        The data values of the searched grids.
    searched_grids_lat : array-like
        The latitudes corresponding to the searched grids.
    searched_grids_lon : array-like
        The longitudes corresponding to the searched grids.
    dst_lat : float
        The latitude of the destination grid.
    dst_lon : float
        The longitude of the destination grid.
    p : int or float, optional
        The power exponent for weighting, controlling the influence of distance. Default is 2.
    missing_value : float or None, optional
        The value representing missing data. If provided, missing data will be removed before interpolation.

    Returns
    -------
    float or None
        The resampled data value obtained using IDW interpolation. If no valid data remains after
        removing missing values, returns `missing_value` or None.
    """
    data = searched_grids_data
    lat = searched_grids_lat
    lon = searched_grids_lon
    
    if len(data) == 0:
        return np.nan if missing_value is None else missing_value
    
    # get distance
    dx = lon - dst_lon
    dy = lat - dst_lat
    d = np.sqrt(dx**2 + dy**2)
    
    # same location as target point
    if np.any(d == 0):
        return float(data[d == 0][0])

    weights = d**(-p)
    weights /= weights.sum()
    
    return float(np.sum(data * weights))


@resample_time_series_wrapper
@resample_missing_wrapper
def resampleMethod_bilinear(
    searched_grids_data,
    searched_grids_lat,
    searched_grids_lon,
    dst_lat,
    dst_lon,
    missing_value=None,
):
    """
    Resamples the input grid data using bilinear interpolation.

    Bilinear interpolation estimates the value at a given point (dst_lat, dst_lon) using
    the weighted average of the four surrounding grid points.

    Schematic representation:

        (lat2, lon1) ----- (lat2, lon2)   -> Corresponding latitudes and longitudes
              |    (x, y)   |
              |             |
        (lat1, lon1) ----- (lat1, lon2)

        - The interpolation first computes intermediate values along the longitude direction.
        - Then, it interpolates along the latitude direction.

    Parameters
    ----------
    searched_grids_data : array-like
        The data values of the searched grids.
    searched_grids_lat : array-like
        The latitudes corresponding to the searched grids.
    searched_grids_lon : array-like
        The longitudes corresponding to the searched grids.
    dst_lat : float
        The latitude of the destination grid.
    dst_lon : float
        The longitude of the destination grid.
    missing_value : float or None, optional
        The value representing missing data. If provided, missing data will be removed before interpolation.

    Returns
    -------
    float or None
        The resampled data value obtained using bilinear interpolation. If no valid data remains after
        removing missing values, returns `missing_value` or None.
    
    Cases:
    - 4+ points: bilinear interpolation (using first 4 sorted points)
    - 3 points: IDW
    - 2 points: linear interpolation
    - 1 point: return that value
    - 0 points: return missing_value
    """
    # remove missing data
    data = searched_grids_data
    lat = searched_grids_lat
    lon = searched_grids_lon
    
    n = len(data)
    
    # all missing
    if n == 0:
        return np.nan if missing_value is None else missing_value

    elif n == 1:
        return data[0]
    
    elif 2 <= n <= 3:
        distances = np.sqrt((lat - dst_lat)**2 + (lon - dst_lon)**2)
        if np.any(distances == 0):
            return data[np.argmin(distances)]
        weights = 1 / distances
        return np.sum(data * weights) / np.sum(weights)
    
    else:
        points = np.array([lat, lon, data]).T
        points = points[np.lexsort((points[:, 1], points[:, 0]))]

        lat1, lon1, q11 = points[0]
        _,    lon2, q12 = points[1]
        lat2, _,    q21 = points[2]
        _,    _,    q22 = points[3]
        
        is_rectangle = (
            np.isclose(lat1, points[1][0]) and
            np.isclose(lat2, points[3][0]) and
            np.isclose(lon1, points[2][1]) and
            np.isclose(lon2, points[3][1])
        )
        
        if not is_rectangle:
            # return to IDW
            distances = np.sqrt((lat - dst_lat)**2 + (lon - dst_lon)**2)
            if np.any(distances == 0):
                return float(data[np.argmin(distances)])
            weights = 1 / distances
            return float(np.sum(data * weights) / np.sum(weights))
        
        f_lon1 = (lon2 - dst_lon) / (lon2 - lon1) * q11 + (dst_lon - lon1) / (lon2 - lon1) * q12
        f_lon2 = (lon2 - dst_lon) / (lon2 - lon1) * q21 + (dst_lon - lon1) / (lon2 - lon1) * q22
        f_lat  = (lat2 - dst_lat) / (lat2 - lat1) * f_lon1 + (dst_lat - lat1) / (lat2 - lat1) * f_lon2

        return f_lat


@resample_time_series_wrapper
@resample_missing_wrapper
def resampleMethod_Majority(
    searched_grids_data,
    searched_grids_lat,
    searched_grids_lon,
    dst_lat=None,
    dst_lon=None,
    missing_value=None,
):
    """
    Resamples the input grid data using majority voting.

    This method finds the most frequently occurring value (mode) in the searched grid data.
    It is useful for categorical data resampling, such as land cover classification.

    Parameters
    ----------
    searched_grids_data : array-like
        The data values of the searched grids.
    searched_grids_lat : array-like
        The latitudes corresponding to the searched grids.
    searched_grids_lon : array-like
        The longitudes corresponding to the searched grids.
    dst_lat : float, optional
        The latitude of the destination grid (not used in computation).
    dst_lon : float, optional
        The longitude of the destination grid (not used in computation).
    missing_value : float or None, optional
        The value representing missing data. If provided, missing data will be removed before computing
        the majority value.

    Returns
    -------
    float or None
        The most frequently occurring value in the searched grid data. If no valid data remains
        after removing missing values, returns `missing_value` or None.
    """
    data = searched_grids_data
    lat = searched_grids_lat
    lon = searched_grids_lon
    
    data = np.array(data)
    
    # all missing
    if len(data) == 0:
        return np.nan if missing_value is None else missing_value

    try:
        counter = Counter(data)
        dst_data = counter.most_common(1)[0][0]
    except Exception:
        dst_data = np.nan if missing_value is None else missing_value

    return dst_data


@resample_time_series_wrapper
@resample_missing_wrapper
def resampleMethod_GeneralFunction(
    searched_grids_data,
    searched_grids_lat,
    searched_grids_lon,
    dst_lat=None,
    dst_lon=None,
    missing_value=None,
    general_function=np.mean,
):
    """
    Resamples the input grid data using a general function, such as max(), min(), or a custom function.

    This function allows the user to apply any aggregation function (e.g., mean, median, max, min)
    to resample the data. The function can also be a frozen parameter function.

    Parameters
    ----------
    searched_grids_data : array-like
        The data values of the searched grids.
    searched_grids_lat : array-like
        The latitudes corresponding to the searched grids.
    searched_grids_lon : array-like
        The longitudes corresponding to the searched grids.
    dst_lat : float, optional
        The latitude of the destination grid (not used in computation).
    dst_lon : float, optional
        The longitude of the destination grid (not used in computation).
    general_function : callable, optional
        A function that aggregates the input data, such as `np.mean`, `np.max`, or `np.min`.
        Default is `np.mean`.
    missing_value : float or None, optional
        The value representing missing data. If provided, missing data will be removed before applying
        the general function.

    Returns
    -------
    float or None
        The resampled data value obtained using the specified general function. If no valid data remains
        after removing missing values, returns `missing_value` or None.
    """
    data = searched_grids_data
    lat = searched_grids_lat
    lon = searched_grids_lon

    data = np.array(data, dtype=float)

    # all missing
    if len(data) == 0:
        return np.nan if missing_value is None else missing_value

    try:
        dst_data = general_function(data)
    except Exception:
        dst_data = np.nan if missing_value is None else missing_value

    return dst_data
