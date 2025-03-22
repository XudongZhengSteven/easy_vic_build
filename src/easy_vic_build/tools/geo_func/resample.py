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


def removeMissData(
    searched_grids_data, searched_grids_lat, searched_grids_lon, missing_value
):
    """
    Remove missing data from the input grids based on a specified missing value.

    This function identifies and removes data entries that match the specified missing
    value from the input data arrays. It also removes the corresponding latitude and
    longitude values if available. The function returns the cleaned data and the indices
    of the missing data for reference.

    Parameters
    ----------
    searched_grids_data : array-like
        The data array from which missing values will be removed.

    searched_grids_lat : array-like, optional
        The latitude values corresponding to the data array. Defaults to None if missing.

    searched_grids_lon : array-like, optional
        The longitude values corresponding to the data array. Defaults to None if missing.

    missing_value : float
        The value that represents missing data in the input arrays.

    Returns
    -------
    tuple
        - searched_grids_data : array
          The data array with missing values removed.

        - searched_grids_lat : array or None
          The latitude array with missing values removed, or None if not provided.

        - searched_grids_lon : array or None
          The longitude array with missing values removed, or None if not provided.

        - miss_index : array
          A boolean array indicating the positions of the missing data.

    Notes
    -----
    - The input arrays should be of the same length.
    - If latitude and longitude arrays are not provided, they will be returned as None.
    """
    miss_index = np.array(searched_grids_data) == float(missing_value)
    searched_grids_data = np.array(searched_grids_data)
    searched_grids_data = searched_grids_data[~miss_index]
    try:
        searched_grids_lat = np.array(searched_grids_lat)
        searched_grids_lat = searched_grids_lat[~miss_index]
        searched_grids_lon = np.array(searched_grids_lon)
        searched_grids_lon = searched_grids_lon[~miss_index]
    except:
        searched_grids_lat = None
        searched_grids_lon = None

    return searched_grids_data, searched_grids_lat, searched_grids_lon, miss_index


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
    if missing_value:
        searched_grids_data, searched_grids_lat, searched_grids_lon, miss_index = (
            removeMissData(
                searched_grids_data,
                searched_grids_lat,
                searched_grids_lon,
                missing_value,
            )
        )

    if len(searched_grids_data) > 0:
        dst_data = sum(searched_grids_data) / len(searched_grids_data)
    else:
        dst_data = float(missing_value) if missing_value else missing_value

    return dst_data


def resampleMethod_IDW(
    searched_grids_data,
    searched_grids_lat,
    searched_grids_lon,
    dst_lat,
    dst_lon,
    p=2,
    missing_value=None,
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
    if missing_value:
        searched_grids_data, searched_grids_lat, searched_grids_lon, miss_index = (
            removeMissData(
                searched_grids_data,
                searched_grids_lat,
                searched_grids_lon,
                missing_value,
            )
        )

    if len(searched_grids_data) > 0:
        # cal distance
        dx = abs(searched_grids_lon - dst_lon)
        dy = abs(searched_grids_lat - dst_lat)
        d = (dx**2 + dy**2) ** 0.5

        # cal weight
        d_p_inverse = 1 / (d**p)
        weight = [d_p_inverse_ / sum(d_p_inverse) for d_p_inverse_ in d_p_inverse]

        # cal dst_variable
        dst_data = sum(np.array(searched_grids_data) * np.array(weight))
    else:
        dst_data = float(missing_value) if missing_value else missing_value

    return dst_data


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
    """
    if missing_value:
        searched_grids_data, searched_grids_lat, searched_grids_lon, miss_index = (
            removeMissData(
                searched_grids_data,
                searched_grids_lat,
                searched_grids_lon,
                missing_value,
            )
        )

    if sum(miss_index) > 0:
        dst_data = float(missing_value) if missing_value else missing_value
    else:
        # combine searched_grids_lat, searched_grids_lon, variable_searched_grids_values
        searched_grids_combined = np.vstack(
            [searched_grids_lat, searched_grids_lon, searched_grids_data]
        )

        # sorted by the first row (ascending), sort based on lat to keep the first lat is same
        sorted = searched_grids_combined.T[
            np.lexsort(searched_grids_combined[::-1, :])
        ].T

        # bilinear interpolation
        linear_lat1 = (sorted[1, 1] - dst_lon) / (sorted[1, 1] - sorted[1, 0]) * sorted[
            2, 0
        ] + (dst_lon - sorted[1, 0]) / (sorted[1, 1] - sorted[1, 0]) * sorted[2, 1]
        linear_lat2 = (sorted[1, 3] - dst_lon) / (sorted[1, 3] - sorted[1, 2]) * sorted[
            2, 2
        ] + (dst_lon - sorted[1, 2]) / (sorted[1, 3] - sorted[1, 2]) * sorted[2, 2]

        dst_data = (sorted[0, 2] - dst_lat) / (
            sorted[0, 2] - sorted[0, 0]
        ) * linear_lat1 + (dst_lat - sorted[0, 0]) / (
            sorted[0, 2] - sorted[0, 0]
        ) * linear_lat2

    return dst_data


def resampleMethod_GeneralFunction(
    searched_grids_data,
    searched_grids_lat,
    searched_grids_lon,
    dst_lat=None,
    dst_lon=None,
    general_function=np.mean,
    missing_value=None,
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
    if missing_value:
        searched_grids_data, searched_grids_lat, searched_grids_lon, miss_index = (
            removeMissData(
                searched_grids_data,
                searched_grids_lat,
                searched_grids_lon,
                missing_value,
            )
        )

    if len(searched_grids_data) > 0:
        dst_data = general_function(searched_grids_data)
    else:
        dst_data = float(missing_value) if missing_value else missing_value

    return dst_data


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
    if missing_value:
        searched_grids_data, searched_grids_lat, searched_grids_lon, miss_index = (
            removeMissData(
                searched_grids_data,
                searched_grids_lat,
                searched_grids_lon,
                missing_value,
            )
        )

    if len(searched_grids_data) > 0:
        dst_data = max(set(searched_grids_data), key=searched_grids_data.count)
    else:
        dst_data = float(missing_value) if missing_value else missing_value

    return dst_data
