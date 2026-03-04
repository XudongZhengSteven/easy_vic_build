# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Resampling methods for grid-based geospatial variables.

This module provides several interpolation/aggregation strategies that map
source-grid values to a destination location, including mean, IDW, bilinear,
majority vote, generic-function aggregation, and conservative remapping.
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
    Resample values using simple arithmetic mean.

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
    float
        Mean value of ``searched_grids_data``. If no valid data is available,
        returns ``missing_value`` or ``np.nan``.
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
    Resample values using inverse-distance weighting (IDW).

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
    float
        IDW-interpolated value at ``(dst_lat, dst_lon)``. If no valid data is
        available, returns ``missing_value`` or ``np.nan``.
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
    Resample values using bilinear interpolation with fallbacks.

    Bilinear interpolation estimates the value at ``(dst_lat, dst_lon)`` using
    four surrounding points. If geometric assumptions are not met, the function
    falls back to distance-based interpolation.

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
    float
        Interpolated destination value.

    Notes
    -----
    Fallback behavior:

    - 4+ points: bilinear interpolation (or IDW when points are not rectangular),
    - 2-3 points: distance-weighted interpolation,
    - 1 point: direct value return,
    - 0 point: ``missing_value`` or ``np.nan``.
    """
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
    Resample categorical values using majority vote.

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
    float
        Most frequent value in ``searched_grids_data``. If no valid data is
        available, returns ``missing_value`` or ``np.nan``.
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
    Resample values using a user-provided aggregation function.

    This method applies ``general_function`` to source values after wrapper-level
    preprocessing.

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
    float
        Aggregated value returned by ``general_function``. If evaluation fails or
        no valid data is available, returns ``missing_value`` or ``np.nan``.
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


@resample_time_series_wrapper
@resample_missing_wrapper
def resampleMethod_conservative(
    searched_grids_data,
    searched_grids_lat,
    searched_grids_lon,
    searched_grids_res=None,
    dst_lat=None,
    dst_lon=None,    
    dst_res=None,
    missing_value=None,
):
    """
    Resample values using overlap-area conservative remapping.

    Parameters
    ----------
    searched_grids_data : array-like
        Values from source grids.
    searched_grids_lat : array-like
        Source-grid center latitudes.
    searched_grids_lon : array-like
        Source-grid center longitudes.
    searched_grids_res : float, optional
        Source-grid resolution.
    dst_lat : float, optional
        Destination-grid center latitude.
    dst_lon : float, optional
        Destination-grid center longitude.
    dst_res : float, optional
        Destination-grid resolution.
    missing_value : float, optional
        Missing-value code used when no valid overlap exists.

    Returns
    -------
    float
        Conservatively remapped destination value. If resolution inputs are
        missing, the function falls back to ``np.nanmean(data)``.
    """
    data = searched_grids_data
    lat = searched_grids_lat
    lon = searched_grids_lon
    
    n = len(data)
    
    # all missing
    if len(data) == 0:
        return np.nan if missing_value is None else missing_value

    if searched_grids_res is None or dst_res is None:
        return np.nanmean(data)  # return to mean
    
    # source bounds
    half_src = searched_grids_res / 2.0
    lat1_src = searched_grids_lat - half_src
    lat2_src = searched_grids_lat + half_src
    lon1_src = searched_grids_lon - half_src
    lon2_src = searched_grids_lon + half_src

    # destination bounds
    half_dst = dst_res / 2.0
    lat1_dst = dst_lat - half_dst
    lat2_dst = dst_lat + half_dst
    lon1_dst = dst_lon - half_dst
    lon2_dst = dst_lon + half_dst

    # overlap lengths
    lat_overlap = np.maximum(0, np.minimum(lat2_src, lat2_dst) - np.maximum(lat1_src, lat1_dst))
    lon_overlap = np.maximum(0, np.minimum(lon2_src, lon2_dst) - np.maximum(lon1_src, lon1_dst))

    overlap_area = lat_overlap * lon_overlap
    total_overlap_area = np.sum(overlap_area)

    if total_overlap_area <= 0:
        return np.nan if missing_value is None else missing_value

    # handle missing values
    if missing_value is None:
        valid_mask = np.isfinite(data)
    else:
        valid_mask = np.isfinite(data) & (data != missing_value)

    if not np.any(valid_mask & (overlap_area > 0)):
        return np.nan if missing_value is None else missing_value

    # destination area (rectangle)
    dst_area = (lat2_dst - lat1_dst) * (lon2_dst - lon1_dst)

    # total "mass"
    total_mass = np.sum(data[valid_mask] * overlap_area[valid_mask])
    
    dst_data = total_mass / dst_area
    
    return dst_data
        
    
    
    
