# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: clip

This module contains functions for clipping geospatial data to a specified region based on
latitude, longitude, and resolution. The clip function extracts a subregion of the source data
that corresponds to the target geographical area defined by the input latitude and longitude
ranges. This is useful for reducing data size and improving computational efficiency when working
with large geospatial datasets.

Functions:
----------
    - clip: Clips the source geospatial data based on the specified latitude and longitude
      ranges, adjusting for the given resolution.

Dependencies:
-------------
    - numpy: Provides numerical operations and array manipulations, especially for array slicing
      and mesh grid creation.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com
"""

import numpy as np


def clip(dst_lat, dst_lon, dst_res, src_lat, src_lon, src_data, reverse_lat=True):
    """
    Clips the source data to match the desired latitude and longitude range, based on the target resolution.
    clip for extracting before to improve speed, avoid to creating too large search array, as below

    This function extracts a subset of the source data, adjusting for the resolution and the specified
    latitude/longitude ranges. It helps to avoid creating overly large search arrays, improving speed during
    geospatial operations.

    Parameters
    ----------
    dst_lat : array-like
        The latitude values of the target grid (destination).

    dst_lon : array-like
        The longitude values of the target grid (destination).

    dst_res : float
        The resolution of the target grid (in degrees).

    src_lat : array-like
        The latitude values of the source data grid.

    src_lon : array-like
        The longitude values of the source data grid.

    src_data : array-like
        The source data to be clipped (must match the dimensions of src_lat and src_lon).

    reverse_lat : bool, optional
        If True, assumes the source latitude values are in descending order (large to small).
        If False, assumes ascending order (small to large). Default is True.

    Returns
    -------
    src_data_clip : array-like
        The clipped source data, matching the specified latitude and longitude range.

    src_lon_clip : array-like
        The clipped longitude values corresponding to the selected source data.

    src_lat_clip : array-like
        The clipped latitude values corresponding to the selected source data.

    Notes
    -----
    This function uses `np.where` to find the indices of the source data that fall within the specified latitude
    and longitude ranges (with an additional buffer based on the target resolution). The clipping operation extracts
    the relevant subset of the source data.
    """
    xindex_start = np.where(src_lon <= min(dst_lon) - dst_res / 2)[0][-1]
    xindex_end = np.where(src_lon >= max(dst_lon) + dst_res / 2)[0][0]

    # if reverse_lat (src_lat, large -> small), else (src_lat, small -> large)
    if reverse_lat:
        yindex_start = np.where(src_lat >= max(dst_lat) + dst_res / 2)[0][-1]
        yindex_end = np.where(src_lat <= min(dst_lat) - dst_res / 2)[0][0]
    else:
        yindex_start = np.where(src_lat <= min(dst_lat) - dst_res / 2)[0][-1]
        yindex_end = np.where(src_lat >= max(dst_lat) + dst_res / 2)[0][0]

    src_data_clip = src_data[
        yindex_start : yindex_end + 1, xindex_start : xindex_end + 1
    ]
    src_lon_clip = src_lon[xindex_start : xindex_end + 1]
    src_lat_clip = src_lat[yindex_start : yindex_end + 1]

    ## old version
    # xindex = np.where((src_lon >= min(dst_lon) - dst_res/2) & (src_lon <= max(dst_lon) + dst_res/2))[0]
    # yindex = np.where((src_lat >= min(dst_lat) - dst_res/2) & (src_lat <= max(dst_lat) + dst_res/2))[0]

    # src_data_clip = src_data[min(yindex): max(yindex), min(xindex): max(xindex)]
    # src_lon_clip = src_lon[min(xindex): max(xindex)]
    # src_lat_clip = src_lat[min(yindex): max(yindex)]

    ## then search grids
    # searched_grids_index = search_grids.search_grids_radius_rectangle(dst_lat=grids_lat, dst_lon=grids_lon,
    #                                                                     src_lat=umd_lat_clip, src_lon=umd_lon_clip,
    #                                                                     lat_radius=grid_shp_res/2, lon_radius=grid_shp_res/2)

    return src_data_clip, src_lon_clip, src_lat_clip
