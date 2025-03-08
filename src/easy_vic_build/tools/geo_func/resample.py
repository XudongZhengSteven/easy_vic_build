# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import numpy as np


def removeMissData(searched_grids_data, searched_grids_lat, searched_grids_lon, missing_value):
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


def resampleMethod_SimpleAverage(searched_grids_data, searched_grids_lat, searched_grids_lon,
                                 dst_lat=None, dst_lon=None, missing_value=None):
    """ simple average """
    if missing_value:
        searched_grids_data, searched_grids_lat, searched_grids_lon, miss_index = removeMissData(
            searched_grids_data, searched_grids_lat, searched_grids_lon, missing_value)

    if len(searched_grids_data) > 0:
        dst_data = sum(searched_grids_data) / len(searched_grids_data)
    else:
        dst_data = float(missing_value) if missing_value else missing_value

    return dst_data


def resampleMethod_IDW(searched_grids_data, searched_grids_lat, searched_grids_lon,
                       dst_lat, dst_lon, p=2, missing_value=None):
    """ Inverse Distance Weight Interpolation
    input:
        p: the power exponent, parameter for resampleMethod_IDW, default = 2
    """
    if missing_value:
        searched_grids_data, searched_grids_lat, searched_grids_lon, miss_index = removeMissData(
            searched_grids_data, searched_grids_lat, searched_grids_lon, missing_value)

    if len(searched_grids_data) > 0:
        # cal distance
        dx = abs(searched_grids_lon - dst_lon)
        dy = abs(searched_grids_lat - dst_lat)
        d = (dx ** 2 + dy ** 2) ** 0.5

        # cal weight
        d_p_inverse = 1 / (d ** p)
        weight = [d_p_inverse_ / sum(d_p_inverse)
                  for d_p_inverse_ in d_p_inverse]

        # cal dst_variable
        dst_data = sum(np.array(searched_grids_data) * np.array(weight))
    else:
        dst_data = float(missing_value) if missing_value else missing_value

    return dst_data


def resampleMethod_bilinear(searched_grids_data, searched_grids_lat, searched_grids_lon,
                            dst_lat, dst_lon, missing_value=None):
    """ Bilinear Interpolation

    (1,2) ----- (2,2)   -> lon (1, 2, 2, 1)     (1, 2) - (2, 2) -> x (lon)
    |     (x, y)    |      lat (2, 2, 1, 1)     (1, 2) - (1, 1) -> y (lat)
    |               |      values (., ., ., .)  -> values_interpolated (., ., ., .)
    (1,1) ----- (2,1)
    """
    if missing_value:
        searched_grids_data, searched_grids_lat, searched_grids_lon, miss_index = removeMissData(
            searched_grids_data, searched_grids_lat, searched_grids_lon, missing_value)

    if sum(miss_index) > 0:
        dst_data = float(missing_value) if missing_value else missing_value
    else:
        # combine searched_grids_lat, searched_grids_lon, variable_searched_grids_values
        searched_grids_combined = np.vstack(
            [searched_grids_lat, searched_grids_lon, searched_grids_data])

        # sorted by the first row (ascending), sort based on lat to keep the first lat is same
        sorted = searched_grids_combined.T[np.lexsort(
            searched_grids_combined[::-1, :])].T

        # bilinear interpolation
        linear_lat1 = (sorted[1, 1] - dst_lon) / (sorted[1, 1] - sorted[1, 0]) * sorted[2, 0] + \
                      (dst_lon - sorted[1, 0]) / \
            (sorted[1, 1] - sorted[1, 0]) * sorted[2, 1]
        linear_lat2 = (sorted[1, 3] - dst_lon) / (sorted[1, 3] - sorted[1, 2]) * sorted[2, 2] + \
                      (dst_lon - sorted[1, 2]) / \
            (sorted[1, 3] - sorted[1, 2]) * sorted[2, 2]

        dst_data = (sorted[0, 2] - dst_lat) / (sorted[0, 2] - sorted[0, 0]) * linear_lat1 + \
            (dst_lat - sorted[0, 0]) / \
            (sorted[0, 2] - sorted[0, 0]) * linear_lat2

    return dst_data


def resampleMethod_GeneralFunction(searched_grids_data, searched_grids_lat, searched_grids_lon,
                                   dst_lat=None, dst_lon=None, general_function=np.mean, missing_value=None):
    """ resample based on general function, such as max(), min(), it can be a frozen parameter function """
    if missing_value:
        searched_grids_data, searched_grids_lat, searched_grids_lon, miss_index = removeMissData(
            searched_grids_data, searched_grids_lat, searched_grids_lon, missing_value)

    if len(searched_grids_data) > 0:
        dst_data = general_function(searched_grids_data)
    else:
        dst_data = float(missing_value) if missing_value else missing_value

    return dst_data


def resampleMethod_Majority(searched_grids_data, searched_grids_lat, searched_grids_lon,
                            dst_lat=None, dst_lon=None, missing_value=None):
    """ Find Majority """
    if missing_value:
        searched_grids_data, searched_grids_lat, searched_grids_lon, miss_index = removeMissData(
            searched_grids_data, searched_grids_lat, searched_grids_lon, missing_value)

    if len(searched_grids_data) > 0:
        dst_data = max(set(searched_grids_data), key=searched_grids_data.count)
    else:
        dst_data = float(missing_value) if missing_value else missing_value

    return dst_data
