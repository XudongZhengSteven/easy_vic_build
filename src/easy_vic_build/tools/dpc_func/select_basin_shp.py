# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: select_basin_shp

This module provides functions to filter and select basins from a GeoDataFrame based on various hydrological and
environmental criteria, such as streamflow data, area, aridity, and elevation slope. These functions are commonly used in
hydrological modeling and environmental research to process and refine basin-level datasets.

Functions:
----------
    - selectBasinremovingStreamflowMissing: Removes basins with missing streamflow data for a specified date range.
    - selectBasinBasedOnArea: Selects basins based on a specified area range.
    - selectBasinStreamflowWithZero: Selects basins with a significant number of zero streamflow values.
    - selectBasinBasedOnAridity: Selects basins based on an aridity threshold (yet to be implemented).
    - selectBasinBasedOnElevSlope: Selects basins based on an elevation slope threshold (yet to be implemented).

Dependencies:
-------------
    - pandas: Used for data manipulation and filtering of basin-related attributes.
    - geopandas: Handles geospatial data in GeoDataFrame format for basin selection.
    - logger: Custom logging module for tracking the filtering process.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com

"""

from ... import logger
from .extractData_func import *


def selectBasinremovingStreamflowMissing(
    basin_shp, date_period=["19980101", "20101231"]
):
    """
    Selects basins and removes those with missing streamflow data within the specified date period.

    Parameters
    ----------
    basin_shp : GeoDataFrame
        A GeoDataFrame containing basin data with streamflow IDs to be filtered.

    date_period : list of str, optional
        A list containing the start and end dates for the period to filter streamflow data. Default is ["19980101", "20101231"].

    Returns
    -------
    basin_shp : GeoDataFrame
        A filtered GeoDataFrame with basins that have valid streamflow data for the specified date period.
    """
    logger.info(
        f"Removing basins with missing streamflow data for period {date_period}."
    )
    # get remove streamflow missing
    streamflows_dict_original, streamflows_dict_removed_missing = (
        Extract_CAMELS_Streamflow.getremoveStreamflowMissing(date_period)
    )
    remove_num = len(streamflows_dict_original["usgs_streamflows"]) - len(
        streamflows_dict_removed_missing["usgs_streamflows"]
    )
    print(f"remove Basin based on StreamflowMissing: remove {remove_num} files")

    # get ids removed missing
    streamflow_ids_removed_missing = streamflows_dict_removed_missing["streamflow_ids"]
    index_removed_missing = [
        id in streamflow_ids_removed_missing for id in basin_shp.hru_id.values
    ]

    # remove
    basin_shp = basin_shp.iloc[index_removed_missing, :]

    logger.info(
        f"Remaining {len(basin_shp)} basins after removing those with missing streamflow data."
    )

    return basin_shp


def selectBasinBasedOnArea(basin_shp, min_area, max_area):
    """
    Selects basins based on the specified area range.

    Parameters
    ----------
    basin_shp : GeoDataFrame
        A GeoDataFrame containing basin data with area information.

    min_area : float
        The minimum area of the basin in square kilometers.

    max_area : float
        The maximum area of the basin in square kilometers.

    Returns
    -------
    basin_shp : GeoDataFrame
        A filtered GeoDataFrame with basins whose area is within the specified range.
    """
    logger.info(f"Selecting basins based on area range: {min_area} - {max_area} km².")
    basin_shp = basin_shp.loc[
        (basin_shp.loc[:, "AREA_km2"] >= min_area)
        & (basin_shp.loc[:, "AREA_km2"] <= max_area),
        :,
    ]
    logger.info(f"Remaining {len(basin_shp)} basins after filtering based on area.")

    return basin_shp


def selectBasinStreamflowWithZero(
    basin_shp, usgs_streamflow, streamflow_id, zeros_min_num=100
):
    """
    Selects basins with a significant number of zero streamflow values.

    Parameters
    ----------
    basin_shp : GeoDataFrame
        A GeoDataFrame containing basin data to be filtered.

    usgs_streamflow : list of DataFrame
        A list of DataFrames containing the streamflow data for each basin.

    streamflow_id : list of str
        A list of streamflow IDs corresponding to each basin.

    zeros_min_num : int, optional
        The minimum number of zero streamflow values required for selecting a basin. Default is 100.

    Returns
    -------
    basin_shp : GeoDataFrame
        A filtered GeoDataFrame with basins that have a significant number of zero streamflow values.
    """
    # loop for each basin
    logger.info(
        f"Selecting basins based on zero streamflow values, with a minimum of {zeros_min_num} zeros."
    )
    selected_id = []

    for i in range(len(usgs_streamflow)):
        usgs_streamflow_ = usgs_streamflow[i]
        streamflow = usgs_streamflow_.iloc[:, 4].values
        zero_count = sum(streamflow == 0)
        if zero_count > zeros_min_num:  # find basin with zero streamflow
            selected_id.append(streamflow_id[i])
            logger.info(
                f"Basin {streamflow_id[i]} has {zero_count} zero streamflow values."
            )
            # plt.plot(streamflow)
            # plt.ylim(bottom=0)
            # plt.show()

    selected_index = [id in selected_id for id in basin_shp.hru_id.values]
    basin_shp = basin_shp.iloc[selected_index, :]

    logger.info(
        f"Remaining {len(basin_shp)} basins after filtering based on zero streamflow."
    )
    return basin_shp


def selectBasinBasedOnAridity(basin_shp, aridity):
    """
    Selects basins based on aridity.

    Parameters
    ----------
    basin_shp : GeoDataFrame
        A GeoDataFrame containing basin data to be filtered.

    aridity : float
        The aridity threshold value for basin selection.

    Returns
    -------
    basin_shp : GeoDataFrame
        A filtered GeoDataFrame with basins that meet the specified aridity condition.
    """
    logger.info(f"Selecting basins based on aridity threshold: {aridity}.")
    # Placeholder for aridity-based filtering
    pass


def selectBasinBasedOnElevSlope(basin_shp, elev_slope):
    """
    Selects basins based on elevation slope.

    Parameters
    ----------
    basin_shp : GeoDataFrame
        A GeoDataFrame containing basin data to be filtered.

    elev_slope : float
        The elevation slope threshold for basin selection.

    Returns
    -------
    basin_shp : GeoDataFrame
        A filtered GeoDataFrame with basins that meet the specified elevation slope condition.
    """
    logger.info(f"Selecting basins based on elevation slope threshold: {elev_slope}.")
    # Placeholder for elevation slope-based filtering
    pass
