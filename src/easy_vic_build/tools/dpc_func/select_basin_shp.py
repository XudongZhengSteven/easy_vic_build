# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Basin selection helpers based on streamflow and basin attributes.

This module provides filtering utilities for basin-level GeoDataFrames, such
as filtering by streamflow completeness, area range, and zero-flow behavior.
"""

from ... import logger
from .extractData_func import *


def selectBasinremovingStreamflowMissing(
    basin_shp, date_period=["19980101", "20101231"]
):
    """Remove basins with missing streamflow within a given date period.

    Parameters
    ----------
    basin_shp : geopandas.GeoDataFrame
        Basin dataset containing ``hru_id`` values.
    date_period : list of str, optional
        Two-element date range ``[start, end]`` in ``YYYYMMDD`` format.

    Returns
    -------
    geopandas.GeoDataFrame
        Filtered basin dataset with valid streamflow records for ``date_period``.
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
    """Select basins by area range.

    Parameters
    ----------
    basin_shp : geopandas.GeoDataFrame
        Basin dataset containing ``AREA_km2``.
    min_area : float
        Minimum basin area in square kilometers.
    max_area : float
        Maximum basin area in square kilometers.

    Returns
    -------
    geopandas.GeoDataFrame
        Filtered basin dataset with ``AREA_km2`` in ``[min_area, max_area]``.
    """
    logger.info(f"Selecting basins based on area range: {min_area} - {max_area} km^2.")
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
    """Select basins that have many zero streamflow records.

    Parameters
    ----------
    basin_shp : geopandas.GeoDataFrame
        Basin dataset containing ``hru_id`` values.
    usgs_streamflow : list of pandas.DataFrame
        Streamflow tables corresponding to basin IDs.
    streamflow_id : list
        Streamflow IDs corresponding to ``usgs_streamflow``.
    zeros_min_num : int, optional
        Minimum zero-flow count threshold. Basins with counts greater than this
        value are retained.

    Returns
    -------
    geopandas.GeoDataFrame
        Filtered basin dataset with selected zero-flow basins.
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
    """Select basins by aridity threshold.

    Parameters
    ----------
    basin_shp : geopandas.GeoDataFrame
        Basin dataset to filter.
    aridity : float
        Aridity threshold value.

    Returns
    -------
    None
        This function is currently not implemented.
    """
    logger.info(f"Selecting basins based on aridity threshold: {aridity}.")
    # Placeholder for aridity-based filtering
    pass


def selectBasinBasedOnElevSlope(basin_shp, elev_slope):
    """Select basins by elevation-slope threshold.

    Parameters
    ----------
    basin_shp : geopandas.GeoDataFrame
        Basin dataset to filter.
    elev_slope : float
        Elevation-slope threshold value.

    Returns
    -------
    None
        This function is currently not implemented.
    """
    logger.info(f"Selecting basins based on elevation slope threshold: {elev_slope}.")
    # Placeholder for elevation slope-based filtering
    pass
