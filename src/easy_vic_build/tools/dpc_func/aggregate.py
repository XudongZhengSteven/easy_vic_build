# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: aggregate

This module contains various functions for aggregating and processing spatial data related to hydrology and climate,
specifically focusing on different types of environmental data like precipitation, soil moisture, snow water equivalent (SWE),
and canopy interception. These functions are designed to aggregate data from grid-based datasets within basins, often used in
hydrological modeling and environmental studies.

Functions:
----------
    - aggregate_TRMM_P: Aggregates TRMM precipitation data for a basin.
    - aggregate_ERA5_SM: Aggregates ERA5 soil moisture data for a basin.
    - aggregate_func_SWE_axis1: Aggregates snow water equivalent (SWE) data along axis 1, removing specific values.
    - aggregate_func_SWE_axis0: Aggregates snow water equivalent (SWE) data along axis 0, removing specific values.
    - aggregate_GlobalSnow_SWE: Aggregates global snow SWE data for a basin.
    - aggregate_GLDAS_CanopInt: Aggregates GLDAS canopy interception data for a basin.

Dependencies:
-------------
    - numpy: Provides numerical operations and array manipulations.
    - pandas: Supports data manipulation and processing of geospatial datasets.
    - tqdm: For displaying progress bars during iterations.
    - functools: For partial function application.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com

"""

from functools import partial

import numpy as np
import pandas as pd
import tqdm

from ..decoractors import *


def aggregate_GLEAMEDaily(basin_shp):
    """
    Aggregates daily GLEAM E values for each basin.

    Parameters
    ----------
    basin_shp : pandas.DataFrame
        A DataFrame containing basin shapefile information, with an "intersects_grids"
        column that holds grid data intersecting each basin.

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with a new column "aggregated_E", containing aggregated
        daily GLEAM E values for each basin.

    """
    aggregate_func = partial(np.nanmean, axis=1)
    aggregate_column = "E"
    aggregate_GLEAMEDaily_list = []
    for i in tqdm(
        basin_shp.index,
        desc="loop for basin to aggregate gleam_e_daily",
        colour="green",
    ):
        intersects_grids_basin = basin_shp.loc[i, "intersects_grids"]

        intersects_grids_basin_df_list = []
        for j in intersects_grids_basin.index:
            intersects_grid = intersects_grids_basin.loc[j, :]
            intersects_grid_gleame_daily = intersects_grid[aggregate_column]
            intersects_grids_basin_df_list.append(intersects_grid_gleame_daily)

        concat_df = pd.concat(intersects_grids_basin_df_list, axis=1)
        if isinstance(concat_df["date"], pd.Series):
            aggregate_basin_date = pd.DataFrame(concat_df["date"])
        else:
            aggregate_basin_date = pd.DataFrame(concat_df["date"].iloc[:, 0])
        if isinstance(concat_df["E"], pd.Series):
            aggregate_basin_value = pd.DataFrame(concat_df["E"])
        else:
            aggregate_basin_value = aggregate_func(concat_df["E"])
        aggregate_basin_date["E"] = aggregate_basin_value
        aggregate_basin = aggregate_basin_date

        aggregate_GLEAMEDaily_list.append(aggregate_basin)

    basin_shp[f"aggregated_{aggregate_column}"] = aggregate_GLEAMEDaily_list

    return basin_shp


def aggregate_GLEAMEpDaily(basin_shp):
    """
    Aggregates daily GLEAM Ep values for each basin.

    Parameters
    ----------
    basin_shp : pandas.DataFrame
        A DataFrame containing basin shapefile information.

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with aggregated daily GLEAM Ep values.
    """
    aggregate_func = partial(np.nanmean, axis=1)
    aggregate_column = "Ep"
    aggregate_GLEAMEpDaily_list = []
    for i in tqdm(
        basin_shp.index,
        desc="loop for basin to aggregate gleam_ep_daily",
        colour="green",
    ):
        intersects_grids_basin = basin_shp.loc[i, "intersects_grids"]

        intersects_grids_basin_df_list = []
        for j in intersects_grids_basin.index:
            intersects_grid = intersects_grids_basin.loc[j, :]
            intersects_grid_gleamep_daily = intersects_grid[aggregate_column]
            intersects_grids_basin_df_list.append(intersects_grid_gleamep_daily)

        concat_df = pd.concat(intersects_grids_basin_df_list, axis=1)
        if isinstance(concat_df["date"], pd.Series):
            aggregate_basin_date = pd.DataFrame(concat_df["date"])
        else:
            aggregate_basin_date = pd.DataFrame(concat_df["date"].iloc[:, 0])
        if isinstance(concat_df["Ep"], pd.Series):
            aggregate_basin_value = pd.DataFrame(concat_df["Ep"])
        else:
            aggregate_basin_value = aggregate_func(concat_df["Ep"])
        aggregate_basin_date["Ep"] = aggregate_basin_value
        aggregate_basin = aggregate_basin_date

        aggregate_GLEAMEpDaily_list.append(aggregate_basin)

    basin_shp[f"aggregated_{aggregate_column}"] = aggregate_GLEAMEpDaily_list

    return basin_shp


def aggregate_TRMM_P(basin_shp):
    """
    Aggregate TRMM precipitation data for each basin by calculating the mean across intersecting grids.

    Parameters
    ----------
    basin_shp : GeoDataFrame
        A GeoDataFrame containing basin shapes and their associated grid intersections.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame with an additional column "aggregated_precipitation" containing the aggregated precipitation values.
    """
    aggregate_func = partial(np.nanmean, axis=1)
    aggregate_column = "precipitation"
    aggregate_list = []

    for i in tqdm(
        basin_shp.index, desc="loop for basins to aggregate TRMM_P", colour="green"
    ):
        intersects_grids_basin = basin_shp.loc[i, "intersects_grids"]

        intersects_grids_basin_df_list = []
        for j in intersects_grids_basin.index:
            intersects_grid = intersects_grids_basin.loc[j, :]
            intersects_grid_daily = intersects_grid[aggregate_column]
            intersects_grids_basin_df_list.append(intersects_grid_daily)

        concat_df = pd.concat(intersects_grids_basin_df_list, axis=1)
        if isinstance(concat_df["date"], pd.Series):
            aggregate_basin_date = pd.DataFrame(concat_df["date"])
        else:
            aggregate_basin_date = pd.DataFrame(concat_df["date"].iloc[:, 0])
        if isinstance(concat_df["precipitation"], pd.Series):
            aggregate_basin_value = pd.DataFrame(concat_df["precipitation"])
        else:
            aggregate_basin_value = aggregate_func(concat_df["precipitation"])
        aggregate_basin_date["precipitation"] = aggregate_basin_value
        aggregate_basin = aggregate_basin_date

        aggregate_list.append(aggregate_basin)

    basin_shp[f"aggregated_{aggregate_column}"] = aggregate_list

    return basin_shp


def aggregate_ERA5_SM(basin_shp, aggregate_column="swvl1"):
    """
    Aggregate ERA5 soil moisture data for each basin by calculating the mean across intersecting grids.

    Parameters
    ----------
    basin_shp : GeoDataFrame
        A GeoDataFrame containing basin shapes and their associated grid intersections.
    aggregate_column : str, optional
        The name of the column to aggregate, by default "swvl1".

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame with an additional column for aggregated soil moisture values.
    """
    aggregate_func = partial(np.nanmean, axis=1)
    aggregate_list = []

    for i in tqdm(
        basin_shp.index, desc="loop for basin to aggregate ERA5 SM", colour="green"
    ):
        intersects_grids_basin = basin_shp.loc[i, "intersects_grids"]

        intersects_grids_basin_df_list = []
        for j in intersects_grids_basin.index:
            intersects_grid = intersects_grids_basin.loc[j, :]
            intersects_grid_daily = intersects_grid[aggregate_column]
            intersects_grids_basin_df_list.append(intersects_grid_daily)

        concat_df = pd.concat(intersects_grids_basin_df_list, axis=1)
        if isinstance(concat_df["date"], pd.Series):
            aggregate_basin_date = pd.DataFrame(concat_df["date"])
        else:
            aggregate_basin_date = pd.DataFrame(concat_df["date"].iloc[:, 0])
        if isinstance(concat_df[aggregate_column], pd.Series):
            aggregate_basin_value = pd.DataFrame(concat_df[aggregate_column])
        else:
            aggregate_basin_value = aggregate_func(concat_df[aggregate_column])
        aggregate_basin_date[aggregate_column] = aggregate_basin_value
        aggregate_basin = aggregate_basin_date

        aggregate_list.append(aggregate_basin)

    basin_shp[f"aggregated_{aggregate_column}"] = aggregate_list

    return basin_shp


@apply_along_axis_decorator(axis=1)
def aggregate_func_SWE_axis1(data_array):
    """
    Aggregate SWE (Snow Water Equivalent) data along axis 1 by calculating the mean,
    while removing invalid values (negative, nan, or melting snow).

    Parameters
    ----------
    data_array : array-like
        Array of SWE values for a specific basin.

    Returns
    -------
    float
        The mean SWE value after removing invalid data.
    """
    data_array = np.array(data_array)
    data_array = data_array.astype(float)

    # create code map
    # 0         : physical values 0 mm
    # 0.001     : melting snow, removed
    # > 0.001   : physical values mm
    # < 0       : masked, removed
    # nan       : nan value, removed
    bool_removed = (data_array < 0) | (np.isnan(data_array)) | (data_array == 0.001)
    data_array = data_array[~bool_removed]

    # mean
    aggregate_value = np.mean(data_array)

    return aggregate_value


@apply_along_axis_decorator(axis=0)
def aggregate_func_SWE_axis0(data_array):
    """
    Aggregate SWE (Snow Water Equivalent) data along axis 0 by calculating the mean,
    while removing invalid values (negative, nan, or melting snow).

    Parameters
    ----------
    data_array : array-like
        Array of SWE values for a specific basin.

    Returns
    -------
    float
        The mean SWE value after removing invalid data.
    """
    data_array = np.array(data_array)

    # create code map
    # 0         : physical values 0 mm
    # 0.001     : melting snow, removed
    # > 0.001   : physical values mm
    # < 0       : masked, removed
    # nan       : nan value, removed
    bool_removed = (data_array < 0) | (np.isnan(data_array)) | (data_array == 0.001)
    data_array = data_array[~bool_removed]

    # mean
    aggregate_value = np.mean(data_array)

    return aggregate_value


def aggregate_GlobalSnow_SWE(basin_shp, aggregate_column="swe"):
    """
    Aggregate Global Snow SWE data for each basin by calculating the mean across intersecting grids.

    Parameters
    ----------
    basin_shp : GeoDataFrame
        A GeoDataFrame containing basin shapes and their associated grid intersections.
    aggregate_column : str, optional
        The name of the column to aggregate, by default "swe".

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame with an additional column for aggregated Global Snow SWE values.
    """
    aggregate_func = aggregate_func_SWE_axis1
    aggregate_list = []

    for i in tqdm(
        basin_shp.index,
        desc="loop for basin to aggregate GlobalSnow_SWE",
        colour="green",
    ):
        intersects_grids_basin = basin_shp.loc[i, "intersects_grids"]

        intersects_grids_basin_df_list = []
        for j in intersects_grids_basin.index:
            intersects_grid = intersects_grids_basin.loc[j, :]
            intersects_grid_daily = intersects_grid[aggregate_column]
            intersects_grids_basin_df_list.append(intersects_grid_daily)

        concat_df = pd.concat(intersects_grids_basin_df_list, axis=1)
        if isinstance(concat_df["date"], pd.Series):
            aggregate_basin_date = pd.DataFrame(concat_df["date"])
        else:
            aggregate_basin_date = pd.DataFrame(concat_df["date"].iloc[:, 0])
        if isinstance(concat_df[aggregate_column], pd.Series):
            aggregate_basin_value = pd.DataFrame(concat_df[aggregate_column])
        else:
            aggregate_basin_value = aggregate_func(concat_df[aggregate_column])
        aggregate_basin_date[aggregate_column] = aggregate_basin_value
        aggregate_basin = aggregate_basin_date

        aggregate_list.append(aggregate_basin)

    basin_shp[f"aggregated_{aggregate_column}"] = aggregate_list

    return basin_shp


def aggregate_GLDAS_CanopInt(basin_shp, aggregate_column="CanopInt_tavg"):
    """
    Aggregate GLDAS canopy interception data for each basin by calculating the mean across intersecting grids.

    Parameters
    ----------
    basin_shp : GeoDataFrame
        A GeoDataFrame containing basin shapes and their associated grid intersections.
    aggregate_column : str, optional
        The name of the column to aggregate, by default "CanopInt_tavg".

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame with an additional column for aggregated canopy interception values.
    """
    aggregate_func = partial(np.nanmean, axis=1)
    aggregate_list = []

    for i in tqdm(
        basin_shp.index,
        desc="loop for basin to aggregate GLDAS_CanopInt",
        colour="green",
    ):
        intersects_grids_basin = basin_shp.loc[i, "intersects_grids"]

        intersects_grids_basin_df_list = []
        for j in intersects_grids_basin.index:
            intersects_grid = intersects_grids_basin.loc[j, :]
            intersects_grid_daily = intersects_grid[aggregate_column]
            intersects_grids_basin_df_list.append(intersects_grid_daily)

        concat_df = pd.concat(intersects_grids_basin_df_list, axis=1)
        if isinstance(concat_df["date"], pd.Series):
            aggregate_basin_date = pd.DataFrame(concat_df["date"])
        else:
            aggregate_basin_date = pd.DataFrame(concat_df["date"].iloc[:, 0])
        if isinstance(concat_df[aggregate_column], pd.Series):
            aggregate_basin_value = pd.DataFrame(concat_df[aggregate_column])
        else:
            aggregate_basin_value = aggregate_func(concat_df[aggregate_column])
        aggregate_basin_date[aggregate_column] = aggregate_basin_value
        aggregate_basin = aggregate_basin_date

        aggregate_list.append(aggregate_basin)

    basin_shp[f"aggregated_{aggregate_column}"] = aggregate_list

    return basin_shp
