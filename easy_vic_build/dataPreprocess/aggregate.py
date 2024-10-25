# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import numpy as np
import pandas as pd
from functools import partial
import tqdm
from ..tools.decoractors import *


def aggregate_GLEAMEDaily(basin_shp):
    aggregate_func = partial(np.nanmean, axis=1)
    aggregate_column = "E"
    aggregate_GLEAMEDaily_list = []
    for i in tqdm(basin_shp.index, desc="loop for basin to aggregate gleam_e_daily", colour="green"):
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
    aggregate_func = partial(np.nanmean, axis=1)
    aggregate_column = "Ep"
    aggregate_GLEAMEpDaily_list = []
    for i in tqdm(basin_shp.index, desc="loop for basin to aggregate gleam_ep_daily", colour="green"):
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
    aggregate_func = partial(np.nanmean, axis=1)
    aggregate_column = "precipitation"
    aggregate_list = []
    
    for i in tqdm(basin_shp.index, desc="loop for basins to aggregate TRMM_P", colour="green"):
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
    aggregate_func = partial(np.nanmean, axis=1)
    aggregate_list = []
    
    for i in tqdm(basin_shp.index, desc="loop for basin to aggregate ERA5 SM", colour="green"):
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
    data_array = np.array(data_array)
    data_array = data_array.astype(float)
    
    # create code map
    # 0         : physical values 0 mm
    # 0.001     : melting snow, removed
    # > 0.001   : physical values mm
    # < 0       : masked, removed
    # nan       : nan value, removed
    bool_removed = ((data_array < 0) | (np.isnan(data_array)) | (data_array == 0.001))
    data_array = data_array[~bool_removed]
    
    # mean
    aggregate_value = np.mean(data_array)
    
    return aggregate_value


@apply_along_axis_decorator(axis=0)
def aggregate_func_SWE_axis0(data_array):
    data_array = np.array(data_array)
    
    # create code map
    # 0         : physical values 0 mm
    # 0.001     : melting snow, removed
    # > 0.001   : physical values mm
    # < 0       : masked, removed
    # nan       : nan value, removed
    bool_removed = ((data_array < 0) | (np.isnan(data_array)) | (data_array == 0.001))
    data_array = data_array[~bool_removed]
    
    # mean
    aggregate_value = np.mean(data_array)
    
    return aggregate_value


def aggregate_GlobalSnow_SWE(basin_shp, aggregate_column="swe"):
    aggregate_func = aggregate_func_SWE_axis1
    aggregate_list = []
    
    for i in tqdm(basin_shp.index, desc="loop for basin to aggregate GlobalSnow_SWE", colour="green"):
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
    aggregate_func = partial(np.nanmean, axis=1)
    aggregate_list = []
    
    for i in tqdm(basin_shp.index, desc="loop for basin to aggregate GLDAS_CanopInt", colour="green"):
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