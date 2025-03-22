# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import *

from ...geo_func import search_grids


def ExtractData(grid_shp, period=["19800101", "20101231"], var_name="E"):
    # period
    period_date = [
        datetime.strptime(period[0], "%Y%m%d"),
        datetime.strptime(period[1], "%Y%m%d"),
    ]
    period_year = [period_date[0].year, period_date[1].year]

    # general
    home = "E:/data/hydrometeorology/GLEAM/data/v3.8a/daily"
    version = "v3.8a"
    suffix = ".nc"

    src_dirs = [
        d
        for d in os.listdir(home)
        if (
            os.path.isdir(os.path.join(home, d))
            and int(d) >= period_year[0]
            and int(d) <= period_year[1]
        )
    ]
    src_fnames = [f"{var_name}_{d}_GLEAM_{version}{suffix}" for d in src_dirs]
    src_paths = [
        os.path.join(home, d, f"{var_name}_{d}_GLEAM_{version}{suffix}")
        for d in src_dirs
    ]

    # get dst_lat, dst_lon from grid_shp
    dst_lon = [grid_shp.point_geometry[i].x for i in grid_shp.index]
    dst_lat = [grid_shp.point_geometry[i].y for i in grid_shp.index]

    # search grids and set date
    with xr.open_dataset(src_paths[0]) as dataset:

        # search grids
        src_lon = np.array(dataset.variables["lon"][:])
        src_lat = np.array(dataset.variables["lat"][:])

        searched_grids_index = search_grids.search_grids_equal(
            dst_lat, dst_lon, src_lat, src_lon
        )

    # set date and search date_period_index
    grid_df_list = []
    for i in tqdm(
        range(len(grid_shp.index)),
        desc="loop for grid to extract gleam_e_daily",
        colour="green",
    ):
        searched_grid_index = searched_grids_index[i]
        searched_grid_lat_index, searched_grid_lon_index = (
            searched_grid_index[0],
            searched_grid_index[1],
        )

        # loop for years to read
        for j in range(len(src_paths)):
            src_path = src_paths[j]
            grid_df = pd.DataFrame(columns=["date", var_name])

            with xr.open_dataset(src_path) as src_dataset:
                # set date
                date = pd.to_datetime(src_dataset["time"].values)
                date_str = date.strftime(date_format="%Y%m%d").to_numpy()
                date_period_index = np.where(
                    (date >= datetime.strptime(period[0], "%Y%m%d"))
                    & (date <= datetime.strptime(period[1], "%Y%m%d"))
                )[0]

                # loop for searched grids to cal grid data
                var_data = src_dataset[var_name].isel(
                    time=date_period_index,
                    lat=xr.DataArray(searched_grid_lat_index, dims="z"),
                    lon=xr.DataArray(searched_grid_lon_index, dims="z"),
                )
                var_data = var_data.values

                # add to df
                grid_df.loc[:, "date"] = date_str[date_period_index]
                grid_df.loc[:, var_name] = var_data

                # combine into grid_df_all
                if j == 0:
                    grid_df_all = grid_df
                else:
                    grid_df_all = pd.concat([grid_df_all, grid_df], axis=0)

        # reset index
        grid_df_all.index = list(range(len(grid_df_all)))

        # save in grid_df_list
        grid_df_list.append(grid_df_all)

    # save in grid_shp
    grid_shp.loc[:, var_name] = grid_df_list

    return grid_shp


if __name__ == "__main__":
    pass
