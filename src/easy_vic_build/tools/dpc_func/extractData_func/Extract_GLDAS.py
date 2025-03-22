# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import *

from ...geo_func import search_grids


def combineGLDAS(var_name="CanopInt_tavg"):
    # general
    home = "H:/data/GLDAS/GLDAS_025_daily_1948_2014_global_CLSM/daily_data"
    suffix = ".nc4"

    src_fnames = np.array(
        [n for n in os.listdir(home) if n.endswith(suffix)], dtype=str
    )
    src_paths = np.array([os.path.join(home, n) for n in src_fnames], dtype=str)

    # split
    src_fnames_list = np.split(src_fnames[:-11], 10)
    src_fnames_list.append(src_fnames[-11:])

    src_fpaths_list = np.split(src_paths[:-11], 10)
    src_fpaths_list.append(src_paths[-11:])

    # loop for each split list
    for j in range(len(src_fpaths_list)):
        src_paths_ = src_fpaths_list[j]
        src_fnames_ = src_fnames_list[j]

        # loop for months files to aggregate hourly to daily
        for i in tqdm(
            range(len(src_paths_)),
            desc="loop for files to combineGLDAS",
            colour="green",
        ):
            # general
            src_path = src_paths_[i]

            # read
            with xr.open_dataset(src_path) as dataset:
                dataset_var_ = dataset[var_name]

            if i == 0:
                dataset_var = dataset_var_
            else:
                dataset_var = xr.concat([dataset_var, dataset_var_], dim="time")

        # save
        start = re.search(r"\d{8}", src_fnames_[0])[0]
        end = re.search(r"\d{8}", src_fnames_[-1])[0]
        dst_fname = f"GLDAS_CLSM025_D.A{start}_{end}_{var_name}_combine.020.nc4"

        dst_fpath = os.path.join(
            "H:/data/GLDAS/GLDAS_025_daily_1948_2014_global_CLSM/", dst_fname
        )
        dataset_var.to_netcdf(dst_fpath)

        dataset_var.close()


def ExtractData(grid_shp, period=["19800101", "20101231"], var_name="CanopInt_tavg"):
    # general
    home = "E:/data/hydrometeorology/GLDAS/GLDAS_025_daily_1948_2014_global_CLSM/CanopInt_tavg"
    suffix = ".nc4"

    src_fnames = np.array([n for n in os.listdir(home) if n.endswith(suffix)])
    src_paths = np.array([os.path.join(home, n) for n in src_fnames])

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

    # set date
    src_datasets = [xr.open_dataset(src_path) for src_path in src_paths]

    # loop for grid to read
    grid_df_list = []
    for i in range(len(grid_shp.index)):
        grid_df = pd.DataFrame(columns=["date", var_name])
        searched_grid_index = searched_grids_index[i]
        searched_grid_lat_index, searched_grid_lon_index = (
            searched_grid_index[0],
            searched_grid_index[1],
        )

        # set date
        date_str_all = np.array([], dtype=object)
        var_data_all = np.array([], dtype=float)
        var_data_all = var_data_all.reshape((-1, 1))

        for j in range(len(src_datasets)):
            src_dataset = src_datasets[j]
            src_date = pd.to_datetime(src_dataset["time"].values)
            date_str = src_date.strftime(date_format="%Y%m%d").to_numpy()
            date_period_index = np.where(
                (src_date >= datetime.strptime(period[0], "%Y%m%d"))
                & (src_date <= datetime.strptime(period[1], "%Y%m%d"))
            )[0]
            if len(date_period_index) > 0:
                date_str = date_str[date_period_index]

                var_data = src_dataset[var_name].isel(
                    time=date_period_index,
                    lat=xr.DataArray(searched_grid_lat_index, dims="z"),
                    lon=xr.DataArray(searched_grid_lon_index, dims="z"),
                )
                var_data = var_data.values

                date_str_all = np.concatenate([date_str_all, date_str], axis=0)
                var_data_all = np.concatenate([var_data_all, var_data], axis=0)

            else:
                continue

        # add to df
        grid_df.loc[:, "date"] = date_str_all
        grid_df.loc[:, var_name] = var_data_all

        # save in grid_df_list
        grid_df_list.append(grid_df)

    # save in grid_shp
    grid_shp.loc[:, var_name] = grid_df_list

    # close
    [dataset.close() for dataset in src_datasets]

    return grid_shp


if __name__ == "__main__":
    # combineGLDAS()
    pass
