# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
from tqdm import *
from ...tools.geo_func import search_grids
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import xarray as xr
from tqdm import *
from functools import partial

def aggregateERA5_LandSM_Hourly2Daily(src_dir, dst_dir, variable_dir, aggregate_func):
    # path
    fdir = os.path.join(src_dir, variable_dir)
    fnames = [n for n in os.listdir(fdir) if n.endswith(".grib")]
    fpaths = [os.path.join(fdir, fn) for fn in fnames]
    
    # loop for months files to aggregate hourly to daily
    for i in tqdm(range(len(fpaths)), desc="loop for files to aggregate hourly to daily", colour="green"):
        # general
        fp = fpaths[i]
        
        # read
        with xr.open_dataset(fp, engine="cfgrib") as dataset:
            dataset_daily_ = dataset.resample(time="1D").reduce(aggregate_func)
            
        if i == 0:
            dataset_daily = dataset_daily_
        else:
            dataset_daily = xr.concat([dataset_daily, dataset_daily_], dim="time")
            
    # save
    dst_fname = fnames[0][:fnames[0].rfind("_") + 1] + "19800101_20141231" + ".nc4"
    
    dst_fpath = os.path.join(dst_dir, dst_fname)
    dataset_daily.to_netcdf(dst_fpath)
    
    # check size
    date_all = pd.date_range("19800101", "20141231", freq="1D")
    print(len(dataset_daily["time"]))
    print(date_all)
    
    # remove idx file
    idx_fnames = [n for n in os.listdir(fdir) if n.endswith(".idx")]
    idx_paths = [os.path.join(fdir, fn) for fn in idx_fnames]
    [os.remove(idx_path) for idx_path in idx_paths]
    

def aggregateERA5_LandSM_Hourly2Daily_func():
    src_dir = "E:/data/hydrometeorology/ERA5/ERA5_Land_hourly_data_from_1950_to_present/data_sm"
    dst_dir = "E:/data/hydrometeorology/ERA5/ERA5_Land_hourly_data_from_1950_to_present/data_sm_daily"
    variable_dir = "volumetric_soil_water_layer_4"
    aggregate_func = np.nanmean
    
    # read
    aggregateERA5_LandSM_Hourly2Daily(src_dir, dst_dir, variable_dir, aggregate_func)
    

def ExtractData(grid_shp, period=["19800101", "20101231"], var_name="1", aggregate_func=partial(np.nanmean, axis=1)):
    # general
    home = "E:/data/hydrometeorology/ERA5/ERA5_Land_hourly_data_from_1950_to_present/data_sm_daily"
    suffix = ".nc4"
    fpath = os.path.join(home, f"volumetric_soil_water_layer_{var_name}_19800101_20141231{suffix}")
    
    var_name = f"swvl{var_name}"
    
    # get dst_lat, dst_lon from grid_shp
    dst_lon = [grid_shp.point_geometry[i].x for i in grid_shp.index]
    dst_lat = [grid_shp.point_geometry[i].y for i in grid_shp.index]
    
    # set date and search date_period_index
    with xr.open_dataset(fpath) as dataset:
        # set date
        date = pd.to_datetime(dataset["time"].values)
        date_str = date.strftime(date_format="%Y%m%d").to_numpy()
        date_period_index = np.where((date >= datetime.strptime(period[0], "%Y%m%d")) &
                                    (date <= datetime.strptime(period[1], "%Y%m%d")))[0]

        # search grids
        src_lon = np.array(dataset["longitude"].values)
        src_lat = np.array(dataset["latitude"].values)
        
        searched_grids_index = search_grids.search_grids_radius_rectangle(dst_lat, dst_lon, src_lat, src_lon, 0.25/2, 0.25/2, leave=False)
        
        # loop for grid to extract GlobalSnow_SWE
        grid_df_list = []
        for i in tqdm(range(len(grid_shp.index)), desc="loop for grid to extract ERA5 SM", colour="green"):
            grid_df = pd.DataFrame(columns=["date", var_name])
            searched_grid_index = searched_grids_index[i]
            searched_grid_lat_index, searched_grid_lon_index = searched_grid_index[0], searched_grid_index[1]
            
            # loop for searched grids to cal grid data
            var_data = dataset[var_name].isel(time=date_period_index, latitude=xr.DataArray(searched_grid_lat_index, dims="z"), longitude=xr.DataArray(searched_grid_lon_index, dims="z"))
            var_data = var_data.values
            var_data = aggregate_func(var_data)
            
            # add to df
            grid_df.loc[:, "date"] = date_str[date_period_index]
            grid_df.loc[:, var_name] = var_data
            
            # save in grid_df_list
            grid_df_list.append(grid_df)
            
    # save in grid_shp
    grid_shp.loc[:, var_name] = grid_df_list
    
    return grid_shp

if __name__ == "__main__":
    pass


    
    
    
    
    