# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
from tqdm import *
from ...tools.geo_func import search_grids
from netCDF4 import Dataset, date2num
import numpy as np
import pandas as pd
from ...tools.nc_func.create_nc import copy_vattributefunc
from datetime import datetime, timedelta
import time


def combineTRMM_P():
    # general
    src_home = "E:/data/hydrometeorology/TRMM_P/TRMM_3B42/data"
    dst_home = "E:/data/hydrometeorology/TRMM_P/TRMM_3B42"
    suffix = ".nc4"
    
    src_names = [n for n in os.listdir(src_home) if n.endswith(suffix)]
    src_paths = [os.path.join(src_home, n) for n in src_names]
    
    # set time
    date_period = pd.date_range("19980101", "20191230", freq="1D")
    date_period_str = list(date_period.strftime("%Y%m%d"))
    date_period_datetime = [datetime.strptime(d, "%Y%m%d") for d in date_period_str]
    
    # create dataset
    dst_path = os.path.join(dst_home, f"3B42_Daily_combined{suffix}")
    dst_dataset = Dataset(dst_path, "w")
    
    # create time dimension
    dst_dataset.createDimension("time", len(src_names))
    dst_times = dst_dataset.createVariable("time", "f8", ("time", ))
    dst_times.units = "days since 1998-01-01"
    dst_times.calendar = "gregorian"
    dst_times[:] = date2num(date_period_datetime, units=dst_times.units, calendar=dst_times.calendar)
    
    # read one file to copy
    with Dataset(src_paths[0], "r") as src_dataset:
        # src dim variables
        src_lon = src_dataset.variables["lon"][:]
        src_lat = src_dataset.variables["lat"][:]
        
        # create lon/lat dimension
        dst_dataset.createDimension("lon", len(src_lon))
        dst_dataset.createDimension("lat", len(src_lat))
        
        # create variables
        dst_lon = dst_dataset.createVariable("lon", "f4", ("lon", ))
        dst_lat = dst_dataset.createVariable("lat", "f4", ("lat", ))
        dst_precipitation = dst_dataset.createVariable("precipitation", "f4", ("time", "lon", "lat", ), fill_value=src_dataset.variables["precipitation"].getncattr("_FillValue"))
        dst_precipitation_cnt = dst_dataset.createVariable("precipitation_cnt", "i2", ("time", "lon", "lat", ))
        dst_IRprecipitation = dst_dataset.createVariable("IRprecipitation", "f4", ("time", "lon", "lat", ), fill_value=src_dataset.variables["IRprecipitation"].getncattr("_FillValue"))
        dst_IRprecipitation_cnt = dst_dataset.createVariable("IRprecipitation_cnt", "i2", ("time", "lon", "lat", ))
        dst_HQprecipitation = dst_dataset.createVariable("HQprecipitation", "f4", ("time", "lon", "lat", ), fill_value=src_dataset.variables["HQprecipitation"].getncattr("_FillValue"))
        dst_HQprecipitation_cnt = dst_dataset.createVariable("HQprecipitation_cnt", "i2", ("time", "lon", "lat", ))
        dst_randomError = dst_dataset.createVariable("randomError", "f4", ("time", "lon", "lat", ), fill_value=src_dataset.variables["randomError"].getncattr("_FillValue"))
        dst_randomError_cnt = dst_dataset.createVariable("randomError_cnt", "i2", ("time", "lon", "lat", ))
        
        # set/copy lon/lat variables
        dst_lon[:] = src_dataset.variables["lon"][:]
        dst_lat[:] = src_dataset.variables["lat"][:]
        
        # copy variable attr
        copy_vattributefunc(src_dataset.variables["lon"], dst_lon)
        copy_vattributefunc(src_dataset.variables["lat"], dst_lat)
        copy_vattributefunc(src_dataset.variables["precipitation"], dst_precipitation)
        copy_vattributefunc(src_dataset.variables["precipitation_cnt"], dst_precipitation_cnt)
        copy_vattributefunc(src_dataset.variables["IRprecipitation"], dst_IRprecipitation)
        copy_vattributefunc(src_dataset.variables["IRprecipitation_cnt"], dst_IRprecipitation_cnt)
        copy_vattributefunc(src_dataset.variables["HQprecipitation"], dst_HQprecipitation)
        copy_vattributefunc(src_dataset.variables["HQprecipitation_cnt"], dst_HQprecipitation_cnt)
        copy_vattributefunc(src_dataset.variables["randomError"], dst_randomError)
        copy_vattributefunc(src_dataset.variables["randomError_cnt"], dst_randomError_cnt)
        
        # add time in variable attr
        add_time_attr = "time " + src_dataset.variables["precipitation"].getncattr("coordinates")
        dst_precipitation.setncattr("coordinates", add_time_attr)
        dst_precipitation_cnt.setncattr("coordinates", add_time_attr)
        dst_IRprecipitation.setncattr("coordinates", add_time_attr)
        dst_IRprecipitation_cnt.setncattr("coordinates", add_time_attr)
        dst_HQprecipitation.setncattr("coordinates", add_time_attr)
        dst_HQprecipitation_cnt.setncattr("coordinates", add_time_attr)
        dst_randomError.setncattr("coordinates", add_time_attr)
        dst_randomError_cnt.setncattr("coordinates", add_time_attr)
        
        # copy dataset attribute
        dst_dataset.FileHeader = src_dataset.FileHeader
        dst_dataset.InputPointer = src_dataset.InputPointer
        dst_dataset.title = src_dataset.title
        dst_dataset.ProductionTime = "2023-09-28"
        dst_dataset.history = src_dataset.history
        dst_dataset.description = f"combine daily TRMM P into one file, by XudongZheng, 28/09/2023T{time.ctime(time.time())}"
    
    # loop for file to read data
    for i in tqdm(range(len(date_period_str)), desc="loop for file to combine", colour="green"):
        src_path = os.path.join(src_home, f"3B42_Daily.{date_period_str[i]}.7.nc4{suffix}")
        with Dataset(src_path, "r") as src_dataset:
            dst_precipitation[i, :, :] = src_dataset.variables["precipitation"][:]
            dst_precipitation_cnt[i, :, :] = src_dataset.variables["precipitation_cnt"][:]
            dst_IRprecipitation[i, :, :] = src_dataset.variables["IRprecipitation"][:]
            dst_IRprecipitation_cnt[i, :, :] = src_dataset.variables["IRprecipitation_cnt"][:]
            dst_HQprecipitation[i, :, :] = src_dataset.variables["HQprecipitation"][:]
            dst_HQprecipitation_cnt[i, :, :] = src_dataset.variables["HQprecipitation_cnt"][:]
            dst_randomError[i, :, :] = src_dataset.variables["randomError"][:]
            dst_randomError_cnt[i, :, :] = src_dataset.variables["randomError_cnt"][:]

    dst_dataset.close()


def ExtractData(grid_shp, period=["19980101", "20111231"], var_name="precipitation"):
    # general
    src_path = "E:/data/hydrometeorology/TRMM_P/TRMM_3B42/3B42_Daily_combined.nc4"

    # date
    date_period = pd.date_range(start=period[0], end=period[1], freq='D')

    # get dst_lat, dst_lon from grid_shp
    dst_lon = [grid_shp.point_geometry[i].x for i in grid_shp.index]
    dst_lat = [grid_shp.point_geometry[i].y for i in grid_shp.index]

    # search grids
    with Dataset(src_path, "r") as dataset:
        src_lon = np.array(dataset.variables["lon"][:])
        src_lat = np.array(dataset.variables["lat"][:])
        searched_grids_index = search_grids.search_grids_equal(dst_lat, dst_lon, src_lat, src_lon)

        # loop for grid to extract TRMM P
        grid_df_list = []
        for i in tqdm(range(len(grid_shp.index)), desc="loop for grid to extract TRMM P", colour="green"):
            grid_df = pd.DataFrame(columns=["date", var_name])
            searched_grid_index = searched_grids_index[i]
            searched_grid_lat_index, searched_grid_lon_index = searched_grid_index[0][0], searched_grid_index[1][0]
            
            # set date and search date_period_index
            date = np.array(dataset.variables["time"][:])
            base_date = datetime(1998, 1, 1)  # since 1998-01-01
            date_datetime = np.array([base_date + timedelta(days=int(d)) for d in date])
            date_str = np.array([d.strftime("%Y%m%d") for d in date_datetime])
            date_period_index = np.where((date_datetime >= datetime.strptime(period[0], "%Y%m%d")) &
                                        (date_datetime <= datetime.strptime(period[1], "%Y%m%d")))[0]
            
            # loop for searched grids to cal grid data
            var_data = dataset.variables[var_name][date_period_index, searched_grid_lon_index, searched_grid_lat_index]
            var_data = var_data.filled(fill_value=np.nan)
            
            # add to df
            grid_df.loc[:, "date"] = date_str[date_period_index]
            grid_df.loc[:, var_name] = var_data

            # save in grid_df_list
            grid_df_list.append(grid_df)

        # save in grid_shp
        grid_shp.loc[:, var_name] = grid_df_list

    return grid_shp

if __name__ == "__main__":
    combineTRMM_P()