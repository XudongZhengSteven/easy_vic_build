# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
import pandas as pd
import numpy as np
from ...tools.geo_func import search_grids
from datetime import datetime, timedelta
from netCDF4 import Dataset, date2num, num2date
from ...tools.nc_func.create_nc import copy_vattributefunc, copy_garrtibutefunc
import time
from tqdm import *
from pyproj import CRS, Transformer
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from Decorators.decoractors import apply_along_axis_decorator
import xarray as xr

def combineGlobalSnow_SWE():
    # general
    src_home = "E:/data/hydrometeorology/globalsnow/archive_v3.0/L3A_daily_SWE/NetCDF4"
    dst_home = "E:/data/hydrometeorology/globalsnow/archive_v3.0/L3A_daily_SWE"
    suffix = ".nc"
    
    src_names = [n for n in os.listdir(src_home) if n.endswith(suffix)]
    src_paths = [os.path.join(src_home, n) for n in src_names]
    
    # set time
    str_date = [n[:n.find("_")] for n in src_names]
    date_period_datetime = [datetime.strptime(d, "%Y%m%d") for d in str_date]
    
    # create dataset
    dst_path = os.path.join(dst_home, f"GlobalSnow_swe_northern_hemisphere_swe_0.25grid_combined{suffix}")
    dst_dataset = Dataset(dst_path, "w")
    
    # create time dimension
    dst_dataset.createDimension("time", len(src_names))
    dst_times = dst_dataset.createVariable("time", "f8", ("time", ))
    dst_times.units = "days since 1979-01-06"
    dst_times.calendar = "gregorian"
    dst_times[:] = date2num(date_period_datetime, units=dst_times.units, calendar=dst_times.calendar)
    
    # read one file to copy
    with Dataset(src_paths[0], "r") as src_dataset:
        src_x = src_dataset.variables["x"][:]
        src_y = src_dataset.variables["y"][:]
        
        # create lon/lat dimension
        dst_dataset.createDimension("x", len(src_x))
        dst_dataset.createDimension("y", len(src_y))
        
        # create variables
        dst_x = dst_dataset.createVariable("x", "f8", ("x", ))
        dst_y = dst_dataset.createVariable("y", "f8", ("y", ))
        dst_swe = dst_dataset.createVariable("swe", "i4", ("time", "y", "x", ),
                                             fill_value=src_dataset.variables["swe"].getncattr("_FillValue"))
        dst_crs = dst_dataset.createVariable("crs", "S1", )
        
        # set/copy lon/lat variables
        dst_x[:] = src_dataset.variables["x"][:]
        dst_y[:] = src_dataset.variables["y"][:]
        
        # copy variable attr
        copy_vattributefunc(src_dataset.variables["x"], dst_x)
        copy_vattributefunc(src_dataset.variables["y"], dst_y)
        copy_vattributefunc(src_dataset.variables["swe"], dst_swe)
        copy_vattributefunc(src_dataset.variables["crs"], dst_crs)
        
        # copy dataset attribute
        copy_garrtibutefunc(src_dataset, dst_dataset)
        dst_dataset.description = f"combine GlobalSnow SWE into one file, by XudongZheng, 08/10/2023T{time.ctime(time.time())}"
    
    # loop for file to read data
    for i in tqdm(range(len(src_paths)), desc="loop for file to combine", colour="green"):
        src_path = src_paths[i]
        with Dataset(src_path, "r") as src_dataset:
            dst_swe[i, :, :] = src_dataset.variables["swe"][:]

    dst_dataset.close()


def tranform_GlobalSnow_SWE_to_WGS84():
    # general
    src_home = "E:/data/hydrometeorology/globalsnow/archive_v3.0/L3A_daily_SWE/NetCDF4"
    dst_home = "E:/data/hydrometeorology/globalsnow/archive_v3.0/L3A_daily_SWE"
    sub_dir = "NetCDF4_WGS84_nearest"
    suffix = ".nc"
    src_names = [n for n in os.listdir(src_home) if n.endswith(suffix)]
    src_paths = [os.path.join(src_home, n) for n in src_names]
    dst_paths = [os.path.join(src_home, sub_dir, n.replace(suffix, ".tif")) for n in src_names]
    
    # set crs
    dst_crs = CRS.from_epsg(4326)
    src_crs = CRS.from_epsg(3408)
    
    # set res
    dst_resolution = 0.25
    
    for i in tqdm(range(len(src_paths)), desc="loop for files to projection into WGS84"):
        src_path = src_paths[i]
        dst_path = dst_paths[i]
    
        with rasterio.open(src_path, "r") as src_dataset:
            transform, width, height = calculate_default_transform(src_crs, dst_crs, src_dataset.width, src_dataset.height, *src_dataset.bounds, resolution=dst_resolution)
            kwargs = src_dataset.meta.copy()
            kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'driver': "GTiff",
            })
        
            with rasterio.open(dst_path, 'w', **kwargs) as dst_dataset:
                for i in tqdm(range(1, src_dataset.count + 1)):
                    reproject(source=rasterio.band(src_dataset, i),
                        destination=rasterio.band(dst_dataset, i),
                        src_transform=src_dataset.transform,
                        src_crs=src_crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest,
                        dst_resolution=dst_resolution)  # using nearest due the code meaning
            

def combineGlobalSnow_SWE_WGS84():
    # general
    sub = "nearest"  # bilinear
    sub_dir = f"NetCDF4_WGS84_{sub}"
    src_home = f"E:/data/hydrometeorology/globalsnow/archive_v3.0/L3A_daily_SWE/NetCDF4/{sub_dir}"
    src_nc_reference = "E:/data/hydrometeorology/globalsnow/archive_v3.0/L3A_daily_SWE/GlobalSnow_swe_northern_hemisphere_swe_0.25grid_combined.nc"
    src_tif_reference = f"E:/data/hydrometeorology/globalsnow/archive_v3.0/L3A_daily_SWE/NetCDF4/NetCDF4_WGS84_{sub}/19790106_northern_hemisphere_swe_0.25grid.tif"
    dst_home = "E:/data/hydrometeorology/globalsnow/archive_v3.0/L3A_daily_SWE"
    suffix = ".tif"
    
    src_names = [n for n in os.listdir(src_home) if n.endswith(suffix)]
    src_paths = [os.path.join(src_home, n) for n in src_names]
    
    # create dataset
    dst_path = os.path.join(dst_home, f"GlobalSnow_swe_northern_hemisphere_swe_0.25grid_combined_WGS84_{sub}.nc")
    dst_dataset = Dataset(dst_path, "w")
    
    # got lat lon
    with rasterio.open(src_tif_reference, "r") as src_dataset:
        src_lat = np.linspace(src_dataset.bounds[1], src_dataset.bounds[3], src_dataset.height)
        src_lon = np.linspace(src_dataset.bounds[0], src_dataset.bounds[2], src_dataset.width)
    
    # read one file to copy
    with Dataset(src_nc_reference, "r") as src_dataset:
        src_time = src_dataset.variables["time"][:]
        
        # create dimension
        dst_dataset.createDimension("time", len(src_time))
        dst_dataset.createDimension("lat", len(src_lat))
        dst_dataset.createDimension("lon", len(src_lon))
        
        # create variables
        dst_lat = dst_dataset.createVariable("lat", "f8", ("lat", ), fill_value=np.NaN)
        dst_lon = dst_dataset.createVariable("lon", "f8", ("lon", ), fill_value=np.NaN)
        dst_swe = dst_dataset.createVariable("swe", "f4", ("time", "lat", "lon", ),
                                             fill_value=src_dataset.variables["swe"].getncattr("_FillValue"))
        dst_time = dst_dataset.createVariable("time", "f8", ("time", ))
        
        # set/copy lon/lat variables
        dst_lat[:] = src_lat
        dst_lon[:] = src_lon
        dst_time[:] = src_time
        
        # copy variable attr
        copy_vattributefunc(src_dataset.variables["swe"], dst_swe)
        copy_vattributefunc(src_dataset.variables["time"], dst_time)
        dst_swe.delncattr("grid_mapping")
        dst_lon.setncattr("units", "degrees")
        dst_lat.setncattr("units", "degrees")
        
        # copy dataset attribute
        # copy_garrtibutefunc(src_dataset, dst_dataset)
        dst_dataset.description = f"combine GlobalSnow SWE WGS84 into one file, by XudongZheng, 08/10/2023T{time.ctime(time.time())}"

        # loop for file to read data
        for i in tqdm(range(len(src_paths)), desc="loop for file to combine", colour="green"):
            src_path = src_paths[i]
            with rasterio.open(src_path, "r") as src_dataset:
                data = src_dataset.read()
                dst_swe[i, :, :] = np.flip(data, axis=1)  # !flip, because the array in rasterio flip in y
        
    dst_dataset.close()

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


def ExtractData(grid_shp, period=["19980101", "20111231"], var_name="swe", aggregate_func=aggregate_func_SWE_axis1):    
    # general
    src_path = "E:/data/hydrometeorology/globalsnow/archive_v3.0/L3A_daily_SWE/GlobalSnow_swe_northern_hemisphere_swe_0.25grid_combined_WGS84_nearest.nc"
    
    # get dst_lat, dst_lon from grid_shp
    dst_lon = [grid_shp.point_geometry[i].x for i in grid_shp.index]
    dst_lat = [grid_shp.point_geometry[i].y for i in grid_shp.index]
    
    # set date and search date_period_index
    with xr.open_dataset(src_path) as dataset:
        # set date
        date = pd.to_datetime(dataset["time"].values)
        date_str = date.strftime(date_format="%Y%m%d").to_numpy()
        date_period_index = np.where((date >= datetime.strptime(period[0], "%Y%m%d")) &
                                    (date <= datetime.strptime(period[1], "%Y%m%d")))[0]

        # search grids
        src_lon = np.array(dataset["lon"].values)
        src_lat = np.array(dataset["lat"].values)
        
        searched_grids_index = search_grids.search_grids_radius_rectangle(dst_lat, dst_lon, src_lat, src_lon, 0.25/2, 0.25/2)
        
        # loop for grid to extract GlobalSnow_SWE
        grid_df_list = []
        for i in tqdm(range(len(grid_shp.index)), desc="loop for grid to extract GlobalSnow_SWE", colour="green"):
            grid_df = pd.DataFrame(columns=["date", var_name])
            searched_grid_index = searched_grids_index[i]
            searched_grid_lat_index, searched_grid_lon_index = searched_grid_index[0], searched_grid_index[1]
            
            # loop for searched grids to cal grid data
            var_data = dataset[var_name].isel(time=date_period_index, lat=xr.DataArray(searched_grid_lat_index, dims="z"), lon=xr.DataArray(searched_grid_lon_index, dims="z"))
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
    # combineGlobalSnow_SWE()
    # tranform_GlobalSnow_SWE_to_WGS84()
    # combineGlobalSnow_SWE_WGS84()
    pass