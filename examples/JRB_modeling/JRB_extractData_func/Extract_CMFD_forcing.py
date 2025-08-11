# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import *
from netCDF4 import Dataset, num2date
from datetime import datetime
import pandas as pd
from copy import deepcopy
import cftime
import xarray as xr

from easy_vic_build.tools.geo_func import search_grids
from easy_vic_build.tools.geo_func.create_gdf import CreateGDF
from easy_vic_build.tools.mete_func.mete_func import cal_VP_from_prs_sh
from easy_vic_build import logger


def combine_CMDF():
    # general, 0.1deg, 3H
    home = "E:\\data\\hydrometeorology\\CMFD China Meteorological Forcing Dataset - 03-hourly (Version 1)\\Data_forcing_03hr_010deg"
    var_names = [
        "LRad",
        "Prec",
        "Pres",
        "SHum",
        "SRad",
        "Temp",
        "Wind",
    ]
    
    for var_name in tqdm(var_names, desc="loop for combining CMDF nc files"):
        fpaths = [os.path.join(home, var_name, fn) for fn in os.listdir(os.path.join(home, var_name)) if fn.endswith('.nc')]
        
        # sort
        fpaths.sort()
        
        # combine
        ds = xr.open_mfdataset(fpaths, combine='nested', concat_dim='time')
        
        # save
        dst_path = os.path.join(home, var_name, var_name.casefold() + "_combined.nc")
        ds.to_netcdf(dst_path)


def resample_to_daily_CMDF():
    # general, 0.1deg
    home = "E:\\data\\hydrometeorology\\CMFD China Meteorological Forcing Dataset - 03-hourly (Version 1)\\Data_forcing_03hr_010deg"
    var_names = [
        "LRad",  # W m-2
        "Prec",  # mm/h
        "Pres",  # Pa
        "SHum",  # kg kg-1
        "SRad",  # W m-2
        "Temp",  # K
        "Wind",  # m s-1
    ]
    
    for var_name in tqdm(var_names, desc="loop for resample CMDF to daily"):
        fpaths = os.path.join(home, var_name, var_name.casefold() + "_combined.nc")
        
        # open
        src_dataset = xr.open_dataset(fpaths)

        # resample
        if var_name == "Temp":
            dst_dataset_daily_mean = src_dataset.resample(time='1D').mean()
            dst_dataset_daily_min = src_dataset.resample(time='1D').min()
            dst_dataset_daily_max = src_dataset.resample(time='1D').max()
            
            # save
            dst_path_mean = os.path.join(home, var_name, "temp" + "_combined_daily.nc")
            dst_path_min = os.path.join(home, var_name, "tempmax" + "_combined_daily.nc")
            dst_path_max = os.path.join(home, var_name, "tempmin" + "_combined_daily.nc")
            
            dst_dataset_daily_mean.to_netcdf(dst_path_mean)
            dst_dataset_daily_min.to_netcdf(dst_path_min)
            dst_dataset_daily_max.to_netcdf(dst_path_max)
            
        else:
            dst_dataset_daily = src_dataset.resample(time='1D').mean()
            # save
            dst_path = os.path.join(home, var_name, var_name.casefold() + "_combined_daily.nc")
            dst_dataset_daily.to_netcdf(dst_path)
            
    
def ExtractData(
    grid_shp, grid_shp_res=0.125,
    date_period=["20080101", "20181231"],
    search_method="radius_rectangle", 
    plot=False, check_search=False
):
    # general, 0.1deg, daily (already transfer into daily)
    home = "E:\\data\\hydrometeorology\\CMFD China Meteorological Forcing Dataset - 03-hourly (Version 1)\\Data_forcing_03hr_010deg"
    var_names = [
        "Prec",  #* mm/h -> *24 -> mm/d
        "Pres",  #* Pa -> /1000 -> kPa
        "SHum",  # kg kg-1
        "SRad",  #* W m-2
        "LRad",  #* W m-2
        "Temp",  #* K -> -273.15 -> C
        "TempMin",  # K -> -273.15 -> C
        "TempMax",  # K -> -273.15 -> C
        "Wind",  #* m s-1
    ]  # need to derive: VP (kPa)
    
    var_names_casefold = [n.casefold() for n in var_names]
    
    year_list = np.arange(int(date_period[0][:4]), int(date_period[1][:4])+ 1, 1, dtype="int")
    month_list = [f"0{m}" if m<10 else f"{m}" for m in range(1, 13)]
    year_month_list = [f"{y}{m}" for y in year_list for m in month_list]
    
    date = pd.date_range(date_period[0], date_period[1], freq="D")
    start_date = datetime.strptime(date_period[0], "%Y%m%d %H:%M:%S")
    end_date = datetime.strptime(date_period[1], "%Y%m%d %H:%M:%S")
    
    infix = "CMFD_V0106_B-01_03hr_010deg"
    
    # set grids_lat, lon
    grids_lat = grid_shp.point_geometry.y.to_list()
    grids_lon = grid_shp.point_geometry.x.to_list()
    
    # read data to get lon, lat
    with Dataset(os.path.join(home, var_names[0], f"{var_names_casefold[0]}_{infix}_{year_month_list[0]}.nc"), "r") as src_dataset:
        # get lat, lon
        forcing_lat = src_dataset.variables["lat"][:]
        forcing_lon = src_dataset.variables["lon"][:]
        
        # get res
        forcing_lat_res = (max(forcing_lat) - min(forcing_lat)) / (len(forcing_lat) - 1)  # 1/10 deg
        forcing_lon_res = (max(forcing_lon) - min(forcing_lon)) / (len(forcing_lon) - 1)
    
    # search grids  
    logger.info("searching grids for CDMet forcing data... ...")
    
    # source data res: 0.1 deg
    if search_method == "radius_rectangle":
        searched_grids_index = search_grids.search_grids_radius_rectangle(
            dst_lat=grids_lat,
            dst_lon=grids_lon,
            src_lat=forcing_lat,
            src_lon=forcing_lon,
            lat_radius=grid_shp_res / 2,
            lon_radius=grid_shp_res / 2,
        )
        
    elif search_method == "radius_rectangle_reverse":
        searched_grids_index = search_grids.search_grids_radius_rectangle_reverse(
            dst_lat=grids_lat,
            dst_lon=grids_lon,
            src_lat=forcing_lat,
            src_lon=forcing_lon,
            lat_radius=forcing_lat_res / 2,
            lon_radius=forcing_lon_res / 2,
        )
    
    elif search_method == "nearest":
        searched_grids_index = search_grids.search_grids_nearest(dst_lat=grids_lat, dst_lon=grids_lon,
                                                                src_lat=forcing_lat, src_lon=forcing_lon,
                                                                search_num=1,
                                                                move_src_lat=None, move_src_lon=None)
    else:
        logger.warning(f"search method {search_method} not supported")
    
    # read forcing for each grid
    forcings_searched_resample_Series = [[] for _ in range(len(var_names))]
    
    for j in range(len(var_names)):
        forcings_searched_resample_Series_v = forcings_searched_resample_Series[j]
        
        # read data
        with Dataset(os.path.join(home, var_names[j], f"{var_names_casefold[j]}_combined_daily.nc"), "r") as src_dataset:
            # get time index
            src_time = src_dataset.variables["time"]
            src_dates = num2date(src_time[:], units=src_time.units, calendar=src_time.calendar)
            start_index = np.where(src_dates >= start_date)[0][0]
            end_index = np.where(src_dates <= end_date)[0][-1] + 1
            
            # get data
            var_names_nc = var_names_casefold[j] if var_names_casefold[j] not in ["tempmax", "tempmin"] else "temp"
            forcing_data_j = src_dataset.variables[var_names_nc][start_index:end_index, :, :]
        
            for i in tqdm(
                grid_shp.index,
                colour="green",
                desc=f"loop for each grid to extract forcing data",
            ):
                # get search grid index, lat, lon for this dst_grid
                searched_grid_index = searched_grids_index[i]
                dst_lat_grid = grid_shp.loc[i, :].point_geometry.y
                dst_lon_grid = grid_shp.loc[i, :].point_geometry.x
                
                searched_grid_lat = [
                    forcing_lat[searched_grid_index[0][j]]
                    for j in range(len(searched_grid_index[0]))
                ]
                
                searched_grid_lon = [
                    forcing_lon[searched_grid_index[1][j]]
                    for j in range(len(searched_grid_index[0]))
                ]
                
                # get searched data
                searched_grid_data = [forcing_data_j[:, searched_grid_index[0][l], searched_grid_index[1][l]] for l in range(len(searched_grid_index[0]))]
                
                searched_grid_data = np.array(searched_grid_data)
                
                searched_grid_data = searched_grid_data.T  # time * searched_grids
            
                # resample
                searched_resample_data_series_v = np.nanmean(
                    searched_grid_data,
                    axis=1,
                )
                
                # append
                forcings_searched_resample_Series_v.append(searched_resample_data_series_v)
            
                # check
                if check_search and j == 0 and i == 0:
                    cgdf = CreateGDF()
                    grid_shp_grid = grid_shp.loc[[i], "geometry"]
                    searched_grids_gdf = cgdf.createGDF_rectangle_central_coord(
                        searched_grid_lon, searched_grid_lat, forcing_lat_res
                    )

                    fig, ax = plt.subplots()
                    grid_shp_grid.boundary.plot(ax=ax, edgecolor="r", linewidth=2)  # target
                    searched_grids_gdf.plot(
                        ax=ax, edgecolor="k", linewidth=0.2, facecolor="b", alpha=0.5
                    )  # searched data from source data
                    
                    ax.set_title("check search")
                    
                    plt.show(block=True)
            
    # save
    for j in range(len(var_names)):
        # [v1, ..., v5], v1 = [grid1, ..., gridn], grid1 = [time1, ..., timek] (series)
        grid_shp[f"{var_names[j]}"] = forcings_searched_resample_Series[j]
    
    # postprocessing: unit change
    # Prec: mm/h -> *24 -> mm/d
    grid_shp["Prec"] = grid_shp["Prec"].apply(lambda row: np.array(row) * 24)  # mm/h -> mm/d
    
    # Pres: Pa -> /1000 -> kPa
    grid_shp["Pres"] = grid_shp["Pres"].apply(lambda row: np.array(row) / 1000.0)  # Pa to kPa
    
    # Temp: K -> -273.15 -> C
    grid_shp["Temp"] = grid_shp["Temp"].apply(lambda row: np.array(row) - 273.15)  # K to C
    
    # calculate VP, kPa
    def compute_vp_series(row):
        prs_kPa = row["Pres"]
        sh_kg_per_kg = row["SHum"]
        
        vp_series = [
            cal_VP_from_prs_sh(prs_kPa_day, sh_kg_per_kg_day) for prs_kPa_day, sh_kg_per_kg_day in zip(prs_kPa, sh_kg_per_kg)
        ]
        return vp_series

    grid_shp["VP"] = grid_shp.apply(compute_vp_series, axis=1)
    
    # rename columns to add units
    grid_shp.rename(
        columns={
            "Temp": "tmp_avg_C",  # C
            "Prec": "pre_mm_per_day",  # mm/day
            "Pres": "prs_kPa",  # kPa
            "SRad": "swd_W_per_m2",  # W m-2
            "SHum": "shu_kg_per_kg",  # kg/kg
            "VP": "vp_kPa",  # kPa
            "LRad": "lwd_W_per_m2",  # W m-2
            "Wind": "wind_m_per_s",  # m/s
        },
        inplace=True
    )
    
    # plot
    if plot:
        # plot timeseries
        grid_i = 0
        plot_var_name = "VP"
        
        plt.figure(figsize=(10, 6))
        plt.plot(date, grid_shp.loc[grid_shp.index[grid_i], f"{plot_var_name}"], label=plot_var_name)
        plt.xlabel("Time")
        plt.ylabel(plot_var_name)
        plt.legend()
        plt.title(f"Time Series of {plot_var_name} at Grid {grid_i}")
        plt.show(block=True)
        
        # plot map
        fig, ax = plt.subplots()
        
        grid_shp_plot = deepcopy(grid_shp)
        grid_shp_plot[f"{plot_var_name}_timemean"] = grid_shp_plot.apply(
            lambda row: np.nanmean(row[f"{plot_var_name}"]), axis=1
        )
        
        grid_shp_plot.plot(
            f"{plot_var_name}_timemean",
            ax=ax,
            edgecolor="k",
            linewidth=0.2,
        )
        ax.set_title(f"{plot_var_name} mean")
        ax.set_xlim(
            [min(grids_lon) - grid_shp_res / 2, max(grids_lon) + grid_shp_res / 2]
        )
        ax.set_ylim(
            [min(grids_lat) - grid_shp_res / 2, max(grids_lat) + grid_shp_res / 2]
        )
        
        plt.show(block=True)

    return grid_shp
        
    

if __name__ == "__main__":
    # combine_CMDF()
    # resample_to_daily_CMDF()
    pass