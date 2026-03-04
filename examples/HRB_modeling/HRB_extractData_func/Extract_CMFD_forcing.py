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
from netCDF4 import date2index

from easy_vic_build.tools.geo_func import search_grids
from easy_vic_build.tools.geo_func.create_gdf import CreateGDF
from easy_vic_build.tools.mete_func.mete_func import cal_VP_from_prs_sh
from easy_vic_build import logger
from multiprocessing import Pool

# TODO search once and reading multiple times

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
            

def ExtractData_month(
    chunk_yms,
    chunk_month_date, forcing_home, var_name, 
    grid_shp, searched_grids_index
):
    infix = "CMFD_V0106_B-01_03hr_010deg"
    var_name_casefold = var_name.casefold()
    fn = f"{var_name_casefold}_{infix}_{chunk_yms}.nc"
    fp = os.path.join(forcing_home, var_name, fn)
    
    ngrid = len(grid_shp)
    ntime = len(chunk_month_date)
    
    searched_resample_data_array = np.empty((ngrid, ntime), dtype=np.float32)  # len=len(grid_shp.index)
    
    with Dataset(fp, "r") as src_dataset:
        src_var = src_dataset.variables[var_name_casefold][:]

        for pos, (y_idx, x_idx) in tqdm(enumerate(searched_grids_index), total=ngrid, desc="reading grids"):
            searched_grid_data = src_var[:, y_idx, x_idx]  # shape = (ntime, n_points)
            searched_resample_data_array[pos, :] = np.nanmean(searched_grid_data, axis=1)
    
    # with Dataset(fp, "r") as src_dataset:
    #     src_var = src_dataset.variables[var_name_casefold]

    #     for pos, gi in tqdm(enumerate(grid_shp.index), desc="loop for grids to reading", colour="grey"):
    #         # get search grid index, lat, lon for this dst_grid
    #         searched_grid_index = searched_grids_index[pos]
            
    #         # get searched data
    #         searched_grid_data = [src_var[:, searched_grid_index[0][l], searched_grid_index[1][l]] for l in range(len(searched_grid_index[0]))]
            
    #         # resample
    #         searched_resample_data = np.nanmean(np.array(searched_grid_data), axis=0)
            
    #         # append
    #         searched_resample_data_array[pos, :] = searched_resample_data
    #         # searched_resample_data_list.append(searched_resample_data)
            
    #         # check
    #         # if check_search and j == 0 and i == 0:
    #         #     searched_grid_lat = [
    #         #         src_dataset.variables["lat"][searched_grid_index[0][j]].item()
    #         #         for j in range(len(searched_grid_index[0]))
    #         #     ]
    #         #     searched_grid_lon = [
    #         #         src_dataset.variables["lon"][searched_grid_index[1][j]].item()
    #         #         for j in range(len(searched_grid_index[0]))
    #         #     ]
                
    #         #     forcing_lat_res = 0.1
                
    #         #     cgdf = CreateGDF()
    #         #     grid_shp_grid = grid_shp.loc[[gi], "geometry"]
    #         #     searched_grids_gdf = cgdf.createGDF_rectangle_central_coord(
    #         #         searched_grid_lon, searched_grid_lat, forcing_lat_res
    #         #     )

    #         #     fig, ax = plt.subplots()
    #         #     grid_shp_grid.boundary.plot(ax=ax, edgecolor="r", linewidth=2)  # target
    #         #     searched_grids_gdf.plot(
    #         #         ax=ax, edgecolor="k", linewidth=0.2, facecolor="b", alpha=0.5
    #         #     )  # searched data from source data
                
    #         #     ax.set_title("check search")
                
    #         #     plt.show(block=True)

    return searched_resample_data_array


def process_date_chunk(args):
    var_name, chunk, grid_shp, searched_grids_index, forcing_home, unique_year_month_str, timestep = args
    
    chunk_unique_year_month_str = np.array(unique_year_month_str)[chunk].tolist()
    
    # get chunk starts and ends
    chunk_starts = pd.to_datetime([f"{ym}01" for ym in chunk_unique_year_month_str], format="%Y%m%d")
    chunk_ends = chunk_starts + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=21)
    
    # get chunk dates    
    chunk_dates = [pd.date_range(start, end, freq=timestep) for start, end in zip(chunk_starts, chunk_ends)]
    chunk_all_dates = [dt for sublist in chunk_dates for dt in sublist]
    
    # init
    chunk_results = np.empty((len(grid_shp.index), len(chunk_all_dates)), dtype=np.float32)
    
    # process
    cursor = 0
    for i, (chunk_yms, chunk_month_date) in enumerate(zip(chunk_unique_year_month_str, chunk_dates)):
        try:
            n_times = len(chunk_month_date)
            
            chunk_results[:, cursor:cursor + n_times] = ExtractData_month(
                chunk_yms,
                chunk_month_date, forcing_home, var_name, 
                grid_shp, searched_grids_index
            )
            
        except Exception as e:
            logger.error(f"Error at chunk_yms={chunk_yms}: {str(e)}")
            chunk_results[:, cursor:cursor + n_times] = -9999.0
        
        cursor += n_times
    
    return chunk_all_dates[0], chunk_results


def ExtractData(
    grid_shp, grid_shp_res=0.125,
    date_period=["20080101", "20181231"],
    search_method="radius_rectangle",
    timestep="3H",
    plot=False, check_search=False,
    N_PROCESS=8,
    CHUNK_SIZE=6,  # months (files)
):
    # general, 0.1deg, 3h
    forcing_home = "E:\\data\\hydrometeorology\\CMFD China Meteorological Forcing Dataset - 03-hourly (Version 1)\\Data_forcing_03hr_010deg"
    var_names = [
        "Prec",  #* mm/h # -> *3 -> mm/3h(step)
        "Pres",  #* Pa -> /1000 -> kPa
        "SHum",  # kg kg-1
        "SRad",  #* W m-2
        "LRad",  #* W m-2
        "Temp",  #* K -> -273.15 -> C
        # "TempMin",  # K -> -273.15 -> C
        # "TempMax",  # K -> -273.15 -> C
        "Wind",  #* m s-1
    ]  # need to derive: VP (kPa)
    
    var_names_casefold = [n.casefold() for n in var_names]
    
    date = pd.date_range(date_period[0], date_period[1], freq=timestep)
    date_to_index = {d: i for i, d in enumerate(date)}
    
    start_date = datetime.strptime(date_period[0], "%Y%m%d %H:%M:%S")
    end_date = datetime.strptime(date_period[1], "%Y%m%d %H:%M:%S")
    
    unique_year_month_str = sorted(date.to_period('M').strftime('%Y%m').unique().tolist())
    
    # np.arange(int(date_period[0][:4]), int(date_period[1][:4])+ 1, 1, dtype="int")
    # month_list = [f"0{m}" if m<10 else f"{m}" for m in range(1, 13)]
    # year_month_list = [f"{y}{m}" for y in year_list for m in month_list]
    
    infix = "CMFD_V0106_B-01_03hr_010deg"
    
    # set grids_lat, lon
    grids_lat = grid_shp.point_geometry.y.to_list()
    grids_lon = grid_shp.point_geometry.x.to_list()
    
    # read data to get lon, lat
    with Dataset(os.path.join(forcing_home, var_names[0], f"{var_names_casefold[0]}_{infix}_{unique_year_month_str[0]}.nc"), "r") as src_dataset:
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
    
    # initialize the array to hold results
    forcings_searched_resample_arrays = np.full(
        (len(var_names), len(grid_shp.index), len(date)),
        fill_value=-9999.0,
        dtype=np.float32
    )
    
    # loop to read NLDAS forcing data for each variable
    for vi, var_name in enumerate(var_names):
        logger.info(f"Processing {var_name}...")        
        
        chunks = [
            range(i, min(i+CHUNK_SIZE, len(unique_year_month_str))) 
            for i in range(0, len(unique_year_month_str), CHUNK_SIZE)
        ]

        with Pool(processes=N_PROCESS) as pool:
            tasks = [
                pool.apply_async(
                    process_date_chunk,
                    ((var_name, chunk, grid_shp, searched_grids_index, forcing_home, unique_year_month_str, timestep),)
                )
                for chunk in chunks
            ]
            
            with tqdm(total=len(chunks), desc=f"{var_name} Progress") as pbar:
                for task in tasks:
                    start_date, chunk_res = task.get()
                    start_di = date_to_index[start_date]
                    end_di = start_di + chunk_res.shape[1]
                    forcings_searched_resample_arrays[vi, :, start_di:end_di] = chunk_res
                    pbar.update(1)
                    
    # save
    for j in range(len(var_names)):
        forcings_searched_resample_arrays_v = forcings_searched_resample_arrays[j]
        # [v1, ..., v5], v1 = [grid1, ..., gridn], grid1 = [time1, ..., timek] (series)
        grid_shp[f"{var_names[j]}"] = list(forcings_searched_resample_arrays_v)
    
    # postprocessing: unit change
    # Prec: mm/h -> mm / 3h
    grid_shp["Prec"] = grid_shp["Prec"].apply(lambda row: np.array(row) * 3)  # mm/h -> mm/3h
    
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
            "Prec": "pre_mm_per_3h",  # mm/3h
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
        plot_var_name = "tmp_avg_C"
        
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