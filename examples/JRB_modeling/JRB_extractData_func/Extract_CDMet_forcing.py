# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import *
from netCDF4 import Dataset
import pandas as pd
from copy import deepcopy

from easy_vic_build.tools.geo_func import search_grids
from easy_vic_build.tools.geo_func.create_gdf import CreateGDF

from easy_vic_build.tools.mete_func.mete_func import cal_SWDOWN_Angstrom_Prescott_eq, cal_es_Tetens_eq
from easy_vic_build.tools.mete_func.mete_func import cal_VP_from_RH_es, cal_LWDOWN_Brutsaert_eq, cal_LWDOWN_CD99_eq, cal_cloud_fraction_from_ssd

from easy_vic_build import logger

def ExtractData(
    grid_shp, grid_shp_res=0.125,
    date_period=["20080101", "20181231"],
    search_method="radius_rectangle", 
    plot=False, check_search=False
):
    # general, 4km, daily
    home = "E:\\data\\hydrometeorology\\CDMet (4 km daily gridded meteorological dataset for China (2000-2020))\\2008_2018"
    prefix = "CDMet"
    var_names = [
        "meantmp", #* K, -> -273.15 -> C
        # "maxtmp",  # K, -> -273.15 -> C
        # "mintmp",  # K, -> -273.15 -> C
        "pre",  #* mm/d
        "prs",  #* hPa, -> /10.0 -> kPa
        "rhu",  # percentage
        "ssd",  # h
        "win"  #* m/s
    ]  # need to derive: SWDOWN (W m-2), LWDOWN (W m-2), VP (kPa)
    
    year_list = np.arange(int(date_period[0][:4]), int(date_period[1][:4])+ 1, 1, dtype="int")
    date = pd.date_range(date_period[0], date_period[1], freq="D")
    
    # set grids_lat, lon
    grids_lat = grid_shp.point_geometry.y.to_list()
    grids_lon = grid_shp.point_geometry.x.to_list()
    
    # read data to get lon, lat
    with Dataset(os.path.join(home, f"{prefix}_{var_names[0]}_{year_list[0]}.nc"), "r") as src_dataset:
        # get lat, lon
        forcing_lat = src_dataset.variables["lat"][:]
        forcing_lon = src_dataset.variables["lon"][:]
        
        # get res
        forcing_lat_res = (max(forcing_lat) - min(forcing_lat)) / (len(forcing_lat) - 1)  # 4km
        forcing_lon_res = (max(forcing_lon) - min(forcing_lon)) / (len(forcing_lon) - 1)
        
        # clip: extract before to improve speed
        xindex_start = np.where(forcing_lon <= min(grids_lon) - grid_shp_res)[0][-1]
        xindex_end = np.where(forcing_lon >= max(grids_lon) + grid_shp_res)[0][0]

        yindex_start = np.where(forcing_lat >= max(grids_lat) + grid_shp_res)[0][-1]  # large -> small
        yindex_end = np.where(forcing_lat <= min(grids_lat) - grid_shp_res)[0][0]

        forcing_lon_clip = forcing_lon[xindex_start : xindex_end + 1]
        forcing_lat_clip = forcing_lat[yindex_start : yindex_end + 1]
    
    # search grids
    logger.info("searching grids for CDMet forcing data... ...")
    
    # source data res: 4km
    if search_method == "radius_rectangle":
        searched_grids_index = search_grids.search_grids_radius_rectangle(
            dst_lat=grids_lat,
            dst_lon=grids_lon,
            src_lat=forcing_lat_clip,
            src_lon=forcing_lon_clip,
            lat_radius=grid_shp_res / 2,
            lon_radius=grid_shp_res / 2,
        )
        
    elif search_method == "radius_rectangle_reverse":
        searched_grids_index = search_grids.search_grids_radius_rectangle_reverse(
            dst_lat=grids_lat,
            dst_lon=grids_lon,
            src_lat=forcing_lat_clip,
            src_lon=forcing_lon_clip,
            lat_radius=forcing_lat_res / 2,
            lon_radius=forcing_lon_res / 2,
        )
    
    elif search_method == "nearest":
        searched_grids_index = search_grids.search_grids_nearest(dst_lat=grids_lat, dst_lon=grids_lon,
                                                                src_lat=forcing_lat_clip, src_lon=forcing_lon_clip,
                                                                search_num=1,
                                                                move_src_lat=None, move_src_lon=None)
    else:
        logger.warning(f"search method {search_method} not supported")
        
    # read forcing for each grid
    forcings_searched_resample_Series = [[] for _ in range(len(var_names))]
    
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
            forcing_lat_clip[searched_grid_index[0][j]]
            for j in range(len(searched_grid_index[0]))
        ]
        searched_grid_lon = [
            forcing_lon_clip[searched_grid_index[1][j]]
            for j in range(len(searched_grid_index[0]))
        ]
        
        # loop for get searched data
        for j in range(len(var_names)):
            forcings_searched_resample_Series_v = forcings_searched_resample_Series[j]
            
            for k in range(len(year_list)):
                # read data
                with Dataset(os.path.join(home, f"{prefix}_{var_names[j]}_{year_list[k]}.nc"), "r") as src_dataset:
                    # get data
                    forcing_data_k = src_dataset.variables[var_names[j]][
                        :,
                        yindex_start : yindex_end + 1,
                        xindex_start : xindex_end + 1,
                    ]
                    
                    # get searched data
                    searched_grid_data_k = [forcing_data_k[:, searched_grid_index[0][l], searched_grid_index[1][l]]
                                          for l in range(len(searched_grid_index[0]))
                    ]
                    
                    searched_grid_data_k = np.array(searched_grid_data_k)
                    
                    searched_grid_data_k = searched_grid_data_k.T  # time * searched_grids
                    
                    if k == 0:
                        searched_grid_data_v = searched_grid_data_k
                    else:
                        searched_grid_data_v = np.concatenate((searched_grid_data_v, searched_grid_data_k), axis=0)
                    
            # resample
            searched_resample_data_series_v = np.nanmean(
                searched_grid_data_v,
                axis=1,
            )
            
            # append
            forcings_searched_resample_Series_v.append(searched_resample_data_series_v)
            
        # check
        if check_search and i == 0:
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
    # meantmp, maxtmp, mintmp: K -> C
    grid_shp["meantmp"] = grid_shp["meantmp"].apply(lambda row: np.array(row) - 273.15)  # K to C
    # grid_shp["maxtmp"] = grid_shp["maxtmp"].apply(lambda row: np.array(row) - 273.15)  # K to C
    # grid_shp["mintmp"] = grid_shp["mintmp"].apply(lambda row: np.array(row) - 273.15)  # K to C
    
    # prs: hPa -> kPa
    grid_shp["prs"] = grid_shp["prs"].apply(lambda row: np.array(row) / 10.0)  # hPa to kPa
    
    # calculate SWDOWN, W m-2
    def compute_swd_series(row):
        lat = row["point_geometry"].y
        ssd_series = row["ssd"]
        swdown_series = [
            cal_SWDOWN_Angstrom_Prescott_eq(ssd_day, lat, day, a=0.25, b=0.50)
            for ssd_day, day in zip(ssd_series, date)
        ]
        return swdown_series
    
    grid_shp['SWDOWN'] = grid_shp.apply(compute_swd_series, axis=1)
    
    # calculate VP, kPa
    def compute_vp_series(row):
        T_C = row["meantmp"]
        RH_100 = row["rhu"]
        es_kPa = [cal_es_Tetens_eq(T_day) for T_day in T_C]
        
        vp_series = [
            cal_VP_from_RH_es(RH_day, es_day) for RH_day, es_day in zip(RH_100, es_kPa)
        ]
        return vp_series

    grid_shp["VP"] = grid_shp.apply(compute_vp_series, axis=1)
    
    # calculate cloud fraction from ssd
    def compute_cloud_fraction(row):
        lat = row["point_geometry"].y
        ssd_h = row["ssd"]
        
        cloud_fraction_series = [
            cal_cloud_fraction_from_ssd(ssd_day, date_day, lat)
            for ssd_day, date_day in zip(ssd_h, date)
        ]
        return cloud_fraction_series
    
    grid_shp["cloud_fraction"] = grid_shp.apply(compute_cloud_fraction, axis=1)
    
    # calculate LWDOWN, W m-2
    def compute_lwd_series(row):
        T_C = row["meantmp"]
        T_K = T_C + 273.15
        cloud_cover = row["cloud_fraction"]
        lwd_series = [cal_LWDOWN_CD99_eq(T_day, cloud_cover_day) for T_day, cloud_cover_day in zip(T_K, cloud_cover)]
        return lwd_series

    grid_shp["LWDOWN"] = grid_shp.apply(compute_lwd_series, axis=1)
    
    # rename columns to add units
    grid_shp.rename(
        columns={
            "meantmp": "tmp_avg_C",
            # "maxtmp": "tmp_max_C",
            # "mintmp": "tmp_min_C",
            "pre": "pre_mm_per_day",
            "prs": "prs_kPa",
            "rhu": "rhu_percentage",
            "ssd": "ssd_h",
            "SWDOWN": "swd_W_per_m2",
            "VP": "vp_kPa",
            "LWDOWN": "lwd_W_per_m2",
            "win": "wind_m_per_s", # win
        },
        inplace=True
    )
    
    # plot
    if plot:
        # plot timeseries
        grid_i = 30
        plot_var_name = "LWDOWN"
        
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

