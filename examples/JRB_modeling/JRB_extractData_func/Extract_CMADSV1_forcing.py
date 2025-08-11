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
import geopandas as gpd
import re

from easy_vic_build.tools.geo_func import search_grids
from easy_vic_build.tools.geo_func.create_gdf import CreateGDF

from easy_vic_build.tools.mete_func.mete_func import cal_es_Tetens_eq, cal_SWDOWN_Angstrom_Prescott_eq, cal_cloud_fraction_from_swdown
from easy_vic_build.tools.mete_func.mete_func import cal_VP_from_prs_sh, cal_LWDOWN_Brutsaert_eq, cal_LWDOWN_CD99_eq, cal_clearsky_SWDOWN_Dudhia89_eq

from easy_vic_build import logger


def format_name_TMP_AVG():
    home = os.path.join("E:\\data\\hydrometeorology\\CMADS\\CMADS-L V1.0", "CMADS-L-For-other-model", "Average-Temperature-txt")
    
    # standard pattern
    standard_pattern = re.compile(r'^CMADS_V1\.0_TMP_AVG_\d+-\d+\.txt$')

    # abnormal_pattern
    abnormal_pattern = re.compile(r'^CMADS_V1\.0_TMP_AVG_CMADS_V1\.0_TMP_AVG_(\d+-\d+)\.txt$')
    
    for filename in tqdm(os.listdir(home), desc="loop for formating name of TMP AVG"):
        if filename.endswith('.txt'):
            # check abnormal
            abnormal_match = abnormal_pattern.match(filename)
            if abnormal_match:
                # extract numbers
                numbers = abnormal_match.group(1)
                
                # construct new filename
                new_filename = f"CMADS_V1.0_TMP_AVG_{numbers}.txt"
                
                # get full path
                old_path = os.path.join(home, filename)
                new_path = os.path.join(home, new_filename)
                
                # check if the new filename already exists
                if os.path.exists(new_path):
                    logger.warning(f"target file {new_filename} exist, skip to rename {filename}")
                else:
                    # rename
                    os.rename(old_path, new_path)
                    print(f"renamed: {filename} -> {new_filename}")
        

def ExtractData(
    grid_shp, grid_shp_res=0.125,
    date_period=["20080101", "20181231"],
    search_method="radius_rectangle", 
    plot=False, check_search=False,
    reverse_lat=True,
    time_UTC=12,
    elevation=0,
):
    # general, 1/3 deg, daily
    home = "E:\\data\\hydrometeorology\\CMADS\\CMADS-L V1.0"
    coord_dir = "CMADS-L V1.0-(station)"
    data_dir = "CMADS-L-For-other-model"
    var_dir_names = [
        "Atmospheric-Pressure-txt",
        "Average-Temperature-txt",
        # "Maximum-Temperature-txt",
        # "Minimum-Temperature-txt",
        "Precipitation-txt",
        "Relative-Humidity-txt",
        "Solar-Radiation-txt",
        "Specific-Humidity-txt",
        "Wind-txt",
    ]  # need to derive: SWDOWN (W m-2), LWDOWN (W m-2), VP (kPa)
    
    prefix = "CMADS_V1.0"
    var_file_names = [
        "PRS",  #* hPa -> /10.0 -> kPa
        "TMP_AVG",  #* C
        # "TMP_MAX",  # C
        # "TMP_MIN",  # C
        "24h_PRE",  #* mm/day
        "RHU",  # fraction
        "SOR",  # MJ/m2
        "SHU",  # g/kg
        "WIND"  # *m/s
    ]  # need to derive: SWDOWN (W m-2), LWDOWN (W m-2), VP (kPa)
    
    date = pd.date_range(date_period[0], date_period[1], freq="D")
    
    # set grids_lat, lon
    grids_lat = grid_shp.point_geometry.y.to_list()
    grids_lon = grid_shp.point_geometry.x.to_list()
    
    # read data to get lon, lat
    coord_path = os.path.join(home, coord_dir, "CMADS1.0.shp")
    coord_gdf = gpd.read_file(coord_path)
    
    forcing_lat = coord_gdf.latitude.to_list()
    forcing_lon = coord_gdf.longitude.to_list()
    
    forcing_lat = list(set(forcing_lat))
    forcing_lon = list(set(forcing_lon))
    
    forcing_lat.sort(reverse=reverse_lat)  # True: large -> small
    forcing_lon.sort(reverse=False)
    
    forcing_lat = np.array(forcing_lat)
    forcing_lon = np.array(forcing_lon)
    
    forcing_lat_res = (forcing_lat.max() - forcing_lat.min()) / (len(forcing_lat) - 1)  # 1/3 deg
    forcing_lon_res = (forcing_lon.max() - forcing_lon.min()) / (len(forcing_lon) - 1)
    
    # search grids
    logger.info("searching grids for CDMet forcing data... ...")
    
    # source data res: 1/3 deg
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
    forcings_searched_resample_Series = [[] for _ in range(len(var_file_names))]
    
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
        
        # loop for get searched data
        for j in range(len(var_dir_names)):
            forcings_searched_resample_Series_v = forcings_searched_resample_Series[j]
            
            for l in range(len(searched_grid_lat)):
                # map station_id
                searched_grid_lat_l = searched_grid_lat[l]
                searched_grid_lon_l = searched_grid_lon[l]
                
                station_id_l = coord_gdf.loc[
                    (coord_gdf.latitude == searched_grid_lat_l) &
                    (coord_gdf.longitude == searched_grid_lon_l), "StationID"
                ].values[0]
                
                # data path
                data_path_l = os.path.join(home, data_dir, var_dir_names[j], f"{prefix}_{var_file_names[j]}_{station_id_l}.txt")
            
                # read data
                data_df_l = pd.read_csv(data_path_l, sep="\t", header=0)
                data_df_l.index = pd.date_range(data_df_l.columns[0], freq="D", periods=len(data_df_l))
                
                # select data period
                data_df_l_period = data_df_l.loc[date_period[0]:date_period[1], :].values
                
                if l == 0:
                    searched_grid_data_v = data_df_l_period
                else:
                    searched_grid_data_v = np.hstack((searched_grid_data_v, data_df_l_period))
                
            # resample #TODO resample method
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
    for j in range(len(var_file_names)):
        # [v1, ..., v5], v1 = [grid1, ..., gridn], grid1 = [time1, ..., timek] (series)
        grid_shp[f"{var_file_names[j]}"] = forcings_searched_resample_Series[j]
    
    # postprocessing: unit change
    # prs: hPa -> kPa
    grid_shp["PRS"] = grid_shp["PRS"].apply(lambda row: np.array(row) / 10.0)  # hPa to kPa
    
    # calculate SWDOWN, W m-2
    grid_shp["SWDOWN"] = grid_shp["SOR"].apply(lambda row: np.array(row) * 1e6 / (24 * 3600))  # MJ/m2 to W m-2
    
    # rhu: fraction to percentage
    grid_shp["RHU"] = grid_shp["RHU"].apply(lambda row: np.array(row) * 100.0)  # fraction to percentage
    
    # shu: g/kg to kg/kg
    grid_shp["SHU"] = grid_shp["SHU"].apply(lambda row: np.array(row) / 1000.0)  # g/kg to kg/kg
    
    # calculate VP, kPa
    def compute_vp_series(row):
        prs_kPa = row["PRS"]
        sh_kg_per_kg = row["SHU"]
        
        vp_series = [
            cal_VP_from_prs_sh(prs_kPa_day, sh_kg_per_kg_day) for prs_kPa_day, sh_kg_per_kg_day in zip(prs_kPa, sh_kg_per_kg)
        ]
        return vp_series

    grid_shp["VP"] = grid_shp.apply(compute_vp_series, axis=1)
    
    # calculate SWDOWN_clearsky, W m-2
    def compute_swd_clearsky_series(row):
        lat = row["point_geometry"].y
        
        swdown_clearsky_series = [cal_clearsky_SWDOWN_Dudhia89_eq(day, lat, elevation, time_UTC, ESRA=False) for day in date]
        return swdown_clearsky_series
    
    grid_shp['SWDOWN_clearsky'] = grid_shp.apply(compute_swd_clearsky_series, axis=1)
    
    # calculate cloud fraction from SWDOWN_clearsky and SWDOWN_measure (SOR)
    def compute_cloud_fraction_series(row):
        SWDOWN_clearsky = row["SWDOWN_clearsky"]
        SWDOWN_measure = row["SWDOWN"]
        
        cloud_fraction_series = [
            cal_cloud_fraction_from_swdown(sw_measure_day, sw_clearsky_day)
            for sw_measure_day, sw_clearsky_day in zip(SWDOWN_measure, SWDOWN_clearsky)
        ]
        return cloud_fraction_series
    
    grid_shp['cloud_fraction'] = grid_shp.apply(compute_cloud_fraction_series, axis=1)
    
    # calculate LWDOWN, W m-2
    def compute_lwd_series(row):
        T_C = row["TMP_AVG"]
        T_K = T_C + 273.15
        cloud_cover = row["cloud_fraction"]
        lwd_series = [cal_LWDOWN_CD99_eq(T_day, cloud_cover_day) for T_day, cloud_cover_day in zip(T_K, cloud_cover)]
        return lwd_series

    grid_shp["LWDOWN"] = grid_shp.apply(compute_lwd_series, axis=1)
    
    # rename columns to add units
    grid_shp.rename(
        columns={
            "TMP_AVG": "tmp_avg_C",  # C
            # "TMP_MAX": "tmp_max_C",  # C
            # "TMP_MIN": "tmp_min_C",  # C
            "24h_PRE": "pre_mm_per_day",  # mm/day
            "PRS": "prs_kPa",  # kPa
            "RHU": "rhu_percentage",  # percentage
            "SOR": "sor_MJ_per_m2",  # MJ/m2
            "SWDOWN": "swd_W_per_m2",  # W m-2
            "SHU": "shu_kg_per_kg",  # kg/kg
            "VP": "vp_kPa",  # kPa
            "LWDOWN": "lwd_W_per_m2",  # W m-2
            "WIND": "wind_m_per_s",  # m/s
        },
        inplace=True
    )
    
    # plot
    if plot:
        # plot timeseries
        grid_i = 0
        plot_var_name = "SWDOWN"
        
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
    # format_name_TMP_AVG()
    pass
    
    