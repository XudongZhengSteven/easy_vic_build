# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import matplotlib.pyplot as plt
import numpy as np
from tqdm import *
from netCDF4 import Dataset
import pandas as pd
from copy import deepcopy
import os

from easy_vic_build.tools.geo_func import search_grids
from easy_vic_build.tools.geo_func.create_gdf import CreateGDF

from easy_vic_build import logger


def ExtractData(
    grid_shp, grid_shp_res=0.125,
    date_period=["20080101", "20181231"],
    search_method="radius_rectangle", 
    plot=False, check_search=False
):
    # general, 4km, daily
    home = "H:\\data\\hydrometeorology\\GLEAM\\data\\v3.8a\\daily"
    # f"SMroot_{year}_GLEAM_v3.8a.nc"
    # f"SMsurf_{year}_GLEAM_v3.8a.nc"
    
    year_list = np.arange(int(date_period[0][:4]), int(date_period[1][:4])+ 1, 1, dtype="int")
    date = pd.date_range(date_period[0], date_period[1], freq="D")
    
    # set grids_lat, lon
    grids_lat = grid_shp.point_geometry.y.to_list()
    grids_lon = grid_shp.point_geometry.x.to_list()
    
    # read data to get lon, lat
    with Dataset(os.path.join(home, str(year_list[0]), f"E_{year_list[0]}_GLEAM_v3.8a.nc"), "r") as src_dataset:
        # get lat, lon
        GLEAM_lat = src_dataset.variables["lat"][:]
        GLEAM_lon = src_dataset.variables["lon"][:]
        
        # get res
        GLEAM_lat_res = (max(GLEAM_lat) - min(GLEAM_lat)) / (len(GLEAM_lat) - 1)  # 0.25 deg
        GLEAM_lon_res = (max(GLEAM_lon) - min(GLEAM_lon)) / (len(GLEAM_lon) - 1)
    
    # search grids  
    logger.info("searching grids for GLEAM data... ...")
    
    # source data res: 0.25 deg
    if search_method == "radius_rectangle":
        searched_grids_index = search_grids.search_grids_radius_rectangle(
            dst_lat=grids_lat,
            dst_lon=grids_lon,
            src_lat=GLEAM_lat,
            src_lon=GLEAM_lon,
            lat_radius=grid_shp_res / 2,
            lon_radius=grid_shp_res / 2,
        )
        
    elif search_method == "radius_rectangle_reverse":
        searched_grids_index = search_grids.search_grids_radius_rectangle_reverse(
            dst_lat=grids_lat,
            dst_lon=grids_lon,
            src_lat=GLEAM_lat,
            src_lon=GLEAM_lon,
            lat_radius=GLEAM_lat_res / 2,
            lon_radius=GLEAM_lon_res / 2,
        )
    
    elif search_method == "nearest":
        searched_grids_index = search_grids.search_grids_nearest(dst_lat=grids_lat, dst_lon=grids_lon,
                                                                src_lat=GLEAM_lat, src_lon=GLEAM_lon,
                                                                search_num=1,
                                                                move_src_lat=None, move_src_lon=None)
    else:
        logger.warning(f"search method {search_method} not supported")
    
    # loop for yeas to read
    E_all = []
    
    for yi, year in tqdm(
      enumerate(year_list),
      colour="green",
      desc="loop for each year to extract GLEAM SM data",  
    ):  
        # read
        with Dataset(os.path.join(home, str(year), f"E_{year}_GLEAM_v3.8a.nc"), "r") as src_dataset_SMroot:
            E_data = src_dataset_SMroot["E"][:]
        
        # loop for grids
        E_yi = []
        for gi, gindex in tqdm(enumerate(grid_shp.index), colour="grey", desc="loop for grids"):
            # get search grid index and data for this dst_grid
            searched_grid_index = searched_grids_index[gi]
            # dst_lat_grid = grid_shp.loc[gindex, :].point_geometry.y
            # dst_lon_grid = grid_shp.loc[gindex, :].point_geometry.x
            
            # searched_grid_lat = [
            #     GLEAM_lat[searched_grid_index[0][j]]
            #     for j in range(len(searched_grid_index[0]))
            # ]
            # searched_grid_lon = [
            #     GLEAM_lon[searched_grid_index[1][j]]
            #     for j in range(len(searched_grid_index[0]))
            # ]

            # get searched data
            ij_list = list(zip(*searched_grid_index))
            E_gi =  np.stack([E_data[:, i, j] for i, j in ij_list], axis=0)  # search grids x date
            
            # resample
            searched_resample_data_E_gi = np.nanmean(
                E_gi, axis=0
            ).reshape(-1, 1)  # dates
            
            # append grids
            E_yi.append(searched_resample_data_E_gi)
            
            # check
            # if check_search and gi + yi == 0:
            #     cgdf = CreateGDF()
            #     grid_shp_grid = grid_shp.loc[[gi], "geometry"]
            #     searched_grids_gdf = cgdf.createGDF_rectangle_central_coord(
            #         searched_grid_lon, searched_grid_lat, GLEAM_lat_res
            #     )

            #     fig, ax = plt.subplots()
            #     grid_shp_grid.boundary.plot(ax=ax, edgecolor="r", linewidth=2)  # target
            #     searched_grids_gdf.plot(
            #         ax=ax, edgecolor="k", linewidth=0.2, facecolor="b", alpha=0.5
            #     )  # searched data from source data
                
            #     ax.set_title("check search for GLEAM SM")
                
            #     plt.show(block=True)
            
        # concatenate grids
        searched_resample_data_E = np.concatenate(E_yi, axis=1)
        
        # append date, check len(date) == SMsurf_all.shape[1]
        E_all.append(searched_resample_data_E)
        
    # concatenate years, mm
    E_all = np.concatenate(E_all, axis=0)
    
    # save
    grid_shp["E(mm)"] = list(E_all.T)
    
    # plot
    if plot:
        # plot timeseries
        grid_i = 0
        plot_var_name = "E(mm)"
        
        plt.figure(figsize=(10, 6))
        plt.plot(date, grid_shp.loc[grid_shp.index[grid_i], "E(mm)"], "g-", label="E(mm)")
        
        plt.xlabel("Time")
        plt.ylabel("E (mm)")
        plt.legend()
        plt.title(f"Time Series of {plot_var_name} at Grid {grid_i}")
        
        # plot map
        fig, ax = plt.subplots()
        
        grid_shp_plot = deepcopy(grid_shp)
        grid_shp_plot[f"E_timemean"] = grid_shp_plot.apply(
            lambda row: np.nanmean(row[plot_var_name]), axis=1
        )
        
        grid_shp_plot.plot(
            f"E_timemean",
            ax=ax,
            edgecolor="k",
            linewidth=0.2,
        )
        ax.set_title(f"E_timemean")
        ax.set_xlim(
            [min(grids_lon) - grid_shp_res / 2, max(grids_lon) + grid_shp_res / 2]
        )
        ax.set_ylim(
            [min(grids_lat) - grid_shp_res / 2, max(grids_lat) + grid_shp_res / 2]
        )
    
        plt.show(block=True)
    
    return grid_shp
    
    
    
    