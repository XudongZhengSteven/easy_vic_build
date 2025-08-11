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
    with Dataset(os.path.join(home, str(year_list[0]), f"SMroot_{year_list[0]}_GLEAM_v3.8a.nc"), "r") as src_dataset:
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
    SMroot_all = []
    SMsurf_all = []
    
    for yi, year in tqdm(
      enumerate(year_list),
      colour="green",
      desc="loop for each year to extract GLEAM SM data",  
    ):  
        # read
        with Dataset(os.path.join(home, str(year), f"SMroot_{year}_GLEAM_v3.8a.nc"), "r") as src_dataset_SMroot:
            SMroot_data = src_dataset_SMroot["SMroot"][:]
        with Dataset(os.path.join(home, str(year), f"SMsurf_{year}_GLEAM_v3.8a.nc"), "r") as src_dataset_SMsurf:
            SMsurf_data = src_dataset_SMsurf["SMsurf"][:]
        
        # loop for grids
        SMroot_yi = []
        SMsurf_yi = []
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
            SMroot_gi =  np.stack([SMroot_data[:, i, j] for i, j in ij_list], axis=0)  # search grids x date
            SMsurf_gi = np.stack([SMsurf_data[:, i, j] for i, j in ij_list], axis=0)
            
            # resample
            searched_resample_data_SMroot_gi = np.nanmean(
                SMroot_gi, axis=0
            ).reshape(-1, 1)  # dates
        
            searched_resample_data_SMsurf_gi = np.nanmean(
                SMsurf_gi, axis=0
            ).reshape(-1, 1)
            
            # append grids
            SMroot_yi.append(searched_resample_data_SMroot_gi)
            SMsurf_yi.append(searched_resample_data_SMsurf_gi)
            
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
        searched_resample_data_SMroot = np.concatenate(SMroot_yi, axis=1)
        searched_resample_data_SMsurf = np.concatenate(SMsurf_yi, axis=1)
        
        # append date, check len(date) == SMsurf_all.shape[1]
        SMroot_all.append(searched_resample_data_SMroot)
        SMsurf_all.append(searched_resample_data_SMsurf)
        
    # concatenate years
    SMroot_all = np.concatenate(SMroot_all, axis=0)
    SMsurf_all = np.concatenate(SMsurf_all, axis=0)
    
    # unit change, m3/m3 -> mm, surf: 0-10cm, root: 10-100cm
    SMsurf_all = SMsurf_all * 100  # mm
    SMroot_all = SMroot_all * 900  # mm
    
    # combine
    SMtotal_all = SMsurf_all + SMroot_all
    
    # save
    grid_shp["SMsurf(mm)"] = list(SMsurf_all.T)
    grid_shp["SMroot(mm)"] = list(SMroot_all.T)
    grid_shp["SMtotal(mm)"] = list(SMtotal_all.T)
    
    # plot
    if plot:
        # plot timeseries
        grid_i = 0
        plot_var_name = "SMsurf(mm)"
        
        plt.figure(figsize=(10, 6))
        plt.plot(date, grid_shp.loc[grid_shp.index[grid_i], "SMsurf(mm)"], "b--", label="SMsurf(mm)")
        plt.plot(date, grid_shp.loc[grid_shp.index[grid_i], "SMroot(mm)"], "r--", label="SMroot(mm)")
        plt.plot(date, grid_shp.loc[grid_shp.index[grid_i], "SMtotal(mm)"], "k-", label="SMtotal(mm)")
        
        plt.xlabel("Time")
        plt.ylabel("SM (mm)")
        plt.legend()
        plt.title(f"Time Series of {plot_var_name} at Grid {grid_i}")
        
        # plot map
        for plot_var_name in ["SMsurf(mm)", "SMroot(mm)", "SMtotal(mm)"]:
            fig, ax = plt.subplots()
            
            grid_shp_plot = deepcopy(grid_shp)
            grid_shp_plot[f"{plot_var_name}_timemean"] = grid_shp_plot.apply(
                lambda row: np.nanmean(row[plot_var_name]), axis=1
            )
            
            grid_shp_plot.plot(
                f"{plot_var_name}_timemean",
                ax=ax,
                edgecolor="k",
                linewidth=0.2,
            )
            ax.set_title(f"{plot_var_name}_timemean")
            ax.set_xlim(
                [min(grids_lon) - grid_shp_res / 2, max(grids_lon) + grid_shp_res / 2]
            )
            ax.set_ylim(
                [min(grids_lat) - grid_shp_res / 2, max(grids_lat) + grid_shp_res / 2]
            )
        
        plt.show(block=True)
    
    return grid_shp
    
    
    
    