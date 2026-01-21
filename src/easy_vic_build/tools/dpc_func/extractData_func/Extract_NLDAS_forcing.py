# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
from netCDF4 import Dataset
from ...geo_func import search_grids, resample
from ...geo_func.create_gdf import CreateGDF
from .... import logger
from ...mete_func.mete_func import cal_VP_from_prs_sh
import pandas as pd
from datetime import datetime
from tqdm import *
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from multiprocessing import Pool

def generate_nldas_filename(dt):
    return f"NLDAS_FORA0125_H.A{dt.strftime('%Y%m%d')}.{dt.strftime('%H%M')}.002.grb.SUB.nc4"
    
    
def process_date_chunk(args):
    var_name, date_chunk, grid_shp, searched_grids_index, NLDAS_forcing_home, date = args
    chunk_results = np.empty((len(grid_shp.index), len(date_chunk)), dtype=np.float32)
    
    for i, di in enumerate(date_chunk):
        try:
            chunk_results[:, i] = ExtractData_di(
                di, date, NLDAS_forcing_home, var_name, 
                grid_shp, searched_grids_index
            )
        except Exception as e:
            logger.error(f"Error at di={di}: {str(e)}")
            chunk_results[:, i] = -9999.0
    
    return date_chunk[0], chunk_results
    

def ExtractData_di(
    di,
    date,
    NLDAS_forcing_home,
    var_name,
    grid_shp,
    searched_grids_index
):
    dt = date[di]
    searched_resample_data_list = []  # len=len(grid_shp.index)
    
    fn = generate_nldas_filename(dt)
    fp = os.path.join(NLDAS_forcing_home, fn)
    
    with Dataset(fp, "r") as src_dataset:
        src_var = src_dataset.variables[var_name]

        for gi in grid_shp.index:
            # get search grid index, lat, lon for this dst_grid
            searched_grid_index = searched_grids_index[gi]
            
            # get searched data
            if len(src_var.dimensions) == 3:
                searched_grid_data = [src_var[0, searched_grid_index[0][l], searched_grid_index[1][l]] for l in range(len(searched_grid_index[0]))]
                
            elif len(src_var.dimensions) == 4:
                searched_grid_data = [src_var[0, 0, searched_grid_index[0][l], searched_grid_index[1][l]] for l in range(len(searched_grid_index[0]))]
                
            else:
                logger.warning(f"Variable {var_name} has unsupported dimensions: {len(src_var.dimensions)}")

            # resample # TODO
            searched_resample_data = np.nanmean(np.array(searched_grid_data))
            
            # append
            searched_resample_data_list.append(searched_resample_data)
            
            # # check
            # if check_search and vi+gi+di == 0:
            #     cgdf = CreateGDF()
            #     grid_shp_grid = grid_shp.loc[[gi], "geometry"]
            #     searched_grids_gdf = cgdf.createGDF_rectangle_central_coord(
            #         searched_grid_lon, searched_grid_lat, NLDAS_forcing_lat_res
            #     )

            #     fig, ax = plt.subplots()
            #     grid_shp_grid.boundary.plot(ax=ax, edgecolor="r", linewidth=2)  # target
            #     searched_grids_gdf.plot(
            #         ax=ax, edgecolor="k", linewidth=0.2, facecolor="b", alpha=0.5
            #     )  # searched data from source data
                
            #     ax.set_title("check search")
                
            #     plt.show(block=True)

    return searched_resample_data_list

def ExtractData(
    grid_shp,
    date_period,
    search_method="nearest",
    grid_shp_res=0.25, 
    plot=True, 
    check_search=False,
    N_PROCESS=8,
    CHUNK_SIZE=60,
):
    # general
    NLDAS_forcing_home = "E:\\data\\hydrometeorology\\NLDAS\\NLDAS2_Primary_Forcing_Data_subset_0.125\\data"
    
    date = pd.date_range(date_period[0], date_period[1], freq="H")
    
    # read NLDAS, lat, lon, res (0.125deg)
    with Dataset(os.path.join(NLDAS_forcing_home, generate_nldas_filename(date[0])), "r") as dataset:
        NLDAS_forcing_lat = dataset.variables["lat"][:]
        NLDAS_forcing_lon = dataset.variables["lon"][:]
        NLDAS_forcing_lat_res = (max(NLDAS_forcing_lat) - min(NLDAS_forcing_lat)) / (len(NLDAS_forcing_lat) - 1)
        NLDAS_forcing_lon_res = (max(NLDAS_forcing_lon) - min(NLDAS_forcing_lon)) / (len(NLDAS_forcing_lon) - 1)
        NLDAS_forcing_lat_res = float(f"{NLDAS_forcing_lat_res:.3g}")
        NLDAS_forcing_lon_res = float(f"{NLDAS_forcing_lon_res:.3g}")
    
    # set grids_lat, lon
    grids_lat = grid_shp.point_geometry.y.to_list()
    grids_lon = grid_shp.point_geometry.x.to_list()
    
    # set var name
    var_names = [
        "TMP",  #* K -> -273.15 -> C
        "APCP",  #* kg/m^2 == mm/step
        "PRES",  #* Pa -> /1000 -> kPa
        "DSWRF",  #* W/m^2
        "DLWRF",  #* W/m^2
        "SPFH",  # kg/kg
        "UGRD",  #* m/s
        "VGRD",  #* m/s
    ]    # need to derive: VP (kPa), Wind
    
    # search grids
    logger.info("searching grids for NLDAS forcing data... ...")
    
    # source data res: 0.125 deg ~= 13.875km
    if search_method == "radius":
        searched_grids_index = search_grids.search_grids_radius(
            dst_lat=grids_lat,
            dst_lon=grids_lon,
            src_lat=NLDAS_forcing_lat,
            src_lon=NLDAS_forcing_lon,
            lat_radius=grid_shp_res / 2,
            lon_radius=grid_shp_res / 2,
        )
        
    if search_method == "radius_rectangle":
        searched_grids_index = search_grids.search_grids_radius_rectangle(
            dst_lat=grids_lat,
            dst_lon=grids_lon,
            src_lat=NLDAS_forcing_lat,
            src_lon=NLDAS_forcing_lon,
            lat_radius=grid_shp_res / 2,
            lon_radius=grid_shp_res / 2,
        )
    
    elif search_method == "radius_rectangle_reverse":
        searched_grids_index = search_grids.search_grids_radius_rectangle_reverse(
            dst_lat=grids_lat,
            dst_lon=grids_lon,
            src_lat=NLDAS_forcing_lat,
            src_lon=NLDAS_forcing_lon,
            lat_radius=NLDAS_forcing_lat_res / 2,
            lon_radius=NLDAS_forcing_lon_res / 2,
        )
    
    elif search_method == "nearest":
        searched_grids_index = search_grids.search_grids_nearest(dst_lat=grids_lat, dst_lon=grids_lon,
                                                                 src_lat=NLDAS_forcing_lat, src_lon=NLDAS_forcing_lon,
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
            range(i, min(i+CHUNK_SIZE, len(date))) 
            for i in range(0, len(date), CHUNK_SIZE)
        ]
        
        with Pool(processes=N_PROCESS) as pool:
            tasks = [
                pool.apply_async(
                    process_date_chunk,
                    ((var_name, chunk, grid_shp, searched_grids_index, NLDAS_forcing_home, date),)
                )
                for chunk in chunks
            ]
            
            with tqdm(total=len(chunks), desc=f"{var_name} Progress") as pbar:
                for task in tasks:
                    start_di, chunk_res = task.get()
                    end_di = start_di + chunk_res.shape[1]
                    forcings_searched_resample_arrays[vi, :, start_di:end_di] = chunk_res
                    pbar.update(1)
    
    # save
    for j in range(len(var_names)):
        forcings_searched_resample_arrays_v = forcings_searched_resample_arrays[j]
        # [v1, ..., v5], v1 = [grid1, ..., gridn], grid1 = [time1, ..., timek] (series)
        grid_shp[f"{var_names[j]}"] = [row.tolist() for row in forcings_searched_resample_arrays_v]
    
    # postprocessing: unit change
    # TMP: K -> -273.15 -> C
    grid_shp["TMP"] = grid_shp["TMP"].apply(lambda row: np.array(row) - 273.15)  # K to C
    
    # PRES: Pa -> /1000 -> kPa
    grid_shp["PRES"] = grid_shp["PRES"].apply(lambda row: np.array(row) / 1000.0)  # Pa to kPa
    
    # calculate VP, kPa
    def compute_vp_series(row):
        prs_kPa = row["PRES"]
        sh_kg_per_kg = row["SPFH"]
        
        vp_series = [
            cal_VP_from_prs_sh(prs_kPa_day, sh_kg_per_kg_day) for prs_kPa_day, sh_kg_per_kg_day in zip(prs_kPa, sh_kg_per_kg)
        ]
        return vp_series

    grid_shp["VP"] = grid_shp.apply(compute_vp_series, axis=1)
    
    # cal wind
    def compute_wind_series(row):
        wind_u = row["UGRD"]
        wind_v = row["VGRD"]
        
        vp_series = [
            (wind_u_day**2 + wind_v_day**2)**0.5  for wind_u_day, wind_v_day in zip(wind_u, wind_v)
        ]
        return vp_series

    grid_shp["WIND"] = grid_shp.apply(compute_wind_series, axis=1)
    
    # rename columns to add units
    grid_shp.rename(
        columns={
            "TMP": "tmp_avg_C",  # C
            "APCP": "pre_mm_per_step",  # mm/step (1h)
            "PRES": "prs_kPa",  # kPa
            "DSWRF": "swd_W_per_m2",  # W m-2
            "DLWRF": "lwd_W_per_m2",  # W m-2
            "SPFH": "shu_kg_per_kg",  # kg/kg
            "VP": "vp_kPa",  # kPa
            "WIND": "wind_m_per_s",  # m/s
            # "UGRD"
            # "VGRD"
        },
        inplace=True
    )
    
    # plot
    if plot:
        # plot timeseries
        grid_i = 0
        plot_var_name = "lwd_W_per_m2"
        
        plt.figure(figsize=(10, 6))
        plt.plot(date, grid_shp.loc[grid_shp.index[grid_i], f"{plot_var_name}"], label=plot_var_name)
        plt.xlabel("Time")
        plt.ylabel(plot_var_name)
        plt.legend()
        plt.title(f"Time Series of {plot_var_name} at Grid {grid_i}")
        # plt.show(block=True)
        
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

# def ExtractData(
#     grid_shp,
#     date_period,
#     search_method="nearest",
#     grid_shp_res=0.25, 
#     plot=True, 
#     check_search=False
# ):
#     # general
#     NLDAS_forcing_home = "E:\\data\\hydrometeorology\\NLDAS\\NLDAS2_Primary_Forcing_Data_subset_0.125\\data"
    
#     def generate_nldas_filename(dt):
#         return f"NLDAS_FORA0125_H.A{dt.strftime('%Y%m%d')}.{dt.strftime('%H%M')}.002.grb.SUB.nc4"
    
#     date = pd.date_range(date_period[0], date_period[1], freq="H")
    
#     # read NLDAS, lat, lon, res (0.125deg)
#     with Dataset(os.path.join(NLDAS_forcing_home, generate_nldas_filename(date[0])), "r") as dataset:
#         NLDAS_forcing_lat = dataset.variables["lat"][:]
#         NLDAS_forcing_lon = dataset.variables["lon"][:]
#         NLDAS_forcing_lat_res = (max(NLDAS_forcing_lat) - min(NLDAS_forcing_lat)) / (len(NLDAS_forcing_lat) - 1)
#         NLDAS_forcing_lon_res = (max(NLDAS_forcing_lon) - min(NLDAS_forcing_lon)) / (len(NLDAS_forcing_lon) - 1)
#         NLDAS_forcing_lat_res = float(f"{NLDAS_forcing_lat_res:.3g}")
#         NLDAS_forcing_lon_res = float(f"{NLDAS_forcing_lon_res:.3g}")
    
#     # set grids_lat, lon
#     grids_lat = grid_shp.point_geometry.y.to_list()
#     grids_lon = grid_shp.point_geometry.x.to_list()
    
#     # set var name
#     var_names = [
#         "TMP",  #* K -> -273.15 -> C
#         "APCP",  #* kg/m^2 == mm/step
#         "PRES",  #* Pa -> /1000 -> kPa
#         "DSWRF",  #* W/m^2
#         "DLWRF",  #* W/m^2
#         "SPFH",  # kg/kg
#         "UGRD",  #* m/s
#         "VGRD",  #* m/s
#     ]    # need to derive: VP (kPa), Wind
    
#     # search grids
#     logger.info("searching grids for NLDAS forcing data... ...")
    
#     # source data res: 0.125 deg
#     if search_method == "radius_rectangle":
#         searched_grids_index = search_grids.search_grids_radius_rectangle(
#             dst_lat=grids_lat,
#             dst_lon=grids_lon,
#             src_lat=NLDAS_forcing_lat,
#             src_lon=NLDAS_forcing_lon,
#             lat_radius=grid_shp_res / 2,
#             lon_radius=grid_shp_res / 2,
#         )
    
#     elif search_method == "radius_rectangle_reverse":
#         searched_grids_index = search_grids.search_grids_radius_rectangle_reverse(
#             dst_lat=grids_lat,
#             dst_lon=grids_lon,
#             src_lat=NLDAS_forcing_lat,
#             src_lon=NLDAS_forcing_lon,
#             lat_radius=NLDAS_forcing_lat_res / 2,
#             lon_radius=NLDAS_forcing_lon_res / 2,
#         )
    
#     elif search_method == "nearest":
#         searched_grids_index = search_grids.search_grids_nearest(dst_lat=grids_lat, dst_lon=grids_lon,
#                                                                  src_lat=NLDAS_forcing_lat, src_lon=NLDAS_forcing_lon,
#                                                                  search_num=1,
#                                                                  move_src_lat=None, move_src_lon=None)
#     else:
#         logger.warning(f"search method {search_method} not supported")
    
#     # loop to read NLDAS forcing data for each grid
#     forcings_searched_resample_arrays = [np.full((len(grid_shp.index), len(date)), fill_value=-9999.0, dtype=float) for _ in range(len(var_names))] # grids * dates
    
#     for vi, var_name in enumerate(var_names):
#         forcings_searched_resample_arrays_v = forcings_searched_resample_arrays[vi]
        
#         logger.info(f"Extracting NLDAS forcing data for variable: {var_name}... ...")
        
#         for di, dt in tqdm(enumerate(date),
#                        colour="green",
#                        desc=f"loop for each date to extract forcing data"):
#             fn = generate_nldas_filename(dt)
#             fp = os.path.join(NLDAS_forcing_home, fn)
            
#             with Dataset(fp, "r") as src_dataset:
#                 src_var = src_dataset.variables[var_name]

#                 for gi in grid_shp.index:
#                     # get search grid index, lat, lon for this dst_grid
#                     searched_grid_index = searched_grids_index[gi]
#                     # dst_lat_grid = grid_shp.loc[gi, :].point_geometry.y
#                     # dst_lon_grid = grid_shp.loc[gi, :].point_geometry.x
        
#                     # searched_grid_lat = [
#                     #     NLDAS_forcing_lat[searched_grid_index[0][j]]
#                     #     for j in range(len(searched_grid_index[0]))
#                     # ]
                    
#                     # searched_grid_lon = [
#                     #     NLDAS_forcing_lon[searched_grid_index[1][j]]
#                     #     for j in range(len(searched_grid_index[0]))
#                     # ]
                    
#                     # get searched data
#                     if len(src_var.dimensions) == 3:
#                         searched_grid_data = [src_var[0, searched_grid_index[0][l], searched_grid_index[1][l]] for l in range(len(searched_grid_index[0]))]
                        
#                     elif len(src_var.dimensions) == 4:
#                         searched_grid_data = [src_var[0, 0, searched_grid_index[0][l], searched_grid_index[1][l]] for l in range(len(searched_grid_index[0]))]
                        
#                     else:
#                         logger.warning(f"Variable {var_name} has unsupported dimensions: {len(src_var.dimensions)}")
        
#                     # resample
#                     searched_resample_data = np.nanmean(np.array(searched_grid_data))
                    
#                     # append
#                     forcings_searched_resample_arrays_v[gi, di] = searched_resample_data
                    
#                     # # check
#                     # if check_search and vi+gi+di == 0:
#                     #     cgdf = CreateGDF()
#                     #     grid_shp_grid = grid_shp.loc[[gi], "geometry"]
#                     #     searched_grids_gdf = cgdf.createGDF_rectangle_central_coord(
#                     #         searched_grid_lon, searched_grid_lat, NLDAS_forcing_lat_res
#                     #     )

#                     #     fig, ax = plt.subplots()
#                     #     grid_shp_grid.boundary.plot(ax=ax, edgecolor="r", linewidth=2)  # target
#                     #     searched_grids_gdf.plot(
#                     #         ax=ax, edgecolor="k", linewidth=0.2, facecolor="b", alpha=0.5
#                     #     )  # searched data from source data
                        
#                     #     ax.set_title("check search")
                        
#                     #     plt.show(block=True)
    
#     # save
#     for j in range(len(var_names)):
#         # [v1, ..., v5], v1 = [grid1, ..., gridn], grid1 = [time1, ..., timek] (series)
#         grid_shp[f"{var_names[j]}"] = [row.tolist() for row in forcings_searched_resample_arrays[j]]
    
#     # postprocessing: unit change
#     # TMP: K -> -273.15 -> C
#     grid_shp["TMP"] = grid_shp["TMP"].apply(lambda row: np.array(row) - 273.15)  # K to C
    
#     # PRES: Pa -> /1000 -> kPa
#     grid_shp["PRES"] = grid_shp["PRES"].apply(lambda row: np.array(row) / 1000.0)  # Pa to kPa
    
#     # calculate VP, kPa
#     def compute_vp_series(row):
#         prs_kPa = row["PRES"]
#         sh_kg_per_kg = row["SPFH"]
        
#         vp_series = [
#             cal_VP_from_prs_sh(prs_kPa_day, sh_kg_per_kg_day) for prs_kPa_day, sh_kg_per_kg_day in zip(prs_kPa, sh_kg_per_kg)
#         ]
#         return vp_series

#     grid_shp["VP"] = grid_shp.apply(compute_vp_series, axis=1)
    
#     # cal wind
#     def compute_wind_series(row):
#         wind_u = row["UGRD"]
#         wind_v = row["VGRD"]
        
#         vp_series = [
#             (wind_u_day**2 + wind_v_day**2)**0.5  for wind_u_day, wind_v_day in zip(wind_u, wind_v)
#         ]
#         return vp_series

#     grid_shp["WIND"] = grid_shp.apply(compute_wind_series, axis=1)
    
#     # rename columns to add units
#     grid_shp.rename(
#         columns={
#             "TMP": "tmp_avg_C",  # C
#             "APCP": "pre_mm_per_day",  # mm/day
#             "PRES": "prs_kPa",  # kPa
#             "DSWRF": "swd_W_per_m2",  # W m-2
#             "DLWRF": "lwd_W_per_m2",  # W m-2
#             "SPFH": "shu_kg_per_kg",  # kg/kg
#             "VP": "vp_kPa",  # kPa
#             "WIND": "wind_m_per_s",  # m/s
#             # "UGRD"
#             # "VGRD"
#         },
#         inplace=True
#     )
    
#     # plot
#     if plot:
#         # plot timeseries
#         grid_i = 0
#         plot_var_name = "lwd_W_per_m2"
        
#         plt.figure(figsize=(10, 6))
#         plt.plot(date, grid_shp.loc[grid_shp.index[grid_i], f"{plot_var_name}"], label=plot_var_name)
#         plt.xlabel("Time")
#         plt.ylabel(plot_var_name)
#         plt.legend()
#         plt.title(f"Time Series of {plot_var_name} at Grid {grid_i}")
#         plt.show(block=True)
        
#         # plot map
#         fig, ax = plt.subplots()
        
#         grid_shp_plot = deepcopy(grid_shp)
#         grid_shp_plot[f"{plot_var_name}_timemean"] = grid_shp_plot.apply(
#             lambda row: np.nanmean(row[f"{plot_var_name}"]), axis=1
#         )
        
#         grid_shp_plot.plot(
#             f"{plot_var_name}_timemean",
#             ax=ax,
#             edgecolor="k",
#             linewidth=0.2,
#         )
#         ax.set_title(f"{plot_var_name} mean")
#         ax.set_xlim(
#             [min(grids_lon) - grid_shp_res / 2, max(grids_lon) + grid_shp_res / 2]
#         )
#         ax.set_ylim(
#             [min(grids_lat) - grid_shp_res / 2, max(grids_lat) + grid_shp_res / 2]
#         )
        
#         plt.show(block=True)
        
#     return grid_shp
        
    
if __name__ == "__main__":
    # ExtractData(None, ["19980101", "20101231"], grid_shp_res=0.25, plot=False, check_search=False)
    pass