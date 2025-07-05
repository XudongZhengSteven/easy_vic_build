# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
build_MeteForcing - A Python module for building meteorological foring files without nco package.

This module provides functions for constructing meteorological forcing data for the VIC model.
It includes the main function `buildMeteForcing` and a helper function `resampleTimeForcing` to:
- Organize and prepare meteorological forcing data for use in the VIC model.
- Resample meteorological forcing data to the required time steps.

Functions:
----------
    - `buildMeteForcing`: The main function that orchestrates the meteorological forcing data preparation.
    - `resampleTimeForcing`: Resample meteorological forcing data from the source directory to a target time step, and save the results as NetCDF files.

Usage:
------
    1. provide an `evb_dir` instance that specifies the directory structure for original meteorological forcing data (`evb_dir.MeteForcing_src_dir`).
    2. Specify the `evb_dir.MeteForcing_src_suffix`.
    3. Call `buildMeteForcing`.
    4. (Optional) Call `resampleTimeForcing` for Resampling time-based forcing data.

Example:
--------
    basin_index = 213
    model_scale = "6km"
    date_period = ["19980101", "19981231"]
    case_name = f"{basin_index}_{model_scale}"

    evb_dir = Evb_dir("./examples")
    evb_dir.builddir(case_name)

    dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level1 = readdpc(evb_dir)

    evb_dir.MeteForcing_src_dir = "E:\\data\\hydrometeorology\\NLDAS\\NLDAS2_Primary_Forcing_Data_subset_0.125\\data"
    evb_dir.MeteForcing_src_suffix = ".nc4"

    buildMeteForcing(evb_dir, dpc_VIC_level1, date_period,
                     reverse_lat=True, check_search=False,
                     time_re_exp=r"\d{8}.\d{4}")

Dependencies:
-------------
    - `os`: For file and directory operations.
    - `numpy`: For numerical operations.
    - `re`: For regular expressions.
    - `datetime`: For date and time handling.
    - `netCDF4`: For reading and writing NetCDF files.
    - `cftime`: For handling time units and calendars in NetCDF files.
    - `tqdm`: For progress bars during file processing.
    - `matplotlib`: For plotting data (if required in the future).
    - `xarray`: For handling multidimensional arrays and working with NetCDF files.

"""

# TODO parallel
import os
import re
from datetime import datetime

import cftime
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from tqdm import *

from . import logger
from .tools.decoractors import clock_decorator
from .tools.dpc_func.basin_grid_func import grids_array_coord_map
from .tools.geo_func import search_grids
from .tools.geo_func.create_gdf import CreateGDF
from .tools.mete_func.resampleTimeForcing import resampleTimeForcing
from .tools.mete_func.createMeteForcingDataset import createMeteForcingDataset


@clock_decorator(print_arg_ret=False)
def buildMeteForcing(
    evb_dir,
    dpc_VIC_level1,
    date_period,
    reverse_lat=True,
    check_search=False,
    time_re_exp=r"\d{8}.\d{4}",
    search_func=search_grids.search_grids_radius_rectangle_reverse,
    dst_time_hours=None,
):
    """
    Build meteorological forcing dataset for VIC model.

    Parameters
    ----------
    evb_dir : `Evb_dir`
        An instance of the `Evb_dir` class, containing paths for VIC deployment.
        
    dpc_VIC_level1 : `dpc_VIC_level1`
        An instance of the `dpc_VIC_level1` class to process data at level 1 of the VIC model.

    date_period : list of str
        A list containing the start and end dates for the desired period in the format ["YYYYMMDD", "YYYYMMDD"].
    
    reverse_lat : bool
        Boolean flag to indicate whether to reverse latitudes (Northern Hemisphere: large -> small, set as True).
        
    check_search : bool, optional
        Whether to perform a search check (default is False).
        
    time_re_exp : str, optional
        Regular expression pattern for extracting time information (default is r"\d{8}.\d{4}").
        
    search_func : function, optional
        Function used to search grid indices (default is `search_grids.search_grids_radius_rectangle_reverse`).
    
    dst_time_hours : int, optional
        The target time step in hours for resampling (default is 24 hours).

    Returns
    -------
    None
        The function creates a NetCDF file for the forcing data and stores it in the specified output directory.
    """
    # Start of the forcing building process, log an info message
    logger.info("Starting to build meteorological forcing files... ...")

    # ====================== set dir and path ======================
    # set path
    src_home = evb_dir.MeteForcing_src_dir
    suffix = evb_dir.MeteForcing_src_suffix
    src_names = [n for n in os.listdir(src_home) if n.endswith(suffix)]
    MeteForcing_dir = evb_dir.MeteForcing_dir

    logger.debug(f"set src home: {src_home}, suffix: {suffix}")
    logger.debug(f"set MeteForcing_dir: {MeteForcing_dir}")

    ## ====================== get grid_shp and basin_shp ======================
    grid_shp = dpc_VIC_level1.grid_shp
    basin_shp = dpc_VIC_level1.basin_shp

    # grids_map_array
    lon_list, lat_list, lon_map_index, lat_map_index = grids_array_coord_map(
        grid_shp, reverse_lat=reverse_lat
    )  # * all lat set as reverse

    ## ====================== loop for read year and create forcing ======================
    # set time
    start_year = int(date_period[0][:4])
    end_year = int(date_period[1][:4])
    logger.debug(f"set forcing date: {start_year}-{end_year}")

    year = start_year
    while year <= end_year:
        logger.info(f"creating forcing for year: {year}, end with year: {end_year}... ...")

        # get files
        src_names_year = [n for n in src_names if "A" + str(year) in n]
        src_names_year.sort()

        # get times
        time_str = [re.search(time_re_exp, n)[0] for n in src_names_year]
        time_datetime = [datetime.strptime(t, "%Y%m%d.%H00") for t in time_str]

        # create nc
        dst_path_year = os.path.join(MeteForcing_dir, f"{evb_dir.forcing_prefix}.{year}.nc")
        logger.debug(f"Creating NetCDF file: {dst_path_year}... ...")
        dst_dataset, time_v, lats, lons, lat_v, lon_v =  createMeteForcingDataset(dst_path_year, lat_list, lon_list, time_datetime, start_time=date_period[0])

        # loop for read data in this year
        for i in tqdm(
            range(len(src_names_year)),
            desc="loop for read data in this year",
            colour="green",
        ):
            src_name_year_i = src_names_year[i]
            src_path_year_i = os.path.join(src_home, src_name_year_i)

            with Dataset(src_path_year_i, "r") as src_dataset:
                logger.debug(f"Reading data from: {src_path_year_i}")

                # get time index
                src_dataset_time = src_dataset.variables["time"]
                src_time_cftime = cftime.num2date(
                    src_dataset_time[:][0],
                    units=src_dataset_time.units,
                    calendar=src_dataset_time.calendar,
                )
                src_time_datetime = datetime(
                    src_time_cftime.year,
                    src_time_cftime.month,
                    src_time_cftime.day,
                    src_time_cftime.hour,
                    src_time_cftime.minute,
                    src_time_cftime.second,
                )
                src_time_datetime_in_dst_dataset_index = int(
                    cftime.date2index(
                        src_time_datetime, time_v, calendar=time_v.calendar
                    )
                )

                # get lat, lon index
                if i == 0:  # just search once when all src_file is consistent
                    src_lat = src_dataset.variables["lat"][:]
                    src_lon = src_dataset.variables["lon"][:]
                    src_lat_res = (max(src_lat) - min(src_lat)) / (len(src_lat) - 1)
                    src_lon_res = (max(src_lon) - min(src_lon)) / (len(src_lon) - 1)

                    lats_flatten = lats[:, :].flatten()
                    lons_flatten = lons[:, :].flatten()
                    dst_lat_res = (max(lat_v) - min(lat_v)) / (len(lat_v) - 1)
                    dst_lon_res = (max(lon_v) - min(lon_v)) / (len(lon_v) - 1)
                    searched_grids_index = search_func(
                        dst_lat=lats_flatten,
                        dst_lon=lons_flatten,
                        src_lat=src_lat,
                        src_lon=src_lon,
                        lat_radius=src_lat_res / 2,
                        lon_radius=src_lon_res / 2,
                        leave=False,
                    )

                # search data
                tas_list = []
                prcp_list = []
                pres_list = []
                dswrf_list = []
                dlwrf_list = []
                spfh_list = []
                wind_u_list = []
                wind_v_list = []
                for j in range(len(lats_flatten)):
                    # lon/lat
                    searched_grid_index = searched_grids_index[j]
                    searched_grid_lat = [
                        src_lat[searched_grid_index[0][k]]
                        for k in range(len(searched_grid_index[0]))
                    ]
                    searched_grid_lon = [
                        src_lon[searched_grid_index[1][k]]
                        for k in range(len(searched_grid_index[0]))
                    ]

                    # tas, AIR_TEMP
                    searched_grid_data_tas = [
                        src_dataset.variables["TMP"][
                            0,
                            0,
                            searched_grid_index[0][k],
                            searched_grid_index[1][k],
                        ]
                        for k in range(len(searched_grid_index[0]))
                    ][0]
                    tas_list.append(searched_grid_data_tas)

                    # prcp, PREC
                    searched_grid_data_prcp = [
                        src_dataset.variables["APCP"][
                            0, searched_grid_index[0][k], searched_grid_index[1][k]
                        ]
                        for k in range(len(searched_grid_index[0]))
                    ][0]
                    prcp_list.append(searched_grid_data_prcp)

                    # pres, PRESSURE
                    searched_grid_data_pres = [
                        src_dataset.variables["PRES"][
                            0, searched_grid_index[0][k], searched_grid_index[1][k]
                        ]
                        for k in range(len(searched_grid_index[0]))
                    ][0]
                    pres_list.append(searched_grid_data_pres)

                    # dswrf, SWDOWN
                    searched_grid_data_dswrf = [
                        src_dataset.variables["DSWRF"][
                            0, searched_grid_index[0][k], searched_grid_index[1][k]
                        ]
                        for k in range(len(searched_grid_index[0]))
                    ][0]
                    dswrf_list.append(searched_grid_data_dswrf)

                    # dlwrf, LWDOWN
                    searched_grid_data_dlwrf = [
                        src_dataset.variables["DLWRF"][
                            0, searched_grid_index[0][k], searched_grid_index[1][k]
                        ]
                        for k in range(len(searched_grid_index[0]))
                    ][0]
                    dlwrf_list.append(searched_grid_data_dlwrf)

                    # vp, VP = (SPFH * PERS) / (0.622 + SPFH)
                    searched_grid_data_SPFH = [
                        src_dataset.variables["SPFH"][
                            0,
                            0,
                            searched_grid_index[0][k],
                            searched_grid_index[1][k],
                        ]
                        for k in range(len(searched_grid_index[0]))
                    ][0]
                    spfh_list.append(searched_grid_data_SPFH)

                    # wind, Wind = (u**2 + v**2) ** (0.5)
                    searched_grid_data_wind_u = [
                        src_dataset.variables["UGRD"][
                            0,
                            0,
                            searched_grid_index[0][k],
                            searched_grid_index[1][k],
                        ]
                        for k in range(len(searched_grid_index[0]))
                    ][0]
                    wind_u_list.append(searched_grid_data_wind_u)
                    searched_grid_data_wind_v = [
                        src_dataset.variables["VGRD"][
                            0,
                            0,
                            searched_grid_index[0][k],
                            searched_grid_index[1][k],
                        ]
                        for k in range(len(searched_grid_index[0]))
                    ][0]
                    wind_v_list.append(searched_grid_data_wind_v)

                    # check
                    if check_search and i == 0:
                        fig, ax = plt.subplots()
                        cgdf = CreateGDF()
                        dst_grids_gdf = cgdf.createGDF_rectangle_central_coord(
                            lons_flatten, lats_flatten, dst_lat_res
                        )
                        src_grids_gdf = cgdf.createGDF_rectangle_central_coord(
                            searched_grid_lon, searched_grid_lat, src_lat_res
                        )

                        src_grids_gdf.boundary.plot(
                            ax=ax, edgecolor="r", linewidth=2
                        )
                        dst_grids_gdf.plot(
                            ax=ax,
                            edgecolor="k",
                            linewidth=0.2,
                            facecolor="b",
                            alpha=0.5,
                        )
                        ax.set_title("check search")

                # reshape to 2D
                tas_array_2D = np.reshape(np.array(tas_list), lats.shape)
                prcp_array_2D = np.reshape(np.array(prcp_list), lats.shape)
                pres_array_2D = np.reshape(np.array(pres_list), lats.shape)
                dswrf_array_2D = np.reshape(np.array(dswrf_list), lats.shape)
                dlwrf_array_2D = np.reshape(np.array(dlwrf_list), lats.shape)
                spfh_array_2D = np.reshape(np.array(spfh_list), lats.shape)
                wind_u_array_2D = np.reshape(np.array(wind_u_list), lats.shape)
                wind_v_array_2D = np.reshape(np.array(wind_v_list), lats.shape)

                ## unit change
                # AIR_TEMP: K->C,  x - 273.15
                tas_array_2D -= 273.15

                # PERS：Pa->Kpa, x / 1000
                pres_array_2D /= 1000

                ## cal other variables
                # cal vp, Kpa, same as PRES
                vp_array_2D = (spfh_array_2D * pres_array_2D) / (
                    0.622 + spfh_array_2D
                )  # VP = (SPFH * PERS) / (0.622 + SPFH)

                # cal wind
                wind_array_2D = (wind_u_array_2D**2 + wind_v_array_2D**2) ** (0.5)

                ## append into dst_dataset
                dst_dataset.variables["tas"][
                    src_time_datetime_in_dst_dataset_index, :, :
                ] = tas_array_2D
                dst_dataset.variables["prcp"][
                    src_time_datetime_in_dst_dataset_index, :, :
                ] = prcp_array_2D
                dst_dataset.variables["pres"][
                    src_time_datetime_in_dst_dataset_index, :, :
                ] = pres_array_2D
                dst_dataset.variables["dswrf"][
                    src_time_datetime_in_dst_dataset_index, :, :
                ] = dswrf_array_2D
                dst_dataset.variables["dlwrf"][
                    src_time_datetime_in_dst_dataset_index, :, :
                ] = dlwrf_array_2D
                dst_dataset.variables["vp"][
                    src_time_datetime_in_dst_dataset_index, :, :
                ] = vp_array_2D
                dst_dataset.variables["wind"][
                    src_time_datetime_in_dst_dataset_index, :, :
                ] = wind_array_2D

        # close dataset
        dst_dataset.close()
            
        # next year
        year += 1
        logger.info(f"Finished processing for year: {year}")

    ## ====================== loop for read year and resample time forcing ======================
    if dst_time_hours is not None:
        resampleTimeForcing(evb_dir, dst_time_hours)

    logger.info("Building meteorological forcing files successfully")



