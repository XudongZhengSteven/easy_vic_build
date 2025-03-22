# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
"""
Module: build_MeteForcing_nco

This module provides functions for constructing meteorological forcing data for the VIC model using NetCDF Operators (NCO). It includes functionalities for clipping source meteorological forcing data to a basin boundary, regridding and formatting data, and resampling data to required time steps.

Functions
---------
    - buildMeteForcingnco: Orchestrates the meteorological forcing data preparation process.
    - clip_src_data_for_basin: Clips the source meteorological forcing data to the given basin boundary.
    - formationForcing: Regrids and formats meteorological forcing data for VIC input.
    - resampleTimeForcing: Resamples meteorological forcing data from the source directory to a target time step and saves the results as NetCDF files.

Usage
-----
To use this module, provide an `evb_dir` instance that specifies the directory structure for meteorological forcing data. The `buildMeteForcingnco` function will handle data preparation, including clipping, formatting, and resampling.

Example
-------
    basin_index = 213
    model_scale = "6km"
    date_period = ["19980101", "19981231"]
    case_name = f"{basin_index}_{model_scale}"

    evb_dir = Evb_dir("./examples")
    evb_dir.builddir(case_name)

    dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2 = readdpc(evb_dir)

    evb_dir.MeteForcing_src_dir = "E:\\data\\hydrometeorology\\NLDAS\\NLDAS2_Primary_Forcing_Data_subset_0.125\\data"
    evb_dir.MeteForcing_src_suffix = ".nc4"

    evb_dir.linux_share_temp_dir = "F:\\Linux\\C_VirtualBox_Share\\temp"

    combineYearly_py_path = "F:\\research\\Research\\easy_vic_build\\easy_vic_build\\scripts\\linux_scripts\\combineYearly.py"
    if not os.path.exists(os.path.join(evb_dir.linux_share_temp_dir, "combineYearly.py")):
        shutil.copy(combineYearly_py_path, os.path.join(evb_dir.linux_share_temp_dir, "combineYearly.py"))

    # build MeteForcingnco: step=1
    buildMeteForcingnco(evb_dir, dpc_VIC_level1, date_period,
                         step=1, reverse_lat=True, check_search=False,
                         year_re_exp=r"\d{4}.nc4")

    # go to linux, run combineYearly.py

    # build MeteForcingnco: step=2
    # buildMeteForcingnco(evb_dir, dpc_VIC_level1, date_period,
    #                     step=2, reverse_lat=True, check_search=False,
    #                     year_re_exp=r"\d{4}.nc4")

    # build MeteForcingnco: step=3
    # buildMeteForcingnco(evb_dir, dpc_VIC_level1, date_period,
    #                     step=3, reverse_lat=True, check_search=False,
    #                     year_re_exp=r"\d{4}.nc4")

    # build MeteForcingnco: step=4, resample
    # buildMeteForcingnco(evb_dir, dpc_VIC_level1, date_period,
    #                     step=4, reverse_lat=True, check_search=False,
    #                     year_re_exp=r"\d{4}.nc4",
    #                     dst_time_hours=24)

Dependencies
------------
    - os: For file and directory operations.
    - numpy: For numerical computations.
    - re: For handling regular expressions.
    - shutil: For file and directory operations.
    - datetime: For managing time and date.
    - netCDF4: For reading and writing NetCDF files.
    - cftime: For handling time units and calendars in NetCDF files.
    - tqdm: For displaying progress bars during processing.
    - matplotlib: For potential data visualization.
    - xarray: For handling multidimensional arrays and working with NetCDF files.
    - nco: For interacting with NetCDF Operators.
    - CreateGDF: For creating geospatial data frames.
    - search_grids: For searching grid points within a given domain.
    - grids_array_coord_map: For mapping basin grid coordinates.
    - check_and_mkdir, remove_and_mkdir: Utility functions for directory management.
    - clock_decorator: A decorator for measuring function execution time.

Author
------
    Xudong Zheng
    Email: z786909151@163.com
"""
# * use nco to increase speed (you need a Linux system, as the ncrcat has not been implemented in pynco)
# * This is particularly useful for large domain
# TODO parallel


import os
import re
import shutil
from datetime import datetime

import cftime
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from nco import Nco
from nco.custom import Limit
from netCDF4 import Dataset
from tqdm import *

from . import logger
from .tools.decoractors import clock_decorator
from .tools.dpc_func.basin_grid_func import grids_array_coord_map
from .tools.geo_func import search_grids
from .tools.geo_func.create_gdf import CreateGDF
from .tools.utilities import check_and_mkdir, remove_and_mkdir


@clock_decorator(print_arg_ret=False)
def buildMeteForcingnco(
    evb_dir,
    dpc_VIC_level1,
    date_period,
    step=1,
    reverse_lat=True,
    check_search=False,
    year_re_exp=r"A\d{4}.nc4",
    dst_time_hours=24,
):
    """
    Build meteorological forcing data for VIC model.

    Parameters
    ----------
    evb_dir : object
        Directory object containing path configurations.
    dpc_VIC_level1 : object
        Input data for VIC model at level 1.
    date_period : tuple of datetime
        Start and end date for the forcing data.
    step : int, optional
        Step number to control the workflow (default is 1).
    reverse_lat : bool, optional
        Whether to reverse latitude (default is True).
    check_search : bool, optional
        Whether to perform a search check (default is False).
    year_re_exp : str, optional
        Regular expression for year matching (default is "A\d{4}.nc4").
    dst_time_hours : int, optional
        Time resolution in hours for resampling (default is 24).

    Returns
    -------
    None
        Function executes a series of steps to generate meteorological forcing data.

    Notes
    -----
    This function consists of multiple steps including:
    1. Clipping data for the basin.
    2. Combining yearly data and forming forcing data.
    3. Cleaning temporary data.
    4. Resampling time-based forcing data.
    """

    # Start of the forcing building process, log an info message
    logger.info(
        f"Starting to build meteorological forcing files for step {step}... ..."
    )

    # ====================== set dir and path ======================
    # set path
    MeteForcing_dir = evb_dir.MeteForcing_dir
    MeteForcing_clip_dir = os.path.join(MeteForcing_dir, "clip")
    MeteForcing_combineYearly_dir = os.path.join(MeteForcing_dir, "combineYearly")
    # combineYearly_py_path = os.path.join(evb_dir.__package_dir__, "linux_scripts\\combineYearly.py")

    linux_share_temp_dir = evb_dir.linux_share_temp_dir
    check_and_mkdir(linux_share_temp_dir)

    linux_share_temp_clip_dir = os.path.join(linux_share_temp_dir, "clip")
    linux_share_temp_combineYearly_dir = os.path.join(
        linux_share_temp_dir, "combineYearly"
    )
    ## ====================== step1: clip for basin ======================
    if step == 1:
        logger.info("Step 1: Clipping data for basin...")
        clip_src_data_for_basin(evb_dir, dpc_VIC_level1, date_period, reverse_lat)

        # * mv MeteForcing_clip_dir to linux_share_temp_dir/clip, this is used for window users
        if os.path.exists(linux_share_temp_clip_dir):
            shutil.rmtree(linux_share_temp_clip_dir)

        shutil.move(MeteForcing_clip_dir, linux_share_temp_dir)

        # -------------------- cp combineYearly.py to share_temp_home --------------------
        # shutil.copy(combineYearly_py_path, os.path.join(linux_share_temp_dir, "combineYearly.py"))

        # -------------------- cd to share_temp_home and run combineYearly.py --------------------
        # * cd to share_temp_home (go to Linux): run combineYearly.py (python combineYearly.py ./)

    elif step == 2:
        # -------------------- formationForcing: regrid, formation --------------------
        # * mv linux_share_temp_dir/combineYearly back to MeteForcing_combineYearly_dir
        # if os.path.exists(dst_dir):
        #     shutil.rmtree(dst_dir)
        logger.info("Step 2: Combining yearly data and forming forcing...")
        try:
            shutil.move(
                linux_share_temp_combineYearly_dir, MeteForcing_combineYearly_dir
            )
        except FileNotFoundError:
            logger.warning("CombineYearly directory already moved")

        # formationForcing
        formationForcing(
            evb_dir, dpc_VIC_level1, date_period, reverse_lat, check_search, year_re_exp
        )

    elif step == 3:
        # -------------------- clean temp data --------------------
        logger.info("Step 3: Cleaning temporary data...")

        logger.info(f"Removing {linux_share_temp_clip_dir}")
        shutil.rmtree(linux_share_temp_clip_dir)

        logger.info(f"Removing {MeteForcing_combineYearly_dir}")
        shutil.rmtree(MeteForcing_combineYearly_dir)

    elif step == 4:
        # -------------------- resample time forcing --------------------
        logger.info("Step 4: Resampling time-based forcing data...")
        resampleTimeForcing(evb_dir, dst_time_hours=dst_time_hours)

    else:
        logger.error("Invalid step number. Please input a valid step number")

    logger.info(
        f"Building meteorological forcing files for step {step} completed successfully"
    )


def clip_src_data_for_basin(evb_dir, dpc_VIC_level1, date_period, reverse_lat=True):
    """
    Clip the meteorological forcing data to the basin's boundary for a given date period.

    Parameters
    ----------
    evb_dir : Evb_dir
        An instance of the `Evb_dir` class, which provides paths to the source and destination directories for the forcing data.
    dpc_VIC_level1 : DPC_VIC_Level1
        An instance of the `DPC_VIC_Level1` class, which provides basin shape and grid shape information.
    date_period : list of str
        A list containing the start and end dates for the desired period in the format ["YYYYMMDD", "YYYYMMDD"].
    reverse_lat : bool, optional
        If True, the latitude values will be reversed. Default is True.

    Returns
    -------
    None
        The function clips the forcing data and saves the results to the specified output directory.

    Notes
    -----
    The clipping is performed for each year within the provided `date_period`. The clipped files are saved
    with the `.clip.nc4` suffix in the `clip` directory.

    """
    # Start of the parameter building process, log an info message
    logger.info("Starting to clip_src_data_for_basin... ...")

    # ====================== set dir and path ======================
    # set path
    src_home = evb_dir.MeteForcing_src_dir
    suffix = evb_dir.MeteForcing_src_suffix
    src_names = [n for n in os.listdir(src_home) if n.endswith(suffix)]

    MeteForcing_dir = evb_dir.MeteForcing_dir
    MeteForcing_clip_dir = os.path.join(MeteForcing_dir, "clip")
    check_and_mkdir(MeteForcing_clip_dir)

    ## ====================== get grid_shp and basin_shp ======================
    grid_shp = dpc_VIC_level1.grid_shp
    basin_shp = dpc_VIC_level1.basin_shp

    # grids_map_array
    lon_list, lat_list, lon_map_index, lat_map_index = grids_array_coord_map(
        grid_shp, reverse_lat=reverse_lat
    )  # * all lat set as reverse

    ## ====================== clip MeteForcing src data ======================
    # set time
    start_year = int(date_period[0][:4])
    end_year = int(date_period[1][:4])

    # get dpc_VIC_level1.grid_shp boundary
    xmin_level1 = dpc_VIC_level1.grid_shp["point_geometry"].x.min()
    xmax_level1 = dpc_VIC_level1.grid_shp["point_geometry"].x.max()
    ymin_level1 = dpc_VIC_level1.grid_shp["point_geometry"].y.min()
    ymax_level1 = dpc_VIC_level1.grid_shp["point_geometry"].y.max()

    # get src_lat, src_lon
    with Dataset(os.path.join(src_home, src_names[0]), "r") as src_dataset:
        src_lat = src_dataset.variables["lat"][:]
        src_lon = src_dataset.variables["lon"][:]

    # set clip boundary
    src_lat_sorted = np.sort(src_lat)
    src_lon_sorted = np.sort(src_lon)

    ymin = src_lat_sorted[np.where(src_lat_sorted <= ymin_level1)][-1]
    ymax = src_lat_sorted[np.where(src_lat_sorted >= ymax_level1)][0]

    xmin = src_lon_sorted[np.where(src_lon_sorted <= xmin_level1)][-1]
    xmax = src_lon_sorted[np.where(src_lon_sorted >= xmax_level1)][0]

    # nco
    year = start_year
    while year <= end_year:
        logger.info(
            f"Clipping forcing data for year {year}, ending with year {end_year}."
        )

        # get files
        src_names_year = [n for n in src_names if "A" + str(year) in n]
        src_names_year.sort()

        # loop for clip files
        nco_ = Nco()
        opt = [Limit("lon", xmin, xmax), Limit("lat", ymin, ymax)]
        for i in tqdm(range(len(src_names_year))):
            src_name_year = src_names_year[i]
            src_path = os.path.join(src_home, src_name_year)
            dst_fname = src_name_year[: src_name_year.find(".nc4")] + ".clip.nc4"
            dst_path = os.path.join(MeteForcing_clip_dir, dst_fname)

            logger.debug(f"Clipping file: {src_name_year}")
            nco_.ncks(input=src_path, output=dst_path, options=opt)

        # next year
        year += 1
        logger.info(f"Finished processing for year: {year}")

    logger.info("clip_src_data_for_basin completed successfully")


def formationForcing(
    evb_dir,
    dpc_VIC_level1,
    date_period,
    reverse_lat=True,
    check_search=False,
    year_re_exp=r"\d{4}.nc4",
):
    """
    Format meteorological forcing data and generate NetCDF files for VIC model.

    Parameters
    ----------
    evb_dir : object
        Directory information for meteorological forcing data.
    dpc_VIC_level1 : object
        Level 1 data processor object containing grid and basin shape.
    date_period : tuple
        A tuple of start and end date for the data period.
    reverse_lat : bool, optional
        If True, reverses the latitude order (default is True).
    check_search : bool, optional
        If True, enables grid search check and visualization (default is False).
    year_re_exp : str, optional
        Regular expression pattern for extracting the year from file names (default is r"\d{4}.nc4").

    Returns
    -------
    None
        The function saves the formatted forcing data to NetCDF files.

    Notes
    -----
    This function processes meteorological forcing data from source files, transforms the data according to
    latitude/longitude mapping, and saves the data in NetCDF format for use in VIC simulations.
    """

    logger.info("Starting to formating meteorological forcing files... ...")

    # ====================== set dir and path ======================
    # set path
    suffix = evb_dir.MeteForcing_src_suffix
    MeteForcing_dir = evb_dir.MeteForcing_dir
    MeteForcing_combineYearly_dir = os.path.join(MeteForcing_dir, "combineYearly")
    src_home = MeteForcing_combineYearly_dir
    src_names = [n for n in os.listdir(src_home) if n.endswith(suffix)]

    logger.debug(f"set src home: {src_home}, suffix: {suffix}")
    logger.debug(f"set MeteForcing_dir: {MeteForcing_dir}")

    ## ====================== get grid_shp and basin_shp ======================
    grid_shp = dpc_VIC_level1.grid_shp
    basin_shp = dpc_VIC_level1.basin_shp

    # grids_map_array
    lon_list, lat_list, lon_map_index, lat_map_index = grids_array_coord_map(
        grid_shp, reverse_lat=reverse_lat
    )  # * all lat set as reverse

    ## ====================== search grids for match ======================
    # src grids
    with Dataset(os.path.join(src_home, src_names[0]), "r") as src_dataset:
        src_lat_ = src_dataset.variables["lat"][:]
        src_lon_ = src_dataset.variables["lon"][:]
        src_lat_res_ = (max(src_lat_) - min(src_lat_)) / (len(src_lat_) - 1)
        src_lon_res_ = (max(src_lon_) - min(src_lon_)) / (len(src_lon_) - 1)

    # dst grids
    grid_array_lons_, grid_array_lats_ = np.meshgrid(
        np.array(lon_list), np.array(lat_list)
    )  # 2D array

    lats_flatten_ = grid_array_lats_.flatten()
    lons_flatten_ = grid_array_lons_.flatten()

    # search
    logger.info("search grids for match src and dst data")
    searched_grids_index = search_grids.search_grids_radius_rectangle_reverse(
        dst_lat=lats_flatten_,
        dst_lon=lons_flatten_,
        src_lat=src_lat_,
        src_lon=src_lon_,
        lat_radius=src_lat_res_ / 2,
        lon_radius=src_lon_res_ / 2,
    )

    ## ====================== loop for forcing formation ======================
    for i in tqdm(
        range(len(src_names)), desc="loop for forcing formation", colour="green"
    ):
        # general
        src_name = src_names[i]
        src_path = os.path.join(src_home, src_name)

        # get year
        year = re.search(year_re_exp, src_name)[0][:4]

        logger.info(f"formating forcing for year: {year}")

        # read src
        with Dataset(src_path, "r") as src_dataset:
            logger.debug(f"Reading data from: {src_path}")

            # get lat, lon index
            src_lat = src_dataset.variables["lat"][:]
            src_lon = src_dataset.variables["lon"][:]
            src_lat_res = (max(src_lat) - min(src_lat)) / (len(src_lat) - 1)
            src_lon_res = (max(src_lon) - min(src_lon)) / (len(src_lon) - 1)

            src_lon_flatten, src_lat_flatten = np.meshgrid(src_lon, src_lat)
            src_lon_flatten = src_lon_flatten.flatten()
            src_lat_flatten = src_lat_flatten.flatten()

            # get time
            src_dataset_time = src_dataset.variables["time"]
            src_time_cftime = [
                cftime.num2date(
                    src_dataset_time[t_i],
                    units=src_dataset_time.units,
                    calendar=src_dataset_time.calendar,
                )
                for t_i in range(len(src_dataset_time[:]))
            ]
            src_time_datetime = [
                datetime(t.year, t.month, t.day, t.hour, t.minute, t.second)
                for t in src_time_cftime
            ]

            # create nc
            dst_path_year = os.path.join(
                MeteForcing_dir, f"{evb_dir.forcing_prefix}.{year}.nc"
            )
            with Dataset(dst_path_year, "w") as dst_dataset:
                # define dimension
                time_dim = dst_dataset.createDimension("time", len(src_time_datetime))
                lat_dim = dst_dataset.createDimension("lat", len(lat_list))
                lon_dim = dst_dataset.createDimension("lon", len(lon_list))

                # define dimension variables
                time_v = dst_dataset.createVariable("time", int, ("time",))
                lat_v = dst_dataset.createVariable("lat", "f8", ("lat",))  # 1D array
                lon_v = dst_dataset.createVariable("lon", "f8", ("lon",))  # 1D array
                lats = dst_dataset.createVariable(
                    "lats",
                    "f8",
                    (
                        "lat",
                        "lon",
                    ),
                )  # 2D array
                lons = dst_dataset.createVariable(
                    "lons",
                    "f8",
                    (
                        "lat",
                        "lon",
                    ),
                )  # 2D array

                # assign attribute for dimension variables
                time_v.calendar = "proleptic_gregorian"
                time_v.units = f"hours since {date_period[0][:4]}-{date_period[0][4:6]}-{date_period[0][6:8]} 00:00:00"

                lat_v.units = "degrees_north"
                lat_v.long_name = "latitude of grid cell center"
                lat_v.standard_name = "latitude"
                lat_v.axis = "Y"

                lon_v.units = "degrees_east"
                lon_v.long_name = "longitude of grid cell center"
                lon_v.standard_name = "longitude"
                lon_v.axis = "X"

                lats.long_name = "lats 2D"
                lats.description = "Latitude of grid cell 2D"
                lats.units = "degrees"

                lons.long_name = "lons 2D"
                lons.description = "longitude of grid cell 2D"
                lons.units = "degrees"

                # assign values for dimension variables
                dst_dataset.variables["time"][:] = [
                    cftime.date2num(t, units=time_v.units, calendar=time_v.calendar)
                    for t in src_time_datetime
                ]
                dst_dataset.variables["lat"][:] = np.array(lat_list)  # 1D array
                dst_dataset.variables["lon"][:] = np.array(lon_list)  # 1D array
                grid_array_lons, grid_array_lats = np.meshgrid(
                    dst_dataset.variables["lon"][:], dst_dataset.variables["lat"][:]
                )  # 2D array
                dst_dataset.variables["lons"][:, :] = grid_array_lons  # 2D array
                dst_dataset.variables["lats"][:, :] = grid_array_lats  # 2D array

                # get time index
                src_time_datetime_in_dst_dataset_index = [
                    int(cftime.date2index(t, time_v, calendar=time_v.calendar))
                    for t in src_time_datetime
                ]

                # define variables
                tas = dst_dataset.createVariable(
                    "tas",
                    "f4",
                    (
                        "time",
                        "lat",
                        "lon",
                    ),
                )
                prcp = dst_dataset.createVariable(
                    "prcp",
                    "f4",
                    (
                        "time",
                        "lat",
                        "lon",
                    ),
                )
                pres = dst_dataset.createVariable(
                    "pres",
                    "f4",
                    (
                        "time",
                        "lat",
                        "lon",
                    ),
                )
                dswrf = dst_dataset.createVariable(
                    "dswrf",
                    "f4",
                    (
                        "time",
                        "lat",
                        "lon",
                    ),
                )
                dlwrf = dst_dataset.createVariable(
                    "dlwrf",
                    "f4",
                    (
                        "time",
                        "lat",
                        "lon",
                    ),
                )
                vp = dst_dataset.createVariable(
                    "vp",
                    "f4",
                    (
                        "time",
                        "lat",
                        "lon",
                    ),
                )
                wind = dst_dataset.createVariable(
                    "wind",
                    "f4",
                    (
                        "time",
                        "lat",
                        "lon",
                    ),
                )

                # assign attribute for variables
                tas.long_name = "AIR_TEMP"
                tas.description = "Average air temperature"
                tas.units = "C"

                prcp.long_name = "PREC"
                prcp.description = "Total precipitation (rain and snow)"
                prcp.units = "mm"

                pres.long_name = "PRESSURE"
                pres.description = "Atmospheric pressure"
                pres.units = "kPa"

                dswrf.long_name = "SWDOWN"
                dswrf.description = "Incoming shortwave radiation"
                dswrf.units = "W/m2"

                dlwrf.long_name = "LWDOWN"
                dlwrf.description = "Incoming longwave radiation"
                dlwrf.units = "W/m2"

                vp.long_name = "VP"
                vp.description = "Vapor pressure"
                vp.units = "kPa"

                wind.long_name = "WIND"
                wind.description = "Wind speed"
                wind.units = "m/s"

                # search grids
                lats_flatten = lats[:, :].flatten()
                lons_flatten = lons[:, :].flatten()
                dst_lat_res = (max(lat_v) - min(lat_v)) / (len(lat_v) - 1)
                dst_lon_res = (max(lon_v) - min(lon_v)) / (len(lon_v) - 1)

                # search data
                tas_array_3D = np.empty(
                    (len(src_time_datetime_in_dst_dataset_index), *lats.shape),
                    dtype=float,
                )
                prcp_array_3D = np.empty(
                    (len(src_time_datetime_in_dst_dataset_index), *lats.shape),
                    dtype=float,
                )
                pres_array_3D = np.empty(
                    (len(src_time_datetime_in_dst_dataset_index), *lats.shape),
                    dtype=float,
                )
                dswrf_array_3D = np.empty(
                    (len(src_time_datetime_in_dst_dataset_index), *lats.shape),
                    dtype=float,
                )
                dlwrf_array_3D = np.empty(
                    (len(src_time_datetime_in_dst_dataset_index), *lats.shape),
                    dtype=float,
                )
                spfh_array_3D = np.empty(
                    (len(src_time_datetime_in_dst_dataset_index), *lats.shape),
                    dtype=float,
                )
                wind_u_array_3D = np.empty(
                    (len(src_time_datetime_in_dst_dataset_index), *lats.shape),
                    dtype=float,
                )
                wind_v_array_3D = np.empty(
                    (len(src_time_datetime_in_dst_dataset_index), *lats.shape),
                    dtype=float,
                )

                for j in tqdm(range(len(lats_flatten)), desc="loop for read variables"):
                    # src_lat/lon
                    src_lat_j = lats_flatten[j]
                    src_lon_j = lons_flatten[j]

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
                            :, 0, searched_grid_index[0][k], searched_grid_index[1][k]
                        ]
                        for k in range(len(searched_grid_index[0]))
                    ][0]
                    tas_array_3D[
                        src_time_datetime_in_dst_dataset_index,
                        lat_map_index[src_lat_j],
                        lon_map_index[src_lon_j],
                    ] = searched_grid_data_tas

                    # prcp, PREC
                    searched_grid_data_prcp = [
                        src_dataset.variables["APCP"][
                            :, searched_grid_index[0][k], searched_grid_index[1][k]
                        ]
                        for k in range(len(searched_grid_index[0]))
                    ][0]
                    prcp_array_3D[
                        src_time_datetime_in_dst_dataset_index,
                        lat_map_index[src_lat_j],
                        lon_map_index[src_lon_j],
                    ] = searched_grid_data_prcp

                    # pres, PRESSURE
                    searched_grid_data_pres = [
                        src_dataset.variables["PRES"][
                            :, searched_grid_index[0][k], searched_grid_index[1][k]
                        ]
                        for k in range(len(searched_grid_index[0]))
                    ][0]
                    pres_array_3D[
                        src_time_datetime_in_dst_dataset_index,
                        lat_map_index[src_lat_j],
                        lon_map_index[src_lon_j],
                    ] = searched_grid_data_pres

                    # dswrf, SWDOWN
                    searched_grid_data_dswrf = [
                        src_dataset.variables["DSWRF"][
                            :, searched_grid_index[0][k], searched_grid_index[1][k]
                        ]
                        for k in range(len(searched_grid_index[0]))
                    ][0]
                    dswrf_array_3D[
                        src_time_datetime_in_dst_dataset_index,
                        lat_map_index[src_lat_j],
                        lon_map_index[src_lon_j],
                    ] = searched_grid_data_dswrf

                    # dlwrf, LWDOWN
                    searched_grid_data_dlwrf = [
                        src_dataset.variables["DLWRF"][
                            :, searched_grid_index[0][k], searched_grid_index[1][k]
                        ]
                        for k in range(len(searched_grid_index[0]))
                    ][0]
                    dlwrf_array_3D[
                        src_time_datetime_in_dst_dataset_index,
                        lat_map_index[src_lat_j],
                        lon_map_index[src_lon_j],
                    ] = searched_grid_data_dlwrf

                    # vp, VP = (SPFH * PERS) / (0.622 + SPFH)
                    searched_grid_data_SPFH = [
                        src_dataset.variables["SPFH"][
                            :, 0, searched_grid_index[0][k], searched_grid_index[1][k]
                        ]
                        for k in range(len(searched_grid_index[0]))
                    ][0]
                    spfh_array_3D[
                        src_time_datetime_in_dst_dataset_index,
                        lat_map_index[src_lat_j],
                        lon_map_index[src_lon_j],
                    ] = searched_grid_data_SPFH

                    # wind, Wind = (u**2 + v**2) ** (0.5)
                    searched_grid_data_wind_u = [
                        src_dataset.variables["UGRD"][
                            :, 0, searched_grid_index[0][k], searched_grid_index[1][k]
                        ]
                        for k in range(len(searched_grid_index[0]))
                    ][0]
                    wind_u_array_3D[
                        src_time_datetime_in_dst_dataset_index,
                        lat_map_index[src_lat_j],
                        lon_map_index[src_lon_j],
                    ] = searched_grid_data_wind_u

                    searched_grid_data_wind_v = [
                        src_dataset.variables["VGRD"][
                            :, 0, searched_grid_index[0][k], searched_grid_index[1][k]
                        ]
                        for k in range(len(searched_grid_index[0]))
                    ][0]
                    wind_v_array_3D[
                        src_time_datetime_in_dst_dataset_index,
                        lat_map_index[src_lat_j],
                        lon_map_index[src_lon_j],
                    ] = searched_grid_data_wind_v

                    # check
                    if check_search and i == 0:
                        fig, ax = plt.subplots()
                        cgdf = CreateGDF()
                        dst_grids_gdf = cgdf.createGDF_rectangle_central_coord(
                            lons_flatten, lats_flatten, dst_lat_res
                        )
                        src_grids_gdf = cgdf.createGDF_rectangle_central_coord(
                            src_lon_flatten, src_lat_flatten, src_lat_res
                        )
                        # src_grids_gdf = cgdf.createGDF_rectangle_central_coord(searched_grid_lon, searched_grid_lat, src_lat_res)

                        src_grids_gdf.boundary.plot(ax=ax, edgecolor="r", linewidth=2)
                        dst_grids_gdf.plot(
                            ax=ax,
                            edgecolor="k",
                            linewidth=0.2,
                            facecolor="b",
                            alpha=0.5,
                        )
                        ax.set_title("check search")

                ## unit change
                # AIR_TEMP: K->C,  x - 273.15
                tas_array_3D -= 273.15

                # PERS：Pa->Kpa, x / 1000
                pres_array_3D /= 1000

                ## cal other variables
                # cal vp
                vp_array_3D = (spfh_array_3D * pres_array_3D) / (
                    0.622 + spfh_array_3D
                )  # VP = (SPFH * PERS) / (0.622 + SPFH)

                # cal wind
                wind_array_3D = (wind_u_array_3D**2 + wind_v_array_3D**2) ** (0.5)

                ## append into dst_dataset
                dst_dataset.variables["tas"][
                    src_time_datetime_in_dst_dataset_index, :, :
                ] = tas_array_3D
                dst_dataset.variables["prcp"][
                    src_time_datetime_in_dst_dataset_index, :, :
                ] = prcp_array_3D
                dst_dataset.variables["pres"][
                    src_time_datetime_in_dst_dataset_index, :, :
                ] = pres_array_3D
                dst_dataset.variables["dswrf"][
                    src_time_datetime_in_dst_dataset_index, :, :
                ] = dswrf_array_3D
                dst_dataset.variables["dlwrf"][
                    src_time_datetime_in_dst_dataset_index, :, :
                ] = dlwrf_array_3D
                dst_dataset.variables["vp"][
                    src_time_datetime_in_dst_dataset_index, :, :
                ] = vp_array_3D
                dst_dataset.variables["wind"][
                    src_time_datetime_in_dst_dataset_index, :, :
                ] = wind_array_3D

                # assign Global attributes
                dst_dataset.title = "VIC5 image meteForcing dataset"
                dst_dataset.note = "meteForcing dataset generated by XudongZheng, zhengxd@sehemodel.club"
                dst_dataset.Conventions = "CF-1.6"

        logger.info(f"Finished processing for year: {year}")

    logger.info("Formating meteorological forcing files completed successfully")


def resampleTimeForcing(evb_dir, dst_time_hours=24):
    """
    Resample the meteorological forcing data to a different time step.

    Parameters
    ----------
    evb_dir : object
        Directory structure containing the paths to the source and destination data.
    dst_time_hours : int, optional
        The target time step in hours for resampling (default is 24 hours).

    Returns
    -------
    None
        The function saves the resampled data as NetCDF files in the destination directory.
    """
    logger.info(
        f"Starting to resample meteorological forcing files, dst_time_hours: {dst_time_hours}... ..."
    )

    suffix = ".nc"
    MeteForcing_dir = evb_dir.MeteForcing_dir
    resample_dir = os.path.join(MeteForcing_dir, "resample_forcing")
    remove_and_mkdir(resample_dir)
    logger.debug(f"Resample directory created at: {resample_dir}")

    dst_home = resample_dir
    src_home = MeteForcing_dir
    src_names = [n for n in os.listdir(src_home) if n.endswith(suffix)]

    # resample_factor = int(dst_time_hours / src_time_hours)

    ## ====================== loop for forcing resample time ======================
    for i in tqdm(
        range(len(src_names)), desc="loop for forcing resample time", colour="green"
    ):
        # general
        src_name = src_names[i]
        src_path = os.path.join(src_home, src_name)
        dst_path = os.path.join(dst_home, src_name)

        logger.info(f"Processing {src_name} from {src_path}")

        # Open source dataset
        try:
            src_dataset = xr.open_dataset(src_path)
            logger.debug(f"Opened source dataset: {src_path}")
        except Exception as e:
            logger.error(f"Error opening source dataset {src_path}: {e}")
            continue

        # resample data
        try:
            dst_dataset = xr.Dataset(
                {
                    "dlwrf": src_dataset["dlwrf"]
                    .resample(time=f"{dst_time_hours}H")
                    .mean(skipna=True),
                    "dswrf": src_dataset["dswrf"]
                    .resample(time=f"{dst_time_hours}H")
                    .mean(skipna=True),
                    "prcp": src_dataset["prcp"]
                    .resample(time=f"{dst_time_hours}H")
                    .sum(skipna=True),
                    "pres": src_dataset["pres"]
                    .resample(time=f"{dst_time_hours}H")
                    .mean(skipna=True),
                    "tas": src_dataset["tas"]
                    .resample(time=f"{dst_time_hours}H")
                    .mean(skipna=True),
                    "vp": src_dataset["vp"]
                    .resample(time=f"{dst_time_hours}H")
                    .mean(skipna=True),
                    "wind": src_dataset["wind"]
                    .resample(time=f"{dst_time_hours}H")
                    .mean(skipna=True),
                }
            )

            dst_dataset["lats"] = src_dataset["lats"]
            dst_dataset["lons"] = src_dataset["lons"]

            dst_dataset.to_netcdf(dst_path)
            logger.info(f"Resampled data saved to: {dst_path}")

        except Exception as e:
            logger.error(f"Error during resampling or saving {src_name}: {e}")
            continue

        finally:
            # close
            src_dataset.close()
            dst_dataset.close()

    logger.info("Resample meteorological forcing files completed successfully")
