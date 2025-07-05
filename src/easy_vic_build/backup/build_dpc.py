# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
build_dpc - A Python module for building dpc (data processing class) classes.

This module provides functionality for building and visualizing the Data Process Class (DPC)
of the VIC model at three levels (level0, level1, and level2). It includes methods for data
processing at each level, generating corresponding plots, and saving the processed data to disk.

Functions:
----------
    - `builddpc`: Main function for building and visualizing the dpc at different levels.

Usage:
------
    1. Set and instantiate your own data processing classes for each level (`dpc_VIC_level0`,
    `dpc_VIC_level1`, and `dpc_VIC_level2`).
    2. Call the `builddpc` function with appropriate arguments to process and plot the data.

Example:
--------
    basin_index = 213
    model_scale = "6km"
    date_period = ["19980101", "19981231"]
    case_name = f"{basin_index}_{model_scale}"
    grid_res_level0=0.00833
    grid_res_level1=scalemap[model_scale]
    grid_res_level2=0.125

    evb_dir = Evb_dir(cases_home="./examples")
    evb_dir.builddir(case_name)

    basin_shp_all, basin_shp = read_one_HCDN_basin_shp(basin_index)

    grid_shp_lon_level1, grid_shp_lat_level1, grid_shp_level1 = createGridForBasin(basin_shp, grid_res_level1, expand_grids_num=1)
    _, _, _, boundary_grids_edge_x_y_level1 = grid_shp_level1.createBoundaryShp()

    grid_shp_lon_level0, grid_shp_lat_level0, grid_shp_level0 = createGridForBasin(basin_shp, grid_res_level0, boundary=boundary_grids_edge_x_y_level1)
    grid_shp_lon_level2, grid_shp_lat_level2, grid_shp_level2 = createGridForBasin(basin_shp, grid_res_level2, boundary=boundary_grids_edge_x_y_level1)

    dpc_VIC_level0 = dataProcess_VIC_level0(basin_shp, grid_shp_level0, grid_res_level0, date_period)
    dpc_VIC_level1 = dataProcess_VIC_level1(basin_shp, grid_shp_level1, grid_res_level1, date_period)
    dpc_VIC_level2 = dataProcess_VIC_level2(basin_shp, grid_shp_level2, grid_res_level2, date_period)

    dpc_VIC_level0_call_kwargs={"readBasindata": False, "readGriddata": True, "readBasinAttribute": False}
    dpc_VIC_level1_call_kwargs={"readBasindata": True, "readGriddata": True, "readBasinAttribute": True}
    dpc_VIC_level2_call_kwargs={"readBasindata": False, "readGriddata": False, "readBasinAttribute": False}
    plot_columns_level0 = ["SrtmDEM_mean_Value", "soil_l1_sand_nearest_Value"]
    plot_columns_level1 = ["annual_P_in_src_grid_Value", "umd_lc_major_Value"]

    builddpc(evb_dir, dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2,
             dpc_VIC_level0_call_kwargs, dpc_VIC_level1_call_kwargs, dpc_VIC_level2_call_kwargs,
             plot_columns_level0, plot_columns_level1)

Dependencies:
-------------
    - `matplotlib`: For plotting the DPCs.
    - `pickle`: For serializing the DPC data.
    - `tools.utilities`: Custom utility functions.
    - `tools.decoractors`: For measuring function execution time.

"""

import pickle

import matplotlib.pyplot as plt

from . import logger
from .tools.decoractors import clock_decorator
from .tools.utilities import *


@clock_decorator(print_arg_ret=False)
def builddpc(
    evb_dir,
    dpc_VIC_level0,
    dpc_VIC_level1,
    dpc_VIC_level2,
    dpc_VIC_level0_call_kwargs=dict(),
    dpc_VIC_level1_call_kwargs=dict(),
    dpc_VIC_level2_call_kwargs=dict(),
    plot_columns_level0=["SrtmDEM_mean_Value", "soil_l1_sand_nearest_Value"],
    plot_columns_level1=["annual_P_in_src_grid_Value", "umd_lc_major_Value"],
):
    """
    Build and visualize the VIC model data process classes (dpc) at three levels (level0/1/2).

    Parameters:
    -----------
    evb_dir : `Evb_dir`
        An instance of the `Evb_dir` class, containing paths for VIC deployment.

    dpc_VIC_level0 : `dpc_VIC_level0`
        An instance of the `dpc_VIC_level0` class to process data at level 0 of the VIC model.

    dpc_VIC_level1 : `dpc_VIC_level1`
        An instance of the `dpc_VIC_level1` class to process data at level 1 of the VIC model.

    dpc_VIC_level2 : `dpc_VIC_level2`
        An instance of the `dpc_VIC_level2` class to process data at level 2 of the VIC model.

    dpc_VIC_level0_call_kwargs : dict, optional
        Keyword arguments to pass to the level 0 processing function.

    dpc_VIC_level1_call_kwargs : dict, optional
        Keyword arguments to pass to the level 1 processing function.

    dpc_VIC_level2_call_kwargs : dict, optional
        Keyword arguments to pass to the level 2 processing function.

    plot_columns_level0 : list of str, optional
        Columns to plot for level 0 data. Default is ["SrtmDEM_mean_Value", "soil_l1_sand_nearest_Value"].

    plot_columns_level1 : list of str, optional
        Columns to plot for level 1 data. Default is ["annual_P_in_src_grid_Value", "umd_lc_major_Value"].

    Returns:
    --------
    None
        The function generates and saves plots and serialized data for the dpc at three levels. Three dpcs
        are stored.

    Notes:
    ------
    - The function processes data at three levels (0, 1, 2) using user-supplied processing classes.
    - Plots are generated for each level and for selected columns.
    - Serialized data is saved to specified paths within the `evb_dir` object.
    - The `clock_decorator` is used to measure execution time of the function.
    """
    logger.info("Starting to build dpc... ...")

    # ====================== build dpc level0 ======================
    logger.info("Building dpc_level0... ...")
    try:
        dpc_VIC_level0(**dpc_VIC_level0_call_kwargs)
        logger.info("dpc_level0 is built")
    except Exception as e:
        logger.error(f"Error while building dpc_level0: {e}")

    # ====================== build dpc level1 ======================
    logger.info("Building dpc_level1... ...")
    try:
        dpc_VIC_level1(**dpc_VIC_level1_call_kwargs)
        logger.info("dpc_level1 is built")
    except Exception as e:
        logger.error(f"Error while building dpc_level1: {e}")

    # ====================== build dpc level2 ======================
    logger.info("Building dpc_level2... ...")
    try:
        dpc_VIC_level2(**dpc_VIC_level2_call_kwargs)  # Read data, not read
        logger.info("dpc_level2 is built")
    except Exception as e:
        logger.error(f"Error while building dpc_level2: {e}")

    # ====================== plot ======================
    logger.info("Plotting dpc... ...")
    try:
        fig_grid_basin, axes_grid_basin = plt.subplots(
            1, 3, figsize=(12, 4), gridspec_kw={"wspace": 0.4}
        )
        dpc_VIC_level0.plot(fig_grid_basin, axes_grid_basin[0])
        dpc_VIC_level1.plot(fig_grid_basin, axes_grid_basin[1])
        dpc_VIC_level2.plot(fig_grid_basin, axes_grid_basin[2])

        axes_grid_basin[0].set_title("dpc level0")
        axes_grid_basin[1].set_title("dpc level1")
        axes_grid_basin[2].set_title("dpc level2")

        if plot_columns_level0 is not None:
            fig_columns, axes_columns = plt.subplots(2, 2, figsize=(12, 8))
            dpc_VIC_level0.plot_grid_column(
                column=plot_columns_level0[0], fig=fig_columns, ax=axes_columns[0, 0]
            )
            dpc_VIC_level0.plot_grid_column(
                column=plot_columns_level0[1], fig=fig_columns, ax=axes_columns[0, 1]
            )
            dpc_VIC_level1.plot_grid_column(
                column=plot_columns_level1[0], fig=fig_columns, ax=axes_columns[1, 0]
            )
            dpc_VIC_level1.plot_grid_column(
                column=plot_columns_level1[1], fig=fig_columns, ax=axes_columns[1, 1]
            )

            axes_columns[0, 0].set_title(plot_columns_level0[0])
            axes_columns[0, 1].set_title(plot_columns_level0[1])
            axes_columns[1, 0].set_title(plot_columns_level1[0])
            axes_columns[1, 1].set_title(plot_columns_level1[1])
        logger.info("Plotting completed successfully")

    except Exception as e:
        logger.error(f"Error while plotting: {e}")

    # ====================== save ======================
    logger.info("Saving dpc... ...")
    try:
        with open(evb_dir.dpc_VIC_level0_path, "wb") as f:
            pickle.dump(dpc_VIC_level0, f)
        logger.info(f"Save dpc_level0 to {evb_dir.dpc_VIC_level0_path}")

        with open(evb_dir.dpc_VIC_level1_path, "wb") as f:
            pickle.dump(dpc_VIC_level1, f)
        logger.info(f"Save dpc_level1 to {evb_dir.dpc_VIC_level1_path}")

        with open(evb_dir.dpc_VIC_level2_path, "wb") as f:
            pickle.dump(dpc_VIC_level2, f)
        logger.info(f"Save dpc_level2 to {evb_dir.dpc_VIC_level2_path}")

        fig_grid_basin.savefig(evb_dir.dpc_VIC_plot_grid_basin_path)
        logger.info(f"Save plot to {evb_dir.dpc_VIC_plot_grid_basin_path}")

        if plot_columns_level0 is not None:
            fig_columns.savefig(evb_dir.dpc_VIC_plot_columns_path)
            logger.info(f"Save columns plot to {evb_dir.dpc_VIC_plot_columns_path}")

        logger.info("Save dpc successfully")
        
    except Exception as e:
        logger.error(f"Error while saving data: {e}")

    logger.info("Building dpc successfully")
