# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
build_Param - A Python module for building VIC parameter file.

This module provides functions for constructing the Parameter file of the VIC model.
It includes capabilities to:
- Build the basic params_dataset_level0
- Build the params_dataset_level0 by g parameters and TF
- Build the params_dataset_level1
- Searching grids for scaling grids from level 0 to level 1
- Scaling params_dataset_level0 to params_dataset_level1

Functions:
----------
    - `buildParam_level0`: Build the parameter dataset for level 0, consisting of two components: `buildParam_level0_basic` and `buildParam_level0_by_g`.
    - `buildParam_level0_basic`: Build the basic parameter dataset for level 0.
    - `buildParam_level0_by_g`: Use global parameter lists and TF to generate the parameter dataset.
    - `buildParam_level1`: Build Level 1 parameters based on TF and dpc information.
    - `scaling_level0_to_level1_search_grids`: Searching grids for scaling grids from level 0 to level 1 (Matching).
    - `scaling_level0_to_level1`: Scaling the grid parameters from level 0 to level 1 based on matching grids.

Usage:
------
    1. Call `buildParam_level0` and provide g_list as well as dpc instances to generate basic params_dataset_level0.
    2. Call `buildParam_level1` to generate params_dataset_level1.
    3. Call `scaling_level0_to_level1_search_grids` to search grids for match grids at level 0 and level 1.
    4. Call `scaling_level0_to_level1` to scale params_dataset_level0 to params_dataset_level1.
    Note: The Transfer function and scaling operator is set in params_func.TransferFunction and params_func.Scaling_operator module.

Example:
--------
    basin_index = 213
    model_scale = "6km"
    date_period = ["19980101", "19981231"]
    case_name = f"{basin_index}_{model_scale}"

    evb_dir = Evb_dir(cases_home="./examples")  # cases_home="/home/xdz/code/VIC_xdz/cases"
    evb_dir.builddir(case_name)

    dpc_VIC_level0, dpc_VIC_level1, dpc_level2 = readdpc(evb_dir)

    domain_dataset = readDomain(evb_dir)

    params_dataset_level0, stand_grids_lat, stand_grids_lon, rows_index, cols_index = buildParam_level0(evb_dir, default_g_list, dpc_VIC_level0, reverse_lat=True)
    params_dataset_level1, stand_grids_lat, stand_grids_lon, rows_index, cols_index = buildParam_level1(evb_dir, dpc_VIC_level1, reverse_lat=True, domain_dataset=domain_dataset)
    params_dataset_level1, searched_grids_bool_index = scaling_level0_to_level1(params_dataset_level0, params_dataset_level1)

    domain_dataset.close()
    params_dataset_level0.close()
    params_dataset_level1.close()

Dependencies:
-------------
    - `numpy`: For numerical computations and array manipulations.
    - `bulid_Domain.cal_mask_frac_area_length`: For calculating mask fraction, area, and length within a domain.
    - `tools.decoractors`: For measuring function execution time.
    - `tools.dpc_func`: For data processing and computation functions.
    - `tools.geo_func`: For geometric calculations and spatial operations.
    - `tools.params_func`: Custom utility functions for parameter handling.
    - `tools.utilities`: Custom utility functions.
    - `tools.decoractors`: For measuring function execution time (duplicate entry, consider consolidating).

"""

import numpy as np
from tqdm import *

from . import logger
from .tools.decoractors import clock_decorator
from .tools.dpc_func.basin_grid_func import *
from .tools.geo_func import search_grids
from .tools.params_func.params_set import *
from .tools.params_func.Scaling_operator import Scaling_operator
from .tools.params_func.TransferFunction import TF_VIC
from .tools.params_func.build_Param_interface import buildParam_level0_interface, buildParam_level1_interface
from .tools.utilities import *


@clock_decorator(print_arg_ret=False)
def buildParam_level0(
    evb_dir,
    g_params,
    soillayerresampler,
    dpc_VIC_level0,
    TF_VIC_class=TF_VIC,
    buildParam_level0_interface_class=buildParam_level0_interface,
    reverse_lat=True,
    stand_grids_lat_level0=None,
    stand_grids_lon_level0=None,
    rows_index_level0=None,
    cols_index_level0=None,
):
    """
    Build the parameter dataset for level 0, consisting of two components: `buildParam_level0_basic` and `buildParam_level0_by_g`.

    Parameters:
    -----------
    evb_dir : `Evb_dir`
        An instance of the `Evb_dir` class, containing paths for VIC deployment.
        
    g_list : list
        A list of global parameters (g-parameters).
        
    dpc_VIC_level0 : `dpc_VIC_level0`
        An instance of the `dpc_VIC_level0` class to process data at level 0 of the VIC model.

    reverse_lat : bool
        Boolean flag to indicate whether to reverse latitudes (Northern Hemisphere: large -> small, set as True).
        
    stand_grids_lat : list, optional
        A list of standard latitudes. If not provided, will be calculated based on the grid shape.
        
    stand_grids_lon : list, optional
        A list of standard longitudes. If not provided, will be calculated based on the grid shape.
        
    rows_index : list, optional
        A list of row indices for the grid. If not provided, will be calculated based on the grid shape.
        
    cols_index : list, optional
        A list of column indices for the grid. If not provided, will be calculated based on the grid shape.

    Returns:
    --------
    params_dataset_level0 : `netCDF.Dataset`
        The parameter dataset for level 0.
        
    stand_grids_lat : list
        A list of standard latitudes.
        
    stand_grids_lon : list
        A list of standard longitudes.
        
    rows_index : list
        A list of row indices for the grid.
        
    cols_index : list
        A list of column indices for the grid.

    Notes:
    ------
    The function generates the parameter dataset for level 0, integrating two sub-components:
    `buildParam_level0_basic` for basic parameter generation and `buildParam_level0_by_g`
    for parameter adjustments based on global parameters.
    """
    # Start of the parameter building process, log an info message
    logger.info("Starting to building params_dataset_level0... ...")

    # initialization
    # if buildParam_level0_interface_class is None:
    #     from .tools.params_func.build_Param_interface import buildParam_level0_interface
    #     buildParam_level0_interface_class = buildParam_level0_interface
    
    buildParam_level0_interface_instance = buildParam_level0_interface_class(
        evb_dir,
        logger,
        dpc_VIC_level0,
        g_params,
        soillayerresampler,
        TF_VIC_class,
        reverse_lat,
        stand_grids_lat_level0,
        stand_grids_lon_level0,
        rows_index_level0,
        cols_index_level0
    )
    
    ## ======================= buildParam_level0_basic =======================
    # Call the buildParam_level0_basic function to generate the base parameters
    logger.info("Calling buildParam_level0_basic... ...")
    buildParam_level0_interface_instance.buildParam_level0_basic()

    ## ======================= buildParam_level0_by_g_tf =======================
    # Call buildParam_level0_by_g_tf to further refine the parameters based on grid list
    logger.info("Calling buildParam_level0_by_g_tf... ...")
    buildParam_level0_interface_instance.buildParam_level0_by_g_tf()
    
    # Log the successful completion of the parameter building
    logger.info(f"Building params_dataset_level0 successfully, params_dataset_level0 file has been saved to {evb_dir.params_dataset_level0_path}")

    # return (
    #     buildParam_level0_interface_instance.params_dataset_level0,
    #     buildParam_level0_interface_instance.stand_grids_lat_level0,
    #     buildParam_level0_interface_instance.stand_grids_lon_level0,
    #     buildParam_level0_interface_instance.rows_index_level0,
    #     buildParam_level0_interface_instance.cols_index_level0,
    # )
    return buildParam_level0_interface_instance


@clock_decorator(print_arg_ret=False)
def buildParam_level1(
    evb_dir,
    dpc_VIC_level1,
    TF_VIC_class=TF_VIC,
    buildParam_level1_interface_class=buildParam_level1_interface,
    reverse_lat=True,
    domain_dataset=None,
    stand_grids_lat_level1=None,
    stand_grids_lon_level1=None,
    rows_index_level1=None,
    cols_index_level1=None,
):
    """
    Build Level 1 parameters.

    Parameters
    ----------
    evb_dir : `Evb_dir`
        An instance of the `Evb_dir` class, containing paths for VIC deployment.
    
    dpc_VIC_level1 : `dpc_VIC_level1`
        An instance of the `dpc_VIC_level1` class to process data at level 0 of the VIC model.
    
    reverse_lat : bool
        Boolean flag to indicate whether to reverse latitudes (Northern Hemisphere: large -> small, set as True).

    domain_dataset : `netCDF.Dataset`, optional
        Domain dataset containing terrain and mask information. If not provided, mask will be computed based on `dpc_VIC_level1`.
    
    stand_grids_lat : list of float, optional
        A list of standard grid latitudes. If not provided, will be calculated based on the grid shape.
        
    stand_grids_lon : list of float, optional
        A list of standard grid longitudes. If not provided, will be calculated based on the grid shape.
        
    rows_index : list of int, optional
        A list of row indices specifying grid positions. If not provided, will be calculated based on the grid shape.
    
    cols_index : list of int, optional
        A list of column indices specifying grid positions. If not provided, will be calculated based on the grid shape.

    Returns
    -------
    params_dataset_level1 : `netCDF.Dataset`
        The parameter dataset for level 1.
        
    stand_grids_lat : list of float
        The list of standard grid latitudes used in the dataset.
        
    stand_grids_lon : list of float
        The list of standard grid longitudes used in the dataset.
        
    rows_index : list of int
        The list of row indices used in the grid.
        
    cols_index : list of int
        The list of column indices used in the grid.
    """
    # Start of the parameter building process, log an info message
    logger.info("Starting to build params_dataset_level1... ...")
    
    # initialization        
    # if buildParam_level1_interface_class is None:
    #     from .tools.params_func.build_Param_interface import buildParam_level1_interface
    #     buildParam_level1_interface_class = buildParam_level1_interface
    
    buildParam_level1_interface_instance = buildParam_level1_interface_class(
        evb_dir,
        logger,
        dpc_VIC_level1,
        TF_VIC_class,
        reverse_lat,
        domain_dataset,
        stand_grids_lat_level1,
        stand_grids_lon_level1,
        rows_index_level1,
        cols_index_level1
    )
    
    ## ======================= buildParam_level1_basic =======================
    # Call the buildParam_level1_basic function to generate the base parameters
    logger.info("Calling buildParam_level1_basic... ...")
    buildParam_level1_interface_instance.buildParam_level1_basic()
    
    ## ======================= buildParam_level1_by_tf =======================
    # Call buildParam_level1_by_tf to further refine the parameters based on tf
    logger.info("Calling buildParam_level0_by_tf... ...")
    buildParam_level1_interface_instance.buildParam_level1_by_tf()

    # Log the successful completion of the parameter building
    logger.info(
        f"Building params_dataset_level1 successfully, params_dataset_level1 file has been saved to {evb_dir.params_dataset_level1_path}"
    )

    # return (
    #     buildParam_level1_interface_instance.params_dataset_level1,
    #     buildParam_level1_interface_instance.stand_grids_lat_level1,
    #     buildParam_level1_interface_instance.stand_grids_lon_level1,
    #     buildParam_level1_interface_instance.rows_index_level1,
    #     buildParam_level1_interface_instance.cols_index_level1,
    # )
    return buildParam_level1_interface_instance


def scaling_level0_to_level1_search_grids(params_dataset_level0, params_dataset_level1):
    """
    Searching grids for scaling grids from level 0 to level 1 (Matching grids at different levels).

    This function reads longitude and latitude values from the parameter datasets of level 0 and level 1,
    calculates the grid resolutions, creates 2D mesh grids for level 1, and searches for the closest
    matching grid indices between level 0 and level 1 with `search_grids.search_grids_radius_rectangle`.
    The function then converts the results into boolean indices for the corresponding grids.

    Parameters
    ----------
    params_dataset_level0 : object
        The parameter dataset for level 0, containing the longitude and latitude values of the original grid.
        
    params_dataset_level1 : object
        The parameter dataset for level 1, containing the longitude and latitude values of the target grid.

    Returns
    -------
    searched_grids_index : array
        The indices of the grids from level 0 that correspond to the grids of level 1.
        
    searched_grids_bool_index : array
        Boolean indices indicating which grids from level 0 match the grids from level 1.
    """
    logger.info(
        "Starting to searching grids for scaling grids from level 0 to level 1... ..."
    )

    # read lon, lat from params, cal res
    logger.debug(
        "Reading longitude and latitude values from level 0 and level 1 datasets... ..."
    )
    lon_list_level0, lat_list_level0 = (
        params_dataset_level0.variables["lon"][:],
        params_dataset_level0.variables["lat"][:],
    )
    lon_list_level1, lat_list_level1 = (
        params_dataset_level1.variables["lon"][:],
        params_dataset_level1.variables["lat"][:],
    )

    # Replace masked values with NaN
    lon_list_level0 = np.ma.filled(lon_list_level0, fill_value=np.NAN)
    lat_list_level0 = np.ma.filled(lat_list_level0, fill_value=np.NAN)
    lon_list_level1 = np.ma.filled(lon_list_level1, fill_value=np.NAN)
    lat_list_level1 = np.ma.filled(lat_list_level1, fill_value=np.NAN)

    # Calculate grid resolution for level 0 and level 1
    res_lon_level0 = (max(lon_list_level0) - min(lon_list_level0)) / (
        len(lon_list_level0) - 1
    )
    res_lat_level0 = (max(lat_list_level0) - min(lat_list_level0)) / (
        len(lat_list_level0) - 1
    )
    res_lon_level1 = (max(lon_list_level1) - min(lon_list_level1)) / (
        len(lon_list_level1) - 1
    )
    res_lat_level1 = (max(lat_list_level1) - min(lat_list_level1)) / (
        len(lat_list_level1) - 1
    )

    logger.debug(f"Resolution for level 0: lon {res_lon_level0}, lat {res_lat_level0}")
    logger.debug(f"Resolution for level 1: lon {res_lon_level1}, lat {res_lat_level1}")

    # Create 2D meshgrid for level 1 and flatten
    logger.debug("Creating 2D meshgrid for level 1... ...")
    lon_list_level1_2D, lat_list_level1_2D = np.meshgrid(
        lon_list_level1, lat_list_level1
    )
    lon_list_level1_2D_flatten = lon_list_level1_2D.flatten()
    lat_list_level1_2D_flatten = lat_list_level1_2D.flatten()

    # Search for corresponding grids between level 0 and level 1
    logger.debug("Searching for matching grids from level 0 to level 1... ...")
    searched_grids_index = search_grids.search_grids_radius_rectangle(
        dst_lat=lat_list_level1_2D_flatten,
        dst_lon=lon_list_level1_2D_flatten,
        src_lat=lat_list_level0,
        src_lon=lon_list_level0,
        lat_radius=res_lat_level1/2,
        lon_radius=res_lon_level1/2,
    )

    # Convert search results into boolean indices
    logger.debug("Converting search results into boolean indices... ...")
    searched_grids_bool_index = searched_grids_index_to_bool_index(
        searched_grids_index, lat_list_level0, lon_list_level0
    )

    logger.info(
        "Searching grids for scaling grids from level 0 to level 1 successfully"
    )
    return searched_grids_index, searched_grids_bool_index


@clock_decorator(print_arg_ret=False)
def scaling_level0_to_level1(
    params_dataset_level0, params_dataset_level1, searched_grids_bool_index=None,
    nlayer_list=[1, 2, 3],
):
    """
    Scaling the parameters from level 0 to level 1 based on matching grids.

    This function takes the parameters from the level 0 and level 1 datasets, and scales the grid
    parameters from the level 0 resolution to the level 1 resolution. It searches for the matching
    grids between the two levels and then returns the level 1 dataset with the corresponding data
    mapped from level 0, along with a boolean index indicating which grids in level 0 correspond
    to the grids in level 1. The scaling operators are applied.

    Parameters
    ----------
    params_dataset_level0 : `netCDF.Dataset`
        The parameter dataset for level 0.
        
    params_dataset_level1 : `netCDF.Dataset`
        The parameter dataset for level 1.
        
    searched_grids_bool_index : array-like, optional, default=None
        Boolean indices indicating which grids from level 0 match the grids from level 1.
        If not provided, it is calculated within the function.

    Returns
    -------
    params_dataset_level1 : `netCDF.Dataset`
        The parameter dataset for level 1, with values from level 0 mapped onto the grids of level 1.
        
    searched_grids_bool_index : array
        Boolean indices indicating which grids from level 0 correspond to grids from level 1.

    Notes
    ------
    - This function performs a search for the closest grids between level 0 and level 1.
    - The mapping process takes into account the resolution of both grids and the spatial alignment.
    """

    logger.info(
        "Starting to scaling params_dataset_level0 to params_dataset_level1... ..."
    )

    # Retrieve grid shape information
    lon_list_level1, lat_list_level1 = (
        params_dataset_level1.variables["lon"][:],
        params_dataset_level1.variables["lat"][:],
    )
    lon_list_level1 = np.ma.filled(lon_list_level1, fill_value=np.NAN)
    lat_list_level1 = np.ma.filled(lat_list_level1, fill_value=np.NAN)

    # search grids
    if searched_grids_bool_index is None:
        searched_grids_index, searched_grids_bool_index = (
            scaling_level0_to_level1_search_grids(
                params_dataset_level0, params_dataset_level1
            )
        )

    # ======================= scaling (resample) =======================
    logger.info("Scaling based on Scaling_operator... ...")
    scaling_operator = Scaling_operator()

    # resample func
    search_and_resample_func_2d = lambda scaling_func, varibale_name: np.array(
        [
            scaling_func(
                params_dataset_level0.variables[varibale_name][
                    searched_grid_bool_index[0], searched_grid_bool_index[1]
                ].flatten()
            )
            for searched_grid_bool_index in searched_grids_bool_index
        ]
    ).reshape((len(lat_list_level1), len(lon_list_level1)))
    
    search_and_resample_func_3d = (
        lambda scaling_func, varibale_name, first_dim: np.array(
            [
                scaling_func(
                    params_dataset_level0.variables[varibale_name][
                        first_dim,
                        searched_grid_bool_index[0],
                        searched_grid_bool_index[1],
                    ].flatten()
                )
                for searched_grid_bool_index in searched_grids_bool_index
            ]
        ).reshape((len(lat_list_level1), len(lon_list_level1)))
    )

    # depth, m
    for i in range(len(nlayer_list)):
        params_dataset_level1.variables["depth"][i, :, :] = search_and_resample_func_3d(
            scaling_operator.Arithmetic_mean, "depth", i
        )
    logger.debug("Scaling depth parameter completed")

    # b_infilt, /NA
    params_dataset_level1.variables["infilt"][:, :] = search_and_resample_func_2d(
        scaling_operator.Arithmetic_mean, "infilt"
    )
    logger.debug("Scaling infilt parameter completed")

    # ksat, mm/s -> mm/day (VIC requirement)
    for i in range(len(nlayer_list)):
        params_dataset_level1.variables["Ksat"][i, :, :] = search_and_resample_func_3d(
            scaling_operator.Harmonic_mean, "Ksat", i
        )
    logger.debug("Scaling Ksat parameter completed")

    # phi_s, m3/m3 or mm/mm
    for i in range(len(nlayer_list)):
        params_dataset_level1.variables["phi_s"][i, :, :] = search_and_resample_func_3d(
            scaling_operator.Arithmetic_mean, "phi_s", i
        )
    logger.debug("Scaling phi_s parameter completed")

    # psis, kPa/cm-H2O
    for i in range(len(nlayer_list)):
        params_dataset_level1.variables["psis"][i, :, :] = search_and_resample_func_3d(
            scaling_operator.Arithmetic_mean, "psis", i
        )
    logger.debug("Scaling psis parameter completed")

    # b_retcurve, /NA
    for i in range(len(nlayer_list)):
        params_dataset_level1.variables["b_retcurve"][i, :, :] = (
            search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "b_retcurve", i)
        )
    logger.debug("Scaling b_retcurve parameter completed")

    # expt, /NA
    for i in range(len(nlayer_list)):
        params_dataset_level1.variables["expt"][i, :, :] = search_and_resample_func_3d(
            scaling_operator.Arithmetic_mean, "expt", i
        )
    logger.debug("Scaling expt parameter completed")

    # fc, % or m3/m3
    for i in range(len(nlayer_list)):
        params_dataset_level1.variables["fc"][i, :, :] = search_and_resample_func_3d(
            scaling_operator.Arithmetic_mean, "fc", i
        )
    logger.debug("Scaling fc parameter completed")

    # D4, /NA, same as c, typically is 2
    params_dataset_level1.variables["D4"][:, :] = search_and_resample_func_2d(
        scaling_operator.Arithmetic_mean, "D4"
    )
    logger.debug("Scaling D4 parameter completed")

    # cexpt
    params_dataset_level1.variables["c"][:, :] = search_and_resample_func_2d(
        scaling_operator.Arithmetic_mean, "c"
    )
    logger.debug("Scaling c parameter completed")

    # D1 ([day^-1]), D2 ([day^-D4])
    params_dataset_level1.variables["D1"][:, :] = search_and_resample_func_2d(
        scaling_operator.Harmonic_mean, "D1"
    )
    params_dataset_level1.variables["D2"][:, :] = search_and_resample_func_2d(
        scaling_operator.Harmonic_mean, "D2"
    )
    logger.debug("Scaling D1/2 parameter completed")

    # D3 ([mm])
    params_dataset_level1.variables["D3"][:, :] = search_and_resample_func_2d(
        scaling_operator.Arithmetic_mean, "D3"
    )
    logger.debug("Scaling D3 parameter completed")

    # Dsmax, mm or mm/day
    params_dataset_level1.variables["Dsmax"][:, :] = search_and_resample_func_2d(
        scaling_operator.Harmonic_mean, "Dsmax"
    )
    logger.debug("Scaling Dsmax parameter completed")

    # Ds, [day^-D4] or fraction
    params_dataset_level1.variables["Ds"][:, :] = search_and_resample_func_2d(
        scaling_operator.Harmonic_mean, "Ds"
    )
    logger.debug("Scaling Ds parameter completed")

    # Ws, fraction
    params_dataset_level1.variables["Ws"][:, :] = search_and_resample_func_2d(
        scaling_operator.Arithmetic_mean, "Ws"
    )
    logger.debug("Scaling Ws parameter completed")

    # init_moist, mm
    for i in range(len(nlayer_list)):
        params_dataset_level1.variables["init_moist"][i, :, :] = (
            search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "init_moist", i)
        )
    logger.debug("Scaling init_moist parameter completed")

    # elev, m
    params_dataset_level1.variables["elev"][:, :] = search_and_resample_func_2d(
        scaling_operator.Arithmetic_mean, "elev"
    )
    logger.debug("Scaling elev parameter completed")

    # dp, m, typically is 4m
    params_dataset_level1.variables["dp"][:, :] = search_and_resample_func_2d(
        scaling_operator.Arithmetic_mean, "dp"
    )
    logger.debug("Scaling dp parameter completed")

    # bubble, cm
    for i in range(len(nlayer_list)):
        params_dataset_level1.variables["bubble"][i, :, :] = search_and_resample_func_3d(
            scaling_operator.Arithmetic_mean, "bubble", i
        )
    logger.debug("Scaling bubble parameter completed")

    # quartz, N/A
    for i in range(len(nlayer_list)):
        params_dataset_level1.variables["quartz"][i, :, :] = search_and_resample_func_3d(
            scaling_operator.Arithmetic_mean, "quartz", i
        )
    logger.debug("Scaling quartz parameter completed")

    # bulk_density, kg/m3 or mm
    for i in range(len(nlayer_list)):
        params_dataset_level1.variables["bulk_density"][i, :, :] = (
            search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "bulk_density", i)
        )
    logger.debug("Scaling bulk_density parameter completed")

    # soil_density, kg/m3
    for i in range(len(nlayer_list)):
        params_dataset_level1.variables["soil_density"][i, :, :] = (
            search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "soil_density", i)
        )
    logger.debug("Scaling soil_density parameter completed")

    # Wcr_FRACT, fraction
    for i in range(len(nlayer_list)):
        params_dataset_level1.variables["Wcr_FRACT"][i, :, :] = search_and_resample_func_3d(
            scaling_operator.Arithmetic_mean, "Wcr_FRACT", i
        )
    logger.debug("Scaling Wcr_FRACT parameter completed")

    # wp, computed field capacity [frac]
    for i in range(len(nlayer_list)):
        params_dataset_level1.variables["wp"][i, :, :] = search_and_resample_func_3d(
            scaling_operator.Arithmetic_mean, "wp", i
        )
    logger.debug("Scaling wp parameter completed")

    # Wpwp_FRACT, fraction
    for i in range(len(nlayer_list)):
        params_dataset_level1.variables["Wpwp_FRACT"][i, :, :] = (
            search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "Wpwp_FRACT", i)
        )
    logger.debug("Scaling Wpwp_FRACT parameter completed")

    # rough, m, Surface roughness of bare soil
    params_dataset_level1.variables["rough"][:, :] = search_and_resample_func_2d(
        scaling_operator.Arithmetic_mean, "rough"
    )
    logger.debug("Scaling rough parameter completed")

    # snow rough, m
    params_dataset_level1.variables["snow_rough"][:, :] = search_and_resample_func_2d(
        scaling_operator.Arithmetic_mean, "snow_rough"
    )
    logger.debug("Scaling snow_rough parameter completed")

    logger.info(
        "Scaling params_dataset_level0 to params_dataset_level1 successfully"
    )

    return params_dataset_level1, searched_grids_bool_index
