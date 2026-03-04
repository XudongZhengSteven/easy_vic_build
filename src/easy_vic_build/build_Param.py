# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Build VIC parameter datasets and scale level-0 parameters to level-1.

Public functions
----------------
``buildParam_level0``
    Build and update level-0 parameter dataset through interface methods.
``buildParam_level1``
    Build and update level-1 parameter dataset through interface methods.
``scaling_level0_to_level1_search_grids``
    Create level-0 to level-1 grid mapping.
``scaling_level0_to_level1``
    Resample level-0 parameters onto level-1 grids.
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
    basin_hierarchy=None,
):
    """
    Build level-0 parameter dataset using the configured interface class.

    Parameters
    ----------
    evb_dir : Evb_dir
        Case directory manager.
    g_params : dict
        Parameter vector/group used by transfer functions at level-0.
    soillayerresampler : object
        Soil-layer resampler used by the level-0 interface.
    dpc_VIC_level0 : object
        DPC instance for level-0 processing.
    TF_VIC_class : type, optional
        Transfer-function class used by the interface.
    buildParam_level0_interface_class : type, optional
        Interface class implementing ``buildParam_level0_basic`` and
        ``buildParam_level0_by_g_tf``.
    reverse_lat : bool, optional
        Whether latitude axis is arranged north-to-south.
    stand_grids_lat_level0, stand_grids_lon_level0, rows_index_level0, cols_index_level0 : optional
        Precomputed grid metadata to reuse for faster repeated runs.
    basin_hierarchy : optional
        Optional basin hierarchy information passed to interface constructor.

    Returns
    -------
    object
        The created level-0 interface instance.
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
        cols_index_level0,
        basin_hierarchy
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
    Build level-1 parameter dataset using the configured interface class.

    Parameters
    ----------
    evb_dir : Evb_dir
        Case directory manager.
    dpc_VIC_level1 : object
        DPC instance for level-1 processing.
    TF_VIC_class : type, optional
        Transfer-function class used by the interface.
    buildParam_level1_interface_class : type, optional
        Interface class implementing ``buildParam_level1_basic`` and
        ``buildParam_level1_by_tf``.
    reverse_lat : bool, optional
        Whether latitude axis is arranged north-to-south.
    domain_dataset : netCDF4.Dataset, optional
        Domain dataset used by level-1 interface.
    stand_grids_lat_level1, stand_grids_lon_level1, rows_index_level1, cols_index_level1 : optional
        Precomputed grid metadata to reuse for faster repeated runs.

    Returns
    -------
    object
        The created level-1 interface instance.
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
    logger.info("Calling buildParam_level1_by_tf... ...")
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
    Build level-0 to level-1 grid mapping by rectangular neighborhood search.

    Parameters
    ----------
    params_dataset_level0 : netCDF4.Dataset
        Source level-0 parameter dataset.
    params_dataset_level1 : netCDF4.Dataset
        Target level-1 parameter dataset.

    Returns
    -------
    tuple
        ``(searched_grids_index, searched_grids_bool_index)``.
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
    lon_list_level0 = np.ma.filled(lon_list_level0, fill_value=np.nan)
    lat_list_level0 = np.ma.filled(lat_list_level0, fill_value=np.nan)
    lon_list_level1 = np.ma.filled(lon_list_level1, fill_value=np.nan)
    lat_list_level1 = np.ma.filled(lat_list_level1, fill_value=np.nan)

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
    nlayer_list=[1, 2, 3], elev_scaling=None,
):
    """
    Scaling level-0 parameters onto level-1 grid cells.

    Parameters
    ----------
    params_dataset_level0 : netCDF4.Dataset
        Source level-0 dataset.
    params_dataset_level1 : netCDF4.Dataset
        Target level-1 dataset to be updated in place.
    searched_grids_bool_index : array-like, optional
        Precomputed level-0 search masks for each level-1 cell.
    nlayer_list : list, optional
        Soil-layer indices used for 3D variables.
    elev_scaling : str, optional
        If set to ``"Arithmetic_min"``, elevation uses min aggregation;
        otherwise arithmetic mean is used.
    
    Returns
    -------
    tuple
        ``(params_dataset_level1, searched_grids_bool_index)``.
    """

    logger.info(
        "Starting to scaling params_dataset_level0 to params_dataset_level1... ..."
    )

    # Retrieve grid shape information
    lon_list_level1, lat_list_level1 = (
        params_dataset_level1.variables["lon"][:],
        params_dataset_level1.variables["lat"][:],
    )
    lon_list_level1 = np.ma.filled(lon_list_level1, fill_value=np.nan)
    lat_list_level1 = np.ma.filled(lat_list_level1, fill_value=np.nan)

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
            scaling_operator.Majority, "depth", i
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

    # d4, /NA, same as c, typically is 2
    params_dataset_level1.variables["d4"][:, :] = search_and_resample_func_2d(
        scaling_operator.Arithmetic_mean, "d4"
    )
    logger.debug("Scaling d4 parameter completed")

    # cexpt
    params_dataset_level1.variables["c"][:, :] = search_and_resample_func_2d(
        scaling_operator.Arithmetic_mean, "c"
    )
    logger.debug("Scaling c parameter completed")

    # d1 ([day^-1]), d2 ([day^-d4])
    params_dataset_level1.variables["d1"][:, :] = search_and_resample_func_2d(
        scaling_operator.Harmonic_mean, "d1"
    )
    params_dataset_level1.variables["d2"][:, :] = search_and_resample_func_2d(
        scaling_operator.Harmonic_mean, "d2"
    )
    logger.debug("Scaling d1/2 parameter completed")

    # d3 ([mm])
    params_dataset_level1.variables["d3"][:, :] = search_and_resample_func_2d(
        scaling_operator.Arithmetic_mean, "d3"
    )
    logger.debug("Scaling d3 parameter completed")

    # Dsmax, mm or mm/day
    params_dataset_level1.variables["Dsmax"][:, :] = search_and_resample_func_2d(
        scaling_operator.Harmonic_mean, "Dsmax"
    )
    logger.debug("Scaling Dsmax parameter completed")

    # Ds, [day^-d4] or fraction
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

    # elev, m]
    if elev_scaling is not None:
        if elev_scaling == "Arithmetic_min":
            params_dataset_level1.variables["elev"][:, :] = search_and_resample_func_2d(
                scaling_operator.Arithmetic_min, "elev"
            )
        else:
            raise ValueError(f"elev_scaling {elev_scaling} not recognized.")
    else:
        params_dataset_level1.variables["elev"][:, :] = search_and_resample_func_2d(
            scaling_operator.Arithmetic_mean, "elev"
        )
    logger.debug("Scaling elev parameter completed")
    
    # slope, m/m
    params_dataset_level1.variables["slope"][:, :] = search_and_resample_func_2d(
        scaling_operator.Arithmetic_mean, "slope"
    )
    logger.debug("Scaling slope parameter completed")

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
