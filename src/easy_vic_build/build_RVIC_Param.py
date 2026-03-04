# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Build RVIC routing inputs and configuration files.

Public functions
----------------
``buildRVICParam_basic``
    Generate flow-direction, pour-point, UHBOX, and RVIC parameter CFG files.
``buildRVICParam``
    Run full RVIC parameter generation (requires ``rvic`` package).
``buildRVICFlowDirectionFile``
    Build ``flow_direction_file.nc`` from hydroanalysis outputs and domain data.
``buildPourPointFile`` / ``buildUHBOXFile`` / ``buildParamCFGFile`` / ``buildConvCFGFile``
    Create individual RVIC input/configuration files.
"""

import os
from copy import deepcopy

import numpy as np
import pandas as pd
import rasterio
import rasterio.transform
from netCDF4 import Dataset

from . import logger
from .tools.decoractors import clock_decorator
from .tools.geo_func.search_grids import *
from .tools.params_func.createParametersDataset import createFlowDirectionFile
from .tools.params_func.TransferFunction import TF_VIC
from .tools.routing_func import create_uh
from .tools.utilities import (read_cfg_to_dict,
                              read_rvic_conv_cfg_file_reference,
                              read_rvic_param_cfg_file_reference)

try:
    from rvic.parameters import parameters as rvic_parameters

    HAS_RVIC = True
except:
    HAS_RVIC = False


def buildRVICParam_basic(
    evb_dir,
    domain_dataset,
    ppf_kwargs=dict(),
    uh_params={
        "createUH_func": create_uh.createGUH,
        "uh_dt": 3600,
        "tp": 1.4,
        "mu": 5.0,
        "m": 3.0,
        "plot_bool": True,
        "max_day": None,
        "max_day_range": (0, 10),
        "max_day_converged_threshold": 0.001,
    },
    cfg_params={
        "VELOCITY": 1.5,  # or variable name: velocity
        "DIFFUSION": 800.0,  # or variable name: diffusion
        "OUTPUT_INTERVAL": 86400,
        "SUBSET_DAYS": 10,
        "CELL_FLOWDAYS": 2,
        "BASIN_FLOWDAYS": 50,
    },
    fd_params={
        "g_velocity": None,
        "g_diffusion": None,
        "slope": None,
        "TF_VIC_class": TF_VIC
    }
):
    """
    Build core RVIC input files without running ``rvic.parameters``.

    Parameters
    ----------
    evb_dir : Evb_dir
        Case directory manager.
    domain_dataset : netCDF4.Dataset
        Domain dataset used to write routing fields.
    ppf_kwargs : dict, optional
        Keyword arguments forwarded to :func:`buildPourPointFile`.
    uh_params : dict, optional
        Keyword arguments forwarded to :func:`buildUHBOXFile`.
    cfg_params : dict, optional
        Keyword arguments forwarded to :func:`buildParamCFGFile`.
    fd_params : dict, optional
        Keyword arguments forwarded to :func:`buildRVICFlowDirectionFile`.

    Returns
    -------
    None
        Files are written in ``evb_dir.RVICParam_dir``.
    """
    logger.info(
        "Starting to generate RVIC parameter file without using rvic_parameters... ..."
    )

    # general RVICParam before using rvic_parameters
    # buildRVICFlowDirectionFile
    buildRVICFlowDirectionFile(evb_dir, domain_dataset, **fd_params)

    # buildPourPointFile
    buildPourPointFile(evb_dir, **ppf_kwargs)

    # buildUHBOXFile
    max_day = buildUHBOXFile(evb_dir, **uh_params)

    # buildParamCFGFile
    if cfg_params["CELL_FLOWDAYS"] is None:
        cfg_params["CELL_FLOWDAYS"] = max_day
        
    buildParamCFGFile(evb_dir, **cfg_params)

    logger.info(
        "RVIC parameter file generation without using rvic_parameters successfully"
    )


@clock_decorator(print_arg_ret=False)
def buildRVICParam(
    evb_dir,
    domain_dataset,
    ppf_kwargs=dict(),
    uh_params={
        "createUH_func": create_uh.createGUH,
        "uh_dt": 3600,
        "tp": 1.4,
        "mu": 5.0,
        "m": 3.0,
        "plot_bool": True,
        "max_day": None,
        "max_day_range": (0, 10),
        "max_day_converged_threshold": 0.001,
    },
    cfg_params={
        "VELOCITY": 1.5,
        "DIFFUSION": 800.0,
        "OUTPUT_INTERVAL": 86400,
        "SUBSET_DAYS": 10,
        "CELL_FLOWDAYS": 2,
        "BASIN_FLOWDAYS": 50,
    },
    fd_params={
        "g_velocity": None,
        "g_diffusion": None,
        "slope": None,
        "TF_VIC_class": TF_VIC
    },
    numofproc=1,
):
    """
    Build RVIC input files and run ``rvic.parameters``.

    Parameters
    ----------
    evb_dir : Evb_dir
        Case directory manager.
    domain_dataset : netCDF4.Dataset
        Domain dataset used to build routing inputs.
    ppf_kwargs : dict, optional
        Keyword arguments forwarded to :func:`buildPourPointFile`.
    uh_params : dict, optional
        Keyword arguments forwarded to :func:`buildUHBOXFile`.
    cfg_params : dict, optional
        Keyword arguments forwarded to :func:`buildParamCFGFile`.
    fd_params : dict, optional
        Keyword arguments forwarded to :func:`buildRVICFlowDirectionFile`.
    numofproc : int, optional
        Number of processes passed to ``rvic.parameters``.

    Returns
    -------
    None
        RVIC parameter files are generated and ``rvic.parameters`` is executed.

    Raises
    ------
    ImportError
        If the RVIC module is not available.

    Notes
    -----
    If the ``rvic`` package is unavailable, this function raises ``ImportError``.
    """
    logger.info("Starting to generate RVIC parameter file... ...")

    # buildRVICParam_general
    buildRVICParam_basic(
        evb_dir,
        domain_dataset,
        ppf_kwargs,
        uh_params,
        cfg_params,
        fd_params,
    )

    # build rvic parameters
    logger.debug(
        f"Reading RVIC parameter configuration from {evb_dir.rvic_param_cfg_file_path}... ..."
    )
    param_cfg_file_dict = read_cfg_to_dict(evb_dir.rvic_param_cfg_file_path)

    if HAS_RVIC:
        logger.info("Executing RVIC parameter computation... ...")
        rvic_parameters(param_cfg_file_dict, numofproc)
        logger.info("RVIC parameter computation completed")
    else:
        logger.error("RVIC module is not available. Cannot proceed with buildRVICParam")
        raise ImportError("no rvic for buildRVICParam")

    logger.info("RVIC parameter file generation successfully")


def buildRVICFlowDirectionFile(evb_dir, domain_dataset, g_velocity=None, g_diffusion=None, slope=None, TF_VIC_class=TF_VIC):
    """
    Build ``flow_direction_file.nc`` from hydroanalysis rasters and domain data.

    Parameters
    ----------
    evb_dir : Evb_dir
        Case directory manager.
    domain_dataset : netCDF4.Dataset
        Domain dataset that provides ``lat``, ``lon``, ``mask``, and ``area``.
    g_velocity : tuple or list, optional
        Parameters for spatial velocity estimation via ``TF_VIC_class.velocity``.
    g_diffusion : tuple or list, optional
        Parameters for spatial diffusion estimation via ``TF_VIC_class.diffusion``.
    slope : array-like, optional
        Slope field used when estimating velocity.
    TF_VIC_class : type, optional
        Transfer-function class for optional velocity/diffusion fields.

    Returns
    -------
    None
        The file is written to ``evb_dir.flow_direction_file_path``.
    """
    logger.info("Starting to generate RVIC flow direction file... ...")
    # ====================== set dir and path ======================
    # set path
    flow_direction_file_path = os.path.join(
        evb_dir.RVICParam_dir, "flow_direction_file.nc"
    )
    
    flow_direction_path = os.path.join(evb_dir.Hydroanalysis_dir, "flow_direction.tif")
    flow_acc_path = os.path.join(evb_dir.Hydroanalysis_dir, "flow_acc.tif")
    flow_distance_path = os.path.join(evb_dir.Hydroanalysis_dir, "flow_distance.tif")
    
    if os.path.exists(flow_direction_file_path):
        logger.info(f"{flow_direction_file_path} already exists, skipping creation")
        
        if g_velocity is not None:
            logger.info("modify velocity... ...")
            
            # read
            flow_direction_dataset = Dataset(flow_direction_file_path, "a")
            domain_area_m2 = domain_dataset.variables["area"][:, :]
            with rasterio.open(flow_acc_path, "r", driver="GTiff") as dataset:
                flow_acc_array = dataset.read(1)
                flow_acc_array = flow_acc_array.astype(float)
            
            with rasterio.open(flow_distance_path, "r", driver="GTiff") as dataset:
                flow_distance_array = dataset.read(1)
                flow_distance_array = flow_distance_array.astype(float)
                
            tf_VIC = TF_VIC_class()
            
            domain_area_km2 = domain_area_m2 * 1e-6
            flow_acc_array_km2 = flow_acc_array * domain_area_km2
            velocity_array = tf_VIC.velocity(flow_acc_array_km2, slope, *g_velocity)
            
            velocity_array = velocity_array.astype(float)
            flow_direction_dataset.variables["velocity"][:, :] = np.array(velocity_array)
            
            if g_diffusion is not None:
                logger.info("modify diffusion... ...")
                diffusion_array = tf_VIC.diffusion(velocity_array, flow_distance_array, *g_diffusion)
                
                diffusion_array = diffusion_array.astype(float)
                flow_direction_dataset.variables["diffusion"][:, :] = np.array(diffusion_array)
            
            flow_direction_dataset.close()
            
        return

    # ====================== read general information ======================
    logger.debug("Reading latitude, longitude, and mask data from VIC parameters... ...")
    domain_lat = domain_dataset.variables["lat"][:]
    domain_lon = domain_dataset.variables["lon"][:]
    domain_mask = domain_dataset.variables["mask"][:, :]
    domain_area_m2 = domain_dataset.variables["area"][:, :]

    # ====================== read flow_direction and flow_acc ======================
    logger.debug(f"Reading flow direction data from {flow_direction_path}... ...")
    with rasterio.open(flow_direction_path, "r", driver="GTiff") as dataset:
        flow_direction_array = dataset.read(1)

    logger.debug(f"Reading flow accumulation data from {flow_acc_path}... ...")
    with rasterio.open(flow_acc_path, "r", driver="GTiff") as dataset:
        flow_acc_array = dataset.read(1)

    logger.debug(f"Reading flow distance data from {flow_distance_path}... ...")
    with rasterio.open(flow_distance_path, "r", driver="GTiff") as dataset:
        flow_distance_array = dataset.read(1)
        
    # ====================== combine them into a nc file ======================
    # create nc file
    logger.debug(f"Creating NetCDF file: {flow_direction_file_path}... ...")
    flow_direction_dataset = createFlowDirectionFile(
        flow_direction_file_path, domain_lat, domain_lon
    )

    # change type
    logger.debug("Processing and masking data... ...")
    domain_mask_array = deepcopy(domain_mask)
    domain_mask_array = domain_mask_array.astype(int)
    flow_direction_array = flow_direction_array.astype(int)
    flow_distance_array = flow_distance_array.astype(float)
    flow_acc_array = flow_acc_array.astype(float)
    
    # ====================== optional: cal velocity and diffusion ======================
    if g_velocity is not None:
        tf_VIC = TF_VIC_class()
        
        domain_area_km2 = domain_area_m2 * 1e-6
        flow_acc_array_km2 = flow_acc_array * domain_area_km2
        velocity_array = tf_VIC.velocity(flow_acc_array_km2, slope, *g_velocity)
        
        velocity_array = velocity_array.astype(float)
        flow_direction_dataset.variables["velocity"][:, :] = np.array(velocity_array)
        
        if g_diffusion is not None:
            diffusion_array = tf_VIC.diffusion(velocity_array, flow_distance_array, *g_diffusion)
            
            diffusion_array = diffusion_array.astype(float)
            flow_direction_dataset.variables["diffusion"][:, :] = np.array(diffusion_array)
            
    # mask
    # domain_mask_array[domain_mask == 0] = int(-9999)
    # flow_direction_array[domain_mask == 0] = int(-9999)
    # flow_distance_array[domain_mask == 0] = float(-9999.0)
    # flow_acc_array[domain_mask == 0] = float(-9999.0)
    # if g_velocity is not None:
    #     velocity_array[domain_mask == 0] = float(-9999.0)
    #     if g_diffusion is not None:
    #         diffusion_array[domain_mask == 0] = float(-9999.0)

    # assign values
    flow_direction_dataset.variables["lat"][:] = np.array(domain_lat)
    flow_direction_dataset.variables["lon"][:] = np.array(domain_lon)
    flow_direction_dataset.variables["Basin_ID"][:, :] = np.array(domain_mask_array)
    flow_direction_dataset.variables["Flow_Direction"][:, :] = np.array(flow_direction_array)
    flow_direction_dataset.variables["Flow_Distance"][:, :] = np.array(flow_distance_array)
    flow_direction_dataset.variables["Source_Area"][:, :] = np.array(flow_acc_array)
    
    flow_direction_dataset.close()

    logger.info(
        f"RVIC flow direction file generation successfully, saved to: {flow_direction_file_path}"
    )


def buildPourPointFile(evb_dir, names=None, lons=None, lats=None):
    """
    Generate a pour point CSV file for RVIC.

    This function writes a CSV with columns ``lons``, ``lats``, and ``names``.

    Parameters
    ----------
    evb_dir : Evb_dir
        Case directory manager.
    names : list, optional
        Pour-point names.
    lons : list, optional
        Pour-point longitudes.
    lats : list, optional
        Pour-point latitudes.

    Returns
    -------
    None
        The function writes the pour point data to a CSV file.
    """
    #! you should check it with FlowAcc (source area)

    logger.info("Starting to generate pour point file... ...")
    # ====================== set dir and path ======================
    RVICParam_dir = evb_dir.RVICParam_dir
    pourpoint_file_path = os.path.join(RVICParam_dir, "pour_points.csv")

    # ====================== build PourPointFile ======================
    # df
    pourpoint_file = pd.DataFrame(columns=["lons", "lats", "names"])
    
    if lons is None or lats is None or names is None:
        logger.error("Missing longitude, latitude, or name data for pour points")
        raise ValueError(
            "Longitude, latitude, and name lists must be provided when dpc_VIC_level1 is None"
        )

    pourpoint_file.lons = lons
    pourpoint_file.lats = lats
    pourpoint_file.names = names

    # ====================== Save pour point file ======================
    pourpoint_file.to_csv(pourpoint_file_path, header=True, index=False)
    logger.info(
        f"Pour point file generation successfully, saved to {pourpoint_file_path}"
    )


def buildUHBOXFile(evb_dir, createUH_func=create_uh.createGUH, **kwargs):
    """
    Generate and save the UHBOX (Unit Hydrograph Box) file.

    This function creates a UHBOX file using a specified unit hydrograph creation function.
    The resulting UHBOX data is then saved to a CSV file.

    Parameters
    ----------
    evb_dir : Evb_dir
        Case directory manager.
    createUH_func : function, optional
        Function used to generate the unit hydrograph, default is
        ``create_uh.createGUH``.
        
    **kwargs : dict
        Additional parameters to be passed to the `createUH_func`.

    Returns
    -------
    max_day : int
        Maximum duration (in days) used in the unit hydrograph generation.
    """
    logger.info("Starting to generate UHBOX file... ...")

    # build
    max_day, UHBOX_file = createUH_func(evb_dir, **kwargs)

    # save
    UHBOX_file.to_csv(evb_dir.uhbox_file_path, header=True, index=False)

    logger.info(
        f"UHBOX file generation successfully, saved to {evb_dir.uhbox_file_path}"
    )

    return max_day


def buildParamCFGFile(
    evb_dir,
    VELOCITY=1.5,
    DIFFUSION=800.0,
    OUTPUT_INTERVAL=86400,
    SUBSET_DAYS=10,
    CELL_FLOWDAYS=2,
    BASIN_FLOWDAYS=50,
    CONSTRAIN_FRACTIONS=True,
):
    """
    Generate and save the RVIC parameter configuration file.

    This function creates a configuration (CFG) file for RVIC parameter settings based on
    a reference configuration file and specified routing parameters.

    Parameters
    ----------
    evb_dir : Evb_dir
        Case directory manager.
    
    VELOCITY : float, optional
        Flow velocity parameter, default is 1.5, and the acceptable range is 1.0 to 3.0 m/s.
        
    DIFFUSION : float, optional
        Diffusion coefficient for routing, default is 800.0, and the acceptable range is 200 to 4000m3/s.
        
    OUTPUT_INTERVAL : int, optional
        Time interval (seconds) for output, default is 86400 seconds (1 day), and should typically be set as a multiple of 60.
        
    SUBSET_DAYS : int, optional
        Number of days used for subset processing, default is 10.
        
    CELL_FLOWDAYS : int, optional
        Flow duration at the cell level (days), default is 2.
        
    BASIN_FLOWDAYS : int, optional
        Flow duration at the basin level (days), default is 50.
        
    CONSTRAIN_FRACTIONS : bool, optional
        Whether RVIC should constrain source fractions.

    Notes
    -----
    The output file is written to ``evb_dir.rvic_param_cfg_file_path``.
    """
    logger.info("Starting to generate RVIC parameter configuration file... ...")
    # ====================== build CFGFile ======================
    # read reference cfg
    # param_cfg_file = ConfigParser()
    # param_cfg_file.optionxform = str  # import to keep case
    # param_cfg_file.read(evb_dir.rvic_param_cfg_file_reference_path)
    param_cfg_file = read_rvic_param_cfg_file_reference()

    # set cfg
    param_cfg_file.set("OPTIONS", "CASEID", evb_dir._case_name)
    param_cfg_file.set("OPTIONS", "CASE_DIR", evb_dir.RVICParam_dir)
    param_cfg_file.set("OPTIONS", "TEMP_DIR", evb_dir.RVICTemp_dir)
    param_cfg_file.set("OPTIONS", "SUBSET_DAYS", str(SUBSET_DAYS))
    param_cfg_file.set("OPTIONS", "CONSTRAIN_FRACTIONS", str(CONSTRAIN_FRACTIONS))
    param_cfg_file.set("POUR_POINTS", "FILE_NAME", evb_dir.pourpoint_file_path)
    param_cfg_file.set("UH_BOX", "FILE_NAME", evb_dir.uhbox_file_path)
    param_cfg_file.set("ROUTING", "FILE_NAME", evb_dir.flow_direction_file_path)
    param_cfg_file.set("ROUTING", "VELOCITY", str(VELOCITY))
    param_cfg_file.set("ROUTING", "DIFFUSION", str(DIFFUSION))
    param_cfg_file.set("ROUTING", "OUTPUT_INTERVAL", str(OUTPUT_INTERVAL))
    param_cfg_file.set("ROUTING", "CELL_FLOWDAYS", str(CELL_FLOWDAYS))
    param_cfg_file.set("ROUTING", "BASIN_FLOWDAYS", str(BASIN_FLOWDAYS))
    param_cfg_file.set("DOMAIN", "FILE_NAME", evb_dir.domainFile_path)

    # write cfg
    with open(evb_dir.rvic_param_cfg_file_path, "w") as configfile:
        param_cfg_file.write(configfile)

    logger.info(
        f"RVIC parameter configuration file generation successfully, saved to {evb_dir.rvic_param_cfg_file_path}"
    )


def buildConvCFGFile(
    evb_dir,
    RUN_STARTDATE="1979-09-01-00",
    DATL_FILE="rasm_sample_runoff.nc",
    PARAM_FILE_PATH="sample_rasm_parameters.nc",
    RVICHIST_MFILT=365,
):
    """
    Generate and save the RVIC convolution configuration file.

    This function creates a configuration (CFG) file for RVIC convolution settings
    based on a reference configuration file and specified parameters.

    Parameters
    ----------
    evb_dir : Evb_dir
        Case directory manager.
    
    RUN_STARTDATE : str, optional
        The start date for the RVIC run in "YYYY-MM-DD-HH" format.
        
    DATL_FILE : str, optional
        The name of the input runoff file. Default is "rasm_sample_runoff.nc".
        
    PARAM_FILE_PATH : str, optional
        The path to the RVIC parameter file. Default is "sample_rasm_parameters.nc".

    Returns
    -------
    None
        The output file is written to ``evb_dir.rvic_conv_cfg_file_path``.
    """
    logger.info("Starting to generate RVIC convolution configuration file... ...")
    # ====================== build CFGFile ======================
    # read reference cfg
    # conv_cfg_file = ConfigParser()
    # conv_cfg_file.optionxform = str  # import to keep case
    # conv_cfg_file.read(evb_dir.rvic_conv_cfg_file_reference_path)
    conv_cfg_file = read_rvic_conv_cfg_file_reference()

    # set cfg
    conv_cfg_file.set("OPTIONS", "CASEID", evb_dir._case_name)
    conv_cfg_file.set("OPTIONS", "CASE_DIR", evb_dir.RVICConv_dir)
    conv_cfg_file.set("OPTIONS", "RUN_STARTDATE", RUN_STARTDATE)
    conv_cfg_file.set("HISTORY", "RVICHIST_MFILT", str(RVICHIST_MFILT))

    conv_cfg_file.set("DOMAIN", "FILE_NAME", evb_dir.domainFile_path)

    conv_cfg_file.set("PARAM_FILE", "FILE_NAME", PARAM_FILE_PATH)

    conv_cfg_file.set("INPUT_FORCINGS", "DATL_PATH", evb_dir.VICResults_dir)
    conv_cfg_file.set("INPUT_FORCINGS", "DATL_FILE", DATL_FILE)

    # write cfg
    with open(evb_dir.rvic_conv_cfg_file_path, "w") as configfile:
        conv_cfg_file.write(configfile)

    logger.info(
        f"RVIC convolution configuration file generation successfully, saved to {evb_dir.rvic_conv_cfg_file_path}"
    )
