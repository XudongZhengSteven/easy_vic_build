# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: build_hydroanalysis

This module provides functions for performing hydroanalysis tasks, such as creating a Digital Elevation Model (DEM), 
calculating flow direction, flow accumulation, and flow distance. It supports two packages for calculating flow direction: 
"arcpy" and "wbw", and allows the user to define a pour point for localized flow direction calculations.

Functions:
----------
    - buildHydroanalysis: Performs the hydroanalysis process, including DEM generation, flow direction and accumulation 
      calculation, and flow distance calculation. The function supports both "arcpy" and "wbw" packages for flow direction calculation.
    
Usage:
------
    To use this module, provide the necessary datasets (e.g., parameters and domain datasets), along with optional configuration 
    settings such as pour point location and flow direction package. Then call `buildHydroanalysis` to perform the entire hydroanalysis 
    process and generate output files such as the DEM, flow direction, flow accumulation, and flow distance.

Example:
--------
    # Example usage:
    basin_index = 213
    model_scale = "6km"
    date_period = ["19980101", "19981231"]
    case_name = f"{basin_index}_{model_scale}"
    
    evb_dir = Evb_dir(cases_home="./examples")  # cases_home="/home/xdz/code/VIC_xdz/cases"
    evb_dir.builddir(case_name)
    remove_and_mkdir(evb_dir.RVICParam_dir)
    evb_dir.builddir(case_name)

    domain_dataset = readDomain(evb_dir)
    params_dataset_level0, params_dataset_level1 = readParam(evb_dir)

    buildHydroanalysis(evb_dir, params_dataset_level1, domain_dataset, reverse_lat=True, flow_direction_pkg="wbw", crs_str="EPSG:4326",
                       create_stream=True,
                       pourpoint_lon=None, pourpoint_lat=None, pourpoint_direction_code=None)
    
    domain_dataset.close()
    params_dataset_level0.close()
    params_dataset_level1.close()
    
Dependencies:
-------------
    - rasterio: For reading and writing geospatial raster data.
    - shutil: For file operations like copying and removing directories.
    - tools.geo_func: For searching grid locations.
    - tools.hydroanalysis_func: For DEM creation and hydroanalysis.
    - tools.utilities: For utility functions like directory cleaning.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com
    
"""

import os
import shutil
import rasterio
from .tools.geo_func.search_grids import *
from .tools.hydroanalysis_func import create_dem, create_flow_distance, hydroanalysis_arcpy, hydroanalysis_wbw
from .tools.utilities import remove_and_mkdir
from . import logger


def buildHydroanalysis(evb_dir, params_dataset_level1, domain_dataset, reverse_lat=True, stream_acc_threshold=100.0, flow_direction_pkg="wbw", crs_str="EPSG:4326",
                       pourpoint_lon=None, pourpoint_lat=None, pourpoint_direction_code=None):
    """
    Perform hydroanalysis tasks to generate DEM, flow direction, flow accumulation, and flow distance. 
    The results are saved in specified directories and can be used for further analysis or modeling.

    Parameters
    ----------
    evb_dir : object
        The directory structure object containing paths for saving output files.

    params_dataset_level1 : Dataset
        A dataset containing the parameters (e.g., latitude, longitude) for DEM creation.

    domain_dataset : Dataset
        A dataset containing domain information such as x and y lengths.

    reverse_lat : bool, optional
        Flag to reverse the latitude direction. Default is True.

    stream_acc_threshold : float, optional
        The threshold value for stream accumulation. Default is 100.0.

    flow_direction_pkg : str, optional
        The package used to calculate flow direction. Options are "arcpy" and "wbw". Default is "wbw".

    crs_str : str, optional
        The coordinate reference system string. Default is "EPSG:4326".

    pourpoint_lon : float, optional
        Longitude of the pour point location. Default is None.

    pourpoint_lat : float, optional
        Latitude of the pour point location. Default is None.

    pourpoint_direction_code : int, optional
        The direction code of the pour point. Default is None.

    Returns
    -------
    None
        The function generates several output files (e.g., DEM, flow direction, flow accumulation, flow distance) 
        and saves them in the specified directory.
    """
    
    logger.info("Starting to building hydroanalysis... ...")
    # ====================== set dir and path ======================
    # set path
    dem_level1_tif_path = os.path.join(evb_dir.Hydroanalysis_dir, "dem_level1.tif")
    flow_direction_path = os.path.join(evb_dir.Hydroanalysis_dir, "flow_direction.tif")
    flow_acc_path = os.path.join(evb_dir.Hydroanalysis_dir, "flow_acc.tif")
    flow_distance_path = os.path.join(evb_dir.Hydroanalysis_dir, "flow_distance.tif")
    
    logger.debug(f"DEM path: {dem_level1_tif_path}")
    logger.debug(f"Flow direction path: {flow_direction_path}")
    
    # ====================== read ======================
    params_lat = params_dataset_level1.variables["lat"][:]
    params_lon = params_dataset_level1.variables["lon"][:]
    x_length_array = domain_dataset.variables["x_length"][:, :]
    y_length_array = domain_dataset.variables["y_length"][:, :]
    
    # get index for pourpoint
    if pourpoint_lat is not None:
        searched_grid_index = search_grids_nearest([pourpoint_lat], [pourpoint_lon], params_lat, params_lon, search_num=1)[0]
        pourpoint_x_index = searched_grid_index[1][0]
        pourpoint_y_index = searched_grid_index[0][0]
        logger.info(f"Pourpoint found at index: ({pourpoint_y_index}, {pourpoint_x_index})")
    else:
        pourpoint_x_index = None
        pourpoint_y_index = None
        logger.info(f"No pourpoint provided, default flow direction calculation will be used")
    
    # ====================== create and save dem_level1.tif ======================
    transform = create_dem.create_dem_from_params(params_dataset_level1, dem_level1_tif_path, crs_str=crs_str, reverse_lat=reverse_lat)
    logger.debug(f"DEM created and saved to: {dem_level1_tif_path}")
    
    # ====================== build flow drection ======================
    if flow_direction_pkg == "arcpy":
        # arcpy related path
        arcpy_python_path = evb_dir.arcpy_python_path
        arcpy_python_script_path = os.path.join(evb_dir.__package_dir__, "arcpy_scripts\\build_flowdirection_arcpy.py")
        
        arcpy_workspace_dir = os.path.join(evb_dir.Hydroanalysis_dir, "arcpy_workspace")
        remove_and_mkdir(arcpy_workspace_dir)
        workspace_dir = arcpy_workspace_dir
        
        # build flow direction based on arcpy
        out = hydroanalysis_arcpy.hydroanalysis_arcpy(workspace_dir, dem_level1_tif_path, arcpy_python_path, arcpy_python_script_path, stream_acc_threshold)
        logger.info("Flow direction and accumulation calculated using arcpy")
        
        # cp data from workspace to RVICParam_dir
        shutil.copy(os.path.join(workspace_dir, "flow_direction.tif"), flow_direction_path)
        shutil.copy(os.path.join(workspace_dir, "flow_acc.tif"), flow_acc_path)
    
    elif flow_direction_pkg == "wbw":
        # wbw related path
        wbw_workspace_dir = os.path.join(evb_dir.Hydroanalysis_dir, "wbw_workspace")
        remove_and_mkdir(wbw_workspace_dir)
        workspace_dir = wbw_workspace_dir
        
        # build flow direction based on wbw
        out = hydroanalysis_wbw.hydroanalysis_wbw(workspace_dir, dem_level1_tif_path, pourpoint_x_index=pourpoint_x_index, pourpoint_y_index=pourpoint_y_index, pourpoint_direction_code=pourpoint_direction_code)
        logger.info("Flow direction and accumulation calculated using wbw")
        
        # cp data from workspace to RVICParam_dir
        shutil.copy(os.path.join(workspace_dir, "flow_direction.tif"), flow_direction_path)
        shutil.copy(os.path.join(workspace_dir, "flow_acc.tif"), flow_acc_path)
    
    else:
        logger.error("Invalid flow_direction_pkg. Please choose 'arcpy' or 'wbw'")
        print("please input correct flow_direction_pkg")
        return
    
    # ====================== read flow_direction ======================
    with rasterio.open(flow_direction_path, 'r', driver='GTiff') as dataset:
        flow_direction_array = dataset.read(1)
        
    logger.debug(f"Flow direction read from: {flow_direction_path}")
    
    # ====================== cal flow distance and save it ======================
    create_flow_distance.create_flow_distance(flow_distance_path, flow_direction_array, x_length_array, y_length_array, transform, crs_str=crs_str)
    logger.info(f"Flow distance file calculated and saved to: {flow_distance_path}")
    
    # clean workspace_dir
    remove_and_mkdir(workspace_dir)
    logger.debug(f"Workspace directory cleaned: {workspace_dir}")
    
    logger.info("Building hydroanalysis completed successfully")