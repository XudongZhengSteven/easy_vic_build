# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import os
import shutil
import rasterio
from .tools.geo_func.search_grids import *
from .tools.hydroanalysis_func import create_dem, create_flow_distance, hydroanalysis_arcpy, hydroanalysis_wbw
from .tools.utilities import remove_and_mkdir


def buildHydroanalysis(evb_dir, params_dataset_level1, domain_dataset, reverse_lat=True, stream_acc_threshold=100.0, flow_direction_pkg="wbw", crs_str="EPSG:4326",
                       pourpoint_lon=None, pourpoint_lat=None, pourpoint_direction_code=None):
    # ====================== set dir and path ======================
    # set path
    dem_level1_tif_path = os.path.join(evb_dir.Hydroanalysis_dir, "dem_level1.tif")
    flow_direction_path = os.path.join(evb_dir.Hydroanalysis_dir, "flow_direction.tif")
    flow_acc_path = os.path.join(evb_dir.Hydroanalysis_dir, "flow_acc.tif")
    flow_distance_path = os.path.join(evb_dir.Hydroanalysis_dir, "flow_distance.tif")
    
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
        
    else:
        pourpoint_x_index = None
        pourpoint_y_index = None
    
    # ====================== create and save dem_level1.tif ======================
    transform = create_dem.create_dem_from_params(params_dataset_level1, dem_level1_tif_path, crs_str=crs_str, reverse_lat=reverse_lat)
    
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

        # cp data from workspace to RVICParam_dir
        shutil.copy(os.path.join(workspace_dir, "flow_direction.tif"), flow_direction_path)
        shutil.copy(os.path.join(workspace_dir, "flow_acc.tif"), flow_acc_path)
    
    else:
        print("please input correct flow_direction_pkg")
    
    # ====================== read flow_direction ======================
    with rasterio.open(flow_direction_path, 'r', driver='GTiff') as dataset:
        flow_direction_array = dataset.read(1)
    
    # ====================== cal flow distance and save it ======================
    create_flow_distance.create_flow_distance(flow_distance_path, flow_direction_array, x_length_array, y_length_array, transform, crs_str=crs_str)
    
    # clean workspace_dir
    remove_and_mkdir(workspace_dir)
    
    
        