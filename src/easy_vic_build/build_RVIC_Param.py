# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import os
import numpy as np
import pandas as pd
import rasterio.transform
import rasterio
from copy import deepcopy
from .tools.params_func.createParametersDataset import createFlowDirectionFile
from .tools.utilities import read_cfg_to_dict, read_rvic_param_cfg_file_reference, read_rvic_conv_cfg_file_reference
from .tools.decoractors import clock_decorator
from .tools.uh_func import create_uh
from .tools.geo_func.search_grids import *
from configparser import ConfigParser
try:
    from rvic.parameters import parameters as rvic_parameters
    HAS_RVIC = True
except:
    HAS_RVIC = False


def buildRVICParam_general(evb_dir, dpc_VIC_level1, params_dataset_level1,
                           ppf_kwargs=dict(), uh_params={"createUH_func": create_uh.createGUH, "uh_dt": 3600, "tp": 1.4, "mu": 5.0, "m": 3.0, "plot_bool": True, "max_day":None, "max_day_range": (0, 10), "max_day_converged_threshold": 0.001},
                           cfg_params={"VELOCITY": 1.5, "DIFFUSION": 800.0, "OUTPUT_INTERVAL": 86400, "SUBSET_DAYS": 10, "CELL_FLOWDAYS": 2, "BASIN_FLOWDAYS": 50}):
    # general RVICParam before using rvic_parameters
    # buildRVICFlowDirectionFile
    buildRVICFlowDirectionFile(evb_dir, params_dataset_level1)
    
    # buildPourPointFile
    buildPourPointFile(evb_dir, dpc_VIC_level1, **ppf_kwargs)
    
    # buildUHBOXFile
    buildUHBOXFile(evb_dir, **uh_params)
    
    # buildParamCFGFile
    buildParamCFGFile(evb_dir, **cfg_params)
    
    
@clock_decorator
def buildRVICParam(evb_dir, dpc_VIC_level1, params_dataset_level1,
                   ppf_kwargs=dict(), uh_params={"createUH_func": create_uh.createGUH, "uh_dt": 3600, "tp": 1.4, "mu": 5.0, "m": 3.0, "plot_bool": True, "max_day":None, "max_day_range": (0, 10), "max_day_converged_threshold": 0.001},
                   cfg_params={"VELOCITY": 1.5, "DIFFUSION": 800.0, "OUTPUT_INTERVAL": 86400, "SUBSET_DAYS": 10, "CELL_FLOWDAYS": 2, "BASIN_FLOWDAYS": 50}):    

    # buildRVICParam_general
    buildRVICParam_general(evb_dir, dpc_VIC_level1, params_dataset_level1, ppf_kwargs, uh_params, cfg_params)
    
    # build rvic parameters
    param_cfg_file_dict = read_cfg_to_dict(evb_dir.rvic_param_cfg_file_path)
    
    if HAS_RVIC:
        rvic_parameters(param_cfg_file_dict, numofproc=1)
    else:
        raise ImportError("no rvic for buildRVICParam")


def buildRVICFlowDirectionFile(evb_dir, params_dataset_level1):
    # ====================== set dir and path ======================
    # set path
    flow_direction_file_path = os.path.join(evb_dir.RVICParam_dir, "flow_direction_file.nc")
    flow_direction_path = os.path.join(evb_dir.Hydroanalysis_dir, "flow_direction.tif")
    flow_acc_path = os.path.join(evb_dir.Hydroanalysis_dir, "flow_acc.tif")
    flow_distance_path = os.path.join(evb_dir.Hydroanalysis_dir, "flow_distance.tif")
    
    # ====================== read general information ======================
    params_lat = params_dataset_level1.variables["lat"][:]
    params_lon = params_dataset_level1.variables["lon"][:]
    params_mask = params_dataset_level1.variables["run_cell"][:, :]
    
    # ====================== read flow_direction and flow_acc ======================
    with rasterio.open(flow_direction_path, 'r', driver='GTiff') as dataset:
        flow_direction_array = dataset.read(1)
    
    with rasterio.open(flow_acc_path, 'r', driver='GTiff') as dataset:
        flow_acc_array = dataset.read(1)

    with rasterio.open(flow_distance_path, 'r', driver='GTiff') as dataset:
        flow_distance_array = dataset.read(1)
    
    # ====================== combine them into a nc file ======================
    # create nc file
    flow_direction_dataset = createFlowDirectionFile(flow_direction_file_path, params_lat, params_lon)
    
    # change type
    params_mask_array = deepcopy(params_mask)
    params_mask_array = params_mask_array.astype(int)
    flow_direction_array = flow_direction_array.astype(int)
    flow_distance_array = flow_distance_array.astype(float)
    flow_acc_array = flow_acc_array.astype(float)
    
    # mask
    params_mask_array[params_mask==0] = int(-9999)
    flow_direction_array[params_mask==0] = int(-9999)
    flow_distance_array[params_mask==0] = float(-9999.0)
    flow_acc_array[params_mask==0] = float(-9999.0)
    
    # assign values
    flow_direction_dataset.variables["lat"][:] = np.array(params_lat)
    flow_direction_dataset.variables["lon"][:] = np.array(params_lon)
    flow_direction_dataset.variables["Basin_ID"][:, :] = np.array(params_mask_array)
    flow_direction_dataset.variables["Flow_Direction"][:, :] = np.array(flow_direction_array)
    flow_direction_dataset.variables["Flow_Distance"][:, :] = np.array(flow_distance_array)
    flow_direction_dataset.variables["Source_Area"][:, :] = np.array(flow_acc_array)
    
    flow_direction_dataset.close()


def buildPourPointFile(evb_dir, dpc_VIC_level1=None, names=None, lons=None, lats=None):
    #* dpc_VIC_level1.basin_shp should contain "camels_topo" attributes
    #! you should check it with FlowAcc (source area)
    # ====================== set dir and path ======================
    RVICParam_dir = evb_dir.RVICParam_dir
    pourpoint_file_path = os.path.join(RVICParam_dir, "pour_points.csv")
    
    # ====================== build PourPointFile ======================
    # df
    pourpoint_file = pd.DataFrame(columns=["lons", "lats", "names"])
    
    if dpc_VIC_level1 is not None:
        x, y = dpc_VIC_level1.basin_shp.loc[:, "camels_topo:gauge_lon"].values[0], dpc_VIC_level1.basin_shp.loc[:, "camels_topo:gauge_lat"].values[0]
        pourpoint_file.lons = [x]
        pourpoint_file.lats = [y]
        pourpoint_file.names = [f"gauge_id:{dpc_VIC_level1.basin_shp.loc[:, 'camels_topo:gauge_id'].values[0]}"]
    else:
        pourpoint_file.lons = lons
        pourpoint_file.lats = lats
        pourpoint_file.names = names
    
    pourpoint_file.to_csv(pourpoint_file_path, header=True, index=False)


def buildUHBOXFile(evb_dir, createUH_func=create_uh.createGUH, **kwargs):
    # build
    max_day, UHBOX_file = createUH_func(evb_dir, **kwargs)
    
    # save
    UHBOX_file.to_csv(evb_dir.uhbox_file_path, header=True, index=False)
    return max_day


def buildParamCFGFile(evb_dir, VELOCITY=1.5, DIFFUSION=800.0, OUTPUT_INTERVAL=86400, SUBSET_DAYS=10, CELL_FLOWDAYS=2, BASIN_FLOWDAYS=50):
    # ====================== build CFGFile ======================
    # read reference cfg
    # param_cfg_file = ConfigParser()
    # param_cfg_file.optionxform = str  # import to keep case
    # param_cfg_file.read(evb_dir.rvic_param_cfg_file_reference_path)
    param_cfg_file = read_rvic_param_cfg_file_reference()
    
    # set cfg
    param_cfg_file.set("OPTIONS", 'CASEID', evb_dir._case_name)
    param_cfg_file.set("OPTIONS", 'CASE_DIR', evb_dir.RVICParam_dir)
    param_cfg_file.set("OPTIONS", 'TEMP_DIR', evb_dir.RVICTemp_dir)
    param_cfg_file.set("OPTIONS", 'SUBSET_DAYS', str(SUBSET_DAYS))
    param_cfg_file.set("POUR_POINTS", 'FILE_NAME', evb_dir.pourpoint_file_path)
    param_cfg_file.set("UH_BOX", 'FILE_NAME', evb_dir.uhbox_file_path)
    param_cfg_file.set("ROUTING", 'FILE_NAME', evb_dir.flow_direction_file_path)
    param_cfg_file.set("ROUTING", 'VELOCITY', str(VELOCITY))
    param_cfg_file.set("ROUTING", 'DIFFUSION', str(DIFFUSION))
    param_cfg_file.set("ROUTING", 'OUTPUT_INTERVAL', str(OUTPUT_INTERVAL))
    param_cfg_file.set("ROUTING", 'CELL_FLOWDAYS', str(CELL_FLOWDAYS))
    param_cfg_file.set("ROUTING", 'BASIN_FLOWDAYS', str(BASIN_FLOWDAYS))
    param_cfg_file.set("DOMAIN", 'FILE_NAME', evb_dir.domainFile_path)
    
    # write cfg
    with open(evb_dir.rvic_param_cfg_file_path, 'w') as configfile:
        param_cfg_file.write(configfile)
        
def buildConvCFGFile(evb_dir, RUN_STARTDATE="1979-09-01-00", DATL_FILE="rasm_sample_runoff.nc", PARAM_FILE_PATH="sample_rasm_parameters.nc"):
    # ====================== build CFGFile ======================
    # read reference cfg
    # conv_cfg_file = ConfigParser()
    # conv_cfg_file.optionxform = str  # import to keep case
    # conv_cfg_file.read(evb_dir.rvic_conv_cfg_file_reference_path)
    conv_cfg_file = read_rvic_conv_cfg_file_reference()
    
    # set cfg
    conv_cfg_file.set("OPTIONS", 'CASEID', evb_dir._case_name)
    conv_cfg_file.set("OPTIONS", 'CASE_DIR', evb_dir.RVICConv_dir)
    conv_cfg_file.set("OPTIONS", 'RUN_STARTDATE', RUN_STARTDATE)

    conv_cfg_file.set("DOMAIN", 'FILE_NAME', evb_dir.domainFile_path)
    
    conv_cfg_file.set("PARAM_FILE", 'FILE_NAME', PARAM_FILE_PATH)
    
    conv_cfg_file.set("INPUT_FORCINGS", 'DATL_PATH', evb_dir.VICResults_dir)
    conv_cfg_file.set("INPUT_FORCINGS", 'DATL_FILE', DATL_FILE)
    
    # write cfg
    with open(evb_dir.rvic_conv_cfg_file_path, 'w') as configfile:
        conv_cfg_file.write(configfile)


def modifyRVICParam_for_pourpoint(evb_dir, pourpoint_lon, pourpoint_lat, pourpoint_direction_code, params_dataset_level1, domain_dataset, 
                                  reverse_lat=True, stream_acc_threshold=100.0, flow_direction_pkg="wbw", crs_str="EPSG:4326"):
    # modify PourPointFile
    buildPourPointFile(evb_dir, None, names=["pourpoint"], lons=[pourpoint_lon], lats=[pourpoint_lat])

    # modify buildRVICFlowDirectionFile, modify 0 direction (edge) of pourpoint to pourpoint_direction_code
    buildRVICFlowDirectionFile(evb_dir, params_dataset_level1, domain_dataset, reverse_lat=reverse_lat, stream_acc_threshold=stream_acc_threshold, flow_direction_pkg=flow_direction_pkg, crs_str=crs_str,
                           pourpoint_lon=pourpoint_lon, pourpoint_lat=pourpoint_lat, pourpoint_direction_code=pourpoint_direction_code)


if __name__ == "__main__":
    tp=1.4
    mu=5.0
    m=3.0