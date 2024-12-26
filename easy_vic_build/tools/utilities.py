# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os

import pandas as pd
import geopandas as gpd
from tqdm import *
import pickle
from netCDF4 import Dataset
import shutil
from .geo_func.search_grids import *
from .params_func.GlobalParamParser import GlobalParamParser
from .params_func.params_set import *
from .dpc_func.basin_grid_class import HCDNBasins
from configparser import ConfigParser


## ------------------------ general utilities ------------------------
def check_and_mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(os.path.abspath(dir))

def remove_and_mkdir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(os.path.abspath(dir))
    
def remove_files(dir):
    # do not remove subdir
    for f in os.listdir(dir):
        fp = os.path.join(dir, f)
        if os.path.isfile(fp):
            os.remove(fp)

def setHomePath(root="E:"):
    home = f"{root}/data/hydrometeorology/CAMELS"
    return root, home


def exportToCsv(basin_shp, fpath_dir):
    columns = basin_shp.columns.to_list()
    streamflow = basin_shp.loc[:, columns.pop(11)]
    prep = basin_shp.loc[:, columns.pop(11)]
    gleam_e_daily = basin_shp.loc[:, columns.pop(-1)]

    columns.remove("geometry")
    # columns.remove("intersects_grids")

    csv_df = basin_shp.loc[:, columns]

    # save
    csv_df.to_csv(os.path.join(fpath_dir, "basin_shp.csv"))

    # save streamflow
    if not os.path.exists(os.path.join(fpath_dir, "streamflow")):
        os.mkdir(os.path.join(fpath_dir, "streamflow"))
    for i in tqdm(streamflow.index, desc="loop for basins to save streamflow", colour="green"):
        streamflow[i].to_csv(os.path.join(fpath_dir, "streamflow", f"{i}.csv"))

    # save prep
    if not os.path.exists(os.path.join(fpath_dir, "prep")):
        os.mkdir(os.path.join(fpath_dir, "prep"))
    for i in tqdm(prep.index, desc="loop for basins to save prep", colour="green"):
        prep[i].to_csv(os.path.join(fpath_dir, "prep", f"{i}.csv"))

    # save gleam_e_daily
    if not os.path.exists(os.path.join(fpath_dir, "gleam_e_daily")):
        os.mkdir(os.path.join(fpath_dir, "gleam_e_daily"))
    for i in tqdm(gleam_e_daily.index, desc="loop for basins to save gleam_e_daily", colour="green"):
        gleam_e_daily[i].to_csv(os.path.join(fpath_dir, "gleam_e_daily", f"{i}.csv"))
        

def checkGaugeBasin(basinShp, usgs_streamflow, BasinAttribute, forcingDaymetGaugeAttributes):
    id_basin_shp = set(basinShp.hru_id.values)
    id_usgs_streamflow = set([usgs_streamflow[i].iloc[0, 0] for i in range(len(usgs_streamflow))])
    id_BasinAttribute = set(BasinAttribute["camels_clim"].gauge_id.values)
    id_forcingDaymet = set([forcingDaymetGaugeAttributes[i]["gauge_id"]
                            for i in range(len(forcingDaymetGaugeAttributes))])
    print(f"id_HCDN_shp: {len(id_basin_shp)}, id_usgs_streamflow: {len(id_usgs_streamflow)}, id_BasinAttribute: {len(id_BasinAttribute)}, id_forcingDatmet: {len(id_forcingDaymet)}")
    print(f"id_usgs_streamflow - id_HCDN_shp: {id_usgs_streamflow - id_basin_shp}")
    print(f"id_BasinAttribute - id_HCDN_shp: {id_BasinAttribute - id_basin_shp}")
    print(f"id_forcingDatmet - id_HCDN_shp: {id_forcingDaymet - id_basin_shp}")
    # result
    # id_usgs_streamflow - id_HCDN_shp: {9535100, 6775500, 6846500}
    # id_forcingDatmet - id_HCDN_shp: {6846500, 6775500, 3448942, 1150900, 2081113, 9535100}

## ------------------------ read function ------------------------
def readHCDNGrids(home="E:\\data\\hydrometeorology\\CAMELS"):
    grid_shp_label_path = os.path.join(home, "map", "grids_0_25_label.shp")
    grid_shp_label = gpd.read_file(grid_shp_label_path)
    print(grid_shp_label)

    grid_shp_path = os.path.join(home, "map", "grids_0_25.shp")
    grid_shp = gpd.read_file(grid_shp_path)
    print(grid_shp)

    # combine grid_shp_lable into grid_shp
    grid_shp["point_geometry"] = grid_shp_label.geometry

    return grid_shp


def readHCDNBasins(home="E:\\data\\hydrometeorology\\CAMELS"):
    # read data: HCDN
    HCDN_shp_path = os.path.join(home, "basin_set_full_res", "HCDN_nhru_final_671.shp")
    HCDN_shp = gpd.read_file(HCDN_shp_path)
    HCDN_shp["AREA_km2"] = HCDN_shp.AREA / 1000000  # m2 -> km2
    print(HCDN_shp)
    return HCDN_shp


def read_one_HCDN_basin_shp(basin_index, home="E:\\data\\hydrometeorology\\CAMELS"):
    basin_shp_all = HCDNBasins(home)
    basin_shp = basin_shp_all.loc[basin_index: basin_index, :]
    return basin_shp_all, basin_shp


def readdpc(evb_dir):
    # read
    with open(evb_dir.dpc_VIC_level0_path, "rb") as f:
        dpc_VIC_level0 = pickle.load(f)
    
    with open(evb_dir.dpc_VIC_level1_path, "rb") as f:
        dpc_VIC_level1 = pickle.load(f)
    
    with open(evb_dir.dpc_VIC_level2_path, "rb") as f:
        dpc_VIC_level2 = pickle.load(f)
    
    return dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2


def readDomain(evb_dir):
    # read
    domain_dataset = Dataset(evb_dir.domainFile_path, "r", format="NETCDF4")
    
    return domain_dataset


def readParam(evb_dir, mode="r"):
    # read
    params_dataset_level0 = Dataset(evb_dir.params_dataset_level0_path, mode, format="NETCDF4")
    params_dataset_level1 = Dataset(evb_dir.params_dataset_level1_path, mode, format="NETCDF4")
    
    return params_dataset_level0, params_dataset_level1


def clearParam(evb_dir):
    if os.path.isfile(evb_dir.params_dataset_level0_path):
        os.remove(evb_dir.params_dataset_level0_path)

    if os.path.isfile(evb_dir.params_dataset_level1_path):
        os.remove(evb_dir.params_dataset_level1_path)


def readRVICParam(evb_dir):
    # read
    flow_direction_dataset = Dataset(evb_dir.flow_direction_file_path, "r", "NETCDF4")
    pourpoint_file = pd.read_csv(evb_dir.pourpoint_file_path)
    uhbox_file = pd.read_csv(evb_dir.uhbox_file_path)
    cfg_file = ConfigParser()
    cfg_file.read(evb_dir.cfg_file_path)
    
    return flow_direction_dataset, pourpoint_file, uhbox_file, cfg_file


def read_cfg_to_dict(cfg_file_path):
    cfg_file = ConfigParser()
    cfg_file.optionxform = str
    cfg_file.read(cfg_file_path)
    cfg_file_dict = {section: dict(cfg_file.items(section)) for section in cfg_file.sections()}
    
    return cfg_file_dict


def readGlobalParam(evb_dir):
    globalParam = GlobalParamParser()
    globalParam.load(evb_dir.globalParam_path)
    
    return globalParam
    

def readCalibrateCp(evb_dir):
    with open(evb_dir.calibrate_cp_path, "rb") as f:
        state = pickle.load(f)
        # current_generation = state["current_generation"]
        # initial_population = state["initial_population"]
        # population = state["population"]
        # history = state["history"]
        # front_fitness = [history[i][1][0][0].fitness.values for i in range(len(history))]
        # plt.plot(front_fitness)
    return state


def readBasinMap(evb_dir):
    stream_gdf = gpd.read_file(os.path.join(evb_dir.BasinMap_dir, "stream_raster_shp_clip.shp"))
    return stream_gdf