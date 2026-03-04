# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from general_info import *
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import geopandas as gpd

from HRB_build_dpc import dataProcess_VIC_level0_HRB, dataProcess_VIC_level1_HRB
from HRB_extractData_func.Extract_SoilGrids1km import SoilGrids_soillayerresampler, set_g_params_soilGrids_layer

from easy_vic_build.tools.params_func.build_Param_interface import buildParam_level0_interface, buildParam_level1_interface, buildParam_level0_interface_ARNO_spatially_uniform, buildParam_level0_interface_Nijssen_spatially_uniform
from easy_vic_build.tools.utilities import readDomain
from easy_vic_build.build_Param import buildParam_level0, buildParam_level1, scaling_level0_to_level1
from easy_vic_build.tools.params_func.params_set import *
from easy_vic_build.tools.params_func.TransferFunction import TF_VIC
from easy_vic_build.tools.dpc_func.basin_grid_func import createEmptyArray_from_gridshp, assignValue_for_grid_array, createStand_grids_lat_lon_from_gridshp
from easy_vic_build.tools.nested_basin_func.nested_basin_func import cal_unique_mask_nested_basin
from easy_vic_build.bulid_Domain import remap_level1_to_level0_mask

class buildParam_level0_interface_HRB(buildParam_level0_interface):
    def set_ele_std(self):
        # ele_std, m
        self.logger.debug("setting ele_std... ...")
        
        self.grid_array_ele_std = createEmptyArray_from_gridshp(
            self.stand_grids_lat_level0, self.stand_grids_lon_level0, dtype=float, missing_value=np.nan
        )
        
        self.grid_array_ele_std = assignValue_for_grid_array(
            self.grid_array_ele_std,
            self.grid_shp_level0.loc[:, "ASTGTM_DEM_std_Value"],
            self.rows_index_level0,
            self.cols_index_level0,
        )
    
    def set_mean_slope(self):
        # mean slope, % (m/m)
        self.logger.debug("setting mean slope... ...")
        
        self.grid_array_mean_slope = createEmptyArray_from_gridshp(
            self.stand_grids_lat_level0, self.stand_grids_lon_level0,
            dtype=float, missing_value=np.nan
        )
        
        self.grid_array_mean_slope = assignValue_for_grid_array(
            self.grid_array_mean_slope,
            self.grid_shp_level0.loc[:, "ASTGTM_DEM_mean_slope_Value%"],
            self.rows_index_level0,
            self.cols_index_level0,
        )
        
        self.params_dataset_level0.variables["slope"][:, :] = self.grid_array_mean_slope
        
    def set_elev(self):
        # elev, m, Arithmetic mean
        self.logger.debug("setting elev... ...")
        
        grid_array_elev = createEmptyArray_from_gridshp(
            self.stand_grids_lat_level0, self.stand_grids_lon_level0,
            dtype=float, missing_value=np.nan
        )
        grid_array_elev = assignValue_for_grid_array(
            grid_array_elev,
            self.grid_shp_level0.loc[:, "ASTGTM_DEM_mean_Value"],
            self.rows_index_level0,
            self.cols_index_level0,
        )

        self.params_dataset_level0.variables["elev"][:, :] = grid_array_elev
    
    
class buildParam_level1_interface_HRB(buildParam_level1_interface):
    def __init__(self, evb_dir, logger, TF_VIC, dpc_VIC_level1, reverse_lat=True, domain_dataset=None, stand_grids_lat_level1=None, stand_grids_lon_level1=None, rows_index_level1=None, cols_index_level1=None):
        super().__init__(evb_dir, logger, TF_VIC, dpc_VIC_level1, reverse_lat, domain_dataset, stand_grids_lat_level1, stand_grids_lon_level1, rows_index_level1, cols_index_level1)

    def set_annual_prec(self):
        # annual_prec, mm
        self.logger.debug("setting annual_prec... ...")
        
        self.grid_array_annual_P = createEmptyArray_from_gridshp(
            self.stand_grids_lat_level1, self.stand_grids_lon_level1, dtype=float, missing_value=np.nan
        )
        
        self.grid_array_annual_P = assignValue_for_grid_array(
            self.grid_array_annual_P,
            self.grid_shp_level1.loc[:, "annual_P_CMFD_mm"],
            self.rows_index_level1,
            self.cols_index_level1,
        )
        
        self.params_dataset_level1.variables["annual_prec"][:, :] = self.grid_array_annual_P
        

class buildParam_level0_interface_ARNO_spatially_uniform_HRB(buildParam_level0_interface_ARNO_spatially_uniform, buildParam_level0_interface_HRB):
    pass

class buildParam_level0_interface_Nijssen_spatially_uniform_HRB(buildParam_level0_interface_Nijssen_spatially_uniform, buildParam_level0_interface_HRB):
    pass

def build_params_HRB(evb_dir_modeling, reverse_lat=True):
    # read dpc
    dpc_VIC_level0 = dataProcess_VIC_level0_HRB(evb_dir_modeling._dpc_VIC_level0_path)
    dpc_VIC_level1 = dataProcess_VIC_level1_HRB(evb_dir_modeling._dpc_VIC_level1_path)
    
    # merge
    dpc_VIC_level0.merge_grid_data()
    dpc_VIC_level1.merge_grid_data()
    
    # read domain
    domain_dataset = readDomain(evb_dir_modeling)

    # build parameters
    default_params_HRB = default_params
    default_params_HRB = set_g_params_soilGrids_layer(default_params_HRB)
    default_params_HRB["g_params"]["soil_layers_breakpoints"]["optimal"] = [2, 4]
    
    # build params_level0 with default params
    buildParam_level0_interface_instance = buildParam_level0(
        evb_dir_modeling,
        default_params_HRB["g_params"],
        SoilGrids_soillayerresampler,
        dpc_VIC_level0,
        TF_VIC_class=TF_VIC,
        buildParam_level0_interface_class=buildParam_level0_interface_HRB,
        reverse_lat=reverse_lat,
        stand_grids_lat_level0=None,
        stand_grids_lon_level0=None,
        rows_index_level0=None,
        cols_index_level0=None,
    )
    
    params_dataset_level0, stand_grids_lat_level0, stand_grids_lon_level0, rows_index_level0, cols_index_level0 = (
        buildParam_level0_interface_instance.params_dataset_level0, 
        buildParam_level0_interface_instance.stand_grids_lat_level0,
        buildParam_level0_interface_instance.stand_grids_lon_level0,
        buildParam_level0_interface_instance.rows_index_level0,
        buildParam_level0_interface_instance.cols_index_level0
    )
    
    # build params_level1
    buildParam_level1_interface_instance = buildParam_level1(
        evb_dir_modeling,
        dpc_VIC_level1,
        TF_VIC_class=TF_VIC,
        buildParam_level1_interface_class=buildParam_level1_interface_HRB,
        reverse_lat=reverse_lat,
        domain_dataset=domain_dataset,
        stand_grids_lat_level1=None,
        stand_grids_lon_level1=None,
        rows_index_level1=None,
        cols_index_level1=None,
    )
    
    params_dataset_level1, stand_grids_lat_level1, stand_grids_lon_level1, rows_index_level1, cols_index_level1 = (
        buildParam_level1_interface_instance.params_dataset_level1, 
        buildParam_level1_interface_instance.stand_grids_lat_level1,
        buildParam_level1_interface_instance.stand_grids_lon_level1,
        buildParam_level1_interface_instance.rows_index_level1,
        buildParam_level1_interface_instance.cols_index_level1,
    )
    
    # scaling_level0_to_level1
    params_dataset_level1, searched_grids_bool_index = scaling_level0_to_level1(
        params_dataset_level0, params_dataset_level1,
        searched_grids_bool_index=None,
        nlayer_list=[1, 2, 3],
        elev_scaling="Arithmetic_min",
    )
    
    # close
    domain_dataset.close()
    params_dataset_level0.close()
    params_dataset_level1.close()


def build_params_nested_HRB_basin_hierarchy(evb_dir_hydroanalysis, evb_dir_modeling, reverse_lat=True):
    # read dpc
    dpc_VIC_level0 = dataProcess_VIC_level0_HRB(evb_dir_modeling._dpc_VIC_level0_path)
    dpc_VIC_level1 = dataProcess_VIC_level1_HRB(evb_dir_modeling._dpc_VIC_level1_path)
    
    # merge
    dpc_VIC_level0.merge_grid_data()
    dpc_VIC_level1.merge_grid_data()
    
    # read domain
    domain_dataset = readDomain(evb_dir_modeling)  # main outlet
    
    # read grid_shp
    grid_shp_level0 = dpc_VIC_level0.get_data_from_cache("grid_shp")[0]
    grid_shp_level1 = dpc_VIC_level1.get_data_from_cache("grid_shp")[0]
    
    stand_grids_lat_level0, stand_grids_lon_level0 = createStand_grids_lat_lon_from_gridshp(
        grid_shp_level0, grid_res=None, reverse_lat=reverse_lat
    )
    
    stand_grids_lat_level1, stand_grids_lon_level1 = createStand_grids_lat_lon_from_gridshp(
        grid_shp_level1, grid_res=None, reverse_lat=reverse_lat
    )
    
    # read basins
    nested_basins_shp = gpd.read_file(os.path.join(evb_dir_hydroanalysis.Hydroanalysis_dir, "wbw_working_directory_level0", "basins_vector_outlets_with_reference.shp"))
    basin_shps = {
        "hanzhong": nested_basins_shp.iloc[0:1, :],
        "yangxian": nested_basins_shp.iloc[1:2, :],
        "lianghekou": nested_basins_shp.iloc[2:3, :],
        "shiquan": nested_basins_shp.iloc[3:4, :],
        "youshui": nested_basins_shp.iloc[4:5, :],
    }
    
    # enforce_unique_masks
    unique_masks_level1, grid_shp_unique_mask = cal_unique_mask_nested_basin(
        station_names,
        grid_shp_level1,
        basin_shps,
        main_basin_shp=basin_shps["shiquan"],
        plot=True
    )
    
    save_grid_shp_unique_mask_bool = False
    if save_grid_shp_unique_mask_bool:
        grid_shp_unique_mask.loc[:, ["geometry", "unique_mask_list"]].to_file(os.path.join(evb_dir_modeling.DomainFile_dir, "grid_shp_unique_mask_level1_grid.shp"))
        grid_shp_unique_mask.loc[:, ["point_geometry", "unique_mask_list"]].set_geometry("point_geometry").to_file(os.path.join(evb_dir_modeling.DomainFile_dir, "grid_shp_unique_mask_level1_center.shp"))

    # get subbasin_masks_level0
    masks_level0 = {}
    for station_name in station_names:
        lon_level1 = stand_grids_lon_level1
        lat_level1 = stand_grids_lat_level1
        lon_level0 = stand_grids_lon_level0
        lat_level0 = stand_grids_lat_level0
        mask_level1 = unique_masks_level1[station_name]
        
        mask_level0 = remap_level1_to_level0_mask(
            lon_level1, lat_level1, mask_level1,
            lon_level0, lat_level0
        )
        
        masks_level0[station_name] = mask_level0
        
    basin_hierarchy = {
        "station_names": station_names,
        "subbasin_masks_level0": masks_level0,
    }

    # build parameters
    default_params_HRB = default_params
    default_params_HRB = set_g_params_soilGrids_layer(default_params_HRB)
    default_params_HRB["g_params"]["soil_layers_breakpoints"]["optimal"] = [2, 4]
    g_params_expand = expand_station_wise_params(default_params_HRB["g_params"], station_num=len(station_names))
    
    for i, s in enumerate(station_names):
        g_params_expand[f"total_depths_{i}"]["optimal"] = [random.uniform(0.1, 4.0)]
        g_params_expand[f"soil_layers_breakpoints_{i}"]["optimal"] = [random.randint(1, 2), random.randint(3, 5)]
    
    default_params_HRB["g_params"] = g_params_expand
    
    # build params_level0 with default params
    buildParam_level0_interface_instance = buildParam_level0(
        evb_dir_modeling,
        default_params_HRB["g_params"],
        SoilGrids_soillayerresampler,
        dpc_VIC_level0,
        TF_VIC_class=TF_VIC,
        buildParam_level0_interface_class=buildParam_level0_interface_HRB,
        reverse_lat=reverse_lat,
        stand_grids_lat_level0=None,
        stand_grids_lon_level0=None,
        rows_index_level0=None,
        cols_index_level0=None,
        basin_hierarchy=basin_hierarchy,
    )
    
    params_dataset_level0, stand_grids_lat_level0, stand_grids_lon_level0, rows_index_level0, cols_index_level0 = (
        buildParam_level0_interface_instance.params_dataset_level0, 
        buildParam_level0_interface_instance.stand_grids_lat_level0,
        buildParam_level0_interface_instance.stand_grids_lon_level0,
        buildParam_level0_interface_instance.rows_index_level0,
        buildParam_level0_interface_instance.cols_index_level0
    )
    
    # build params_level1
    buildParam_level1_interface_instance = buildParam_level1(
        evb_dir_modeling,
        dpc_VIC_level1,
        TF_VIC_class=TF_VIC,
        buildParam_level1_interface_class=buildParam_level1_interface_HRB,
        reverse_lat=reverse_lat,
        domain_dataset=domain_dataset,
        stand_grids_lat_level1=None,
        stand_grids_lon_level1=None,
        rows_index_level1=None,
        cols_index_level1=None,
    )
    
    params_dataset_level1, stand_grids_lat_level1, stand_grids_lon_level1, rows_index_level1, cols_index_level1 = (
        buildParam_level1_interface_instance.params_dataset_level1, 
        buildParam_level1_interface_instance.stand_grids_lat_level1,
        buildParam_level1_interface_instance.stand_grids_lon_level1,
        buildParam_level1_interface_instance.rows_index_level1,
        buildParam_level1_interface_instance.cols_index_level1,
    )
    
    # scaling_level0_to_level1
    params_dataset_level1, searched_grids_bool_index = scaling_level0_to_level1(
        params_dataset_level0, params_dataset_level1,
        searched_grids_bool_index=None,
        nlayer_list=[1, 2, 3],
        elev_scaling="Arithmetic_min",
    )
    
    # close
    domain_dataset.close()
    params_dataset_level0.close()
    params_dataset_level1.close()


def build_params_HRB_spatially_uniform(evb_dir_modeling, reverse_lat=True, baseflow_scheme="ARNO"):
    # read dpc
    dpc_VIC_level0 = dataProcess_VIC_level0_HRB(evb_dir_modeling._dpc_VIC_level0_path)
    dpc_VIC_level1 = dataProcess_VIC_level1_HRB(evb_dir_modeling._dpc_VIC_level1_path)
    
    # merge
    dpc_VIC_level0.merge_grid_data()
    dpc_VIC_level1.merge_grid_data()
    
    # read domain
    domain_dataset = readDomain(evb_dir_modeling)

    # build parameters
    default_params_HRB = default_params
    
    if baseflow_scheme == "ARNO":
        default_params_HRB["g_params"] = g_params_ARNO_spatially_uniform_minimal
        buildParam_level0_interface_class = buildParam_level0_interface_ARNO_spatially_uniform_HRB
    elif baseflow_scheme == "Nijssen":
        default_params_HRB["g_params"] = g_params_Nijssen_spatially_uniform_minimal
        buildParam_level0_interface_class = buildParam_level0_interface_Nijssen_spatially_uniform_HRB
    
    default_params_HRB["g_params"] = set_g_params_soilGrids_layer(default_params_HRB["g_params"])
    default_params_HRB["g_params"]["soil_layers_breakpoints"]["optimal"] = [2, 4]
    default_params_HRB = set_default_params(default_params_HRB)
    
    # build params_level0 with default params
    buildParam_level0_interface_instance = buildParam_level0(
        evb_dir_modeling,
        default_params_HRB["g_params"],
        SoilGrids_soillayerresampler,
        dpc_VIC_level0,
        TF_VIC_class=TF_VIC,
        buildParam_level0_interface_class=buildParam_level0_interface_class,
        reverse_lat=reverse_lat,
        stand_grids_lat_level0=None,
        stand_grids_lon_level0=None,
        rows_index_level0=None,
        cols_index_level0=None,
    )
    
    params_dataset_level0, stand_grids_lat_level0, stand_grids_lon_level0, rows_index_level0, cols_index_level0 = (
        buildParam_level0_interface_instance.params_dataset_level0, 
        buildParam_level0_interface_instance.stand_grids_lat_level0,
        buildParam_level0_interface_instance.stand_grids_lon_level0,
        buildParam_level0_interface_instance.rows_index_level0,
        buildParam_level0_interface_instance.cols_index_level0
    )
    
    # build params_level1
    buildParam_level1_interface_instance = buildParam_level1(
        evb_dir_modeling,
        dpc_VIC_level1,
        TF_VIC_class=TF_VIC,
        buildParam_level1_interface_class=buildParam_level1_interface_HRB,
        reverse_lat=reverse_lat,
        domain_dataset=domain_dataset,
        stand_grids_lat_level1=None,
        stand_grids_lon_level1=None,
        rows_index_level1=None,
        cols_index_level1=None,
    )
    
    params_dataset_level1, stand_grids_lat_level1, stand_grids_lon_level1, rows_index_level1, cols_index_level1 = (
        buildParam_level1_interface_instance.params_dataset_level1, 
        buildParam_level1_interface_instance.stand_grids_lat_level1,
        buildParam_level1_interface_instance.stand_grids_lon_level1,
        buildParam_level1_interface_instance.rows_index_level1,
        buildParam_level1_interface_instance.cols_index_level1,
    )
    
    # scaling_level0_to_level1
    params_dataset_level1, searched_grids_bool_index = scaling_level0_to_level1(
        params_dataset_level0, params_dataset_level1,
        searched_grids_bool_index=None,
        nlayer_list=[1, 2, 3],
        elev_scaling="Arithmetic_min",
    )
    
    # close
    domain_dataset.close()
    params_dataset_level0.close()
    params_dataset_level1.close()    

if __name__ == "__main__":
    # build params based on default
    # build_params_HRB(evb_dir_modeling, reverse_lat)
    build_params_nested_HRB_basin_hierarchy(evb_dir_hydroanalysis, evb_dir_modeling, reverse_lat)
    # build_params_HRB_spatially_uniform(evb_dir_modeling, reverse_lat, baseflow_scheme="Nijssen")
    
    