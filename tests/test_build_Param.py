# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from general_info import *
from easy_vic_build.tools.utilities import readdpc, readDomain
from easy_vic_build.tools.dpc_func.dpc_subclass import dataProcess_VIC_level0, dataProcess_VIC_level1
from easy_vic_build.Evb_dir_class import Evb_dir
from easy_vic_build.build_Param import buildParam_level0, buildParam_level1
from easy_vic_build.build_Param import scaling_level0_to_level1
from easy_vic_build.tools.params_func.params_set import default_params
from easy_vic_build.tools.dpc_func.extractData_func.Extract_CONUS_SOIL import CONUS_soillayerresampler
from easy_vic_build.tools.params_func.TransferFunction import TF_VIC
from easy_vic_build.tools.params_func.build_Param_interface import buildParam_level0_interface, buildParam_level1_interface

"""
general information:

basin set
106(10_100_km_humid); 240(10_100_km_semi_humid); 648(10_100_km_semi_arid); 
213(100_1000_km_humid); 38(100_1000_km_semi_humid); 670(10_100_km_semi_arid);
397(1000_larger_km_humid); 636(1000_larger_km_semi_humid); 580(1000_larger_km_semi_arid) 

grid_res_level0=1km(0.00833)
grid_res_level1=3km(0.025), 6km(0.055), 8km(0.072), 12km(0.11)

""" 


def test():
    # general set
    case_name = f"{basin_index}_{model_scale}"
    
    # build dir
    evb_dir = Evb_dir(cases_home="./examples")
    evb_dir.builddir(case_name)
    
    # read dpc
    dpc_VIC_level0 = readdpc(evb_dir.dpc_VIC_level0_path, dataProcess_VIC_level0)
    dpc_VIC_level1 = readdpc(evb_dir.dpc_VIC_level1_path, dataProcess_VIC_level1)
    
    # merge
    dpc_VIC_level0.merge_grid_data()
    dpc_VIC_level1.merge_grid_data()
    
    # read domain
    domain_dataset = readDomain(evb_dir)

    # build parameters
    # build params_level0 with default params
    buildParam_level0_interface_instance = buildParam_level0(
        evb_dir,
        default_params["g_params"],
        CONUS_soillayerresampler,
        dpc_VIC_level0,
        TF_VIC_class=TF_VIC,
        buildParam_level0_interface_class=buildParam_level0_interface,
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
        evb_dir,
        dpc_VIC_level1,
        TF_VIC_class=TF_VIC,
        buildParam_level1_interface_class=buildParam_level1_interface,
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
    )
    
    # close
    domain_dataset.close()
    params_dataset_level0.close()
    params_dataset_level1.close()


if __name__ == "__main__":
    test()
    