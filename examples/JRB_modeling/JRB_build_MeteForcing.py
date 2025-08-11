# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from general_info import *
from JRB_build_dpc import (dataProcess_VIC_level0_JRB, dataProcess_VIC_level1_JRB, dataProcess_VIC_level2_CMADSV1_JRB, dataProcess_VIC_level2_CMFD_JRB, 
                           dataProcess_VIC_level2_CDMet_JRB, dataProcess_VIC_level3_JRB)
from JRB_build_evb_dir import build_modeling_dir
from easy_vic_build.tools.mete_func.build_MeteForcing_interface import buildMeteForcing_interface
from easy_vic_build import build_MeteForcing


def JRB_build_MeteForcing():
    # build evb
    evb_dir_modeling = build_modeling_dir(subname=f"{station_name}_{model_scale}")

    # read dpc_VIC_level2
    # dpc_VIC_level2_CMADSV1 = dataProcess_VIC_level2_CMADSV1_JRB(evb_dir_modeling._dpc_VIC_level2_path.replace(".pkl", "_CMADSV1.pkl"))
    # dpc_VIC_level2_CMFD = dataProcess_VIC_level2_CMFD_JRB(evb_dir_modeling._dpc_VIC_level2_path.replace(".pkl", "_CMFD.pkl"))
    dpc_VIC_level2_CDMet = dataProcess_VIC_level2_CDMet_JRB(evb_dir_modeling._dpc_VIC_level2_path.replace(".pkl", "_CDMet.pkl"))
    
    # merge
    # dpc_VIC_level2_CMADSV1.merge_grid_data()
    # dpc_VIC_level2_CMFD.merge_grid_data()
    dpc_VIC_level2_CDMet.merge_grid_data()
    
    # build
    buildMeteForcing_interface_instance = build_MeteForcing.buildMeteForcing(
        evb_dir_modeling,
        dpc_VIC_level2_CDMet,
        date_period,
        date_period,
        buildMeteForcing_interface,
        reverse_lat=True,
        stand_grids_lat_level2=None,
        stand_grids_lon_level2=None,
        rows_index_level2=None,
        cols_index_level2=None,
        file_format="NETCDF4",
    )
    

if __name__ == "__main__":
    JRB_build_MeteForcing()