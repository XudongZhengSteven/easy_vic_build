# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from general_info import *
from HRB_build_dpc import dataProcess_VIC_level2_CMFD_HRB
from easy_vic_build.tools.mete_func.build_MeteForcing_interface import buildMeteForcing_interface
from easy_vic_build import build_MeteForcing


def HRB_build_MeteForcing(evb_dir_modeling):
    # read dpc_VIC_level2
    dpc_VIC_level2_CMFD = dataProcess_VIC_level2_CMFD_HRB(evb_dir_modeling._dpc_VIC_level2_path.replace(".pkl", "_CMFD.pkl"))
    
    # merge
    dpc_VIC_level2_CMFD.merge_grid_data()
    
    # build
    buildMeteForcing_interface_instance = build_MeteForcing.buildMeteForcing(
        evb_dir_modeling,
        dpc_VIC_level2_CMFD,
        date_period,
        date_period,
        buildMeteForcing_interface,
        reverse_lat=True,
        stand_grids_lat_level2=None,
        stand_grids_lon_level2=None,
        rows_index_level2=None,
        cols_index_level2=None,
        file_format="NETCDF4",
        timestep=timestep,
    )
    

if __name__ == "__main__":
    HRB_build_MeteForcing(evb_dir_modeling)