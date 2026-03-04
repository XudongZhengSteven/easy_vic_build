# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
from general_info import *
from easy_vic_build.bulid_Domain import buildDomain, addElevIntoDomain
from easy_vic_build.tools.utilities import readParam
from HRB_build_dpc import dataProcess_VIC_level0_HRB, dataProcess_VIC_level1_HRB, dataProcess_VIC_level3_HRB


def build_domain_HRB(evb_dir_modeling, reverse_lat=True):    
    # read dpc_VIC_level
    dpc_VIC_level0 = dataProcess_VIC_level0_HRB(evb_dir_modeling._dpc_VIC_level0_path)
    dpc_VIC_level1 = dataProcess_VIC_level1_HRB(evb_dir_modeling._dpc_VIC_level1_path)
    dpc_VIC_level3 = dataProcess_VIC_level3_HRB(evb_dir_modeling._dpc_VIC_level3_path)
    basin_shps = dpc_VIC_level3.get_data_from_cache("basin_shps")[0]
    
    # build domain for each station
    for station_name in station_names:
        basin_shp = basin_shps[station_name]
        
        # level1
        evb_dir_modeling.domainFile_path = os.path.join(evb_dir_modeling.DomainFile_dir, f"domain_{station_name}.nc")
        buildDomain(
            evb_dir_modeling,
            dpc_VIC_level1,
            reverse_lat,
            basin_shp
        )

def add_elev_into_domain(evb_dir_modeling):
    # read params
    _, params_dataset_level1 = readParam(evb_dir_modeling)
    
    # add elev into domain
    addElevIntoDomain(evb_dir_modeling, params_dataset_level1)
    
    
if __name__ == "__main__":
    # build domain
    build_domain_HRB(evb_dir_modeling, reverse_lat)
    
    # add elev into domain
    # add_elev_into_domain(evb_dir_modeling)
    