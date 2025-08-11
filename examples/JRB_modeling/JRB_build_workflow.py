# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com


from general_info import *
from JRB_build_dpc import (dataProcess_VIC_level0_JRB, dataProcess_VIC_level1_JRB, dataProcess_VIC_level2_CMADSV1_JRB, dataProcess_VIC_level2_CMFD_JRB, 
                           dataProcess_VIC_level2_CDMet_JRB, dataProcess_VIC_level3_JRB)
from JRB_build_evb_dir import build_modeling_dir
from JRB_build_dpc import build_dpc_VIC_JRB
from JRB_build_domain import build_domain_JRB
from JRB_build_Param import build_params_JRB
from JRB_hydroanalysis import hydroanalysis_level1_JRB
from JRB_plot_Basin_map import plot_basin_map_JRB

def JRB_build_workflow(build_dpc=False):
    # hydroanalysis
    evb_dir_hydroanalysis = build_modeling_dir(subname="hydroanalysis")
    
    # build dpc
    if build_dpc:
        build_dpc_VIC_JRB(evb_dir_hydroanalysis, station_name, model_scale, date_period)
    
    # build domain
    build_domain_JRB(model_scale, station_name, reverse_lat)
    
    # build params
    build_params_JRB(station_name, model_scale, reverse_lat)
    
    # hydroanalysis at level1
    hydroanalysis_level1_JRB(station_name, model_scale, reverse_lat)
    
    # plot basin map
    plot_basin_map_JRB(evb_dir_hydroanalysis, model_scale, station_name)
    
    
if __name__ == "__main__":
    JRB_build_workflow()


