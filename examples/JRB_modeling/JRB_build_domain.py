# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from general_info import *
from JRB_build_evb_dir import build_modeling_dir
from easy_vic_build.bulid_Domain import buildDomain
from JRB_build_dpc import dataProcess_VIC_level1_JRB


def build_domain_JRB(model_scale, station_name, reverse_lat=True):
    # build evb
    evb_dir_modeling = build_modeling_dir(subname=f"{station_name}_{model_scale}")
    
    # read dpc_VIC_level1
    dpc_VIC_level1 = dataProcess_VIC_level1_JRB(evb_dir_modeling._dpc_VIC_level1_path)
    
    # build domain
    buildDomain(evb_dir_modeling, dpc_VIC_level1, reverse_lat)
    

if __name__ == "__main__":
    # build domain
    build_domain_JRB(model_scale, station_name, reverse_lat)
    