# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from easy_vic_build.bulid_Domain import buildDomain
from easy_vic_build.tools.utilities import readdpc
from easy_vic_build.tools.dpc_func.dpc_subclass import dataProcess_VIC_level1
from easy_vic_build.Evb_dir_class import Evb_dir
from general_info import *

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
    dpc_VIC_level1 = readdpc(evb_dir.dpc_VIC_level1_path, dataProcess_VIC_level1)
    
    # build domain
    buildDomain(evb_dir, dpc_VIC_level1, reverse_lat)

if __name__ == "__main__":
    test()