# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import sys
sys.path.append("../easy_vic_build")
from easy_vic_build.build_dpc import builddpc
from easy_vic_build import Evb_dir

"""
general information:

basin set
106(10_100_km_humid); 240(10_100_km_semi_humid); 648(10_100_km_semi_arid); 
213(100_1000_km_humid); 38(100_1000_km_semi_humid); 670(10_100_km_semi_arid);
397(1000_larger_km_humid); 636(1000_larger_km_semi_humid); 580(1000_larger_km_semi_arid) 

grid_res_level0=1km(0.00833)
grid_res_level1=3km(0.025), 6km(0.055), 12km(0.11)

""" 


if __name__ == "__main__":
    basin_index = 106
    date_period = ["19980101", "20101231"]
    case_name = "106_3km"
    
    # build dir
    evb_dir = Evb_dir()
    evb_dir.builddir(case_name)
    
    # build dpc
    dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2 = builddpc(evb_dir, basin_index, date_period,
                                                              grid_res_level0=0.00833, grid_res_level1=0.025, grid_res_level2=0.125,
                                                              dpc_VIC_level0_call_kwargs={"readGriddata": True},
                                                              dpc_VIC_level1_call_kwargs={"readBasindata": True, "readGriddata": True, "readBasinAttribute": True})