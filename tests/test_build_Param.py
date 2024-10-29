# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import sys
sys.path.append("../easy_vic_build")
from easy_vic_build.tools.utilities import readdpc
from easy_vic_build import Evb_dir
from easy_vic_build.bulid_Param import buildParam_level0, buildParam_level1
from easy_vic_build.bulid_Param import scaling_level0_to_level1
from easy_vic_build.tools.calibrate_func.params_set import default_g_list, g_boundary

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
    
    # read dpc
    dpc_VIC_level0, dpc_VIC_level1, dpc_level2 = readdpc(evb_dir)

    # build parameters
    # build params_level0 with default params
    default_g_list, g_boundary =  get_default_g_list()
    params_dataset_level0 = buildParam_level0(default_g_list, dpc_VIC_level0, evb_dir, reverse_lat=True)
    
    # build params_level1
    params_dataset_level1 = buildParam_level1(dpc_VIC_level1, evb_dir, reverse_lat=True)
    
    # scaling_level0_to_level1
    params_dataset_level1 = scaling_level0_to_level1(params_dataset_level0, params_dataset_level1)
    
    # close
    params_dataset_level0.close()
    params_dataset_level1.close()
    
    