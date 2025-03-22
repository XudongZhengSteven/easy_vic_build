# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from easy_vic_build.build_RVIC_Param import *
from easy_vic_build.build_dpc import readdpc
from easy_vic_build.Evb_dir_class import Evb_dir
from easy_vic_build.tools.utilities import readParam, readDomain, remove_and_mkdir
from easy_vic_build.tools.params_func.params_set import default_uh_params

"""
general information:

basin set
106(10_100_km_humid); 240(10_100_km_semi_humid); 648(10_100_km_semi_arid); 
213(100_1000_km_humid); 38(100_1000_km_semi_humid); 670(10_100_km_semi_arid);
397(1000_larger_km_humid); 636(1000_larger_km_semi_humid); 580(1000_larger_km_semi_arid) 

grid_res_level0=1km(0.00833)
grid_res_level1=3km(0.025), 6km(0.055), 8km(0.072), 12km(0.11)

""" 

scalemap = {"3km": 0.025, "6km": 0.055, "8km": 0.072, "12km": 0.11}

if __name__ == "__main__":
    basin_index = 213
    model_scale = "6km"
    date_period = ["19980101", "19981231"]
    case_name = f"{basin_index}_{model_scale}"
    
    # build dir
    evb_dir = Evb_dir("./examples")  # cases_home="/home/xdz/code/VIC_xdz/cases"
    evb_dir.builddir(case_name)
    
    # remake RVICParam_dir
    # remove_and_mkdir(evb_dir.RVICParam_dir)
    # evb_dir.builddir(case_name)
    
    # read dpc
    dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2 = readdpc(evb_dir)
    
    # read params
    params_dataset_level0, params_dataset_level1 = readParam(evb_dir)
    
    # build RVICParam_general
    buildRVICParam_general(evb_dir, dpc_VIC_level1, params_dataset_level1,
                           ppf_kwargs=dict(), uh_params={"createUH_func": create_uh.createGUH, "uh_dt": 3600,
                                                         "tp": default_uh_params[0], "mu": default_uh_params[1], "m": default_uh_params[2],
                                                         "plot_bool": True, "max_day":None, "max_day_range": (0, 10), "max_day_converged_threshold": 0.001},
                           cfg_params={"VELOCITY": 1.5, "DIFFUSION": 800.0, "OUTPUT_INTERVAL": 86400, "SUBSET_DAYS": 10, "CELL_FLOWDAYS": 2, "BASIN_FLOWDAYS": 50})
    
    # buildRVICParam: note, rvic should be installed
    # buildRVICParam(evb_dir, dpc_VIC_level1, params_dataset_level1,
    #                ppf_kwargs=dict(), uh_params={"createUH_func": create_uh.createGUH, "uh_dt": 3600,
    #                                              "tp": default_uh_params[0], "mu": default_uh_params[1], "m": default_uh_params[2],
    #                                              "plot_bool": True, "max_day":None, "max_day_range": (0, 10), "max_day_converged_threshold": 0.001},
    #                cfg_params={"VELOCITY": 1.5, "DIFFUSION": 800.0, "OUTPUT_INTERVAL": 86400, "SUBSET_DAYS": 10, "CELL_FLOWDAYS": 2, "BASIN_FLOWDAYS": 50})
    
    
    # close
    params_dataset_level0.close()
    params_dataset_level1.close()
