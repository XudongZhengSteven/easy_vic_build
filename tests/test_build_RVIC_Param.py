# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import sys
sys.path.append("../easy_vic_build")
from easy_vic_build.build_RVIC_Param import *
from easy_vic_build.build_dpc import readdpc
from easy_vic_build import Evb_dir
from easy_vic_build.tools.utilities import readParam, readDomain

"""
general information:

basin set
106(10_100_km_humid); 240(10_100_km_semi_humid); 648(10_100_km_semi_arid); 
213(100_1000_km_humid); 38(100_1000_km_semi_humid); 670(10_100_km_semi_arid);
397(1000_larger_km_humid); 636(1000_larger_km_semi_humid); 580(1000_larger_km_semi_arid) 

grid_res_level0=1km(0.00833)
grid_res_level1=3km(0.025), 6km(0.055), 8km(0.072), 12km(0.11)

""" 

if __name__ == "__main__":
    basin_index = 636
    date_period = ["19980101", "20101231"]
    case_name = "636_12km"
    
    # build dir
    evb_dir = Evb_dir()
    evb_dir.builddir(case_name)
    
    # read dpc
    dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2 = readdpc(evb_dir)

    # read domain
    domain_dataset = readDomain(evb_dir)
    
    # read params
    params_dataset_level0, params_dataset_level1 = readParam(evb_dir)
    
    # set arcpy_python_path
    evb_dir.arcpy_python_path = "C:\\Python27\\ArcGIS10.5\\python.exe"
    
    # build RVIC_Param
    buildRVICParam_general(evb_dir, dpc_VIC_level1, params_dataset_level1, domain_dataset,
                           flowdirection_kwargs={"reverse_lat": True, "stream_acc_threshold": 100.0, "flow_direction_pkg": "wbw", "crs_str": "EPSG:4326"},
                           ppf_kwargs=dict(), uh_params={"createUH_func": create_uh.createGUH, "uh_dt": 3600, "tp": 1.4, "mu": 5.0, "m": 3.0, "plot_bool": True, "max_day":None, "max_day_range": (0, 10), "max_day_converged_threshold": 0.001},
                           cfg_params={"VELOCITY": 1.5, "DIFFUSION": 800.0, "OUTPUT_INTERVAL": 86400, "SUBSET_DAYS": 10, "CELL_FLOWDAYS": 2, "BASIN_FLOWDAYS": 50})
    
    # close
    domain_dataset.close()
    params_dataset_level0.close()
    params_dataset_level1.close()
