# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import sys
sys.path.append("../easy_vic_build")
from easy_vic_build.build_RVIC_Param import buildRVICParam
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
    dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2 = readdpc(evb_dir)

    # read domain
    domain_dataset = readDomain(evb_dir)
    
    # read params
    params_dataset_level0, params_dataset_level1 = readParam(evb_dir)
    
    # set arcpy_python_path
    evb_dir.arcpy_python_path = "C:\\Python27\\ArcGIS10.5\\python.exe"
    
    # build RVIC_Param
    buildRVICParam(dpc_VIC_level1, evb_dir, params_dataset_level1, domain_dataset, reverse_lat=True, stream_acc_threshold=100.0,
                   ppf_kwargs=dict(), uh_params={"tp": 1.4, "mu": 5.0, "m": 3.0}, uh_plot_bool=True,
                   cfg_params={"VELOCITY": 1.5, "DIFFUSION": 800.0, "OUTPUT_INTERVAL": 86400, "RVIC_input_name": "fluxes.nc"})
