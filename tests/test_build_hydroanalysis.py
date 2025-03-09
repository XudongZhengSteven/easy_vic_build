# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from easy_vic_build.build_RVIC_Param import *
from easy_vic_build import Evb_dir
from easy_vic_build.tools.utilities import readParam, readDomain
from easy_vic_build.build_hydroanalysis import buildHydroanalysis
from easy_vic_build.tools.utilities import remove_and_mkdir

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
    basin_index = 213
    model_scale = "6km"
    date_period = ["19980101", "19981231"]
    case_name = f"{basin_index}_{model_scale}"
    
    # build dir
    evb_dir = Evb_dir(cases_home="./examples")  # cases_home="/home/xdz/code/VIC_xdz/cases"
    evb_dir.builddir(case_name)
    remove_and_mkdir(evb_dir.RVICParam_dir)
    evb_dir.builddir(case_name)

    # read domain
    domain_dataset = readDomain(evb_dir)
    
    # read params
    params_dataset_level0, params_dataset_level1 = readParam(evb_dir)
    
    # set arcpy_python_path
    # evb_dir.arcpy_python_path = "C:\\Python27\\ArcGIS10.5\\python.exe"
    
    # build Hydroanalysis
    buildHydroanalysis(evb_dir, params_dataset_level1, domain_dataset, reverse_lat=True, flow_direction_pkg="wbw", crs_str="EPSG:4326",
                       create_stream=True,
                       pourpoint_lon=None, pourpoint_lat=None, pourpoint_direction_code=None)
    
    # close
    domain_dataset.close()
    params_dataset_level0.close()
    params_dataset_level1.close()