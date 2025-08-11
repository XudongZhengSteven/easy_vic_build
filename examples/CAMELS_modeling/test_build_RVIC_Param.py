# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from easy_vic_build.build_RVIC_Param import *
from easy_vic_build.Evb_dir_class import Evb_dir
from easy_vic_build.tools.utilities import readParam, readdpc, readDomain
from easy_vic_build.tools.routing_func.create_uh import createGUH
from easy_vic_build.tools.params_func.params_set import guh_params
from easy_vic_build.tools.dpc_func.dpc_subclass import dataProcess_VIC_level3
import geopandas as gpd

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
    
    # read params
    domain_dataset = readDomain(evb_dir)
    
    # read dpc
    dpc_VIC_level3 = readdpc(evb_dir.dpc_VIC_level3_path, dataProcess_VIC_level3)
    
    # read snaped outlet gdf
    snaped_outlet_gdf = gpd.read_file(os.path.join(
        evb_dir.Hydroanalysis_dir,
        "wbw_working_directory_level1",
        f"snaped_outlets_with_reference.shp"
    ))
    
    # read station id
    basin_shp = dpc_VIC_level3.get_data_from_cache("basin_shp")[0]
    station_id = basin_shp["hru_id"].values[0]
    
    # build RVICParam_general
    buildRVICParam_basic(evb_dir, domain_dataset,
                           ppf_kwargs={
                               "names": [station_id],
                               "lons": [snaped_outlet_gdf.geometry.x.values[0]],
                               "lats": [snaped_outlet_gdf.geometry.y.values[0]]
                            },
                           
                           uh_params={
                               "createUH_func": createGUH,
                               "uh_dt": 3600,
                               "tp": guh_params["tp"]["default"][0],
                               "mu": guh_params["mu"]["default"][0],
                               "m": guh_params["m"]["default"][0],
                               "plot_bool": True, "max_day":None, "max_day_range": (0, 10), "max_day_converged_threshold": 0.001
                            },
                           
                           cfg_params={
                               "VELOCITY": 1.5,
                               "DIFFUSION": 800.0,
                               "OUTPUT_INTERVAL": 86400,
                               "SUBSET_DAYS": 10,
                               "CELL_FLOWDAYS": 2,
                               "BASIN_FLOWDAYS": 50
                           }
                        )
    
    # close
    domain_dataset.close()
    
    
if __name__ == "__main__":
    test()
