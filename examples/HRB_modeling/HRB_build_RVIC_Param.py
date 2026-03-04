# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from general_info import *
from easy_vic_build.build_RVIC_Param import buildRVICParam_basic
from easy_vic_build.tools.utilities import *
from easy_vic_build.tools.routing_func.create_uh import createGUH
from easy_vic_build.tools.params_func.params_set import *
from HRB_build_dpc import dataProcess_VIC_level3_HRB

def HRB_build_RVIC_Param(evb_dir_modeling):
    # read params
    domain_dataset = readDomain(evb_dir_modeling)
    
    # read outlets
    dpc_VIC_level3 = dataProcess_VIC_level3_HRB(
            load_path=evb_dir_modeling._dpc_VIC_level3_path,
            reset_on_load_failure=False,
    )
    
    gauge_info = dpc_VIC_level3.get_data_from_cache("gauge_info")[0]
    
    station_lons = [gauge_info[name]["gauge_coord(lon, lat)_level1"][0] for name in station_names]
    station_lats = [gauge_info[name]["gauge_coord(lon, lat)_level1"][1] for name in station_names]
    
    # read snaped outlet gdf
    # station_id = basin_outlets_reference_i_map[station_name]
    # snaped_outlet_gdf = gpd.read_file(os.path.join(
    #     evb_dir_modeling.Hydroanalysis_dir,
    #     "wbw_working_directory_level1",
    #     f"snaped_outlet_with_reference_{station_id}.shp"
    # ))
    
    # build RVICParam_general
    buildRVICParam_basic(evb_dir_modeling, domain_dataset,
                           ppf_kwargs={
                               "names": [station_name],
                               "lons": station_lons, # [snaped_outlet_gdf.geometry.x.values[0]],
                               "lats": station_lats, # [snaped_outlet_gdf.geometry.y.values[0]]
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
    
    domain_dataset.close()
    
if __name__ == "__main__":
    HRB_build_RVIC_Param(evb_dir_modeling)
    