# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import sys
sys.path.append("../easy_vic_build")
from easy_vic_build.build_dpc import builddpc
from easy_vic_build.tools.dpc_func.dpc_subclass import dataProcess_VIC_level0, dataProcess_VIC_level1, dataProcess_VIC_level2
from easy_vic_build import Evb_dir
from easy_vic_build.tools.utilities import *
from easy_vic_build.tools.dpc_func.basin_grid_func import createGridForBasin

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
    basin_index = 670
    date_period = ["19980101", "20101231"]
    case_name = "670_6km"
    grid_res_level0=0.00833
    grid_res_level1=0.055
    grid_res_level2=0.125
    
    # build dir
    evb_dir = Evb_dir()
    evb_dir.builddir(case_name)
    
    # read shpfile and get basin_shp (Basins)
    basin_shp_all, basin_shp = read_one_HCDN_basin_shp(basin_index)
    
    # build grid_shp (Grids)
    grid_shp_lon_level0, grid_shp_lat_level0, grid_shp_level0 = createGridForBasin(basin_shp, grid_res_level0)
    grid_shp_lon_level1, grid_shp_lat_level1, grid_shp_level1 = createGridForBasin(basin_shp, grid_res_level1)
    grid_shp_lon_level2, grid_shp_lat_level2, grid_shp_level2 = createGridForBasin(basin_shp, grid_res_level2)
    
    # build dpc
    dpc_VIC_level0 = dataProcess_VIC_level0(basin_shp, grid_shp_level0, grid_res_level0, date_period)
    dpc_VIC_level1 = dataProcess_VIC_level1(basin_shp, grid_shp_level1, grid_res_level1, date_period)
    dpc_VIC_level2 = dataProcess_VIC_level2(basin_shp, grid_shp_level2, grid_res_level2, date_period)
    
    dpc_VIC_level0_call_kwargs={"readBasindata": False, "readGriddata": True, "readBasinAttribute": False}
    dpc_VIC_level1_call_kwargs={"readBasindata": True, "readGriddata": True, "readBasinAttribute": True}
    dpc_VIC_level2_call_kwargs={"readBasindata": False, "readGriddata": False, "readBasinAttribute": False}
    plot_columns_level0 = ["SrtmDEM_mean_Value", "soil_l1_sand_nearest_Value"]
    plot_columns_level1 = ["annual_P_in_src_grid_Value", "umd_lc_major_Value"]
    
    builddpc(evb_dir, dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2,
             dpc_VIC_level0_call_kwargs, dpc_VIC_level1_call_kwargs, dpc_VIC_level2_call_kwargs,
             plot_columns_level0, plot_columns_level1)

if __name__ == "__main__":
    test()