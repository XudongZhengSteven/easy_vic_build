# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import sys
sys.path.append("../easy_vic_build")
from easy_vic_build import Evb_dir
from easy_vic_build.tools.utilities import *
from easy_vic_build.tools.plot_func.plot_func import *
from easy_vic_build.tools.hydroanalysis_func.hydroanalysis_for_BasinMap import *

"""
general information:

basin set
106(10_100_km_humid); 240(10_100_km_semi_humid); 648(10_100_km_semi_arid); 
213(100_1000_km_humid); 38(100_1000_km_semi_humid); 670(10_100_km_semi_arid);
397(1000_larger_km_humid); 636(1000_larger_km_semi_humid); 580(1000_larger_km_semi_arid) 

grid_res_level0=1km(0.00833)
grid_res_level1=3km(0.025), 6km(0.055), 8km(0.072), 12km(0.11)

""" 

def plot_basin_map():
    case_name = "397_6km"
    x_locator_interval = 0.3 # 0.1 # 0.3
    y_locator_interval = 0.2 # 0.1 # 0.2
    
    # set evb_dir
    evb_dir = Evb_dir()
    evb_dir.builddir(case_name)
    
    # read and build bool
    read_dpc_bool = True
    read_domain_dataset_bool = True
    read_params_bool = True
    read_BasinMap_bool = True
    
    hydroanalysis_for_basin_bool = False
    
    # hydroanalysis for BasinMap
    if hydroanalysis_for_basin_bool:
        hydroanalysis_for_basin(evb_dir)
        
    # read
    if read_dpc_bool:
        dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2 = readdpc(evb_dir)
    
    if read_domain_dataset_bool:
        domain_dataset = readDomain(evb_dir)
    
    if read_params_bool:
        params_dataset_level0, params_dataset_level1 = readParam(evb_dir, mode="r")
    
    if read_BasinMap_bool:
        stream_gdf = readBasinMap(evb_dir)
    
    fig_dict, ax_dict = plot_Basin_map(dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2, stream_gdf, x_locator_interval=x_locator_interval, y_locator_interval=y_locator_interval, fig=None, ax=None)
    fig_dict["fig_Basin_map"].savefig(os.path.join(evb_dir.BasinMap_dir, "fig_Basin_map.tiff"), dpi=300)
    fig_dict["fig_grid_basin_level0"].savefig(os.path.join(evb_dir.BasinMap_dir, "fig_grid_basin_level0.tiff"), dpi=300)
    fig_dict["fig_grid_basin_level1"].savefig(os.path.join(evb_dir.BasinMap_dir, "fig_grid_basin_level1.tiff"), dpi=300)
    fig_dict["fig_grid_basin_level2"].savefig(os.path.join(evb_dir.BasinMap_dir, "fig_grid_basin_level2.tiff"), dpi=300)
    
    # close
    if read_domain_dataset_bool:
        domain_dataset.close()
    
    if read_params_bool:
        params_dataset_level0.close()
        params_dataset_level1.close()


def plot_basin_map_location():
    # case_names
    cases_names = ["397_12km", "213_12km", "670_12km"]
    
    # set evb_dir
    evb_dir1 = Evb_dir()
    evb_dir1.builddir(cases_names[0])
    
    evb_dir2 = Evb_dir()
    evb_dir2.builddir(cases_names[1])
    
    evb_dir3 = Evb_dir()
    evb_dir3.builddir(cases_names[2])
    
    # read
    dpc_VIC_level0_case1, dpc_VIC_level1_case1, dpc_VIC_level2_case1 = readdpc(evb_dir1)
    dpc_VIC_level0_case2, dpc_VIC_level1_case2, dpc_VIC_level2_case2 = readdpc(evb_dir2)
    dpc_VIC_level0_case3, dpc_VIC_level1_case3, dpc_VIC_level2_case3 = readdpc(evb_dir3)
    
    # plot location
    fig, ax = plot_US_basemap(fig=None, ax=None, set_xyticks=False)
    dpc_VIC_level1_case1.basin_shp.plot(ax=ax, facecolor="#8ECFC9", edgecolor="k", linewidth=0.3, alpha=1)
    dpc_VIC_level1_case2.basin_shp.plot(ax=ax, facecolor="#FFBE7A", edgecolor="k", linewidth=0.3, alpha=1)
    dpc_VIC_level1_case3.basin_shp.plot(ax=ax, facecolor="#FA7F6F", edgecolor="k", linewidth=0.3, alpha=1)
    
    # save
    fig.savefig(os.path.join(Evb_dir.__package_dir__, "cases/fig_Basin_map_location.tiff"), dpi=300)
    
    # finer xticks
    fig, ax = plot_US_basemap(fig=None, ax=None, set_xyticks=True, x_locator_interval=1, y_locator_interval=1)
    dpc_VIC_level1_case1.basin_shp.plot(ax=ax, facecolor="#8ECFC9", edgecolor="k", linewidth=0.3, alpha=1)
    dpc_VIC_level1_case2.basin_shp.plot(ax=ax, facecolor="#FFBE7A", edgecolor="k", linewidth=0.3, alpha=1)
    dpc_VIC_level1_case3.basin_shp.plot(ax=ax, facecolor="#FA7F6F", edgecolor="k", linewidth=0.3, alpha=1)
    
    
if __name__ == "__main__":
    plot_basin_map()
    # plot_basin_map_location()
    