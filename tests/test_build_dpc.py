# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from easy_vic_build.tools.dpc_func.dpc_subclass import dataProcess_VIC_level0, dataProcess_VIC_level1, dataProcess_VIC_level2, dataProcess_VIC_level3
from easy_vic_build.Evb_dir_class import Evb_dir
from easy_vic_build.tools.utilities import *
from easy_vic_build.tools.dpc_func.basin_grid_func import createGridForBasin
from general_info import *
import matplotlib.pyplot as plt

"""
general information:

basin set
106(10_100_km_humid); 240(10_100_km_semi_humid); 648(10_100_km_semi_arid); 
213(100_1000_km_humid); 38(100_1000_km_semi_humid); 670(10_100_km_semi_arid);
397(1000_larger_km_humid); 636(1000_larger_km_semi_humid); 580(1000_larger_km_semi_arid) 

grid_res_level0=1km(0.00833)
grid_res_level1=3km(0.025), 6km(0.055), 8km(0.072), 12km(0.11)

""" 

def build_basin_shp(basin_index):
    # read shpfile and get basin_shp (Basins)
    basin_shp_all, basin_shp = read_one_HCDN_basin_shp(basin_index)
    return basin_shp_all, basin_shp


def build_grid_shp(
    basin_shp,
    grid_res_level0,
    grid_res_level1,
    grid_res_level2,
    expand_grids_num=1,
    plot=False,
):
    # build grid_shp (Grids) for level1 (modeling scale), expand_grids_num=1 to avoid 0 (edge) flow direction in hydroanalysis
    grid_shp_lon_level1, grid_shp_lat_level1, grid_shp_level1 = createGridForBasin(basin_shp, grid_res_level1, expand_grids_num=expand_grids_num)
    _, _, _, boundary_grids_edge_x_y_level1 = grid_shp_level1.createBoundaryShp()
    
    # build grid_shp for level0 and level2 based on the boundary of level1
    grid_shp_lon_level0, grid_shp_lat_level0, grid_shp_level0 = createGridForBasin(basin_shp, grid_res_level0, boundary=boundary_grids_edge_x_y_level1)
    grid_shp_lon_level2, grid_shp_lat_level2, grid_shp_level2 = createGridForBasin(basin_shp, grid_res_level2, boundary=boundary_grids_edge_x_y_level1)
    
    # build grid_shp for level3 based on the shp file
    grid_shp_lon_level3, grid_shp_lat_level3, grid_shp_level3 = createGridForBasin(basin_shp, None, boundary=boundary_grids_edge_x_y_level1)
    
    # plot
    if plot:
        fig, axes = plt.subplots(1, 4)
        basin_shp.plot(ax=axes[0], edgecolor="k", alpha=0.5, facecolor="b")
        grid_shp_level0.plot(ax=axes[0], alpha=0.5, edgecolor="k", linewidth=0.5)
        
        basin_shp.plot(ax=axes[1], edgecolor="k", alpha=0.5, facecolor="b")
        grid_shp_level1.plot(ax=axes[1], alpha=0.5, edgecolor="k", linewidth=0.5)
        grid_shp_level1.point_geometry.plot(ax=axes[1], alpha=0.5, color="darkblue", markersize=1)
        
        basin_shp.plot(ax=axes[2], edgecolor="k", alpha=0.5, facecolor="b")
        grid_shp_level2.plot(ax=axes[2], alpha=0.5, edgecolor="k", linewidth=0.5)
        grid_shp_level2.point_geometry.plot(ax=axes[2], alpha=0.5, color="darkblue", markersize=1)
        
        basin_shp.plot(ax=axes[3], edgecolor="k", alpha=0.5, facecolor="b")
        plt.show(block=True)
        
    return grid_shp_level0, grid_shp_level1, grid_shp_level2, grid_shp_level3


def test():
    # general set
    case_name = f"{basin_index}_{model_scale}"
    
    # build dir
    evb_dir = Evb_dir(cases_home="./examples")
    evb_dir.builddir(case_name)
    
    # build basin shp
    basin_shp_all, basin_shp = build_basin_shp(basin_index)
    
    # build grid shp
    grid_shp_level0, grid_shp_level1, grid_shp_level2, grid_shp_level3 = build_grid_shp(
        basin_shp,
        grid_res_level0,
        grid_res_level1,
        grid_res_level2,
        expand_grids_num=1,
        plot=True
    )
    
    # build dpc level0 (need to re-build for each scale, as the grid resolution is different, the cover area might be different)
    build_dpc_VIC_level0 = False
    if build_dpc_VIC_level0:
        dpc_VIC_level0 = dataProcess_VIC_level0(
            load_path=evb_dir._dpc_VIC_level0_path,
            reset_on_load_failure=True,
        )
        
        dpc_VIC_level0.loaddata_pipeline(
            save_path=evb_dir._dpc_VIC_level0_path,
            loaddata_kwargs={
                "basin_shp": basin_shp,
                "grid_shp": grid_shp_level0,
                "grid_res": grid_res_level0,
            }
        )
        
        dpc_VIC_level0.plot()
        dpc_VIC_level0.save_state(evb_dir._dpc_VIC_level0_path)
    
    # build dpc level1
    build_dpc_VIC_level1 = False
    if build_dpc_VIC_level1:
        dpc_VIC_level1 = dataProcess_VIC_level1(
            load_path=evb_dir._dpc_VIC_level1_path,
            reset_on_load_failure=True,
        )
        
        dpc_VIC_level1.loaddata_pipeline(
            save_path=evb_dir._dpc_VIC_level1_path,
            loaddata_kwargs={
                "basin_shp": basin_shp,
                "grid_shp": grid_shp_level1,
                "grid_res": grid_res_level1,
                "date_period": date_period,
                "evb_dir": evb_dir,
                "reverse_lat": reverse_lat,
                "search_method_st": "radius_rectangle_reverse",  # src: 0.1 deg ~= 11.1km
                "search_method_annual_P": "radius_rectangle_reverse",  # src: 0.125 deg ~= 13.875km
            }
        )
        
        dpc_VIC_level1.plot()
        plt.show(block=True)
        dpc_VIC_level1.save_state(evb_dir._dpc_VIC_level1_path)
        
    # build dpc level2
    build_dpc_VIC_level2 = False
    if build_dpc_VIC_level2:
        dpc_VIC_level2 = dataProcess_VIC_level2(
            load_path=evb_dir._dpc_VIC_level2_path,
            reset_on_load_failure=True,
        )
        
        dpc_VIC_level2.loaddata_pipeline(
            save_path=evb_dir._dpc_VIC_level2_path,
            loaddata_kwargs={
                "basin_shp": basin_shp,
                "grid_shp": grid_shp_level2,
                "grid_res": grid_res_level2,
                "date_period":  date_period,
                "reverse_lat": reverse_lat,
                "search_method": "radius_rectangle_reverse",  # src: 0.125 deg ~= 13.875km
            }
        )
        
        dpc_VIC_level2.save_state(evb_dir._dpc_VIC_level2_path)
        
    # build dpc level3
    build_dpc_VIC_level3 = False
    if build_dpc_VIC_level3:
        dpc_VIC_level3 = dataProcess_VIC_level3(
            load_path=evb_dir._dpc_VIC_level3_path,
            reset_on_load_failure=True,
        )

        dpc_VIC_level3.loaddata_pipeline(
            save_path=evb_dir._dpc_VIC_level3_path,
            loaddata_kwargs={
                "basin_shp": basin_shp,
                "date_period": date_period,
                "k_list": ["camels_topo"],
            }
        )

        dpc_VIC_level3.save_state(evb_dir._dpc_VIC_level3_path)
        

if __name__ == "__main__":
    test()