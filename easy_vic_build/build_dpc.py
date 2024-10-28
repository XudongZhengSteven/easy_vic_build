# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
from .dataPreprocess.basin_grid import read_one_basin_shp, createGridForBasin
from .dataPreprocess.dpc_subclass import dataProcess_VIC_level0, dataProcess_VIC_level1, dataProcess_VIC_level2
from .tools.utilities import *
import pickle
# you can set your own dataProcess_VIC_level0, dataProcess_VIC_level1, dataProcess_VIC_level2

def builddpc(evb_dir, basin_index, date_period,
             grid_res_level0=0.00833, grid_res_level1=0.025, grid_res_level2=0.125,
             dpc_VIC_level0_call_kwargs=dict(), dpc_VIC_level1_call_kwargs=dict(),
             plot_columns_level0=["SrtmDEM_mean_Value", "soil_l1_sand_nearest_Value"],
             plot_columns_level1=["annual_P_in_src_grid_Value", "umd_lc_major_Value"]):
    
    # ====================== set dir and path ======================
    # set path
    dpc_VIC_level0_path = os.path.join(evb_dir.dpcFile_dir, "dpc_VIC_level0.pkl")
    dpc_VIC_level1_path = os.path.join(evb_dir.dpcFile_dir, "dpc_VIC_level1.pkl")
    dpc_VIC_level2_path = os.path.join(evb_dir.dpcFile_dir, "dpc_VIC_level2.pkl")
    dpc_VIC_plot_grid_basin_path = os.path.join(evb_dir.dpcFile_dir, "dpc_VIC_plot_grid_basin.tiff")
    dpc_VIC_plot_columns_path = os.path.join(evb_dir.dpcFile_dir, "dpc_VIC_plot_columns.tiff")
    
    # ====================== get basin_shp ======================
    basin_shp_all, basin_shp = read_one_basin_shp(basin_index)
    
    # ====================== build dpc level0 ======================
    # build grid_shp
    grid_shp_lon_level0, grid_shp_lat_level0, grid_shp_level0 = createGridForBasin(basin_shp, grid_res_level0)
    
    # build dpc_level0
    print("========== build dpc_level0 ==========")
    dpc_VIC_level0 = dataProcess_VIC_level0(basin_shp, grid_shp_level0, grid_res_level0, date_period)
    
    # read data
    dpc_VIC_level0_call_kwargs_ = {"readBasindata": False, "readGriddata": True, "readBasinAttribute": False}
    dpc_VIC_level0_call_kwargs_.update(dpc_VIC_level0_call_kwargs)
    dpc_VIC_level0(**dpc_VIC_level0_call_kwargs_)
    
    # ====================== build dpc level1 ======================
    # build grid_shp
    print("========== build dpc_level1 ==========")
    grid_shp_lon_level1, grid_shp_lat_level1, grid_shp_level1 = createGridForBasin(basin_shp, grid_res_level1)
    
    # build dpc_level1
    dpc_VIC_level1 = dataProcess_VIC_level1(basin_shp, grid_shp_level1, grid_res_level1, date_period)
    
    # read data
    dpc_VIC_level1_call_kwargs_ = {"readBasindata": True, "readGriddata": True, "readBasinAttribute": True}
    dpc_VIC_level1_call_kwargs_.update(dpc_VIC_level1_call_kwargs)
    dpc_VIC_level1(**dpc_VIC_level1_call_kwargs_)
    
    # ====================== build dpc level2 ======================
    # build grid_shp
    print("========== build dpc_level2 ==========")
    grid_shp_lon_level2, grid_shp_lat_level2, grid_shp_level2 = createGridForBasin(basin_shp, grid_res_level2)
    
    # build dpc_level2
    dpc_VIC_level2 = dataProcess_VIC_level2(basin_shp, grid_shp_level2, grid_res_level2, date_period)
    
    # read data, not read
    dpc_VIC_level2()
    
    # ====================== plot ======================
    fig_grid_basin, axes_grid_basin = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={"wspace": 0.4})
    dpc_VIC_level0.plot(fig_grid_basin, axes_grid_basin[0])
    dpc_VIC_level1.plot(fig_grid_basin, axes_grid_basin[1])
    dpc_VIC_level2.plot(fig_grid_basin, axes_grid_basin[2])
    
    axes_grid_basin[0].set_title("dpc level0")
    axes_grid_basin[1].set_title("dpc level1")
    axes_grid_basin[2].set_title("dpc level2")
    
    if plot_columns_level0 is not None:
        fig_columns, axes_columns = plt.subplots(2, 2, figsize=(12, 8))
        dpc_VIC_level0.plot_grid(column=plot_columns_level0[0], fig=fig_columns, ax=axes_columns[0, 0])
        dpc_VIC_level0.plot_grid(column=plot_columns_level0[1], fig=fig_columns, ax=axes_columns[0, 1])
        dpc_VIC_level1.plot_grid(column=plot_columns_level1[0], fig=fig_columns, ax=axes_columns[1, 0])
        dpc_VIC_level1.plot_grid(column=plot_columns_level1[1], fig=fig_columns, ax=axes_columns[1, 1])
        
        axes_columns[0, 0].set_title(plot_columns_level0[0])
        axes_columns[0, 1].set_title(plot_columns_level0[1])
        axes_columns[1, 0].set_title(plot_columns_level1[0])
        axes_columns[1, 1].set_title(plot_columns_level1[1])
    
    # ====================== save ======================
    with open(dpc_VIC_level0_path, "wb") as f:
        pickle.dump(dpc_VIC_level0, f)
    
    with open(dpc_VIC_level1_path, "wb") as f:
        pickle.dump(dpc_VIC_level1, f)
    
    with open(dpc_VIC_level2_path, "wb") as f:
        pickle.dump(dpc_VIC_level2, f)

    fig_grid_basin.savefig(dpc_VIC_plot_grid_basin_path)
    if plot_columns_level0 is not None:
        fig_columns.savefig(dpc_VIC_plot_columns_path)
    
    return dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2
