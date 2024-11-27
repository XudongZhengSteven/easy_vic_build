# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
import matplotlib.pyplot as plt
from .tools.utilities import *
import pickle
from .tools.decoractors import clock_decorator
# you can set your own dataProcess_VIC_level0, dataProcess_VIC_level1, dataProcess_VIC_level2


@clock_decorator(print_arg_ret=False)
def builddpc(evb_dir, dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2,
             dpc_VIC_level0_call_kwargs=dict(), dpc_VIC_level1_call_kwargs=dict(), dpc_VIC_level2_call_kwargs=dict(),
             plot_columns_level0=["SrtmDEM_mean_Value", "soil_l1_sand_nearest_Value"],
             plot_columns_level1=["annual_P_in_src_grid_Value", "umd_lc_major_Value"]):
    
    # ====================== build dpc level0 ======================
    # build dpc_level0
    print("========== build dpc_level0 ==========")
    dpc_VIC_level0(**dpc_VIC_level0_call_kwargs)
    
    # ====================== build dpc level1 ======================
    # build dpc_level1
    print("========== build dpc_level1 ==========")
    dpc_VIC_level1(**dpc_VIC_level1_call_kwargs)
    
    # ====================== build dpc level2 ======================
    # build dpc_level2
    print("========== build dpc_level2 ==========")
    dpc_VIC_level2(**dpc_VIC_level2_call_kwargs)  # read data, not read
    
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
    with open(evb_dir.dpc_VIC_level0_path, "wb") as f:
        pickle.dump(dpc_VIC_level0, f)
    
    with open(evb_dir.dpc_VIC_level1_path, "wb") as f:
        pickle.dump(dpc_VIC_level1, f)
    
    with open(evb_dir.dpc_VIC_level2_path, "wb") as f:
        pickle.dump(dpc_VIC_level2, f)

    fig_grid_basin.savefig(evb_dir.dpc_VIC_plot_grid_basin_path)
    if plot_columns_level0 is not None:
        fig_columns.savefig(evb_dir.dpc_VIC_plot_columns_path)