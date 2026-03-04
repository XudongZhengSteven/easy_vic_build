# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from easy_vic_build.tools.dpc_func.basin_grid_func import createArray_from_gridshp
import matplotlib.pyplot as plt

import os
from copy import deepcopy
import numpy as np
import pandas as pd


def ExtractData(
    grid_shp,
    evb_dir,
    date_period=["20080101", "20181231"],
    plot=False,
    reverse_lat=True
):
    # import
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from HRB_build_dpc import dataProcess_VIC_level2_CMFD_HRB
    
    dpc_VIC_level2_CMFD = dataProcess_VIC_level2_CMFD_HRB(
        load_path=evb_dir._dpc_VIC_level2_path.replace(".pkl", "_CMFD.pkl"),
        reset_on_load_failure=False,
    )
    
    foricng_df_CMFD, _ = dpc_VIC_level2_CMFD.get_data_from_cache("cmfd_forcing")
    
    # get annual pre
    # date = pd.date_range(date_period[0], date_period[1], freq="D")
    date = pd.date_range(date_period[0], date_period[1], freq="3H")
    
    # calculate annual pre, mm
    def compute_annual_pre_series(row, column="pre_mm_per_day"):
        pre_series = row[column]
        pre_df = pd.DataFrame(pre_series, index=date, columns=['precipitation'])
        annual_pre = pre_df.groupby(pre_df.index.year)['precipitation'].sum()
        annual_pre_mean = annual_pre.mean()
        return annual_pre_mean
    
    annual_P_CMFD = foricng_df_CMFD.loc[:, ["pre_mm_per_3h"]].apply(compute_annual_pre_series, args=("pre_mm_per_3h", ), axis=1)  # pre_mm_per_day
    
    # create array
    grid_shp_annual_P = deepcopy(grid_shp)
    grid_shp_annual_P["annual_P_CMFD_mm"] = annual_P_CMFD
    
    annual_P_CMFD_array, stand_grids_lon, stand_grids_lat, rows_index, cols_index = createArray_from_gridshp(grid_shp_annual_P, "annual_P_CMFD_mm", reverse_lat=reverse_lat)
    annual_P_mean = np.array(annual_P_CMFD_array)
    
    # plot
    if plot:
        (   boundary_point_center_shp,
            boundary_point_center_x_y,
            boundary_grids_edge_shp,
            boundary_grids_edge_x_y,
        ) = grid_shp_annual_P.createBoundaryShp()
        
        extent = boundary_grids_edge_x_y
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # 4
        
        handle2 = axes[0].imshow(annual_P_CMFD_array, cmap='RdBu', extent=[extent[0], extent[2], extent[1], extent[3]])  # , vmin=100, vmax=700
        axes[0].set_title('CMFD')
        
        handle3 = axes[1].imshow(annual_P_mean, cmap='RdBu', extent=[extent[0], extent[2], extent[1], extent[3]])  # , vmin=100, vmax=700
        axes[1].set_title('All data mean')
        
        plt.colorbar(mappable=handle2, label='Annual Precipitation (mm)', orientation='horizontal')
        plt.colorbar(mappable=handle3, label='Annual Precipitation (mm)', orientation='horizontal')
        
        plt.savefig(os.path.join(evb_dir.ParamFile_dir, "annual_P_comparison.png"), dpi=300)
        plt.show(block=True)
        plt.close(fig)
        
    # save
    np.savetxt(os.path.join(evb_dir.ParamFile_dir, "annual_P_stand_grids_lon.txt"), stand_grids_lon)
    np.savetxt(os.path.join(evb_dir.ParamFile_dir, "annual_P_stand_grids_lat.txt"), stand_grids_lat)
    np.savetxt(os.path.join(evb_dir.ParamFile_dir, "annual_P_rows_index.txt"), rows_index)
    np.savetxt(os.path.join(evb_dir.ParamFile_dir, "annual_P_cols_index.txt"), cols_index)

    np.save(os.path.join(evb_dir.ParamFile_dir, "annual_P_CMFD.npy"), annual_P_CMFD_array)
    np.save(os.path.join(evb_dir.ParamFile_dir, "annual_P_all_data_mean.npy"), annual_P_mean)
    
    grid_shp["annual_P_CMFD_mm"] = annual_P_CMFD
    
    return grid_shp
    
    
if __name__ == "__main__":
    pass
    
