# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from easy_vic_build.Evb_dir_class import Evb_dir
from easy_vic_build.tools.utilities import *
from easy_vic_build.tools.plot_func.plot_func import *
from easy_vic_build.tools.plot_func import plot_func
from easy_vic_build.tools.dpc_func.dpc_subclass import dataProcess_VIC_level0, dataProcess_VIC_level1, dataProcess_VIC_level2, dataProcess_VIC_level3
import matplotlib.gridspec as gridspec
import rasterio
from general_info import *
plt.rcParams['font.family']='Arial'
plt.rcParams['font.size']=12
plt.rcParams['font.weight']='normal'

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
    case_name_hydroanalysis = f"{basin_index}_hydroanalysis"
    case_name = f"{basin_index}_{model_scale}"
    
    # build dir
    evb_dir_hydroanalysis = Evb_dir(cases_home="./examples")
    evb_dir_hydroanalysis.builddir(case_name_hydroanalysis)
    evb_dir = Evb_dir(cases_home="./examples")
    evb_dir.builddir(case_name)
    
    # read dpc
    dpc_VIC_level0 = readdpc(evb_dir.dpc_VIC_level0_path, dataProcess_VIC_level0)
    dpc_VIC_level1 = readdpc(evb_dir.dpc_VIC_level1_path, dataProcess_VIC_level1)
    dpc_VIC_level2 = readdpc(evb_dir.dpc_VIC_level2_path, dataProcess_VIC_level2)
    dpc_VIC_level3 = readdpc(evb_dir.dpc_VIC_level3_path, dataProcess_VIC_level3)
    
    # read stream gdf
    stream_gdf = gpd.read_file(os.path.join(
        evb_dir_hydroanalysis.Hydroanalysis_dir,
        "wbw_working_directory_level0",
        f"stream_raster_clip_vector.shp"
    ))
    
    basin_attribute = dpc_VIC_level3.get_data_from_cache("basin_attribute")[0]
    
    # plot
    fig_dict, ax_dict = plot_func.plot_Basin_map(
        dpc_VIC_level0, 
        dpc_VIC_level1,
        dpc_VIC_level2,
        stream_gdf,
        [[basin_attribute["camels_topo:gauge_lon"].values[0]], [basin_attribute["camels_topo:gauge_lat"].values[0]]],
        x_locator_interval=0.5, y_locator_interval=0.5,
        fig=None, ax=None,
        dem_column="SrtmDEM_mean_Value",
        figsize=(8, 6),
    )
    
    fig_dict["fig_Basin_map"].savefig(os.path.join(evb_dir_hydroanalysis.BasinMap_dir, "fig_Basin_map.tiff"), dpi=300)
    fig_dict["fig_Basin_map"].savefig(os.path.join(evb_dir.BasinMap_dir, "fig_Basin_map.tiff"), dpi=300)
    fig_dict["fig_grid_basin_level0"].savefig(os.path.join(evb_dir.BasinMap_dir, "fig_grid_basin_level0.tiff"), dpi=300)
    fig_dict["fig_grid_basin_level1"].savefig(os.path.join(evb_dir.BasinMap_dir, "fig_grid_basin_level1.tiff"), dpi=300)
    fig_dict["fig_grid_basin_level2"].savefig(os.path.join(evb_dir.BasinMap_dir, "fig_grid_basin_level2.tiff"), dpi=300)
    
if __name__ == "__main__":
    test()
    