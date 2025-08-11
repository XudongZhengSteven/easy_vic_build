# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from easy_vic_build.Evb_dir_class import Evb_dir
from easy_vic_build.tools.utilities import *
from easy_vic_build.tools.plot_func.plot_func import *
from easy_vic_build.tools.plot_func import plot_func
from easy_vic_build.tools.dpc_func.dpc_subclass import dataProcess_VIC_level0, dataProcess_VIC_level1, dataProcess_VIC_level2, dataProcess_VIC_level3
import matplotlib.gridspec as gridspec
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

def plot_basin_map_combine(
    figsize=(12, 8),
    grid_kwarg={"left": 0.06, "right": 0.99, "bottom": 0.05, "top": 0.98, "hspace": 0.1, "wspace": 0.15},
    ax1_box_aspect_factor=1.5,
    x_locator_interval_landsurface=0.47, y_locator_interval_landsurface=0.5,
    x_locator_interval_grid=0.24, y_locator_interval_grid=0.3
):
    # general set
    case_name_hydroanalysis = f"{basin_index}_hydroanalysis"
    case_name = f"{basin_index}_{model_scale}"
    
    # build dir
    evb_dir_hydroanalysis = Evb_dir(cases_home="../")
    evb_dir_hydroanalysis.builddir(case_name_hydroanalysis)
    evb_dir = Evb_dir(cases_home="../")
    evb_dir.builddir(case_name)
    
    # read
    dpc_VIC_level0 = readdpc(evb_dir.dpc_VIC_level0_path, dataProcess_VIC_level0)
    dpc_VIC_level1 = readdpc(evb_dir.dpc_VIC_level1_path, dataProcess_VIC_level1)
    dpc_VIC_level3 = readdpc(evb_dir.dpc_VIC_level3_path, dataProcess_VIC_level3)
    
    dpc_VIC_level0.merge_grid_data()
    grid_shp_level0 = dpc_VIC_level0.get_data_from_cache("merged_grid_shp")[0]
    
    dpc_VIC_level1.merge_grid_data()
    grid_shp_level1 = dpc_VIC_level1.get_data_from_cache("merged_grid_shp")[0]
    
    basin_shp = dpc_VIC_level3.get_data_from_cache("basin_shp")[0]
    
    stream_gdf = gpd.read_file(os.path.join(
        evb_dir_hydroanalysis.Hydroanalysis_dir,
        "wbw_working_directory_level0",
        f"stream_raster_clip_vector.shp"
    ))
    
    basin_attribute = dpc_VIC_level3.get_data_from_cache("basin_attribute")[0]
    basin_center_coord = [basin_attribute.lon_cen.values[0], basin_attribute.lat_cen.values[0]]  # [lon, lat]
    gauge_lon = basin_attribute["camels_topo:gauge_lon"].values[0]
    gauge_lat = basin_attribute["camels_topo:gauge_lat"].values[0]
    
    # plot
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 4, figure=fig, **grid_kwarg)
    ax1 = plt.subplot(gs[0, 0], projection=ccrs.PlateCarree())
    
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[:, 2:])
    ax4 = plt.subplot(gs[1, 0])
    ax5 = plt.subplot(gs[1, 1])
    
    # plot US
    fig, ax1 = plot_US_basemap(fig=fig, ax=ax1, set_xyticks_bool=True, x_locator_interval=8, y_locator_interval=8, yticks_rotation=90)
    ax1.plot(basin_center_coord[0], basin_center_coord[1], "r*", markersize=10, mec="k", mew=1, zorder=50)  # location
    zoom_center(ax1, basin_center_coord[0], basin_center_coord[1], zoom_factor=2)
    set_ax_box_aspect(ax1, ax1_box_aspect_factor)
    # ax1.set_aspect('equal', adjustable='datalim')
    
    # plot dem
    grid_shp_level0.plot(ax=ax2, column="SrtmDEM_mean_Value", alpha=1, legend=False, colormap="terrain", zorder=1,
                                 legend_kwds={"label": "Elevation (m)"})  # terrain gray
    # grid_shp_level0.plot(ax=ax2, facecolor="none", linewidth=0.1, alpha=1, edgecolor="k", zorder=2)
    # grid_shp_level0.plot(ax=ax2, facecolor="k", alpha=0.2, zorder=3)
    stream_gdf.plot(ax=ax2, color="b", zorder=4)

    ax2.plot(gauge_lon, gauge_lat, "r^", markersize=8, mec="k", mew=1, zorder=5)
    basin_shp.plot(ax=ax2, edgecolor="k", alpha=1, facecolor="none", zorder=4)
    # fig, ax2 = dpc_VIC_level1.plot(fig, ax2, basin_shp_kwargs={"edgecolor": "k", "alpha": 0.1, "facecolor": "b"})  # grid
    set_boundary(ax2, grid_shp_level0.createBoundaryShp()[-1])
    set_xyticks(ax2, x_locator_interval=x_locator_interval_landsurface, y_locator_interval=y_locator_interval_landsurface, yticks_rotation=90)
    
    # plot grid
    basin_shp.plot(ax=ax3, edgecolor="k", alpha=0.5, facecolor="b")
    grid_shp_level1.plot(ax=ax3, alpha=0.5, edgecolor="k", facecolor="none", linewidth=0.5)
    grid_shp_level1.point_geometry.plot(ax=ax3, alpha=0.5, color="darkblue", markersize=1)        
    # fig, ax3 = grid_shp_level1.plot(fig, ax3)
    set_boundary(ax3, grid_shp_level1.createBoundaryShp()[-1])
    set_xyticks(ax3, x_locator_interval=x_locator_interval_grid, y_locator_interval=y_locator_interval_grid, yticks_rotation=90)
    
    # plot LULC
    UMD_LULC_cmap, UMD_LULC_norm, UMD_LULC_ticks, UMD_LULC_ticks_position, UMD_LULC_colorlist, UMD_LULC_colorlevel = get_UMD_LULC_cmap()
    grid_shp_level1.plot(ax=ax4, column="umd_lc_major_Value", alpha=1, legend=False, colormap=UMD_LULC_cmap, zorder=1, norm=UMD_LULC_norm,
                                 legend_kwds={"label": "UMD LULC"})  # terrain gray
    set_boundary(ax4, grid_shp_level1.createBoundaryShp()[-1])
    set_xyticks(ax4, x_locator_interval=x_locator_interval_landsurface, y_locator_interval=y_locator_interval_landsurface, yticks_rotation=90)
    
    # plot Veg
    ndvi_cmap = get_NDVI_cmap()
    grid_shp_level1["MODIS_NDVI_mean_Value_month7_scaled"] = grid_shp_level1["MODIS_NDVI_mean_Value_month7"] * 0.0001 * 0.0001
    grid_shp_level1.plot(ax=ax5, column="MODIS_NDVI_mean_Value_month7_scaled", alpha=1, legend=False, colormap=ndvi_cmap, zorder=1,
                                 legend_kwds={"label": "NDVI"}, vmin=0, vmax=1)  # Greens
    set_boundary(ax5, grid_shp_level1.createBoundaryShp()[-1])
    set_xyticks(ax5, x_locator_interval=x_locator_interval_landsurface, y_locator_interval=y_locator_interval_landsurface, yticks_rotation=90)
    
    # ------------ plot colorbar ------------
    # dem cb
    dem_values = grid_shp_level0["SrtmDEM_mean_Value"].values
    dem_vmin = dem_values.min()
    dem_vmax = dem_values.max()
    dem_cmap = "terrain"
    fig_dem_cb, ax_dem_cb, _, _ = get_colorbar(dem_vmin, dem_vmax, dem_cmap, figsize=(4, 2), subplots_adjust={"right": 0.5}, cb_label="", cb_label_kwargs={}, cb_kwargs={"orientation":"vertical"})
    
    # lulc cb
    lulc_vmin = -0.5
    lulc_vmax = 13.5
    lulc_cmap = UMD_LULC_cmap
    fig_lulc_cb, ax_lulc_cb, _, _ = get_colorbar(lulc_vmin, lulc_vmax, lulc_cmap, figsize=(6, 1), subplots_adjust={"bottom": 0.5}, cb_label="UMD LULC Classification", cb_label_kwargs={}, cb_kwargs={"orientation":"horizontal", "ticks": UMD_LULC_ticks_position})
    
    # NDVI cb
    ndvi_vmin = 0
    ndvi_vmax = 1
    ndvi_cmap = ndvi_cmap
    fig_ndvi_cb, ax_ndvi_cb, _, _ = get_colorbar(ndvi_vmin, ndvi_vmax, ndvi_cmap, figsize=(6, 1), subplots_adjust={"bottom": 0.5}, cb_label="NDVI", cb_label_kwargs={}, cb_kwargs={"orientation":"horizontal"})
    
    # ------------ save fig ------------
    fig.savefig(os.path.join(evb_dir.BasinMap_dir, "fig_Basin_map_combine.tiff"), dpi=300)
    fig_dem_cb.savefig(os.path.join(evb_dir.BasinMap_dir, "fig_dem_cb.svg"), dpi=300)
    fig_lulc_cb.savefig(os.path.join(evb_dir.BasinMap_dir, "fig_lulc_cb.svg"), dpi=300)
    fig_ndvi_cb.savefig(os.path.join(evb_dir.BasinMap_dir, "fig_ndvi_cb.svg"), dpi=300)
    

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
    # test()
    plot_basin_map_combine()