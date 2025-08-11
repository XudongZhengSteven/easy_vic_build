# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os

import geopandas as gpd

from general_info import *
from JRB_build_evb_dir import build_modeling_dir
from easy_vic_build.tools.plot_func.plot_func import set_boundary, set_xyticks, plot_Basin_map, get_NDVI_cmap, get_colorbar, get_UMD_LULC_cmap
from easy_vic_build.tools.utilities import readdpc
from JRB_build_dpc import dataProcess_VIC_level0_JRB, dataProcess_VIC_level1_JRB, dataProcess_VIC_level2_CDMet_JRB, dataProcess_VIC_level2_CMADSV1_JRB, dataProcess_VIC_level2_CMFD_JRB, dataProcess_VIC_level3_JRB

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs


def plot_basin_map_combine(
    evb_dir_hydroanalysis,
    model_scale,
    station_name,
    figsize=(7, 8),
    grid_kwarg={"left": 0.06, "right": 0.99, "bottom": 0.05, "top": 0.98, "hspace": 0.1, "wspace": 0.08},
    x_locator_interval_landsurface=1, y_locator_interval_landsurface=0.8,
    x_locator_interval_grid=1, y_locator_interval_grid=0.8
):  
    # build evb
    evb_dir_modeling = build_modeling_dir(subname=f"{station_name}_{model_scale}")
    
    # read
    dpc_VIC_level0 = readdpc(evb_dir_modeling.dpc_VIC_level0_path, dataProcess_VIC_level0_JRB)
    dpc_VIC_level1 = readdpc(evb_dir_modeling.dpc_VIC_level1_path, dataProcess_VIC_level1_JRB)
    dpc_VIC_level3 = readdpc(evb_dir_modeling.dpc_VIC_level3_path, dataProcess_VIC_level3_JRB)
    
    # merge
    dpc_VIC_level0.merge_grid_data()
    grid_shp_level0 = dpc_VIC_level0.get_data_from_cache("merged_grid_shp")[0]
    
    dpc_VIC_level1.merge_grid_data()
    grid_shp_level1 = dpc_VIC_level1.get_data_from_cache("merged_grid_shp")[0]
    
    basin_shp = dpc_VIC_level3.get_data_from_cache("basin_shp")[0]
    
    stream_gdf = gpd.read_file(os.path.join(
        evb_dir_hydroanalysis.Hydroanalysis_dir,
        "wbw_working_directory_level0",
        f"clipped_stream_vector_basin_vector_outlet_with_reference_{basin_outlets_reference_i_map[station_name]}.shp"
    ))
    
    basin_attribute = dpc_VIC_level3.get_data_from_cache("gauge_info")[0]
    gauge_lon = basin_attribute["gauge_coord(lon, lat)"][0]
    gauge_lat = basin_attribute["gauge_coord(lon, lat)"][1]
    
    # plot
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig, **grid_kwarg)
    
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])
    
    # plot dem
    grid_shp_level0.plot(ax=ax1, column="ASTGTM_DEM_mean_Value", alpha=1, legend=False, colormap="terrain", zorder=1,
                                 legend_kwds={"label": "Elevation (m)"})  # terrain gray
    # grid_shp_level0.plot(ax=ax2, facecolor="none", linewidth=0.1, alpha=1, edgecolor="k", zorder=2)
    # grid_shp_level0.plot(ax=ax2, facecolor="k", alpha=0.2, zorder=3)
    stream_gdf.plot(ax=ax1, color="b", zorder=4)

    ax1.plot(gauge_lon, gauge_lat, "r^", markersize=8, mec="k", mew=1, zorder=5)
    basin_shp.plot(ax=ax1, edgecolor="k", alpha=1, facecolor="none", zorder=4)
    # fig, ax2 = dpc_VIC_level1.plot(fig, ax2, basin_shp_kwargs={"edgecolor": "k", "alpha": 0.1, "facecolor": "b"})  # grid
    set_boundary(ax1, grid_shp_level0.createBoundaryShp()[-1])
    set_xyticks(ax1, x_locator_interval=x_locator_interval_landsurface, y_locator_interval=y_locator_interval_landsurface, yticks_rotation=90)
    
    # plot grid
    basin_shp.plot(ax=ax2, edgecolor="k", alpha=0.5, facecolor="b")
    grid_shp_level1.plot(ax=ax2, alpha=0.5, edgecolor="k", facecolor="none", linewidth=0.5)
    grid_shp_level1.point_geometry.plot(ax=ax2, alpha=0.5, color="darkblue", markersize=1)        
    # fig, ax3 = grid_shp_level1.plot(fig, ax3)
    set_boundary(ax2, grid_shp_level1.createBoundaryShp()[-1])
    set_xyticks(ax2, x_locator_interval=x_locator_interval_grid, y_locator_interval=y_locator_interval_grid, yticks_rotation=90)
    
    # plot LULC
    UMD_LULC_cmap, UMD_LULC_norm, UMD_LULC_ticks, UMD_LULC_ticks_position, UMD_LULC_colorlist, UMD_LULC_colorlevel = get_UMD_LULC_cmap()
    grid_shp_level1.plot(ax=ax3, column="umd_lc_major_Value", alpha=1, legend=False, colormap=UMD_LULC_cmap, zorder=1, norm=UMD_LULC_norm,
                                 legend_kwds={"label": "UMD LULC"})  # terrain gray
    set_boundary(ax3, grid_shp_level1.createBoundaryShp()[-1])
    set_xyticks(ax3, x_locator_interval=x_locator_interval_landsurface, y_locator_interval=y_locator_interval_landsurface, yticks_rotation=90)
    
    # plot Veg
    ndvi_cmap = get_NDVI_cmap()
    grid_shp_level1["MODIS_NDVI_mean_Value_month7_scaled"] = grid_shp_level1["MODIS_NDVI_mean_Value_month7"] * 0.0001 * 0.0001
    grid_shp_level1.plot(ax=ax4, column="MODIS_NDVI_mean_Value_month7_scaled", alpha=1, legend=False, colormap=ndvi_cmap, zorder=1,
                                 legend_kwds={"label": "NDVI"}, vmin=0, vmax=1)  # Greens
    set_boundary(ax4, grid_shp_level1.createBoundaryShp()[-1])
    set_xyticks(ax4, x_locator_interval=x_locator_interval_landsurface, y_locator_interval=y_locator_interval_landsurface, yticks_rotation=90)
    
    # ------------ plot colorbar ------------
    # dem cb
    dem_values = grid_shp_level0["ASTGTM_DEM_mean_Value"].values
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
    fig.savefig(os.path.join(evb_dir_modeling.BasinMap_dir, "fig_Basin_map_combine.tiff"), dpi=300)
    fig_dem_cb.savefig(os.path.join(evb_dir_modeling.BasinMap_dir, "fig_dem_cb.svg"), dpi=300)
    fig_lulc_cb.savefig(os.path.join(evb_dir_modeling.BasinMap_dir, "fig_lulc_cb.svg"), dpi=300)
    fig_ndvi_cb.savefig(os.path.join(evb_dir_modeling.BasinMap_dir, "fig_ndvi_cb.svg"), dpi=300)
    
    
def plot_basin_map_JRB(evb_dir_hydroanalysis, model_scale, station_name):
    # build evb
    evb_dir_modeling = build_modeling_dir(subname=f"{station_name}_{model_scale}")
    
    # read dpc_VIC_level1
    dpc_VIC_level0 = dataProcess_VIC_level0_JRB(evb_dir_modeling._dpc_VIC_level0_path)
    dpc_VIC_level1 = dataProcess_VIC_level1_JRB(evb_dir_modeling._dpc_VIC_level1_path)
    dpc_VIC_level2_CDMet = dataProcess_VIC_level2_CDMet_JRB(evb_dir_modeling._dpc_VIC_level2_path.replace(".pkl", "_CDMet.pkl"))
    dpc_VIC_level3 = dataProcess_VIC_level3_JRB(evb_dir_modeling._dpc_VIC_level3_path)
    
    
    # read stream gdf
    station_id = basin_outlets_reference_i_map[station_name]
    
    stream_gdf = gpd.read_file(os.path.join(
        evb_dir_hydroanalysis.Hydroanalysis_dir,
        "wbw_working_directory_level0",
        f"clipped_stream_vector_basin_vector_outlet_with_reference_{station_id}.shp"
    ))
    
    # plot
    fig_dict, ax_dict = plot_Basin_map(
        dpc_VIC_level0,
        dpc_VIC_level1,
        dpc_VIC_level2_CDMet,
        stream_gdf,
        dpc_VIC_level3.get_data_from_cache("gauge_info")[0]["gauge_coord(lon, lat)"],
        x_locator_interval=1, y_locator_interval=0.5,
        fig=None, ax=None,
        dem_column="ASTGTM_DEM_mean_Value",
    )
    
    fig_dict["fig_Basin_map"].savefig(os.path.join(evb_dir_modeling.BasinMap_dir, "fig_Basin_map.tiff"), dpi=300)
    fig_dict["fig_grid_basin_level0"].savefig(os.path.join(evb_dir_modeling.BasinMap_dir, "fig_grid_basin_level0.tiff"), dpi=300)
    fig_dict["fig_grid_basin_level1"].savefig(os.path.join(evb_dir_modeling.BasinMap_dir, "fig_grid_basin_level1.tiff"), dpi=300)
    fig_dict["fig_grid_basin_level2"].savefig(os.path.join(evb_dir_modeling.BasinMap_dir, "fig_grid_basin_level2.tiff"), dpi=300)
    

if __name__ == "__main__":
    # build hydroanalysis evb_dir for read stream
    evb_dir_hydroanalysis = build_modeling_dir(subname="hydroanalysis")
    
    # plot basin map
    # plot_basin_map_JRB(evb_dir_hydroanalysis, model_scale, station_name)
    
    # plot_basin_map_combine
    plot_basin_map_combine(evb_dir_hydroanalysis, model_scale, station_name)
    