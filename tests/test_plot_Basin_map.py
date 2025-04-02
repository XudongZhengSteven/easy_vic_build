# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from easy_vic_build.Evb_dir_class import Evb_dir
from easy_vic_build.tools.utilities import *
from easy_vic_build.tools.plot_func.plot_func import *
from easy_vic_build.tools.hydroanalysis_func.hydroanalysis_wbw.hydroanalysis_for_BasinMap import *
import matplotlib.gridspec as gridspec
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

scalemap = {"3km": 0.025, "6km": 0.055, "8km": 0.072, "12km": 0.11}

def read_data_for_plot(evb_dir):
    # read
    dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2 = readdpc(evb_dir)
    domain_dataset = readDomain(evb_dir)
    params_dataset_level0, params_dataset_level1 = readParam(evb_dir, mode="r")
    stream_gdf = readBasinMap(evb_dir)
    
    return evb_dir, dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2, domain_dataset, params_dataset_level0, params_dataset_level1, stream_gdf


def close_data_for_plot(domain_dataset, params_dataset_level0, params_dataset_level1):
    domain_dataset.close()
    params_dataset_level0.close()
    params_dataset_level1.close()
    
    
def plot_basin_map(evb_dir, x_locator_interval=0.3, y_locator_interval=0.2):
    # read
    evb_dir, dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2, domain_dataset, params_dataset_level0, params_dataset_level1, stream_gdf = read_data_for_plot(evb_dir)
    
    # plot
    fig_dict, ax_dict = plot_Basin_map(dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2, stream_gdf, x_locator_interval=x_locator_interval, y_locator_interval=y_locator_interval, fig=None, ax=None)
    fig_dict["fig_Basin_map"].savefig(os.path.join(evb_dir.BasinMap_dir, "fig_Basin_map.tiff"), dpi=300)
    fig_dict["fig_grid_basin_level0"].savefig(os.path.join(evb_dir.BasinMap_dir, "fig_grid_basin_level0.tiff"), dpi=300)
    fig_dict["fig_grid_basin_level1"].savefig(os.path.join(evb_dir.BasinMap_dir, "fig_grid_basin_level1.tiff"), dpi=300)
    fig_dict["fig_grid_basin_level2"].savefig(os.path.join(evb_dir.BasinMap_dir, "fig_grid_basin_level2.tiff"), dpi=300)
    
    # close
    close_data_for_plot(domain_dataset, params_dataset_level0, params_dataset_level1)
    

def plot_basin_map_combine(evb_dir, figsize=(12, 8), grid_kwarg={"left": 0.06, "right": 0.99, "bottom": 0.05, "top": 0.98, "hspace": 0.1, "wspace": 0.15}, 
                           ax1_box_aspect_factor=1,
                           x_locator_interval_landsurface=0.47, y_locator_interval_landsurface=0.5,
                           x_locator_interval_grid=0.24, y_locator_interval_grid=0.3):
    # ------------ read ------------
    evb_dir, dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2, domain_dataset, params_dataset_level0, params_dataset_level1, stream_gdf = read_data_for_plot(evb_dir)
    
    # ------------ plot ------------
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 4, figure=fig, **grid_kwarg)
    ax1 = plt.subplot(gs[0, 0], projection=ccrs.PlateCarree())
    
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[:, 2:])
    ax4 = plt.subplot(gs[1, 0])
    ax5 = plt.subplot(gs[1, 1])
    
    # plot US
    fig, ax1 = plot_US_basemap(fig=fig, ax=ax1, set_xyticks_bool=True, x_locator_interval=8, y_locator_interval=8, yticks_rotation=90)
    ax1.plot(dpc_VIC_level1.basin_shp.lon_cen.values[0], dpc_VIC_level1.basin_shp.lat_cen.values[0], "r*", markersize=10, mec="k", mew=1, zorder=50)  # location
    zoom_center(ax1, dpc_VIC_level1.basin_shp.lon_cen.values[0], dpc_VIC_level1.basin_shp.lat_cen.values[0], zoom_factor=2)
    set_ax_box_aspect(ax1, ax1_box_aspect_factor)
    # ax1.set_aspect('equal', adjustable='datalim')
    
    # plot dem
    dpc_VIC_level0.grid_shp.plot(ax=ax2, column="SrtmDEM_mean_Value", alpha=1, legend=False, colormap="terrain", zorder=1,
                                 legend_kwds={"label": "Elevation (m)"})  # terrain gray
    dpc_VIC_level0.basin_shp.plot(ax=ax2, facecolor="none", linewidth=2, alpha=1, edgecolor="k", zorder=2)
    dpc_VIC_level0.basin_shp.plot(ax=ax2, facecolor="k", alpha=0.2, zorder=3)
    stream_gdf.plot(ax=ax2, color="b", zorder=4)
    gauge_lon = dpc_VIC_level1.basin_shp["camels_topo:gauge_lon"].values[0]
    gauge_lat = dpc_VIC_level1.basin_shp["camels_topo:gauge_lat"].values[0]
    ax2.plot(gauge_lon, gauge_lat, "r^", markersize=8, mec="k", mew=1, zorder=5)
    # fig, ax2 = dpc_VIC_level1.plot(fig, ax2, basin_shp_kwargs={"edgecolor": "k", "alpha": 0.1, "facecolor": "b"})  # grid
    set_boundary(ax2, dpc_VIC_level0.boundary_grids_edge_x_y)
    set_xyticks(ax2, x_locator_interval=x_locator_interval_landsurface, y_locator_interval=y_locator_interval_landsurface, yticks_rotation=90)
    
    # plot grid
    fig, ax3 = dpc_VIC_level1.plot(fig, ax3)
    set_boundary(ax3, dpc_VIC_level1.boundary_grids_edge_x_y)
    set_xyticks(ax3, x_locator_interval=x_locator_interval_grid, y_locator_interval=y_locator_interval_grid, yticks_rotation=90)
    
    # plot LULC
    UMD_LULC_cmap, UMD_LULC_norm, UMD_LULC_ticks, UMD_LULC_ticks_position, UMD_LULC_colorlist, UMD_LULC_colorlevel = get_UMD_LULC_cmap()
    dpc_VIC_level1.grid_shp.plot(ax=ax4, column="umd_lc_major_Value", alpha=1, legend=False, colormap=UMD_LULC_cmap, zorder=1,
                                 legend_kwds={"label": "UMD LULC"})  # terrain gray
    set_boundary(ax4, dpc_VIC_level1.boundary_grids_edge_x_y)
    set_xyticks(ax4, x_locator_interval=x_locator_interval_landsurface, y_locator_interval=y_locator_interval_landsurface, yticks_rotation=90)
    
    # plot Veg
    ndvi_cmap = get_NDVI_cmap()
    dpc_VIC_level1.grid_shp["MODIS_NDVI_mean_Value_month7_scaled"] = dpc_VIC_level1.grid_shp["MODIS_NDVI_mean_Value_month7"] * 0.0001 * 0.0001
    dpc_VIC_level1.grid_shp.plot(ax=ax5, column="MODIS_NDVI_mean_Value_month7_scaled", alpha=1, legend=False, colormap=ndvi_cmap, zorder=1,
                                 legend_kwds={"label": "NDVI"}, vmin=0, vmax=1)  # Greens
    set_boundary(ax5, dpc_VIC_level1.boundary_grids_edge_x_y)
    set_xyticks(ax5, x_locator_interval=x_locator_interval_landsurface, y_locator_interval=y_locator_interval_landsurface, yticks_rotation=90)
    
    # ------------ plot colorbar ------------
    # dem cb
    dem_values = dpc_VIC_level0.grid_shp["SrtmDEM_mean_Value"].values
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
    
    # ------------ close ------------
    close_data_for_plot(domain_dataset, params_dataset_level0, params_dataset_level1)
    
    
if __name__ == "__main__":
    # general set
    basin_index = 213
    model_scale = "6km"
    date_period = ["19980101", "19981231"]
    case_name = f"{basin_index}_{model_scale}"

    # build dir
    evb_dir = Evb_dir(cases_home="./examples")
    evb_dir.builddir(case_name)
    
    # hydroanalysis for BasinMap
    hydroanalysis_for_basin_bool = False
    if hydroanalysis_for_basin_bool:
        hydroanalysis_for_basin(evb_dir)
    
    # plot_basin_map
    plot_basin_map(evb_dir, x_locator_interval=0.3, y_locator_interval=0.2)
    
    # plot_basin_map_combine
    plot_basin_map_combine(evb_dir, figsize=(18, 8), grid_kwarg={"left": 0.06, "right": 0.99, "bottom": 0.05, "top": 0.98, "hspace": 0.01, "wspace": 0.1}, ax1_box_aspect_factor=0.8)
    