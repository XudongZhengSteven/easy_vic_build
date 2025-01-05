# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from ..params_func.params_set import *
from matplotlib.ticker import FuncFormatter, MultipleLocator, MaxNLocator
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter, LatitudeLocator
# plt.rcParams['font.family'] = 'Arial'

## ------------------------ plot utilities ------------------------
def set_xyticks(ax, x_locator_interval, y_locator_interval):
    # set xy ticks
    ax.xaxis.set_major_locator(MultipleLocator(x_locator_interval))
    ax.yaxis.set_major_locator(MultipleLocator(y_locator_interval))
    
    format_lon = lambda lon, pos: f"{abs(lon):.1f}°W" if lon < 0 else f"{abs(lon):.1f}°E"
    format_lat = lambda lat, pos: f"{abs(lat):.1f}°S" if lat < 0 else f"{abs(lat):.1f}°N"
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))


def set_boundary(ax, boundary_x_y):
    ax.set_xlim(boundary_x_y[0], boundary_x_y[2])
    ax.set_ylim(boundary_x_y[1], boundary_x_y[3])
    
    
def plotBackground(basin_shp, grid_shp, fig=None, ax=None):
    if not ax:
        fig, ax = plt.subplots()
    plot_kwgs = {"facecolor": "none", "alpha": 0.7, "edgecolor": "k"}
    fig, ax = plotBasins(basin_shp, None, fig, ax, plot_kwgs)
    fig, ax = plotGrids(grid_shp, None, fig, ax)

    return fig, ax


def plotGrids(grid_shp, column=None, fig=None, ax=None, plot_kwgs1=None, plot_kwgs2=None):
    if not ax:
        fig, ax = plt.subplots()
    plot_kwgs1 = dict() if not plot_kwgs1 else plot_kwgs1
    plot_kwgs2 = dict() if not plot_kwgs2 else plot_kwgs2
    plot_kwgs1_ = {"facecolor": "none", "alpha": 0.2, "edgecolor": "gray"}
    plot_kwgs2_ = {"facecolor": "none", "alpha": 0.5, "edgecolor": "gray", "markersize": 0.5}

    plot_kwgs1_.update(plot_kwgs1)
    plot_kwgs2_.update(plot_kwgs2)

    grid_shp.plot(ax=ax, column=column, **plot_kwgs1_)
    grid_shp["point_geometry"].plot(ax=ax, **plot_kwgs2_)
    return fig, ax


def plotBasins(basin_shp, column=None, fig=None, ax=None, plot_kwgs=None):
    if not ax:
        fig, ax = plt.subplots()
    plot_kwgs = dict() if not plot_kwgs else plot_kwgs
    plot_kwgs_ = {"legend": True}
    plot_kwgs_.update(plot_kwgs)
    basin_shp.plot(ax=ax, column=column, **plot_kwgs_)

    return fig, ax


def setBoundary(ax, boundary_x_min, boundary_x_max, boundary_y_min, boundary_y_max):
    ax.set_xlim([boundary_x_min, boundary_x_max])
    ax.set_ylim([boundary_y_min, boundary_y_max])
    return ax


def plot_US_basemap(fig=None, ax=None, set_xyticks=True, x_locator_interval=15, y_locator_interval=10):
    proj = ccrs.PlateCarree()
    extent = [-125, -66.5, 24.5, 50.5]
    
    # get fig, ax
    if not ax:
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0, 0.85, 1], projection=proj)

    # add background
    alpha=0.3
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=10)
    ax.add_feature(cfeature.LAND, alpha=alpha)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=0.5, zorder=10, alpha=alpha)
    ax.add_feature(cfeature.LAKES.with_scale('50m'), linewidth=0.2, edgecolor="k", zorder=10, alpha=alpha)
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="gray", zorder=10)

    # set ticks
    if set_xyticks:
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks([-90, -45, 0, 45, 90])
        set_xyticks(ax, x_locator_interval=x_locator_interval, y_locator_interval=y_locator_interval)

    # set boundary
    set_boundary(ax, extent)  # or ax.set_extent(extent, crs=proj)
    
    # # set gridliner, use this may lead to different sizes between xticks and yticks
    # gridliner = ax.gridlines(crs=proj, draw_labels=True)  # , linewidth=2, color='gray', alpha=0.5, linestyle='--'
    # gridliner.top_labels = False
    # gridliner.right_labels = False
    # gridliner.xlines = False
    # gridliner.ylines = False
    # gridliner.xformatter = LongitudeFormatter()
    # gridliner.yformatter = LatitudeFormatter()
    # gridliner.xlabel_style = {'size': 12, 'color': 'k'}
    # gridliner.xlabel_style = {'size': 12, 'color': 'k'}

    return fig, ax


def plot_selected_map(basin_index, dpc, text_name="basin_index", plot_solely=True, column=None, plot_kwgs_set=dict(), fig=None, ax=None):
    """_summary_

    Args:
        basin_index (_type_): _description_
        dpc (_type_): _description_
        text_name (str, optional): _description_. Defaults to "basin_index".
        plot_solely (bool, optional): _description_. Defaults to True.
        column (_type_, optional): _description_. Defaults to None.
        plot_kwgs_set (_type_, optional): _description_. Defaults to dict().

    Returns:
        _type_: _description_
    
    usages:
    fig, ax, fig_solely = plot_selected_map(basin_shp_area_excluding.index.to_list(), # [0, 1, 2]
                                        dpc,
                                        text_name="basin_index",  # "basin_index", None,
                                        plot_solely=False, 
                                        column=None,  # "camels_clim:aridity",  # None
                                        plot_kwgs_set=dict()) # {"cmap": plt.cm.hot})  # dict()
    """
    # background
    proj = ccrs.PlateCarree()
    extent = [-125, -66.5, 24.5, 50.5]
    alpha=0.3
    if not fig:
        fig = plt.figure(dpi=300)
        ax = fig.add_axes([0.05, 0, 0.9, 1], projection=proj)

        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=10)
        ax.add_feature(cfeature.LAND, alpha=alpha)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=0.5, zorder=10, alpha=alpha)
        ax.add_feature(cfeature.LAKES.with_scale('50m'), linewidth=0.2, edgecolor="k", zorder=10, alpha=alpha)
        
    ax.set_extent(extent,crs=proj)

    # plot
    plot_kwgs = {"facecolor": "r", "alpha": 0.7, "edgecolor": "k", "linewidth": 0.2}
    plot_kwgs.update(plot_kwgs_set)
    if len(basin_index) > 1:
        fig, ax = plotBasins(dpc.basin_shp.loc[basin_index, :].to_crs(proj), fig=fig, ax=ax, plot_kwgs=plot_kwgs, column=column)
    elif len(basin_index) == 1:
        fig, ax = plotBasins(dpc.basin_shp.loc[[basin_index[0], basin_index[0]], :].to_crs(proj), fig=fig, ax=ax, plot_kwgs=plot_kwgs, column=column)
    else:
        return fig, ax, None
    
    # annotation
    if text_name:  # None means not to plot text
        basinLatCens = np.array([dpc.basin_shp.loc[key, "lat_cen"] for key in basin_index])
        basinLonCens = np.array([dpc.basin_shp.loc[key, "lon_cen"] for key in basin_index])
        
        for i in range(len(basinLatCens)):
            basinLatCen = basinLatCens[i]
            basinLonCen = basinLonCens[i]
            text_names_dict = {"basin_index": basin_index[i],
                            "hru_id": dpc.basin_shp.loc[basin_index[i], "hru_id"],
                            "gauge_id": dpc.basin_shp.loc[basin_index[i], "camels_hydro:gauge_id"]}
            
            text_name_plot = text_names_dict[text_name]
            
            ax.text(basinLonCen, basinLatCen, f"{text_name_plot}",
                    horizontalalignment='right',
                    transform=proj,
                    fontdict={"family": "Arial", "fontsize": 5, "color": "b", "weight": "bold"})
    
    # plot solely
    fig_solely = {}
    if plot_solely:
        for i in range(len(basin_index)):
            fig_, ax_ = plotBasins(dpc.basin_shp.loc[[basin_index[i], basin_index[i]], :].to_crs(proj), fig=None, ax=None, plot_kwgs=None)
            fig_solely[i] = {"fig": fig_, "ax": ax_}
            
            text_names_dict = {"basin_index": basin_index[i],
                               "hru_id": dpc.basin_shp.loc[basin_index[i], "hru_id"],
                               "gauge_id": dpc.basin_shp.loc[basin_index[i], "camels_hydro:gauge_id"]}
            text_name_plot = text_names_dict[text_name]
            
            ax_.set_title(text_name_plot)
    else:
        fig_solely = None
    
    return fig, ax, fig_solely


def plotShp(basinShp_original, basinShp, grid_shp, intersects_grids,
            boundary_x_min, boundary_x_max, boundary_y_min, boundary_y_max,
            fig=None, ax=None):
    if not ax:
        fig, ax = plt.subplots()
    basinShp_original.plot(ax=ax, facecolor="none", alpha=0.7, edgecolor="k")
    basinShp.plot(ax=ax)
    grid_shp["geometry"].plot(ax=ax, facecolor="none", edgecolor="gray", alpha=0.2)
    grid_shp["point_geometry"].plot(ax=ax, markersize=0.5, edgecolor="gray", facecolor="gray", alpha=0.5)
    intersects_grids.plot(ax=ax, facecolor="r", edgecolor="gray", alpha=0.2)
    ax.set_xlim([boundary_x_min, boundary_x_max])
    ax.set_ylim([boundary_y_min, boundary_y_max])

    return fig, ax


def plotLandCover(basinShp_original, basinShp, grid_shp, intersects_grids,
                  boundary_x_min, boundary_x_max, boundary_y_min, boundary_y_max,
                  fig=None, ax=None):
    colorlevel = [-0.5 + i for i in range(15)]
    colordict = cm.get_cmap("RdBu_r", 14)
    colordict = colordict(range(14))
    ticks = list(range(14))
    ticks_position = list(range(14))
    cmap = mcolors.ListedColormap(colordict)
    norm = mcolors.BoundaryNorm(colorlevel, cmap.N)

    if not ax:
        fig, ax = plt.subplots()
    basinShp_original.plot(ax=ax, facecolor="none", alpha=0.7, edgecolor="k")
    basinShp.plot(ax=ax)
    grid_shp.plot(ax=ax, column="major_umd_landcover_classification_grids", alpha=0.4,
                  legend=True, colormap=cmap, norm=norm,
                  legend_kwds={"label": "major_umd_landcover_classification_grids", "shrink": 0.8})
    intersects_grids.plot(ax=ax, facecolor="none", edgecolor="k", alpha=0.7)
    ax.set_xlim([boundary_x_min, boundary_x_max])
    ax.set_ylim([boundary_y_min, boundary_y_max])

    ax_cb = fig.axes[1]
    ax_cb.set_yticks(ticks_position)
    ax_cb.set_yticklabels(ticks)

    return fig, ax


def plotHWSDSoilData(basinShp_original, basinShp, grid_shp, intersects_grids,
                     boundary_x_min, boundary_x_max, boundary_y_min, boundary_y_max,
                     fig=None, ax=None, fig_T=None, ax_T=None, fig_S=None, ax_S=None):
    if not ax:
        fig, ax = plt.subplots()
    basinShp_original.plot(ax=ax, facecolor="none", alpha=0.7, edgecolor="k")
    basinShp.plot(ax=ax)
    grid_shp.plot(ax=ax, column="HWSD_BIL_Value", alpha=0.4,
                  legend=True, colormap="Accent",
                  legend_kwds={"label": "HWSD_BIL_Value", "shrink": 0.8})
    intersects_grids.plot(ax=ax, facecolor="none", edgecolor="k", alpha=0.7)
    ax.set_xlim([boundary_x_min, boundary_x_max])
    ax.set_ylim([boundary_y_min, boundary_y_max])

    # T_USDA_TEX_CLASS
    if not ax_T:
        fig_T, ax_T = plt.subplots()
    basinShp_original.plot(ax=ax_T, facecolor="none", alpha=0.7, edgecolor="k")
    basinShp.plot(ax=ax_T)
    grid_shp.plot(ax=ax_T, column="T_USDA_TEX_CLASS", alpha=0.4,
                  legend=True, colormap="Accent",
                  legend_kwds={"label": "T_USDA_TEX_CLASS", "shrink": 0.8})
    intersects_grids.plot(ax=ax_T, facecolor="none", edgecolor="k", alpha=0.7)
    ax_T.set_xlim([boundary_x_min, boundary_x_max])
    ax_T.set_ylim([boundary_y_min, boundary_y_max])

    # S_USDA_TEX_CLASS
    if not ax_S:
        fig_S, ax_S = plt.subplots()
    basinShp_original.plot(ax=ax_S, facecolor="none", alpha=0.7, edgecolor="k")
    basinShp.plot(ax=ax_S)
    grid_shp.plot(ax=ax_S, column="S_USDA_TEX_CLASS", alpha=0.4,
                  legend=True, colormap="Accent",
                  legend_kwds={"label": "S_USDA_TEX_CLASS", "shrink": 0.8})
    intersects_grids.plot(ax=ax_S, facecolor="none", edgecolor="k", alpha=0.7)
    ax_S.set_xlim([boundary_x_min, boundary_x_max])
    ax_S.set_ylim([boundary_y_min, boundary_y_max])

    return fig, ax, fig_S, ax_S, fig_T, ax_T


def plotStrmDEM(basinShp_original, basinShp, grid_shp, intersects_grids,
                boundary_x_min, boundary_x_max, boundary_y_min, boundary_y_max,
                fig=None, ax=None):
    if not ax:
        fig, ax = plt.subplots()
    basinShp_original.plot(ax=ax, facecolor="none", alpha=0.7, edgecolor="k")
    basinShp.plot(ax=ax)
    grid_shp.plot(ax=ax, column="SrtmDEM_mean_Value", alpha=1,
                  legend=True, colormap="gray",
                  legend_kwds={"label": "SrtmDEM_mean_Value", "shrink": 0.8})
    intersects_grids.plot(ax=ax, facecolor="none", edgecolor="k", alpha=0.7)
    ax.set_xlim([boundary_x_min, boundary_x_max])
    ax.set_ylim([boundary_y_min, boundary_y_max])

    return fig, ax


def plot_Calibrate_cp_SO(cp_state):
    # get value
    populations = [h[0] for h in cp_state["history"]]
    fronts = [h[1][0][0] for h in cp_state["history"]]
    fronts_fitness = [f.fitness.values[0] for f in fronts]
    fronts_params = lambda param_index: [all_params_types[param_index](f[param_index]) for f in fronts]
    
    # plot fitness
    plt.plot(fronts_fitness)
    
    # plot params
    plt.plot(fronts_params(1))
    plt.show()
      
    
def plot_Basin_map(dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2, stream_gdf, x_locator_interval=0.3, y_locator_interval=0.2, fig=None, ax=None):
    # =========== plot Basin_map ===========
    # get fig, ax
    if not ax:
        fig_Basin_map, ax_Basin_map = plt.subplots()
        
    # plot dem at level0
    dpc_VIC_level0.grid_shp.plot(ax=ax_Basin_map, column="SrtmDEM_mean_Value", alpha=1, legend=True, colormap="terrain", zorder=1,
                                 legend_kwds={"label": "Elevation (m)"})  # terrain gray
    
    # plot basin boundary
    dpc_VIC_level0.basin_shp.plot(ax=ax_Basin_map, facecolor="none", linewidth=2, alpha=1, edgecolor="k", zorder=2)
    dpc_VIC_level0.basin_shp.plot(ax=ax_Basin_map, facecolor="k", alpha=0.2, zorder=3)
    
    # plot river
    stream_gdf.plot(ax=ax_Basin_map, color="b", zorder=4)
    
    # plot gauge
    gauge_lon = dpc_VIC_level1.basin_shp["camels_topo:gauge_lon"].values[0]
    gauge_lat = dpc_VIC_level1.basin_shp["camels_topo:gauge_lat"].values[0]
    ax_Basin_map.plot(gauge_lon, gauge_lat, "r*", markersize=10, mec="k", mew=1, zorder=5)
    
    # set plot boundary and ticks
    set_boundary(ax_Basin_map, dpc_VIC_level0.boundary_grids_edge_x_y)
    set_xyticks(ax_Basin_map, x_locator_interval, y_locator_interval)
    
    # =========== plot grid basin ===========
    fig_grid_basin_level0, ax_grid_basin_level0 = dpc_VIC_level0.plot()
    fig_grid_basin_level1, ax_grid_basin_level1 = dpc_VIC_level1.plot()
    fig_grid_basin_level2, ax_grid_basin_level2 = dpc_VIC_level2.plot()
    
    set_boundary(ax_grid_basin_level0, dpc_VIC_level0.boundary_grids_edge_x_y)
    set_boundary(ax_grid_basin_level1, dpc_VIC_level1.boundary_grids_edge_x_y)
    set_boundary(ax_grid_basin_level2, dpc_VIC_level2.boundary_grids_edge_x_y)
    
    set_xyticks(ax_grid_basin_level0, x_locator_interval, y_locator_interval)
    set_xyticks(ax_grid_basin_level1, x_locator_interval, y_locator_interval)
    set_xyticks(ax_grid_basin_level2, x_locator_interval, y_locator_interval)
    
    # =========== store ===========
    fig_dict = {"fig_Basin_map": fig_Basin_map,
                "fig_grid_basin_level0": fig_grid_basin_level0,
                "fig_grid_basin_level1": fig_grid_basin_level1,
                "fig_grid_basin_level2": fig_grid_basin_level2}

    ax_dict = {"ax_Basin_map": ax_Basin_map,
                "ax_grid_basin_level0": ax_grid_basin_level0,
                "ax_grid_basin_level1": ax_grid_basin_level1,
                "ax_grid_basin_level2": ax_grid_basin_level2}
    
    return fig_dict, ax_dict