# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

## ------------------------ plot utilities ------------------------
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


def plot_selected_map(basin_index, dpc_base, text_name="basin_index", plot_solely=True, column=None, plot_kwgs_set=dict(), fig=None, ax=None):
    """_summary_

    Args:
        basin_index (_type_): _description_
        dpc_base (_type_): _description_
        text_name (str, optional): _description_. Defaults to "basin_index".
        plot_solely (bool, optional): _description_. Defaults to True.
        column (_type_, optional): _description_. Defaults to None.
        plot_kwgs_set (_type_, optional): _description_. Defaults to dict().

    Returns:
        _type_: _description_
    
    usages:
    fig, ax, fig_solely = plot_selected_map(basin_shp_area_excluding.index.to_list(), # [0, 1, 2]
                                        dpc_base,
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
        fig, ax = plotBasins(dpc_base.basin_shp.loc[basin_index, :].to_crs(proj), fig=fig, ax=ax, plot_kwgs=plot_kwgs, column=column)
    elif len(basin_index) == 1:
        fig, ax = plotBasins(dpc_base.basin_shp.loc[[basin_index[0], basin_index[0]], :].to_crs(proj), fig=fig, ax=ax, plot_kwgs=plot_kwgs, column=column)
    else:
        return fig, ax, None
    
    # annotation
    if text_name:  # None means not to plot text
        basinLatCens = np.array([dpc_base.basin_shp.loc[key, "lat_cen"] for key in basin_index])
        basinLonCens = np.array([dpc_base.basin_shp.loc[key, "lon_cen"] for key in basin_index])
        
        for i in range(len(basinLatCens)):
            basinLatCen = basinLatCens[i]
            basinLonCen = basinLonCens[i]
            text_names_dict = {"basin_index": basin_index[i],
                            "hru_id": dpc_base.basin_shp.loc[basin_index[i], "hru_id"],
                            "gauge_id": dpc_base.basin_shp.loc[basin_index[i], "camels_hydro:gauge_id"]}
            
            text_name_plot = text_names_dict[text_name]
            
            ax.text(basinLonCen, basinLatCen, f"{text_name_plot}",
                    horizontalalignment='right',
                    transform=proj,
                    fontdict={"family": "Arial", "fontsize": 5, "color": "b", "weight": "bold"})
    
    # plot solely
    fig_solely = {}
    if plot_solely:
        for i in range(len(basin_index)):
            fig_, ax_ = plotBasins(dpc_base.basin_shp.loc[[basin_index[i], basin_index[i]], :].to_crs(proj), fig=None, ax=None, plot_kwgs=None)
            fig_solely[i] = {"fig": fig_, "ax": ax_}
            
            text_names_dict = {"basin_index": basin_index[i],
                               "hru_id": dpc_base.basin_shp.loc[basin_index[i], "hru_id"],
                               "gauge_id": dpc_base.basin_shp.loc[basin_index[i], "camels_hydro:gauge_id"]}
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