# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Module ``easy_vic_build.tools.plot_func.plot_map``."""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import geopandas as gpd
import os

from ..params_func.params_set import *
from ..dpc_func.basin_grid_func import createArray_from_gridshp
from .plot_utilities import *

## ------------------------ plot map ------------------------
def plotBackground(basin_shp, grid_shp, fig=None, ax=None):
    """
    Plot the background for a given basin and grid shape.

    This function plots the background for a given basin and grid shape on a figure
    and axis. If no axis object is provided, a new figure and axis are created.

    Parameters
    ----------
    basin_shp : shapefile
        The shapefile object containing basin boundaries.
    grid_shp : shapefile
        The shapefile object containing grid boundaries.
    fig : matplotlib.figure.Figure, optional
        The figure object to plot on (default is None, a new figure is created).
    ax : matplotlib.axes.Axes, optional
        The axis object to plot on (default is None, a new axis is created).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object with the background plot.
    ax : matplotlib.axes.Axes
        The axis object with the background plot.
    """
    if not ax:
        fig, ax = plt.subplots()
    plot_kwgs = {"facecolor": "none", "alpha": 0.7, "edgecolor": "k"}
    fig, ax = plotBasins(basin_shp, None, fig, ax, plot_kwgs)
    fig, ax = plotGrids(grid_shp, None, fig, ax)

    return fig, ax


def plotGrids(
    grid_shp, column=None, fig=None, ax=None, plot_kwgs1=None, plot_kwgs2=None
):
    """
    Plot grid shapes and point geometries on a given axis.

    This function plots the grid shapes from the `grid_shp` object on the given `ax` (or creates a new
    figure and axis if `ax` is not provided). Two sets of keyword arguments can be used to customize
    the appearance of the grid shapes and the point geometries.

    Parameters
    ----------
    grid_shp : geopandas.GeoDataFrame
        A GeoDataFrame containing the grid shapes and point geometries to be plotted.
    column : str, optional
        The column in `grid_shp` to use for coloring the grid shapes. If not provided, no coloring is applied.
    fig : matplotlib.figure.Figure, optional
        The figure to plot on. If not provided, a new figure is created.
    ax : matplotlib.axes.Axes, optional
        The axis to plot on. If not provided, a new axis is created.
    plot_kwgs1 : dict, optional
        A dictionary of keyword arguments for customizing the grid shape plot. Defaults to an empty dictionary.
    plot_kwgs2 : dict, optional
        A dictionary of keyword arguments for customizing the point geometry plot. Defaults to an empty dictionary.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object with the grid shapes and point geometries plotted.
    ax : matplotlib.axes.Axes
        The axis object with the grid shapes and point geometries plotted.
    """
    if not ax:
        fig, ax = plt.subplots()
    plot_kwgs1 = dict() if not plot_kwgs1 else plot_kwgs1
    plot_kwgs2 = dict() if not plot_kwgs2 else plot_kwgs2
    plot_kwgs1_ = {"facecolor": "none", "alpha": 0.2, "edgecolor": "gray"}
    plot_kwgs2_ = {
        "facecolor": "none",
        "alpha": 0.5,
        "edgecolor": "gray",
        "markersize": 0.5,
    }

    plot_kwgs1_.update(plot_kwgs1)
    plot_kwgs2_.update(plot_kwgs2)

    grid_shp.plot(ax=ax, column=column, **plot_kwgs1_)
    grid_shp["point_geometry"].plot(ax=ax, **plot_kwgs2_)
    return fig, ax


def plotBasins(basin_shp, column=None, fig=None, ax=None, plot_kwgs=None):
    """
    Plot basin shapes on a given axis.

    This function plots the basin shapes from the `basin_shp` object on the given `ax` (or creates a new
    figure and axis if `ax` is not provided). Additional customization can be done using the `plot_kwgs`
    dictionary.

    Parameters
    ----------
    basin_shp : geopandas.GeoDataFrame
        A GeoDataFrame containing the basin shapes to be plotted.
    column : str, optional
        The column in `basin_shp` to use for coloring the basin shapes. If not provided, no coloring is applied.
    fig : matplotlib.figure.Figure, optional
        The figure to plot on. If not provided, a new figure is created.
    ax : matplotlib.axes.Axes, optional
        The axis to plot on. If not provided, a new axis is created.
    plot_kwgs : dict, optional
        A dictionary of keyword arguments for customizing the basin plot. Defaults to an empty dictionary.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object with the basin shapes plotted.
    ax : matplotlib.axes.Axes
        The axis object with the basin shapes plotted.
    """
    if not ax:
        fig, ax = plt.subplots()
    plot_kwgs = dict() if not plot_kwgs else plot_kwgs
    plot_kwgs_ = {"legend": True}
    plot_kwgs_.update(plot_kwgs)
    basin_shp.plot(ax=ax, column=column, **plot_kwgs_)

    return fig, ax


def setBoundary(ax, boundary_x_min, boundary_x_max, boundary_y_min, boundary_y_max):
    """
    Set the boundary limits for the x and y axes.

    This function adjusts the x and y axis limits of the provided axis object (`ax`) based on the
    given boundary values.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object to set the boundaries on.
    boundary_x_min : float
        The minimum value for the x-axis.
    boundary_x_max : float
        The maximum value for the x-axis.
    boundary_y_min : float
        The minimum value for the y-axis.
    boundary_y_max : float
        The maximum value for the y-axis.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis object with the updated boundaries.
    """
    ax.set_xlim([boundary_x_min, boundary_x_max])
    ax.set_ylim([boundary_y_min, boundary_y_max])
    return ax


def plot_basemap(
    fig=None,
    ax=None,
    set_xyticks_bool=True,
    extent=None,
    x_locator_interval=15,
    y_locator_interval=10,
    yticks_rotation=0,
):
    """
    Plot a basemap of the United States with customizable tick settings.

    This function creates a map of the United States, including coastlines, rivers, lakes, and state
    boundaries, and allows for customization of x and y axis ticks, including their interval and rotation.

    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        The figure to plot on. If not provided, a new figure is created.
    ax : matplotlib.axes.Axes, optional
        The axis to plot on. If not provided, a new axis is created.
    set_xyticks_bool : bool, optional
        If True, sets the x and y axis ticks to a specified interval (default is True).
    x_locator_interval : int, optional
        The interval for the x-axis ticks (default is 15).
    y_locator_interval : int, optional
        The interval for the y-axis ticks (default is 10).
    yticks_rotation : int, optional
        The rotation angle for the y-axis ticks (default is 0).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object with the US basemap plotted.
    ax : matplotlib.axes.Axes
        The axis object with the US basemap plotted.
    """
    proj = ccrs.PlateCarree()
    # extent = [-125, -66.5, 24.5, 50.5]
    # extent = [-125, 24.5, -66.5, 50.5]

    # get fig, ax
    if not ax:
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0, 0.85, 1], projection=proj)

    # add background
    alpha = 0.3
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, zorder=10)
    ax.add_feature(cfeature.LAND, alpha=alpha)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(
        cfeature.RIVERS.with_scale("50m"), linewidth=0.5, zorder=10, alpha=alpha
    )
    ax.add_feature(
        cfeature.LAKES.with_scale("50m"),
        linewidth=0.2,
        edgecolor="k",
        zorder=10,
        alpha=alpha,
    )
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="gray", zorder=10)

    # set ticks
    if set_xyticks_bool:
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks([-90, -45, 0, 45, 90])
        set_xyticks(
            ax,
            x_locator_interval=x_locator_interval,
            y_locator_interval=y_locator_interval,
            yticks_rotation=yticks_rotation,
        )

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


def plot_selected_map(
    basin_index,
    dpc,
    text_name="basin_index",
    plot_solely=True,
    column=None,
    plot_kwgs_set=dict(),
    fig=None,
    ax=None,
):
    """
    Plot selected basins on a map and optionally annotate and plot basins individually.

    Parameters
    ----------
    basin_index : list
        List of basin indices to be plotted.
    dpc : object
        Data processing class instance containing the basin shapefile and related data.
    text_name : str, optional
        The type of text annotation to plot. Defaults to "basin_index".
    plot_solely : bool, optional
        Whether to plot each selected basin separately. Defaults to True.
    column : str, optional
        The column name from the shapefile to be used for coloring. Defaults to None.
    plot_kwgs_set : dict, optional
        Additional keyword arguments to customize the plot (e.g., color map). Defaults to an empty dictionary.
    fig : matplotlib.figure.Figure, optional
        The figure object to use for the plot. If None, a new figure will be created. Defaults to None.
    ax : matplotlib.axes.Axes, optional
        The axis object to use for the plot. If None, a new axis will be created. Defaults to None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axis object containing the plot.
    fig_solely : dict or None
        A dictionary of figures and axes for each basin if `plot_solely` is True, otherwise None.

    Notes
    -----
    - The function uses a PlateCarree projection for the map.
    - The function can annotate each basin with its `basin_index` or other attributes (e.g., `hru_id` or `gauge_id`).
    - The basins are plotted using the `plotBasins` function.

    Usages:
    -----
    fig, ax, fig_solely = plot_selected_map(basin_shp_area_excluding.index.to_list(), # [0, 1, 2]
                                        dpc,
                                        text_name="basin_index",  # "basin_index", None,
                                        plot_solely=False,
                                        column=None,  # "camels_clim:aridity",  # None
                                        plot_kwgs_set=dict()) # {"cmap": plt.cm.hot})  # dict()
    """
    # background
    proj = ccrs.PlateCarree()
    extent = [-125, 24.5, -66.5, 50.5]  # [-125, -66.5, 24.5, 50.5]
    alpha = 0.3
    if not fig:
        fig = plt.figure(dpi=300)
        ax = fig.add_axes([0.05, 0, 0.9, 1], projection=proj)

        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, zorder=10)
        ax.add_feature(cfeature.LAND, alpha=alpha)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(
            cfeature.RIVERS.with_scale("50m"), linewidth=0.5, zorder=10, alpha=alpha
        )
        ax.add_feature(
            cfeature.LAKES.with_scale("50m"),
            linewidth=0.2,
            edgecolor="k",
            zorder=10,
            alpha=alpha,
        )

    ax.set_extent(extent, crs=proj)

    # plot
    plot_kwgs = {"facecolor": "r", "alpha": 0.7, "edgecolor": "k", "linewidth": 0.2}
    plot_kwgs.update(plot_kwgs_set)
    if len(basin_index) > 1:
        fig, ax = plotBasins(
            dpc.basin_shp.loc[basin_index, :].to_crs(proj),
            fig=fig,
            ax=ax,
            plot_kwgs=plot_kwgs,
            column=column,
        )
    elif len(basin_index) == 1:
        fig, ax = plotBasins(
            dpc.basin_shp.loc[[basin_index[0], basin_index[0]], :].to_crs(proj),
            fig=fig,
            ax=ax,
            plot_kwgs=plot_kwgs,
            column=column,
        )
    else:
        return fig, ax, None

    # annotation
    if text_name:  # None means not to plot text
        basinLatCens = np.array(
            [dpc.basin_shp.loc[key, "lat_cen"] for key in basin_index]
        )
        basinLonCens = np.array(
            [dpc.basin_shp.loc[key, "lon_cen"] for key in basin_index]
        )

        for i in range(len(basinLatCens)):
            basinLatCen = basinLatCens[i]
            basinLonCen = basinLonCens[i]
            text_names_dict = {
                "basin_index": basin_index[i],
                "hru_id": dpc.basin_shp.loc[basin_index[i], "hru_id"],
                "gauge_id": dpc.basin_shp.loc[basin_index[i], "camels_hydro:gauge_id"],
            }

            text_name_plot = text_names_dict[text_name]

            ax.text(
                basinLonCen,
                basinLatCen,
                f"{text_name_plot}",
                horizontalalignment="right",
                transform=proj,
                fontdict={
                    "family": "Arial",
                    "fontsize": 5,
                    "color": "b",
                    "weight": "bold",
                },
            )

    # plot solely
    fig_solely = {}
    if plot_solely:
        for i in range(len(basin_index)):
            fig_, ax_ = plotBasins(
                dpc.basin_shp.loc[[basin_index[i], basin_index[i]], :].to_crs(proj),
                fig=None,
                ax=None,
                plot_kwgs=None,
            )
            fig_solely[i] = {"fig": fig_, "ax": ax_}

            text_names_dict = {
                "basin_index": basin_index[i],
                "hru_id": dpc.basin_shp.loc[basin_index[i], "hru_id"],
                "gauge_id": dpc.basin_shp.loc[basin_index[i], "camels_hydro:gauge_id"],
            }
            text_name_plot = text_names_dict[text_name]

            ax_.set_title(text_name_plot)
    else:
        fig_solely = None

    return fig, ax, fig_solely


def plotShp(
    basinShp_original,
    basinShp,
    grid_shp,
    intersects_grids,
    boundary_x_min,
    boundary_x_max,
    boundary_y_min,
    boundary_y_max,
    fig=None,
    ax=None,
):
    """
    Plot shapefiles of basins, grids, and intersecting grids on a map with specified boundaries.

    Parameters
    ----------
    basinShp_original : geopandas.GeoDataFrame
        Original basin shapefile to be plotted with an outline.
    basinShp : geopandas.GeoDataFrame
        Basin shapefile to be plotted on top of the original basin shapefile.
    grid_shp : geopandas.GeoDataFrame
        Grid shapefile containing the geometry and point geometry to be plotted.
    intersects_grids : geopandas.GeoDataFrame
        Shapefile representing the intersection of grids to be plotted.
    boundary_x_min : float
        Minimum x-coordinate (longitude) for the plot boundary.
    boundary_x_max : float
        Maximum x-coordinate (longitude) for the plot boundary.
    boundary_y_min : float
        Minimum y-coordinate (latitude) for the plot boundary.
    boundary_y_max : float
        Maximum y-coordinate (latitude) for the plot boundary.
    fig : matplotlib.figure.Figure, optional
        The figure object to use for the plot. If None, a new figure will be created. Defaults to None.
    ax : matplotlib.axes.Axes, optional
        The axis object to use for the plot. If None, a new axis will be created. Defaults to None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axis object containing the plot.

    Notes
    -----
    - The plot includes multiple layers: original basin, basin, grid geometry, grid points, and intersecting grids.
    - The boundaries of the plot are set using the provided min and max x and y coordinates.
    """
    if not ax:
        fig, ax = plt.subplots()
    basinShp_original.plot(ax=ax, facecolor="none", alpha=0.7, edgecolor="k")
    basinShp.plot(ax=ax)
    grid_shp["geometry"].plot(ax=ax, facecolor="none", edgecolor="gray", alpha=0.2)
    grid_shp["point_geometry"].plot(
        ax=ax, markersize=0.5, edgecolor="gray", facecolor="gray", alpha=0.5
    )
    intersects_grids.plot(ax=ax, facecolor="r", edgecolor="gray", alpha=0.2)
    ax.set_xlim([boundary_x_min, boundary_x_max])
    ax.set_ylim([boundary_y_min, boundary_y_max])

    return fig, ax


def plotLandCover(
    basinShp_original,
    basinShp,
    grid_shp,
    intersects_grids,
    boundary_x_min,
    boundary_x_max,
    boundary_y_min,
    boundary_y_max,
    fig=None,
    ax=None,
):
    """
    Plot land cover data along with basin, grid, and intersection information on a map.

    Parameters
    ----------
    basinShp_original : geopandas.GeoDataFrame
        Original basin shapefile to be plotted with an outline.
    basinShp : geopandas.GeoDataFrame
        Basin shapefile to be plotted on top of the original basin shapefile.
    grid_shp : geopandas.GeoDataFrame
        Grid shapefile containing land cover classification data.
    intersects_grids : geopandas.GeoDataFrame
        Shapefile representing the intersection of grids to be plotted.
    boundary_x_min : float
        Minimum x-coordinate (longitude) for the plot boundary.
    boundary_x_max : float
        Maximum x-coordinate (longitude) for the plot boundary.
    boundary_y_min : float
        Minimum y-coordinate (latitude) for the plot boundary.
    boundary_y_max : float
        Maximum y-coordinate (latitude) for the plot boundary.
    fig : matplotlib.figure.Figure, optional
        The figure object to use for the plot. If None, a new figure will be created. Defaults to None.
    ax : matplotlib.axes.Axes, optional
        The axis object to use for the plot. If None, a new axis will be created. Defaults to None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axis object containing the plot.

    Notes
    -----
    - The plot includes multiple layers: original basin, basin, grid land cover classification,
      and intersecting grids.
    - A color map is applied to the land cover classification with corresponding ticks in the legend.
    - The boundaries of the plot are set using the provided min and max x and y coordinates.
    """
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
    grid_shp.plot(
        ax=ax,
        column="major_umd_landcover_classification_grids",
        alpha=0.4,
        legend=True,
        colormap=cmap,
        norm=norm,
        legend_kwds={
            "label": "major_umd_landcover_classification_grids",
            "shrink": 0.8,
        },
    )
    intersects_grids.plot(ax=ax, facecolor="none", edgecolor="k", alpha=0.7)
    ax.set_xlim([boundary_x_min, boundary_x_max])
    ax.set_ylim([boundary_y_min, boundary_y_max])

    ax_cb = fig.axes[1]
    ax_cb.set_yticks(ticks_position)
    ax_cb.set_yticklabels(ticks)

    return fig, ax


def plotHWSDSoilData(
    basinShp_original,
    basinShp,
    grid_shp,
    intersects_grids,
    boundary_x_min,
    boundary_x_max,
    boundary_y_min,
    boundary_y_max,
    fig=None,
    ax=None,
    fig_T=None,
    ax_T=None,
    fig_S=None,
    ax_S=None,
):
    """
    Plot soil data from HWSD, USDA texture classes, and basin information on multiple maps.

    Parameters
    ----------
    basinShp_original : geopandas.GeoDataFrame
        Original basin shapefile to be plotted with an outline.
    basinShp : geopandas.GeoDataFrame
        Basin shapefile to be plotted on top of the original basin shapefile.
    grid_shp : geopandas.GeoDataFrame
        Grid shapefile containing soil data including HWSD and USDA texture classes.
    intersects_grids : geopandas.GeoDataFrame
        Shapefile representing the intersection of grids to be plotted.
    boundary_x_min : float
        Minimum x-coordinate (longitude) for the plot boundary.
    boundary_x_max : float
        Maximum x-coordinate (longitude) for the plot boundary.
    boundary_y_min : float
        Minimum y-coordinate (latitude) for the plot boundary.
    boundary_y_max : float
        Maximum y-coordinate (latitude) for the plot boundary.
    fig : matplotlib.figure.Figure, optional
        The figure object to use for the plot. If None, a new figure will be created. Defaults to None.
    ax : matplotlib.axes.Axes, optional
        The axis object to use for the plot. If None, a new axis will be created. Defaults to None.
    fig_T : matplotlib.figure.Figure, optional
        The figure object for T_USDA_TEX_CLASS plot. If None, a new figure will be created. Defaults to None.
    ax_T : matplotlib.axes.Axes, optional
        The axis object for T_USDA_TEX_CLASS plot. If None, a new axis will be created. Defaults to None.
    fig_S : matplotlib.figure.Figure, optional
        The figure object for S_USDA_TEX_CLASS plot. If None, a new figure will be created. Defaults to None.
    ax_S : matplotlib.axes.Axes, optional
        The axis object for S_USDA_TEX_CLASS plot. If None, a new axis will be created. Defaults to None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the main plot.
    ax : matplotlib.axes.Axes
        The axis object containing the main plot.
    fig_S : matplotlib.figure.Figure
        The figure object containing the S_USDA_TEX_CLASS plot.
    ax_S : matplotlib.axes.Axes
        The axis object containing the S_USDA_TEX_CLASS plot.
    fig_T : matplotlib.figure.Figure
        The figure object containing the T_USDA_TEX_CLASS plot.
    ax_T : matplotlib.axes.Axes
        The axis object containing the T_USDA_TEX_CLASS plot.

    Notes
    -----
    - Three different maps are created for HWSD soil data, T_USDA_TEX_CLASS, and S_USDA_TEX_CLASS.
    - Each map is plotted with the corresponding soil classification data overlaid on the basin and grid shapefiles.
    - The boundaries of the plots are set using the provided min and max x and y coordinates.
    - Legends for each map are created with the respective soil classification.
    """
    if not ax:
        fig, ax = plt.subplots()
    basinShp_original.plot(ax=ax, facecolor="none", alpha=0.7, edgecolor="k")
    basinShp.plot(ax=ax)
    grid_shp.plot(
        ax=ax,
        column="HWSD_BIL_Value",
        alpha=0.4,
        legend=True,
        colormap="Accent",
        legend_kwds={"label": "HWSD_BIL_Value", "shrink": 0.8},
    )
    intersects_grids.plot(ax=ax, facecolor="none", edgecolor="k", alpha=0.7)
    ax.set_xlim([boundary_x_min, boundary_x_max])
    ax.set_ylim([boundary_y_min, boundary_y_max])

    # T_USDA_TEX_CLASS
    if not ax_T:
        fig_T, ax_T = plt.subplots()
    basinShp_original.plot(ax=ax_T, facecolor="none", alpha=0.7, edgecolor="k")
    basinShp.plot(ax=ax_T)
    grid_shp.plot(
        ax=ax_T,
        column="T_USDA_TEX_CLASS",
        alpha=0.4,
        legend=True,
        colormap="Accent",
        legend_kwds={"label": "T_USDA_TEX_CLASS", "shrink": 0.8},
    )
    intersects_grids.plot(ax=ax_T, facecolor="none", edgecolor="k", alpha=0.7)
    ax_T.set_xlim([boundary_x_min, boundary_x_max])
    ax_T.set_ylim([boundary_y_min, boundary_y_max])

    # S_USDA_TEX_CLASS
    if not ax_S:
        fig_S, ax_S = plt.subplots()
    basinShp_original.plot(ax=ax_S, facecolor="none", alpha=0.7, edgecolor="k")
    basinShp.plot(ax=ax_S)
    grid_shp.plot(
        ax=ax_S,
        column="S_USDA_TEX_CLASS",
        alpha=0.4,
        legend=True,
        colormap="Accent",
        legend_kwds={"label": "S_USDA_TEX_CLASS", "shrink": 0.8},
    )
    intersects_grids.plot(ax=ax_S, facecolor="none", edgecolor="k", alpha=0.7)
    ax_S.set_xlim([boundary_x_min, boundary_x_max])
    ax_S.set_ylim([boundary_y_min, boundary_y_max])

    return fig, ax, fig_S, ax_S, fig_T, ax_T



def plot_Basin_map(
    dpc_VIC_level0,
    dpc_VIC_level1,
    dpc_VIC_level2,
    stream_gdf,
    gauge_coord,
    x_locator_interval=0.3,
    y_locator_interval=0.2,
    fig=None,
    ax=None,
    dem_column="SrtmDEM_mean_Value",
    **kwargs
):
    """
    Plot the basin map including elevation, basin boundary, river network, and gauge location.

    Parameters
    ----------
    dpc_VIC_level0 : object
        A VIC model object at level 0 containing grid and basin shapefiles.
    dpc_VIC_level1 : object
        A VIC model object at level 1 containing grid and basin shapefiles.
    dpc_VIC_level2 : object
        A VIC model object at level 2 containing grid and basin shapefiles.
    stream_gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing the river network to be plotted.
    gauge_coord: [lon, lat]
    x_locator_interval : float, optional
        The interval for x-axis ticks. Defaults to 0.3.
    y_locator_interval : float, optional
        The interval for y-axis ticks. Defaults to 0.2.
    fig : matplotlib.figure.Figure, optional
        The figure object to use for the plot. If None, a new figure will be created. Defaults to None.
    ax : matplotlib.axes.Axes, optional
        The axis object to use for the plot. If None, a new axis will be created. Defaults to None.

    Returns
    -------
    fig_dict : dict
        A dictionary containing the figure objects for the basin map and grid basins.
    ax_dict : dict
        A dictionary containing the axis objects for the basin map and grid basins.

    Notes
    -----
    - The function plots the basin map with layers including the DEM (elevation),
      basin boundary, river network, and gauge location.
    - It also generates plots for different grid levels (level 0, 1, and 2) of the VIC model.
    """
    # =========== plot Basin_map ===========
    # get fig, ax
    if not ax:
        fig_Basin_map, ax_Basin_map = plt.subplots(**kwargs)

    # get data
    basin_shp_level0 = dpc_VIC_level0.get_data_from_cache("basin_shp")[0]
    grid_shp_level0 = dpc_VIC_level0.get_data_from_cache("grid_shp")[0]
    
    basin_shp_level1 = dpc_VIC_level1.get_data_from_cache("basin_shp")[0]
    grid_shp_level1 = dpc_VIC_level1.get_data_from_cache("grid_shp")[0]
    
    basin_shp_level2 = dpc_VIC_level2.get_data_from_cache("basin_shp")[0]
    grid_shp_level2 = dpc_VIC_level2.get_data_from_cache("grid_shp")[0]
    
    # plot dem at level0
    dpc_VIC_level0.get_data_from_cache("dem")[0].plot(
        ax=ax_Basin_map,
        column=dem_column,
        alpha=1,
        legend=True,
        colormap="terrain",
        zorder=1,
        legend_kwds={"label": "Elevation (m)"},
    )  # terrain gray
    
    # plot basin boundary
    basin_shp_level0.plot(
        ax=ax_Basin_map, facecolor="none", linewidth=2, alpha=1, edgecolor="k", zorder=2
    )
    basin_shp_level0.plot(ax=ax_Basin_map, facecolor="k", alpha=0.2, zorder=3)

    # plot river
    stream_gdf.plot(ax=ax_Basin_map, color="b", zorder=4)
        
    # plot gauge
    ax_Basin_map.plot(
        gauge_coord[0], gauge_coord[1], "r*", markersize=10, mec="k", mew=1, zorder=5
    )  # gauge_coord[lon, lat]

    # set plot boundary and ticks
    set_boundary(ax_Basin_map, grid_shp_level0.createBoundaryShp()[-1])
    set_xyticks(ax_Basin_map, x_locator_interval, y_locator_interval)

    # =========== plot grid basin ===========
    fig_grid_basin_level0, ax_grid_basin_level0 = dpc_VIC_level0.plot()
    fig_grid_basin_level1, ax_grid_basin_level1 = dpc_VIC_level1.plot()
    fig_grid_basin_level2, ax_grid_basin_level2 = dpc_VIC_level2.plot()
    
    set_boundary(ax_grid_basin_level0, grid_shp_level0.createBoundaryShp()[-1])
    set_boundary(ax_grid_basin_level1, grid_shp_level1.createBoundaryShp()[-1])
    set_boundary(ax_grid_basin_level2, grid_shp_level2.createBoundaryShp()[-1])

    set_xyticks(ax_grid_basin_level0, x_locator_interval, y_locator_interval)
    set_xyticks(ax_grid_basin_level1, x_locator_interval, y_locator_interval)
    set_xyticks(ax_grid_basin_level2, x_locator_interval, y_locator_interval)

    # =========== store ===========
    fig_dict = {
        "fig_Basin_map": fig_Basin_map,
        "fig_grid_basin_level0": fig_grid_basin_level0,
        "fig_grid_basin_level1": fig_grid_basin_level1,
        "fig_grid_basin_level2": fig_grid_basin_level2,
    }

    ax_dict = {
        "ax_Basin_map": ax_Basin_map,
        "ax_grid_basin_level0": ax_grid_basin_level0,
        "ax_grid_basin_level1": ax_grid_basin_level1,
        "ax_grid_basin_level2": ax_grid_basin_level2,
    }

    return fig_dict, ax_dict


def plot_basin_map_combine(
    evb_dir,
    evb_dir_hydroanalysis,
    dpc_VIC_level0,
    dpc_VIC_level1,
    dpc_VIC_level3,
    figsize=(12, 8),
    grid_kwarg={"left": 0.06, "right": 0.99, "bottom": 0.05, "top": 0.98, "hspace": 0.1, "wspace": 0.15},
    ax1_box_aspect_factor=1.5,
    x_locator_interval_landsurface=0.47, y_locator_interval_landsurface=0.5,
    x_locator_interval_grid=0.24, y_locator_interval_grid=0.3
):  
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
    fig, ax1 = plot_basemap(fig=fig, ax=ax1, set_xyticks_bool=True, x_locator_interval=8, y_locator_interval=8, yticks_rotation=90)
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


def plot_river_network(G, river_paths=None, figsize=(12, 8), mask_by=None, threshold_label=None, labeled_nodes=None):
    import networkx as nx
    from ..routing_func.river_network import get_display_positions
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # plot set
    color_scheme = {
        'river_node': '#fde725', # '#440154',
        'hillslope_node': '#9E9E9E',
        'sink_node': 'k',
        'river_edge': 'r',
        'normal_edge': '#E0E0E0',
        'background': '#F8F8F8'
    }

    basic_node_size = 50
    basic_edge_width = 0.8
    
    # background
    fig.patch.set_facecolor('white')
    ax.set_facecolor(color_scheme['background'])
    
    #* pos
    pos = get_display_positions(G)
    
    if not pos:
        print("no pos founded, using spring layout")
        pos = nx.spring_layout(G)
    
    # nodes
    all_nodes = list(G.nodes())
    if mask_by is not None:
        all_nodes = [n for n in all_nodes if G.nodes[n].get("mask", 0) == 1]
    river_nodes = [n for n in all_nodes if G.nodes[n].get("node_type", "hillslope") == "river"]
    hillslope_sink_nodes = [n for n in all_nodes if n not in river_nodes]
    
    nx.draw_networkx_nodes(
        G, pos, nodelist=hillslope_sink_nodes, 
        node_color=[color_scheme['hillslope_node'] if G.nodes[n].get('node_type') != 'sink' else color_scheme['sink_node'] for n in hillslope_sink_nodes],
        node_size=[basic_node_size-30 if G.nodes[n].get('node_type') != 'sink' else basic_node_size+30 for n in hillslope_sink_nodes],
        alpha=0.5, ax=ax
    )
    
    river_nodes_cmap = plt.cm.viridis_r
    original_acc_values = [G.nodes[node].get('flow_acc', 0) for node in river_nodes]
    log_acc_values = np.log1p(original_acc_values)
    river_nodes_norm = mcolors.Normalize(vmin=min(log_acc_values), vmax=max(log_acc_values))
    vmin_adjusted = min(log_acc_values) + 0.5 * (max(log_acc_values) - min(log_acc_values))
    river_nodes_norm = mcolors.Normalize(vmin=vmin_adjusted, vmax=max(log_acc_values))
    
    nx.draw_networkx_nodes(
        G, pos, nodelist=river_nodes,
        node_color=river_nodes_cmap(river_nodes_norm(log_acc_values)),
        node_size=[basic_node_size + min(G.nodes[n].get('flow_acc', 0)/100, 100) for n in river_nodes],
        edgecolors='darkblue',
        alpha=0.8, ax=ax
    )
    
    if labeled_nodes is not None:
        if not isinstance(labeled_nodes, list):
            labeled_nodes = [labeled_nodes]
        
        nx.draw_networkx_nodes(
            G, pos, nodelist=labeled_nodes,
            node_color='none',
            node_shape='^',
            node_size=basic_node_size*2.5,
            edgecolors='blue',  # royalblue
            linewidths=1.8,
            alpha=1, ax=ax
        )
        
        label_offset = 0.025
        for n in labeled_nodes:
            x, y = pos[n]
            ax.text(x + label_offset, y + label_offset, str(n),
                    fontsize=9, fontweight='bold', color='darkblue',
                    ha='left', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))
    
    # normal edges
    all_edges = list(G.edges())
    if mask_by is not None:
        if mask_by == "from":
            all_edges = [e for e in all_edges if G.edges[e].get('from_mask') == 1]
        elif mask_by == "to":
            all_edges = [e for e in all_edges if G.edges[e].get('to_mask') == 1]
        elif mask_by == "both":
            all_edges = [e for e in all_edges if G.edges[e].get('from_mask') == 1 and G.edges[e].get('to_mask') == 1]
        elif mask_by == "either":
            all_edges = [e for e in all_edges if G.edges[e].get('from_mask') == 1 or G.edges[e].get('to_mask') == 1]
    
    normal_edges = [e for e in all_edges if not 
                    (G.nodes[e[0]].get('is_river', False) and 
                     G.nodes[e[1]].get('is_river', False))]
    
    nx.draw_networkx_edges(
        G, pos, edgelist=normal_edges,
        edge_color=color_scheme['normal_edge'],
        width=basic_edge_width, alpha=0.8, arrows=False, ax=ax
    )
    
    # river edges (river paths)
    river_edges = []
    if river_paths is not None:
        for path in river_paths:
            for i in range(len(path) - 1):
                if G.has_edge(path[i], path[i+1]) and (path[i], path[i+1]) not in river_edges:
                    edge = (path[i], path[i+1])
                    
                    if mask_by is not None:
                        edge_data = G.edges[edge]
                        from_mask = edge_data.get('from_mask', 0)
                        to_mask = edge_data.get('to_mask', 0)
                        
                        if mask_by == "from" and from_mask != 1:
                            continue
                        elif mask_by == "to" and to_mask != 1:
                            continue
                        elif mask_by == "both" and (from_mask != 1 or to_mask != 1):
                            continue
                        elif mask_by == "either" and (from_mask != 1 and to_mask != 1):
                            continue
                        
                    river_edges.append((path[i], path[i+1]))
    else:
        river_edges = [e for e in all_edges if 
                      G.nodes[e[0]].get('is_river', False) and 
                      G.nodes[e[1]].get('is_river', False)]
    
    nx.draw_networkx_edges(
        G, pos, edgelist=river_edges,
        edge_color=color_scheme["river_edge"], width=basic_edge_width+0.5, alpha=0.8,
        arrows=True, arrowsize=10, arrowstyle="->",
        node_size=20,
    )
    
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_scheme['river_node'], markersize=10, markeredgecolor='darkblue', label='River Node'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_scheme['hillslope_node'], markersize=8, label='Hillslope Node'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_scheme['sink_node'], markersize=10, alpha=0.5, label='Sink Node'),
        Line2D([0], [0], color=color_scheme['river_edge'], linewidth=2, label='River Edge'),
        Line2D([0], [0], color=color_scheme['normal_edge'], linewidth=2, label='Normal Edge')
    ]
    
    ax.legend(
        handles=legend_handles, loc='best', frameon=True, facecolor='white', edgecolor='black',
        prop={'family': 'Arial', 'size': 12}
    )
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)
    
    cbar_norm = mcolors.Normalize(vmin=np.expm1(vmin_adjusted), vmax=max(original_acc_values))
    sm = plt.cm.ScalarMappable(cmap=river_nodes_cmap, norm=cbar_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label(
        'Flow Accumulation', rotation=270, labelpad=15,
        fontfamily="Arial", fontsize=14,
    )
    
    title = "River network topology" if threshold_label is None else f"River network topology (threshold {threshold_label})"
    ax.set_title(
        title,
        fontdict={
            "fontfamily": 'Arial',
            "fontweight": 'bold',
            "fontsize": 16,
        },
        pad=10.0
    )
    
    plt.tight_layout(pad=3.0)
    plt.show(block=True)
    
    return fig, ax


def soft_hillshade(dem, vert_exag=0.5, azdeg=280, altdeg=20):
    x, y = np.gradient(dem * vert_exag)
    slope = np.pi/2 - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)

    az = np.deg2rad(azdeg)
    alt = np.deg2rad(altdeg)

    shaded = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    shaded = (shaded - shaded.min()) / (shaded.max() - shaded.min())
    return shaded


def light_gray_dem(dem):
    d = (dem - np.nanmin(dem)) / (np.nanmax(dem) - np.nanmin(dem))
    d = 0.7 * d + 0.3
    d = d**0.8
    return d


def plot_dem(grid_shp_level0, ax=None):
    # plot
    if ax is None:
        fig, ax = plt.subplots()
        
    grid_array_dem, stand_grids_lon, stand_grids_lat, rows_index, cols_index = createArray_from_gridshp(
        grid_shp_level0,
        value_column="ASTGTM_DEM_mean_Value",
        values_list=None,
        grid_res=None,
        dtype=float,
        missing_value=np.nan,
        plot=False,
        reverse_lat=True,
    )
    
    xmin = np.min(stand_grids_lon)
    xmax = np.max(stand_grids_lon)
    ymin = np.min(stand_grids_lat)
    ymax = np.max(stand_grids_lat)
    
    dem_light = light_gray_dem(grid_array_dem)
    hs = soft_hillshade(grid_array_dem)
    
    ax.imshow(
        dem_light,
        cmap="terrain",
        extent=[xmin, xmax, ymin, ymax],
        origin="upper",
        alpha=0.35,
        interpolation="bicubic",
    )
    
    ax.imshow(
        hs,
        cmap="gray",
        extent=[xmin, xmax, ymin, ymax],
        origin="upper",
        alpha=0.25,
        interpolation="bicubic",
    )
    
    return ax


def plot_Orthographic_basin_shp():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(-10, 45))
    
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black')

    ax.set_global()
    ax.gridlines()
    ax.set_aspect('equal', 'box')
    
    return fig, ax