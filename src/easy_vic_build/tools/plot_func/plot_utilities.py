# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Module ``easy_vic_build.tools.plot_func.plot_utilities``."""

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, MultipleLocator

from ..geo_func.create_gdf import CreateGDF
from ..params_func.params_set import *

## ------------------------ plot utilities ------------------------
def get_colorbar(
    vmin,
    vmax,
    cmap,
    figsize=(2, 6),
    subplots_adjust={"left": 0.5},
    cb_label="",
    cb_label_kwargs={},
    cb_kwargs={},
):
    """
    Create a colorbar for visualizing data range using a given colormap.

    Parameters
    ----------
    vmin : float
        The minimum value for the colormap normalization.
    vmax : float
        The maximum value for the colormap normalization.
    cmap : matplotlib.colors.Colormap
        The colormap to be used for the colorbar.
    figsize : tuple, optional
        The size of the figure (width, height). Default is (2, 6).
    subplots_adjust : dict, optional
        A dictionary to adjust the subplot layout. The default is {"left": 0.5}.
        This allows fine control over the subplot positioning (e.g., "top", "bottom", "right").
    cb_label : str, optional
        The label for the colorbar. Default is an empty string.
    cb_label_kwargs : dict, optional
        Additional keyword arguments to customize the colorbar label. Default is an empty dictionary.
    cb_kwargs : dict, optional
        Additional keyword arguments to customize the colorbar itself. This can include parameters like
        `orientation` (either 'vertical' or 'horizontal'). Default is an empty dictionary.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the colorbar.
    ax : matplotlib.axes.Axes
        The axis object containing the colorbar.
    norm : matplotlib.colors.Normalize
        The normalization used for the colormap.
    sm : matplotlib.cm.ScalarMappable
        The scalar mappable object for the colorbar.

    Notes
    -----
    The function creates a colorbar using a given colormap and normalizes the values between
    `vmin` and `vmax`. The subplot layout can be adjusted via `subplots_adjust`, and the colorbar
    label can be set with `cb_label` and additional arguments provided through `cb_label_kwargs`.
    """
    # cb_kwargs:
    #   orientation：None or {'vertical', 'horizontal'}
    #   subplots_adjust={"left": 0.5} top/bottom/right, set  0.5
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    # usage
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(**subplots_adjust)

    cbar = fig.colorbar(sm, cax=ax, **cb_kwargs)

    if cb_label is not None:
        cbar.set_label(cb_label, **cb_label_kwargs)
    
    if "ticks" in cb_kwargs:
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda i, _: cb_kwargs["ticks"][int(i)]))

    return fig, ax, norm, sm


def get_NDVI_cmap():
    """
    Create a custom colormap for NDVI (Normalized Difference Vegetation Index) values.

    This function generates a colormap that represents NDVI values, ranging from 0 to 1, using
    a series of colors that reflect vegetation health. The colormap starts with a light brown
    color for low NDVI values and transitions to deep green for high NDVI values.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        A custom colormap designed for NDVI visualization, with color transitions at specified NDVI values.

    Notes
    -----
    The NDVI values are mapped to the following colors:
        - NDVI = 0: Light brown (near white).
        - NDVI = 0.15: Light brown.
        - NDVI = 0.3: Dark brown and light green.
        - NDVI = 0.65: Light green.
        - NDVI = 1: Deep green.

    This colormap can be used to visualize NDVI data in a meaningful way, where lower NDVI values
    represent less vegetation and higher NDVI values represent more vegetation.
    """
    colors = [
        (0.0, "#F5F5DC"),  # NDVI = 0: Light brown (near white)
        (0.15, "#D2B48C"),  # NDVI = 0.15: Light brown
        (0.3, "#8B4513"),  # NDVI = 0.3: Dark brown
        (0.3, "#F0FFF0"),  # NDVI = 0.3: Light green (near white)
        (0.65, "#90EE90"),  # NDVI = 0.65: Light green
        (1.0, "#006400"),  # NDVI = 1: Deep green
    ]

    ndvi_cmap = mcolors.LinearSegmentedColormap.from_list(
        name="ndvi_cmap", colors=colors
    )
    return ndvi_cmap


def get_UMD_LULC_cmap():
    """
    Create a colormap for UMD Land Use and Land Cover (LULC) classification.

    This function generates a colormap for the UMD LULC classes, which categorize land cover
    types such as water bodies, forests, grasslands, and urban areas. The colormap maps specific
    colors to each LULC class and returns the colormap, normalization, and other related information
    for visualization.

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
        A colormap object that maps LULC classes to specific colors.
    norm : matplotlib.colors.BoundaryNorm
        A normalization object that maps the LULC class values to the colormap.
    ticks : list of str
        A list of LULC class labels, corresponding to the categories.
    ticks_position : list of int
        A list of positions for each LULC class, used for tick placement in a colorbar.
    colorlist : list of str
        A list of hex color values representing the colors for each LULC class.
    colorlevel : numpy.ndarray
        An array of color boundaries that corresponds to the LULC class levels.

    Notes
    -----
    The UMD LULC classes and their corresponding colors are as follows:
        0.0: Water               -> Deep Blue
        1.0: Evergreen Needleleaf Forest -> Dark Green
        2.0: Evergreen Broadleaf Forest -> Light Green
        3.0: Deciduous Needleleaf Forest -> Brownish
        4.0: Deciduous Broadleaf Forest -> Orange
        5.0: Mixed Forest        -> Purple
        6.0: Woodland            -> Yellow-Green
        7.0: Wooded Grassland    -> Gray
        8.0: Closed Shrubland    -> Red
        9.0: Open Shrubland      -> Light Orange
        10.0: Grassland          -> Light Green
        11.0: Cropland           -> Golden Yellow
        12.0: Bare Ground        -> Light Brown
        13.0: Urban and Built-up -> Bright Cyan

    This colormap can be used for visualizing land cover data based on the UMD classification scheme.
    """

    colordict = {
        "Water": "#444f89",  # Water (Deep Blue)
        "Evergreen Needleleaf Forest": "#016400",  # Evergreen Needleleaf Forest (Dark Green)
        "Evergreen Broadleaf Forest": "#018200",  # Evergreen Broadleaf Forest (Light Green)
        "Deciduous Needleleaf Forest": "#97bf47",  # Deciduous Needleleaf Forest (Brownish)
        "Deciduous Broadleaf Forest": "#02dc00",  # Deciduous Broadleaf Forest (Orange)
        "Mixed Forest": "#00ff00",  # Mixed Forest (Purple)
        "Woodland": "#92ae2f",  # Woodland (Yellow-Green)
        "Wooded Grassland": "#dcce00",  # Wooded Grassland (Gray)
        "Closed Shrubland": "#ffad00",  # Closed Shrubland (Red)
        "Open Shrubland": "#fffbc3",  # Open Shrubland (Light Orange)
        "Grassland": "#8c4809",  # Grassland (Light Green)
        "Cropland": "#f7a5ff",  # Cropland (Golden Yellow)
        "Bare Ground": "#ffc7ae",  # Bare Ground (Light Brown)
        "Urban and Built-up": "#00ffff",  # Urban and Built-up (Bright Cyan)
    }
    ticks = list(colordict.keys())
    ticks_position = list(range(len(ticks)))  # 0~13
    colorlist = list(colordict.values())
    colorlevel = np.arange(-0.5, len(ticks) + 0.5)  # -0.5~13.5

    cmap = mcolors.ListedColormap(colorlist)
    norm = mcolors.BoundaryNorm(colorlevel, cmap.N)

    return cmap, norm, ticks, ticks_position, colorlist, colorlevel


def format_lon(lon, pos):
    """
    Format longitude value as a string with direction (East/West).

    This function takes a longitude value and returns it as a string with one decimal place,
    followed by the corresponding directional suffix ('°E' for positive values and '°W' for negative values).

    Parameters
    ----------
    lon : float
        The longitude value to be formatted.
    pos : float
        The position of the tick (not used in this function, but required by the formatter).

    Returns
    -------
    str
        The formatted longitude string, e.g., "45.0°E" or "90.0°W".
    """
    return f"{abs(lon):.1f}°W" if lon < 0 else f"{abs(lon):.1f}°E"


def format_lat(lat, pos):
    """
    Format latitude value as a string with direction (North/South).

    This function takes a latitude value and returns it as a string with one decimal place,
    followed by the corresponding directional suffix ('°N' for positive values and '°S' for negative values).

    Parameters
    ----------
    lat : float
        The latitude value to be formatted.
    pos : float
        The position of the tick (not used in this function, but required by the formatter).

    Returns
    -------
    str
        The formatted latitude string, e.g., "45.0°N" or "90.0°S".
    """
    return f"{abs(lat):.1f}°S" if lat < 0 else f"{abs(lat):.1f}°N"


def rotate_yticks(ax, yticks_rotation=0):
    """
    Rotate the y-axis tick labels on a given axis.

    This function modifies the y-axis tick labels of the given axis object by setting their
    rotation angle and vertical alignment.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object whose y-tick labels will be rotated.
    yticks_rotation : float, optional
        The angle of rotation for the y-tick labels in degrees (default is 0).
    """
    for tick in ax.get_yticklabels():
        tick.set_rotation(yticks_rotation)
        tick.set_va("center")


def set_xyticks(ax, x_locator_interval, y_locator_interval, yticks_rotation=0):
    """
    Set the x and y axis ticks with specified intervals and rotation.

    This function configures the major tick locations and formatting for both the x and y axes.
    It also allows for rotation of the y-axis tick labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object where the ticks and labels will be applied.
    x_locator_interval : float
        The interval between x-axis ticks.
    y_locator_interval : float
        The interval between y-axis ticks.
    yticks_rotation : float, optional
        The angle of rotation for the y-tick labels in degrees (default is 0).
    """
    # set xy ticks
    # for tick in ax.get_yticklabels():
    #     tick.set_rotation(yticks_rotation)
    #     tick.set_va('center')
    rotate_yticks(ax, yticks_rotation)

    ax.xaxis.set_major_locator(MultipleLocator(x_locator_interval))
    ax.yaxis.set_major_locator(MultipleLocator(y_locator_interval))

    # format_lon = lambda lon, pos: f"{abs(lon):.1f}°W" if lon < 0 else f"{abs(lon):.1f}°E"
    # format_lat = lambda lat, pos: f"{abs(lat):.1f}°S" if lat < 0 else f"{abs(lat):.1f}°N"
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))


def set_boundary(ax, boundary_x_y):
    """
    Set the axis limits based on the given boundary values.

    This function adjusts the x and y axis limits of the provided axis object (`ax`)
    using the boundary values specified in `boundary_x_y`. The first two elements in
    `boundary_x_y` correspond to the x-axis limits, and the last two elements correspond
    to the y-axis limits.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object for which the limits will be set.
    boundary_x_y : list or tuple of 4 floats
        The boundary values in the format [x_min, y_min, x_max, y_max], which define
        the x and y axis limits.
    """
    ax.set_xlim(boundary_x_y[0], boundary_x_y[2])
    ax.set_ylim(boundary_x_y[1], boundary_x_y[3])


def zoom_center(ax, x_center, y_center, zoom_factor=2):
    """
    Zoom into a specific region around a central point on the axis.

    This function zooms into the region centered at (`x_center`, `y_center`) by a
    factor specified by `zoom_factor`. The zooming is applied symmetrically, maintaining
    the aspect ratio of the plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object to apply the zoom on.
    x_center : float
        The x-coordinate of the center point for zooming.
    y_center : float
        The y-coordinate of the center point for zooming.
    zoom_factor : float, optional
        The factor by which to zoom the plot. A value greater than 1 will zoom in
        (default is 2).
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x_range = (xlim[1] - xlim[0]) / zoom_factor
    y_range = (ylim[1] - ylim[0]) / zoom_factor

    ax.set_xlim(x_center - x_range / 2, x_center + x_range / 2)
    ax.set_ylim(y_center - y_range / 2, y_center + y_range / 2)


def set_ax_box_aspect(ax, hw_factor=1):
    """
    Set the aspect ratio of the axis box.

    This function adjusts the aspect ratio of the axis box. The aspect ratio is set
    according to the value of `hw_factor`, where a value of 1 maintains a square aspect
    ratio. Values greater than 1 will stretch the plot horizontally, and values less
    than 1 will stretch it vertically.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object for which the box aspect ratio will be set.
    hw_factor : float, optional
        The ratio of the width to the height of the axis box. A value of 1 maintains
        a square aspect ratio (default is 1).
    """
    ax.set_box_aspect(hw_factor)

def plot_check_search(
    searched_grid_lat, searched_grid_lon,
    target_grid_lat, target_grid_lon,
    searched_grid_res=None,
    target_grid_res=None,
):
    """ 
    Usage:
    i = 205
    searched_grid_lon = [src_lon[l] for l in searched_grids_index[i][0]]
    searched_grid_lat = [src_lat[l] for l in searched_grids_index[i][1]]
    searched_grid_res = src_dlon
    target_grid_lat = [dst_lat[i]]
    target_grid_lon = [dst_lon[i]]
    target_grid_res = dst_dlon
    plot_check_search(
        searched_grid_lat, searched_grid_lon,
        target_grid_lat, target_grid_lon,
        searched_grid_res,
        target_grid_res,
    )
    """
    cgdf = CreateGDF()
    
    # create searched_grids_gdf
    if searched_grid_res is not None:
        searched_grids_gdf = cgdf.createGDF_rectangle_central_coord(
            searched_grid_lon, searched_grid_lat, searched_grid_res
        )
    else:
        searched_grids_gdf = cgdf.createGDF_points(
            lon=searched_grid_lon,
            lat=searched_grid_lat,
        )
    
    # create target_grid
    if target_grid_res is not None:
        target_grids_gdf = cgdf.createGDF_rectangle_central_coord(
            target_grid_lon, target_grid_lat, target_grid_res
        )
    else:
        target_grids_gdf = cgdf.createGDF_points(
            lon=target_grid_lon,
            lat=target_grid_lat,
        )
    
    # plot
    fig, ax = plt.subplots()
    target_grids_gdf.boundary.plot(ax=ax, edgecolor="r", linewidth=2)  # target
    searched_grids_gdf.plot(
        ax=ax, edgecolor="k", linewidth=0.2, facecolor="b", alpha=0.5
    )
    
    ax.set_title("check search")
    
    plt.show(block=True)