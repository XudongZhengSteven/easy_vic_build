# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from tqdm import *
import pickle
from netCDF4 import Dataset
import shutil
from .geo_func.search_grids import *
from .geo_func.create_gdf import CreateGDF
from .params_func.GlobalParamParser import GlobalParamParser
from configparser import ConfigParser


## ------------------------ general utilities ------------------------
def check_and_mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(os.path.abspath(dir))


def remove_and_mkdir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(os.path.abspath(dir))


def setHomePath(root="E:"):
    home = f"{root}/data/hydrometeorology/CAMELS"
    return root, home


def exportToCsv(basin_shp, fpath_dir):
    columns = basin_shp.columns.to_list()
    streamflow = basin_shp.loc[:, columns.pop(11)]
    prep = basin_shp.loc[:, columns.pop(11)]
    gleam_e_daily = basin_shp.loc[:, columns.pop(-1)]

    columns.remove("geometry")
    # columns.remove("intersects_grids")

    csv_df = basin_shp.loc[:, columns]

    # save
    csv_df.to_csv(os.path.join(fpath_dir, "basin_shp.csv"))

    # save streamflow
    if not os.path.exists(os.path.join(fpath_dir, "streamflow")):
        os.mkdir(os.path.join(fpath_dir, "streamflow"))
    for i in tqdm(streamflow.index, desc="loop for basins to save streamflow", colour="green"):
        streamflow[i].to_csv(os.path.join(fpath_dir, "streamflow", f"{i}.csv"))

    # save prep
    if not os.path.exists(os.path.join(fpath_dir, "prep")):
        os.mkdir(os.path.join(fpath_dir, "prep"))
    for i in tqdm(prep.index, desc="loop for basins to save prep", colour="green"):
        prep[i].to_csv(os.path.join(fpath_dir, "prep", f"{i}.csv"))

    # save gleam_e_daily
    if not os.path.exists(os.path.join(fpath_dir, "gleam_e_daily")):
        os.mkdir(os.path.join(fpath_dir, "gleam_e_daily"))
    for i in tqdm(gleam_e_daily.index, desc="loop for basins to save gleam_e_daily", colour="green"):
        gleam_e_daily[i].to_csv(os.path.join(fpath_dir, "gleam_e_daily", f"{i}.csv"))
        

def checkGaugeBasin(basinShp, usgs_streamflow, BasinAttribute, forcingDaymetGaugeAttributes):
    id_basin_shp = set(basinShp.hru_id.values)
    id_usgs_streamflow = set([usgs_streamflow[i].iloc[0, 0] for i in range(len(usgs_streamflow))])
    id_BasinAttribute = set(BasinAttribute["camels_clim"].gauge_id.values)
    id_forcingDaymet = set([forcingDaymetGaugeAttributes[i]["gauge_id"]
                            for i in range(len(forcingDaymetGaugeAttributes))])
    print(f"id_HCDN_shp: {len(id_basin_shp)}, id_usgs_streamflow: {len(id_usgs_streamflow)}, id_BasinAttribute: {len(id_BasinAttribute)}, id_forcingDatmet: {len(id_forcingDaymet)}")
    print(f"id_usgs_streamflow - id_HCDN_shp: {id_usgs_streamflow - id_basin_shp}")
    print(f"id_BasinAttribute - id_HCDN_shp: {id_BasinAttribute - id_basin_shp}")
    print(f"id_forcingDatmet - id_HCDN_shp: {id_forcingDaymet - id_basin_shp}")
    # result
    # id_usgs_streamflow - id_HCDN_shp: {9535100, 6775500, 6846500}
    # id_forcingDatmet - id_HCDN_shp: {6846500, 6775500, 3448942, 1150900, 2081113, 9535100}
    

## ------------------------ grid_shp, basin_shp utilities ------------------------
def createBoundaryShp(grid_shp):
    # boundary: point center
    cgdf_point = CreateGDF()
    boundary_x_min = min(grid_shp["point_geometry"].x)
    boundary_x_max = max(grid_shp["point_geometry"].x)
    boundary_y_min = min(grid_shp["point_geometry"].y)
    boundary_y_max = max(grid_shp["point_geometry"].y)
    boundary_point_center_shp = cgdf_point.createGDF_polygons(lon=[[boundary_x_min, boundary_x_max, boundary_x_max, boundary_x_min]],
                                           lat=[[boundary_y_max, boundary_y_max, boundary_y_min, boundary_y_min]],
                                           crs=grid_shp.crs)
    boundary_point_center_x_y = [boundary_x_min, boundary_x_max, boundary_y_min, boundary_y_max]
    
    # boundary: grids edge
    boundary_x_min = min(grid_shp["geometry"].get_coordinates().x)
    boundary_x_max = max(grid_shp["geometry"].get_coordinates().x)
    boundary_y_min = min(grid_shp["geometry"].get_coordinates().y)
    boundary_y_max = max(grid_shp["geometry"].get_coordinates().y)
    
    boundary_grids_edge_shp = cgdf_point.createGDF_polygons(lon=[[boundary_x_min, boundary_x_max, boundary_x_max, boundary_x_min]],
                                           lat=[[boundary_y_max, boundary_y_max, boundary_y_min, boundary_y_min]],
                                           crs=grid_shp.crs)
    boundary_grids_edge_x_y = [boundary_x_min, boundary_x_max, boundary_y_min, boundary_y_max]
    
    return boundary_point_center_shp, boundary_point_center_x_y, boundary_grids_edge_shp, boundary_grids_edge_x_y


def createStand_grids_lat_lon_from_gridshp(grid_shp, grid_res=None, reverse_lat=True):
    #  grid_res is None: grid_shp is a Complete rectangular grids set, else is may be a uncomplete grids set
    # create sorted stand grids
    if grid_res is None:
        stand_grids_lon = list(set(grid_shp["point_geometry"].x.to_list()))
        stand_grids_lat = list(set(grid_shp["point_geometry"].y.to_list()))
        
        stand_grids_lon = np.array(sorted(stand_grids_lon, reverse=False))  # small -> large, left is zero
        stand_grids_lat = np.array(sorted(stand_grids_lat, reverse=reverse_lat))  # if True, large -> small, top is zero, it is useful for plot directly
    
    else:
        min_lon = min(grid_shp["point_geometry"].x.to_list())
        max_lon = max(grid_shp["point_geometry"].x.to_list())
        num_lon = (max_lon - min_lon) / grid_res + 1
        stand_grids_lon = np.linspace(start=min_lon, stop=max_lon, num=num_lon)
        
        min_lat = min(grid_shp["point_geometry"].y.to_list())
        max_lat = max(grid_shp["point_geometry"].y.to_list())
        num_lat = (max_lat - min_lat) / grid_res + 1
        stand_grids_lat = np.linspace(start=max_lat, stop=min_lat, num=num_lat) if reverse_lat else np.linspace(start=min_lat, stop=max_lat, num=num_lat)
    
    return stand_grids_lat, stand_grids_lon


def createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan):
    # empty array, shape is [lat(large -> small), lon(small -> large)]
    grid_array = np.full((len(stand_grids_lat), len(stand_grids_lon)), fill_value=missing_value, dtype=dtype)
    
    return grid_array


def gridshp_index_to_grid_array_index(grid_shp, stand_grids_lat, stand_grids_lon):
    grid_shp_point_lon = [grid_shp.loc[i, "point_geometry"].x for i in grid_shp.index]
    grid_shp_point_lat = [grid_shp.loc[i, "point_geometry"].y for i in grid_shp.index]
    
    searched_grids_index = search_grids_equal(dst_lat=grid_shp_point_lat, dst_lon=grid_shp_point_lon,
                                              src_lat=stand_grids_lat, src_lon=stand_grids_lon, leave=False)
    
    rows_index, cols_index = searched_grids_index_to_rows_cols_index(searched_grids_index)
    return rows_index, cols_index
    
    
def assignValue_for_grid_array(empty_grid_array, values_list, rows_index, cols_index):
    # values_list can be grid_shp.loc[:, value_column]
    grid_array = empty_grid_array
    grid_array[rows_index, cols_index] = values_list
    
    return grid_array


def createEmptyArray_and_assignValue_from_gridshp(stand_grids_lat, stand_grids_lon, values_list, rows_index, cols_index, dtype=float, missing_value=np.nan):
    # empty array, shape is [lat(large -> small), lon(small -> large)]
    grid_array = np.full((len(stand_grids_lat), len(stand_grids_lon)), fill_value=missing_value, dtype=dtype)
    
    # assign values
    grid_array[rows_index, cols_index] = values_list
    return grid_array
    
    
def createArray_from_gridshp(grid_shp, value_column, grid_res=None, dtype=float, missing_value=np.nan, plot=False, reverse_lat=True):
    # create stand grids lat, lon
    stand_grids_lat, stand_grids_lon = createStand_grids_lat_lon_from_gridshp(grid_shp, grid_res, reverse_lat)

    # create empty array
    grid_array = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype, missing_value)
    
    # grid_shp.index to grid_array index
    rows_index, cols_index = gridshp_index_to_grid_array_index(grid_shp, stand_grids_lat, stand_grids_lon)
    
    # assign values
    grid_array = assignValue_for_grid_array(grid_array, grid_shp, value_column, rows_index, cols_index)
    
    # plot
    if plot:
        plt.imshow(grid_array)
    
    return grid_array, stand_grids_lon, stand_grids_lat


def grids_array_coord_map(grid_shp, reverse_lat=True):
    # lon/lat grid map into index to construct array
    lon_list = sorted(list(set(grid_shp["point_geometry"].x.values)))
    lat_list = sorted(list(set(grid_shp["point_geometry"].y.values)), reverse=reverse_lat)  # if True, large -> small, top is zero, it is useful for plot directly
    
    lon_map_index = dict(zip(lon_list, list(range(len(lon_list)))))
    lat_map_index = dict(zip(lat_list, list(range(len(lat_list)))))
    
    return lon_list, lat_list, lon_map_index, lat_map_index


def cal_ssc_percentile_grid_array(grid_shp_level0, depth_layer_start, depth_layer_end,
                                  stand_grids_lat, stand_grids_lon, rows_index, cols_index):
    # vertical aggregation for sand, silt, clay percentile
    grid_array_sand = [createEmptyArray_and_assignValue_from_gridshp(stand_grids_lat, stand_grids_lon, grid_shp_level0.loc[:, f"soil_l{i+1}_sand_nearest_Value"], rows_index, cols_index, dtype=float, missing_value=np.nan) for i in range(depth_layer_start, depth_layer_end)]
    grid_array_silt = [createEmptyArray_and_assignValue_from_gridshp(stand_grids_lat, stand_grids_lon, grid_shp_level0.loc[:, f"soil_l{i+1}_silt_nearest_Value"], rows_index, cols_index, dtype=float, missing_value=np.nan) for i in range(depth_layer_start, depth_layer_end)]
    grid_array_clay = [createEmptyArray_and_assignValue_from_gridshp(stand_grids_lat, stand_grids_lon, grid_shp_level0.loc[:, f"soil_l{i+1}_clay_nearest_Value"], rows_index, cols_index, dtype=float, missing_value=np.nan) for i in range(depth_layer_start, depth_layer_end)]
    
    grid_array_sand = np.mean(grid_array_sand, axis=0)
    grid_array_silt = np.mean(grid_array_silt, axis=0)
    grid_array_clay = np.mean(grid_array_clay, axis=0)
    
    # keep sum = 100
    grid_array_sum = grid_array_sand + grid_array_silt + grid_array_clay
    adjustment = 100 - grid_array_sum
    
    grid_array_sand += (grid_array_sand / grid_array_sum) * adjustment
    grid_array_silt += (grid_array_silt / grid_array_sum) * adjustment
    grid_array_clay += (grid_array_clay / grid_array_sum) * adjustment
    
    return grid_array_sand, grid_array_silt, grid_array_clay


def cal_bd_grid_array(grid_shp_level0, depth_layer_start, depth_layer_end,
                      stand_grids_lat, stand_grids_lon, rows_index, cols_index):
    #  vertical aggregation for bulk_density
    grid_array_bd = [createEmptyArray_and_assignValue_from_gridshp(stand_grids_lat, stand_grids_lon, grid_shp_level0.loc[:, f"soil_l{i+1}_bulk_density_nearest_Value"], rows_index, cols_index, dtype=float, missing_value=np.nan) for i in range(depth_layer_start, depth_layer_end)]
    grid_array_bd = np.mean(grid_array_bd, axis=0)
    grid_array_bd /= 100 # 100 * kg/m3 -> kg/m3
    
    return grid_array_bd

## ------------------------ read function ------------------------
def readHCDNGrids(home="E:\\data\\hydrometeorology\\CAMELS"):
    grid_shp_label_path = os.path.join(home, "map", "grids_0_25_label.shp")
    grid_shp_label = gpd.read_file(grid_shp_label_path)
    print(grid_shp_label)

    grid_shp_path = os.path.join(home, "map", "grids_0_25.shp")
    grid_shp = gpd.read_file(grid_shp_path)
    print(grid_shp)

    # combine grid_shp_lable into grid_shp
    grid_shp["point_geometry"] = grid_shp_label.geometry

    return grid_shp


def readHCDNBasins(home="E:\\data\\hydrometeorology\\CAMELS"):
    # read data: HCDN
    HCDN_shp_path = os.path.join(home, "basin_set_full_res", "HCDN_nhru_final_671.shp")
    HCDN_shp = gpd.read_file(HCDN_shp_path)
    HCDN_shp["AREA_km2"] = HCDN_shp.AREA / 1000000  # m2 -> km2
    print(HCDN_shp)
    return HCDN_shp


def readdpc(evb_dir):
    # ====================== read ======================
    # read
    with open(evb_dir.dpc_VIC_level0_path, "rb") as f:
        dpc_VIC_level0 = pickle.load(f)
    
    with open(evb_dir.dpc_VIC_level1_path, "rb") as f:
        dpc_VIC_level1 = pickle.load(f)
    
    with open(evb_dir.dpc_VIC_level2_path, "rb") as f:
        dpc_VIC_level2 = pickle.load(f)
    
    return dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2


def readDomain(evb_dir):
    # ====================== read ======================
    # read
    domain_dataset = Dataset(evb_dir.domainFile_path, "r", format="NETCDF4")
    
    return domain_dataset


def readParam(evb_dir, mode="r"):
    # ====================== read ======================
    # read
    params_dataset_level0 = Dataset(evb_dir.params_dataset_level0_path, mode, format="NETCDF4")
    params_dataset_level1 = Dataset(evb_dir.params_dataset_level1_path, mode, format="NETCDF4")
    
    return params_dataset_level0, params_dataset_level1


def readRVICParam(evb_dir):
    # ====================== read ======================
    # read
    flow_direction_dataset = Dataset(evb_dir.flow_direction_file_path, "r", "NETCDF4")
    pourpoint_file = pd.read_csv(evb_dir.pourpoint_file_path)
    uhbox_file = pd.read_csv(evb_dir.uhbox_file_path)
    cfg_file = ConfigParser()
    cfg_file.read(evb_dir.cfg_file_path)
    
    return flow_direction_dataset, pourpoint_file, uhbox_file, cfg_file


def readGlobalParam(evb_dir):
    # ====================== read ======================
    globalParam = GlobalParamParser()
    globalParam.load(evb_dir.globalParam_path)
    
    return globalParam
    
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

## ------------------------ Intersects grids with shp ------------------------
# def combineDataframeDropDuplicates(df1, df2, drop_based_columns=None):
#     combine_df = df1.append(df2)
#     combine_df.index = list(range(len(combine_df)))
#     remain_index = combine_df.loc[:, drop_based_columns].drop_duplicates().index
#     combine_df = combine_df.loc[remain_index, :]
#     return combine_df


# def IntersectsGridsWithHCDN(grid_shp, basinShp):
#     intersects_grids_list = []
#     intersects_grids = pd.DataFrame()
#     for i in basinShp.index:
#         basin = basinShp.loc[i, "geometry"]
#         intersects_grids_ = grid_shp[grid_shp.intersects(basin)]
#         intersects_grids = pd.concat([intersects_grids, intersects_grids_])
#         intersects_grids_list.append(intersects_grids_)

#     intersects_grids["grids_index"] = intersects_grids.index
#     intersects_grids.index = list(range(len(intersects_grids)))
#     droped_index = intersects_grids["grids_index"].drop_duplicates().index
#     intersects_grids = intersects_grids.loc[droped_index, :]

#     basinShp["intersects_grids"] = intersects_grids_list
#     return basinShp, intersects_grids






