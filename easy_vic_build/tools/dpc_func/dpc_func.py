# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from matplotlib import pyplot as plt
import numpy as np
from ..geo_func.create_gdf import CreateGDF
from ..geo_func.search_grids import *
from ..params_func.params_set import *

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
    
    # weight mean
    weights = [CONUS_layers_depths[i] for i in range(depth_layer_start, depth_layer_end)]
    weights /= sum(weights)
    
    grid_array_sand = np.average(grid_array_sand, axis=0, weights=weights)
    grid_array_silt = np.average(grid_array_silt, axis=0, weights=weights)
    grid_array_clay = np.average(grid_array_clay, axis=0, weights=weights)
    
    # keep sum = 100
    grid_array_sum = grid_array_sand + grid_array_silt + grid_array_clay
    adjustment = 100 - grid_array_sum
    
    grid_array_sand += (grid_array_sand / grid_array_sum) * adjustment
    grid_array_silt += (grid_array_silt / grid_array_sum) * adjustment
    grid_array_clay += (grid_array_clay / grid_array_sum) * adjustment
    
    return grid_array_sand, grid_array_silt, grid_array_clay


def cal_bd_grid_array(grid_shp_level0, depth_layer_start, depth_layer_end,
                      stand_grids_lat, stand_grids_lon, rows_index, cols_index):
    # vertical aggregation for bulk_density
    grid_array_bd = [createEmptyArray_and_assignValue_from_gridshp(stand_grids_lat, stand_grids_lon, grid_shp_level0.loc[:, f"soil_l{i+1}_bulk_density_nearest_Value"], rows_index, cols_index, dtype=float, missing_value=np.nan) for i in range(depth_layer_start, depth_layer_end)]
    
    # weight mean
    weights = [CONUS_layers_depths[i] for i in range(depth_layer_start, depth_layer_end)]
    weights /= sum(weights)
    
    grid_array_bd = np.average(grid_array_bd, axis=0, weights=weights)
    
    grid_array_bd *= 10 # / 100 (multiple) * 1000 (g/cm3 -> kg/m3)
    
    return grid_array_bd