# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
"""
Two basic idea:

(1) index search, and x[index]
(2) mask array [0, 1], and use matrix multiplication

usage:
searched_grids_index = search_grids.search_grids_nearest(dst_lat=grids_lat, dst_lon=grids_lon,
                                                        src_lat=soil_lat_clip, src_lon=soil_lon_clip,
                                                        search_num=1)
                                                        
for i in tqdm(grid_shp.index, colour="green", desc=f"loop for each grid to extract soil{l} data", leave=False):
        # lon/lat
        searched_grid_index = searched_grids_index[i]
        sand_searched_grid_data = [sand_clip[l, searched_grid_index[0][j], searched_grid_index[1][j]]
                                    for j in range(len(searched_grid_index[0]))]  # index: (lat, lon), namely (row, col)

"""

import numpy as np
from tqdm import *

#* note: Slicing trap in netcdf4 and xarray, transfer it as np.ndarray first

# TODO parallel
def parallel_function():
    pass


def Uniform_precision(coord, percision):
    """ uniform percision coord based on percision 
    Args:
        coord: 1D array, the lat or lon for grids
        percision: int, min percision to be uniform
    """
    coord = np.array(coord)
    return coord.round(percision)


def search_grids_equal(dst_lat, dst_lon, src_lat, src_lon, lat_radius=None, lon_radius=None, **tqdm_kwargs):
    """ search grids with same coord (src_lat == dst_lat and src_lon == src_lon)

    Args:
        dst_lat/dst_lon: 1D array, the lat/lon of the destination grids
        src_lat/src_lon: 1D array, the lat/lon of the sources grids
        *note: src_lat/lon must be a sorted set, such as sorted(list(set(coord.lon))), dont contain same value
        tqdm_kwargs: you can set leave=False for nest tqdm (Only set leave to True for the outermost tqdm)
    return:
        searched_grids_index: list of tuple, len = len(destination), lat/lon index of dst grids in src grids,
            each tuple: (lat_index, lon_index), len(lat_index) = len(lon_index) = number of searched grids
            i.e. (array([0, 1, 1, 2], dtype=int64), array([0, 0, 1, 0], dtype=int64)) = (lat_index, lon_index)
            lat_index = row = array([0, 1, 1, 2], dtype=int64)
            lon_index = col = array([0, 0, 1, 0], dtype=int64)
            *return index = (row, col)
            *searched_grids_index is one-to-one, len(lat_index) == len(lon_index), there are 4 points above rather than 16
            *it should be careful when using netCDF.Dataset.variable[:, lat_index, lon_index], it is recommended that transfer it into np.ndarray first, such as array = netCDF.Dataset.variable[:, :, :],
            *array[:, lat_index, lon_index]
    """
    src_lon = np.array(src_lon)
    src_lat = np.array(src_lat)
    
    searched_grids_index = []
    for j in tqdm(range(len(dst_lat)), desc="search for dst grids", colour="green", **tqdm_kwargs):
        searched_grids_index_ = (np.where(src_lat == dst_lat[j])[0], np.where(src_lon == dst_lon[j])[0])
        searched_grids_index.append(searched_grids_index_)
    return searched_grids_index


def search_grids_radius(dst_lat, dst_lon, src_lat, src_lon, lat_radius, lon_radius=None, **tqdm_kwargs):
    """ search near grids in circle, based on radius
    input:
        lat_radius: search_radius, float, search radius, define the search domain (a circle centered at the destination grid),
             default it can be destination grid resolution / 2 = res / 2
        tqdm_kwargs: you can set leave=False for nest tqdm (Only set leave to True for the outermost tqdm)
    *note: index in netCDF4.Dataset.variable will get all grids
    *such as: variable[:, [1, 2, 3], [1, 2, 3]] -> ret.shape will be (:, 3, 3), namely take 9 points rather than 3 points, transfer it as np.ndarray first
    """
    search_radius = lat_radius
    src_lon = np.array(src_lon)
    src_lat = np.array(src_lat)
    
    src_lon_mesh, src_lat_mesh = np.meshgrid(src_lon, src_lat)  # 2D array
    searched_grids_index = []

    for j in tqdm(range(len(dst_lat)), desc="search for dst grids", colour="green", **tqdm_kwargs):
        # cal distance
        dx = abs(src_lon_mesh - dst_lon[j])
        dy = abs(src_lat_mesh - dst_lat[j])
        d = (dx ** 2 + dy ** 2) ** 0.5

        # find grids in ncfile which distance <= search_radius
        searched_grids_index_ = np.where(d <= search_radius)
        searched_grids_index.append(searched_grids_index_)

    return searched_grids_index


def search_grids_radius_rectangle(dst_lat, dst_lon, src_lat, src_lon, lat_radius, lon_radius, **tqdm_kwargs):
    """ search near grids in rectangle, based on lat_radius and lon_radius
    note: dst grids cover src grids, dst_grids is large and src_grids is small
    input:
        search_radius: float, search radius, define the search domain (lat+-lat_radius, lon+-lon_radius),
             default it can be destination grid resolution / 2 = res / 2
        tqdm_kwargs: you can set leave=False for nest tqdm (Only set leave to True for the outermost tqdm)
    """
    src_lon = np.array(src_lon)
    src_lat = np.array(src_lat)
    
    src_lon_mesh, src_lat_mesh = np.meshgrid(src_lon, src_lat)  # 2D array
    searched_grids_index = []

    for j in tqdm(range(len(dst_lat)), desc="search for dst grids", colour="green", **tqdm_kwargs):
        # cal distance
        dx = abs(src_lon_mesh - dst_lon[j])
        dy = abs(src_lat_mesh - dst_lat[j])
        
        # find grids in ncfile which distance <= search_radius
        # searched_grids_index_dx_bool_re = dx >= lon_radius
        # searched_grids_index_dy_bool_re = dy >= lat_radius
        # searched_grids_index_dx_dy_bool_re = searched_grids_index_dx_bool_re + searched_grids_index_dy_bool_re
        
        # searched_grids_index_ = np.where(searched_grids_index_dx_dy_bool_re == 0)
        
        # old version
        searched_grids_index_ = np.where((dx <= lon_radius) & (dy <= lat_radius))
        
        searched_grids_index.append(searched_grids_index_)

    return searched_grids_index


def search_grids_radius_rectangle_reverse(dst_lat, dst_lon, src_lat, src_lon, lat_radius, lon_radius, **tqdm_kwargs):
    """ search near grids in rectangle, based on lat_radius and lon_radius
    reverse: src grids cover dst grids, dst_grids is small and src_grids is large
    
    input:
        search_radius: float, search radius, define the search domain (lat+-lat_radius, lon+-lon_radius),
             default it can be src grid resolution / 2 = res / 2
        tqdm_kwargs: you can set leave=False for nest tqdm (Only set leave to True for the outermost tqdm)
             
    """
    src_lon = np.array(src_lon)
    src_lat = np.array(src_lat)
    
    # dst_lon_mesh, dst_lat_mesh = np.meshgrid(dst_lon, dst_lat)  # 2D array
    src_lon_mesh, src_lat_mesh = np.meshgrid(src_lon, src_lat)  # 2D array
    searched_grids_index = []

    for j in tqdm(range(len(dst_lat)), desc="search for dst grids", colour="green", **tqdm_kwargs):
        # cal distance
        dx = abs(src_lon_mesh - dst_lon[j])
        dy = abs(src_lat_mesh - dst_lat[j])
        
        # old version
        searched_grids_index_ = np.where((dx <= lon_radius) & (dy <= lat_radius))
        
        searched_grids_index.append(searched_grids_index_)

    return searched_grids_index
    
    
def search_grids_nearest(dst_lat, dst_lon, src_lat, src_lon, lat_radius=None, lon_radius=None,
                         search_num=4, move_src_lat=None, move_src_lon=None, **tqdm_kwargs):
    """ search nearest grids, based on search_num
    input:
        search_num: int, search number, define the number of nearest grids to be searched,
            default it can be 4
        move_src_lat/lon: float or None, if used, the src file lat/lon will be move the value of move_nc
            this option is used when the dst_grid line with two nc grids, it can not find out nearest four grids
            generally, it can be set as src_res / 5 (small value)

            src_grid               src_grid            src_grid               src_grid

            src_grid   dst_grid    src_grid     ->                dst_grid

            src_grid               src_grid            src_grid               src_grid
            make (lon_flatten[min_index], lat_flatten[min_index]) like this
                [102.335 102.335 102.385 102.385] [32.385 32.335 32.335 32.385]
            rather
                [102.425 102.375 102.375 102.325] [32.375 32.375 32.425 32.375]
        tqdm_kwargs: you can set leave=False for nest tqdm (Only set leave to True for the outermost tqdm)
    """
    src_lon = np.array(src_lon)
    src_lat = np.array(src_lat)
    
    src_lon_mesh_index, src_lat_mesh_index = np.meshgrid(np.arange(len(src_lon)), np.arange(len(src_lat)))
    src_lon_flatten_index = src_lon_mesh_index.flatten()  # 1D array
    src_lat_flatten_index = src_lat_mesh_index.flatten()

    src_lon_mesh, src_lat_mesh = np.meshgrid(src_lon, src_lat)
    src_lon_flatten = src_lon_mesh.flatten()  # 1D array
    src_lat_flatten = src_lat_mesh.flatten()

    # move
    if move_src_lon:
        src_lon_flatten += move_src_lon
    if move_src_lat:
        src_lat_flatten += move_src_lat

    searched_grids_index = []

    for j in tqdm(range(len(dst_lat)), desc="search for dst grids", colour="green", **tqdm_kwargs):
        # cal distance
        dx = abs(src_lon_flatten - dst_lon[j])
        dy = abs(src_lat_flatten - dst_lat[j])
        d = (dx ** 2 + dy ** 2) ** 0.5

        # find grids in src which nearest with dst at search_num th
        min_index = np.argpartition(d, search_num)[:search_num]
        searched_grids_index_ = (
            src_lat_flatten_index[min_index], src_lon_flatten_index[min_index])
        searched_grids_index.append(searched_grids_index_)

    return searched_grids_index
    
    
    
def print_ret(searched_grids_index, src_lat, src_lon):
    # print result
    searched_grids_index = searched_grids_index[0]
    
    print(f"grids: {len(searched_grids_index[0])}")
    for i in range(len(searched_grids_index[0])):
        print(searched_grids_index[0][i], searched_grids_index[1][i])
        print(src_lat[searched_grids_index[0][i]], src_lon[searched_grids_index[1][i]])

    
    