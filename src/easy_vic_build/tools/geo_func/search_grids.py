# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: search_grids

This module contains functions for searching and matching grids between different spatial 
resolutions. It includes methods for locating exact, nearest, or radius-based matches, as well 
as transforming grid indices for advanced data extraction. These functions are useful for geospatial 
applications such as climate models, remote sensing, and other grid-based datasets.

Functions:
----------
    - Uniform_precision: Adjusts the precision of coordinates to ensure consistent comparison 
      between destination and source grids.
    - search_grids_equal: Finds grids with exactly matching coordinates (latitude and longitude).
    - search_grids_radius: Searches for grids within a specified radius based on latitude and 
      longitude distance.
    - search_grids_radius_rectangle: Searches for grids within a rectangular region defined by 
      latitude and longitude radii.
    - search_grids_radius_rectangle_reverse: Searches for grids where the source grids cover 
      the destination grids, in a reverse scenario.
    - search_grids_nearest: Identifies the nearest grids to a given set of destination grids, 
      based on a specified number of nearest neighbors.
    - print_ret: Prints the results of grid searches, displaying indices and their corresponding 
      coordinates.
    - searched_grids_index_to_rows_cols_index: Converts grid indices into row and column indices 
      for use in array indexing.
    - searched_grids_index_to_bool_index: Converts grid indices into boolean arrays for advanced 
      indexing in datasets.

Dependencies:
-------------
    - numpy: Provides numerical operations for array manipulation, distance calculations, and 
      indexing operations.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com

Two basic idea:
-------

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
    """Round coordinates to a uniform precision.

    Parameters
    ----------
    coord : array_like
        1D array containing latitude or longitude values for grids.
    precision : int
        Minimum precision to which values should be rounded.

    Returns
    -------
    ndarray
        Array with values rounded to the specified precision.
    """
    coord = np.array(coord)
    return coord.round(percision)


def search_grids_equal(dst_lat, dst_lon, src_lat, src_lon, lat_radius=None, lon_radius=None, **tqdm_kwargs):
    """Search for grids with matching coordinates (src_lat == dst_lat and src_lon == dst_lon).

    Parameters
    ----------
    dst_lat : array_like
        1D array of latitude values for the destination grids.
    dst_lon : array_like
        1D array of longitude values for the destination grids.
    src_lat : array_like
        1D array of latitude values for the source grids. Must be a sorted set 
        (e.g., `sorted(list(set(coord.lat)))`), ensuring unique values.
    src_lon : array_like
        1D array of longitude values for the source grids. Must be a sorted set 
        (e.g., `sorted(list(set(coord.lon)))`), ensuring unique values.
    lat_radius : optional
        Not used in this function but reserved for potential extensions.
    lon_radius : optional
        Not used in this function but reserved for potential extensions.
    **tqdm_kwargs : dict, optional
        Additional keyword arguments passed to `tqdm`. For nested progress bars, 
        set `leave=False`, keeping `leave=True` only for the outermost `tqdm`.

    Returns
    -------
    list of tuple
        A list of tuples with length equal to `len(dst_lat)`, where each tuple 
        contains two arrays: `(lat_index, lon_index)`. These represent the indices 
        of the destination grid points found in the source grid.

        - `lat_index`: 1D array of latitude indices.
        - `lon_index`: 1D array of longitude indices.

        Example output:
        `(array([0, 1, 1, 2], dtype=int64), array([0, 0, 1, 0], dtype=int64))`
        corresponds to:
        ```
        lat_index = array([0, 1, 1, 2], dtype=int64)  # Row indices
        lon_index = array([0, 0, 1, 0], dtype=int64)  # Column indices
        ```

    Notes
    -----
    - The returned `(lat_index, lon_index)` pairs are one-to-one, meaning 
      `len(lat_index) == len(lon_index)`, ensuring accurate matching.
    - When using `netCDF4.Dataset.variable[:, lat_index, lon_index]`, it is recommended 
      to first convert the dataset variable into a `numpy.ndarray` to avoid potential 
      errors, e.g., `array = netCDF4.Dataset.variable[:, :, :]`, then use `array[:, lat_index, lon_index]`.
    """
    
    src_lon = np.array(src_lon)
    src_lat = np.array(src_lat)
    
    searched_grids_index = []
    for j in tqdm(range(len(dst_lat)), desc="search for dst grids", colour="green", **tqdm_kwargs):
        searched_grids_index_ = (np.where(src_lat == dst_lat[j])[0], np.where(src_lon == dst_lon[j])[0])
        searched_grids_index.append(searched_grids_index_)
    return searched_grids_index


def search_grids_radius(dst_lat, dst_lon, src_lat, src_lon, lat_radius, lon_radius=None, **tqdm_kwargs):
    """Search for nearby grids within a circular radius.

    Parameters
    ----------
    dst_lat : array_like
        1D array of latitude values for the destination grids.
    dst_lon : array_like
        1D array of longitude values for the destination grids.
    src_lat : array_like
        1D array of latitude values for the source grids.
    src_lon : array_like
        1D array of longitude values for the source grids.
    lat_radius : float
        Search radius defining the circular search domain centered at each destination grid point.
        By default, it can be set to half of the destination grid resolution (i.e., `res / 2`).
    lon_radius : optional
        Reserved for potential extensions but not used in this function.
    **tqdm_kwargs : dict, optional
        Additional keyword arguments passed to `tqdm`. For nested progress bars, 
        set `leave=False`, keeping `leave=True` only for the outermost `tqdm`.

    Returns
    -------
    list of tuple
        A list of tuples with length equal to `len(dst_lat)`, where each tuple 
        contains two arrays: `(lat_index, lon_index)`, representing the indices 
        of source grid points found within the search radius.

        - `lat_index`: 1D array of latitude indices.
        - `lon_index`: 1D array of longitude indices.

        Example output:
        ```
        (array([1, 2, 3], dtype=int64), array([1, 2, 3], dtype=int64))
        ```
        corresponds to selecting a 3x3 grid around the destination point.

    Notes
    -----
    - The returned `(lat_index, lon_index)` pairs identify all source grids 
      within the given `lat_radius`.
    - When using `netCDF4.Dataset.variable[:, lat_index, lon_index]`, note that 
      this operation retrieves all grid points within the index range, potentially 
      increasing dimensionality. To avoid unintended broadcasting, convert the dataset 
      variable into a `numpy.ndarray` first, e.g., 
      ```
      array = netCDF4.Dataset.variable[:, :, :]
      array[:, lat_index, lon_index]
      ```
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
    """Search for nearby grids within a rectangular domain.

    Parameters
    ----------
    dst_lat : array_like
        1D array of latitude values for the destination grids.
    dst_lon : array_like
        1D array of longitude values for the destination grids.
    src_lat : array_like
        1D array of latitude values for the source grids.
    src_lon : array_like
        1D array of longitude values for the source grids.
    lat_radius : float
        Search radius in the latitude direction, defining the search domain as 
        `lat ± lat_radius`.
    lon_radius : float
        Search radius in the longitude direction, defining the search domain as 
        `lon ± lon_radius`.
    **tqdm_kwargs : dict, optional
        Additional keyword arguments passed to `tqdm`. For nested progress bars, 
        set `leave=False`, keeping `leave=True` only for the outermost `tqdm`.

    Returns
    -------
    list of tuple
        A list of tuples with length equal to `len(dst_lat)`, where each tuple 
        contains two arrays: `(lat_index, lon_index)`, representing the indices 
        of source grid points found within the rectangular search region.

        - `lat_index`: 1D array of latitude indices.
        - `lon_index`: 1D array of longitude indices.

        Example output:
        ```
        (array([1, 2, 3], dtype=int64), array([1, 2, 3], dtype=int64))
        ```
        corresponds to selecting a rectangular region around the destination grid.

    Notes
    -----
    - The destination grids (`dst_lat`, `dst_lon`) typically cover a larger area 
      than the source grids (`src_lat`, `src_lon`).
    - The rectangular search domain is defined by `lat ± lat_radius` and `lon ± lon_radius`.
    - When using `netCDF4.Dataset.variable[:, lat_index, lon_index]`, be aware that 
      this operation retrieves all grid points within the index range, potentially 
      increasing dimensionality. To avoid unintended broadcasting, convert the dataset 
      variable into a `numpy.ndarray` first, e.g., 
      ```
      array = netCDF4.Dataset.variable[:, :, :]
      array[:, lat_index, lon_index]
      ```
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
    """Search for destination grids within a rectangular domain of source grids.

    This function finds destination grid points (`dst_lat`, `dst_lon`) that are covered by
    a larger set of source grid points (`src_lat`, `src_lon`). The search domain is defined 
    as a rectangle centered on each destination grid point, extending `±lat_radius` in latitude 
    and `±lon_radius` in longitude.

    Parameters
    ----------
    dst_lat : array_like
        1D array of latitude values for the destination grids.
    dst_lon : array_like
        1D array of longitude values for the destination grids.
    src_lat : array_like
        1D array of latitude values for the source grids.
    src_lon : array_like
        1D array of longitude values for the source grids.
    lat_radius : float
        Search radius in the latitude direction, defining the search domain as 
        `lat ± lat_radius`.
    lon_radius : float
        Search radius in the longitude direction, defining the search domain as 
        `lon ± lon_radius`.
    **tqdm_kwargs : dict, optional
        Additional keyword arguments passed to `tqdm`. For nested progress bars, 
        set `leave=False`, keeping `leave=True` only for the outermost `tqdm`.

    Returns
    -------
    list of tuple
        A list of tuples with length equal to `len(dst_lat)`, where each tuple 
        contains two arrays: `(lat_index, lon_index)`, representing the indices 
        of source grid points that cover the corresponding destination grid.

        - `lat_index`: 1D array of latitude indices.
        - `lon_index`: 1D array of longitude indices.

        Example output:
        ```
        (array([1, 2, 3], dtype=int64), array([1, 2, 3], dtype=int64))
        ```
        corresponds to selecting a rectangular region around the destination grid.

    Notes
    -----
    - This function assumes that the source grids are larger and cover the smaller destination grids.
    - The rectangular search domain is defined by `lat ± lat_radius` and `lon ± lon_radius`.
    - When using `netCDF4.Dataset.variable[:, lat_index, lon_index]`, be aware that 
      this operation retrieves all grid points within the index range, potentially 
      increasing dimensionality. To avoid unintended broadcasting, convert the dataset 
      variable into a `numpy.ndarray` first, e.g., 
      ```
      array = netCDF4.Dataset.variable[:, :, :]
      array[:, lat_index, lon_index]
      ```
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
    """Search for the nearest source grids based on the number of neighbors.

    This function identifies the `search_num` closest source grid points (`src_lat`, `src_lon`)
    for each destination grid point (`dst_lat`, `dst_lon`), optionally shifting the source
    grid positions to improve alignment.

    Parameters
    ----------
    dst_lat : array_like
        1D array of latitude values for the destination grids.
    dst_lon : array_like
        1D array of longitude values for the destination grids.
    src_lat : array_like
        1D array of latitude values for the source grids.
    src_lon : array_like
        1D array of longitude values for the source grids.
    lat_radius : float, optional
        Not used in the current implementation but reserved for potential constraints on latitude.
    lon_radius : float, optional
        Not used in the current implementation but reserved for potential constraints on longitude.
    search_num : int, default=4
        The number of nearest source grid points to be identified for each destination grid.
    move_src_lat : float or None, optional
        If specified, shifts the source latitude values by this amount. This is useful when the
        destination grid lies between two source grid points and standard searching does not
        identify the correct nearest neighbors. Generally, it can be set as src_res / 5 (small value).
        ```
        src_grid               src_grid            src_grid               src_grid

        src_grid   dst_grid    src_grid     ->                dst_grid

        src_grid               src_grid            src_grid               src_grid
        ```
        make (lon_flatten[min_index], lat_flatten[min_index]) like this
            [102.335 102.335 102.385 102.385] [32.385 32.335 32.335 32.385]
        rather
            [102.425 102.375 102.375 102.325] [32.375 32.375 32.425 32.375]
    move_src_lon : float or None, optional
        If specified, shifts the source longitude values by this amount, similar to `move_src_lat`.
    **tqdm_kwargs : dict, optional
        Additional keyword arguments passed to `tqdm`. For nested progress bars, 
        set `leave=False`, keeping `leave=True` only for the outermost `tqdm`.

    Returns
    -------
    list of tuple
        A list of tuples with length equal to `len(dst_lat)`, where each tuple 
        contains two arrays: `(lat_index, lon_index)`, representing the indices 
        of the nearest source grid points corresponding to each destination grid.

        Example output:
        ```
        (array([1, 2, 3, 4], dtype=int64), array([5, 6, 7, 8], dtype=int64))
        ```
        corresponds to selecting the four nearest source grids.

    Notes
    -----
    - The function computes Euclidean distances in the latitude-longitude space, which may
      not be accurate for large-scale datasets due to Earth's curvature.
    - To ensure the nearest four grid points form a structured set (e.g., a square around
      each destination grid), `move_src_lat` and `move_src_lon` can be adjusted.
    - When using `netCDF4.Dataset.variable[:, lat_index, lon_index]`, be aware that 
      this operation retrieves all grid points within the index range, potentially 
      increasing dimensionality. To avoid unintended broadcasting, convert the dataset 
      variable into a `numpy.ndarray` first.

    """
    src_lon = np.array(src_lon)
    src_lat = np.array(src_lat)
    
    # Create 2D grid indices for source grid
    src_lon_mesh_index, src_lat_mesh_index = np.meshgrid(np.arange(len(src_lon)), np.arange(len(src_lat)))
    src_lon_flatten_index = src_lon_mesh_index.flatten()  # 1D array
    src_lat_flatten_index = src_lat_mesh_index.flatten()

    # Create 2D coordinate grid from source grids
    src_lon_mesh, src_lat_mesh = np.meshgrid(src_lon, src_lat)
    src_lon_flatten = src_lon_mesh.flatten()  # 1D array
    src_lat_flatten = src_lat_mesh.flatten()

    # Apply optional shifts to source grid positions
    if move_src_lon:
        src_lon_flatten += move_src_lon
    if move_src_lat:
        src_lat_flatten += move_src_lat

    searched_grids_index = []

    for j in tqdm(range(len(dst_lat)), desc="search for dst grids", colour="green", **tqdm_kwargs):
        # Compute Euclidean distance to all source grid points
        dx = abs(src_lon_flatten - dst_lon[j])
        dy = abs(src_lat_flatten - dst_lat[j])
        d = (dx ** 2 + dy ** 2) ** 0.5

        # find grids in src which nearest with dst at search_num th
        min_index = np.argpartition(d, search_num)[:search_num]
        searched_grids_index_ = (src_lat_flatten_index[min_index], src_lon_flatten_index[min_index])
        searched_grids_index.append(searched_grids_index_)

    return searched_grids_index
    
    
def print_ret(searched_grids_index, src_lat, src_lon):
    """Prints the indices and coordinates of the searched grid points.

    This function extracts the first set of searched grid indices from `searched_grids_index`
    and prints both their indices and corresponding latitude-longitude coordinates.

    Parameters
    ----------
    searched_grids_index : list of tuple
        A list where each element is a tuple containing two arrays: `(lat_index, lon_index)`,
        representing the indices of the nearest source grid points for each destination grid.
        Only the first set of indices (`searched_grids_index[0]`) is printed.
    src_lat : array_like
        1D array of latitude values for the source grids.
    src_lon : array_like
        1D array of longitude values for the source grids.

    Returns
    -------
    None
        This function prints the grid indices and coordinates but does not return any values.

    """
    # print result
    searched_grids_index = searched_grids_index[0]
    
    print(f"grids: {len(searched_grids_index[0])}")
    for i in range(len(searched_grids_index[0])):
        print(searched_grids_index[0][i], searched_grids_index[1][i])
        print(src_lat[searched_grids_index[0][i]], src_lon[searched_grids_index[1][i]])


def searched_grids_index_to_rows_cols_index(searched_grids_index):
    """Converts searched grid indices to row and column indices.

    This function transforms a list of grid indices (latitude and longitude indices)
    into separate row and column index arrays. It is designed for cases where each
    destination grid corresponds to exactly one source grid (i.e., a one-to-one mapping).

    Parameters
    ----------
    searched_grids_index : list of tuple
        A list where each element is a tuple containing two arrays: `(lat_index, lon_index)`.
        Each tuple represents the indices of the nearest source grid point for a given destination grid.
        **Note**: This function assumes that `len(searched_grids_index[0]) == 1`, meaning that 
        each destination grid has exactly one matched source grid.

    Returns
    -------
    rows_index : ndarray
        A 1D array of row indices corresponding to the latitude indices.
    cols_index : ndarray
        A 1D array of column indices corresponding to the longitude indices.

    Notes
    -----
    - This function is only valid for one-to-one searches, where each destination grid has exactly one matched source grid.
    - The output can be directly used for indexing a 2D array, such as:
      
      ```python
      grids_array[rows_index, cols_index] = list_values
      ```
    """
    # usage: grids_array[rows_index, cols_index] = list_values (transfer the data into a 1D array, coord (lat, lon): value)
    # note that this fucntion can only used for one to one search, len(searched_grid_index[0]) == 1
    searched_grids_index_trans = np.array(list(map(lambda index_: [index_[0][0], index_[1][0]], searched_grids_index)))
    rows_index, cols_index = searched_grids_index_trans[:, 0], searched_grids_index_trans[:, 1]
    
    return rows_index, cols_index


def searched_grids_index_to_bool_index(searched_grids_index, src_lat, src_lon):
    """Converts searched grid indices to boolean index arrays.

    This function generates boolean masks for latitude and longitude indices based on 
    the searched grid indices. The boolean masks can be used for advanced indexing 
    to extract the corresponding grid points efficiently.

    Parameters
    ----------
    searched_grids_index : list of tuple
        A list where each element is a tuple containing two arrays: `(lat_indices, lon_indices)`.
        Each tuple represents the indices of the matched source grid points for a given destination grid.

    src_lat : array-like
        A 1D array of latitude values corresponding to the source grid.

    src_lon : array-like
        A 1D array of longitude values corresponding to the source grid.

    Returns
    -------
    searched_grids_bool_index : list of tuple
        A list where each element is a tuple containing two boolean arrays: 
        `(lat_bool_mask, lon_bool_mask)`. 
        - `lat_bool_mask` is a boolean mask for latitude indices.
        - `lon_bool_mask` is a boolean mask for longitude indices.

    Notes
    -----
    - Using integer indexing on a dataset with multiple matches results in retrieving all cross-points, 
      leading to an array of shape `(N, N)`, where `N` is the number of matched grids.
    - Using boolean indexing ensures that only the exact matched points are selected, preserving 
      a one-to-one mapping. This results in a more compact array of shape `(M, K)`, where `M * K = N`.
      
    """
    # useage: params_dataset_level0.variables["depth"][0, searched_grids_bool_index[0][0], searched_grids_bool_index[0][1]].shape
    # use integer index will return all corss points: a searched_grids_index with length 143 will return (143 x 143)
    # use bool index will return one-by-one points: a searched_grids_bool_index with length 143 will return (11 x 13) (size=143)
    searched_grids_bool_index = []
    for i in range(len(searched_grids_index)):
        searched_grid_index = searched_grids_index[i]
        searched_grid_bool_index_lat = np.full((len(src_lat), ), fill_value=False, dtype=bool)
        searched_grid_bool_index_lon = np.full((len(src_lon), ), fill_value=False, dtype=bool)
        
        searched_grid_bool_index_lat[searched_grid_index[0]] = True
        searched_grid_bool_index_lon[searched_grid_index[1]] = True
        
        searched_grids_bool_index.append((searched_grid_bool_index_lat, searched_grid_bool_index_lon))
    
    return searched_grids_bool_index
    