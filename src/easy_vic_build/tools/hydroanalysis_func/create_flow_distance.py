# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: create_flow_distance

This module contains the function `create_flow_distance`, which calculates the flow distance
for a given flow direction array and its associated x and y grid length arrays. The function
uses a mapping of flow directions to respective distance types (zonal, meridional, diagonal, and edge)
and applies the appropriate formula to compute the flow distance for each grid cell. The result
is then saved as a GeoTIFF file, preserving the spatial transformation and coordinate reference system.

Functions:
----------
    - create_flow_distance: Computes the flow distance for a flow direction array based on specified
      distance types, and saves the result as a GeoTIFF file.

Dependencies:
-------------
    - rasterio: Provides functionality for reading and writing raster data, as well as managing spatial references.
    - numpy: Used for efficient numerical operations, particularly for array manipulations and vectorization.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com
"""


import numpy as np
import rasterio
from rasterio import CRS


def create_flow_distance(
    flow_distance_path,
    flow_direction_array,
    x_length_array,
    y_length_array,
    transform,
    crs_str="EPSG:4326",
):
    """
    Calculates the flow distance based on the given flow direction array and the respective x and y grid lengths.
    The function maps flow directions to corresponding distance types and computes the flow distance for each grid
    cell accordingly. The result is saved as a GeoTIFF file.

    Parameters:
    -----------
    flow_distance_path : str
        The file path where the calculated flow distance will be saved as a GeoTIFF file.
    flow_direction_array : numpy.ndarray
        A 2D array representing the flow directions of each grid cell.
    x_length_array : numpy.ndarray
        A 2D array representing the horizontal length (in meters) of each grid cell.
    y_length_array : numpy.ndarray
        A 2D array representing the vertical length (in meters) of each grid cell.
    transform : affine.Affine
        The affine transform representing the spatial reference of the data.
    crs_str : str, optional
        The coordinate reference system in EPSG format. Default is "EPSG:4326".

    Returns:
    --------
    None
        The flow distance is saved directly to the specified file path as a GeoTIFF.
    """
    flow_direction_distance_map = {
        "zonal": [64, 4],
        "meridional": [1, 16],
        "diagonal": [32, 128, 8, 2],
        "edge": [0],
    }
    flow_distance_func_map = {
        "zonal": lambda x_length, y_length: y_length,
        "meridional": lambda x_length, y_length: x_length,
        "diagonal": lambda x_length, y_length: (x_length**2 + y_length**2) ** 0.5,
        "edge": lambda x_length, y_length: (x_length**2 + y_length**2) ** 0.5,
    }

    def flow_distance_funcion(flow_direction, x_length, y_length):
        """
        Determines the distance type based on the flow direction and computes the flow distance.

        Parameters:
        -----------
        flow_direction : int
            The flow direction for the current grid cell.
        x_length : float
            The horizontal length of the grid cell.
        y_length : float
            The vertical length of the grid cell.

        Returns:
        --------
        float
            The calculated flow distance for the grid cell.
        """
        for k in flow_direction_distance_map:
            if flow_direction in flow_direction_distance_map[k]:
                distance_type = k
                break

        flow_distance_func = flow_distance_func_map[distance_type]
        return flow_distance_func(x_length, y_length)

    flow_distance_funcion_vect = np.vectorize(flow_distance_funcion)
    flow_distance_array = flow_distance_funcion_vect(
        flow_direction_array, x_length_array, y_length_array
    )

    # save as tif file, transform same as dem
    with rasterio.open(
        flow_distance_path,
        "w",
        driver="GTiff",
        height=flow_distance_array.shape[0],
        width=flow_distance_array.shape[1],
        count=1,
        dtype=flow_distance_array.dtype,
        crs=CRS.from_string(crs_str),
        transform=transform,
    ) as dst:
        dst.write(flow_distance_array, 1)
