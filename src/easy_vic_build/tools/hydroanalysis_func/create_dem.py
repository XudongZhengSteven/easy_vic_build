# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: create_dem

This module contains the function `create_dem_from_params`, which generates a Digital Elevation Model (DEM) 
from input parameters and saves it as a GeoTIFF file. The function reads the latitude, longitude, and elevation 
data from a dataset, calculates the spatial transformation, and writes the DEM to the specified file path with 
the desired coordinate reference system (CRS). Optionally, the latitude values can be reversed during the 
transformation process, providing flexibility in handling different data formats.

Functions:
----------
    - create_dem_from_params: Creates a DEM from latitude, longitude, and elevation data, 
      and saves it as a GeoTIFF file. Returns the spatial transformation used for the DEM.

Dependencies:
-------------
    - os: Provides functions for interacting with the operating system, particularly for file path management.
    - rasterio: Used for reading and writing raster data, as well as defining the transformation for spatial reference.
    - numpy: Supports numerical operations, particularly for handling the input data arrays.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com
"""

import os
import rasterio
from rasterio import CRS


def create_dem_from_params(params_dataset_level1, save_path, crs_str="EPSG:4326", reverse_lat=True):
    """
    Create a Digital Elevation Model (DEM) from input parameters and save it as a GeoTIFF file.

    This function reads the latitude, longitude, and elevation data from a given dataset 
    and generates a DEM in the form of a GeoTIFF file. The resulting DEM is saved to the specified 
    path with the given coordinate reference system (CRS). Optionally, the latitude values can 
    be reversed during the transformation process.

    Parameters
    ----------
    params_dataset_level1 : Dataset
        The input dataset containing latitude, longitude, and elevation data. It should have 
        variables 'lat', 'lon', and 'elev'.
    save_path : str
        The file path where the generated DEM will be saved as a GeoTIFF file.
    crs_str : str, optional
        The coordinate reference system (CRS) to use for the output DEM. The default is "EPSG:4326".
    reverse_lat : bool, optional
        Whether to reverse the latitude values during transformation. The default is True.

    Returns
    -------
    rasterio.transform.Affine
        The transformation object used for the DEM's spatial reference.
    
    Notes
    -----
    - The function assumes that the input dataset contains variables named 'lat', 'lon', and 'elev'.
    - The function uses the rasterio library to write the DEM as a GeoTIFF file.
    """
    # read
    params_lat = params_dataset_level1.variables["lat"][:]
    params_lon = params_dataset_level1.variables["lon"][:]
    params_elev = params_dataset_level1.variables["elev"][:, :]
    
    # ====================== create and save dem_level1.tif ======================
    ulx = min(params_lon)
    uly = max(params_lat)
    xres = round((max(params_lon) - min(params_lon)) / (len(params_lon) - 1), 6)
    yres = round((max(params_lat) - min(params_lat)) / (len(params_lat) - 1), 6)
    if reverse_lat:
        transform = rasterio.transform.from_origin(ulx-xres/2, uly+yres/2, xres, yres)
    else:
        transform = rasterio.transform.from_origin(ulx-xres/2, uly-yres/2, xres, yres)
    
    if not os.path.exists(save_path):
        with rasterio.open(save_path, 'w', driver='GTiff',
                           height=params_elev.shape[0],
                           width=params_elev.shape[1],
                           count=1,
                           dtype=params_elev.dtype,
                           crs=CRS.from_string(crs_str),
                           transform=transform,
                           ) as dst:
            dst.write(params_elev, 1)
    
    return transform