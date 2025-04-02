
# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
outlet_detection - A Python module for detecting and processing watershed outlet points.

This module provides functions for identifying main outlets from flow accumulation data
and snapping outlet points to stream networks for accurate watershed delineation.

Functions:
----------
    - `detect_main_outlet`: Identifies the main watershed outlet from flow accumulation data.
    - `snap_outlet_to_stream`: Snaps outlet points to the nearest stream pixel.
    - `detect_outlets_with_reference`: Creates and processes outlet points from reference coordinates.

Usage:
------
    1. Use `detect_main_outlet` to find the primary outlet from flow accumulation data.
    2. Refine outlet positions with `snap_outlet_to_stream` for accurate watershed delineation.
    3. Process custom outlet locations with `detect_outlets_with_reference`.

Example:
--------
    >>> from hydroanalysis_wbw import detect_main_outlet
    >>> coords, indices = detect_main_outlet("flow_accumulation.tif")
    >>> snapped_outlet = snap_outlet_to_stream(wbe, "outlet.shp", stream_raster)

Dependencies:
-------------
    - `numpy`: For numerical array operations and calculations.
    - `rasterio`: For reading and processing raster data.
    - `whitebox_workflows`: For geospatial processing operations.
    - `...geo_func.create_gdf`: For creating geographic data frame objects.
    
"""

import numpy as np
import rasterio
from ...geo_func.create_gdf import CreateGDF

def detect_main_outlet(
    flow_acc_path,
    output_file_path="main_outlet.shp",
    crs_str="EPSG:4326",
):
    """
    Detect the main outlet point from flow accumulation data.

    The main outlet is identified as the pixel with maximum flow accumulation value.

    Parameters
    ----------
    flow_acc_path : str
        Path to the flow accumulation raster file.
        
    output_file_path : str, optional
        Output path for the main outlet shapefile (default "main_outlet.shp").
        
    crs_str : str, optional
        Coordinate reference system string (default "EPSG:4326").

    Returns
    -------
    tuple
        A tuple containing:
        - (x_coord, y_coord) : tuple of floats
            Geographic coordinates of the main outlet
            
        - (max_col, max_row) : tuple of ints
            Column and row indices of the main outlet in the raster

    Notes
    -----
    The function performs these steps:
    1. Reads the flow accumulation raster
    2. Finds the pixel with maximum accumulation value
    3. Converts pixel coordinates to geographic coordinates
    4. Creates and saves a point feature for the outlet
    """
    with rasterio.open(flow_acc_path) as src:
        flow_acc_array = src.read(1)
        transform = src.transform
        
        masked_array = np.ma.masked_equal(flow_acc_array, src.nodata)
        max_row, max_col = np.unravel_index(
            np.argmax(masked_array),
            masked_array.shape
        )
        
        x_coord, y_coord = transform * (max_col + 0.5, max_row + 0.5)
        
        meta = src.meta.copy()
        meta.update(dtype=rasterio.uint8, count=1, nodata=0)

        # create outlet gdf
        cgdf = CreateGDF()
        outlet_gdf = cgdf.createGDF_points([x_coord], [y_coord], crs=crs_str)
        
        # save
        outlet_gdf.to_file(output_file_path)
    
    return (x_coord, y_coord), (max_col, max_row)


def snap_outlet_to_stream(
    wbe,
    outlet_vector_path,
    stream_raster,
    output_file="snaped_outlet.shp",
    **kwargs
):
    """
    Snap outlet points to the nearest stream pixel.

    Parameters
    ----------
    wbe : `WbEnvironment`
        WhiteboxTools workflow environment instance.
        
    outlet_vector_path : str
        Path to the outlet point vector file.
        
    stream_raster : `WbRaster`
        Extracted stream raster (binary).
                
    output_file : str, optional
        Output path for the snapped outlet shapefile (default "snaped_outlet.shp").
        
    **kwargs : dict
        Additional keyword arguments passed to `jenson_snap_pour_points` method.
            - snap_dist: float
                Maximum snap distance (in meters) for snaping outlet to stream. 

    Returns
    -------
    snaped_outlet_vector: `WbVector`
        Vector file containing the snapped outlet points.

    Notes
    -----
    This function ensures outlet points are precisely positioned on stream pixels,
    which is important for accurate watershed delineation.
    """
    # Let's extract the watershed for a specific outlet point
    outlet_vector = wbe.read_vector(outlet_vector_path) # This is a vector point that was included when we downloaded the `mill_brook` dataset.
    
    # Make sure that the outlet is positioned along the stream
    snaped_outlet_vector = wbe.jenson_snap_pour_points(outlet_vector, stream_raster, **kwargs)
    
    wbe.write_vector(snaped_outlet_vector, output_file)
    
    return snaped_outlet_vector


def detect_outlets_with_reference(
    wbe,
    x_coords,
    y_coords,
    stream_raster,
    crs_str="EPSG:4326",
    output_file_path="outlets_with_reference.shp",
    snaped_output_file_path="snaped_outlets_with_reference.shp",
    **snap_outlet_to_stream_kwargs,
):
    """
    Create and snap outlet points to streams based on reference coordinates.

    Parameters
    ----------
    wbe : `WbEnvironment`
        WhiteboxTools workflow environment instance.
        
    x_coords : list of float
        List of x-coordinates for outlet points.
        
    y_coords : list of float
        List of y-coordinates for outlet points.
        
    stream_raster : `WbRaster`
        Extracted stream raster (binary).
        
    crs_str : str, optional
        Coordinate reference system string (default "EPSG:4326").
        
    output_file_path : str, optional
        Output path for initial outlet points shapefile (default "outlets_with_reference.shp").
        
    snaped_output_file_path : str, optional
        Output path for snapped outlet points shapefile (default "snaped_outlets_with_reference.shp").
        
    **snap_outlet_to_stream_kwargs : dict
        Additional keyword arguments passed to snap_outlet_to_stream function.
            - snap_dist: float
                Maximum snap distance (in meters) for snaping outlet to stream. 

    Returns
    -------
    tuple
        A tuple containing:
        - outlet_gdf : GeoDataFrame
            Initial outlet points before snapping.
            
        - snaped_outlet_vector : GeoDataFrame
            Outlet points after snapping to streams.

    Notes
    -----
    The function performs these steps:
    1. Creates point features from the input coordinates
    2. Saves the initial points to a shapefile
    3. Snaps each point to the nearest stream pixel
    4. Saves the snapped points to another shapefile
    """
    # create outlet gdf
    cgdf = CreateGDF()
    outlet_gdf = cgdf.createGDF_points(x_coords, y_coords, crs=crs_str)
    
    # save
    outlet_gdf.to_file(output_file_path)
    
    # snap to stream
    snaped_outlet_vector = snap_outlet_to_stream(
        wbe,
        outlet_vector_path=output_file_path,
        stream_raster=stream_raster,
        output_file=snaped_output_file_path,
        **snap_outlet_to_stream_kwargs,
    )
    
    return outlet_gdf, snaped_outlet_vector
    
