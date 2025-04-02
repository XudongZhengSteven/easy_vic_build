# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
basin_delineation - A Python module for watershed and subbasin delineation from DEM data.

This module provides functions for delineating watershed basins and subbasins using flow direction
and stream network data, with options for output smoothing and format conversion.

Functions:
----------
    - `delineate_basins_for_snaped_outlets`: Delineates watersheds for specified outlet points.
    - `delineate_all_basins`: Delineates all watersheds draining to DEM edges.
    - `delineate_subbasins`: Delineates subbasins for each link in stream network.

Usage:
------
    1. Prepare flow direction and stream network data
    2. Choose appropriate delineation function based on needs:
       - Use `delineate_basins_for_snaped_outlets` for custom outlets
       - Use `delineate_all_basins` for complete watershed partitioning
       - Use `delineate_subbasins` for stream network-based subcatchments
    3. Specify output paths and smoothing parameters as needed

Example:
--------
    >>> from hydroanalysis_wbw import basin_delineation
    >>> basins_raster, basins_vector = delineate_basins_for_snaped_outlets(
    ...     wbe, flow_dir, outlets_vector)
    >>> all_basins = delineate_all_basins(wbe, flow_dir)
    >>> subbasins = delineate_subbasins(wbe, flow_dir, stream_raster)

Dependencies:
-------------
    - `whitebox_workflows`: Required for core watershed delineation algorithms
    - `rasterio`: Used for raster data handling (indirect dependency)
    - `geopandas`: Used for vector data handling (indirect dependency)

Notes:
------
    - All functions support both raster and vector outputs
    - Vector outputs can be smoothed for cartographic quality
    - ESRI and non-ESRI flow direction conventions supported
    - Designed to work with D8 flow direction algorithm
"""
import geopandas as gpd
from copy import deepcopy

def delineate_basins_for_snaped_outlets(
    wbe,
    flow_direction,
    snaped_outlets_vector,
    output_file_basins_raster="basins_raster.tif",
    output_file_basins_vector="basins_vector.shp",
    smooth_vector=False,
    smooth_filter_size=5,
    esri_pointer=True,
):
    """
    Delineate watershed basins for snapped outlet points.

    Parameters
    ----------
    wbe : `WbEnvironment`
        WhiteboxTools workflow environment instance.
        
    flow_direction : `WbRaster`
        D8 flow direction raster (path or object). Values should follow WhiteboxTools
        D8 encoding (0=East, 1=NE, 2=North, etc.).
        
    snaped_outlets_vector : `WbVector`
        Vector file containing snapped outlet points.
        
    output_file_basins_raster : str, optional
        Output path for basins raster (default "basins_raster.tif").
        
    output_file_basins_vector : str, optional
        Output path for basins vector (default "basins_vector.shp").
    
    smooth_vector : bool, optional
        Whether to smooth output vector polygons (default False).
        Note: smooth_vector may cause edges to not align properly.
        
    smooth_filter_size : int, optional
        Size of smoothing filter (default 5).
        
    esri_pointer : bool, optional
        Whether flow direction uses ESRI pointer convention (default True).

    Returns
    -------
    tuple
        A tuple containing:
        - basins_raster : `WbRaster`
            Raster of delineated basins.
            
        - basins_vector : `WbVector`
            Vector polygons of basins.

    Notes
    -----
    This function:
    1. Uses watershed algorithm to delineate basins
    2. Converts results to vector polygons
    3. Optionally smooths vector boundaries
    4. Saves both raster and vector outputs
    """
    # read
    if isinstance(flow_direction, str):
        flow_direction = wbe.read_raster(flow_direction)
    
    if isinstance(snaped_outlets_vector, str):
        snaped_outlets_vector = wbe.read_vector(snaped_outlets_vector)
    
    # Call watershed to delineate basin for given outlet
    basins_raster = wbe.watershed(
        flow_direction,
        snaped_outlets_vector,
        esri_pointer
    )
    
    wbe.write_raster(basins_raster, output_file_basins_raster)

    # Converting raster to vector
    basins_vector = wbe.raster_to_vector_polygons(basins_raster)
    
    if smooth_vector:
        basins_vector = wbe.smooth_vectors(basins_vector, filter_size=smooth_filter_size)
    
    wbe.write_vector(basins_vector, output_file_basins_vector)
    
    return basins_raster, basins_vector


def delineate_all_basins(
    wbe,
    flow_direction,
    output_file_all_basins_raster="all_basins_raster.tif",
    output_file_all_basins_vector="all_basins_vector.shp",
    smooth_vector=True,
    smooth_filter_size=5,
    esri_pointer=True,
):
    """
    Delineate all watershed basins draining to DEM edges.

    Parameters
    ----------
    wbe : `WbEnvironment`
        WhiteboxTools workflow environment instance.
        
    flow_direction : `WbRaster`
        D8 flow direction raster (path or object). Values should follow WhiteboxTools
        D8 encoding (0=East, 1=NE, 2=North, etc.).
        
    output_file_all_basins_raster : str, optional
        Output path for basins raster (default "all_basins_raster.tif").
        
    output_file_all_basins_vector : str, optional
        Output path for basins vector (default "all_basins_vector.shp").
        
    smooth_vector : bool, optional
        Whether to smooth output vector polygons (default True).
        
    smooth_filter_size : int, optional
        Size of smoothing filter (default 5).
        
    esri_pointer : bool, optional
        Whether flow direction uses ESRI pointer convention (default True).

    Returns
    -------
    tuple
        A tuple containing:
        - all_basins_raster : `WbRaster`
            Raster of all delineated basins
            
        - all_basins_vector : `WbVector`
            Vector polygons of all basins

    Notes
    -----
    This function:
    1. Identifies basins draining to each DEM edge outlet
    2. Converts results to vector polygons
    3. Optionally smooths vector boundaries
    4. Saves both raster and vector outputs
    """
    # Extract all of the watersheds, draining to each outlet on the edge of the DEM using the 'basins' function.
    all_basins_raster = wbe.basins(flow_direction, esri_pointer)
    wbe.write_raster(all_basins_raster, output_file_all_basins_raster)
    
    # Converting raster to vector
    all_basins_vector = wbe.raster_to_vector_polygons(all_basins_raster)
    
    if smooth_vector:
        all_basins_vector = wbe.smooth_vectors(all_basins_vector, filter_size=smooth_filter_size)
    
    wbe.write_vector(all_basins_vector, output_file_all_basins_vector)
    
    return all_basins_raster, all_basins_vector
    
def delineate_subbasins(
    wbe,
    flow_direction,
    stream_raster,
    output_file_subbasins_raster="subbasins_raster.tif",
    output_file_subbasins_vector="subbasins_vector.shp",
    smooth_vector=True,
    smooth_filter_size=5,
    esri_pointer=True,
):
    """
    Delineate subbasins for each stream network link.

    Parameters
    ----------
    wbe : `WbEnvironment`
        WhiteboxTools workflow environment instance.
        
    flow_direction : `WbRaster`
        D8 flow direction raster (path or object). Values should follow WhiteboxTools
        D8 encoding (0=East, 1=NE, 2=North, etc.).
        
    stream_raster : `WbRaster`
        Extracted stream raster (binary).
        
    output_file_subbasins_raster : str, optional
        Output path for subbasins raster (default "subbasins_raster.tif").
        
    output_file_subbasins_vector : str, optional
        Output path for subbasins vector (default "subbasins_vector.shp").
        
    smooth_vector : bool, optional
        Whether to smooth output vector polygons (default True).
        
    smooth_filter_size : int, optional
        Size of smoothing filter (default 5).
        
    esri_pointer : bool, optional
        Whether flow direction uses ESRI pointer convention (default True).

    Returns
    -------
    tuple
        A tuple containing:
        - subbasins_raster : `WbRaster`
            Raster of delineated subbasins.
            
        - subbasins_vector : `WbVector`
            Vector polygons of subbasins.

    Notes
    -----
    This function:
    1. Identifies subcatchments draining to each stream link
    2. Converts results to vector polygons
    3. Optionally smooths vector boundaries
    4. Saves both raster and vector outputs
    """
    # How about extracting subcatchments, i.e. the areas draining directly to each link in the stream network?
    subbasins_raster = wbe.subbasins(flow_direction, stream_raster, esri_pointer)
    wbe.write_raster(subbasins_raster, output_file_subbasins_raster)
    
    # Converting raster to vector
    subbasins_vector = wbe.raster_to_vector_polygons(subbasins_raster)
    
    if smooth_vector:
        subbasins_vector = wbe.smooth_vectors(subbasins_vector, filter_size=smooth_filter_size)
    
    wbe.write_vector(subbasins_vector, output_file_subbasins_vector)
    
    return subbasins_raster, subbasins_vector
    
    
def repair_basins_vector(
    basins_vector_path,
    output_file_basins_vector_path="repaired_basins_vector.shp"
):
    """
    Repair invalid geometries in a vector file (e.g., Shapefile) containing basins.

    This function reads a vector file, fixes topological issues (e.g., self-intersections,
    duplicate nodes) using Shapely's `make_valid()`, and saves the repaired geometries
    to a new file.

    Parameters
    ----------
    basins_vector_path : str
        Path to the input vector file (e.g., Shapefile, GeoJSON) containing basin polygons.
        
    output_file_basins_vector_path : str, optional
        Path to save the repaired vector file. Defaults to "repaired_basins_vector.shp".

    Returns
    -------
    repaired_basins_vector_gdf: `gpd.GeoDataFrame`
        GeoDataFrame with repaired geometries.

    Examples
    --------
    >>> repaired_gdf = repair_basins_vector("basins.shp", "repaired_basins.shp")
    >>> print(repaired_gdf.head())
    """
    # read
    basins_vector_gdf = gpd.read_file(basins_vector_path)
    repaired_basins_vector_gdf = deepcopy(basins_vector_gdf)
    
    # repair
    repaired_basins_vector_gdf["geometry"] = basins_vector_gdf.geometry.make_valid()
    
    # convert into polygon
    repaired_basins_vector_gdf["geometry"] = repaired_basins_vector_gdf.geometry.apply(force_multipolygon_to_polygon)
    
    return repaired_basins_vector_gdf
