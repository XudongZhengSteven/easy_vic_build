# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Basin and subbasin delineation helpers based on flow-direction rasters."""
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
    Delineate basins for user-provided snapped outlet points.

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
        ``(basins_raster, basins_vector)``.
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
    Delineate all edge-draining basins from a flow-direction raster.

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
        ``(all_basins_raster, all_basins_vector)``.
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
    Delineate subbasins draining to each stream link.

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
        ``(subbasins_raster, subbasins_vector)``.
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
    Repair invalid basin-vector geometries using ``make_valid``.

    Parameters
    ----------
    basins_vector_path : str
        Path to the input vector file (e.g., Shapefile, GeoJSON) containing basin polygons.
        
    output_file_basins_vector_path : str, optional
        Output path placeholder. The current implementation does not write to file.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with repaired geometries.
    """
    # read
    basins_vector_gdf = gpd.read_file(basins_vector_path)
    repaired_basins_vector_gdf = deepcopy(basins_vector_gdf)
    
    # repair
    repaired_basins_vector_gdf["geometry"] = basins_vector_gdf.geometry.make_valid()
    
    # convert into polygon
    repaired_basins_vector_gdf["geometry"] = repaired_basins_vector_gdf.geometry.apply(force_multipolygon_to_polygon)
    
    return repaired_basins_vector_gdf
