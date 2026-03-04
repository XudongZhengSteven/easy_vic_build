
# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Outlet detection and snapping helpers for watershed delineation."""

import numpy as np
import rasterio
from ...geo_func.create_gdf import CreateGDF

def detect_main_outlet(
    flow_acc_path,
    output_file_path="main_outlet.shp",
    crs_str="EPSG:4326",
):
    """
    Detect the primary outlet from a flow-accumulation raster.

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
        ``((x_coord, y_coord), (max_col, max_row))`` where the first tuple is
        outlet coordinates and the second is raster index.
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
    Snap outlet points to the nearest stream cell.

    Parameters
    ----------
    wbe : `WbEnvironment`
        WhiteboxTools workflow environment instance.
        
    outlet_vector_path : str
        Path to the outlet point vector file.
        
    stream_raster : `WbRaster`
        Extracted stream raster (binary).
                
    output_file : str, optional
        Output path for the snapped outlet shapefile (default
        ``"snaped_outlet.shp"``).
        
    **kwargs : dict, optional
        Additional keyword arguments passed to ``jenson_snap_pour_points``.
            - snap_dist: float
                Maximum snap distance for snapping outlets to streams.

    Returns
    -------
    `WbVector`
        Snapped outlet vector object.
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
    Create outlet points from coordinates and snap them to streams.

    Parameters
    ----------
    wbe : `WbEnvironment`
        WhiteboxTools workflow environment instance.
        
    x_coords : sequence of float
        X coordinates for outlet points.
    y_coords : sequence of float
        Y coordinates for outlet points.
        
    stream_raster : `WbRaster`
        Extracted stream raster (binary).
        
    crs_str : str, optional
        Coordinate reference system string (default "EPSG:4326").
        
    output_file_path : str, optional
        Output path for initial outlet points shapefile (default "outlets_with_reference.shp").
        
    snaped_output_file_path : str, optional
        Output path for snapped outlet points shapefile (default "snaped_outlets_with_reference.shp").
        
    **snap_outlet_to_stream_kwargs : dict
        Additional keyword arguments forwarded to ``snap_outlet_to_stream``.
            - snap_dist: float
                Maximum snap distance for snapping outlets to streams.

    Returns
    -------
    tuple
        ``(outlet_gdf, snaped_outlet_vector)``.
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
    
