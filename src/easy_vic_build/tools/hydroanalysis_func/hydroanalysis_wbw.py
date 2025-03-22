# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: hydroanalysis_wbw

This module contains the `hydroanalysis_wbw` function, which performs hydrological analysis 
using the WhiteboxTools (WBW) library. The function includes operations such as filling DEM depressions, 
computing flow direction and accumulation, and optionally generating stream raster and vector data. 
The module is designed to work with geospatial raster data for hydrological modeling and analysis.

Functions:
----------
    - hydroanalysis_wbw: Performs hydrological analysis on a DEM using WhiteboxTools. 
      This includes filling depressions, calculating flow direction and accumulation, and 
      optionally creating stream raster and vector data.

Dependencies:
-------------
    - os: Provides interaction with the operating system, such as path handling.
    - whitebox_workflows: A library that facilitates geospatial processing tasks, such as DEM filling, 
      flow direction, and stream extraction.
    - ..geo_func.format_conversion: Contains functions for converting raster data to shapefiles.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com
"""


import os
from whitebox_workflows import WbEnvironment, show
from ..geo_func.format_conversion import *


def hydroanalysis_wbw(workspace_dir, dem_level1_tif_path, create_stream=False, pourpoint_x_index=None, pourpoint_y_index=None, pourpoint_direction_code=None):
    """
    Perform hydrological analysis using WhiteboxTools on a DEM. This includes filling depressions, 
    calculating flow direction and accumulation, and optionally generating stream raster and vector data.

    Parameters:
    -----------
    workspace_dir : str
        The directory where output files will be saved and where the input DEM file is located.

    dem_level1_tif_path : str
        The path to the DEM file (GeoTIFF format) for which the analysis is to be performed.

    create_stream : bool, optional
        If True, stream raster and vector data will be generated (default is False).

    pourpoint_x_index : int, optional
        The x index of the pour point to modify the flow direction (default is None).

    pourpoint_y_index : int, optional
        The y index of the pour point to modify the flow direction (default is None).

    pourpoint_direction_code : int, optional
        The flow direction code to assign at the pour point (default is None).

    Returns:
    --------
    bool
        Returns True if the analysis is completed successfully.
    """
    # build flow direction based on wbw
    
    # env set
    wbe = WbEnvironment()
    wbe.working_directory = workspace_dir
    
    # read dem
    dem = wbe.read_raster(dem_level1_tif_path)
    # show(dem, colorbar_kwargs={'label': 'Elevation (m)'})
    
    # fill depressions
    filled_dem = wbe.breach_depressions_least_cost(dem)
    # filled_dem = wbe.fill_depressions(filled_dem)
    wbe.write_raster(filled_dem, 'filled_dem.tif')
    # show(filled_dem, colorbar_kwargs={'label': 'Elevation (m)'})
    # show(filled_dem - dem, colorbar_kwargs={'label': 'fill (m)'})

    # flow direction
    flow_direction = wbe.d8_pointer(filled_dem, esri_pointer=True)
    if pourpoint_x_index is not None:
        flow_direction[pourpoint_y_index, pourpoint_x_index] = pourpoint_direction_code
    
    wbe.write_raster(flow_direction, 'flow_direction.tif')
    # show(flow_direction, colorbar_kwargs={'label': 'flow direction (D8)'})
    
    # flow accumulation
    flow_acc = wbe.d8_flow_accum(flow_direction, out_type="cells", log_transform=False, input_is_pointer=True, esri_pntr=True)
    wbe.write_raster(flow_acc, 'flow_acc.tif')
    # show(flow_acc, colorbar_kwargs={'label': 'flow acc (number)'}, vmin=200)
    
    if create_stream:
        # stream raster
        stream_raster = wbe.extract_streams(flow_acc, threshold=100.0)
        wbe.write_raster(stream_raster, 'stream_raster.tif')
        # show(stream_raster, colorbar_kwargs={'label': 'stream raster (1, bool)'})
        
        # stream raster to shp
        stream_raster_shp_gdf = raster_to_shp(os.path.join(workspace_dir, 'stream_raster.tif'), os.path.join(workspace_dir, 'stream_raster_shp.shp'))
        
        # stream vector
        stream_vector = wbe.raster_streams_to_vector(stream_raster, flow_direction)
        stream_vector, tmp1, tmp2, tmp3 = wbe.vector_stream_network_analysis(stream_vector, filled_dem) # We only want the streams output
        wbe.write_vector(stream_vector, 'stream_vector.shp')
        # show(stream_vector, colorbar_kwargs={'label': 'stream vector(1, bool)'})
    
    return True