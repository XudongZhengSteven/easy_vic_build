# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
""" 
flow_direction - A Python module for calculating flow direction from a depression-filled DEM.

This module provides functions for calculating flow direction with multiple algorithms.

Functions:
----------
    - `d8_flowdirection`: Calculate D8 flow direction from a depression-filled DEM.

Usage:
------
    1. Call `d8_flowdirection` function with the WhiteboxTools environment and input DEM file.

Example:
--------
    >>> from hydroanalysis_wbw import d8_flowdirection
    >>> flow_dir = d8_flowdirection(wbe, "filled_dem.tif")

Dependencies:
-------------
    - `whitebox_workflows`: A library that facilitates geospatial processing tasks, such as DEM filling,
      flow direction, and stream extraction.

"""

def d8_flowdirection(
    wbe,
    filled_dem,
    output_file="flow_direction.tif",
    pourpoint_x_index=None,
    pourpoint_y_index=None,
    pourpoint_direction_code=None,
    **kwargs
):
    """Calculate D8 flow direction from a depression-filled DEM.
    
    Parameters
    ----------
    wbe : `WbEnvironment`
        WhiteboxTools workflow environment instance
        
    filled_dem : `WbRaster`
        Path to filled DEM raster file or WbRaster object. Must be hydrologically
        conditioned (depressions filled and flats resolved)
        
    output_file : str, optional
        Output file path for flow direction raster (default="flow_direction.tif")
        
    pourpoint_x_index : int, optional
        Column index of pour point where flow direction should be manually set
        
    pourpoint_y_index : int, optional
        Row index of pour point where flow direction should be manually set
        
    pourpoint_direction_code : int, optional
        D8 flow direction code (1-8) to assign at pour point location. 
        Only used if pourpoint indices are provided.
        
    **kwargs : dict, optional
        Additional parameters for d8_pointer:
        
        - esri_pointer : bool, optional
            Whether to use ESRI-style flow direction encoding (default=True)
            
        - num_procs : int, optional
            Number of processors to use for calculation

    Returns
    -------
    flow_direction: `WbRaster`
        D8 flow direction raster where values represent flow direction:
        - 1: East
        - 2: Southeast
        - 4: South
        - 8: Southwest
        - 16: West
        - 32: Northwest
        - 64: North
        - 128: Northeast
        (ESRI encoding style when esri_pointer=True)

    Examples
    --------
    >>> # Basic flow direction calculation
    >>> flow_dir = d8_flowdirection(wbe, "filled_dem.tif")
    
    >>> # With manual pour point adjustment
    >>> flow_dir = d8_flowdirection(wbe, "filled_dem.tif",
    ...                           pourpoint_x_index=100,
    ...                           pourpoint_y_index=200,
    ...                           pourpoint_direction_code=1)

    Notes
    -----
    1. Input DEM must be properly hydrologically conditioned (depressions filled
       and flat areas resolved) before flow direction calculation.
    2. ESRI encoding (default) uses powers of 2 (1,2,4,8,16,32,64,128) while
       alternative encodings may use simple 1-8 values.
    3. Pour point modification is typically used to enforce watershed outlets or
       correct known flow direction errors.
    4. Flow direction raster should be checked using `validate_flow_direction`
       before subsequent analysis.
    """
    # kwargs
    kwargs_ = {"esri_pointer": True}
    kwargs_.update(kwargs)
    kwargs = kwargs_
    
    # flow direction
    flow_direction = wbe.d8_pointer(filled_dem, **kwargs)
    
    # modify code at pourpoint if provided
    if pourpoint_x_index is not None:
        flow_direction[pourpoint_y_index, pourpoint_x_index] = pourpoint_direction_code
    
    # write
    wbe.write_raster(flow_direction, output_file)
    # show(flow_direction, colorbar_kwargs={'label': 'flow direction (D8)'})

    return flow_direction
