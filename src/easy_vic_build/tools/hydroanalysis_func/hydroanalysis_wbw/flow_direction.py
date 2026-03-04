# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
"""Flow-direction utilities for conditioned DEM rasters."""

def d8_flowdirection(
    wbe,
    filled_dem,
    output_file="flow_direction.tif",
    **kwargs
):
    """Calculate D8 flow direction from a hydrologically conditioned DEM.
    
    Parameters
    ----------
    wbe : `WbEnvironment`
        WhiteboxTools workflow environment instance
        
    filled_dem : `WbRaster`
        Path to filled DEM raster file or WbRaster object. Must be hydrologically
        conditioned (depressions filled and flats resolved)
        
    output_file : str, optional
        Output file path for flow direction raster (default="flow_direction.tif")

    **kwargs : dict, optional
        Additional parameters for d8_pointer:
        
        - esri_pointer : bool, optional
            Whether to use ESRI-style flow direction encoding (default=True)
            
        - num_procs : int, optional
            Number of processors to use for calculation

    Returns
    -------
    `WbRaster`
        Flow-direction raster written to ``output_file`` and returned.

    Notes
    -----
    The input DEM should be depression-filled before calling this function.
    """
    # kwargs
    kwargs_ = {"esri_pointer": True}
    kwargs_.update(kwargs)
    kwargs = kwargs_
    
    # flow direction
    flow_direction = wbe.d8_pointer(filled_dem, **kwargs)
    
    # write
    wbe.write_raster(flow_direction, output_file)
    # show(flow_direction, colorbar_kwargs={'label': 'flow direction (D8)'})

    return flow_direction
