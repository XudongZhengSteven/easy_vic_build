# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
"""Flow-accumulation utilities for D8 flow-direction rasters."""

def d8_flowaccumulation(
    wbe,
    flow_direction,
    output_file="flow_acc.tif",
    **kwargs
):
    """Calculate D8 flow accumulation from a flow-direction raster.

    Computes the number of upstream cells that drain into each cell, representing
    contributing area or flow accumulation.

    Parameters
    ----------
    wbe : `WbEnvironment`
        WhiteboxTools workflow environment instance
        
    flow_direction : `WbRaster`
        D8 flow-direction raster path or ``WbRaster`` object.
        
    output_file : str, optional
        Output file path for flow accumulation raster (default="flow_acc.tif")
        
    **kwargs : dict, optional
        Additional parameters for d8_flow_accum:
        
        - out_type : {'cells', 'sca', 'specific'}, optional
            Output type (default='cells'):
            - 'cells': Number of contributing cells
            - 'sca': Specific catchment area (cells * cell area)
            - 'specific': Same as 'sca'
            
        - log_transform : bool, optional
            Whether to apply logarithmic transform to output (default=False)
            
        - input_is_pointer : bool, optional
            Whether input is pointer-type (default=True)
            
        - esri_pntr : bool, optional
            Whether input uses ESRI pointer encoding (default=True)
            
        - num_procs : int, optional
            Number of processors to use for calculation

    Returns
    -------
    `WbRaster`
        Flow-accumulation raster written to ``output_file`` and returned.
    """
    # kwargs
    kwargs_ = {"out_type": "cells",
               "log_transform": False,
               "input_is_pointer": True,
               "esri_pntr": True
               }
    kwargs_.update(kwargs)
    kwargs = kwargs_
    
    # flow accumulation
    flow_acc = wbe.d8_flow_accum(flow_direction, **kwargs)
    
    # write
    wbe.write_raster(flow_acc, output_file)
    # show(flow_acc, colorbar_kwargs={'label': 'flow acc (number)'}, vmin=200)
    
    return flow_acc
