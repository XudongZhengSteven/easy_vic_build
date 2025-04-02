# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
""" 
flow_accumulation - A Python module for calculating flow accumulation from flow direction raster.

This module provides functions for calculating flow accumulation from flow direction raster.

Functions:
----------
    - `d8_flowaccumulation`: Calculate D8 flow accumulation from flow direction raster.

Usage:
------
    1. Call `d8_flowaccumulation` function with the WhiteboxTools environment and input flow direction raster file.

Example:
--------
    >>> from hydroanalysis_wbw import d8_flowaccumulation
    >>> flow_acc = d8_flowaccumulation(wbe, "flow_dir.tif")

Dependencies:
-------------
    - `whitebox_workflows`: A library that facilitates geospatial processing tasks, such as DEM filling,
      flow direction, and stream extraction.

"""

def d8_flowaccumulation(
    wbe,
    flow_direction,
    output_file="flow_acc.tif",
    **kwargs
):
    """Calculate D8 flow accumulation from flow direction raster.

    Computes the number of upstream cells that drain into each cell, representing
    contributing area or flow accumulation.

    Parameters
    ----------
    wbe : `WbEnvironment`
        WhiteboxTools workflow environment instance
        
    flow_direction : `WbRaster`
        Path to D8 flow direction raster file or WbRaster object. Must use
        ESRI-style encoding (powers of 2) if esri_pntr=True.
        
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
    flow_acc: `WbRaster`
        Flow accumulation raster where each cell value represents:
        - When out_type='cells': Count of upstream cells
        - When out_type='sca'/'specific': Contributing area in square map units

    Examples
    --------
    >>> # Basic flow accumulation calculation
    >>> flow_acc = d8_flowaccumulation(wbe, "flow_dir.tif")

    >>> # With specific catchment area output
    >>> flow_acc = d8_flowaccumulation(wbe, "flow_dir.tif",
    ...                              out_type='sca')

    Notes
    -----
    1. Input flow direction must be calculated from a properly conditioned DEM
       (depressions filled and flats resolved).
    2. For hydrological applications, 'cells' output is typically used for
       stream delineation while 'sca' is used for soil moisture modeling.
    3. Logarithmic transform (log_transform=True) can help visualize large
       value ranges but should not be used for quantitative analysis.
    4. Result should be checked for artifacts using visual inspection before
       use in subsequent analysis.
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
