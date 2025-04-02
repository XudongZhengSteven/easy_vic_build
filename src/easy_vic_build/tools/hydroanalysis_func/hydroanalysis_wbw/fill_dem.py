# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
""" 
fill_dem - A Python module for filling depressions in a DEM using least-cost breaching algorithm.

This module contains the `filldem` function, which fills depressions in a DEM using the least-cost breaching algorithm.
The function is designed to work with geospatial raster data for hydrological modeling and analysis.

Functions:
----------
    - `filldem`: Fills depressions in a DEM using the least-cost breaching algorithm.

Usage:
------
    1. Call the `filldem` function with the input DEM raster file path.
    2. Optionally specify additional parameters such as `max_dist`, `flat_increment`, and `min_dist`.

Dependencies:
-------------
    - `whitebox_workflows`: A library that facilitates geospatial processing tasks, such as DEM filling and flow direction.

"""
def filldem(
    wbe,
    dem_path,
    output_file="filled_dem.tif",
    **kwargs
):
    """Fill depressions in a DEM using least-cost breaching algorithm.
    
    Parameters
    ----------
    wbe : `WbEnvironment`
        WhiteboxTools workflow environment instance
        
    dem_path : str
        Path to input DEM raster file
        
    output_file : str, optional
        Output file path for filled DEM (default="filled_dem.tif")
        
    **kwargs : dict, optional
        Additional parameters for breach_depressions_least_cost:
        
        - max_dist : float
            Maximum breach channel length (in meters). Recommended value is DEM 
            resolution multiplied by terrain complexity factor:
            - 10-20x for simple terrain
            - 30-50x for complex/mountainous terrain
            Example: 500.0 for 30m resolution DEM (e.g., SRTM)
            
        - flat_increment : float
            Elevation increment applied to flat areas (prevents flow direction 
            artifacts). Recommended:
            - 0.001 for meter-level DEMs
            - 0.0001 for sub-meter DEMs
            
        - min_dist : bool
            Whether to enforce minimum distance paths (default=True). Set to False
            may create shorter but less natural breach paths.

    Returns
    -------
    filled_dem: `WbRaster`
        Depression-filled DEM raster object

    Examples
    --------
    >>> # Basic usage with default parameters
    >>> filled = filldem(wbe, "input_dem.tif")
    
    >>> # Advanced usage with custom parameters
    >>> filled = filldem(wbe, "input_dem.tif", 
    ...                 output_file="output_dem.tif",
    ...                 max_dist=100.0, 
    ...                 flat_increment=0.001)

    Notes
    -----
    1. This function uses WhiteboxTools' breach_depressions_least_cost algorithm
       which is generally preferred over simple filling for hydrological applications.
    2. The filled DEM should typically be followed by flat area resolution 
       (resolve_flats) before flow direction calculation.
    3. For large datasets, consider setting `num_procs` in WbEnvironment for
       parallel processing.
    """
    # read
    dem = wbe.read_raster(dem_path)
    # show(dem, colorbar_kwargs={'label': 'Elevation (m)'})
    
    # kwargs
    kwargs_ = {
        "flat_increment": 0.001,
        "max_dist": 500
    }
    kwargs_.update(kwargs)
    kwargs = kwargs_
    
    # fill depressions
    filled_dem = wbe.breach_depressions_least_cost(dem, **kwargs)
    filled_dem = wbe.fill_depressions(filled_dem, flat_increment=kwargs["flat_increment"])
    # filled_dem = wbe.resolve_flats(filled_dem) # resolve flats
    # show(filled_dem, colorbar_kwargs={'label': 'Elevation (m)'})
    # show(filled_dem - dem, colorbar_kwargs={'label': 'fill (m)'})
    
    # write
    wbe.write_raster(filled_dem, output_file)
    
    return filled_dem
