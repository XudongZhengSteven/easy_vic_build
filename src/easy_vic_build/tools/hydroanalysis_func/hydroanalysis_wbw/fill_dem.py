# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
"""DEM conditioning helpers for hydroanalysis workflows.

This module provides utilities to estimate terrain-scale increments, add
deterministic perturbations to break flat ties, and generate hydrologically
conditioned DEM rasters using WhiteboxTools operations.
"""
import numpy as np
from scipy import ndimage
import rasterio

def estimate_typical_dz(dem, nodata=None):
    """Estimate representative local elevation difference for a DEM.

    Parameters
    ----------
    dem : numpy.ndarray
        Input DEM array.
    nodata : float, optional
        Nodata value to be ignored during estimation.

    Returns
    -------
    float
        Median absolute difference between each cell and its 8 neighbors.
    """
    if nodata is not None:
        mask = (dem == nodata)
        dem = dem.copy()
        dem[mask] = np.nan
    diffs = []
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for dy, dx in shifts:
        shifted = ndimage.shift(dem, shift=(dy, dx), order=0, mode='nearest')
        diff = np.abs(dem - shifted)
        diffs.append(diff)
        
    all_diffs = np.stack(diffs, axis=0)
    vals = all_diffs.reshape(-1)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return 0.0
    return float(np.median(vals))

def add_deterministic_perturbation(dem, epsilon_ratio=0.01):
    """Add tiny monotonic perturbation to break flat-elevation ties.

    Parameters
    ----------
    dem : numpy.ndarray
        Input DEM array.
    epsilon_ratio : float, optional
        Ratio used to scale perturbation based on minimum non-zero elevation
        difference in the DEM.

    Returns
    -------
    numpy.ndarray
        Perturbed DEM array with deterministic tie-breaking offsets.
    """
    diffs = np.abs(dem[1:, :] - dem[:-1, :])
    diffs = np.append(diffs, np.abs(dem[:, 1:] - dem[:, :-1]))
    min_dz = np.min(diffs[diffs > 0])
    epsilon = min_dz * epsilon_ratio

    rows, cols = dem.shape
    idx = (np.arange(rows)[:, None] * cols + np.arange(cols)[None, :]).astype(np.float64)
    idx_norm = idx / idx.max()
    dem_pert = dem + epsilon * idx_norm
    return dem_pert

def filldem(
    wbe,
    dem_path,
    output_file="filled_dem.tif",
    add_perturbation=False,
    burn_streams_path=None,
    fill_depressions_bool=True,
    **kwargs
):
    """Condition a DEM using least-cost breaching and optional post-processing.
    
    Parameters
    ----------
    wbe : `WbEnvironment`
        WhiteboxTools workflow environment instance
        
    dem_path : str
        Path to input DEM raster file
        
    output_file : str, optional
        Output file path for filled DEM (default="filled_dem.tif")

    add_perturbation : bool, optional
        If ``True``, add a small deterministic perturbation before breaching to
        reduce flat-area tie effects.

    burn_streams_path : str, optional
        Optional path to stream vectors used by ``fill_burn`` after filling.

    fill_depressions_bool : bool, optional
        If ``True``, run ``wbe.fill_depressions`` after breaching.
        
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
    `WbRaster`
        Conditioned DEM raster object. If ``burn_streams_path`` is provided,
        returns the burned DEM.

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
    ``flat_increment`` is auto-estimated from local elevation differences and
    can be overridden through ``**kwargs``.
    """
    # add_deterministic_perturbation
    
    with rasterio.open(dem_path) as src:
        dem_array = src.read(1)
        nodata = src.nodata
        profile = src.profile
        dz_typ = estimate_typical_dz(dem_array)
        flat_increment = max(1e-6, 0.01*dz_typ)
        
        if add_perturbation:
            dem_pert = add_deterministic_perturbation(dem_array.astype(np.float64), epsilon_ratio=0.01)
        
            dem_path_pert = dem_path.replace(".tif", "_pert.tif")
            with rasterio.open(dem_path_pert, 'w', **profile) as dst:
                dst.write(dem_pert.astype(np.float64), 1)
            
            dem_path = dem_path_pert
        
    # read
    dem = wbe.read_raster(dem_path)
    # show(dem, colorbar_kwargs={'label': 'Elevation (m)'})
    
    # kwargs
    kwargs_ = {
        "flat_increment": flat_increment, # flat_increment,
        # "max_dist": 500
    }
    kwargs_.update(kwargs)
    kwargs = kwargs_
    
    # fill depressions
    filled_dem = wbe.breach_depressions_least_cost(dem, **kwargs)
    if fill_depressions_bool:
        filled_dem = wbe.fill_depressions(filled_dem, flat_increment=kwargs["flat_increment"])
    
    # write
    wbe.write_raster(filled_dem, output_file)
    
    # filled_dem = wbe.resolve_flats(filled_dem) # resolve flats
    # show(filled_dem, colorbar_kwargs={'label': 'Elevation (m)'})
    # show(filled_dem - dem, colorbar_kwargs={'label': 'fill (m)'})
    
    # burn streams into DEM
    if burn_streams_path is not None:
        streams = wbe.read_vector(burn_streams_path)
        burned_dem = wbe.fill_burn(filled_dem, streams)  # , decrement_value=10.0, gradient_distance=8
        burned_dem_path = dem_path.replace(".tif", "_burned.tif")
        wbe.write_raster(burned_dem, burned_dem_path)
        
        return burned_dem
    
    return filled_dem
