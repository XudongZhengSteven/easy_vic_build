# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
""" 
stream_network - A Python module for extracting stream network from flow accumulation data.

This module provides convenience functions for extracting stream network from flow accumulation data.

Functions:
----------
    - `d8_streamnetwork`: Extract stream network from flow accumulation data.
    - `calculate_streamnetwork_threshold`: Calculate threshold for stream extraction using multiple methods.
    - `threshold_max_ratio`: Calculate stream threshold as a ratio of maximum flow accumulation.
    - `threshold_percentile`: Calculate stream threshold based on flow accumulation percentile.
    - `threshold_drainage_area`: Calculate threshold based on minimum drainage area.
    - `threshold_dynamic_elbow`: Calculate threshold using curvature-based elbow detection.
    - `threshold_multiscale`: Calculate threshold using multi-scale adaptive approach.

Usage:
------
    1. Call `d8_streamnetwork` function with the WhiteboxTools environment and input data.
    2. Optionally, use `calculate_streamnetwork_threshold` to calculate stream threshold.

Example:
--------
    >>> from hydroanalysis_wbw import d8_streamnetwork
    >>> streams = d8_streamnetwork(wbe, flow_acc, flow_direction, filled_dem)
    
Dependencies:
-------------
    - `whitebox_workflows`: A library that facilitates geospatial processing tasks, such as DEM filling,
      flow direction, and stream extraction.
    - `numpy`: A library for numerical computing with Python.
    - `rasterio`: A library for reading and writing raster data.
    - `..geo_func.format_conversion`: Contains functions for converting raster data to shapefiles.

"""

import rasterio
import numpy as np
from ...geo_func.format_conversion import *
from .... import logger

def d8_streamnetwork(
    wbe,
    flow_acc,
    flow_direction,
    filled_dem,
    stream_acc_threshold=100.0,
    output_file_stream_raster="stream_raster.tif",
    output_file_stream_raster_link="stream_raster_link.tif",
    output_file_stream_vector="stream_vector.shp",
    output_file_stream_vector_repaired="stream_vector_repaired.shp",
    output_file_stream_lines_vector="stream_lines_vector.shp",
    output_file_confluences_points_vector="confluences_points_vector.shp",
    output_file_outlet_points_vector="outlet_points_vector.shp",
    output_file_channel_head_points_vector="channel_head_points_vector.shp",
    kwargs_extract_streams={},
    kwargs_vector_stream_network_analysis={},
    snap_dist=0.001,
    esri_pointer=True,
):
    """Extract and analyze D8-based stream network from accumulation and DEM data.

    This function performs a complete stream network extraction workflow including:
    - Stream raster generation from flow accumulation
    - Vector conversion and topological analysis
    - Network repair and feature extraction (confluences, outlets, channel heads)

    Parameters
    ----------
    wbe : `WbEnvironment`
        WhiteboxTools workflow environment instance.
        
    flow_acc : `WbRaster`
        Input flow accumulation raster. Values represent upstream contributing area in number of cells.
        
    flow_direction : `WbRaster`
        D8 flow direction raster (path or object). Values should follow WhiteboxTools
        D8 encoding (0=East, 1=NE, 2=North, etc.).
        
    filled_dem : `WbRaster`
        Hydrologically corrected DEM (path or object). Must be depression-filled.
        
    stream_acc_threshold : float, optional
        Flow accumulation threshold for stream initiation (in number of cells).
        Default=100.0.
        
    output_file_stream_raster : str, optional
        Output path for stream raster (1=stream, 0=non-stream). Default="stream_raster.tif".
    
    output_file_stream_raster_link: str, optional
        Output path for stream raster link. Default="stream_raster_link.tif".
        
    output_file_stream_vector : str, optional
        Output path for raw stream vector before repair. Default="stream_vector.shp".
        
    output_file_stream_vector_repaired : str, optional
        Output path for topologically repaired stream vector. Default="stream_vector_repaired.shp".
        
    output_file_stream_lines_vector : str, optional
        Output path for finalized stream lines vector. Default="stream_lines_vector.shp".
        
    output_file_confluences_points_vector : str, optional
        Output path for confluence points vector. Default="confluences_points_vector.shp".
        
    output_file_outlet_points_vector : str, optional
        Output path for outlet points vector. Default="outlet_points_vector.shp".
        
    output_file_channel_head_points_vector : str, optional
        Output path for channel head points vector. Default="channel_head_points_vector.shp".
        
    kwargs_extract_streams : dict, optional
        Additional parameters for extract_streams operation. Common options:
        
        - zero_background: bool
            Zero background (default=False)

    kwargs_vector_stream_network_analysis : dict, optional
        Additional parameters for vector_stream_network_analysis. Common options:
        
        - max_ridge_cutting_height: float
            The maximum ridge-cutting height, in DEM z units (cutting_height) (default=10.0)
            
    snap_dist : float, optional
        Maximum snap distance (in meters) for repairing stream topology. 
        Recommended: 2-5x DEM resolution. Default=0.001.
    
    esri_pointer: bool, optional
        Whether to use the esri pointer, which should be same as the flow_direction (default=True).

    Returns
    -------
    tuple
        Returns a tuple containing:
        
        - stream_raster : `WbRaster`
            Extracted stream raster (binary)
            
        - stream_vector : `WbVector`
            Raw stream vector before repair
            
        - repaired_stream_vector : `WbVector`
            Topologically repaired stream vector
            
        - vector_stream_network_analysis_result: tuple
            Tuple containing (stream_lines_vector, confluences_points_vector,
            outlet_points_vector, channel_head_points_vector)

    Notes
    -----
    1. DEM must be properly hydrologically conditioned (breached/filled) prior to use.
    2. For small watersheds (<10 km²), consider reducing stream_acc_threshold to 50-80.
    3. snap_dist should be adjusted based on DEM resolution:
       - 10m DEM: 20-50m
       - 30m DEM: 60-150m

    Examples
    --------
    >>> # Basic extraction with default parameters
    >>> results = d8_streamnetwork(wbe, "flow_acc.tif", "flow_dir.tif", "dem_filled.tif")

    >>> # Customized extraction for high-resolution data
    >>> results = d8_streamnetwork(
    ...     wbe,
    ...     "hr_flow_acc.tif",
    ...     "hr_flow_dir.tif",
    ...     "hr_dem_filled.tif",
    ...     stream_acc_threshold=50.0,
    ...     snap_dist=0.001,
    ... )
    """
    
    # stream raster
    logger.info("Extracting stream_raster... ...")
    stream_raster = wbe.extract_streams(flow_acc, threshold=stream_acc_threshold, **kwargs_extract_streams)
    wbe.write_raster(stream_raster, output_file_stream_raster)
    # show(stream_raster, colorbar_kwargs={'label': 'stream raster (1, bool)'})
    
    # stream link
    logger.info("Linking stream_raster... ...")
    stream_raster_link = wbe.stream_link_class(flow_direction, stream_raster, esri_pntr=esri_pointer)
    wbe.write_raster(stream_raster_link, output_file_stream_raster_link)
    
    # stream vector
    logger.info("Converting stream_raster to stream_vector... ...")
    stream_vector = wbe.raster_streams_to_vector(stream_raster, flow_direction)
    stream_vector, tmp1, tmp2, tmp3 = wbe.vector_stream_network_analysis(
        stream_vector, filled_dem
    )
    
    wbe.write_vector(stream_vector, output_file_stream_vector)
    # show(stream_vector, colorbar_kwargs={'label': 'stream vector(1, bool)'})
    
    # repair_stream_vector_topology
    logger.info("Repairing stream_vector... ...")
    repaired_stream_vector = wbe.repair_stream_vector_topology(
        stream_vector,
        snap_dist,
    )
    
    wbe.write_vector(repaired_stream_vector, output_file_stream_vector_repaired)
    
    # vector_stream_network_analysis
    logger.info("Analyzing stream_vector network... ...")
    stream_lines_vector, confluences_points_vector, outlet_points_vector, channel_head_points_vector = wbe.vector_stream_network_analysis(
        repaired_stream_vector,
        filled_dem,
        snap_distance=snap_dist,
        **kwargs_vector_stream_network_analysis,
    )
    
    vector_stream_network_analysis_result = (stream_lines_vector, confluences_points_vector, outlet_points_vector, channel_head_points_vector)
    
    wbe.write_vector(stream_lines_vector, output_file_stream_lines_vector)
    wbe.write_vector(confluences_points_vector, output_file_confluences_points_vector)
    wbe.write_vector(outlet_points_vector, output_file_outlet_points_vector)
    wbe.write_vector(channel_head_points_vector, output_file_channel_head_points_vector)
    
    return stream_raster, stream_vector, repaired_stream_vector, vector_stream_network_analysis_result
    

def calculate_streamnetwork_threshold(
    flow_acc_path,
    dem_path=None,
    method='hybrid',
    **kwargs                        
):
    """Calculate adaptive threshold for stream extraction using multiple methods.

    This function implements six different approaches for determining optimal
    flow accumulation thresholds for stream network extraction, including both
    statistical and physically-based methods. The hybrid method combines all
    approaches for robust results.

    Parameters
    ----------
    flow_acc_path : str
        Path to flow accumulation raster file. Values should represent upstream
        contributing area in number of cells.
        
    dem_path : str, optional
        Path to digital elevation model (DEM) raster file. Required only for
        'drainage_area' and 'hybrid' methods. Should be hydrologically corrected.
        
    method : {'hybrid', 'max_ratio', 'percentile', 'drainage_area', 'dynamic_elbow', 'multi_scale'}, optional
        Threshold calculation method (default='hybrid'):
        - 'max_ratio': Uses ratio of maximum flow accumulation value
        - 'percentile': Uses specified percentile of flow accumulation values
        - 'drainage_area': Based on minimum drainage area (requires DEM)
        - 'dynamic_elbow': Automatic curvature-based threshold detection
        - 'multi_scale': Multi-scale adaptive thresholding
        - 'hybrid': Weighted combination of all methods (recommended)
        
    **kwargs : dict, optional
        Method-specific parameters:
        
        - max_ratio : float, optional
            For 'max_ratio' method: ratio of max flow accumulation to use as
            threshold (default=0.001, range 0.0001-0.1)
            
        - percentile : float, optional
            For 'percentile' method: percentile value to use (default=99.5)
            
        - drainage_area_km2 : float, optional
            For 'drainage_area' method: minimum drainage area in km²
            (default=0.01)
            
        - elbow_sensitivity : float, optional
            For 'dynamic_elbow' method: sensitivity factor (default=0.1)
            
        - scale_levels : list of float, optional
            For 'multi_scale' method: scale levels as proportions of basin area
            (default=[0.1, 0.2, 0.3, 0.4, 0.5])

    Returns
    -------
    float
        Calculated threshold value in flow accumulation units (cell count).
        Represents the minimum upstream contributing area required to initiate
        a stream channel.

    Raises
    ------
    ValueError
        If invalid method is specified or required parameters are missing
    RuntimeError
        If flow accumulation data is invalid (no positive values) or
        DEM is required but not provided

    Notes
    -----
    1. For most applications, the 'hybrid' method with default parameters
       provides the most robust results.
    2. The 'drainage_area' method requires:
       - DEM with proper projection (for cell area calculation)
       - Pre-processing with depression filling/breaching
    3. Recommended parameter ranges based on DEM resolution:
       | Resolution | drainage_area_km2 | max_ratio  |
       |------------|-------------------|-----------|
       | 1-5m       | 0.001-0.01        | 0.0001-0.005 |
       | 10-30m     | 0.01-0.1          | 0.001-0.01  |
       | >30m       | 0.1-1.0           | 0.01-0.05   |

    Examples
    --------
    >>> # Hybrid method with default parameters
    >>> threshold = calculate_streamnetwork_threshold("flow_acc.tif")

    >>> # Drainage area method for high-resolution data
    >>> threshold = calculate_streamnetwork_threshold(
    ...     "flow_acc.tif",
    ...     dem_path="dem_2m.tif",
    ...     method='drainage_area',
    ...     drainage_area_km2=0.005
    ... )

    >>> # Custom hybrid approach with adjusted weights
    >>> threshold = calculate_streamnetwork_threshold(
    ...     "flow_acc.tif",
    ...     dem_path="dem.tif",
    ...     method='hybrid',
    ...     percentile=99.7,
    ...     elbow_sensitivity=0.2
    ... )
    """
    kwargs_ = {
        'max_ratio': 0.001,
        'percentile': 99.5,
        'drainage_area_km2': 0.01,
        'elbow_sensitivity': 0.1,
        'scale_levels': [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    kwargs_.update(kwargs)
    kwargs = kwargs_
    
    with rasterio.open(flow_acc_path) as src:
        flow_acc = src.read(1)
        valid_acc = flow_acc[flow_acc > 0]
    
    if len(valid_acc) == 0:
        logger.error("No valid flow accumulation values found")
        raise RuntimeError
    
    if method == 'max_ratio':
        return threshold_max_ratio(valid_acc, kwargs['max_ratio'])
    
    elif method == 'percentile':
        return threshold_percentile(valid_acc, kwargs['percentile'])
    
    elif method == 'drainage_area':
        if not dem_path:
            logger.error("DEM path is required for drainage_area method")
            raise RuntimeError
        
        return threshold_drainage_area(dem_path, flow_acc_path, drainage_area_km2=kwargs['drainage_area_km2'], max_ratio=kwargs['max_ratio'])
    
    elif method == "dynamic_elbow":
        return threshold_dynamic_elbow(valid_acc, kwargs['elbow_sensitivity'])
    
    elif method == "multiscale":
        return threshold_multiscale(valid_acc, kwargs['scale_levels'])
    
    elif method == 'hybrid':
        thresholds = [
            threshold_max_ratio(valid_acc, kwargs['max_ratio']),
            threshold_percentile(valid_acc, kwargs['percentile']),
            threshold_dynamic_elbow(valid_acc, kwargs['elbow_sensitivity']),
            threshold_multiscale(flow_acc, kwargs['scale_levels'])
        ]
        
        if dem_path:
            thresholds.append(
                threshold_drainage_area(dem_path, flow_acc_path, drainage_area_km2=kwargs['drainage_area_km2'], max_ratio=kwargs['max_ratio'])
            )
        
        # Use weighted average instead of simple min
        weights = [0.2, 0.3, 0.25, 0.25]  # Customizable weights
        if dem_path:
            weights = [0.15, 0.25, 0.2, 0.2, 0.2]  # Adjust if drainage area is included
        
        logger.info(f"calculated thresholds\nmax_ratio: {thresholds[0]}\npercentile: {thresholds[1]}\ndynamic_elbow: {thresholds[2]}\nmultiscale: {thresholds[3]}\ndrainage_area: {thresholds[4]}")
        weighted_avg = sum(t*w for t,w in zip(thresholds, weights)) / sum(weights)
        
        return min(weighted_avg, np.max(valid_acc) * 0.1)
    
    else:
        raise ValueError("Invalid method or missing required parameters")
    
def threshold_max_ratio(
    flow_acc,
    max_ratio
):
    """Calculate stream threshold as a ratio of maximum flow accumulation.
    
    Parameters
    ----------
    flow_acc : `numpy.ndarray`
        2D array of flow accumulation values (cell counts)
        
    max_ratio : float
        Ratio of maximum flow accumulation to use as threshold (0-1)
        
    Returns
    -------
    float
        Threshold value in flow accumulation units (cell count)
        
    Examples
    --------
    >>> acc = np.array([[1, 5], [10, 100]])
    >>> threshold_max_ratio(acc, 0.05)
    5.0  # 100 * 0.05
    """
    return np.max(flow_acc) * max_ratio

def threshold_percentile(
    flow_acc,
    percentile
):
    """Calculate stream threshold based on flow accumulation percentile.
    
    Parameters
    ----------
    flow_acc : `numpy.ndarray`
        2D array of flow accumulation values (cell counts)
        
    percentile : float
        Percentile value to use (0-100)
        
    Returns
    -------
    float
        Threshold value in flow accumulation units (cell count)
        
    Notes
    -----
    Common percentile values:
    - 95%: Conservative (extracts more streams)
    - 99%: Moderate
    - 99.5%: Aggressive (extracts only major channels)
    """
    return np.percentile(flow_acc, percentile)

def threshold_drainage_area(
    dem_path,
    flow_acc_path,
    drainage_area_km2=0.1,
    max_ratio=0.1,
    min_cells=30,
):
    """Calculate threshold based on minimum drainage area.
    
    Parameters
    ----------
    dem_path : str
        Path to DEM raster file (must have projection info)
        
    flow_acc_path : str
        Path to flow accumulation raster file
        
    drainage_area_km2 : float, optional
        Minimum drainage area in square kilometers (default=0.1)
        
    max_ratio : float, optional
        Maximum ratio of max flow accumulation to use (default=0.1)
        
    min_cells : int, optional
        Minimum cell count threshold (default=30)
        
    Returns
    -------
    float
        Threshold value in flow accumulation units (cell count)
        
    Raises
    ------
    RuntimeError
        If DEM or flow accumulation files cannot be read
        
    Notes
    -----
    The final threshold is constrained by:
    1. Physical area: drainage_area_km2 / cell_area
    2. Maximum ratio: max_flow_acc * max_ratio
    3. Minimum cells: min_cells
    """
    with rasterio.open(dem_path) as dem:
        cell_area_km2 = dem.res[0] * dem.res[1] / 1e6
        threshold_cells = drainage_area_km2 / cell_area_km2
    
    with rasterio.open(flow_acc_path) as src:
        flow_acc = src.read(1)
        valid_acc = flow_acc[flow_acc > 0]
        max_acc = valid_acc.max()
    
    threshold = max(min_cells, min(threshold_cells, max_acc * max_ratio))
    
    return threshold


def threshold_dynamic_elbow(
    flow_acc,
    elbow_sensitivity=0.3
):
    """Calculate threshold using curvature-based elbow detection.
    
    Parameters
    ----------
    flow_acc : `numpy.ndarray`
        2D array of flow accumulation values (cell counts)
        
    elbow_sensitivity : float, optional
        Sensitivity factor (0-1) where lower values produce more conservative
        thresholds (default=0.3)
        
    Returns
    -------
    float
        Threshold value in flow accumulation units (cell count)
        
    Notes
    -----
    1. Applies log-transform to enhance curvature features
    2. Finds maximum second derivative point (elbow) in distribution
    3. Applies sensitivity scaling to the detected elbow value
    """
    flow_acc = np.sort(flow_acc[flow_acc > 0])
    if len(flow_acc) == 0:
        return 0
    
    x = np.arange(len(flow_acc))
    y = np.log(flow_acc + 1)
    
    dy = np.gradient(y, x)
    d2y = np.gradient(dy, x)
    
    elbow_idx = np.argmax(d2y)
    threshold = flow_acc[elbow_idx]
    
    return threshold * elbow_sensitivity

def threshold_multiscale(
    flow_acc,
    scale_levels
):
    """Calculate threshold using multi-scale adaptive approach.
    
    Parameters
    ----------
    flow_acc : `numpy.ndarray`
        2D array of flow accumulation values (cell counts)
        
    scale_levels : list of float
        Scale levels as proportions (0-1) of basin area to analyze
        
    Returns
    -------
    float
        Mean threshold across all scales (95th percentile at each scale)
        
    Examples
    --------
    >>> acc = np.random.randint(0, 1000, (100,100))
    >>> threshold_multiscale(acc, [0.1, 0.3, 0.5])
    245.3  # Example output
    """
    thresholds  = []
    valid_acc = flow_acc[flow_acc > 0]
    
    for scale in scale_levels:
        mask = flow_acc >= np.percentile(valid_acc, 100*(1-scale))
        sub_acc = flow_acc[mask]
        
        if len(sub_acc) > 0:
            thresholds.append(np.percentile(sub_acc, 95))
    
    return np.mean(thresholds) if thresholds else np.percentile(valid_acc, 95)

def clip_stream_for_basin(
    wbe,
    stream_vector,
    basin_vector,
    output_file_clipped_stream_vector="clipped_stream_vector.shp",
):
    if isinstance(stream_vector, str):
        stream_vector = wbe.read_vector(stream_vector)
        
    if isinstance(basin_vector, str):
        basin_vector = wbe.read_vector(basin_vector)
    
    clipped_stream_vector = wbe.clip(
        stream_vector,
        basin_vector
    )
    
    wbe.write_vector(clipped_stream_vector, output_file_clipped_stream_vector)
    
    return clipped_stream_vector
    