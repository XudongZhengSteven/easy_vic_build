# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
"""Stream-network extraction and threshold utilities."""

import rasterio
import numpy as np
from ...geo_func.format_conversion import *
from .... import logger
from whitebox_workflows import show

def d8_streamnetwork(
    wbe,
    flow_acc,
    flow_direction,
    filled_dem,
    stream_acc_threshold=100.0,
    output_file_stream_raster="stream_raster.tif",
    output_file_stream_raster_vector="stream_raster_vector.shp",
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
    """Extract and analyze a D8-based stream network.

    Parameters
    ----------
    wbe : `WbEnvironment`
        WhiteboxTools workflow environment instance.
    flow_acc : `WbRaster`
        Flow-accumulation raster or raster object.
    flow_direction : `WbRaster`
        D8 flow-direction raster or raster object.
    filled_dem : `WbRaster`
        Hydrologically conditioned DEM raster or raster object.
    stream_acc_threshold : float, optional
        Flow-accumulation threshold for stream initiation.
    output_file_stream_raster : str, optional
        Output path for stream raster.
    output_file_stream_raster_vector : str, optional
        Output path for vectorized stream raster.
    output_file_stream_raster_link : str, optional
        Output path for stream-link raster.
    output_file_stream_vector : str, optional
        Output path for stream vector before topology repair.
    output_file_stream_vector_repaired : str, optional
        Output path for repaired stream vector.
    output_file_stream_lines_vector : str, optional
        Output path for stream-line vector.
    output_file_confluences_points_vector : str, optional
        Output path for confluence points vector.
    output_file_outlet_points_vector : str, optional
        Output path for outlet points vector.
    output_file_channel_head_points_vector : str, optional
        Output path for channel-head points vector.
    kwargs_extract_streams : dict, optional
        Additional keyword arguments passed to ``wbe.extract_streams``.
    kwargs_vector_stream_network_analysis : dict, optional
        Additional keyword arguments passed to
        ``wbe.vector_stream_network_analysis``.
    snap_dist : float, optional
        Snap distance used in topology repair and vector network analysis.
    esri_pointer : bool, optional
        Whether ``flow_direction`` uses ESRI D8 pointer encoding.

    Returns
    -------
    tuple
        ``(stream_raster, stream_vector, repaired_stream_vector,
        vector_stream_network_analysis_result)``.

    Notes
    -----
    ``filled_dem`` should be hydrologically conditioned before use.
    """
    
    # stream raster
    logger.info("Extracting stream_raster... ...")
    stream_raster = wbe.extract_streams(flow_acc, threshold=stream_acc_threshold, **kwargs_extract_streams)
    wbe.write_raster(stream_raster, output_file_stream_raster)
    # show(stream_raster, colorbar_kwargs={'label': 'stream raster (1, bool)'})
    
    # stream raster vector
    stream_raster_vector = wbe.raster_to_vector_lines(stream_raster)
    wbe.write_vector(stream_raster_vector, output_file_stream_raster_vector)
    # show(stream_raster_vector, colorbar_kwargs={'label': 'stream raster vector(1, bool)'})
    
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
    
    if len(confluences_points_vector.records) > 0:
        wbe.write_vector(confluences_points_vector, output_file_confluences_points_vector)
    else:
        logger.warning("Confluences points vector could not be written. It may be empty.")
    
    if len(outlet_points_vector.records) > 0:
        wbe.write_vector(outlet_points_vector, output_file_outlet_points_vector)
    else:
        logger.warning("Outlet points vector could not be written. It may be empty.")
    
    if len(channel_head_points_vector.records) > 0:
        wbe.write_vector(channel_head_points_vector, output_file_channel_head_points_vector)
    else:
        logger.warning("Channel head points vector could not be written. It may be empty.")
    
    return stream_raster, stream_vector, repaired_stream_vector, vector_stream_network_analysis_result
    

def calculate_streamnetwork_threshold(
    flow_acc_path,
    dem_path=None,
    method='hybrid',
    **kwargs                        
):
    """Estimate a stream-extraction threshold from flow accumulation.

    Parameters
    ----------
    flow_acc_path : str
        Path to the flow-accumulation raster file.
    dem_path : str, optional
        Path to DEM raster. Required when ``method='drainage_area'``.
    method : {'hybrid', 'max_ratio', 'percentile', 'drainage_area', 'dynamic_elbow', 'multiscale'}, optional
        Threshold estimation method.
    **kwargs : dict, optional
        Optional method parameters: ``max_ratio``, ``percentile``,
        ``drainage_area_km2``, ``elbow_sensitivity``, and ``scale_levels``.

    Returns
    -------
    float
        Stream-extraction threshold in flow-accumulation units.

    Raises
    ------
    ValueError
        If ``method`` is not supported.
    RuntimeError
        If no valid flow-accumulation values are found, or DEM is missing
        for ``method='drainage_area'``.
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
    """Calculate threshold as a fraction of maximum flow accumulation.

    Parameters
    ----------
    flow_acc : `numpy.ndarray`
        Flow-accumulation values.
    max_ratio : float
        Ratio applied to the maximum value.

    Returns
    -------
    float
        Threshold value.
    """
    return np.max(flow_acc) * max_ratio

def threshold_percentile(
    flow_acc,
    percentile
):
    """Calculate threshold from a percentile of flow accumulation.

    Parameters
    ----------
    flow_acc : `numpy.ndarray`
        Flow-accumulation values.
    percentile : float
        Percentile value in ``[0, 100]``.

    Returns
    -------
    float
        Threshold value.
    """
    return np.percentile(flow_acc, percentile)

def threshold_drainage_area(
    dem_path,
    flow_acc_path,
    drainage_area_km2=0.1,
    max_ratio=0.1,
    min_cells=30,
):
    """Calculate threshold from a minimum drainage-area constraint.

    Parameters
    ----------
    dem_path : str
        Path to DEM raster file.
    flow_acc_path : str
        Path to flow-accumulation raster file.
    drainage_area_km2 : float, optional
        Minimum drainage area in square kilometers.
    max_ratio : float, optional
        Upper-limit ratio relative to maximum flow accumulation.
    min_cells : int, optional
        Lower bound of threshold in cell count.

    Returns
    -------
    float
        Threshold value.
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
        Flow-accumulation values.
    elbow_sensitivity : float, optional
        Sensitivity factor applied to the detected elbow value.

    Returns
    -------
    float
        Threshold value.
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
    """Calculate threshold using a multi-scale adaptive approach.

    Parameters
    ----------
    flow_acc : `numpy.ndarray`
        Flow-accumulation values.
    scale_levels : list of float
        Proportions used to build sub-scale masks.

    Returns
    -------
    float
        Mean threshold across scales.
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
    """Clip stream vectors by a basin polygon and save the clipped output.

    Parameters
    ----------
    wbe : WbEnvironment
        WhiteboxTools workflow environment.
    stream_vector : str or WbVector
        Stream layer path or an already loaded stream vector.
    basin_vector : str or WbVector
        Basin polygon path or an already loaded basin vector.
    output_file_clipped_stream_vector : str, optional
        Output path for clipped stream vectors.

    Returns
    -------
    WbVector
        Clipped stream vector object.
    """
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
    







