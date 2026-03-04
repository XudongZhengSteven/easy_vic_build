# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com


"""Module ``easy_vic_build.tools.hydroanalysis_func.hydroanalysis_wbw.hydroanalysis``."""

import os

from ...geo_func.format_conversion import *
from ...utilities import *
from .... import logger
from . import (set_workenv, fill_dem, flow_direction, flow_accumulation, stream_network, outlet_detection, basin_delineation)


def hydroanalysis_for_level0(
    working_directory,
    dem_level0_path,
    stream_acc_threshold=None,
    filldem_kwargs={},
    d8_flowdirection_kwargs={},
    d8_flowaccumulation_kwargs={},
    calculate_streamnetwork_threshold_kwargs={},
    d8_streamnetwork_kwargs={},
    snap_outlet_to_stream_kwargs={},
    delineate_basins_for_snaped_outlets_kwargs={},
    crs_str="EPSG:4326",
    esri_pointer=True,
    outlets_with_reference_coords=None,
):
    """
    Run full hydrological analysis at level-0 DEM resolution.

    The workflow includes DEM filling, flow direction/accumulation, stream
    extraction, outlet detection/snapping, basin delineation, and stream clipping.

    Parameters
    ----------
    working_directory : str
        Path to working directory where outputs will be saved.
    dem_level0_path : str
        Path to input DEM file for level 0 analysis.
    stream_acc_threshold : float, optional
        Stream extraction threshold. If ``None``, it is estimated from inputs.
    filldem_kwargs : dict, optional
        Extra keyword arguments forwarded to ``fill_dem.filldem``.
    d8_flowdirection_kwargs : dict, optional
        Extra keyword arguments for D8 flow direction.
    d8_flowaccumulation_kwargs : dict, optional
        Extra keyword arguments for D8 flow accumulation.
    calculate_streamnetwork_threshold_kwargs : dict, optional
        Extra keyword arguments for threshold estimation.
    d8_streamnetwork_kwargs : dict, optional
        Extra keyword arguments for stream extraction.
    snap_outlet_to_stream_kwargs : dict, optional
        Extra keyword arguments for outlet snapping.
    delineate_basins_for_snaped_outlets_kwargs : dict, optional
        Extra keyword arguments for basin delineation.
    crs_str : str, optional
        CRS used when creating outlet vectors.
    esri_pointer : bool, optional
        Whether D8 direction encoding follows ESRI convention.
    outlets_with_reference_coords : tuple, optional
        Reference outlet coordinates as ``(x_coords, y_coords)``.
        Each element should be an iterable of equal length.

    Returns
    -------
    bool
        True if analysis completes successfully.

    Notes
    -----
    Typical outputs include ``filled_dem.tif``, ``flow_direction.tif``,
    ``flow_acc.tif``, stream/outlet shapefiles, and basin polygons.
    """
    logger.info("Starting to performing hydroanalysis at level 0... ...")
    
    # env set
    logger.info("Setting up WhiteboxTools environment... ...")
    wbe = set_workenv.setWorkenv(working_directory)
    
    # fill dem
    logger.info("Filling DEM... ...")
    dst_filled_dem = fill_dem.filldem(
        wbe,
        dem_path=dem_level0_path,
        output_file="filled_dem.tif",
        **filldem_kwargs
    )
    
    # flow direction
    logger.info("Calculating flow direction... ...")
    d8_flowdirection_kwargs["esri_pointer"] = esri_pointer
    dst_flow_direction = flow_direction.d8_flowdirection(
        wbe, dst_filled_dem, output_file="flow_direction.tif",
        **d8_flowdirection_kwargs        
    )
    
    # flow accumulation
    logger.info("Calculating flow accumulation... ...")
    dst_flow_acc = flow_accumulation.d8_flowaccumulation(
        wbe,
        dst_flow_direction,
        output_file="flow_acc.tif",
        **d8_flowaccumulation_kwargs
    )
    
    # stream network
    logger.info("Calculating stream accumulation threshold... ...")
    stream_acc_threshold = stream_acc_threshold if stream_acc_threshold is not None else stream_network.calculate_streamnetwork_threshold(
        os.path.join(working_directory, "flow_acc.tif"),
        os.path.join(working_directory, "filled_dem.tif"),
        esri_pointer=esri_pointer,
        **calculate_streamnetwork_threshold_kwargs
    )
    
    logger.info(f"Stream accumulation threshold: {stream_acc_threshold}")
    
    logger.info("Extracting stream network... ...")
    dst_stream_raster, dst_stream_vector, dst_repaired_stream_vector, dst_vector_stream_network_analysis_result = stream_network.d8_streamnetwork(
        wbe,
        dst_flow_acc,
        dst_flow_direction,
        dst_filled_dem,
        stream_acc_threshold=stream_acc_threshold,
        **d8_streamnetwork_kwargs
    )

    # main outlet
    logger.info("Detecting main outlet... ...")
    dst_main_outlet_coord, dst_main_outlet_col_row_index = outlet_detection.detect_main_outlet(
        os.path.join(working_directory, "flow_acc.tif"),
        os.path.join(working_directory, "main_outlet.shp"),
        crs_str=crs_str,
    )
    
    dst_snaped_main_outlet_vector = outlet_detection.snap_outlet_to_stream(
        wbe,
        os.path.join(working_directory, "main_outlet.shp"),
        dst_stream_raster,
        os.path.join(working_directory, "snaped_main_outlet.shp"),
        **snap_outlet_to_stream_kwargs
    )
    
    # outlets with reference
    logger.info(f"Detecting outlets with reference: {list(zip(outlets_with_reference_coords, outlets_with_reference_coords))}... ...")
    if outlets_with_reference_coords is not None:
        dst_outlets_gdf_with_reference, dst_snaped_outlets_vector_with_reference = outlet_detection.detect_outlets_with_reference(
            wbe,
            x_coords=outlets_with_reference_coords[0],
            y_coords=outlets_with_reference_coords[1],
            stream_raster=dst_stream_raster,
            crs_str=crs_str,
            output_file_path=os.path.join(working_directory, "outlets_with_reference.shp"),
            snaped_output_file_path=os.path.join(working_directory, "snaped_outlets_with_reference.shp"),
            **snap_outlet_to_stream_kwargs
        )
        
        # detect_outlets_with_reference for each outlet
        if len(outlets_with_reference_coords[0]) > 1:
            for i in range(len(outlets_with_reference_coords[0])):
                outlet_detection.detect_outlets_with_reference(
                    wbe,
                    x_coords=[outlets_with_reference_coords[0][i]],
                    y_coords=[outlets_with_reference_coords[1][i]],
                    stream_raster=dst_stream_raster,
                    crs_str=crs_str,
                    output_file_path=os.path.join(working_directory, f"outlet_with_reference_{i}.shp"),
                    snaped_output_file_path=os.path.join(working_directory, f"snaped_outlet_with_reference_{i}.shp"),
                    **snap_outlet_to_stream_kwargs
                )
            
    # basin delineation
    logger.info("Delineating basins for snaped main outlet... ...")
    dst_basin_raster_main_outlet, dst_basin_vector_main_outlet = basin_delineation.delineate_basins_for_snaped_outlets(
        wbe,
        dst_flow_direction,
        dst_snaped_main_outlet_vector,
        output_file_basins_raster="basin_raster_main_outlet.tif",
        output_file_basins_vector="basin_vector_main_outlet.shp",
        esri_pointer=esri_pointer,
        **delineate_basins_for_snaped_outlets_kwargs,
    )
    
    if outlets_with_reference_coords is not None:
        logger.info("Delineating basins for snaped outlets with reference coords... ...")
        dst_basins_raster_outlets_with_reference, dst_basins_vector_outlets_with_reference = basin_delineation.delineate_basins_for_snaped_outlets(
            wbe,
            dst_flow_direction,
            dst_snaped_outlets_vector_with_reference,
            output_file_basins_raster="basins_raster_outlets_with_reference.tif",
            output_file_basins_vector="basins_vector_outlets_with_reference.shp",
            esri_pointer=esri_pointer,
            **delineate_basins_for_snaped_outlets_kwargs,
        )
        
        # delineate_basins_for_snaped_outlets for each outlet
        if len(outlets_with_reference_coords[0]) > 1:
            for i in range(len(outlets_with_reference_coords[0])):
                basin_delineation.delineate_basins_for_snaped_outlets(
                    wbe,
                    dst_flow_direction,
                    f"snaped_outlet_with_reference_{i}.shp",
                    output_file_basins_raster=f"basin_raster_outlet_with_reference_{i}.tif",
                    output_file_basins_vector=f"basin_vector_outlet_with_reference_{i}.shp",
                    esri_pointer=esri_pointer,
                    **delineate_basins_for_snaped_outlets_kwargs,
            )
    
    # clip streams within basins
    logger.info("Clipping streams within basins for main outlet... ...")
    
    dst_clipped_stream_vector_basin_vector_main_outlet = stream_network.clip_stream_for_basin(
        wbe,
        "stream_raster_vector.shp", # "stream_vector.shp",  stream_raster_vector
        "basin_vector_main_outlet.shp",
        output_file_clipped_stream_vector="clipped_stream_vector_basin_vector_main_outlet.shp"
    )
    
    if outlets_with_reference_coords is not None:
        logger.info("Clipping streams within basins for outlets with reference... ...")
        dst_clipped_stream_vector_basins_vector_outlets_with_reference = stream_network.clip_stream_for_basin(
            wbe,
            "stream_raster_vector.shp",
            "basins_vector_outlets_with_reference.shp",
            output_file_clipped_stream_vector="clipped_stream_vector_basins_vector_outlets_with_reference.shp"
        )
        
        if len(outlets_with_reference_coords[0]) > 1:
            for i in range(len(outlets_with_reference_coords[0])):
                stream_network.clip_stream_for_basin(
                    wbe,
                    "stream_raster_vector.shp",
                    f"basin_vector_outlet_with_reference_{i}.shp",
                    output_file_clipped_stream_vector=f"clipped_stream_vector_basin_vector_outlet_with_reference_{i}.shp"
                )
                
    return True
    
    
def hydroanalysis_for_level1(
    working_directory,
    dem_level1_path,
    stream_acc_threshold=None,
    filldem_kwargs={},
    d8_flowdirection_kwargs={},
    d8_flowaccumulation_kwargs={},
    calculate_streamnetwork_threshold_kwargs={},
    d8_streamnetwork_kwargs={},
    snap_outlet_to_stream_kwargs={},
    crs_str="EPSG:4326",
    esri_pointer=True,
    outlets_with_reference_coords=None,
    fill_dem_bool=True,
):
    """
    Run hydrological analysis at level-1 (routing/model grid scale).

    Outputs are used by downstream routing preparation workflows (for example,
    RVIC parameter construction).

    Parameters
    ----------
    working_directory : str
        Path to working directory where outputs will be saved.
    dem_level1_path : str
        Path to input DEM file for level 1 analysis.
    stream_acc_threshold : float, optional
        Stream extraction threshold. If ``None``, it is estimated from inputs.
    filldem_kwargs : dict, optional
        Extra keyword arguments forwarded to ``fill_dem.filldem``.
    d8_flowdirection_kwargs : dict, optional
        Extra keyword arguments for D8 flow direction.
    d8_flowaccumulation_kwargs : dict, optional
        Extra keyword arguments for D8 flow accumulation.
    calculate_streamnetwork_threshold_kwargs : dict, optional
        Extra keyword arguments for threshold estimation.
    d8_streamnetwork_kwargs : dict, optional
        Extra keyword arguments for stream extraction.
    snap_outlet_to_stream_kwargs : dict, optional
        Extra keyword arguments for outlet snapping.
    crs_str : str, optional
        CRS used when creating outlet vectors.
    esri_pointer : bool, optional
        Whether D8 direction encoding follows ESRI convention.
    outlets_with_reference_coords : tuple, optional
        Reference outlet coordinates as ``(x_coords, y_coords)``.
        Each element should be an iterable of equal length.
    fill_dem_bool : bool, optional
        If ``True``, run DEM filling first; otherwise use ``dem_level1_path``
        directly as analysis DEM.

    Returns
    -------
    bool
        True if analysis completes successfully.

    Notes
    -----
    Main outputs include ``flow_direction.tif``, ``flow_acc.tif``,
    ``stream_raster.tif``, and snapped outlet shapefiles.
    """
    logger.info("Starting to performing hydroanalysis at level 1... ...")
    
    # env set
    logger.info("Setting up WhiteboxTools environment... ...")
    wbe = set_workenv.setWorkenv(working_directory)

    # fill dem
    if fill_dem_bool:
        logger.info("Filling DEM... ...")
        dst_filled_dem = fill_dem.filldem(
            wbe,
            dem_path=dem_level1_path,
            output_file="filled_dem.tif",
            **filldem_kwargs
        )
    else:
        dst_filled_dem = wbe.read_raster(dem_level1_path)
    
    # flow direction
    logger.info("Calculating flow direction... ...")
    d8_flowdirection_kwargs["esri_pointer"] = esri_pointer
    dst_flow_direction = flow_direction.d8_flowdirection(
        wbe, dst_filled_dem, output_file="flow_direction.tif",
        **d8_flowdirection_kwargs
    )
    
    # flow accumulation
    logger.info("Calculating flow accumulation... ...")
    dst_flow_acc = flow_accumulation.d8_flowaccumulation(
        wbe,
        dst_flow_direction,
        output_file="flow_acc.tif",
        **d8_flowaccumulation_kwargs
    )
    
    # stream network
    logger.info("Calculating stream accumulation threshold... ...")
    stream_acc_threshold = stream_acc_threshold if stream_acc_threshold is not None else stream_network.calculate_streamnetwork_threshold(
        os.path.join(working_directory, "flow_acc.tif"),
        os.path.join(working_directory, "filled_dem.tif"),
        esri_pointer=esri_pointer,
        **calculate_streamnetwork_threshold_kwargs
    )
    
    logger.info(f"Stream accumulation threshold: {stream_acc_threshold}")
    
    logger.info("Extracting stream network... ...")
    dst_stream_raster, dst_stream_vector, dst_repaired_stream_vector, dst_vector_stream_network_analysis_result = stream_network.d8_streamnetwork(
        wbe,
        dst_flow_acc,
        dst_flow_direction,
        dst_filled_dem,
        stream_acc_threshold=stream_acc_threshold,
        **d8_streamnetwork_kwargs
    )
    
    # outlets with reference
    if outlets_with_reference_coords is not None:
        logger.info(f"Detecting outlets with reference: {list(zip(outlets_with_reference_coords, outlets_with_reference_coords))}... ...")
        dst_outlet_gdf_with_reference, dst_snaped_outlet_vector_with_reference = outlet_detection.detect_outlets_with_reference(
            wbe,
            x_coords=outlets_with_reference_coords[0],
            y_coords=outlets_with_reference_coords[1],
            stream_raster=dst_stream_raster,
            crs_str=crs_str,
            output_file_path=os.path.join(working_directory, "outlets_with_reference.shp"),
            snaped_output_file_path=os.path.join(working_directory, "snaped_outlets_with_reference.shp"),
            **snap_outlet_to_stream_kwargs
        )
        
        # detect_outlets_with_reference for each outlet
        if len(outlets_with_reference_coords[0]) > 1:
            for i in range(len(outlets_with_reference_coords[0])):
                outlet_detection.detect_outlets_with_reference(
                    wbe,
                    x_coords=[outlets_with_reference_coords[0][i]],
                    y_coords=[outlets_with_reference_coords[1][i]],
                    stream_raster=dst_stream_raster,
                    crs_str=crs_str,
                    output_file_path=os.path.join(working_directory, f"outlet_with_reference_{i}.shp"),
                    snaped_output_file_path=os.path.join(working_directory, f"snaped_outlet_with_reference_{i}.shp"),
                    **snap_outlet_to_stream_kwargs
                )
                
    return True
