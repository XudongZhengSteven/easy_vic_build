# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com


import os

from ...geo_func.format_conversion import *
from ...utilities import *
from ..create_dem import create_dem_from_params
from .... import logger
from . import (set_workenv, fill_dem, flow_direction, flow_accumulation, stream_network, outlet_detection, basin_delineation)


def hydroanalysis_for_level0(
    working_directory,
    dem_level0_path,
    pourpoint_x_index=None,
    pourpoint_y_index=None,
    pourpoint_direction_code=None,
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
    Perform complete hydrological analysis at level 0 (original DEM resolution).

    This function executes a complete hydrological workflow including DEM filling,
    flow direction/accumulation calculation, stream network extraction, outlet
    detection, and watershed delineation.

    Parameters
    ----------
    working_directory : str
        Path to working directory where outputs will be saved.
        
    dem_level0_path : str
        Path to input DEM file for level 0 analysis.
        
    pourpoint_x_index : int, optional
        Column index for pour point adjustment.
        
    pourpoint_y_index : int, optional
        Row index for pour point adjustment.
        
    pourpoint_direction_code : int, optional
        Flow direction code for pour point enforcement.
        
    stream_acc_threshold : float, optional
        Threshold value for stream extraction. If None, will be calculated.
        
    filldem_kwargs : dict, optional
        Additional arguments for DEM filling (default {}).
        
    d8_flowdirection_kwargs : dict, optional
        Additional arguments for flow direction calculation (default {}).
        
    d8_flowaccumulation_kwargs : dict, optional
        Additional arguments for flow accumulation (default {}).
        
    calculate_streamnetwork_threshold_kwargs : dict, optional
        Arguments for stream threshold calculation (default {}).
        
    d8_streamnetwork_kwargs : dict, optional
        Arguments for stream network extraction (default {}).
        
    snap_outlet_to_stream_kwargs : dict, optional
        Arguments for outlet snapping (default {}).
        
    delineate_basins_for_snaped_outlets_kwargs : dict, optional
        Arguments for basin delineation (default {}).
        
    crs_str : str, optional
        Coordinate reference system (default "EPSG:4326").
        
    esri_pointer : bool, optional
        Whether to use ESRI flow direction convention (default True).
        
    outlets_with_reference_coords : tuple, optional
        Tuple of (x_coords, y_coords) for reference outlets.

    Returns
    -------
    bool
        True if analysis completes successfully.

    Notes
    -----
    Processing Steps:
    1. DEM filling
    2. Flow direction calculation
    3. Flow accumulation calculation
    4. Stream network extraction
    5. Outlet detection and snapping
    6. Watershed delineation
    7. Stream network clipping

    Output Files:
    - filled_dem.tif
    - flow_direction.tif
    - flow_acc.tif
    - stream network rasters/vectors
    - outlet shapefiles
    - basin delineation rasters/vectors
    - clipped stream networks
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
        pourpoint_x_index=pourpoint_x_index,
        pourpoint_y_index=pourpoint_y_index,
        pourpoint_direction_code=pourpoint_direction_code,
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
        "stream_vector.shp",
        "basin_vector_main_outlet.shp",
        output_file_clipped_stream_vector="clipped_stream_vector_basin_vector_main_outlet.shp"
    )
    
    if outlets_with_reference_coords is not None:
        logger.info("Clipping streams within basins for outlets with reference... ...")
        dst_clipped_stream_vector_basins_vector_outlets_with_reference = stream_network.clip_stream_for_basin(
            wbe,
            "stream_vector.shp",
            "basins_vector_outlets_with_reference.shp",
            output_file_clipped_stream_vector="clipped_stream_vector_basins_vector_outlets_with_reference.shp"
        )
        
        if len(outlets_with_reference_coords[0]) > 1:
            for i in range(len(outlets_with_reference_coords[0])):
                stream_network.clip_stream_for_basin(
                    wbe,
                    "stream_vector.shp",
                    f"basin_vector_outlet_with_reference_{i}.shp",
                    output_file_clipped_stream_vector=f"clipped_stream_vector_basin_vector_outlet_with_reference_{i}.shp"
                )
                
    return True
    
    
def hydroanalysis_for_level1(
    working_directory,
    dem_level1_path,
    pourpoint_x_index=None,
    pourpoint_y_index=None,
    pourpoint_direction_code=None,
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
):
    """
    Performs hydrological analysis at Level 1 (modeling scale) to generate files for runoff routing modeling.
    These outputs are passed to build_hydroanalysis and subsequently to build_RVIC_Params. 
    The input DEM at level 1 can be generated by `create_dem.create_dem_from_params` function.

    Key Target Files:
    - flow_direction.tif : D8 flow direction raster, used to derive flow_distance file.
    - flow_acc.tif : Flow accumulation raster, used to derive flow_distance file.
    - stream_raster.tif : Extracted stream network binary raster.
    - snaped_outlet_vector_with_reference.shp : Verified outlet points snapped to stream channels.

    This function executes hydrological workflow including DEM filling,
    flow direction/accumulation calculation, stream network extraction, and outlet
    detection.

    Parameters
    ----------
    working_directory : str
        Path to working directory where outputs will be saved.
        
    dem_level1_path : str
        Path to input DEM file for level 1 analysis.
        
    pourpoint_x_index : int, optional
        Column index for pour point adjustment.
        
    pourpoint_y_index : int, optional
        Row index for pour point adjustment.
        
    pourpoint_direction_code : int, optional
        Flow direction code for pour point enforcement.
        
    stream_acc_threshold : float, optional
        Threshold value for stream extraction. If None, will be calculated.
        
    filldem_kwargs : dict, optional
        Additional arguments for DEM filling (default {}).
        
    d8_flowdirection_kwargs : dict, optional
        Additional arguments for flow direction calculation (default {}).
        
    d8_flowaccumulation_kwargs : dict, optional
        Additional arguments for flow accumulation (default {}).
        
    calculate_streamnetwork_threshold_kwargs : dict, optional
        Arguments for stream threshold calculation (default {}).
        
    d8_streamnetwork_kwargs : dict, optional
        Arguments for stream network extraction (default {}).
        
    snap_outlet_to_stream_kwargs : dict, optional
        Arguments for outlet snapping (default {}).
        
    crs_str : str, optional
        Coordinate reference system (default "EPSG:4326").
        
    esri_pointer : bool, optional
        Whether to use ESRI flow direction convention (default True).
        
    outlets_with_reference_coords : tuple, optional
        Tuple of (x_coords, y_coords) for reference outlets.

    Returns
    -------
    bool
        True if analysis completes successfully.

    Notes
    -----
    Processing Steps:
    1. DEM filling
    2. Flow direction calculation
    3. Flow accumulation calculation
    4. Stream network extraction
    5. Outlet detection and snapping

    Output Files:
    - filled_dem.tif
    - flow_direction.tif
    - flow_acc.tif
    - stream network rasters/vectors
    - outlet shapefiles
    """
    logger.info("Starting to performing hydroanalysis at level 1... ...")
    
    # env set
    logger.info("Setting up WhiteboxTools environment... ...")
    wbe = set_workenv.setWorkenv(working_directory)

    # fill dem
    logger.info("Filling DEM... ...")
    dst_filled_dem = fill_dem.filldem(
        wbe,
        dem_path=dem_level1_path,
        output_file="filled_dem.tif",
        **filldem_kwargs
    )
    
    # flow direction
    logger.info("Calculating flow direction... ...")
    d8_flowdirection_kwargs["esri_pointer"] = esri_pointer
    dst_flow_direction = flow_direction.d8_flowdirection(
        wbe, dst_filled_dem, output_file="flow_direction.tif",
        pourpoint_x_index=pourpoint_x_index,
        pourpoint_y_index=pourpoint_y_index,
        pourpoint_direction_code=pourpoint_direction_code,
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
    logger.info(f"Detecting outlets with reference: {list(zip(outlets_with_reference_coords, outlets_with_reference_coords))}... ...")
    if outlets_with_reference_coords is not None:
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
    
    return True


# def hydroanalysis_for_basinMap(
#     evb_dir,
#     crs_str="EPSG:4326",
#     reverse_lat=True,
# ):
#     """
#     Perform hydrological analysis for plotting basin map. This includes reading DEM data,
#     conducting hydrological analysis using the `hydroanalysis_wbw` function, and clipping
#     stream geometries based on the basin's boundary.

#     Parameters:
#     -----------
#     evb_dir : object
#         The directory object containing paths to the necessary input and output directories,
#         including `BasinMap_dir`, which holds DEM and stream data.

#     Returns:
#     --------
#     None
#         This function does not return any value but generates and saves the necessary hydrological
#         analysis outputs to files in the `BasinMap_dir`.
#     """
    # # read dpc
    # dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2 = readdpc(evb_dir)

    # # read params
    # params_dataset_level0, params_dataset_level1 = readParam(evb_dir, mode="r")

    # # hydroanalysis for level0
    # transform = create_dem_from_params(
    #     params_dataset_level0,
    #     os.path.join(evb_dir.BasinMap_dir, "dem_level0.tif"),
    #     crs_str=crs_str,
    #     reverse_lat=reverse_lat,
    # )
        
    # hydroanalysis_for_level1(
    #     evb_dir.BasinMap_dir,
    #     os.path.join(evb_dir.BasinMap_dir, "dem_level0.tif"),
    # )

    # # clip stream gdf within basin shp
    # stream_gdf = gpd.read_file(
    #     os.path.join(evb_dir.BasinMap_dir, "stream_raster_shp.shp")
    # )
    
    # stream_gdf_clip = gpd.overlay(
    #     stream_gdf,
    #     dpc_VIC_level0.basin_shp.loc[:, "geometry":"geometry"],
    #     how="intersection",
    # )
    
    # stream_gdf_clip.to_file(
    #     os.path.join(evb_dir.BasinMap_dir, "stream_raster_shp_clip.shp")
    # )

    # # close params
    # params_dataset_level0.close()
    # params_dataset_level1.close()
