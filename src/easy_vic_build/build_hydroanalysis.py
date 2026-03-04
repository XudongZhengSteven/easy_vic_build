# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Hydroanalysis utilities for level-0 and level-1 VIC workflows.

Public functions
----------------
``buildHydroanalysis_level0``
    Run level-0 hydroanalysis from an existing DEM file.
``buildHydroanalysis_level1``
    Create level-1 DEM, flow direction/accumulation rasters, and flow distance.
``buildRivernetwork_level1``
    Build river-network graphs from level-1 hydroanalysis outputs.

Notes
-----
The current implementation only supports ``flow_direction_pkg="wbw"``.
"""

import os
import shutil

import rasterio
from netCDF4 import Dataset

from . import logger
from .tools.geo_func.search_grids import *
from .tools.hydroanalysis_func import (create_dem, create_flow_distance)
from .tools.utilities import remove_and_mkdir
from .tools.routing_func.river_network import create_river_network_graph, find_river_paths, sort_river_paths_by_lengths, extract_connected_river_network
from .tools.plot_func.plot_map import plot_river_network


def buildHydroanalysis_level0(
    evb_dir,
    dem_level0_path,
    flow_direction_pkg="wbw",
    **kwargs,
):
    """Run level-0 hydroanalysis from a prepared DEM raster.

    Parameters
    ----------
    evb_dir : Evb_dir
        Case directory manager. Results are written under
        ``evb_dir.Hydroanalysis_dir``.
    dem_level0_path : str
        Path to the level-0 DEM raster.
    flow_direction_pkg : str, optional
        Flow-direction backend. Only ``"wbw"`` is supported.
    **kwargs
        Extra keyword arguments forwarded to
        ``hydroanalysis_wbw.hydroanalysis_for_level0``.

    Returns
    -------
    None
    """
    logger.info(f"Starting to performing hydroanalysis for level0 based on {flow_direction_pkg}... ...")
    # ====================== set dir and path ======================
    logger.debug(f"DEM path: {dem_level0_path}")
    
    # ====================== perform hydrological analysis ======================
    if flow_direction_pkg == "wbw":
        # import
        from .tools.hydroanalysis_func.hydroanalysis_wbw import hydroanalysis
        
        # wbw related path
        wbw_working_directory = os.path.join(evb_dir.Hydroanalysis_dir, "wbw_working_directory_level0")
        remove_and_mkdir(wbw_working_directory)
        working_directory = wbw_working_directory

        # perform hydrological analysis for level0 based on wbw
        hydroanalysis.hydroanalysis_for_level0(
            working_directory,
            dem_level0_path,
            **kwargs,
        )
        
        logger.info("hydroanalysis for level0 based on wbw has been completed successfully")
        
    else:
        logger.error("Invalid flow_direction_pkg. Please choose 'wbw'")
        print("please input correct flow_direction_pkg")
        return
    
def buildHydroanalysis_level1(
    evb_dir,
    params_dataset_level1,
    domain_dataset,
    reverse_lat=True,
    stream_acc_threshold=None,
    flow_direction_pkg="wbw",
    crs_str="EPSG:4326",
    **kwargs,
):
    """
    Build level-1 hydroanalysis products for one case.

    Parameters
    ----------
    evb_dir : Evb_dir
        Case directory manager. Output files are written to
        ``evb_dir.Hydroanalysis_dir``.
    params_dataset_level1 : netCDF4.Dataset
        Parameter dataset that provides level-1 ``lat`` and ``lon``.
    domain_dataset : netCDF4.Dataset
        Domain dataset that provides ``x_length`` and ``y_length``.
    reverse_lat : bool, optional
        Whether latitude order should be reversed when exporting DEM.
    stream_acc_threshold : float, optional
        Threshold forwarded to the WBW hydroanalysis implementation.
    flow_direction_pkg : str, optional
        Flow-direction backend. Only ``"wbw"`` is supported.
    crs_str : str, optional
        CRS used when writing output rasters.
    **kwargs
        Extra keyword arguments forwarded to
        ``hydroanalysis_wbw.hydroanalysis_for_level1``.

    Returns
    -------
    None
        This function writes:
        ``dem_level1.tif``, ``flow_direction.tif``, ``flow_acc.tif``,
        and ``flow_distance.tif``.
    """

    logger.info(f"Starting to performing hydroanalysis for level1 based on {flow_direction_pkg}... ...")
    # ====================== set dir and path ======================
    # set path
    dem_level1_path = os.path.join(evb_dir.Hydroanalysis_dir, "dem_level1.tif")
    flow_direction_path = os.path.join(evb_dir.Hydroanalysis_dir, "flow_direction.tif")
    flow_acc_path = os.path.join(evb_dir.Hydroanalysis_dir, "flow_acc.tif")
    flow_distance_path = os.path.join(evb_dir.Hydroanalysis_dir, "flow_distance.tif")

    logger.debug(f"DEM path: {dem_level1_path}")
    logger.debug(f"Flow direction path: {flow_direction_path}")

    # ====================== read ======================
    params_lat = params_dataset_level1.variables["lat"][:]
    params_lon = params_dataset_level1.variables["lon"][:]
    x_length_array = domain_dataset.variables["x_length"][:, :]
    y_length_array = domain_dataset.variables["y_length"][:, :]

    # ====================== create and save dem_level1.tif ======================
    transform = create_dem.create_dem_from_params(
        params_dataset_level1,
        dem_level1_path,
        crs_str=crs_str,
        reverse_lat=reverse_lat,
    )
    logger.debug(f"DEM created and saved to: {dem_level1_path}")

    # ====================== build flow direction ======================
    if flow_direction_pkg == "wbw":
        # import
        from .tools.hydroanalysis_func.hydroanalysis_wbw import hydroanalysis
        
        # wbw related path
        wbw_working_directory = os.path.join(evb_dir.Hydroanalysis_dir, "wbw_working_directory_level1")
        remove_and_mkdir(wbw_working_directory)
        working_directory = wbw_working_directory

        # Perform level-1 hydroanalysis in WBW workspace.
        out = hydroanalysis.hydroanalysis_for_level1(
            working_directory,
            dem_level1_path,
            stream_acc_threshold=stream_acc_threshold,
            crs_str=crs_str,
            **kwargs,
        )
        logger.info("Flow direction and accumulation calculated using wbw")

    else:
        logger.error("Invalid flow_direction_pkg. Please choose 'wbw'")
        print("please input correct flow_direction_pkg")
        return

    # cp data from workspace to Hydroanalysis_dir
    shutil.copy(os.path.join(working_directory, "flow_direction.tif"), flow_direction_path)
    shutil.copy(os.path.join(working_directory, "flow_acc.tif"), flow_acc_path)
    
    # ====================== read flow_direction ======================
    with rasterio.open(flow_direction_path, "r", driver="GTiff") as dataset:
        flow_direction_array = dataset.read(1)

    logger.debug(f"Flow direction read from: {flow_direction_path}")

    # ====================== cal flow distance and save it ======================
    create_flow_distance.create_flow_distance(
        flow_distance_path,
        flow_direction_array,
        x_length_array,
        y_length_array,
        transform,
        crs_str=crs_str,
    )
    logger.info(f"Flow distance file calculated and saved to: {flow_distance_path}")

    # clean working_directory
    # remove_and_mkdir(working_directory)
    # logger.debug(f"Workspace directory cleaned: {working_directory}")

    logger.info(f"Building hydroanalysis successfully, the results have been saved to {evb_dir.Hydroanalysis_dir}")


def buildRivernetwork_level1(
    evb_dir,
    threshold=None,
    domain_dataset=None,
    plot_bool=False,
    labeled_nodes=None,
):
    """Build river-network graphs from level-1 hydroanalysis outputs.

    Parameters
    ----------
    evb_dir : Evb_dir
        Case directory manager. ``flow_direction.tif`` and ``flow_acc.tif`` are
        read from ``evb_dir.Hydroanalysis_dir``.
    threshold : float, optional
        Flow-accumulation threshold used to extract the network.
    domain_dataset : netCDF4.Dataset, optional
        Domain dataset that provides the ``mask`` variable. If ``None``,
        ``evb_dir.domainFile_path`` is opened internally.
    plot_bool : bool, optional
        If ``True``, generate plotting figures and include them in the output.
    labeled_nodes : iterable, optional
        Optional node labels forwarded to the plotting routine.

    Returns
    -------
    dict
        Dictionary containing river-network graphs, path statistics, and
        optional figure objects.
    """
    # read flow_direction, domain dataset
    flow_direction_path = os.path.join(evb_dir.Hydroanalysis_dir, "flow_direction.tif")
    flow_acc_path = os.path.join(evb_dir.Hydroanalysis_dir, "flow_acc.tif")
    
    with rasterio.open(flow_direction_path, "r", driver="GTiff") as dataset:
        flow_direction = dataset.read(1)
    
    with rasterio.open(flow_acc_path, "r", driver="GTiff") as dataset:
        flow_acc = dataset.read(1)
    
    if domain_dataset is None:
        domain_dataset = Dataset(evb_dir.domainFile_path, "r")

    domain_mask = domain_dataset.variables["mask"][:, :]
    domain_dataset.close()
        
    # create graph
    river_network_graph, node_positions, threshold = create_river_network_graph(flow_direction, flow_acc, threshold=threshold, mask=domain_mask)
    
    # create graph full
    river_network_graph_full, *_ = create_river_network_graph(flow_direction, flow_acc, threshold=0, mask=None)
    
    # find river
    river_paths = find_river_paths(river_network_graph)
    sorted_river_paths, length_info = sort_river_paths_by_lengths(river_paths, descending=True)
    
    # get river network graph connected
    river_network_graph_connected = extract_connected_river_network(river_network_graph, min_size=10, mask=True)
    
    # plot
    if plot_bool:
        fig_river_network, ax_river_network = plot_river_network(river_network_graph, mask_by="both", threshold_label=threshold, labeled_nodes=labeled_nodes)  # sorted_river_paths[:2], "both"
        fig_river_network_full, ax_river_network_full = plot_river_network(river_network_graph_full, mask_by=None, threshold_label=0, labeled_nodes=labeled_nodes)  # sorted_river_paths[:2], "both"
        fig_river_network_connected, ax_river_network_connected = plot_river_network(river_network_graph_connected, mask_by="both", threshold_label=threshold, labeled_nodes=labeled_nodes)

    
    river_network = {
        "river_network_graph": river_network_graph,
        "river_network_graph_full": river_network_graph_full,
        "river_network_graph_connected": river_network_graph_connected,
        "node_positions": node_positions,
        "threshold": threshold,
        "sorted_river_paths": sorted_river_paths,
        "length_info": length_info,
        "figs": {
            "fig_river_network": fig_river_network,
            "fig_river_network_full": fig_river_network_full,
            "fig_river_network_connected": fig_river_network_connected
        }
    }
    
    return river_network
