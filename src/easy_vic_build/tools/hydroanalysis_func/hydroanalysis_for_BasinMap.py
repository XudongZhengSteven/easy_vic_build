# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: hydroanalysis_for_BasinMap

This module contains the `hydroanalysis_for_basin` function, which performs hydrological analysis
for a specific basin based on input data. It integrates multiple steps, including reading
Digital Elevation Model (DEM) data, performing hydrological analysis with the `hydroanalysis_wbw`
function, and clipping stream geometries based on the basin boundary. The module leverages geospatial
data structures (GeoDataFrames) and external utilities for hydrological analysis and map generation.

Functions:
----------
    - hydroanalysis_for_basin: Conducts a series of hydrological analyses for a basin, including
      DEM processing, stream analysis, and clipping of stream data based on basin geometry.

Dependencies:
-------------
    - geopandas: Provides support for handling geospatial data and operations on GeoDataFrames,
      such as clipping and overlaying geometries.
    - os: Used for interacting with the file system, such as path handling.
    - ..utilities: Provides various utility functions required for data processing and analysis.
    - .create_dem: Contains the `create_dem_from_params` function to generate DEM files from parameters.
    - .hydroanalysis_wbw: Contains the `hydroanalysis_wbw` function for performing hydrological analysis.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com
"""


import geopandas as gpd

from ..utilities import *
from .create_dem import create_dem_from_params
from .hydroanalysis_wbw import hydroanalysis_wbw


def hydroanalysis_for_basin(evb_dir):
    """
    Perform hydrological analysis for a specific basin. This includes reading DEM data,
    conducting hydrological analysis using the `hydroanalysis_wbw` function, and clipping
    stream geometries based on the basin's boundary.

    Parameters:
    -----------
    evb_dir : object
        The directory object containing paths to the necessary input and output directories,
        including `BasinMap_dir`, which holds DEM and stream data.

    Returns:
    --------
    None
        This function does not return any value but generates and saves the necessary hydrological
        analysis outputs to files in the `BasinMap_dir`.
    """
    # read dpc
    dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2 = readdpc(evb_dir)

    # read params
    params_dataset_level0, params_dataset_level1 = readParam(evb_dir, mode="r")

    # hydroanalysis for level0
    transform = create_dem_from_params(
        params_dataset_level0,
        os.path.join(evb_dir.BasinMap_dir, "dem_level0.tif"),
        crs_str="EPSG:4326",
        reverse_lat=True,
    )
    hydroanalysis_wbw(
        evb_dir.BasinMap_dir,
        os.path.join(evb_dir.BasinMap_dir, "dem_level0.tif"),
        create_stream=True,
    )

    # clip stream gdf within basin shp
    stream_gdf = gpd.read_file(
        os.path.join(evb_dir.BasinMap_dir, "stream_raster_shp.shp")
    )
    stream_gdf_clip = gpd.overlay(
        stream_gdf,
        dpc_VIC_level0.basin_shp.loc[:, "geometry":"geometry"],
        how="intersection",
    )
    stream_gdf_clip.to_file(
        os.path.join(evb_dir.BasinMap_dir, "stream_raster_shp_clip.shp")
    )

    # close params
    params_dataset_level0.close()
    params_dataset_level1.close()
