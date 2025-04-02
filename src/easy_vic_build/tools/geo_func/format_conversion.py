# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: format_conversion

This module contains functions for converting raster data to shapefiles. Specifically,
it includes the `raster_to_shp` function, which reads a raster file, extracts its geometries,
and converts them into a shapefile format. This is useful for transforming raster-based data
into vector-based formats that can be more easily analyzed and visualized in GIS software.

Functions:
----------
    - raster_to_shp: Converts a raster file to a shapefile by extracting the geometries
      of raster features and saving them as vector data.

Dependencies:
-------------
    - rasterio: Provides functions for reading and processing raster data.
    - shapely: Used to create and manipulate geometries, particularly converting raster features to shapes.
    - geopandas: Provides support for handling geospatial data, specifically creating GeoDataFrames
      and writing them to shapefiles.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com
"""

import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


def raster_to_shp(raster_path, shp_path):
    """
    Convert a raster file to a shapefile.

    This function reads a raster file, extracts its geometries (features), and saves them
    as a shapefile.

    Parameters:
    -----------
    raster_path : str
        Path to the input raster file.
    shp_path : str
        Path where the output shapefile will be saved.
    attribute_name : str, optional
        Name of the attribute field storing raster values (default="value").
    
    Returns:
    --------
    gdf : GeoDataFrame
        A GeoDataFrame containing the geometries of the features from the raster file.
    """
    with rasterio.open(raster_path, "r") as src:
        data = src.read(1)
        mask = data != src.nodata

        results = shapes(
            data,
            mask=mask,
            transform=src.transform
        )

        geoms = []
        for geom, value in results:
            geom = shape(geom)
            geoms.append(geom)

        gdf = gpd.GeoDataFrame(geometry=geoms, crs=src.crs)

        gdf.to_file(shp_path)
    