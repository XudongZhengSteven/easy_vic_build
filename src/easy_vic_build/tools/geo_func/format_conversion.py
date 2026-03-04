# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Format-conversion helpers for geospatial datasets.

This module currently provides raster-to-vector conversion utilities.
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

    Parameters
    ----------
    raster_path : str
        Path to the input raster file.
    shp_path : str
        Path where the output shapefile will be saved.
    
    Returns
    -------
    None
        Output shapefile is written to ``shp_path``.
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
    
