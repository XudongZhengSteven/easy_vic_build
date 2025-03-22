"""
Subpackage: geo_func

A Subpackage of easy_vic_build.tools

This subpackage contains a collection of modules that provide functions for processing geospatial data.
These functions support common tasks such as clipping, grid search, format conversion, and resampling,
which are often required in environmental and hydrological modeling.

Modules:
--------
    - clip: Provides functions for clipping geospatial data (e.g., shapefiles, raster) to a specified region.
    - create_gdf: Provides utilities for creating GeoDataFrames from various data formats.
    - format_conversion: Contains functions for converting between different geospatial data formats (e.g., GeoJSON, Shapefile).
    - resample: Provides functions for resampling geospatial data (e.g., raster data) to a different resolution or grid.
    - search_grids: Implements functions for searching for grid data within specified bounds or conditions.


Author:
-------
    Xudong Zheng
    Email: z786909151@163.com

"""

# Importing submodules for ease of access
from . import clip, create_gdf, format_conversion, resample, search_grids

# Define the package's public API and version
__all__ = ["clip", "create_gdf", "format_conversion", "resample", "search_grids"]
