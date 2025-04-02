"""
Subpackage: hydroanalysis_func

A Subpackage of easy_vic_build.tools

This subpackage contains a collection of modules that provide functions for hydrological analysis,
including terrain preprocessing, flow distance calculations, and basin-scale hydrological assessments.
These tools facilitate the extraction and analysis of hydrological features from digital elevation
models (DEMs) and other spatial datasets.

Modules:
--------
    - create_dem: Generates digital elevation models (DEMs) from input topographic data.
    - create_flow_distance: Computes flow distance from a given source, aiding in hydrological modeling.
    - mosaic_dem: Mosaics multiple DEMs into a single raster dataset.
    - hydroanalysis_arcpy: Provides hydrological analysis functions utilizing the ArcPy library.
    - hydroanalysis_wbw: Contains tools for watershed and basin-wide hydrological analysis.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com

"""

# Importing submodules for ease of access
from . import (create_dem, create_flow_distance, mosaic_dem, hydroanalysis_wbw, hydroanalysis_arcpy)

# Define the package's public API and version
__all__ = [
    "create_dem",
    "create_flow_distance",
    "mosaic_dem",
    "hydroanalysis_wbw",
    "hydroanalysis_arcpy",
]
