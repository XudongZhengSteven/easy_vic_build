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
    - hydroanalysis_arcpy: Provides hydrological analysis functions utilizing the ArcPy library.
    - hydroanalysis_wbw: Contains tools for watershed and basin-wide hydrological analysis.
    - hydroanalysis_for_BasinMap: Supports hydrological feature extraction for basin-scale mapping.


Author:
-------
    Xudong Zheng
    Email: z786909151@163.com
    
"""

# Importing submodules for ease of access
from . import create_dem
from . import create_flow_distance
from . import hydroanalysis_arcpy
from . import hydroanalysis_wbw
from . import hydroanalysis_for_BasinMap

# Define the package's public API and version
__all__ = ["create_dem", "create_flow_distance", "hydroanalysis_arcpy", "hydroanalysis_wbw", "hydroanalysis_for_BasinMap"]