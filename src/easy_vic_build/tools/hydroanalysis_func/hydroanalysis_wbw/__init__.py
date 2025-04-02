# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

""" 
hydroanalysis_wbw: A Python subpackage for performing hydrological analysis based on wbw package.

This subpackage contains a collection of modules that provide functions for hydrological analysis,
including terrain preprocessing, flow distance calculations, and basin-scale hydrological assessments.
These tools facilitate the extraction and analysis of hydrological features from digital elevation
models (DEMs) and other spatial datasets.

Modules:
--------
    - `set_workenv`:
    - `hydroanalysis`:
    - `fill_dem`:
    - `flow_direction`:
    - `flow_accumulation`: 
    - `stream_network`:
    - `outlet_detection`:
    - `basin_delineation`:

"""

# Importing submodules for ease of access
from . import (set_workenv, hydroanalysis, fill_dem, flow_direction, flow_accumulation, stream_network, outlet_detection, basin_delineation)

# Define the package's public API and version
__all__ = [
    "set_workenv",
    "hydroanalysis",
    "fill_dem",
    "flow_direction",
    "flow_accumulation",
    "stream_network",
    "outlet_detection",
    "basin_delineation"
]
