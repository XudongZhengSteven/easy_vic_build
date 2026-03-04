# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Whitebox-based hydrological analysis components.

Modules
-------
set_workenv
    Whitebox work-environment setup helpers.
hydroanalysis
    End-to-end orchestration helpers.
fill_dem
    DEM filling and conditioning routines.
flow_direction
    Flow-direction derivation utilities.
flow_accumulation
    Flow-accumulation derivation utilities.
stream_network
    Stream extraction and network analysis helpers.
outlet_detection
    Outlet detection utilities.
basin_delineation
    Basin delineation utilities.
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
