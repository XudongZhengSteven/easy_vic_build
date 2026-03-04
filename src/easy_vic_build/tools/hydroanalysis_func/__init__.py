"""Hydrological analysis modules.

This subpackage contains terrain preprocessing and watershed-analysis tools,
including DEM preparation, flow-distance generation, DEM mosaicking, and
Whitebox-based hydrological workflows.
"""

# Importing submodules for ease of access
from . import (create_dem, create_flow_distance, mosaic_dem, hydroanalysis_wbw)

# Define the package's public API and version
__all__ = [
    "create_dem",
    "create_flow_distance",
    "mosaic_dem",
    "hydroanalysis_wbw",
]
