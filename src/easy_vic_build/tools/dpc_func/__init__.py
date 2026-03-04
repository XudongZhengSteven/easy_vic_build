"""Data-processing modules for basin and grid workflows.

This subpackage contains core modules used to construct basin/grid data and run
data-processing pipelines.

Modules
-------
aggregate
    Basin-level aggregation helpers for grid-based variables.
basin_grid_class
    Basin/grid GeoDataFrame classes and constructors.
basin_grid_func
    Grid construction and array-mapping utility functions.
dpc_base
    Base pipeline class with step registration and cache management.
dpc_subclass
    Level-specific processing pipelines built on ``dpc_base``.
select_basin_shp
    Basin filtering helpers based on hydrological criteria.
"""

# Importing submodules for ease of access
from . import (aggregate, basin_grid_class, basin_grid_func, dpc_base,
               dpc_subclass, select_basin_shp)

# Define the package's public API and version
__all__ = [
    "basin_grid_class",
    "basin_grid_func",
    "aggregate",
    "dpc_base",
    "dpc_subclass",
    "select_basin_shp",
]
