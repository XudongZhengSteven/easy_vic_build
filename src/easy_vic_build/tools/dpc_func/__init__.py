"""
Subpackage: dpc_func

A Subpackage of easy_vic_build.tools

This subpackage contains a collection of modules that provide functions for data processing classes (dpc class).

Modules:
--------
    - basin_grid_class: Contains the class definitions for basin grid-related functionality.
      
    - basin_grid_func: Provides various functions for manipulating basin grids.
    
    - aggregate: Implements functions for aggregating data.
    
    - dpc_base: Provides base functionality for data processing classes.
    
    - dpc_subclass: Defines subclasses for different types of data processing.
    
    - readdataIntoBasins_interface: Interfaces for reading data into basins.
    
    - readdataIntoGrids_interface: Interfaces for reading data into grids.
    
    - select_basin_shp: Functions for selecting basin shapes from spatial data.


Author:
-------
    Xudong Zheng
    Email: z786909151@163.com
    
"""

# Importing submodules for ease of access
from . import basin_grid_class
from . import basin_grid_func
from . import aggregate

from . import dpc_base
from . import dpc_subclass
from . import readdataIntoBasins_interface
from . import readdataIntoGrids_interface
from . import select_basin_shp

# Define the package's public API and version
__all__ = ["basin_grid_class", "basin_grid_func", "aggregate", "dpc_base", "dpc_subclass", "readdataIntoBasins_interface", "readdataIntoGrids_interface", "select_basin_shp"]