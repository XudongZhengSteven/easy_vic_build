"""
Subpackage: nc_func

A Subpackage of easy_vic_build.tools

This subpackage contains a collection of modules for NetCDF (NC) data operations. The modules focus on creating and masking
NetCDF files, which are commonly used for storing multi-dimensional scientific data.

Modules:
--------
    - create_nc: Provides functionality for creating NetCDF files from hydrological data.
    - mask_nc: Provides functionality for applying masks to NetCDF files based on specific conditions.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com

"""

# Importing submodules for ease of access
from . import create_nc, mask_nc

# Define the package's public API and version
__all__ = ["create_nc", "mask_nc"]
