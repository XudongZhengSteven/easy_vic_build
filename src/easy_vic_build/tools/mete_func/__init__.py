"""
Subpackage: mete_func

A Subpackage of easy_vic_build.tools

This subpackage contains a collection of modules focused on meteorological data processing.
The primary functionality includes handling meteorological inputs.

Modules:
--------
    - mete_func: Provides functions for generating digital elevation models (DEMs) from input topographic data.
      It handles the creation of elevation grids based on latitude, longitude, and elevation data.


Author:
-------
    Xudong Zheng
    Email: z786909151@163.com

"""

# Importing submodules for ease of access
from . import mete_func

# Define the package's public API and version
__all__ = ["mete_func"]
