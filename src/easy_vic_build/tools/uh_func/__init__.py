"""
Subpackage: uh_func

A Subpackage of easy_vic_build.tools

This subpackage contains a collection of modules for creating and managing unit hydrographs
and related functions for hydrological modeling. It provides tools for generating, analyzing,
and visualizing unit hydrographs, which are critical for simulating runoff responses in hydrological models.

Modules:
--------
    - create_uh: Provides functionality for creating unit hydrographs based on different methods
      and parameters, suitable for use in hydrological simulations.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com

"""

# Importing submodules for ease of access
from . import create_uh

# Define the package's public API and version
__all__ = ["create_uh"]
