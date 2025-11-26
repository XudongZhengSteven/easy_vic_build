"""
Subpackage: plot_func

A Subpackage of easy_vic_build.tools

This subpackage contains a collection of modules for creating visualizations related to
hydrological modeling and environmental data analysis. The modules within this subpackage
offer functionality to plot various hydrological parameters, model outputs, and results
in a user-friendly manner for better data interpretation and presentation.

Modules:
--------
    - plot_func: Contains functions to generate different types of plots, such as time series,
      spatial distributions, and comparison plots, for visualizing hydrological data and model outputs.


Author:
-------
    Xudong Zheng
    Email: z786909151@163.com

"""

# Importing submodules for ease of access
from . import plot_utilities, plot_evaluation, plot_map

# Define the package's public API and version
__all__ = ["plot_utilities", "plot_evaluation", "plot_map"]

# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import matplotlib.colors as mcolors
# import matplotlib.gridspec as gridspec
# import numpy as np
# import pandas as pd
# from matplotlib import cm
# from matplotlib import pyplot as plt
# from matplotlib.cm import ScalarMappable
# from matplotlib.colors import Normalize
# from matplotlib.offsetbox import AnchoredText
# from matplotlib.ticker import FuncFormatter, MultipleLocator
# from matplotlib.lines import Line2D

# from netCDF4 import num2date
# import geopandas as gpd
# import os

# from ..calibrate_func.evaluate_metrics import EvaluationMetric
# from ..geo_func.create_gdf import CreateGDF

# from ..params_func.params_set import *

# # plt.rcParams['font.family'] = 'Arial'
