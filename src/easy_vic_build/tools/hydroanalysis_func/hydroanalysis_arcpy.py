# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: hydroanalysis_arcpy

This module contains the `hydroanalysis_arcpy` function, which performs hydrological analysis 
using ArcPy. Specifically, the function processes Digital Elevation Model (DEM) data to generate 
various hydrological outputs, including flow direction, flow accumulation, stream accumulation, 
stream link, and stream features. The analysis is conducted by invoking an external ArcPy Python 
script and passing the necessary parameters.

Functions:
----------
    - hydroanalysis_arcpy: Executes hydrological analysis based on ArcPy, generating several 
      hydrological maps and features, including flow direction and stream features.

Dependencies:
-------------
    - os: Provides functions for interacting with the operating system, such as file and directory management.
    - arcpy (external): Used for geospatial data analysis, particularly for hydrological modeling.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com
"""


import os

def hydroanalysis_arcpy(workspace_path, dem_tiff_path, arcpy_python_path, arcpy_python_script_path, stream_acc_threshold):
    """
    Performs hydrological analysis based on ArcPy by processing DEM data to generate hydrological outputs 
    such as flow direction, flow accumulation, and stream features.

    This function calls an external ArcPy script, passing the paths to the necessary input files 
    and the analysis threshold, and then executes the script using the specified Python environment.

    Parameters:
    -----------
    workspace_path : str
        The directory where the output files from the hydrological analysis will be saved.
    dem_tiff_path : str
        The file path to the input Digital Elevation Model (DEM) in TIFF format.
    arcpy_python_path : str
        The path to the Python executable for ArcPy.
    arcpy_python_script_path : str
        The path to the ArcPy Python script that will perform the hydrological analysis.
    stream_acc_threshold : float
        The threshold value for stream accumulation to define stream features.

    Returns:
    --------
    out : int
        The output of the system command execution. Typically, a value indicating success or failure of the command.
    """
    # hydroanalysis based on arcpy
    stream_acc_threshold = str(stream_acc_threshold) #* set this threshold each time
    filled_dem_file_path = os.path.join(workspace_path, "filled_dem")
    flow_direction_file_path = os.path.join(workspace_path, "flow_direction")
    flow_acc_file_path = os.path.join(workspace_path, "flow_acc")
    stream_acc_file_path = os.path.join(workspace_path, "stream_acc")
    stream_link_file_path = os.path.join(workspace_path, "stream_link")
    stream_feature_file_path = "stream_feature"
    command_arcpy = " ".join([arcpy_python_script_path, workspace_path, stream_acc_threshold, dem_tiff_path, filled_dem_file_path,
                              flow_direction_file_path, flow_acc_file_path, stream_acc_file_path, stream_link_file_path,
                              stream_feature_file_path])
    
    # conduct arcpy file
    out = os.system(f'{arcpy_python_path} {command_arcpy}')
    
    return out