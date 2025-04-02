# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

""" 
mosaic_dem.py - A Python module for merging multiple DEM files into a single mosaic with optional boundary clipping.

This module provides a function to merge multiple DEM files into a single mosaic.

Functions:
----------
    `merge_dems`: Merge multiple DEM files into a single mosaic with optional boundary clipping.

Example:
--------
    >>> merge_dems("/path/to/dems/", ".tif", output_file="output.tif")
    
Dependencies:
-------------
    - `os`: Operating system specific module.
    - `gdal`: Geospatial Data Abstraction Library (GDAL) module.

"""

from osgeo import gdal
import os
from ... import logger

def merge_dems(input_dir, suffix=".tif", output_file="merged_dem.tif",
               srcSRS="EPSG:4326", dstSRS="EPSG:4326",
               **gdal_warp_kwargs):
    """
    Merge multiple DEM files into a single mosaic with optional boundary clipping.

    Parameters
    ----------
    input_dir : str
        Directory containing input DEM files (.tif format).
    
    suffix: str
        Suffix of the DEM files (default: ".tif").
    
    output_file : str, optional
        Output file path (default: "merged_dem.tif").
        
    srcSRS : str, optional
        Source coordinate reference system (default: "EPSG:4326").
        
    dstSRS : str, optional
        Target coordinate reference system (default: "EPSG:4326").
        
    **gdal_warp_kwargs : dict
        Additional GDAL Warp options (override defaults).

    Returns
    -------
    None
        Output is written directly to the specified file.

    Notes
    -----
    Default processing parameters:
    - Cubic resampling
    - Multithreaded processing
    - LZW compression
    - Automatic BIGTIFF handling

    Examples
    --------
    >>> merge_dems("/path/to/dems/", output_file="output.tif",
    ...           cutline_file="aoi.shp", blendDistance=30)
    """
    logger.info(f"Starting to merge dems in {input_dir}... ...")
    
    # get all dem files in the input directory
    dem_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(suffix)]
    if not dem_files:
        logger.error("No DEM files found in the input directory.")
        return
    
    # set kwargs
    warp_options  = {
        "format": "GTiff",
        "outputType": gdal.GDT_Float32,
        "resampleAlg": "cubic",
        "srcNodata": -9999,
        "dstNodata": -9999,
        "multithread": True,  # use multithread
        "warpMemoryLimit": 2048,  # memory limit (MB)
        "srcSRS": srcSRS,
        "dstSRS": dstSRS,
        "creationOptions": ["COMPRESS=LZW", "BIGTIFF=IF_NEEDED"],  # creation options
    }
    
    warp_options.update(gdal_warp_kwargs)
    
    # merge dems
    try:
        gdal.Warp(
            output_file,
            dem_files,
            options=gdal.WarpOptions(**warp_options),
        )
        
        logger.info(f"Merge dems sucessfully, saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to merge dems: {e}")
        return