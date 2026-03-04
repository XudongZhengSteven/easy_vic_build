# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Merge multiple DEM tiles into one output raster."""

import os
from ... import logger

def merge_dems(input_dir, suffix=".tif", output_file="merged_dem.tif",
               srcSRS="EPSG:4326", dstSRS="EPSG:4326",
               **gdal_warp_kwargs):
    """Merge DEM files in a directory into one raster using GDAL Warp.

    Parameters
    ----------
    input_dir : str
        Directory containing DEM files.
    suffix : str, optional
        Filename suffix used to select DEM files.
    output_file : str, optional
        Output raster path.
    srcSRS : str, optional
        Source CRS string.
    dstSRS : str, optional
        Destination CRS string.
    **gdal_warp_kwargs : dict
        Additional keyword arguments passed to ``gdal.WarpOptions``.

    Returns
    -------
    None
        The merged raster is written to ``output_file``.

    Notes
    -----
    Default Warp settings include cubic resampling, LZW compression,
    and multithreaded processing.
    """
    try:
        from osgeo import gdal
    except ImportError:
        logger.error("gdal is not avaiable for mosaic_dem module")
    
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
