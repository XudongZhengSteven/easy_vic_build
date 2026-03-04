# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Clipping utilities for array grids and GeoTIFF rasters."""

import numpy as np
import os
import rasterio
from rasterio.mask import mask
from rasterio import windows
import geopandas as gpd


def clip(dst_lat, dst_lon, dst_res, src_lat, src_lon, src_data, reverse_lat=True):
    """
    Clip source gridded arrays by destination extent and resolution buffer.

    This function is typically used as a pre-step before grid searching to
    reduce search-space size and improve runtime.

    Parameters
    ----------
    dst_lat : array-like
        The latitude values of the target grid (destination).

    dst_lon : array-like
        The longitude values of the target grid (destination).

    dst_res : float
        The resolution of the target grid (in degrees).

    src_lat : array-like
        The latitude values of the source data grid.

    src_lon : array-like
        The longitude values of the source data grid.

    src_data : array-like
        Source 2D data array indexed by ``[lat, lon]``.

    reverse_lat : bool, optional
        If True, assumes the source latitude values are in descending order (large to small).
        If False, assumes ascending order (small to large). Default is True.

    Returns
    -------
    tuple
        ``(src_data_clip, src_lon_clip, src_lat_clip)``.

    Notes
    -----
    ``dst_res / 2`` is used as an outer buffer when locating source indices.
    """
    xindex_start = np.where(src_lon <= min(dst_lon) - dst_res / 2)[0][-1]
    xindex_end = np.where(src_lon >= max(dst_lon) + dst_res / 2)[0][0]

    # if reverse_lat (src_lat, large -> small), else (src_lat, small -> large)
    if reverse_lat:
        yindex_start = np.where(src_lat >= max(dst_lat) + dst_res / 2)[0][-1]
        yindex_end = np.where(src_lat <= min(dst_lat) - dst_res / 2)[0][0]
    else:
        yindex_start = np.where(src_lat <= min(dst_lat) - dst_res / 2)[0][-1]
        yindex_end = np.where(src_lat >= max(dst_lat) + dst_res / 2)[0][0]

    src_data_clip = src_data[
        yindex_start : yindex_end + 1, xindex_start : xindex_end + 1
    ]
    src_lon_clip = src_lon[xindex_start : xindex_end + 1]
    src_lat_clip = src_lat[yindex_start : yindex_end + 1]

    ## old version
    # xindex = np.where((src_lon >= min(dst_lon) - dst_res/2) & (src_lon <= max(dst_lon) + dst_res/2))[0]
    # yindex = np.where((src_lat >= min(dst_lat) - dst_res/2) & (src_lat <= max(dst_lat) + dst_res/2))[0]

    # src_data_clip = src_data[min(yindex): max(yindex), min(xindex): max(xindex)]
    # src_lon_clip = src_lon[min(xindex): max(xindex)]
    # src_lat_clip = src_lat[min(yindex): max(yindex)]

    ## then search grids
    # searched_grids_index = search_grids.search_grids_radius_rectangle(dst_lat=grids_lat, dst_lon=grids_lon,
    #                                                                     src_lat=umd_lat_clip, src_lon=umd_lon_clip,
    #                                                                     lat_radius=grid_shp_res/2, lon_radius=grid_shp_res/2)

    return src_data_clip, src_lon_clip, src_lat_clip


def clip_tiff(
    input_tiff: str,
    output_tiff: str,
    bbox: tuple = None,
    shp_path: str = None
):
    """
    Clip a GeoTIFF by bounding box or shapefile boundary.

    Parameters
    ----------
    input_tiff : str
        Path to the input GeoTIFF file.
    output_tiff : str
        Path to the output clipped GeoTIFF file.
    bbox : tuple, optional
        Geographic extent (xmin, ymin, xmax, ymax), e.g. (105.2, 33.5, 106.8, 34.9).
    shp_path : str, optional
        Path to a shapefile used as the clipping boundary.

    Returns
    -------
    None
        Clipped raster is written to ``output_tiff``.

    Raises
    ------
    FileNotFoundError
        If ``input_tiff`` or ``shp_path`` does not exist.
    ValueError
        If neither ``bbox`` nor ``shp_path`` is provided.
    """
    # ---------------- Check input validity ----------------
    if not os.path.exists(input_tiff):
        raise FileNotFoundError(f"Input file not found: {input_tiff}")

    if bbox is None and shp_path is None:
        raise ValueError("Either 'bbox' or 'shp_path' must be provided.")

    with rasterio.open(input_tiff) as src:
        # ---------------- Clip by shapefile ----------------
        if shp_path is not None:
            if not os.path.exists(shp_path):
                raise FileNotFoundError(f"Shapefile not found: {shp_path}")

            # Read shapefile and convert CRS to match raster
            gdf = gpd.read_file(shp_path)
            gdf = gdf.to_crs(src.crs)

            # Extract geometry
            geoms = [feature["geometry"] for feature in gdf.__geo_interface__["features"]]

            # Perform masking and cropping
            out_image, out_transform = mask(src, geoms, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

        # ---------------- Clip by bounding box ----------------
        elif bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            # Compute the pixel window corresponding to the bbox
            window = windows.from_bounds(xmin, ymin, xmax, ymax, src.transform)

            # Read data within the window
            out_image = src.read(window=window)
            out_transform = windows.transform(window, src.transform)

            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

    # ---------------- Write clipped raster ----------------
    os.makedirs(os.path.dirname(output_tiff), exist_ok=True)
    with rasterio.open(output_tiff, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f"Clipping completed. Output saved to: {output_tiff}")

    
    
