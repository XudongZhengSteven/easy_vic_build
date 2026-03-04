# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Module ``easy_vic_build.tools.geo_func.reproject``."""

import rasterio
from rasterio.warp import reproject, calculate_default_transform, Resampling

def reproject_raster(src_fname, dst_fname, dst_crs):
    """
    Reproject a raster file to a target CRS.

    Parameters
    ----------
    src_fname : str
        Source raster path.
    dst_fname : str
        Output raster path.
    dst_crs : str or rasterio.crs.CRS
        Target coordinate reference system.

    Returns
    -------
    None
        Output is written to ``dst_fname``.
    """
    with rasterio.open(src_fname) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(dst_fname, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
