# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd


def raster_to_shp(raster_path, shp_path):
    with rasterio.open(raster_path, "r") as raster_dataset:
        raster = raster_dataset.read(1)
        mask = raster != raster_dataset.nodata
        results = shapes(raster, mask=mask, transform=raster_dataset.transform)
        geoms = []
        for geom, value in results:
            geom = shape(geom)
            geoms.append(geom)

        gdf = gpd.GeoDataFrame(geometry=geoms, crs=raster_dataset.crs)

        gdf.to_file(shp_path)

    return gdf
