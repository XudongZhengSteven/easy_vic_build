# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import os
import rasterio
from rasterio import CRS


def create_dem_from_params(params_dataset_level1, save_path, crs_str="EPSG:4326", reverse_lat=True):
    # read
    params_lat = params_dataset_level1.variables["lat"][:]
    params_lon = params_dataset_level1.variables["lon"][:]
    params_elev = params_dataset_level1.variables["elev"][:, :]
    
    # ====================== create and save dem_level1.tif ======================
    ulx = min(params_lon)
    uly = max(params_lat)
    xres = round((max(params_lon) - min(params_lon)) / (len(params_lon) - 1), 6)
    yres = round((max(params_lat) - min(params_lat)) / (len(params_lat) - 1), 6)
    if reverse_lat:
        transform = rasterio.transform.from_origin(ulx-xres/2, uly+yres/2, xres, yres)
    else:
        transform = rasterio.transform.from_origin(ulx-xres/2, uly-yres/2, xres, yres)
    
    if not os.path.exists(save_path):
        with rasterio.open(save_path, 'w', driver='GTiff',
                           height=params_elev.shape[0],
                           width=params_elev.shape[1],
                           count=1,
                           dtype=params_elev.dtype,
                           crs=CRS.from_string(crs_str),
                           transform=transform,
                           ) as dst:
            dst.write(params_elev, 1)
    
    return transform