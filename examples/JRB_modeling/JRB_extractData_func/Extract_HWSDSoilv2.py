# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
import rasterio
import numpy as np
from matplotlib import pyplot as plt
from tqdm import *

from easy_vic_build.tools.geo_func import resample, search_grids
from easy_vic_build.tools.geo_func.create_gdf import CreateGDF
from easy_vic_build.tools.params_func.TansferFunction import SoilLayerResampler

import pyodbc

HWSD_Soil_layers_depths = [
    0.20,
    0.20,
    0.20,
    0.20,
    0.20,
    0.50,
    0.50,
]  # 7 layers m, 0–20cm, 20–40cm, 40–60cm, 60–80cm, 80–100cm, 100–150cm and 150–200cm, D1-D7

def ExtractData(
    grid_shp,
    grid_shp_res=0.125,
    plot_layer=True,
    check_search=False,
):
    # plot_layer: start from 1
    layer_names = [f"D{i}" for i in range(1, 8)]
    
    # read HWSD raster (MU)
    HWSDSoil_BIL_path = "E:\\data\\LULC\\HWSD_Harmonized World Soil database\\HWSD2.0\\HWSD2_RASTER\\HWSD2.bil"
    mdb_path = "E:\\data\\LULC\\HWSD_Harmonized World Soil database\\HWSD2.0\\HWSD2.mdb"
    
    with rasterio.open(HWSDSoil_BIL_path) as src:
        HWSDSoil_BIL = src.read(1)
        HWSDSoil_BIL_meta = src.meta
        
        src_transform = src.transform
        width = src.width
        height = src.height

        ul = src_transform * (0, 0)
        lr = src_transform * (width, height)

        HWSDSoil_lon = np.linspace(ul[0], lr[0], width)
        HWSDSoil_lat = np.linspace(ul[1], lr[1], height)  # large -> small

        HWSDSoil_lat_res = (max(HWSDSoil_lat) - min(HWSDSoil_lat)) / (len(HWSDSoil_lat) - 1)
        HWSDSoil_lon_res = (max(HWSDSoil_lon) - min(HWSDSoil_lon)) / (len(HWSDSoil_lon) - 1)      
        
        # clip: extract before to improve speed
        xindex_start = np.where(HWSDSoil_lon <= min(grids_lon) - grid_shp_res)[0][-1]
        xindex_end = np.where(HWSDSoil_lon >= max(grids_lon) + grid_shp_res)[0][0]

        yindex_start = np.where(HWSDSoil_lat >= max(grids_lat) + grid_shp_res)[0][-1]  # large -> small
        yindex_end = np.where(HWSDSoil_lat <= min(grids_lat) - grid_shp_res)[0][0]
        
        HWSDSoil_lon_clip = HWSDSoil_lon[xindex_start : xindex_end + 1]
        HWSDSoil_lat_clip = HWSDSoil_lat[yindex_start : yindex_end + 1]
        
        HWSDSoil_BIL_clip = HWSDSoil_BIL[yindex_start : yindex_end + 1, xindex_start : xindex_end + 1]
    
    # set grids_lat, lon
    grids_lat = grid_shp.point_geometry.y.to_list()
    grids_lon = grid_shp.point_geometry.x.to_list()
    # grids_lat = [grid_shp.loc[i, :].point_geometry.y for i in grid_shp.index]
    # grids_lon = [grid_shp.loc[i, :].point_geometry.x for i in grid_shp.index]
    
    # search grids
    print("========== search grids for SoilGrids 1km ==========")
    searched_grids_index = search_grids.search_grids_nearest(
        dst_lat=grids_lat,
        dst_lon=grids_lon,
        src_lat=HWSDSoil_lat_clip,
        src_lon=HWSDSoil_lon_clip,
        search_num=1,
    )
    
    # search and inquire
    conn = pyodbc.connect(
        "DRIVER={Microsoft Access Driver (*.mdb)};"
        f"DBQ={mdb_path};"
    )
    
    cursor = conn.cursor()
    
    for l in range(len(layer_names)):
        silt_nearest_Value = []
        clay_nearest_Value = []
        sand_nearest_Value = []
        bulk_density_nearest_Value = []
        
        for i in tqdm(
            grid_shp.index,
            colour="green",
            desc=f"loop for each grid to extract soil{l} data",
            leave=False,
        ):
            # lon/lat
            searched_grid_index = searched_grids_index[i]
            
            searched_grid_lat = [
                HWSDSoil_lat_clip[searched_grid_index[0][j]]
                for j in range(len(searched_grid_index[0]))
            ]
            
            searched_grid_lon = [
                HWSDSoil_lon_clip[searched_grid_index[1][j]]
                for j in range(len(searched_grid_index[0]))
            ]
            
            # search
            BIL_searched_grid_data = [
                HWSDSoil_BIL_clip[searched_grid_index[0][j], searched_grid_index[1][j]]
                for j in range(len(searched_grid_index[0]))
            ]
            
            # inquire 
            # TODO
            # sql = f"""
            #     SELECT
            #         Silt, Clay, Sand, BulkDensity
            #     FROM
            #         HWSD2_LAYERS
            #     WHERE
            #         d.HWSD2_SMU_ID = {}
            # # """
            # cursor.execute(sql)
            # result = cursor.fetchone()
            
        
        

    
    
    