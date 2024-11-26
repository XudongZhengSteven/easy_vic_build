# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from copy import copy
from tqdm import *
import geopandas as gpd
import rasterio
from rasterio.plot import show
#import pyodbc
from ...geo_func import search_grids, resample
from rasterio import mask
from copy import copy

import warnings
warnings.filterwarnings('ignore')


def ExtractData(grid_shp, boundary_shp, plot=True):
    # read HWSD soil type
    HWSD_Soil_BIL_path = "E:/data/LULC/HWSD_Harmonized World Soil database/hwsd.bil"
    HWSD_Soil_BIL = rasterio.open(HWSD_Soil_BIL_path)
    HWSD_Soil_array = HWSD_Soil_BIL.read(1)

    # grids in grid_shp
    grid_shp_x = grid_shp.point_geometry.x.values
    grid_shp_y = grid_shp.point_geometry.y.values

    # HWSD grids, bounds: left, bottom, right, top
    HWSD_x = np.linspace(HWSD_Soil_BIL.bounds[0], HWSD_Soil_BIL.bounds[2], HWSD_Soil_BIL.width)
    HWSD_y = np.linspace(HWSD_Soil_BIL.bounds[1], HWSD_Soil_BIL.bounds[3], HWSD_Soil_BIL.height)

    # clip
    # grid_x_boundary
    grid_x_boundary = [grid_shp_x.min(), grid_shp_x.max()]
    grid_y_boundary = [grid_shp_y.min(), grid_shp_y.max()]

    # search x y of HWSD grids located in grid_boundary, x is lon/col, y is lat/row
    index_x_boundary = [np.where(HWSD_x <= grid_x_boundary[0])[0][-1], np.where(HWSD_x >= grid_x_boundary[1])[0][0]]
    index_y_boundary = [np.where(HWSD_y <= grid_y_boundary[0])[0][-1], np.where(HWSD_y >= grid_y_boundary[1])[0][0]]
    clip_HWSD_x = HWSD_x[index_x_boundary[0]: index_x_boundary[1] + 1]
    clip_HWSD_y = HWSD_y[index_y_boundary[0]: index_y_boundary[1] + 1]
    # !note: index for array not suitable for the rasterio.dataset, need to use the dataset.index(x, y)
    index_clip_array_x = [HWSD_Soil_BIL.index(clip_HWSD_x[i], clip_HWSD_y[0])[1] for i in range(len(clip_HWSD_x))]
    index_clip_array_y = [HWSD_Soil_BIL.index(clip_HWSD_x[0], clip_HWSD_y[i])[0] for i in range(len(clip_HWSD_y))]

    index_clip_array_x_mesh, index_clip_array_y_mesh = np.meshgrid(index_clip_array_x, index_clip_array_y)
    # index_clip_array_y_mesh, index_clip_array_x_mesh = np.meshgrid(index_clip_array_y, index_clip_array_x) will rotate the array

    # clip grid_boundary array from HWSD array, x is lon/col, y is lat/row
    clip_HWSD_Soil_array = HWSD_Soil_array[index_clip_array_y_mesh, index_clip_array_x_mesh]
    clip_HWSD_Soil_array_flip = np.flip(clip_HWSD_Soil_array, axis=0)

    # search HWSD grids for each grid in grid_shp
    searched_grids_index = search_grids.search_grids_radius_rectangle(dst_lat=grid_shp_y, dst_lon=grid_shp_x,
                                                                      src_lat=clip_HWSD_y, src_lon=clip_HWSD_x,
                                                                      lat_radius=0.125, lon_radius=0.125)

    # resample for Majority
    HWSD_BIL_Value = []
    for i in range(len(searched_grids_index)):
        searched_grid_index = searched_grids_index[i]
        searched_grid_data = [clip_HWSD_Soil_array[searched_grid_index[0][j],
                                                   searched_grid_index[1][j]]
                              for j in range(len(searched_grid_index[0]))]  # index: (lat, lon), namely (row, col)
        dst_data = resample.resampleMethod_Majority(searched_grid_data, None, None, None, None, missing_value=0)
        HWSD_BIL_Value.append(dst_data)

    # save in grid_shp
    grid_shp["HWSD_BIL_Value"] = HWSD_BIL_Value

    # plot
    if plot:
        ax1 = show(HWSD_Soil_BIL)

        # mask
        boundary_shp_ = copy(boundary_shp)
        boundary_shp_ = boundary_shp_.to_crs(HWSD_Soil_BIL.crs)
        geo = boundary_shp_.geometry
        clip_image, clip_transform = mask.mask(HWSD_Soil_BIL, geo, crop=True)
        clip_meta = HWSD_Soil_BIL.meta
        clip_meta.update({"driver": "GTiff",
                          "height": clip_image.shape[1],
                          "width": clip_image.shape[2],
                          "transform": clip_transform})
        ax2 = show(clip_image, transform=clip_transform)

        # clip_HWSD_Soil_array
        ax3 = show(clip_HWSD_Soil_array_flip, transform=clip_transform)

    HWSD_Soil_BIL.close()

    return grid_shp


def inquireHWSDSoilData(MU_GLOBALS):
    # connect
    driver = '{Microsoft Access Driver (*.mdb, *.accdb)}'
    HWSD_Soil_mdb_path = "E:/data/LULC/HWSD_Harmonized World Soil database/HWSD.mdb"
    cnxn = pyodbc.connect(f'Driver={driver};DBQ={HWSD_Soil_mdb_path}')

    # cursor
    crsr = cnxn.cursor()

    # inquire data
    T_USDA_TEX_CLASS = []
    S_USDA_TEX_CLASS = []
    for MU_GLOBAL in tqdm(MU_GLOBALS, colour="green", desc="inquire T/S_USDA_TEX_CLASS for MU_GLOBALS"):
        try:
            crsr.execute(f'SELECT {"T_USDA_TEX_CLASS"} FROM HWSD_DATA where MU_GLOBAL = {MU_GLOBAL}')
            T_USDA_TEX_CLASS_ = crsr.fetchone()[0]
        except:
            T_USDA_TEX_CLASS_ = None

        try:
            crsr.execute(f'SELECT {"S_USDA_TEX_CLASS"} FROM HWSD_DATA where MU_GLOBAL = {MU_GLOBAL}')
            S_USDA_TEX_CLASS_ = crsr.fetchone()[0]
        except:
            S_USDA_TEX_CLASS_ = None

        T_USDA_TEX_CLASS.append(T_USDA_TEX_CLASS_)
        S_USDA_TEX_CLASS.append(S_USDA_TEX_CLASS_)

    cnxn.close()

    # !find: T_USDA_TEX_CLASS {3.0, 4.0, 5.0, 7.0, 9.0, 10.0, 11.0, 12.0, 13.0, nan}
    # !find: S_USDA_TEX_CLASS {2.0, 3.0, 5.0, 7.0, 9.0, 10.0, 11.0, 12.0, nan}

    return T_USDA_TEX_CLASS, S_USDA_TEX_CLASS


if __name__ == "__main__":
    inquireHWSDSoilData()
