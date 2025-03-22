# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os

import numpy as np
from tqdm import *

from ...geo_func.search_grids import search_grids_equal


def ExtractData(grid_shp):
    # read landcover, classes: 0-14
    umd_landcover_qdeg_path = "E:/data/LULC/UMD_landcover_classification/ISLSCP_II_University_of_Maryland_Global_Land_Cover_Classifications/umd_landcover_qdeg"
    umd_path = [
        os.path.join(umd_landcover_qdeg_path, f"umd_landcover_qd_c0{i}.asc")
        for i in range(10)
    ]
    umd_path.extend(
        os.path.join(umd_landcover_qdeg_path, f"umd_landcover_qd_c{i}.asc")
        for i in range(10, 15)
    )
    umd_path.sort()
    umd_landcover_Cv_all = [
        np.loadtxt(path, skiprows=6) for path in umd_path
    ]  # 0-14 class

    major_umd_landcover_qdeg_path = os.path.join(
        umd_landcover_qdeg_path, "umd_landcover_class_qd.asc"
    )
    major_umd_landcover_qdeg = np.loadtxt(major_umd_landcover_qdeg_path, skiprows=6)

    # umd lat lon
    umd_lat = np.arange(
        89.875, -90.125, -0.25
    )  # -90.125 = -89.875 - 0.25 (last not contain)
    umd_lon = np.arange(-179.875, 180.125, 0.25)

    # search grids
    grids_lat = [grid_shp.loc[i, :].point_geometry.y for i in grid_shp.index]
    grids_lon = [grid_shp.loc[i, :].point_geometry.x for i in grid_shp.index]

    searched_grids_index = search_grids_equal(
        dst_lat=grids_lat, dst_lon=grids_lon, src_lat=umd_lat, src_lon=umd_lon
    )

    # read umd_landcover_classification for each grid
    umd_landcover_classification_grids = []
    major_umd_landcover_classification_grids = []
    for i in tqdm(
        grid_shp.index, colour="green", desc="loop for each grid to extract LC"
    ):
        umd_landcover_classification_grid = []
        searched_grid_index = searched_grids_index[i]
        for j in range(len(umd_landcover_Cv_all)):
            umd_landcover_Cv = umd_landcover_Cv_all[j]
            umd_landcover_classification_grid.append(
                umd_landcover_Cv[searched_grid_index[0][0], searched_grid_index[1][0]]
            )

        umd_landcover_classification_grids.append(umd_landcover_classification_grid)
        major_umd_landcover_classification_grids.append(
            major_umd_landcover_qdeg[
                searched_grid_index[0][0], searched_grid_index[1][0]
            ]
        )

    grid_shp["umd_landcover_classification"] = umd_landcover_classification_grids
    grid_shp["major_umd_landcover_classification_grids"] = (
        major_umd_landcover_classification_grids
    )

    return grid_shp
