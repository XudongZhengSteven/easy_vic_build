# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import numpy as np
from matplotlib import pyplot as plt
from tqdm import *
import rasterio
from rasterio.plot import show
import geopandas as gpd
from ...geo_func import search_grids, resample
from ...geo_func.create_gdf import CreateGDF

"""     
        0.0	water
		1.0	Evergreen Needleleaf Forest
		2.0	Evergreen Broadleaf Forest
		3.0	Deciduous Needleleaf Forest
		4.0	Deciduous Broadleaf Forest
		5.0	Mixed Forest
		6.0	Woodland
		7.0	Wooded Grassland
		8.0	Closed Shrubland
		9.0	Open Shrubland
		10.0	Grassland
		11.0	Cropland
		12.0	Bare Ground
		13.0	Urban and Built-up
  
"""

def ExtractData(grid_shp, grid_shp_res=0.125, plot=True, save_original=False, check_search=False):
    # read landcover, classes: 0-14
    umd_landcover_1km_path = "E:\\data\\LULC\\UMD_landcover_classification\\UMD_GLCF_GLCDS_data\\differentFormat\\data.tiff"
    
    # read
    with rasterio.open(umd_landcover_1km_path, mode="r") as dataset:
        # read array
        data_lulc = dataset.read(1)
    
        # umd lat lon
        width = dataset.width
        height = dataset.height
        
        umd_lon = [dataset.xy(0, i)[0] for i in range(width)]
        umd_lat = [dataset.xy(i, 0)[1] for i in range(height)]
        # test: row, column = dataset.index(umd_lon[103], umd_lat[95])
    
    umd_lat_res = (max(umd_lat) - min(umd_lat)) / (len(umd_lat) - 1)
    umd_lon_res = (max(umd_lon) - min(umd_lon)) / (len(umd_lon) - 1)
    
    # set grids_lat, lon
    grids_lat = [grid_shp.loc[i, :].point_geometry.y for i in grid_shp.index]
    grids_lon = [grid_shp.loc[i, :].point_geometry.x for i in grid_shp.index]
    
    # clip: extract before to improve speed
    xindex_start = np.where(umd_lon <= min(grids_lon) - grid_shp_res)[0][-1]
    xindex_end = np.where(umd_lon >= max(grids_lon) + grid_shp_res)[0][0]
    
    yindex_start = np.where(umd_lat >= max(grids_lat) + grid_shp_res)[0][-1]  # large -> small
    yindex_end = np.where(umd_lat <= min(grids_lat) - grid_shp_res)[0][0]
    
    data_lulc_clip = data_lulc[yindex_start: yindex_end+1, xindex_start: xindex_end+1]
    umd_lon_clip = umd_lon[xindex_start: xindex_end+1]
    umd_lat_clip = umd_lat[yindex_start: yindex_end+1]

    # search grids
    # searched_grids_index = search_grids.search_grids_nearest(dst_lat=grids_lat, dst_lon=grids_lon, 
    #                                                          src_lat=umd_lat_clip, src_lon=umd_lon_clip,
    #                                                          search_num=1)
    print("========== search grids for UMD 1km ==========")
    searched_grids_index = search_grids.search_grids_radius_rectangle(dst_lat=grids_lat, dst_lon=grids_lon,
                                                                      src_lat=umd_lat_clip, src_lon=umd_lon_clip,
                                                                      lat_radius=grid_shp_res/2, lon_radius=grid_shp_res/2)

    # read umd_landcover_classification for each grid
    # umd_lc_nearest_Value = []
    umd_lc_major_Value = []
    if save_original:
        original_Value = []
        original_lat = []
        original_lon = []
        original_Cv = []
    
    for i in tqdm(grid_shp.index, colour="green", desc="loop for each grid to extract LC"):
        searched_grid_index = searched_grids_index[i]
        searched_grid_lat = [umd_lat_clip[searched_grid_index[0][j]] for j in range(len(searched_grid_index[0]))]
        searched_grid_lon = [umd_lon_clip[searched_grid_index[1][j]] for j in range(len(searched_grid_index[0]))]        
        searched_grid_data = [data_lulc_clip[searched_grid_index[0][j], searched_grid_index[1][j]]
                              for j in range(len(searched_grid_index[0]))]  # index: (lat, lon), namely (row, col)
        
        major_value = resample.resampleMethod_Majority(searched_grid_data, searched_grid_lat, searched_grid_lon)
        
        # cal cv
        cgdf = CreateGDF()
        grid_shp_grid = grid_shp.loc[i:i, "geometry"]
        grid_shp_gdf = grid_shp.loc[i:i, :]
        searched_grids_gdf = cgdf.createGDF_rectangle_central_coord(searched_grid_lon, searched_grid_lat, umd_lon_res)
        
        overlay_grids = gpd.overlay(grid_shp_gdf, searched_grids_gdf, how="intersection")
        grid_shp_area = grid_shp_gdf.area[i]
        overlay_grids_area = [overlay_grids.area[oi] for oi in overlay_grids.index]
        
        Cv = [a/grid_shp_area for a in overlay_grids_area]
        
        # check
        if check_search and i == 0:
            fig, ax = plt.subplots()
            grid_shp_grid.boundary.plot(ax=ax, edgecolor="r", linewidth=2)
            searched_grids_gdf.plot(ax=ax, edgecolor="k", linewidth=0.2, facecolor="b", alpha=0.5)
        
        # umd_lc_nearest_Value.append(searched_grid_data[0])
        umd_lc_major_Value.append(major_value)
        
        if save_original:
            original_Value.append(searched_grid_data)
            original_lat.append(searched_grid_lat)
            original_lon.append(searched_grid_lon)
            original_Cv.append(Cv)

    # save in grid_shp
    # grid_shp["umd_lc_nearest_Value"] = np.array(umd_lc_nearest_Value)
    grid_shp["umd_lc_major_Value"] = umd_lc_major_Value

    if save_original:
        grid_shp["umd_lc_original_Value"] = original_Value
        grid_shp["umd_lc_original_lat"] = original_lat
        grid_shp["umd_lc_original_lon"] = original_lon
        grid_shp["umd_lc_original_Cv"] = original_Cv

    # plot
    if plot:
        # original, total 
        plt.figure()
        show(data_lulc, title="total_data_lulc",
             extent=[umd_lon[0], umd_lon[-1],
                     umd_lat[-1], umd_lat[0]])
        
        # original, clip
        plt.figure()
        show(data_lulc_clip, title="clip_data_lulc",
             extent=[umd_lon_clip[0], umd_lon_clip[-1],
                     umd_lat_clip[-1], umd_lat_clip[0]])
        
        # readed major
        fig, ax = plt.subplots()
        grid_shp.plot("umd_lc_major_Value", ax=ax, edgecolor="k", linewidth=0.2)
        ax.set_title("readed major")
        
        ax.set_xlim([min(grids_lon)-grid_shp_res/2, max(grids_lon)+grid_shp_res/2])  # [-80.6344, -80.10127999999995]
        ax.set_ylim([min(grids_lat)-grid_shp_res/2, max(grids_lat)+grid_shp_res/2])  # [36.79361, 37.10182000000003]

    return grid_shp


def get_grids_coord_from_UMDLand(): # 1km
        # read landcover, classes: 0-14
    umd_landcover_1km_path = "E:\\data\\LULC\\UMD_landcover_classification\\UMD_GLCF_GLCDS_data\\differentFormat\\data.tiff"
    
    # read
    with rasterio.open(umd_landcover_1km_path, mode="r") as dataset:
        # umd lat lon
        width = dataset.width
        height = dataset.height
        
        umd_lon = [dataset.xy(0, i)[0] for i in range(width)]
        umd_lat = [dataset.xy(i, 0)[1] for i in range(height)]
        
    umd_res = umd_lon[1] - umd_lon[0]

    return umd_lon, umd_lat, umd_res


if __name__ == "__main__":
    pass
    # umd_landcover_1km_path = "E:\\data\\LULC\\UMD_landcover_classification\\UMD_GLCF_GLCDS_data\\differentFormat\\data.tiff"
    
    # dataset = rasterio.open(umd_landcover_1km_path, mode="r")
    
    