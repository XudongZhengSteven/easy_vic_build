# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import rasterio
from rasterio.plot import show
from rasterio.enums import Resampling
from tqdm import *
import numpy as np
import matplotlib.pyplot as plt
from ...tools.geo_func import search_grids, resample
from ...tools.geo_func.create_gdf import CreateGDF


def ExtractData(grid_shp, grid_shp_res=0.25, plot=True, save_original=False, check_search=False):
    # grid_shp_res
    grid_shp_res_m = grid_shp_res * 111.32 * 1000
    
    # read dem data
    SrtmDEM_path = "E:/data/LULC/DEM/SRTM/US/Combine/srtm_11_03.tif"
    SrtmDEM = rasterio.open(SrtmDEM_path)
    SrtmDEM_data = SrtmDEM.read(1)
    
    # downscale to increase process speed, requirement: SrtmDEM_downscale.res < target.res
    downscale_factor = 1/8  # mannual set
    SrtmDEM_downscale = SrtmDEM.read(1,
                                     out_shape=(int(SrtmDEM.height * downscale_factor),
                                                int(SrtmDEM.width * downscale_factor)
                                                ),
                                     resampling=Resampling.average
                                     )

    transform = SrtmDEM.transform * SrtmDEM.transform.scale(
        (SrtmDEM.width / SrtmDEM_downscale.shape[-1]),
        (SrtmDEM.height / SrtmDEM_downscale.shape[-2])
    )

    # set grids_lat, lon
    grids_lat = [grid_shp.loc[i, :].point_geometry.y for i in grid_shp.index]
    grids_lon = [grid_shp.loc[i, :].point_geometry.x for i in grid_shp.index]
    
    # SrtmDEM grids, corresponding to the array index of data
    ul = SrtmDEM.transform * (0, 0)
    lr = SrtmDEM.transform * (SrtmDEM.shape[1], SrtmDEM.shape[0])

    SrtmDEM_lon = np.linspace(ul[0], lr[0], SrtmDEM.shape[1])
    SrtmDEM_lat = np.linspace(ul[1], lr[1], SrtmDEM.shape[0])  # large -> small
    
    # res
    SrtmDEM_lat_res = (max(SrtmDEM_lat) - min(SrtmDEM_lat)) / (len(SrtmDEM_lat) - 1)
    SrtmDEM_lon_res = (max(SrtmDEM_lat) - min(SrtmDEM_lat)) / (len(SrtmDEM_lat) - 1)
    
    # clip: extract before to improve speed
    xindex_start = np.where(SrtmDEM_lon <= min(grids_lon) - grid_shp_res)[0][-1]
    xindex_end = np.where(SrtmDEM_lon >= max(grids_lon) + grid_shp_res)[0][0]
    
    yindex_start = np.where(SrtmDEM_lat >= max(grids_lat) + grid_shp_res)[0][-1]  # large -> small
    yindex_end = np.where(SrtmDEM_lat <= min(grids_lat) - grid_shp_res)[0][0]
    
    SrtmDEM_data_clip = SrtmDEM_data[yindex_start: yindex_end+1, xindex_start: xindex_end+1]
    SrtmDEM_lon_clip = SrtmDEM_lon[xindex_start: xindex_end+1]
    SrtmDEM_lat_clip = SrtmDEM_lat[yindex_start: yindex_end+1]
    
    # close
    SrtmDEM.close()
    
    # search SrtmDEM grids for each grid in grid_shp
    print("========== search grids for SrtmDEM ==========")
    searched_grids_index = search_grids.search_grids_radius_rectangle(
        dst_lat=grids_lat, dst_lon=grids_lon, src_lat=SrtmDEM_lat_clip, src_lon=SrtmDEM_lon_clip,
        lat_radius=grid_shp_res/2, lon_radius=grid_shp_res/2)

    # resample for mean
    SrtmDEM_mean_Value = []
    SrtmDEM_std_Value = []
    SrtmDEM_mean_slope_Value = []
    if save_original:
        original_Value = []
        original_lat = []
        original_lon = []
        
    for i in tqdm(range(len(searched_grids_index)), desc="loop for grids extract SrtmDEM", colour="g"):
        searched_grid_index = searched_grids_index[i]
        searched_grid_lat = [SrtmDEM_lat_clip[searched_grid_index[0][j]] for j in range(len(searched_grid_index[0]))]
        searched_grid_lon = [SrtmDEM_lon_clip[searched_grid_index[1][j]] for j in range(len(searched_grid_index[0]))]
        searched_grid_data = [SrtmDEM_data_clip[searched_grid_index[0][j], searched_grid_index[1][j]]
                              for j in range(len(searched_grid_index[0]))]  # index: (lat, lon), namely (row, col)
        
        mean_value = resample.resampleMethod_GeneralFunction(
            searched_grid_data, searched_grid_lat, searched_grid_lon, None, None,
            general_function=np.mean, missing_value=32767)
        
        std_value = resample.resampleMethod_GeneralFunction(
            searched_grid_data, searched_grid_lat, searched_grid_lon, None, None,
            general_function=np.std, missing_value=32767)
        
        SrtmDEM_mean_slope_value = (max(searched_grid_data) - min(searched_grid_data)) / ((2 * grid_shp_res_m **2) ** 0.5) * 100
        
        SrtmDEM_mean_Value.append(mean_value)
        SrtmDEM_std_Value.append(std_value)
        SrtmDEM_mean_slope_Value.append(SrtmDEM_mean_slope_value)
        
        # check
        if check_search and i == 0:
            cgdf = CreateGDF()
            grid_shp_grid = grid_shp.loc[i:i, "geometry"]
            searched_grids_gdf = cgdf.createGDF_rectangle_central_coord(searched_grid_lon, searched_grid_lat, SrtmDEM_lat_res)
            
            fig, ax = plt.subplots()
            grid_shp_grid.boundary.plot(ax=ax, edgecolor="r", linewidth=2)
            searched_grids_gdf.plot(ax=ax, edgecolor="k", linewidth=0.2, facecolor="b", alpha=0.5)
            ax.set_title("check search")
        
        if save_original:
            original_Value.append(searched_grid_data)
            original_lat.append(searched_grid_lat)
            original_lon.append(searched_grid_lon)

    # set missing_value as none
    SrtmDEM_mean_Value = np.array(SrtmDEM_mean_Value)
    SrtmDEM_mean_Value[SrtmDEM_mean_Value == 32767] = np.NAN
    SrtmDEM_std_Value = np.array(SrtmDEM_std_Value)
    SrtmDEM_mean_slope_Value = np.array(SrtmDEM_mean_slope_Value)

    if save_original:
        for i in range(len(original_Value)):
            original_Value_grid = original_Value[i]
            original_Value_grid = np.array(original_Value_grid, float)
            original_Value_grid[original_Value_grid == 32767] = np.NAN
            original_Value[i] = original_Value_grid.tolist()

    # save in grid_shp
    grid_shp["SrtmDEM_mean_Value"] = SrtmDEM_mean_Value
    grid_shp["SrtmDEM_std_Value"] = SrtmDEM_std_Value
    grid_shp["SrtmDEM_mean_slope_Value%"] = SrtmDEM_mean_slope_Value

    if save_original:
        grid_shp["SrtmDEM_original_Value"] = original_Value
        grid_shp["SrtmDEM_original_lat"] = original_lat
        grid_shp["SrtmDEM_original_lon"] = original_lon
    
    # plot
    if plot:
        # original, total, check lat corresponding to the array
        plt.figure()
        show(SrtmDEM_downscale, title="total_SrtmDEM_downscale",
             extent=[SrtmDEM_lon[0], SrtmDEM_lon[-1],
                     SrtmDEM_lat[-1], SrtmDEM_lat[0]])
        
        # original, clip
        plt.figure()
        show(SrtmDEM_data_clip, title="clip_SrtmDEM",
             extent=[SrtmDEM_lon_clip[0], SrtmDEM_lon_clip[-1],
                     SrtmDEM_lat_clip[-1], SrtmDEM_lat_clip[0]])
        
        # readed mean
        fig, ax = plt.subplots()
        grid_shp.plot("SrtmDEM_mean_Value", ax=ax, edgecolor="k", linewidth=0.2)
        ax.set_title("readed mean")
        ax.set_xlim([min(grids_lon)-grid_shp_res/2, max(grids_lon)+grid_shp_res/2])
        ax.set_ylim([min(grids_lat)-grid_shp_res/2, max(grids_lat)+grid_shp_res/2])

        # readed std
        fig, ax = plt.subplots()
        grid_shp.plot("SrtmDEM_std_Value", ax=ax, edgecolor="k", linewidth=0.2)
        ax.set_title("readed std")
        ax.set_xlim([min(grids_lon)-grid_shp_res/2, max(grids_lon)+grid_shp_res/2])
        ax.set_ylim([min(grids_lat)-grid_shp_res/2, max(grids_lat)+grid_shp_res/2])
        
        # readed slope
        fig, ax = plt.subplots()
        grid_shp.plot("SrtmDEM_mean_slope_Value%", ax=ax, edgecolor="k", linewidth=0.2)
        ax.set_title("readed mean slope value")
        ax.set_xlim([min(grids_lon)-grid_shp_res/2, max(grids_lon)+grid_shp_res/2])
        ax.set_ylim([min(grids_lat)-grid_shp_res/2, max(grids_lat)+grid_shp_res/2])

    return grid_shp


def get_grids_coord_from_StrmDEM(): # 90m
    # SrtmDEM grids
    # read
    SrtmDEM_path = "E:/data/LULC/DEM/SRTM/US/Combine/srtm_11_03.tif"
    SrtmDEM = rasterio.open(SrtmDEM_path)
    
    ul = SrtmDEM.transform * (0, 0)
    lr = SrtmDEM.transform * (SrtmDEM.shape[1], SrtmDEM.shape[0])

    SrtmDEM_lon = np.linspace(ul[0], lr[0], SrtmDEM.shape[1])
    SrtmDEM_lat = np.linspace(ul[1], lr[1], SrtmDEM.shape[0])
    
    SrtmDEM_res = SrtmDEM_lon[1] - SrtmDEM_lon[0]
    
    return SrtmDEM_lon, SrtmDEM_lat, SrtmDEM_res


def readSrtmDEMAsPCRasterMap():
    # read dem data
    # SrtmDEM_path = "E:/data/LULC/DEM/SRTM/US/Combine/srtm_11_03.tif"
    # SrtmDEM = rasterio.open(SrtmDEM_path)
    pass