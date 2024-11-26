# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
from tqdm import *
import numpy as np
import xarray as xr
from tqdm import *
import matplotlib.pyplot as plt
from rasterio.plot import show
from ...geo_func import search_grids, resample
from ...geo_func.create_gdf import CreateGDF

def ExtractData(grid_shp, grid_shp_res=0.125, plot_layer=False, check_search=False):
    # plot_layer: start from 1
    # general
    home = "E:\\data\\hydrometeorology\\ERA5\\ERA5-Land monthly averaged data from 1950 to present\\data_soilTemperature"
    
    # read data
    stls_data = []
    
    for i in range(1, 5):
        fn =  os.path.join(home, f"Soil_temperature_level{i}_CDS_Beta.grib")
        with xr.open_dataset(fn, engine="cfgrib") as dataset:
            stl_data_ = dataset.variables[f"stl{i}"]
            stl_data_ = np.nanmean(stl_data_, axis=0)
            
            # unit: K->C
            stl_data_ -= 273.15
            
            stls_data.append(stl_data_)
            if i == 1:
                stl_lat = dataset.variables["latitude"].values  # large -> small
                stl_lon = dataset.variables["longitude"].values
                stl_all_layers_mean = stl_data_
            else:
                stl_all_layers_mean += stl_data_
    
    stl_all_layers_mean /= 4
    
    stl_lat_res = (stl_lat.max() - stl_lat.min()) / (len(stl_lat) - 1)
    stl_lon_res = (stl_lon.max() - stl_lon.min()) / (len(stl_lon) - 1)
    
    # set grids_lat, lon
    grids_lat = [grid_shp.loc[i, :].point_geometry.y for i in grid_shp.index]
    grids_lon = [grid_shp.loc[i, :].point_geometry.x for i in grid_shp.index]
    
    # search grids
    print("========== search grids for ERA5 ST ==========")
    searched_grids_index = search_grids.search_grids_radius_rectangle_reverse(dst_lat=grids_lat, dst_lon=grids_lon,
                                                                              src_lat=stl_lat, src_lon=stl_lon,
                                                                              lat_radius=stl_lat_res/2, lon_radius=stl_lon_res/2)
    # searched_grids_index = search_grids.search_grids_nearest(dst_lat=grids_lat, dst_lon=grids_lon,
    #                                                          src_lat=stl_lat, src_lon=stl_lon,
    #                                                          search_num=1,
    #                                                          move_src_lat=None, move_src_lon=None)
    
    # read soil temperature for each grid
    for l in range(1, 5):
        stl_in_src_grid_Value = []
        # stl_IDW_mean_Value = []
        if l == 1:
            stl_all_layers_mean_Value = []
        
        for i in tqdm(grid_shp.index, colour="green", desc=f"loop for each grid to extract ST layer{l}"):
            searched_grid_index = searched_grids_index[i]
            dst_lat_grid = grid_shp.loc[i, :].point_geometry.y
            dst_lon_grid = grid_shp.loc[i, :].point_geometry.x
            searched_grid_lat = [stl_lat[searched_grid_index[0][j]] for j in range(len(searched_grid_index[0]))]
            searched_grid_lon = [stl_lon[searched_grid_index[1][j]] for j in range(len(searched_grid_index[0]))]
            searched_grid_data = [stls_data[l-1][searched_grid_index[0][j], searched_grid_index[1][j]]
                                for j in range(len(searched_grid_index[0]))]  # index: (lat, lon), namely (row, col)
            
            stl_in_src_grid_value = searched_grid_data[0]
            
            # IDW_mean_value = resample.resampleMethod_IDW(searched_grid_data, searched_grid_lat, searched_grid_lon,
            #                                             dst_lat_grid, dst_lon_grid, p=2)
            
                        
            # stl_IDW_mean_Value.append(IDW_mean_value)
            stl_in_src_grid_Value.append(stl_in_src_grid_value)

            if l == 1:
                searched_grid_data_all_layer_mean = [stl_all_layers_mean[searched_grid_index[0][j], searched_grid_index[1][j]]
                                                     for j in range(len(searched_grid_index[0]))]  # index: (lat, lon), namely (row, col)

                # IDW_mean_value_all_layer_mean = resample.resampleMethod_IDW(searched_grid_data_all_layer_mean, searched_grid_lat, searched_grid_lon,
                #                                                             dst_lat_grid, dst_lon_grid, p=2)
                
                stl_all_layers_mean_value = searched_grid_data_all_layer_mean[0]
                
                
                stl_all_layers_mean_Value.append(stl_all_layers_mean_value)
                
                # check
                if check_search and i == 0:
                    cgdf = CreateGDF()
                    grid_shp_grid = grid_shp.loc[i:i, "geometry"]
                    searched_grids_gdf = cgdf.createGDF_rectangle_central_coord(searched_grid_lon, searched_grid_lat, stl_lat_res)
                    
                    fig, ax = plt.subplots()
                    grid_shp_grid.boundary.plot(ax=ax, edgecolor="r", linewidth=2)
                    searched_grids_gdf.plot(ax=ax, edgecolor="k", linewidth=0.2, facecolor="b", alpha=0.5)
                    ax.set_title("check search")
                
        # save
        # grid_shp[f"stl{l}_IDW_mean_Value"] = np.array(stl_IDW_mean_Value)
        grid_shp[f"stl{l}_in_src_grid_Value"] = np.array(stl_in_src_grid_Value)
        
        if l == 1:
            grid_shp[f"stl_all_layers_mean_Value"] = np.array(stl_all_layers_mean_Value)
    
    # plot
    if plot_layer:
        # for clip
        xindex_start = np.where(stl_lon <= min(grids_lon) - grid_shp_res)[0][-1]
        xindex_end = np.where(stl_lon >= max(grids_lon) + grid_shp_res)[0][0]
        
        yindex_start = np.where(stl_lat >= max(grids_lat) + grid_shp_res)[0][-1]  # large -> small
        yindex_end = np.where(stl_lat <= min(grids_lat) - grid_shp_res)[0][0]
        stl_lon_clip = stl_lon[xindex_start: xindex_end+1]
        stl_lat_clip = stl_lat[yindex_start: yindex_end+1]
        
        # original, total
        plt.figure()
        show(stls_data[plot_layer-1], title=f"total_data_stl{plot_layer}",
             extent=[stl_lon[0], stl_lon[-1],
                     stl_lat[-1], stl_lat[0]])
        
        # original, clip
        plt.figure()
        data_stl_clip = stls_data[plot_layer-1][yindex_start: yindex_end+1, xindex_start: xindex_end+1]
        show(data_stl_clip, title=f"clip_data_stl{plot_layer}",
             extent=[stl_lon_clip[0], stl_lon_clip[-1],
                     stl_lat_clip[-1], stl_lat_clip[0]])
        
        # original, clip, all_layer_mean
        plt.figure()
        data_stl_all_layers_mean_clip = stl_all_layers_mean[yindex_start: yindex_end+1, xindex_start: xindex_end+1]
        show(data_stl_all_layers_mean_clip, title=f"clip_data_stl_all_layers_mean",
             extent=[stl_lon_clip[0], stl_lon_clip[-1],
                     stl_lat_clip[-1], stl_lat_clip[0]])
        
        # readed mean
        fig, ax = plt.subplots()
        grid_shp.plot(f"stl{plot_layer}_in_src_grid_Value", ax=ax, edgecolor="k", linewidth=0.2)
        ax.set_title(f"stl{plot_layer}_in_src_grid_Value")
        ax.set_xlim([min(grids_lon)-grid_shp_res/2, max(grids_lon)+grid_shp_res/2])
        ax.set_ylim([min(grids_lat)-grid_shp_res/2, max(grids_lat)+grid_shp_res/2])

        # readed all_layer_mean
        fig, ax = plt.subplots()
        grid_shp.plot(f"stl_all_layers_mean_Value", ax=ax, edgecolor="k", linewidth=0.2)
        ax.set_title(f"stl_all_layers_mean_Value")
        ax.set_xlim([min(grids_lon)-grid_shp_res/2, max(grids_lon)+grid_shp_res/2])
        ax.set_ylim([min(grids_lat)-grid_shp_res/2, max(grids_lat)+grid_shp_res/2])

    return grid_shp
        
if __name__ == "__main__":
    # # general
    # home = "E:\\data\\hydrometeorology\\ERA5\\ERA5-Land monthly averaged data from 1950 to present\\data_soilTemperature"
    # fpath = os.path.join(home, "Soil_temperature_level1_CDS_Beta.grib")
    
    # # read data
    # dataset = xr.open_dataset(fpath, engine="cfgrib")
    pass
    