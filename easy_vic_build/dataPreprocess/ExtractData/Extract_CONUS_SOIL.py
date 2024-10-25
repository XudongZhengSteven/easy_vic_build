# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import os
import numpy as np
from tqdm import *
from matplotlib import pyplot as plt
from rasterio.plot import show
from ...tools.geo_func import search_grids, resample
from ...tools.geo_func.create_gdf import CreateGDF

""" 
potential issues: NOTE that geographic coordinates are NOT a projection -- scales vary with latitude and are substantially different in the x and y directions, and the 30-arcsec
grid cells are neither square nor equal in area.
"""

CONUS_layers_depths = [0.05, 0.05, 0.10, 0.10, 0.10, 0.20, 0.20, 0.20, 0.50, 0.50, 0.50]  # 11 layers. m


def ExtractData(grid_shp, grid_shp_res=0.125, plot_layer=1, save_original=False, check_search=False):
    # plot_layer: start from 1
    # read
    home = "E:\\data\\LULC\\CONUS-SOIL"
    
    sand_path = os.path.join(home, "SandSiltClayFraction\\sand.bsq\\sand.bsq")
    silt_path = os.path.join(home, "SandSiltClayFraction\\silt.bsq\\silt.bsq")
    clay_path = os.path.join(home, "SandSiltClayFraction\\clay.bsq\\clay.bsq")
    bulk_density_path = os.path.join(home, "BulkDensity\\bd.bsq\\bd.bsq")
    
    # decode
    layers=11
    height=2984
    width=6936
    
    dtype_particle = 'uint8'
    dtype_bulk_density = '<u2'
    
    with open(sand_path, "rb") as f: # %
        sand = np.fromfile(f, dtype=dtype_particle).reshape((layers, height, width))
    
    with open(silt_path, "rb") as f: # %
        silt = np.fromfile(f, dtype=dtype_particle).reshape((layers, height, width))
        
    with open(clay_path, "rb") as f: # %
        clay = np.fromfile(f, dtype=dtype_particle).reshape((layers, height, width))
    
    with open(bulk_density_path, "rb") as f: # kg/m3 * 100 
        bulk_density = np.fromfile(f, dtype=dtype_bulk_density).reshape((layers, height, width))
    
    # set grids_lat, lon
    grids_lat = [grid_shp.loc[i, :].point_geometry.y for i in grid_shp.index]
    grids_lon = [grid_shp.loc[i, :].point_geometry.x for i in grid_shp.index]
    
    # soil data lon, lat
    soil_data_Xmin = -124 - 45/60
    soil_data_Xmax = -66 - 57/60
    soil_data_Ymin = 24 + 32/60
    soil_data_Ymax = 49 + 24/60
    
    soil_lon = np.linspace(soil_data_Xmin, soil_data_Xmax, width)
    soil_lat = np.linspace(soil_data_Ymax, soil_data_Ymin, height)  # large->small, top(90, large lat) is zero
    
    # res
    soil_lat_res = (max(soil_lat) - min(soil_lat)) / (len(soil_lat) - 1)
    soil_lon_res = (max(soil_lon) - min(soil_lon)) / (len(soil_lon) - 1)
    
    # clip: extract before to improve speed
    xindex_start = np.where(soil_lon <= min(grids_lon) - grid_shp_res)[0][-1]
    xindex_end = np.where(soil_lon >= max(grids_lon) + grid_shp_res)[0][0]
    
    yindex_start = np.where(soil_lat >= max(grids_lat) + grid_shp_res)[0][-1]  # large -> small
    yindex_end = np.where(soil_lat <= min(grids_lat) - grid_shp_res)[0][0]
    
    soil_lon_clip = soil_lon[xindex_start: xindex_end+1]
    soil_lat_clip = soil_lat[yindex_start: yindex_end+1]
    
    sand_clip = sand[:, yindex_start: yindex_end+1, xindex_start: xindex_end+1]
    silt_clip = silt[:, yindex_start: yindex_end+1, xindex_start: xindex_end+1]
    clay_clip = clay[:, yindex_start: yindex_end+1, xindex_start: xindex_end+1]
    bulk_density_clip = bulk_density[:, yindex_start: yindex_end+1, xindex_start: xindex_end+1]

    # search grids
    searched_grids_index = search_grids.search_grids_nearest(dst_lat=grids_lat, dst_lon=grids_lon,
                                                             src_lat=soil_lat_clip, src_lon=soil_lon_clip,
                                                             search_num=1)

    # searched_grids_index = search_grids.search_grids_radius_rectangle(dst_lat=grids_lat, dst_lon=grids_lon,
    #                                                                   src_lat=soil_lat_clip, src_lon=soil_lon_clip,
    #                                                                   lat_radius=grid_shp_res/2, lon_radius=grid_shp_res/2)
    
    # loop for layers
    for l in range(layers):
    # read soil data for each grid
        # sand_mean_Value = []
        # silt_mean_Value = []
        # bulk_density_mean_Value = []
        
        sand_nearest_Value = []
        silt_nearest_Value = []
        clay_nearest_Value = []
        bulk_density_nearest_Value = []
        
        if save_original:
            sand_original_Value = []
            silt_original_Value = []
            clay_original_Value = []
            bulk_density_original_Value = []
            
        if l == 0:
            original_lat = []
            original_lon = []

        for i in tqdm(grid_shp.index, colour="green", desc="loop for each grid to extract soil data"):
            # lon/lat
            searched_grid_index = searched_grids_index[i]
            searched_grid_lat = [soil_lat_clip[searched_grid_index[0][j]] for j in range(len(searched_grid_index[0]))]
            searched_grid_lon = [soil_lon_clip[searched_grid_index[1][j]] for j in range(len(searched_grid_index[0]))]
            
            # data
            sand_searched_grid_data = [sand_clip[l, searched_grid_index[0][j], searched_grid_index[1][j]]
                                    for j in range(len(searched_grid_index[0]))]  # index: (lat, lon), namely (row, col)
            silt_searched_grid_data = [silt_clip[l, searched_grid_index[0][j], searched_grid_index[1][j]]
                                    for j in range(len(searched_grid_index[0]))]  # index: (lat, lon), namely (row, col)
            clay_searched_grid_data = [clay_clip[l, searched_grid_index[0][j], searched_grid_index[1][j]]
                                    for j in range(len(searched_grid_index[0]))]  # index: (lat, lon), namely (row, col)
            bulk_density_searched_grid_data = [bulk_density_clip[l, searched_grid_index[0][j], searched_grid_index[1][j]]
                                    for j in range(len(searched_grid_index[0]))]  # index: (lat, lon), namely (row, col)
            
            # resample
            # sand_mean_value = resample.resampleMethod_SimpleAverage(sand_searched_grid_data, searched_grid_lat, searched_grid_lon)
            # silt_mean_value = resample.resampleMethod_SimpleAverage(silt_searched_grid_data, searched_grid_lat, searched_grid_lon)
            # bulk_density_mean_value = resample.resampleMethod_SimpleAverage(bulk_density_searched_grid_data, searched_grid_lat, searched_grid_lon)
            
            # sand_mean_Value.append(sand_mean_value)
            # silt_mean_Value.append(silt_mean_value)
            # bulk_density_mean_Value.append(bulk_density_mean_value)
            
            # check
            if check_search and l+i == 0:
                cgdf = CreateGDF()
                grid_shp_grid = grid_shp.loc[i:i, "geometry"]
                searched_grids_gdf = cgdf.createGDF_rectangle_central_coord(searched_grid_lon, searched_grid_lat, soil_lon_res)
                
                fig, ax = plt.subplots()
                grid_shp_grid.boundary.plot(ax=ax, edgecolor="r", linewidth=2)
                searched_grids_gdf.plot(ax=ax, edgecolor="k", linewidth=0.2, facecolor="b", alpha=0.5)
                ax.set_title("check search")
            
            if sum([sand_searched_grid_data[0], silt_searched_grid_data[0], clay_searched_grid_data[0]]) == 0:
                sand_searched_grid_data[0] = 100/3
                silt_searched_grid_data[0] = 100/3
                clay_searched_grid_data[0] = 100/3
            
            sand_nearest_Value.append(sand_searched_grid_data[0])
            silt_nearest_Value.append(silt_searched_grid_data[0])
            clay_nearest_Value.append(clay_searched_grid_data[0])
            bulk_density_nearest_Value.append(bulk_density_searched_grid_data[0])
            
            if save_original:
                sand_original_Value.append(sand_searched_grid_data)
                silt_original_Value.append(silt_searched_grid_data)
                clay_original_Value.append(clay_searched_grid_data)
                bulk_density_original_Value.append(bulk_density_searched_grid_data)
                
                if l == 0:
                    original_lat.append(searched_grid_lat)
                    original_lon.append(searched_grid_lon)
                
        # save in grid_shp
        # grid_shp[f"soil_l{l+1}_sand_mean_Value"] = sand_mean_Value
        # grid_shp[f"soil_l{l+1}_silt_mean_Value"] = silt_mean_Value
        # grid_shp[f"soil_l{l+1}_bulk_density_mean_Value"] = bulk_density_mean_Value
        grid_shp[f"soil_l{l+1}_sand_nearest_Value"] = np.array(sand_nearest_Value)
        grid_shp[f"soil_l{l+1}_silt_nearest_Value"] = np.array(silt_nearest_Value)
        grid_shp[f"soil_l{l+1}_clay_nearest_Value"] = np.array(clay_nearest_Value)
        grid_shp[f"soil_l{l+1}_bulk_density_nearest_Value"] = np.array(bulk_density_nearest_Value)

        if save_original:
            grid_shp[f"soil_l{l+1}_sand_original_Value"] = sand_original_Value
            grid_shp[f"soil_l{l+1}_silt_original_Value"] = silt_original_Value
            grid_shp[f"soil_l{l+1}_clay_original_Value"] = clay_original_Value
            grid_shp[f"soil_l{l+1}_bulk_density_original_Value"] = bulk_density_original_Value
            
            if l == 0:
                grid_shp["soil_original_lat"] = original_lat
                grid_shp["soil_original_lon"] = original_lon

    # plot
    if plot_layer:
        # original, total 
        plt.figure()
        show(sand[plot_layer-1, :, :], title=f"total_data_sand_l{plot_layer}",
             extent=[soil_lon[0], soil_lon[-1],
                     soil_lat[-1], soil_lat[0]])
        
        plt.figure()
        show(silt[plot_layer-1, :, :], title=f"total_data_silt_l{plot_layer}",
             extent=[soil_lon[0], soil_lon[-1],
                     soil_lat[-1], soil_lat[0]])
        
        plt.figure()
        show(clay[plot_layer-1, :, :], title=f"total_data_clay_l{plot_layer}",
             extent=[soil_lon[0], soil_lon[-1],
                     soil_lat[-1], soil_lat[0]])
        
        plt.figure()
        show(bulk_density[plot_layer-1, :, :], title=f"total_data_bulk_density_l{plot_layer}",
             extent=[soil_lon[0], soil_lon[-1],
                     soil_lat[-1], soil_lat[0]])
        
        # original, clip
        plt.figure()
        show(sand[plot_layer-1, yindex_start: yindex_end+1, xindex_start: xindex_end+1], title=f"clip_data_sand_l{plot_layer}",
             extent=[soil_lon_clip[0], soil_lon_clip[-1],
                     soil_lat_clip[-1], soil_lat_clip[0]])
        
        plt.figure()
        show(silt[plot_layer-1, yindex_start: yindex_end+1, xindex_start: xindex_end+1], title=f"clip_data_silt_l{plot_layer}",
             extent=[soil_lon_clip[0], soil_lon_clip[-1],
                     soil_lat_clip[-1], soil_lat_clip[0]])
        
        plt.figure()
        show(clay[plot_layer-1, yindex_start: yindex_end+1, xindex_start: xindex_end+1], title=f"clip_data_clay_l{plot_layer}",
             extent=[soil_lon_clip[0], soil_lon_clip[-1],
                     soil_lat_clip[-1], soil_lat_clip[0]])
        
        plt.figure()
        show(bulk_density[plot_layer-1, yindex_start: yindex_end+1, xindex_start: xindex_end+1], title=f"clip_data_bulk_density_l{plot_layer}",
             extent=[soil_lon_clip[0], soil_lon_clip[-1],
                     soil_lat_clip[-1], soil_lat_clip[0]])
        
        # readed mean
        fig, ax = plt.subplots()
        grid_shp.plot(f"soil_l{plot_layer}_sand_nearest_Value", ax=ax, edgecolor="k", linewidth=0.2)
        ax.set_title(f"readed nearest sand l{plot_layer}")
        ax.set_xlim([min(grids_lon)-grid_shp_res/2, max(grids_lon)+grid_shp_res/2])
        ax.set_ylim([min(grids_lat)-grid_shp_res/2, max(grids_lat)+grid_shp_res/2])
        
        fig, ax = plt.subplots()
        grid_shp.plot(f"soil_l{plot_layer}_silt_nearest_Value", ax=ax, edgecolor="k", linewidth=0.2)
        ax.set_title(f"readed nearest silt l{plot_layer}")
        ax.set_xlim([min(grids_lon)-grid_shp_res/2, max(grids_lon)+grid_shp_res/2])
        ax.set_ylim([min(grids_lat)-grid_shp_res/2, max(grids_lat)+grid_shp_res/2])
        
        fig, ax = plt.subplots()
        grid_shp.plot(f"soil_l{plot_layer}_clay_nearest_Value", ax=ax, edgecolor="k", linewidth=0.2)
        ax.set_title(f"readed nearest clay l{plot_layer}")
        ax.set_xlim([min(grids_lon)-grid_shp_res/2, max(grids_lon)+grid_shp_res/2])
        ax.set_ylim([min(grids_lat)-grid_shp_res/2, max(grids_lat)+grid_shp_res/2])
        
        fig, ax = plt.subplots()
        grid_shp.plot(f"soil_l{plot_layer}_bulk_density_nearest_Value", ax=ax, edgecolor="k", linewidth=0.2)
        ax.set_title(f"readed nearest bulk_density l{plot_layer}")
        ax.set_xlim([min(grids_lon)-grid_shp_res/2, max(grids_lon)+grid_shp_res/2])
        ax.set_ylim([min(grids_lat)-grid_shp_res/2, max(grids_lat)+grid_shp_res/2])

    return grid_shp


def get_grids_coord_from_COUNS_SOIL(): # 1km
    # soil data lon, lat
    height=2984
    width=6936
    
    soil_data_Xmin = -124 - 45/60
    soil_data_Xmax = -66 - 57/60
    soil_data_Ymin = 24 + 32/60
    soil_data_Ymax = 49 + 24/60
    
    soil_lon = np.linspace(soil_data_Xmin, soil_data_Xmax, width)
    soil_lat = np.linspace(soil_data_Ymax, soil_data_Ymin, height)  # large->small, top(90, large lat) is zero
    
    soil_res = soil_lon[1] - soil_lon[0]
    
    return soil_lon, soil_lat, soil_res


if __name__ == "__main__":
    home = "E:\\data\\LULC\\CONUS-SOIL"
    x = 1
    pass