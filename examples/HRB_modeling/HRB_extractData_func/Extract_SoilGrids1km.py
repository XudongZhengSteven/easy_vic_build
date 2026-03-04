# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
import numpy as np
from matplotlib import pyplot as plt
import rasterio
from rasterio.plot import show
from tqdm import *

from easy_vic_build.tools.geo_func import search_grids
from easy_vic_build.tools.geo_func.create_gdf import CreateGDF
from easy_vic_build.tools.geo_func.reproject import reproject_raster
from easy_vic_build.tools.geo_func.fill_gap import nearest_neighbor_fill
from easy_vic_build.tools.params_func.TransferFunction import SoilLayerResampler
from easy_vic_build import logger

SoilGrids1km_layers_depths = np.array([
    0.05,
    0.10,
    0.15,
    0.30,
    0.40,
    1.00,
])  # 6 layers m, 0~5cm，5~15cm，15~30cm，30~60cm，60~100cm，100~200cm


SoilGrids_soillayerresampler = SoilLayerResampler(SoilGrids1km_layers_depths)

def set_g_params_soilGrids_layer(g_params):
    g_params["soil_layers_breakpoints"] = {
        "default": [1, 5],
        "boundary": [[1, 3], [2, 5]],
        "type": int,
        "optimal": [None, None],
        "free": True,
    }
    return g_params

def ExtractData(
    grid_shp,
    grid_shp_res=0.125,
    plot_layer=True,
    check_search=False,
):
    # plot_layer: start from 1
    layer_names = [
        "0-5cm",
        "5-15cm",
        "15-30cm",
        "30-60cm",
        "60-100cm",
        "100-200cm"
    ]
    
    # read
    home = "E:\\data\\LULC\\SoilGrids\\data_aggregated\\1000m"

    silt_paths = [os.path.join(home, f"silt\\silt_{layer_name}_mean_1000_WGS84.tif") for layer_name in layer_names]
    clay_paths = [os.path.join(home, f"clay\\clay_{layer_name}_mean_1000_WGS84.tif") for layer_name in layer_names]
    sand_paths = [os.path.join(home, f"sand\\sand_{layer_name}_mean_1000_WGS84.tif") for layer_name in layer_names]
    bulk_density_paths = [os.path.join(home, f"bdod\\bdod_{layer_name}_mean_1000_WGS84.tif") for layer_name in layer_names]
    
    # set grids_lat, lon
    grids_lat = grid_shp.point_geometry.y.to_list()
    grids_lon = grid_shp.point_geometry.x.to_list()
    # grids_lat = [grid_shp.loc[i, :].point_geometry.y for i in grid_shp.index]
    # grids_lon = [grid_shp.loc[i, :].point_geometry.x for i in grid_shp.index]
    
    # soil lat lon res
    logger.info("reading soil data to get lat lon res and clip... ...")
    with rasterio.open(silt_paths[0], mode="r") as src:
        src_transform = src.transform
        width = src.width
        height = src.height
        
        ul = src_transform * (0, 0)
        lr = src_transform * (width, height)

        soilGrids_lon = np.linspace(ul[0], lr[0], width)
        soilGrids_lat = np.linspace(ul[1], lr[1], height)  # large -> small

        soilGrids_lat_res = (max(soilGrids_lat) - min(soilGrids_lat)) / (len(soilGrids_lat) - 1)  # 1km
        soilGrids_lon_res = (max(soilGrids_lon) - min(soilGrids_lon)) / (len(soilGrids_lon) - 1)      
        
        # clip: extract before to improve speed
        xindex_start = np.where(soilGrids_lon <= min(grids_lon) - grid_shp_res)[0][-1]
        xindex_end = np.where(soilGrids_lon >= max(grids_lon) + grid_shp_res)[0][0]

        yindex_start = np.where(soilGrids_lat >= max(grids_lat) + grid_shp_res)[0][-1]  # large -> small
        yindex_end = np.where(soilGrids_lat <= min(grids_lat) - grid_shp_res)[0][0]
        
        soilGrids_lon_clip = soilGrids_lon[xindex_start : xindex_end + 1]
        soilGrids_lat_clip = soilGrids_lat[yindex_start : yindex_end + 1]

    # search grids
    logger.info("searching grids for soil data... ...")
    
    # level0: 1km, source data res: 1km
    searched_grids_index = search_grids.search_grids_nearest(
        dst_lat=grids_lat,
        dst_lon=grids_lon,
        src_lat=soilGrids_lat_clip,
        src_lon=soilGrids_lon_clip,
        search_num=1,
    )
    
    for l in range(len(layer_names)):
        silt_nearest_Value = []
        clay_nearest_Value = []
        sand_nearest_Value = []
        bulk_density_nearest_Value = []
        
        # read clip data
        with rasterio.open(silt_paths[l], mode="r") as dataset:
            silt_data = dataset.read(1, masked=True)
            silt_data = silt_data[yindex_start : yindex_end + 1, xindex_start : xindex_end + 1]
        
        with rasterio.open(clay_paths[l], mode="r") as dataset:
            clay_data = dataset.read(1, masked=True)
            clay_data = clay_data[yindex_start : yindex_end + 1, xindex_start : xindex_end + 1]
        
        with rasterio.open(sand_paths[l], mode="r") as dataset:
            sand_data = dataset.read(1, masked=True)
            sand_data = sand_data[yindex_start : yindex_end + 1, xindex_start : xindex_end + 1]
        
        with rasterio.open(bulk_density_paths[l], mode="r") as dataset:
            bulk_density_data = dataset.read(1, masked=True)
            bulk_density_data = bulk_density_data[yindex_start : yindex_end + 1, xindex_start : xindex_end + 1]
        
        # fill gap
        silt_data_filled = nearest_neighbor_fill(silt_data)
        clay_data_filled = nearest_neighbor_fill(clay_data)
        sand_data_filled = nearest_neighbor_fill(sand_data)
        bulk_density_data_filled = nearest_neighbor_fill(bulk_density_data)
        
        for i in tqdm(
            grid_shp.index,
            colour="green",
            desc=f"loop for each grid to extract soil{l} data",
            leave=False,
        ):
            # searched lon/lat, index
            searched_grid_index = searched_grids_index[i]
            searched_grid_lat = [
                soilGrids_lat_clip[searched_grid_index[0][j]]
                for j in range(len(searched_grid_index[0]))
            ]
            
            searched_grid_lon = [
                soilGrids_lon_clip[searched_grid_index[1][j]]
                for j in range(len(searched_grid_index[0]))
            ]
            
            # searched data
            silt_searched_grid_data = [
                silt_data_filled[searched_grid_index[0][j], searched_grid_index[1][j]]
                for j in range(len(searched_grid_index[0]))
            ]
            clay_searched_grid_data = [
                clay_data_filled[searched_grid_index[0][j], searched_grid_index[1][j]]
                for j in range(len(searched_grid_index[0]))
            ]
            sand_searched_grid_data = [
                sand_data_filled[searched_grid_index[0][j], searched_grid_index[1][j]]
                for j in range(len(searched_grid_index[0]))
            ]
            bulk_density_searched_grid_data = [
                bulk_density_data_filled[searched_grid_index[0][j], searched_grid_index[1][j]
                ]
                for j in range(len(searched_grid_index[0]))
            ]
            
            # check not valid value
            # case1: > 100 or < 100
            # case2: < 0 (-nodata)
            # check_valid_value = silt_searched_grid_data[0] + clay_searched_grid_data[0] + sand_searched_grid_data[0]
            # if abs(check_valid_value - 1000) != 0:
            #     logger.warning(f"texture sum ({check_valid_value}) is not 1000")
            
            # check
            if check_search and l + i == 0:
                cgdf = CreateGDF()
                grid_shp_grid = grid_shp.loc[[i], "geometry"]
                searched_grids_gdf = cgdf.createGDF_rectangle_central_coord(
                    searched_grid_lon, searched_grid_lat, soilGrids_lon_res
                )

                fig, ax = plt.subplots()
                grid_shp_grid.boundary.plot(ax=ax, edgecolor="r", linewidth=2)  # target grid
                searched_grids_gdf.plot(
                    ax=ax, edgecolor="k", linewidth=0.2, facecolor="b", alpha=0.5
                )  # searched grids from source data
                ax.set_title("check search")
            
            # resample: nearest
            sand_searched_resample_data = sand_searched_grid_data[0]
            silt_searched_resample_data = silt_searched_grid_data[0]
            clay_searched_resample_data = clay_searched_grid_data[0]
            bulk_density_searched_resample_data = bulk_density_searched_grid_data[0]
            
            # append
            sand_nearest_Value.append(sand_searched_resample_data)
            silt_nearest_Value.append(silt_searched_resample_data)
            clay_nearest_Value.append(clay_searched_resample_data)
            bulk_density_nearest_Value.append(bulk_density_searched_resample_data)
        
        # save in grid_shp
        grid_shp[f"soil_l{l+1}_sand_nearest_Value"] = np.array(sand_nearest_Value)
        grid_shp[f"soil_l{l+1}_silt_nearest_Value"] = np.array(silt_nearest_Value)
        grid_shp[f"soil_l{l+1}_clay_nearest_Value"] = np.array(clay_nearest_Value)
        grid_shp[f"soil_l{l+1}_bulk_density_nearest_Value"] = np.array(
            bulk_density_nearest_Value
        )

    # plot
    if plot_layer:
        # source data
        # read clip data
        with rasterio.open(silt_paths[plot_layer], mode="r") as dataset:
            silt_data = dataset.read(1, masked=True)
            silt_data = silt_data[yindex_start : yindex_end + 1, xindex_start : xindex_end + 1]
        
        with rasterio.open(clay_paths[plot_layer], mode="r") as dataset:
            clay_data = dataset.read(1, masked=True)
            clay_data = clay_data[yindex_start : yindex_end + 1, xindex_start : xindex_end + 1]
        
        with rasterio.open(sand_paths[plot_layer], mode="r") as dataset:
            sand_data = dataset.read(1, masked=True)
            sand_data = sand_data[yindex_start : yindex_end + 1, xindex_start : xindex_end + 1]
        
        with rasterio.open(bulk_density_paths[plot_layer], mode="r") as dataset:
            bulk_density_data = dataset.read(1, masked=True)
            bulk_density_data = bulk_density_data[yindex_start : yindex_end + 1, xindex_start : xindex_end + 1]
        
        # fill gap
        silt_data_filled = nearest_neighbor_fill(silt_data)
        clay_data_filled = nearest_neighbor_fill(clay_data)
        sand_data_filled = nearest_neighbor_fill(sand_data)
        bulk_density_data_filled = nearest_neighbor_fill(bulk_density_data)
        
        
        # plot original data：sand
        show(
            sand_data_filled,
            title=f"original data, sand_data_filled",
            extent=[
                soilGrids_lon_clip[0],
                soilGrids_lon_clip[-1],
                soilGrids_lat_clip[-1],
                soilGrids_lat_clip[0],
            ],
        )
        
        # plot readed nearest: sand
        fig, ax = plt.subplots()
        grid_shp.plot(
            f"soil_l{plot_layer}_sand_nearest_Value",
            ax=ax,
            edgecolor="k",
            linewidth=0.2,
        )
        ax.set_title(f"readed nearest sand l{plot_layer}")
        ax.set_xlim(
            [min(grids_lon) - grid_shp_res / 2, max(grids_lon) + grid_shp_res / 2]
        )
        ax.set_ylim(
            [min(grids_lat) - grid_shp_res / 2, max(grids_lat) + grid_shp_res / 2]
        )

        # plot original data：silt
        show(
            silt_data_filled,
            title=f"original data, silt_data_filled",
            extent=[
                soilGrids_lon_clip[0],
                soilGrids_lon_clip[-1],
                soilGrids_lat_clip[-1],
                soilGrids_lat_clip[0],
            ],
        )
        
        # plot readed nearest: silt
        fig, ax = plt.subplots()
        grid_shp.plot(
            f"soil_l{plot_layer}_silt_nearest_Value",
            ax=ax,
            edgecolor="k",
            linewidth=0.2,
        )
        ax.set_title(f"readed nearest silt l{plot_layer}")
        ax.set_xlim(
            [min(grids_lon) - grid_shp_res / 2, max(grids_lon) + grid_shp_res / 2]
        )
        ax.set_ylim(
            [min(grids_lat) - grid_shp_res / 2, max(grids_lat) + grid_shp_res / 2]
        )

        # plot original data：clay
        show(
            clay_data_filled,
            title=f"original data, clay_data_filled",
            extent=[
                soilGrids_lon_clip[0],
                soilGrids_lon_clip[-1],
                soilGrids_lat_clip[-1],
                soilGrids_lat_clip[0],
            ],
        )
        
        # plot readed nearest: clay
        fig, ax = plt.subplots()
        grid_shp.plot(
            f"soil_l{plot_layer}_clay_nearest_Value",
            ax=ax,
            edgecolor="k",
            linewidth=0.2,
        )
        ax.set_title(f"readed nearest clay l{plot_layer}")
        ax.set_xlim(
            [min(grids_lon) - grid_shp_res / 2, max(grids_lon) + grid_shp_res / 2]
        )
        ax.set_ylim(
            [min(grids_lat) - grid_shp_res / 2, max(grids_lat) + grid_shp_res / 2]
        )

        # plot original data：bd
        show(
            bulk_density_data_filled,
            title=f"original data, bulk_density_data_filled",
            extent=[
                soilGrids_lon_clip[0],
                soilGrids_lon_clip[-1],
                soilGrids_lat_clip[-1],
                soilGrids_lat_clip[0],
            ],
        )
        
        # plot readed nearest: bd
        fig, ax = plt.subplots()
        grid_shp.plot(
            f"soil_l{plot_layer}_bulk_density_nearest_Value",
            ax=ax,
            edgecolor="k",
            linewidth=0.2,
        )
        ax.set_title(f"readed nearest bulk_density l{plot_layer}")
        ax.set_xlim(
            [min(grids_lon) - grid_shp_res / 2, max(grids_lon) + grid_shp_res / 2]
        )
        ax.set_ylim(
            [min(grids_lat) - grid_shp_res / 2, max(grids_lat) + grid_shp_res / 2]
        )
            
    return grid_shp   

    
def reproject_soilGrids1km(dst_crs="EPSG:4326"):
    # plot_layer: start from 1
    layer_names = [
        "0-5cm",
        "5-15cm",
        "15-30cm",
        "30-60cm",
        "60-100cm",
        "100-200cm"
    ]
    
    # read
    home = "E:\\data\\LULC\\SoilGrids\\data_aggregated\\1000m"

    src_silt_paths = [os.path.join(home, f"silt\\silt_{layer_name}_mean_1000.tif") for layer_name in layer_names]
    src_clay_paths = [os.path.join(home, f"clay\\clay_{layer_name}_mean_1000.tif") for layer_name in layer_names]
    src_sand_paths = [os.path.join(home, f"sand\\sand_{layer_name}_mean_1000.tif") for layer_name in layer_names]
    src_bulk_density_paths = [os.path.join(home, f"bdod\\bdod_{layer_name}_mean_1000.tif") for layer_name in layer_names]
    
    dst_silt_paths = [os.path.join(home, f"silt\\silt_{layer_name}_mean_1000_WGS84.tif") for layer_name in layer_names]
    dst_clay_paths = [os.path.join(home, f"clay\\clay_{layer_name}_mean_1000_WGS84.tif") for layer_name in layer_names]
    dst_sand_paths = [os.path.join(home, f"sand\\sand_{layer_name}_mean_1000_WGS84.tif") for layer_name in layer_names]
    dst_bulk_density_paths = [os.path.join(home, f"bdod\\bdod_{layer_name}_mean_1000_WGS84.tif") for layer_name in layer_names]
    
    for l in range(len(layer_names)):
        logger.info(f"reproject soil data layer {layer_names[l]}: silt... ...")
        reproject_raster(
            src_silt_paths[l],
            dst_silt_paths[l],
            dst_crs
        )

        logger.info(f"reproject soil data layer {layer_names[l]}: clay... ...")
        reproject_raster(
            src_clay_paths[l],
            dst_clay_paths[l],
            dst_crs
        )
        
        logger.info(f"reproject soil data layer {layer_names[l]}: sand... ...")
        reproject_raster(
            src_sand_paths[l],
            dst_sand_paths[l],
            dst_crs
        )

        logger.info(f"reproject soil data layer {layer_names[l]}: bulk_density... ...")
        reproject_raster(
            src_bulk_density_paths[l],
            dst_bulk_density_paths[l],
            dst_crs
        )


if __name__  == "__main__":
    # reproject_soilGrids1km(dst_crs="EPSG:4326")
    pass
