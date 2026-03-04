# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.plot import show

from easy_vic_build.tools.geo_func import search_grids, resample
from easy_vic_build.tools.geo_func.create_gdf import CreateGDF
from easy_vic_build import logger

from tqdm import tqdm

def ExtractData(
    grid_shp,
    grid_shp_res=0.125,
    plot=True,
    save_original=False,
    check_search=False,
):
    # grid_shp_res
    grid_shp_res_m = grid_shp_res * 111.32 * 1000
    
    # read
    ASTGTM_DEM_path = "F:\\research\\Research\\ModelingUncertainty_hanjiang\\data\\DEM\\ASTGTM2_mosaic_clip.tif"  # 30m
    ASTGTM_DEM = rasterio.open(ASTGTM_DEM_path)
    ASTGTM_DEM_data = ASTGTM_DEM.read(1)
    
    # downscale to increase process speed, requirement: SrtmDEM_downscale.res < target.res
    logger.info("downscaling ASTGTM_DEM data to increase processing speed... ...")
    downscale_factor = 1 / 8  # mannual set
    ASTGTM_DEM_downscale = ASTGTM_DEM.read(
        1,
        out_shape=(
            int(ASTGTM_DEM.height * downscale_factor),
            int(ASTGTM_DEM.width * downscale_factor),
        ),
        resampling=Resampling.average,
    )

    transform = ASTGTM_DEM.transform * ASTGTM_DEM.transform.scale(
        (ASTGTM_DEM.width / ASTGTM_DEM_downscale.shape[-1]),
        (ASTGTM_DEM.height / ASTGTM_DEM_downscale.shape[-2]),
    )
    
    ASTGTM_DEM_data = ASTGTM_DEM_downscale
    
    # set grids_lat, lon
    grids_lat = grid_shp.point_geometry.y.to_list()
    grids_lon = grid_shp.point_geometry.x.to_list()

    # SrtmDEM grids, corresponding to the array index of data
    ul = transform * (0, 0)
    lr = transform * (ASTGTM_DEM_downscale.shape[1], ASTGTM_DEM_downscale.shape[0])

    ASTGTM_DEM_lon = np.linspace(ul[0], lr[0], ASTGTM_DEM_downscale.shape[1])
    ASTGTM_DEM_lat = np.linspace(ul[1], lr[1], ASTGTM_DEM_downscale.shape[0])  # large -> small

    # res
    ASTGTM_DEM_lat_res = (max(ASTGTM_DEM_lat) - min(ASTGTM_DEM_lat)) / (len(ASTGTM_DEM_lat) - 1)  # 30m
    ASTGTM_DEM_lon_res = (max(ASTGTM_DEM_lat) - min(ASTGTM_DEM_lat)) / (len(ASTGTM_DEM_lat) - 1)

    # # clip: extract before to improve speed
    # xindex_start = np.where(ASTGTM_DEM_lon <= min(grids_lon) - grid_shp_res/2)[0][-1]
    # xindex_end = np.where(ASTGTM_DEM_lon >= max(grids_lon) + grid_shp_res/2)[0][0]

    # yindex_start = np.where(ASTGTM_DEM_lat >= max(grids_lat) + grid_shp_res/2)[0][
    #     -1
    # ]  # large -> small
    # yindex_end = np.where(ASTGTM_DEM_lat <= min(grids_lat) - grid_shp_res/2)[0][0]

    # ASTGTM_DEM_data_clip = ASTGTM_DEM_data[
    #     yindex_start : yindex_end + 1, xindex_start : xindex_end + 1
    # ]
    # ASTGTM_DEM_lon_clip = ASTGTM_DEM_lon[xindex_start : xindex_end + 1]
    # ASTGTM_DEM_lat_clip = ASTGTM_DEM_lat[yindex_start : yindex_end + 1]

    # close
    ASTGTM_DEM.close()
    
    # search ASTGTM_DEM grids for each grid in grid_shp
    logger.info("searching grids for DEM data... ...")
    
    # level0: 1km, source data res: 30m
    searched_grids_index = search_grids.search_grids_radius_rectangle(
        dst_lat=grids_lat,
        dst_lon=grids_lon,
        src_lat=ASTGTM_DEM_lat,
        src_lon=ASTGTM_DEM_lon,
        lat_radius=grid_shp_res / 2,
        lon_radius=grid_shp_res / 2,
    )
    
    # resample for mean
    ASTGTM_DEM_mean_Value = []
    ASTGTM_DEM_std_Value = []
    ASTGTM_DEM_mean_slope_Value = []
    
    if save_original:
        original_Value = []
        original_lat = []
        original_lon = []

    for i in tqdm(
        range(len(searched_grids_index)),
        desc="loop for grids extract ASTGTM_DEM",
        colour="g",
    ):
        # searched index and data for this dst_grid
        searched_grid_index = searched_grids_index[i]
        searched_grid_lat = [
            ASTGTM_DEM_lat[searched_grid_index[0][j]]
            for j in range(len(searched_grid_index[0]))
        ]
        searched_grid_lon = [
            ASTGTM_DEM_lon[searched_grid_index[1][j]]
            for j in range(len(searched_grid_index[0]))
        ]
        searched_grid_data = [
            ASTGTM_DEM_data[searched_grid_index[0][j], searched_grid_index[1][j]]
            for j in range(len(searched_grid_index[0]))
        ]  # index: (lat, lon), namely (row, col)

        mean_value = resample.resampleMethod_GeneralFunction(
            searched_grid_data,
            searched_grid_lat,
            searched_grid_lon,
            None,
            None,
            general_function=np.mean,
            missing_value=-9999.0,
        )

        std_value = resample.resampleMethod_GeneralFunction(
            searched_grid_data,
            searched_grid_lat,
            searched_grid_lon,
            None,
            None,
            general_function=np.std,
            missing_value=-9999.0,
        )

        ASTGTM_DEM_mean_slope_value = (
            (max(searched_grid_data) - min(searched_grid_data))
            / ((2 * grid_shp_res_m**2) ** 0.5)
            * 100
        )

        ASTGTM_DEM_mean_Value.append(mean_value)
        ASTGTM_DEM_std_Value.append(std_value)
        ASTGTM_DEM_mean_slope_Value.append(ASTGTM_DEM_mean_slope_value)

        # check
        if check_search and i == 0:
            cgdf = CreateGDF()
            grid_shp_grid = grid_shp.loc[[i], "geometry"]
            searched_grids_gdf = cgdf.createGDF_rectangle_central_coord(
                searched_grid_lon, searched_grid_lat, ASTGTM_DEM_lat_res
            )

            fig, ax = plt.subplots()
            grid_shp_grid.boundary.plot(ax=ax, edgecolor="r", linewidth=2)  # target, grid_shp
            searched_grids_gdf.plot(
                ax=ax, edgecolor="k", linewidth=0.2, facecolor="b", alpha=0.5
            )  # searched data
            
            ax.set_title("check search")
            
            plt.show(block=True)

        if save_original:
            original_Value.append(searched_grid_data)
            original_lat.append(searched_grid_lat)
            original_lon.append(searched_grid_lon)

    # set missing_value as none
    ASTGTM_DEM_mean_Value = np.array(ASTGTM_DEM_mean_Value)
    ASTGTM_DEM_mean_Value[ASTGTM_DEM_mean_Value == -9999.0] = np.NAN
    ASTGTM_DEM_std_Value = np.array(ASTGTM_DEM_std_Value)
    ASTGTM_DEM_mean_slope_Value = np.array(ASTGTM_DEM_mean_slope_Value)

    if save_original:
        for i in range(len(original_Value)):
            original_Value_grid = original_Value[i]
            original_Value_grid = np.array(original_Value_grid, float)
            original_Value_grid[original_Value_grid == -9999.0] = np.NAN
            original_Value[i] = original_Value_grid.tolist()

    # save in grid_shp
    grid_shp["ASTGTM_DEM_mean_Value"] = ASTGTM_DEM_mean_Value
    grid_shp["ASTGTM_DEM_std_Value"] = ASTGTM_DEM_std_Value
    grid_shp["ASTGTM_DEM_mean_slope_Value%"] = ASTGTM_DEM_mean_slope_Value

    if save_original:
        grid_shp["ASTGTM_DEM_original_Value"] = original_Value
        grid_shp["ASTGTM_DEM_original_lat"] = original_lat
        grid_shp["ASTGTM_DEM_original_lon"] = original_lon
    
    # plot
    if plot:
        # original, total, check lat corresponding to the array
        plt.figure()
        show(
            ASTGTM_DEM_downscale,
            title="total_ASTGTM_DEM_downscale",
            extent=[ASTGTM_DEM_lon[0], ASTGTM_DEM_lon[-1], ASTGTM_DEM_lat[-1], ASTGTM_DEM_lat[0]],
        )

        # readed mean
        fig, ax = plt.subplots()
        grid_shp.plot("ASTGTM_DEM_mean_Value", ax=ax, edgecolor="k", linewidth=0.2)
        ax.set_title("readed mean")
        ax.set_xlim(
            [min(grids_lon) - grid_shp_res / 2, max(grids_lon) + grid_shp_res / 2]
        )
        ax.set_ylim(
            [min(grids_lat) - grid_shp_res / 2, max(grids_lat) + grid_shp_res / 2]
        )

        # readed std
        fig, ax = plt.subplots()
        grid_shp.plot("ASTGTM_DEM_std_Value", ax=ax, edgecolor="k", linewidth=0.2)
        ax.set_title("readed std")
        ax.set_xlim(
            [min(grids_lon) - grid_shp_res / 2, max(grids_lon) + grid_shp_res / 2]
        )
        ax.set_ylim(
            [min(grids_lat) - grid_shp_res / 2, max(grids_lat) + grid_shp_res / 2]
        )

        # readed slope
        fig, ax = plt.subplots()
        grid_shp.plot("ASTGTM_DEM_mean_slope_Value%", ax=ax, edgecolor="k", linewidth=0.2)
        ax.set_title("readed mean slope value")
        ax.set_xlim(
            [min(grids_lon) - grid_shp_res / 2, max(grids_lon) + grid_shp_res / 2]
        )
        ax.set_ylim(
            [min(grids_lat) - grid_shp_res / 2, max(grids_lat) + grid_shp_res / 2]
        )
        
        plt.show()

    return grid_shp
