# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com


import os

import numpy as np
import pandas as pd
import rasterio
from matplotlib import pyplot as plt
from netCDF4 import Dataset
from rasterio.plot import show
from tqdm import *
from easy_vic_build.tools.geo_func import resample, search_grids
from easy_vic_build.tools.geo_func.create_gdf import CreateGDF
from easy_vic_build import logger


def combine_MODIS_BSA_data(date_period=["2003", "2018"]):
    top = "G:\\data\\hydrometeorology\\MODIS\\MCD43D51 v061_Black_Sky_Albedo_Shortwave_daily_1km\\Global"
    suffix = ".hdf"
    
    Albedo_BSA_months = {}
    height = None
    
    for dirpath, dirnames, filenames in os.walk(top):
        for filename in filenames:
            if filename.endswith(suffix):
                src_path = os.path.join(dirpath, filename)
                
                # get date
                src_date = filename[filename.find(".A") + 2 :]
                src_date = src_date[: src_date.find(".")]
                src_year = int(src_date[:4])
                src_day_num = int(src_date[4:])

                stand_date = pd.date_range(f"{src_year}0101", f"{src_year}1231", freq="D")
                src_stand_date = stand_date[src_day_num - 1]

                src_month = src_stand_date.month
                
                if int(src_year) >= int(date_period[0]) and int(src_year) <= int(date_period[1]):
                    print("reading data: ", src_path)
                    
                    # read data
                    with Dataset(src_path, "r", format="NETCDF4_CLASSIC") as dataset:
                        Albedo_BSA = dataset.variables["BRDF_Albedo_BSA_Shortwave"]

                        # clip us first
                        if height is None:
                            height, width = Albedo_BSA.shape
                            lat_Albedo_BSA = np.linspace(90, -90, height)  # 7000  large -> small
                            lon_Albedo_BSA = np.linspace(-180, 180, width)  # 8000
                            lat_Albedo_BSA_res = 180 / height  # 1km
                            lon_Albedo_BSA_res = 360 / width
                            
                            # HRB [105.6, 32, 109, 34.8]
                            lat_JRB_min = 32
                            lat_JRB_max = 34.8
                            lon_JRB_min = 105.6
                            lon_JRB_max = 109
                            
                            lat_index = np.where(
                                (lat_Albedo_BSA >= lat_JRB_min) & (lat_Albedo_BSA <= lat_JRB_max)
                            )[0]
                            lon_index = np.where(
                                (lon_Albedo_BSA >= lon_JRB_min) & (lon_Albedo_BSA <= lon_JRB_max)
                            )[0]
                            # plt.imshow(Albedo_BSA[lat_index[0]: lat_index[-1], lon_index[0]: lon_index[-1]])
                            lat_Albedo_BSA_JRB = lat_Albedo_BSA[lat_index[0] : lat_index[-1] + 1]
                            lon_Albedo_BSA_JRB = lon_Albedo_BSA[lon_index[0] : lon_index[-1] + 1]

                        try:
                            Albedo_BSA_array = Albedo_BSA[
                                lat_index[0] : lat_index[-1] + 1, lon_index[0] : lon_index[-1] + 1
                            ]
                            Albedo_BSA_array = Albedo_BSA_array.filled(np.NAN)
                        except:
                            continue

                    # combine
                    try:
                        Albedo_BSA_month = Albedo_BSA_months[src_month]
                        Albedo_BSA_month = Albedo_BSA_month.reshape((*Albedo_BSA_month.shape, 1))  # 3D
                        Albedo_BSA_array = Albedo_BSA_array.reshape((*Albedo_BSA_array.shape, 1))
                        Albedo_BSA_month = np.concatenate(
                            [Albedo_BSA_month, Albedo_BSA_array], axis=2
                        )
                        
                        Albedo_BSA_month_mean = np.nanmean(Albedo_BSA_month, axis=2)
                        Albedo_BSA_month_temp = {src_month: Albedo_BSA_month_mean}
                        
                    except:
                        Albedo_BSA_month = Albedo_BSA_array
                        Albedo_BSA_month_temp = {src_month: Albedo_BSA_month}

                    Albedo_BSA_months.update(Albedo_BSA_month_temp)

    # save
    for k in Albedo_BSA_months.keys():
        Albedo_BSA_month_array = Albedo_BSA_months[k]
        np.save(
            os.path.join(top, f"combine_MODIS_BSA_HRB_month{k}_{date_period[0]}_{date_period[1]}.npy"),
            Albedo_BSA_month_array,
        )
        np.savetxt(os.path.join(top, f"MODIS_BSA_lat_HRB.txt"), lat_Albedo_BSA_JRB)
        np.savetxt(os.path.join(top, f"MODIS_BSA_lon_HRB.txt"), lon_Albedo_BSA_JRB)


def gapfillingBSA():
    src_home = "E:\\data\\hydrometeorology\\MODIS\\MCD43D51 v061_Black_Sky_Albedo_Shortwave_daily_1km\\HRB\\2003_2018"
    for i in range(1, 13):
        src_path = os.path.join(src_home, f"combine_MODIS_BSA_HRB_month{i}_2003_2018.npy")
        src_data = np.load(src_path)
        src_data = src_data.reshape((src_data.shape[0], src_data.shape[1]))

        # fill
        df = pd.DataFrame(src_data)
        filled_df = df.fillna(method="ffill", axis=1)
        filled_data = filled_df.to_numpy()

        # save
        np.save(
            os.path.join(
                src_home, f"combine_MODIS_BSA_HRB_month{i}_filled_2008_2018.npy"
            ),
            filled_data,
        )


def ExtractData(
    grid_shp,
    grid_shp_res=0.125,
    plot_month=False,
    save_original=False,
    check_search=False,
):
    # read BSA, months: 1-12
    BSA_home = "E:\\data\\hydrometeorology\\MODIS\\MCD43D51 v061_Black_Sky_Albedo_Shortwave_daily_1km\\HRB\\2003_2018"

    # read landcover, classes: 0-14
    umd_landcover_1km_path = "E:\\data\\LULC\\UMD_landcover_classification\\UMD_GLCF_GLCDS_data\\differentFormat\\data.tiff"

    # read lat, lon, res
    BSA_lat = np.loadtxt(os.path.join(BSA_home, f"MODIS_BSA_lat_HRB.txt"))
    BSA_lon = np.loadtxt(os.path.join(BSA_home, f"MODIS_BSA_lon_HRB.txt"))

    BSA_lat_res = (max(BSA_lat) - min(BSA_lat)) / (len(BSA_lat) - 1)
    BSA_lon_res = (max(BSA_lon) - min(BSA_lon)) / (len(BSA_lon) - 1)

    # set grids_lat, lon
    grids_lat = grid_shp.point_geometry.y.to_list()
    grids_lon = grid_shp.point_geometry.x.to_list()

    # read BSA_months
    BSA_months_clip = dict(zip(list(range(1, 13)), [[] for m in range(1, 13)]))
    BSA_months = dict(zip(list(range(1, 13)), [[] for m in range(1, 13)]))
    logger.info("read BSA_months data... ...")
    for m in range(1, 13):
        # read original data
        BSA_month_path = os.path.join(
            BSA_home, f"combine_MODIS_BSA_HRB_month{m}_filled_2008_2018.npy"
        )
        BSA_month = np.load(BSA_month_path)

        # clip
        xindex_start = np.where(BSA_lon <= min(grids_lon) - grid_shp_res)[0][-1]
        xindex_end = np.where(BSA_lon >= max(grids_lon) + grid_shp_res)[0][0]

        yindex_start = np.where(BSA_lat >= max(grids_lat) + grid_shp_res)[0][
            -1
        ]  # large -> small
        yindex_end = np.where(BSA_lat <= min(grids_lat) - grid_shp_res)[0][0]

        BSA_clip = BSA_month[
            yindex_start : yindex_end + 1, xindex_start : xindex_end + 1
        ]
        BSA_lon_clip = BSA_lon[xindex_start : xindex_end + 1]
        BSA_lat_clip = BSA_lat[yindex_start : yindex_end + 1]

        # append
        BSA_months[m] = BSA_month
        BSA_months_clip[m] = BSA_clip

    # read umd
    with rasterio.open(umd_landcover_1km_path, mode="r") as dataset:
        # umd lat lon
        width = dataset.width
        height = dataset.height

        umd_lon = np.array([dataset.xy(0, i)[0] for i in range(width)])
        umd_lat = np.array([dataset.xy(i, 0)[1] for i in range(height)])
        # test: row, column = dataset.index(umd_lon[103], umd_lat[95])

    umd_lat_res = (max(umd_lat) - min(umd_lat)) / (len(umd_lat) - 1)
    umd_lon_res = (max(umd_lon) - min(umd_lon)) / (len(umd_lon) - 1)

    # clip umd
    xindex_start = np.where(umd_lon <= min(grids_lon) - grid_shp_res)[0][-1]
    xindex_end = np.where(umd_lon >= max(grids_lon) + grid_shp_res)[0][0]

    yindex_start = np.where(umd_lat >= max(grids_lat) + grid_shp_res)[0][
        -1
    ]  # large -> small
    yindex_end = np.where(umd_lat <= min(grids_lat) - grid_shp_res)[0][0]

    umd_lon_clip = umd_lon[xindex_start : xindex_end + 1]
    umd_lat_clip = umd_lat[yindex_start : yindex_end + 1]

    # search grids
    logger.info("searching grids for BSA... ...")
    
    # level1 > 1km, source data res: 1km, search LULC grids, then match BSA grids with LULC grids
    searched_grids_index = search_grids.search_grids_radius_rectangle(
        dst_lat=grids_lat,
        dst_lon=grids_lon,
        src_lat=umd_lat_clip,
        src_lon=umd_lon_clip,
        lat_radius=grid_shp_res / 2,
        lon_radius=grid_shp_res / 2,
    )

    # read umd grids and extract BSA
    BSA_mean_Value = dict(zip(list(range(1, 13)), [[] for m in range(1, 13)]))
    if save_original:
        original_Value = dict(zip(list(range(1, 13)), [[] for m in range(1, 13)]))
        original_lat = []
        original_lon = []

    for i in tqdm(
        grid_shp.index, colour="green", desc="loop for each grid to extract BSA"
    ):
        # umd grids
        searched_grid_index = searched_grids_index[i]
        searched_grids_lat_umd = [
            umd_lat_clip[searched_grid_index[0][j]]
            for j in range(len(searched_grid_index[0]))
        ]
        searched_grids_lon_umd = [
            umd_lon_clip[searched_grid_index[1][j]]
            for j in range(len(searched_grid_index[0]))
        ]

        # BSA grids
        searched_match_grids_data = dict(
            zip(list(range(1, 13)), [[] for m in range(1, 13)])
        )
        searched_match_grids_lat = []
        searched_match_grids_lon = []

        # match BSA grid with umd grid
        for j in range(len(searched_grid_index[0])):
            # search lat, lon
            searched_grid_lat = umd_lat_clip[searched_grid_index[0][j]]
            searched_grid_lon = umd_lon_clip[searched_grid_index[1][j]]

            # search neartest BSA grid with umd grid
            searched_grids_index_match = search_grids.search_grids_nearest(
                dst_lat=[searched_grid_lat],
                dst_lon=[searched_grid_lon],
                src_lat=BSA_lat_clip,
                src_lon=BSA_lon_clip,
                search_num=1,
                leave=False,
            )[0]

            searched_match_grid_lat = BSA_lat_clip[searched_grids_index_match[0][0]]
            searched_match_grid_lon = BSA_lon_clip[searched_grids_index_match[1][0]]

            searched_match_grids_lat.append(searched_match_grid_lat)
            searched_match_grids_lon.append(searched_match_grid_lon)

            # loop for months
            for m in range(1, 13):
                BSA_month_clip = BSA_months_clip[m]
                searched_match_grid_data = BSA_month_clip[
                    searched_grids_index_match[0][0], searched_grids_index_match[1][0]
                ]
                searched_match_grids_data[m].append(searched_match_grid_data)

        # resample and save
        for m in range(1, 13):
            BSA_mean_value = resample.resampleMethod_SimpleAverage(
                searched_match_grids_data[m],
                searched_match_grids_lat,
                searched_match_grids_lon,
            )

            # save
            BSA_mean_Value[m].append(BSA_mean_value)

            if save_original:
                original_Value[m].append(searched_match_grids_data[m])
                if m == 1:
                    original_lat.append(searched_match_grids_lat)
                    original_lon.append(searched_match_grids_lon)

        # check
        if check_search and i == 0:
            cgdf = CreateGDF()
            grid_shp_grid = grid_shp.loc[[i], "geometry"]
            searched_umd_grids_gdf = cgdf.createGDF_rectangle_central_coord(
                searched_grids_lon_umd, searched_grids_lat_umd, umd_lat_res
            )
            searched_match_BSA_grids_gdf = cgdf.createGDF_rectangle_central_coord(
                searched_match_grids_lon, searched_match_grids_lat, BSA_lat_res
            )

            fig, ax = plt.subplots()
            grid_shp_grid.boundary.plot(ax=ax, edgecolor="r", linewidth=2)
            
            searched_umd_grids_gdf.plot(
                ax=ax, edgecolor="k", linewidth=2, facecolor="none", alpha=0.5
            )
            searched_match_BSA_grids_gdf.plot(
                ax=ax, edgecolor="k", linewidth=1, facecolor="b", alpha=0.1
            )
            ax.set_title("check search")
            
            plt.show(block=True)

    # save in grid_shp
    for m in range(1, 13):
        grid_shp[f"MODIS_BSA_mean_Value_month{m}"] = np.array(BSA_mean_Value[m])
        if save_original:
            grid_shp["MODIS_BSA_original_lat"] = original_lat
            grid_shp["MODIS_BSA_original_lon"] = original_lon
            grid_shp[f"MODIS_BSA_original_Value_month{m}"] = original_Value[m]

    # plot
    if plot_month:
        # original, total
        plt.figure()
        show(
            BSA_months[plot_month],
            title=f"total_data_BSA_month{plot_month}",
            extent=[BSA_lon[0], BSA_lon[-1], BSA_lat[-1], BSA_lat[0]],
        )

        # original, clip
        plt.figure()
        show(
            BSA_months_clip[plot_month],
            title=f"total_data_BSA_clip_month{plot_month}",
            extent=[
                BSA_lon_clip[0],
                BSA_lon_clip[-1],
                BSA_lat_clip[-1],
                BSA_lat_clip[0],
            ],
        )

        # readed mean
        fig, ax = plt.subplots()
        grid_shp.plot(
            f"MODIS_BSA_mean_Value_month{plot_month}",
            ax=ax,
            edgecolor="k",
            linewidth=0.2,
        )
        ax.set_title(f"readed mean BSA month{plot_month}")
        ax.set_xlim(
            [min(grids_lon) - grid_shp_res / 2, max(grids_lon) + grid_shp_res / 2]
        )
        ax.set_ylim(
            [min(grids_lat) - grid_shp_res / 2, max(grids_lat) + grid_shp_res / 2]
        )

    return grid_shp


if __name__ == "__main__":
    # combine_MODIS_BSA_data()
    # gapfillingBSA()
    pass
