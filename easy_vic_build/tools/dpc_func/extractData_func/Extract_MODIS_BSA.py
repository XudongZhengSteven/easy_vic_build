# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com


import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import *
import rasterio
from rasterio.plot import show
from netCDF4 import Dataset
from ...geo_func import search_grids, resample
from ...geo_func.create_gdf import CreateGDF


def combine_MODIS_BSA_data():
    # data1: 2000055 - 2002008; data2: 2002009 - 2004366; data3: 2005
    # src_home = "E:\data\hydrometeorology\\MODIS\\MCD43D51 v061_Black_Sky_Albedo_Shortwave_daily_1km\\US\\data2"
    src_home = "G:\\data\\hydrometeorology\\MODIS\\MCD43D51 v061_Black_Sky_Albedo_Shortwave_daily_1km\\US\\data3"
    suffix = ".hdf"
    src_names = [n for n in os.listdir(src_home) if n.endswith(suffix)]
    
    Albedo_BSA_months = {}
    for i in tqdm(range(len(src_names)), desc="loop for read all data", colour="g"):
        src_name = src_names[i]
        src_path = os.path.join(src_home, src_name)
        
        # get date
        src_date = src_name[src_name.find(".A")+2:]
        src_date = src_date[:src_date.find(".")]
        src_year = int(src_date[:4])
        src_day_num = int(src_date[4:])
        
        stand_date = pd.date_range(f"{src_year}0101", f"{src_year}1231", freq="D")
        src_stand_date = stand_date[src_day_num - 1]
        
        src_month = src_stand_date.month
        
        # read data
        with Dataset(src_path, "r", format="NETCDF4_CLASSIC") as dataset:
            Albedo_BSA = dataset.variables["BRDF_Albedo_BSA_Shortwave"]
            
            # clip us first
            if i == 0:
                height, width = Albedo_BSA.shape
                lat_Albedo_BSA = np.linspace(90, -90, height)  # 7000  large -> small
                lon_Albedo_BSA = np.linspace(-180, 180, width)  # 8000
                lat_Albedo_BSA_res = 180 / height  # 1km
                lon_Albedo_BSA_res = 360 / width
                # US [-125,   26,  -67,   48.5]
                lat_US_min = 26
                lat_US_max = 48.5
                lon_US_min = -125
                lon_US_max = -67
                lat_index = np.where((lat_Albedo_BSA >= lat_US_min) & (lat_Albedo_BSA <= lat_US_max))[0]
                lon_index = np.where((lon_Albedo_BSA >= lon_US_min) & (lon_Albedo_BSA <= lon_US_max))[0]
                # plt.imshow(Albedo_BSA_months[2][lat_index[0]: lat_index[-1], lon_index[0]: lon_index[-1]])
                lat_Albedo_BSA_US = lat_Albedo_BSA[lat_index[0]: lat_index[-1]+1]
                lon_Albedo_BSA_US = lon_Albedo_BSA[lon_index[0]: lon_index[-1]+1]
                
            try:
                Albedo_BSA_array = Albedo_BSA[lat_index[0]: lat_index[-1]+1, lon_index[0]: lon_index[-1]+1]
                Albedo_BSA_array = Albedo_BSA_array.filled(np.NAN)
            except:
                continue
        
        # combine
        try:
            Albedo_BSA_month = Albedo_BSA_months[src_month]
            Albedo_BSA_month = Albedo_BSA_month.reshape((*Albedo_BSA_month.shape, 1))
            Albedo_BSA_array = Albedo_BSA_array.reshape((*Albedo_BSA_array.shape, 1))
            Albedo_BSA_month = np.concatenate([Albedo_BSA_month, Albedo_BSA_array], axis=2)
            Albedo_BSA_month_temp = {src_month: Albedo_BSA_month}
        except:
            Albedo_BSA_month = Albedo_BSA_array
            Albedo_BSA_month_temp = {src_month: Albedo_BSA_month}
        
        Albedo_BSA_months.update(Albedo_BSA_month_temp)
    
    # save
    for k in Albedo_BSA_months.keys():
        Albedo_BSA_month_array = Albedo_BSA_months[k]
        np.save(os.path.join(src_home, f"combine_MODIS_BSA_US_month{k}.npy"), Albedo_BSA_month_array)
        np.savetxt(os.path.join(src_home, f"MODIS_BSA_lat_US.txt"), lat_Albedo_BSA_US)
        np.savetxt(os.path.join(src_home, f"MODIS_BSA_lon_US.txt"), lon_Albedo_BSA_US)
        

def combineThreeData():
    src_home = "E:\\data\\hydrometeorology\\MODIS\\MCD43D51 v061_Black_Sky_Albedo_Shortwave_daily_1km\\US"
    dirs = ["data1", "data2", "data3"]
    
    for i in range(1, 13):
        save_path = os.path.join(src_home, f"combine_MODIS_BSA_US_month{i}_all_data.npy")
        f_paths = [os.path.join(src_home, d, f"combine_MODIS_BSA_US_month{i}.npy") for d in dirs]
        MODIS_BSA_arrays = [np.load(fp) for fp in f_paths]
        MODIS_BSA_arrays = [a.reshape((*a.shape, 1)) for a in MODIS_BSA_arrays]
        MODIS_BSA_arrays = np.concatenate(MODIS_BSA_arrays, axis=2)
        MODIS_BSA_arrays_mean = np.nanmean(MODIS_BSA_arrays, axis=2)
        
        np.save(save_path, MODIS_BSA_arrays_mean)


def gapfillingBSA():
    src_home = "E:\\data\\hydrometeorology\\MODIS\\MCD43D51 v061_Black_Sky_Albedo_Shortwave_daily_1km\\US"
    for i in range(1, 13):
        src_path = os.path.join(src_home, f"combine_MODIS_BSA_US_month{i}_all_data.npy")
        src_data = np.load(src_path)
        src_data = src_data.reshape((src_data.shape[0], src_data.shape[1]))
        
        # fill
        df = pd.DataFrame(src_data)
        filled_df = df.fillna(method='ffill', axis=1)
        filled_data =filled_df.to_numpy()
        
        # save
        np.save(os.path.join(src_home, f"combine_MODIS_BSA_US_month{i}_all_data_filled.npy"), filled_data)
        
    
def ExtractData(grid_shp, grid_shp_res=0.125, plot_month=False, save_original=False, check_search=False):
    # read BSA, months: 1-12
    BSA_home = "E:\\data\\hydrometeorology\\MODIS\\MCD43D51 v061_Black_Sky_Albedo_Shortwave_daily_1km\\US"
    
    # read landcover, classes: 0-14
    umd_landcover_1km_path = "E:\\data\\LULC\\UMD_landcover_classification\\UMD_GLCF_GLCDS_data\\differentFormat\\data.tiff"
    
    # read lat, lon, res
    BSA_lat = np.loadtxt(os.path.join(BSA_home, f"MODIS_BSA_lat_US.txt"))
    BSA_lon = np.loadtxt(os.path.join(BSA_home, f"MODIS_BSA_lon_US.txt"))
    
    BSA_lat_res = (max(BSA_lat) - min(BSA_lat)) / (len(BSA_lat) - 1)
    BSA_lon_res = (max(BSA_lon) - min(BSA_lon)) / (len(BSA_lon) - 1)
    
    # set grids_lat, lon
    grids_lat = [grid_shp.loc[i, :].point_geometry.y for i in grid_shp.index]
    grids_lon = [grid_shp.loc[i, :].point_geometry.x for i in grid_shp.index]
    
    # read BSA_months
    BSA_months_clip = dict(zip(list(range(1, 13)), [[]for m in range(1, 13)]))
    BSA_months = dict(zip(list(range(1, 13)), [[]for m in range(1, 13)]))
    for m in range(1, 13):
        # read original data
        BSA_month_path = os.path.join(BSA_home, f"combine_MODIS_BSA_US_month{m}_all_data_filled.npy")
        BSA_month = np.load(BSA_month_path)
        
        # clip
        xindex_start = np.where(BSA_lon <= min(grids_lon) - grid_shp_res)[0][-1]
        xindex_end = np.where(BSA_lon >= max(grids_lon) + grid_shp_res)[0][0]
        
        yindex_start = np.where(BSA_lat >= max(grids_lat) + grid_shp_res)[0][-1]  # large -> small
        yindex_end = np.where(BSA_lat <= min(grids_lat) - grid_shp_res)[0][0]
        
        BSA_clip = BSA_month[yindex_start: yindex_end+1, xindex_start: xindex_end+1]
        BSA_lon_clip = BSA_lon[xindex_start: xindex_end+1]
        BSA_lat_clip = BSA_lat[yindex_start: yindex_end+1]
        
        # append
        BSA_months[m] = BSA_month
        BSA_months_clip[m] = BSA_clip
    
    # read umd
    with rasterio.open(umd_landcover_1km_path, mode="r") as dataset:
        # umd lat lon
        width = dataset.width
        height = dataset.height
        
        umd_lon = [dataset.xy(0, i)[0] for i in range(width)]
        umd_lat = [dataset.xy(i, 0)[1] for i in range(height)]
        # test: row, column = dataset.index(umd_lon[103], umd_lat[95])
        
    umd_lat_res = (max(umd_lat) - min(umd_lat)) / (len(umd_lat) - 1)
    umd_lon_res = (max(umd_lon) - min(umd_lon)) / (len(umd_lon) - 1)
    
    # clip umd
    xindex_start = np.where(umd_lon <= min(grids_lon) - grid_shp_res)[0][-1]
    xindex_end = np.where(umd_lon >= max(grids_lon) + grid_shp_res)[0][0]
    
    yindex_start = np.where(umd_lat >= max(grids_lat) + grid_shp_res)[0][-1]  # large -> small
    yindex_end = np.where(umd_lat <= min(grids_lat) - grid_shp_res)[0][0]
    
    umd_lon_clip = umd_lon[xindex_start: xindex_end+1]
    umd_lat_clip = umd_lat[yindex_start: yindex_end+1]
    
    # search grids
    print("========== search grids for BSA ==========")
    searched_grids_index = search_grids.search_grids_radius_rectangle(dst_lat=grids_lat, dst_lon=grids_lon,
                                                                      src_lat=umd_lat_clip, src_lon=umd_lon_clip,
                                                                      lat_radius=grid_shp_res/2, lon_radius=grid_shp_res/2)
        
    # read umd grids and extract BSA
    BSA_mean_Value = dict(zip(list(range(1, 13)), [[]for m in range(1, 13)]))
    if save_original:
        original_Value = dict(zip(list(range(1, 13)), [[]for m in range(1, 13)]))
        original_lat = []
        original_lon = []
    
    for i in tqdm(grid_shp.index, colour="green", desc="loop for each grid to extract BSA"):
        # umd grids
        searched_grid_index = searched_grids_index[i]
        searched_grids_lat_umd = [umd_lat_clip[searched_grid_index[0][j]] for j in range(len(searched_grid_index[0]))]
        searched_grids_lon_umd = [umd_lon_clip[searched_grid_index[1][j]] for j in range(len(searched_grid_index[0]))]
        
        # BSA grids
        searched_match_grids_data = dict(zip(list(range(1, 13)), [[]for m in range(1, 13)]))
        searched_match_grids_lat = []
        searched_match_grids_lon = []
        
        # match BSA grid with umd grid
        for j in range(len(searched_grid_index[0])):
            searched_grid_lat = umd_lat_clip[searched_grid_index[0][j]]
            searched_grid_lon = umd_lon_clip[searched_grid_index[1][j]]
            
            # search neartest BSA grid with umd grid
            searched_grids_index_match = search_grids.search_grids_nearest(dst_lat=[searched_grid_lat], dst_lon=[searched_grid_lon],
                                                                            src_lat=BSA_lat_clip, src_lon=BSA_lon_clip,
                                                                            search_num=1, leave=False)[0]

            searched_match_grid_lat = BSA_lat_clip[searched_grids_index_match[0][0]]
            searched_match_grid_lon = BSA_lon_clip[searched_grids_index_match[1][0]]
            
            searched_match_grids_lat.append(searched_match_grid_lat)
            searched_match_grids_lon.append(searched_match_grid_lon)
            
            # loop for months
            for m in range(1, 13):
                BSA_month_clip = BSA_months_clip[m]
                searched_match_grid_data = BSA_month_clip[searched_grids_index_match[0][0], searched_grids_index_match[1][0]]
                searched_match_grids_data[m].append(searched_match_grid_data)
                
        # resample and save
        for m in range(1, 13):
            BSA_mean_value = resample.resampleMethod_SimpleAverage(searched_match_grids_data[m], searched_match_grids_lat, searched_match_grids_lon)
            
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
            grid_shp_grid = grid_shp.loc[i:i, "geometry"]
            searched_umd_grids_gdf = cgdf.createGDF_rectangle_central_coord(searched_grids_lon_umd, searched_grids_lat_umd, umd_lat_res)
            searched_match_BSA_grids_gdf = cgdf.createGDF_rectangle_central_coord(searched_match_grids_lon, searched_match_grids_lat, BSA_lat_res)
            
            fig, ax = plt.subplots()
            grid_shp_grid.boundary.plot(ax=ax, edgecolor="r", linewidth=2)
            searched_umd_grids_gdf.plot(ax=ax, edgecolor="k", linewidth=1, facecolor="g", alpha=0.5)
            searched_match_BSA_grids_gdf.plot(ax=ax, edgecolor="k", linewidth=1, facecolor="b", alpha=0.5)
            ax.set_title("check search")
    
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
        show(BSA_months[plot_month], title=f"total_data_BSA_month{plot_month}",
             extent=[BSA_lon[0], BSA_lon[-1],
                     BSA_lat[-1], BSA_lat[0]])
        
        # original, clip
        plt.figure()
        show(BSA_months_clip[plot_month], title=f"total_data_BSA_clip_month{plot_month}",
             extent=[BSA_lon_clip[0], BSA_lon_clip[-1],
                     BSA_lat_clip[-1], BSA_lat_clip[0]])
        
        # readed mean
        fig, ax = plt.subplots()
        grid_shp.plot(f"MODIS_BSA_mean_Value_month{plot_month}", ax=ax, edgecolor="k", linewidth=0.2)
        ax.set_title(f"readed mean BSA month{plot_month}")
        ax.set_xlim([min(grids_lon)-grid_shp_res/2, max(grids_lon)+grid_shp_res/2])
        ax.set_ylim([min(grids_lat)-grid_shp_res/2, max(grids_lat)+grid_shp_res/2])

    return grid_shp


if __name__ == "__main__":
    # combine_MODIS_BSA_data()
    # combineThreeData()
    # gapfillingBSA()
    pass