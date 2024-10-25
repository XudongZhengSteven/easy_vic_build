# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
# TODO parallel
import os
import numpy as np
import re
from datetime import datetime
from netCDF4 import Dataset
import cftime
from tqdm import *
import matplotlib.pyplot as plt
from .tools.geo_func import search_grids
from .tools.geo_func.create_gdf import CreateGDF
from .tools.utilities import grids_array_coord_map


def buildMeteForcing(dpc_VIC_level1, evb_dir, date_period,
                     reverse_lat=True, check_search=False,
                     time_re_exp=r"\d{8}.\d{4}",
                     search_func=search_grids.search_grids_radius_rectangle_reverse):
    # ====================== set dir and path ======================
    # set path
    src_home = evb_dir.MeteForcing_src_dir
    suffix = evb_dir.MeteForcing_src_suffix
    src_names = [n for n in os.listdir(src_home) if n.endswith(suffix)]
    
    MeteForcing_dir = evb_dir.MeteForcing_dir
    
    ## ====================== get grid_shp and basin_shp ======================
    grid_shp = dpc_VIC_level1.grid_shp
    basin_shp = dpc_VIC_level1.basin_shp
    
    # grids_map_array
    lon_list, lat_list, lon_map_index, lat_map_index = grids_array_coord_map(grid_shp, reverse_lat=reverse_lat)  #* all lat set as reverse
    
    ## ====================== loop for read year and create forcing ======================
    # set time
    start_year = int(date_period[0][:4])
    end_year = int(date_period[1][:4])
    
    year = start_year
    while year <= end_year:
        print(f"creating forcing: {year}, end with year: {end_year}")
        
        # get files
        src_names_year = [n for n in src_names if "A" + str(year) in n]
        src_names_year.sort()
        
        # get times
        time_str = [re.search(time_re_exp, n)[0] for n in src_names_year]
        time_datetime = [datetime.strptime(t, "%Y%m%d.%H00") for t in time_str]
        
        # create nc
        dst_path_year = os.path.join(MeteForcing_dir, f"forcings.{year}.nc")
        with Dataset(dst_path_year, "w") as dst_dataset:
            # define dimension
            time_dim = dst_dataset.createDimension("time", len(src_names_year))
            lat_dim = dst_dataset.createDimension("lat", len(lat_list))
            lon_dim = dst_dataset.createDimension("lon", len(lon_list))
            
            # define dimension variables
            time_v = dst_dataset.createVariable("time", int, ("time",))
            lat_v = dst_dataset.createVariable("lat", "f8", ("lat",))  # 1D array
            lon_v = dst_dataset.createVariable("lon", "f8", ("lon",))   # 1D array
            lats = dst_dataset.createVariable("lats", "f8", ("lat", "lon",))  # 2D array
            lons = dst_dataset.createVariable("lons", "f8", ("lat", "lon",))  # 2D array
            
            # assign attribute for dimension variables
            time_v.calendar = "proleptic_gregorian"
            time_v.units = f"hours since {date_period[0][:4]}-{date_period[0][4:6]}-{date_period[0][6:8]} 00:00:00"
            
            lat_v.units = "degrees_north"
            lat_v.long_name = "latitude of grid cell center"
            lat_v.standard_name = "latitude"
            lat_v.axis = "Y"
            
            lon_v.units = "degrees_east"
            lon_v.long_name = "longitude of grid cell center"
            lon_v.standard_name = "longitude"
            lon_v.axis = "X"
            
            lats.long_name = "lats 2D"
            lats.description = "Latitude of grid cell 2D"
            lats.units = "degrees"
            
            lons.long_name = "lons 2D"
            lons.description = "longitude of grid cell 2D"
            lons.units = "degrees"
            
            # assign values for dimension variables
            dst_dataset.variables["time"][:] = cftime.date2num(time_datetime, units=time_v.units,calendar=time_v.calendar)
            dst_dataset.variables["lat"][:] = np.array(lat_list)  # 1D array
            dst_dataset.variables["lon"][:] = np.array(lon_list)  # 1D array
            grid_array_lons, grid_array_lats = np.meshgrid(dst_dataset.variables["lon"][:], dst_dataset.variables["lat"][:])  # 2D array
            dst_dataset.variables["lons"][:, :] = grid_array_lons  # 2D array
            dst_dataset.variables["lats"][:, :] = grid_array_lats  # 2D array
            
            # define variables
            tas = dst_dataset.createVariable("tas", "f4", ("time", "lat", "lon",))
            prcp = dst_dataset.createVariable("prcp", "f4", ("time", "lat", "lon",))
            pres = dst_dataset.createVariable("pres", "f4", ("time", "lat", "lon",))
            dswrf = dst_dataset.createVariable("dswrf", "f4", ("time", "lat", "lon",))
            dlwrf = dst_dataset.createVariable("dlwrf", "f4", ("time", "lat", "lon",))
            vp = dst_dataset.createVariable("vp", "f4", ("time", "lat", "lon",))
            wind = dst_dataset.createVariable("wind", "f4", ("time", "lat", "lon",))
            
            # assign attribute for variables
            tas.long_name = "AIR_TEMP"
            tas.description = "Average air temperature"
            tas.units = "C"
            
            prcp.long_name = "PREC"
            prcp.description = "Total precipitation (rain and snow)"
            prcp.units = "mm"

            pres.long_name = "PRESSURE"
            pres.description = "Atmospheric pressure"
            pres.units = "kPa"

            dswrf.long_name = "SWDOWN"
            dswrf.description = "Incoming shortwave radiation"
            dswrf.units = "W/m2"

            dlwrf.long_name = "LWDOWN"
            dlwrf.description = "Incoming longwave radiation"
            dlwrf.units = "W/m2"

            vp.long_name = "VP"
            vp.description = "Vapor pressure"
            vp.units = "kPa"

            wind.long_name = "WIND"
            wind.description = "Wind speed"
            wind.units = "m/s"
            
            # loop for read data in this year
            for i in tqdm(range(len(src_names_year)), desc="loop for read data in this year", colour="green"):
                src_name_year_i = src_names_year[i]
                src_path_year_i = os.path.join(src_home, src_name_year_i)
                
                with Dataset(src_path_year_i, "r") as src_dataset:
                    # get time index
                    src_dataset_time = src_dataset.variables["time"]
                    src_time_cftime = cftime.num2date(src_dataset_time[:][0], units=src_dataset_time.units, calendar=src_dataset_time.calendar)
                    src_time_datetime = datetime(src_time_cftime.year, src_time_cftime.month, src_time_cftime.day,
                                                 src_time_cftime.hour, src_time_cftime.minute, src_time_cftime.second)
                    src_time_datetime_in_dst_dataset_index = int(cftime.date2index(src_time_datetime, time_v, calendar=time_v.calendar))
                    
                    # get lat, lon index
                    if i == 0: # just search once when all src_file is consistent
                        src_lat = src_dataset.variables["lat"][:]
                        src_lon = src_dataset.variables["lon"][:]
                        src_lat_res = (max(src_lat) - min(src_lat)) / (len(src_lat) - 1)
                        src_lon_res = (max(src_lon) - min(src_lon)) / (len(src_lon) - 1)
                        
                        lats_flatten = lats[:, :].flatten()
                        lons_flatten = lons[:, :].flatten()
                        dst_lat_res = (max(lat_v) - min(lat_v)) / (len(lat_v) - 1)
                        dst_lon_res = (max(lon_v) - min(lon_v)) / (len(lon_v) - 1)
                        searched_grids_index = search_func(dst_lat=lats_flatten, dst_lon=lons_flatten, src_lat=src_lat, src_lon=src_lon,
                                                           lat_radius=src_lat_res/2, lon_radius=src_lon_res/2, leave=False)
                    
                    # search data
                    tas_list = []
                    prcp_list = []
                    pres_list = []
                    dswrf_list = []
                    dlwrf_list = []
                    spfh_list = []
                    wind_u_list = []
                    wind_v_list = []
                    for j in range(len(lats_flatten)):
                        # lon/lat
                        searched_grid_index = searched_grids_index[j]
                        searched_grid_lat = [src_lat[searched_grid_index[0][k]] for k in range(len(searched_grid_index[0]))]
                        searched_grid_lon = [src_lon[searched_grid_index[1][k]] for k in range(len(searched_grid_index[0]))]
                        
                        # tas, AIR_TEMP
                        searched_grid_data_tas = [src_dataset.variables["TMP"][0, 0, searched_grid_index[0][k], searched_grid_index[1][k]] for k in range(len(searched_grid_index[0]))][0]
                        tas_list.append(searched_grid_data_tas)
                        
                        # prcp, PREC
                        searched_grid_data_prcp = [src_dataset.variables["APCP"][0, searched_grid_index[0][k], searched_grid_index[1][k]] for k in range(len(searched_grid_index[0]))][0]
                        prcp_list.append(searched_grid_data_prcp)
                        
                        # pres, PRESSURE
                        searched_grid_data_pres = [src_dataset.variables["PRES"][0, searched_grid_index[0][k], searched_grid_index[1][k]] for k in range(len(searched_grid_index[0]))][0]
                        pres_list.append(searched_grid_data_pres)
                        
                        # dswrf, SWDOWN
                        searched_grid_data_dswrf = [src_dataset.variables["DSWRF"][0, searched_grid_index[0][k], searched_grid_index[1][k]] for k in range(len(searched_grid_index[0]))][0]
                        dswrf_list.append(searched_grid_data_dswrf)
                        
                        # dlwrf, LWDOWN
                        searched_grid_data_dlwrf = [src_dataset.variables["DLWRF"][0, searched_grid_index[0][k], searched_grid_index[1][k]] for k in range(len(searched_grid_index[0]))][0]
                        dlwrf_list.append(searched_grid_data_dlwrf)
                        
                        # vp, VP = (SPFH * PERS) / (0.622 + SPFH)
                        searched_grid_data_SPFH = [src_dataset.variables["SPFH"][0, 0, searched_grid_index[0][k], searched_grid_index[1][k]] for k in range(len(searched_grid_index[0]))][0]
                        spfh_list.append(searched_grid_data_SPFH)
                        
                        # wind, Wind = (u**2 + v**2) ** (0.5)
                        searched_grid_data_wind_u = [src_dataset.variables["UGRD"][0, 0, searched_grid_index[0][k], searched_grid_index[1][k]] for k in range(len(searched_grid_index[0]))][0]
                        wind_u_list.append(searched_grid_data_wind_u)
                        searched_grid_data_wind_v = [src_dataset.variables["VGRD"][0, 0, searched_grid_index[0][k], searched_grid_index[1][k]] for k in range(len(searched_grid_index[0]))][0]
                        wind_v_list.append(searched_grid_data_wind_v)
                        
                        # check
                        if check_search and i==0:
                            fig, ax = plt.subplots()
                            cgdf = CreateGDF()
                            dst_grids_gdf = cgdf.createGDF_rectangle_central_coord(lons_flatten, lats_flatten, dst_lat_res)
                            src_grids_gdf = cgdf.createGDF_rectangle_central_coord(searched_grid_lon, searched_grid_lat, src_lat_res)
                            
                            src_grids_gdf.boundary.plot(ax=ax, edgecolor="r", linewidth=2)
                            dst_grids_gdf.plot(ax=ax, edgecolor="k", linewidth=0.2, facecolor="b", alpha=0.5)
                            ax.set_title("check search")
                    
                    # reshape to 2D
                    tas_array_2D = np.reshape(np.array(tas_list), lats.shape)
                    prcp_array_2D = np.reshape(np.array(prcp_list), lats.shape)
                    pres_array_2D = np.reshape(np.array(pres_list), lats.shape)
                    dswrf_array_2D = np.reshape(np.array(dswrf_list), lats.shape)
                    dlwrf_array_2D = np.reshape(np.array(dlwrf_list), lats.shape)
                    spfh_array_2D = np.reshape(np.array(spfh_list), lats.shape)
                    wind_u_array_2D = np.reshape(np.array(wind_u_list), lats.shape)
                    wind_v_array_2D = np.reshape(np.array(wind_v_list), lats.shape)
                    
                    ## unit change
                    # AIR_TEMP: K->C,  x - 273.15
                    tas_array_2D -= 273.15
                    
                    # PERS：Pa->Kpa, x / 1000
                    pres_array_2D /= 1000
                    
                    ## cal other variables
                    # cal vp
                    vp_array_2D = (spfh_array_2D * pres_array_2D) / (0.622 + spfh_array_2D) # VP = (SPFH * PERS) / (0.622 + SPFH)
                    
                    # cal wind
                    wind_array_2D = (wind_u_array_2D**2 + wind_v_array_2D**2) ** (0.5)
                    
                    ## append into dst_dataset
                    dst_dataset.variables["tas"][src_time_datetime_in_dst_dataset_index, :, :] = tas_array_2D
                    dst_dataset.variables["prcp"][src_time_datetime_in_dst_dataset_index, :, :] = prcp_array_2D
                    dst_dataset.variables["pres"][src_time_datetime_in_dst_dataset_index, :, :] = pres_array_2D
                    dst_dataset.variables["dswrf"][src_time_datetime_in_dst_dataset_index, :, :] = dswrf_array_2D
                    dst_dataset.variables["dlwrf"][src_time_datetime_in_dst_dataset_index, :, :] = dlwrf_array_2D
                    dst_dataset.variables["vp"][src_time_datetime_in_dst_dataset_index, :, :] = vp_array_2D
                    dst_dataset.variables["wind"][src_time_datetime_in_dst_dataset_index, :, :] = wind_array_2D
            
            # assign Global attributes
            dst_dataset.title = "VIC5 image meteForcing dataset"
            dst_dataset.note = "meteForcing dataset generated by XudongZheng, zhengxd@sehemodel.club"
            dst_dataset.Conventions = "CF-1.6"
        
        # next year
        year += 1