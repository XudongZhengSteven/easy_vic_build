# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
from netCDF4 import Dataset
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from .tools.dpc_func.dpc_func import grids_array_coord_map
from tqdm import *
from .tools.decoractors import clock_decorator

UTM_proj_map = {"UTM Zone 10N": {"lon_min": -126, "lon_max": -120, "crs_code": "EPSG:32610"},
                "UTM Zone 11N": {"lon_min": -120, "lon_max": -114, "crs_code": "EPSG:32611"},
                "UTM Zone 12N": {"lon_min": -114, "lon_max": -108, "crs_code": "EPSG:32612"},
                "UTM Zone 13N": {"lon_min": -108, "lon_max": -102, "crs_code": "EPSG:32613"},
                "UTM Zone 14N": {"lon_min": -102, "lon_max": -96, "crs_code": "EPSG:32614"},
                "UTM Zone 15N": {"lon_min": -96, "lon_max": -90, "crs_code": "EPSG:32615"},
                "UTM Zone 16N": {"lon_min": -90, "lon_max": -84, "crs_code": "EPSG:32616"},
                "UTM Zone 17N": {"lon_min": -84, "lon_max": -78, "crs_code": "EPSG:32617"},
                "UTM Zone 18N": {"lon_min": -78, "lon_max": -72, "crs_code": "EPSG:32618"},
                "UTM Zone 19N": {"lon_min": -72, "lon_max": -66, "crs_code": "EPSG:32619"}}


def cal_mask_frac_area_length(dpc_VIC, reverse_lat=True, plot=False):
    # get grid_shp and basin_shp
    grid_shp = dpc_VIC.grid_shp
    basin_shp = dpc_VIC.basin_shp
    
    # search proj_crs
    lon_cen = basin_shp["lon_cen"].values[0]
    for k in UTM_proj_map.keys():
        if lon_cen >= UTM_proj_map[k]["lon_min"] and lon_cen <= UTM_proj_map[k]["lon_max"]:
            proj_crs = UTM_proj_map[k]["crs_code"]
    
    # get projection grid_shp
    grid_shp_projection = deepcopy(grid_shp)
    grid_shp_projection = grid_shp_projection.to_crs(proj_crs)
    
    # lon/lat grid map into index to construct array
    lon_list, lat_list, lon_map_index, lat_map_index = grids_array_coord_map(grid_shp, reverse_lat=reverse_lat)
    
    ## mask and frac
    # array init
    mask = np.empty((len(lat_list), len(lon_list)), dtype=int)
    frac = np.empty((len(lat_list), len(lon_list)), dtype=float)
    for i in tqdm(grid_shp.index, colour="green", desc="loop for grids to cal mask, frac"):
        center = grid_shp.loc[i, "point_geometry"]
        cen_lon = center.x
        cen_lat = center.y
        
        # grid
        grid_i = grid_shp.loc[i:i, :]
        # fig, ax = plt.subplots()  # plot for testing
        # grid_i.plot(ax=ax)
        # basin_shp.plot(ax=ax, alpha=0.5)
        
        # intersection
        overlay_gdf = grid_i.overlay(basin_shp, how="intersection")
        if len(overlay_gdf) == 0:
            mask[lat_map_index[cen_lat], lon_map_index[cen_lon]] = 0
            frac[lat_map_index[cen_lat], lon_map_index[cen_lon]] = 0
        else:
            mask[lat_map_index[cen_lat], lon_map_index[cen_lon]] = 1
            frac[lat_map_index[cen_lat], lon_map_index[cen_lon]] = overlay_gdf.area.values[0] / grid_i.area.values[0]
    
    ## area
    # array init
    area = np.empty((len(lat_list), len(lon_list)), dtype=float)
    x_length = np.empty((len(lat_list), len(lon_list)), dtype=float)
    y_length = np.empty((len(lat_list), len(lon_list)), dtype=float)
    
    # loop for grids to calculate area
    for i in tqdm(grid_shp_projection.index, colour="green", desc="loop for grids to cal area, x(y)_length"):
        center = grid_shp_projection.loc[i, "point_geometry"]
        cen_lon = center.x
        cen_lat = center.y
        area[lat_map_index[cen_lat], lon_map_index[cen_lon]] = grid_shp_projection.loc[i, "geometry"].area
        x_length[lat_map_index[cen_lat], lon_map_index[cen_lon]] = grid_shp_projection.loc[i, "geometry"].bounds[2] - grid_shp_projection.loc[i, "geometry"].bounds[0]
        y_length[lat_map_index[cen_lat], lon_map_index[cen_lon]] = grid_shp_projection.loc[i, "geometry"].bounds[3] - grid_shp_projection.loc[i, "geometry"].bounds[1]
    
    # flip for plot
    if not reverse_lat:
        mask_flip = np.flip(mask, axis=0)
        frac_flip = np.flip(frac, axis=0)
        area_flip = np.flip(area, axis=0)
        extent = [lon_list[0], lon_list[-1], lat_list[0], lat_list[-1]]
    else:
        mask_flip = mask
        frac_flip = frac
        area_flip = area
        extent = [lon_list[0], lon_list[-1], lat_list[-1], lat_list[0]]
    
    # plot
    if plot:
        fig, axes = plt.subplots(2, 2)
        dpc_VIC.plot(fig=fig, ax=axes[0, 0], )
        axes[0, 0].set_xlim([extent[0], extent[1]])
        axes[0, 0].set_ylim([extent[2], extent[3]])
        axes[0, 1].imshow(mask_flip, extent=extent)
        axes[1, 0].imshow(frac_flip, extent=extent)
        axes[1, 1].imshow(area_flip, extent=extent)
        
        axes[0, 0].set_title("dpc_VIC")
        axes[0, 1].set_title("mask")
        axes[1, 0].set_title("frac")
        axes[1, 1].set_title("area")
    
    return mask, frac, area, x_length, y_length


@clock_decorator
def buildDomain(dpc_VIC, evb_dir, reverse_lat=True):
    # ====================== build Domain ======================
    # create domain file
    with Dataset(evb_dir.domainFile_path, "w", format="NETCDF4") as dst_dataset:
        # get lon/lat
        lon_list, lat_list, lon_map_index_level0, lat_map_index_level0 = grids_array_coord_map(dpc_VIC.grid_shp, reverse_lat=reverse_lat)
        
        # dimensions
        lat = dst_dataset.createDimension("lat", len(lat_list))
        lon = dst_dataset.createDimension("lon", len(lon_list))
        
        # variables
        lat_v = dst_dataset.createVariable("lat", "f8", ("lat",))
        lon_v = dst_dataset.createVariable("lon", "f8", ("lon",))
        lats = dst_dataset.createVariable("lats", "f8", ("lat", "lon",))  # 2D array
        lons = dst_dataset.createVariable("lons", "f8", ("lat", "lon",))  # 2D array
    
        mask = dst_dataset.createVariable("mask", "i4", ("lat", "lon",))
        area = dst_dataset.createVariable("area", "f8", ("lat", "lon",))
        frac = dst_dataset.createVariable("frac", "f8", ("lat", "lon",))
        x_length = dst_dataset.createVariable("x_length", "f8", ("lat", "lon",))
        y_length = dst_dataset.createVariable("y_length", "f8", ("lat", "lon",))
        
        # assign variables values
        lat_v[:] = np.array(lat_list)
        lon_v[:] = np.array(lon_list)
        grid_array_lons, grid_array_lats = np.meshgrid(lon_v[:], lat_v[:])  # 2D array
        lons[:, :] = grid_array_lons
        lats[:, :] = grid_array_lats
        
        mask_array, frac_array, area_array, x_length_array, y_length_array = cal_mask_frac_area_length(dpc_VIC, reverse_lat=reverse_lat, plot=False)
        mask[:, :] = mask_array
        area[:, :] = area_array
        frac[:, :] = frac_array
        x_length[:, :] = x_length_array
        y_length[:, :] = y_length_array
        
        # Variables attributes
        lat_v.standard_name = "latitude"
        lat_v.long_name = "latitude of grid cell center"
        lat_v.units = "degrees_north"
        lat_v.axis = "Y"
        
        lon_v.standard_name = "longitude"
        lon_v.long_name = "longitude of grid cell center"
        lon_v.units = "degrees_east"
        lon_v.axis = "X"
        
        lats.long_name = "lats 2D"
        lats.description = "Latitude of grid cell 2D"
        lats.units = "degrees"
        
        lons.long_name = "lons 2D"
        lons.description = "longitude of grid cell 2D"
        lons.units = "degrees"
        
        mask.long_name = "domain mask"
        mask.comment = "1=inside domain, 0=outside"
        mask.unit = "binary"
        
        area.standard_name = "area"
        area.long_name = "area"
        area.description = "area of grid cell"
        area.units = "m2"
        
        frac.long_name = "frac"
        frac.description = "fraction of grid cell that is active"
        frac.units = "fraction"
        
        # Global attributes
        dst_dataset.title = "VIC5 image domain dataset"
        dst_dataset.note = "domain dataset generated by XudongZheng, zhengxd@sehemodel.club"
        dst_dataset.Conventions = "CF-1.6"
    
    