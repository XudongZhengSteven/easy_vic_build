# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

""" 
Module: build_Domain

This module provides functions for constructing and modifying the domain file.
It includes capabilities to:
- Calculate mask, fractional areas, and grid lengths for the domain.
- Build a NetCDF domain file for VIC model simulations.
- Modify the domain by adjusting the mask for a specified pour point.

Functions:
----------
    - cal_mask_frac_area_length: Computes the mask, fractional area, and grid length based on the basin shape and grid.
    - buildDomain: Builds the VIC domain NetCDF file using grid and basin data, creating variables for latitude, longitude, mask, area, and other domain attributes.
    - modifyDomain_for_pourpoint: Modifies the VIC domain file by updating the mask for the pour point location.
    
Usage:
------
    To use this module, provide a domain directory and the required grid and basin shape data. You can then call `buildDomain` to generate the domain file and `modifyDomain_for_pourpoint` to modify the domain with a pour point.

Example:
------
    # Example usage:
    basin_index = 213
    model_scale = "6km"
    case_name = f"{basin_index}_{model_scale}"

    evb_dir = Evb_dir("./examples") # cases_home="/home/xdz/code/VIC_xdz/cases"
    evb_dir.builddir(case_name)
    
    dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2 = readdpc(evb_dir)
    buildDomain(evb_dir, dpc_VIC_level1, reverse_lat=True)
    
Dependencies:
-------------
    - matplotlib: For plotting the DPCs.
    - pickle: For serializing the DPC data.
    - tools.utilities: Custom utility functions.
    - tools.decoractors: For measuring function execution time.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com

"""

from netCDF4 import Dataset
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from .tools.dpc_func.basin_grid_func import grids_array_coord_map
from tqdm import *
from .tools.decoractors import clock_decorator
from .tools.geo_func.search_grids import *
from . import logger

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


@clock_decorator(print_arg_ret=False)
def buildDomain(evb_dir, dpc_VIC, reverse_lat=True, pourpoint_xindex=None, pourpoint_yindex=None):
    """
    Build the domain file for the VIC model, including variables like latitude, longitude, mask, area, and others.

    Parameters:
    -----------
    evb_dir : str
        The directory where the domain file will be saved.
    dpc_VIC : object
        An object containing grid and basin shapefiles used to define the VIC grid.
    reverse_lat : bool, optional, default=True
        Flag to indicate whether to reverse latitudes (True for Northern Hemisphere: large -> small).
    pourpoint_xindex : int, optional
        X-index of the pour point to modify the mask. If not provided, no modification is made.
    pourpoint_yindex : int, optional
        Y-index of the pour point to modify the mask. If not provided, no modification is made.

    Returns:
    --------
    None
        This function does not return anything. It saves the domain file to the specified directory.

    Notes:
    ------
    The function will generate a domain file for the VIC model that includes latitude, longitude, mask, area, and other necessary variables.
    """
    # ====================== build Domain ======================
    logger.info("Starting to build domain file... ...")
    # create domain file
    logger.debug(f"open {evb_dir.domainFile_path} file for saving as domain file")
    with Dataset(evb_dir.domainFile_path, "w", format="NETCDF4") as dst_dataset:
        
        # get lon/lat
        logger.debug(f"get lon_list and lat_list from the dpc")
        lon_list, lat_list, lon_map_index_level0, lat_map_index_level0 = grids_array_coord_map(dpc_VIC.grid_shp, reverse_lat=reverse_lat)
        
        logger.debug(f"define dimension and variables in the domain file")
        # Define dimensions
        lat = dst_dataset.createDimension("lat", len(lat_list))
        lon = dst_dataset.createDimension("lon", len(lon_list))
        
        # Create variables for latitudes, longitudes, and other grid data
        lat_v = dst_dataset.createVariable("lat", "f8", ("lat",))
        lon_v = dst_dataset.createVariable("lon", "f8", ("lon",))
        lats = dst_dataset.createVariable("lats", "f8", ("lat", "lon",))  # 2D array
        lons = dst_dataset.createVariable("lons", "f8", ("lat", "lon",))  # 2D array
    
        mask = dst_dataset.createVariable("mask", "i4", ("lat", "lon",))
        area = dst_dataset.createVariable("area", "f8", ("lat", "lon",))
        frac = dst_dataset.createVariable("frac", "f8", ("lat", "lon",))
        frac_grid_in_basin = dst_dataset.createVariable("frac_grid_in_basin", "f8", ("lat", "lon",))
        x_length = dst_dataset.createVariable("x_length", "f8", ("lat", "lon",))
        y_length = dst_dataset.createVariable("y_length", "f8", ("lat", "lon",))
        
        # Assign values to variables
        lat_v[:] = np.array(lat_list)
        lon_v[:] = np.array(lon_list)
        grid_array_lons, grid_array_lats = np.meshgrid(lon_v[:], lat_v[:])  # 2D array
        lons[:, :] = grid_array_lons
        lats[:, :] = grid_array_lats
        
        mask_array, frac_array, frac_grid_in_basin_array, area_array, x_length_array, y_length_array = cal_mask_frac_area_length(dpc_VIC, reverse_lat=reverse_lat, plot=False,
                                                                                                                                 pourpoint_xindex=None, pourpoint_yindex=None)
        mask[:, :] = mask_array
        area[:, :] = area_array
        frac[:, :] = frac_array
        frac_grid_in_basin[:, :] = frac_grid_in_basin_array
        x_length[:, :] = x_length_array
        y_length[:, :] = y_length_array
        
        # Add attributes to variables
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
        
        frac_grid_in_basin.long_name = "frac_grid_in_basin"
        frac_grid_in_basin.description = "fraction of grid cell that in basin"
        frac_grid_in_basin.units = "fraction"
        
        # Global attributes
        dst_dataset.title = "VIC5 image domain dataset"
        dst_dataset.note = "domain dataset generated by XudongZheng, zhengxd@sehemodel.club"
        dst_dataset.Conventions = "CF-1.6"
    
    logger.info(f"Building domain completed sucessfully, domain file has been built at {evb_dir.domainFile_path}")
    
    
def cal_mask_frac_area_length(dpc_VIC, reverse_lat=True, plot=False, pourpoint_xindex=None, pourpoint_yindex=None):
    """
    Calculate the mask, fractional area, and grid dimensions (x/y lengths) for the given VIC grid.

    Parameters:
    -----------
    dpc_VIC : object
        An object containing grid and basin information.
    reverse_lat : bool, optional, default=True
        Flag to indicate whether to reverse latitudes (True for Northern Hemisphere: large -> small).
    plot : bool, optional, default=False
        Flag to determine whether to plot the results.
    pourpoint_xindex : int, optional
        X-index of the pour point to modify the mask. If not provided, no modification is made.
    pourpoint_yindex : int, optional
        Y-index of the pour point to modify the mask. If not provided, no modification is made.

    Returns:
    --------
    mask : array
        The computed mask array for the grid.
    frac : array
        The fractional area of the active grid cells.
    frac_grid_in_basin : array
        The fraction of the grid area that falls within the basin.
    area : array
        The area of each grid cell.
    x_length : float
        The x-length of the grid cells.
    y_length : float
        The y-length of the grid cells.

    Notes:
    ------
    The function optionally generates a plot of the mask and grid dimensions if the `plot` flag is set to True.
    """
    logger.info("Starting cal_mask_frac_area_length... ...")
    
    # get grid_shp and basin_shp from the dpc_VIC
    grid_shp = dpc_VIC.grid_shp
    basin_shp = dpc_VIC.basin_shp
    
    # Determine the UTM CRS based on the longitude of the basin center
    lon_cen = basin_shp["lon_cen"].values[0]
    for k in UTM_proj_map.keys():
        if lon_cen >= UTM_proj_map[k]["lon_min"] and lon_cen <= UTM_proj_map[k]["lon_max"]:
            proj_crs = UTM_proj_map[k]["crs_code"]
    
    # Convert grid shapefile to the chosen projection
    grid_shp_projection = deepcopy(grid_shp)
    grid_shp_projection = grid_shp_projection.to_crs(proj_crs)
    
    # lon/lat grid map into index to construct array
    lon_list, lat_list, lon_map_index, lat_map_index = grids_array_coord_map(grid_shp, reverse_lat=reverse_lat)
    
    # Initialize arrays for mask, frac, and frac_grid_in_basin
    mask = np.empty((len(lat_list), len(lon_list)), dtype=int)
    frac = np.full((len(lat_list), len(lon_list)), fill_value=1.0, dtype=float)
    frac_grid_in_basin = np.empty((len(lat_list), len(lon_list)), dtype=float)
    
    logger.debug("Calculating mask and fraction for grid cells...")
    for i in tqdm(grid_shp.index, colour="green", desc="loop for grids to cal mask, frac"):
        center = grid_shp.loc[i, "point_geometry"]
        cen_lon = center.x
        cen_lat = center.y
        
        # Get the grid at the current index
        grid_i = grid_shp.loc[i:i, :]
        # fig, ax = plt.subplots()  # plot for testing
        # grid_i.plot(ax=ax)
        # basin_shp.plot(ax=ax, alpha=0.5)
        
        # intersection
        overlay_gdf = grid_i.overlay(basin_shp, how="intersection")
        
        # Update mask and fraction based on intersection
        if len(overlay_gdf) == 0:
            mask[lat_map_index[cen_lat], lon_map_index[cen_lon]] = 0
            frac_grid_in_basin[lat_map_index[cen_lat], lon_map_index[cen_lon]] = 0
        else:
            mask[lat_map_index[cen_lat], lon_map_index[cen_lon]] = 1
            frac_grid_in_basin[lat_map_index[cen_lat], lon_map_index[cen_lon]] = overlay_gdf.area.values[0] / grid_i.area.values[0]
    
    logger.debug("Calculating mask and fraction successfully")
    
    # Modify pourpoint if provided
    if pourpoint_xindex is not None:
        mask[pourpoint_yindex, pourpoint_xindex] = 1
    
    # Initialize arrays for area and grid cell dimensions
    area = np.empty((len(lat_list), len(lon_list)), dtype=float)
    x_length = np.empty((len(lat_list), len(lon_list)), dtype=float)
    y_length = np.empty((len(lat_list), len(lon_list)), dtype=float)
    
    logger.debug("Calculating area and grid dimensions...")
    for i in tqdm(grid_shp_projection.index, colour="green", desc="loop for grids to cal area, x(y)_length"):
        center = grid_shp_projection.loc[i, "point_geometry"]
        cen_lon = center.x
        cen_lat = center.y
        area[lat_map_index[cen_lat], lon_map_index[cen_lon]] = grid_shp_projection.loc[i, "geometry"].area
        x_length[lat_map_index[cen_lat], lon_map_index[cen_lon]] = grid_shp_projection.loc[i, "geometry"].bounds[2] - grid_shp_projection.loc[i, "geometry"].bounds[0]
        y_length[lat_map_index[cen_lat], lon_map_index[cen_lon]] = grid_shp_projection.loc[i, "geometry"].bounds[3] - grid_shp_projection.loc[i, "geometry"].bounds[1]
    
    
    # Optionally flip arrays based on latitude orientation
    if not reverse_lat:
        mask_flip = np.flip(mask, axis=0)
        frac_grid_in_basin_flip = np.flip(frac_grid_in_basin, axis=0)
        area_flip = np.flip(area, axis=0)
        extent = [lon_list[0], lon_list[-1], lat_list[0], lat_list[-1]]
    else:
        mask_flip = mask
        frac_grid_in_basin_flip = frac_grid_in_basin
        area_flip = area
        extent = [lon_list[0], lon_list[-1], lat_list[-1], lat_list[0]]
    
    # Plot the results if requested
    if plot:
        fig, axes = plt.subplots(2, 2)
        dpc_VIC.plot(fig=fig, ax=axes[0, 0], )
        axes[0, 0].set_xlim([extent[0], extent[1]])
        axes[0, 0].set_ylim([extent[2], extent[3]])
        axes[0, 1].imshow(mask_flip, extent=extent)
        axes[1, 0].imshow(frac_grid_in_basin_flip, extent=extent)
        axes[1, 1].imshow(area_flip, extent=extent)
        
        axes[0, 0].set_title("dpc_VIC")
        axes[0, 1].set_title("mask")
        axes[1, 0].set_title("frac_grid_in_basin")
        axes[1, 1].set_title("area")
    
    logger.info("cal_mask_frac_area_length completed successfully")
    
    return mask, frac, frac_grid_in_basin, area, x_length, y_length


def modifyDomain_for_pourpoint(evb_dir, pourpoint_lon, pourpoint_lat):
    """
    Modify the VIC domain file for the pour point, updating the mask to 1 at the pour point location.

    Parameters:
    -----------
    evb_dir : str
        The directory where the domain file is stored.
    pourpoint_lon : float
        The longitude of the pour point.
    pourpoint_lat : float
        The latitude of the pour point.

    Returns:
    --------
    None
        This function does not return anything. It updates the VIC domain file at the specified location.

    Notes:
    ------
    The function updates the VIC domain file's mask, setting the value to 1 at the pour point location, to reflect the flow direction.
    """
    logger.info(f"Starting to modify domain for pour point at ({pourpoint_lat}, {pourpoint_lon})... ...")
    
    # Open the existing domain file in append mode
    with Dataset(evb_dir.domainFile_path, "a", format="NETCDF4") as src_dataset:
        # get lat, lon
        lat = src_dataset.variables["lat"][:]
        lon = src_dataset.variables["lon"][:]
        
        # Search for the grid cell closest to the provided pour point coordinates
        searched_grid_index = search_grids_nearest([pourpoint_lat], [pourpoint_lon], lat, lon, search_num=1)[0]
        
        # Log the found grid index for the pour point
        logger.debug(f"Found nearest grid index for pour point: {searched_grid_index}")
        
        # Update the mask at the nearest grid cell to 1, indicating the pour point
        src_dataset.variables["mask"][searched_grid_index[0][0], searched_grid_index[1][0]] = 1
        
        # Log the successful update of the mask
        logger.debug(f"Mask updated to 1 at grid ({searched_grid_index[0][0]}, {searched_grid_index[1][0]}) for the pour point.")
        
    logger.info(f"Modifying domain completed sucessfully")