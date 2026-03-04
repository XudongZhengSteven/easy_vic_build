# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Build and modify VIC domain files.

Public functions
----------------
``buildDomain``
    Create ``domain.nc`` with mask, area, fraction, and grid-length variables.
``cal_mask_frac_area_length``
    Compute domain mask/fraction/area/length arrays from grid and basin geometry.
``modifyDomain_for_pourpoint``
    Force the domain mask at the nearest grid cell to a pour-point location.
``addElevIntoDomain``
    Append elevation from level-1 parameters into the domain file.
"""

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from tqdm import *
import geopandas as gpd

from . import logger
from .tools.decoractors import clock_decorator
from .tools.dpc_func.basin_grid_func import grids_array_coord_map, createStand_grids_lat_lon_from_gridshp, gridshp_index_to_grid_array_index
from .tools.geo_func.search_grids import *

# UTM_proj_map = {
#     "UTM Zone 10N": {"lon_min": -126, "lon_max": -120, "crs_code": "EPSG:32610"},
#     "UTM Zone 11N": {"lon_min": -120, "lon_max": -114, "crs_code": "EPSG:32611"},
#     "UTM Zone 12N": {"lon_min": -114, "lon_max": -108, "crs_code": "EPSG:32612"},
#     "UTM Zone 13N": {"lon_min": -108, "lon_max": -102, "crs_code": "EPSG:32613"},
#     "UTM Zone 14N": {"lon_min": -102, "lon_max": -96, "crs_code": "EPSG:32614"},
#     "UTM Zone 15N": {"lon_min": -96, "lon_max": -90, "crs_code": "EPSG:32615"},
#     "UTM Zone 16N": {"lon_min": -90, "lon_max": -84, "crs_code": "EPSG:32616"},
#     "UTM Zone 17N": {"lon_min": -84, "lon_max": -78, "crs_code": "EPSG:32617"},
#     "UTM Zone 18N": {"lon_min": -78, "lon_max": -72, "crs_code": "EPSG:32618"},
#     "UTM Zone 19N": {"lon_min": -72, "lon_max": -66, "crs_code": "EPSG:32619"},
# }

def generate_utm_proj_map() -> dict:
    """Generate a global UTM zone dictionary (Zone 1-60, N/S)"""
    utm_proj_map = {}
    
    for zone in range(1, 61):
        # Calculate the longitude range of each zone (each zone is 6 degrees wide)
        lon_min = -180 + (zone - 1) * 6
        lon_max = lon_min + 6
        
        # Northern Hemisphere (N) - EPSG:326XX
        utm_proj_map[f"UTM Zone {zone}N"] = {
            "lon_min": lon_min,
            "lon_max": lon_max,
            "crs_code": f"EPSG:326{zone:02d}"  # Zero-padded, e.g., 1 -> 01
        }
        
        # Southern Hemisphere (S) - EPSG:327XX
        utm_proj_map[f"UTM Zone {zone}S"] = {
            "lon_min": lon_min,
            "lon_max": lon_max,
            "crs_code": f"EPSG:327{zone:02d}"
        }
    
    return utm_proj_map

UTM_proj_map = generate_utm_proj_map()

@clock_decorator(print_arg_ret=False)
def buildDomain(
    evb_dir, dpc_VIC,
    reverse_lat=True, basin_shp=None,
    prefine_mask=None
):
    """
    Build ``domain.nc`` for the current case.

    Parameters
    ----------
    evb_dir : Evb_dir
        Case directory manager. Output is written to ``evb_dir.domainFile_path``.
    dpc_VIC : object
        DPC object that provides at least ``grid_shp`` (and optionally ``basin_shp``
        when ``basin_shp`` is not passed explicitly).
    reverse_lat : bool, optional
        Whether latitude axis is arranged north-to-south.
    basin_shp : geopandas.GeoDataFrame, optional
        Basin polygon used to compute active-grid fractions. If ``None``, it is
        taken from ``dpc_VIC`` cache.
    prefine_mask : numpy.ndarray, optional
        If provided, this array is written to ``mask`` instead of computed mask.

    Returns
    -------
    None
    """
    # ====================== build Domain ======================
    logger.info("Starting to build domain file... ...")
    # create domain file
    logger.debug(f"open {evb_dir.domainFile_path} file for saving as domain file")
    with Dataset(evb_dir.domainFile_path, "w", format="NETCDF4") as dst_dataset:

        # get lon/lat
        logger.debug(f"get lon_list and lat_list from the dpc")
        grid_shp = dpc_VIC.get_data_from_cache("grid_shp")[0]
        lon_list, lat_list, lon_map_index_level0, lat_map_index_level0 = (
            grids_array_coord_map(grid_shp, reverse_lat=reverse_lat)
        )

        logger.debug(f"define dimension and variables in the domain file")
        # Define dimensions
        lat = dst_dataset.createDimension("lat", len(lat_list))
        lon = dst_dataset.createDimension("lon", len(lon_list))

        # Create variables for latitudes, longitudes, and other grid data
        lat_v = dst_dataset.createVariable("lat", "f8", ("lat",))
        lon_v = dst_dataset.createVariable("lon", "f8", ("lon",))
        lats = dst_dataset.createVariable(
            "lats",
            "f8",
            (
                "lat",
                "lon",
            ),
        )  # 2D array
        lons = dst_dataset.createVariable(
            "lons",
            "f8",
            (
                "lat",
                "lon",
            ),
        )  # 2D array

        mask = dst_dataset.createVariable(
            "mask",
            "i4",
            (
                "lat",
                "lon",
            ),
        )
        area = dst_dataset.createVariable(
            "area",
            "f8",
            (
                "lat",
                "lon",
            ),
        )
        frac = dst_dataset.createVariable(
            "frac",
            "f8",
            (
                "lat",
                "lon",
            ),
        )
        frac_full_one = dst_dataset.createVariable(
            "frac_full_one",
            "f8",
            (
                "lat",
                "lon",
            ),
        )
        x_length = dst_dataset.createVariable(
            "x_length",
            "f8",
            (
                "lat",
                "lon",
            ),
        )
        y_length = dst_dataset.createVariable(
            "y_length",
            "f8",
            (
                "lat",
                "lon",
            ),
        )

        # Assign values to variables
        lat_v[:] = np.array(lat_list)
        lon_v[:] = np.array(lon_list)
        grid_array_lons, grid_array_lats = np.meshgrid(lon_v[:], lat_v[:])  # 2D array
        lons[:, :] = grid_array_lons
        lats[:, :] = grid_array_lats

        (
            mask_array,
            frac_grid_in_basin_array,
            frac_full_one_array,
            area_array,
            x_length_array,
            y_length_array,
        ) = cal_mask_frac_area_length(
            dpc_VIC,
            reverse_lat=reverse_lat,
            plot=False,
            basin_shp=basin_shp,
        )
        
        mask[:, :] = mask_array if prefine_mask is None else prefine_mask
        area[:, :] = area_array
        frac[:, :] = frac_grid_in_basin_array
        frac_full_one[:, :] = frac_full_one_array
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

        frac_full_one.long_name = "frac_full_one"
        frac_full_one.description = "all value set to 1"
        frac_full_one.units = "fraction"

        # Global attributes
        dst_dataset.title = "VIC5 image domain dataset"
        dst_dataset.note = (
            "domain dataset generated by XudongZheng, zhengxd@sehemodel.club"
        )
        dst_dataset.Conventions = "CF-1.6"

    logger.info(
        f"Building domain sucessfully, domain file has been saved to {evb_dir.domainFile_path}"
    )
    

def cal_mask_frac_area_length(
    dpc_VIC,
    reverse_lat=True,
    plot=False,
    basin_shp=None,
):
    """
    Compute mask/fraction/area/length arrays for domain generation.

    Parameters
    ----------
    dpc_VIC : object
        DPC object that provides cached ``grid_shp`` and optionally ``basin_shp``.
    reverse_lat : bool, optional
        Whether latitude axis is arranged north-to-south.
    plot : bool, optional
        If ``True``, create a quick diagnostic plot.
    basin_shp : geopandas.GeoDataFrame, optional
        Basin polygon for intersection area calculation.

    Returns
    -------
    tuple
        ``(mask, frac_grid_in_basin, frac_full_one, area, x_length, y_length)``.
    """
    logger.info("Starting to cal_mask_frac_area_length... ...")

    # get grid_shp and basin_shp from the dpc_VIC
    grid_shp = dpc_VIC.get_data_from_cache("grid_shp")[0]
    basin_shp = dpc_VIC.get_data_from_cache("basin_shp")[0] if basin_shp is None else basin_shp

    # Determine the UTM CRS based on the longitude of the basin center
    try:
        lon_cen = basin_shp["lon_cen"].values[0]
    except:
        lon_cen = basin_shp.centroid.x[0]
        
    for k in UTM_proj_map.keys():
        if (
            lon_cen >= UTM_proj_map[k]["lon_min"]
            and lon_cen <= UTM_proj_map[k]["lon_max"]
        ):
            proj_crs = UTM_proj_map[k]["crs_code"]

    # Convert grid shapefile to the chosen projection
    grid_shp_projection = deepcopy(grid_shp)
    grid_shp_projection = grid_shp_projection.to_crs(proj_crs)
    basin_shp_projection = deepcopy(basin_shp)
    basin_shp_projection = basin_shp_projection.to_crs(proj_crs)
    
    # lon/lat grid map into index to construct array
    # lon_list, lat_list, lon_map_index, lat_map_index = grids_array_coord_map(
    #     grid_shp, reverse_lat=reverse_lat
    # )
    stand_grids_lat, stand_grids_lon = createStand_grids_lat_lon_from_gridshp(grid_shp, reverse_lat=reverse_lat)
    rows_index, cols_index = gridshp_index_to_grid_array_index(
        grid_shp, stand_grids_lat, stand_grids_lon
    )

    # Initialize arrays for mask, frac, and frac_grid_in_basin
    # mask = np.empty((len(lat_list), len(lon_list)), dtype=int)
    # frac_full_one = np.full((len(lat_list), len(lon_list)), fill_value=1.0, dtype=float)
    # frac_grid_in_basin = np.empty((len(lat_list), len(lon_list)), dtype=float)
    mask = np.empty((len(stand_grids_lat), len(stand_grids_lon)), dtype=int)
    frac_full_one = np.full((len(stand_grids_lat), len(stand_grids_lon)), fill_value=1.0, dtype=float)
    frac_grid_in_basin = np.empty((len(stand_grids_lat), len(stand_grids_lon)), dtype=float)
    area = np.empty((len(stand_grids_lat), len(stand_grids_lon)), dtype=float)
    x_length = np.empty((len(stand_grids_lat), len(stand_grids_lon)), dtype=float)
    y_length = np.empty((len(stand_grids_lat), len(stand_grids_lon)), dtype=float)
    
    logger.debug("Calculating variables for grid cells...")
    
    # overlay
    inter = gpd.overlay(
        grid_shp_projection.reset_index().rename(columns={'index': 'grid_idx'}),
        basin_shp_projection,
        how='intersection'
    )
    
    # cal variables
    grid_area = grid_shp_projection.geometry.area.values
    bounds = grid_shp_projection.geometry.bounds 
    
    area_overlay = np.zeros((len(grid_shp_projection),), dtype=float)
    for grid_id, group in inter.groupby('grid_idx'):
        area_overlay[grid_id] = group.geometry.area.sum()
    
    frac_matrix = area_overlay / grid_area
    frac_matrix[frac_matrix > 1.0] = 1.0
    frac_matrix[frac_matrix <= 0.0] = np.nan
    
    mask[rows_index, cols_index] = frac_matrix > 0
    frac_grid_in_basin[rows_index, cols_index] = frac_matrix
    area[rows_index, cols_index] = grid_area
    x_length[rows_index, cols_index] = (bounds['maxx'] - bounds['minx']).to_numpy()
    y_length[rows_index, cols_index] = (bounds['maxy'] - bounds['miny']).to_numpy()
    
    # for i in tqdm(
    #     grid_shp.index, colour="green", desc="loop for grids to cal mask, frac"
    # ):
    #     center = grid_shp.loc[i, "point_geometry"]
    #     cen_lon = center.x
    #     cen_lat = center.y

    #     # Get the grid at the current index
    #     grid_i = grid_shp.loc[[i], :]
    #     # fig, ax = plt.subplots()  # plot for testing
    #     # grid_i.plot(ax=ax)
    #     # basin_shp.plot(ax=ax, alpha=0.5)

    #     # intersection
    #     overlay_gdf = grid_i.overlay(basin_shp, how="intersection")

    #     # Update mask and fraction based on intersection
    #     if len(overlay_gdf) == 0:
    #         mask[lat_map_index[cen_lat], lon_map_index[cen_lon]] = 0
    #         frac_grid_in_basin[lat_map_index[cen_lat], lon_map_index[cen_lon]] = np.nan  # 0
    #         frac_full_one[lat_map_index[cen_lat], lon_map_index[cen_lon]] = np.nan  # 0
    #     else:
    #         mask[lat_map_index[cen_lat], lon_map_index[cen_lon]] = 1
    #         frac_grid_in_basin[lat_map_index[cen_lat], lon_map_index[cen_lon]] = (
    #             overlay_gdf.area.values[0] / grid_i.area.values[0]
    #         )

    logger.debug("Calculating mask and fraction successfully")

    # Initialize arrays for area and grid cell dimensions
    # area = np.empty((len(lat_list), len(lon_list)), dtype=float)
    # x_length = np.empty((len(lat_list), len(lon_list)), dtype=float)
    # y_length = np.empty((len(lat_list), len(lon_list)), dtype=float)

    # logger.debug("Calculating area and grid dimensions...")
    # for i in tqdm(
    #     grid_shp_projection.index,
    #     colour="green",
    #     desc="loop for grids to cal area, x(y)_length",
    # ):
    #     center = grid_shp_projection.loc[i, "point_geometry"]
    #     cen_lon = center.x
    #     cen_lat = center.y
    #     area[lat_map_index[cen_lat], lon_map_index[cen_lon]] = grid_shp_projection.loc[
    #         i, "geometry"
    #     ].area
    #     x_length[lat_map_index[cen_lat], lon_map_index[cen_lon]] = (
    #         grid_shp_projection.loc[i, "geometry"].bounds[2]
    #         - grid_shp_projection.loc[i, "geometry"].bounds[0]
    #     )
    #     y_length[lat_map_index[cen_lat], lon_map_index[cen_lon]] = (
    #         grid_shp_projection.loc[i, "geometry"].bounds[3]
    #         - grid_shp_projection.loc[i, "geometry"].bounds[1]
    #     )

    # Optionally flip arrays based on latitude orientation
    if not reverse_lat:
        mask_flip = np.flip(mask, axis=0)
        frac_grid_in_basin_flip = np.flip(frac_grid_in_basin, axis=0)
        area_flip = np.flip(area, axis=0)
        # extent = [lon_list[0], lon_list[-1], lat_list[0], lat_list[-1]]
        extent = [stand_grids_lon[0], stand_grids_lon[-1], stand_grids_lat[0], stand_grids_lat[-1]]
    else:
        mask_flip = mask
        frac_grid_in_basin_flip = frac_grid_in_basin
        area_flip = area
        # extent = [lon_list[0], lon_list[-1], lat_list[-1], lat_list[0]]
        extent = [stand_grids_lon[0], stand_grids_lon[-1], stand_grids_lat[0], stand_grids_lat[-1]]
        
    # Plot the results if requested
    if plot:
        fig, axes = plt.subplots(2, 2)
        dpc_VIC.plot(
            fig=fig,
            ax=axes[0, 0],
        )
        axes[0, 0].set_xlim([extent[0], extent[1]])
        axes[0, 0].set_ylim([extent[2], extent[3]])
        axes[0, 1].imshow(mask_flip, extent=extent)
        axes[1, 0].imshow(frac_grid_in_basin_flip, extent=extent)
        axes[1, 1].imshow(area_flip, extent=extent)

        axes[0, 0].set_title("dpc_VIC")
        axes[0, 1].set_title("mask")
        axes[1, 0].set_title("frac_grid_in_basin")
        axes[1, 1].set_title("area")

    logger.info("cal_mask_frac_area_length successfully")

    return mask, frac_grid_in_basin, frac_full_one, area, x_length, y_length


def modifyDomain_for_pourpoint(evb_dir, pourpoint_lon, pourpoint_lat):
    """
    Set domain mask to ``1`` at the grid nearest to the pour point.

    Parameters
    ----------
    evb_dir : Evb_dir
        Case directory manager.
    pourpoint_lon : float
        Pour-point longitude.
    pourpoint_lat : float
        Pour-point latitude.

    Returns
    -------
    None
    """
    logger.info(
        f"Starting to modify domain for pour point at ({pourpoint_lat}, {pourpoint_lon})... ..."
    )

    # Open the existing domain file in append mode
    with Dataset(evb_dir.domainFile_path, "a", format="NETCDF4") as src_dataset:
        # get lat, lon
        lat = src_dataset.variables["lat"][:]
        lon = src_dataset.variables["lon"][:]

        # Search for the grid cell closest to the provided pour point coordinates
        searched_grid_index = search_grids_nearest(
            [pourpoint_lat], [pourpoint_lon], lat, lon, search_num=1
        )[0]

        # Log the found grid index for the pour point
        logger.debug(f"Found nearest grid index for pour point: {searched_grid_index}")

        # Update the mask at the nearest grid cell to 1, indicating the pour point
        src_dataset.variables["mask"][
            searched_grid_index[0][0], searched_grid_index[1][0]
        ] = 1

        # Log the successful update of the mask
        logger.debug(
            f"Mask updated to 1 at grid ({searched_grid_index[0][0]}, {searched_grid_index[1][0]}) for the pour point."
        )

    logger.info(f"Modifying domain sucessfully")


def addElevIntoDomain(evb_dir, params_dataset_level1):
    """Append ``elev`` variable into an existing domain file.

    Parameters
    ----------
    evb_dir : Evb_dir
        Case directory manager.
    params_dataset_level1 : netCDF4.Dataset
        Parameter dataset that contains ``elev`` with shape ``(lat, lon)``.

    Returns
    -------
    None
    """
    logger.info(
        f"Starting to add elev (from param_level1) into domain... ..."
    )
    
    # Open the existing domain file in append mode
    with Dataset(evb_dir.domainFile_path, "a", format="NETCDF4") as src_dataset:
        
        # Create variables for elev
        elev = src_dataset.createVariable("elev", "f8", ("lat", "lon"))
        
        # Add attributes to variables
        elev.standard_name = "elevation"
        elev.long_name = "elevation of grid cell"
        elev.units = "m"
        
        # Assign values to elev variable
        assert params_dataset_level1["elev"].shape == elev.shape, "Shape of elev from params_dataset_level1 does not match domain shape"
        elev[:, :] = params_dataset_level1.variables["elev"][:, :]

    logger.info(f"Adding elev into domain sucessfully")
    
    
def centers_to_edges(centers):
    """Compute grid edges from center coordinates."""
    centers = np.array(centers)
    diffs = np.diff(centers)
    left = centers[0] - diffs[0] / 2
    right = centers[-1] + diffs[-1] / 2
    mid = centers[:-1] + diffs / 2
    edges = np.concatenate(([left], mid, [right]))
    return edges


def remap_level1_to_level0_mask(lon_level1, lat_level1, mask_level1,
                                lon_level0, lat_level0):
    """
    Map level-1 mask to level-0 grid by cell-center lookup.

    Each level-0 grid cell inherits the mask value of the level-1 cell
    in which its center falls.

    Returns
    -------
    numpy.ndarray
        Remapped mask with ``(len(lat_level0), len(lon_level0))`` shape.
    """

    lon_level1 = np.array(lon_level1)
    lat_level1 = np.array(lat_level1)
    lon_level0   = np.array(lon_level0)
    lat_level0   = np.array(lat_level0)

    lon_edges = centers_to_edges(lon_level1)
    lat_edges = centers_to_edges(lat_level1)

    # flip if descending
    if lon_edges[1] < lon_edges[0]:
        lon_edges = lon_edges[::-1]
        lon_level1 = lon_level1[::-1]
        mask_level1 = mask_level1[:, ::-1]

    if lat_edges[1] < lat_edges[0]:
        lat_edges = lat_edges[::-1]
        lat_level1 = lat_level1[::-1]
        mask_level1 = mask_level1[::-1, :]

    # For each level0 cell, locate its level1 parent cell
    xi = np.searchsorted(lon_edges, lon_level0) - 1
    yi = np.searchsorted(lat_edges, lat_level0) - 1

    xi = np.clip(xi, 0, mask_level1.shape[1] - 1)
    yi = np.clip(yi, 0, mask_level1.shape[0] - 1)

    XI, YI = np.meshgrid(xi, yi)
    mask_level0 = mask_level1[YI, XI]

    return mask_level0
