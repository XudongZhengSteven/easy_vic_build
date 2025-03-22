# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: basin_grid_class

This module provides functionality for defining and managing basin-level grid structures
in hydrological models. It includes classes and methods for grid creation, manipulation,
and interaction with basin-specific data. This module is particularly useful in spatially
distributed hydrological models, where basin grids are crucial for discretizing the model domain.

Class:
--------
    - BasinGrid: A class that represents a grid for a specific basin, managing its spatial
      layout and related data.

Class Methods:
---------------
    - __init__: Initializes the BasinGrid class with the necessary parameters, including
      grid resolution and basin boundaries.
    - create_grid: Creates a grid representation for the basin, dividing the basin into
      smaller cells for simulation.
    - assign_data: Assigns specific basin data (e.g., elevation, land use, soil type)
      to each grid cell.
    - update_grid: Updates the properties of the grid cells based on new data or model results.
    - get_cell_data: Retrieves the data associated with a specific grid cell.
    - visualize_grid: Generates a visual representation of the grid, typically showing
      elevation, land-use distribution, or other spatially distributed data.
    - check_grid_integrity: Verifies the integrity of the grid, ensuring no missing data
      or inconsistencies in the cell structure.
    - load_basin_data: Loads basin-specific data (e.g., from netCDF, CSV files) for integration
      into the grid.
    - save_basin_grid: Saves the grid and its associated data to a file for later use.
    - resample_grid: Resamples the grid to a different resolution, useful for downscaling or
      upscaling data.

Dependencies:
-------------
    - numpy: Provides array manipulation and mathematical operations for grid data.
    - matplotlib: Used for visualizing grid structures and spatially distributed data.
    - pandas: Helps with managing and processing basin-related tabular data.
    - netCDF4: For reading and writing netCDF files containing basin and grid-related data.
    - os: For file path management and operations related to saving and loading grid data.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com

"""

import math
import os

import geopandas as gpd
import numpy as np
import shapely

from ..geo_func.create_gdf import CreateGDF


class Basins(gpd.GeoDataFrame):
    """
    A class for handling basin-related operations.

    Inherits from GeoDataFrame to handle basin geometries.

    Methods
    -------
    __add__(self, basins)
        Add two basins objects (not yet implemented).

    __sub__(self, basins)
        Subtract two basins objects (not yet implemented).

    __and__(self, basins)
        Perform an 'and' operation between two basins objects (not yet implemented).
    """

    def __add__(self, basins):
        pass

    def __sub__(self, basins):
        pass

    def __and__(self, basins):
        pass


class Grids(gpd.GeoDataFrame):
    """
    A class for handling grid-related operations.

    Inherits from GeoDataFrame to handle grid geometries.

    Methods
    -------
    __add__(self, grids)
        Add two grids objects (not yet implemented).

    __sub__(self, grids)
        Subtract two grids objects (not yet implemented).

    __and__(self, grids)
        Perform an 'and' operation between two grids objects (not yet implemented).
    """

    def __add__(self, grids):
        pass

    def __sub__(self, grids):
        pass

    def __and__(self, grids):
        pass


class Basins_from_shapefile(Basins):
    """
    A class to initialize basins from a shapefile.

    Parameters
    ----------
    shapefile_path : str
        Path to the shapefile containing the basin geometries.
    data : optional
        Data associated with the basins.
    geometry : optional
        Geometry column name.
    crs : optional
        Coordinate Reference System (CRS) of the basins.
    """

    def __init__(
        self, shapefile_path, data=None, *args, geometry=None, crs=None, **kwargs
    ):
        shp_gdf = gpd.read_file(shapefile_path)
        super().__init__(shp_gdf, *args, geometry=geometry, crs=crs, **kwargs)


class HCDNBasins(Basins):
    """
    A class to initialize basins for HCDN data.

    Parameters
    ----------
    home : str, optional
        Path to the home directory containing the HCDN shapefile. Default is "E:\\data\\hydrometeorology\\CAMELS".
    data : optional
        Data associated with the basins.
    geometry : optional
        Geometry column name.
    crs : optional
        Coordinate Reference System (CRS) of the basins.
    """

    def __init__(
        self,
        home="E:\\data\\hydrometeorology\\CAMELS",
        data=None,
        *args,
        geometry=None,
        crs=None,
        **kwargs,
    ):
        HCDN_shp_path = os.path.join(
            home, "basin_set_full_res", "HCDN_nhru_final_671.shp"
        )
        HCDN_shp = gpd.read_file(HCDN_shp_path)
        HCDN_shp["AREA_km2"] = HCDN_shp.AREA / 1000000  # m2 -> km2
        super().__init__(HCDN_shp, *args, geometry=geometry, crs=crs, **kwargs)


class HCDNGrids(Grids):
    """
    A class to initialize grids for HCDN data.

    Parameters
    ----------
    home : str
        Path to the home directory containing the grid shapefiles.
    data : optional
        Data associated with the grids.
    geometry : optional
        Geometry column name.
    crs : optional
        Coordinate Reference System (CRS) of the grids.
    """

    def __init__(self, home, *args, data=None, geometry=None, crs=None, **kwargs):
        grid_shp_label_path = os.path.join(home, "map", "grids_0_25_label.shp")
        grid_shp_label = gpd.read_file(grid_shp_label_path)
        grid_shp_path = os.path.join(home, "map", "grids_0_25.shp")
        grid_shp = gpd.read_file(grid_shp_path)
        grid_shp["point_geometry"] = grid_shp_label.geometry

        super().__init__(grid_shp, *args, geometry=geometry, crs=crs, **kwargs)

    def createBoundaryShp(self):
        """
        Create boundary shapefiles for the grid.

        Returns
        -------
        tuple
            A tuple containing the boundary shapefiles for the center and edge of the grid.
        """

        (
            boundary_point_center_shp,
            boundary_point_center_x_y,
            boundary_grids_edge_shp,
            boundary_grids_edge_x_y,
        ) = createBoundaryShp(self)
        return (
            boundary_point_center_shp,
            boundary_point_center_x_y,
            boundary_grids_edge_shp,
            boundary_grids_edge_x_y,
        )


def createBoundaryShp(grid_shp):
    """
    Create boundary shapefiles for the given grid.

    Parameters
    ----------
    grid_shp : GeoDataFrame
        The GeoDataFrame containing the grid geometries.

    Returns
    -------
    tuple
        A tuple containing the boundary shapefiles for the center and edge of the grid, along with their coordinates.
    """
    # boundary: point center
    cgdf_point = CreateGDF()
    boundary_x_min = min(grid_shp["point_geometry"].x)
    boundary_x_max = max(grid_shp["point_geometry"].x)
    boundary_y_min = min(grid_shp["point_geometry"].y)
    boundary_y_max = max(grid_shp["point_geometry"].y)
    boundary_point_center_shp = cgdf_point.createGDF_polygons(
        lon=[[boundary_x_min, boundary_x_max, boundary_x_max, boundary_x_min]],
        lat=[[boundary_y_max, boundary_y_max, boundary_y_min, boundary_y_min]],
        crs=grid_shp.crs,
    )
    boundary_point_center_x_y = [
        boundary_x_min,
        boundary_y_min,
        boundary_x_max,
        boundary_y_max,
    ]

    # boundary: grids edge
    boundary_x_min = min(grid_shp["geometry"].get_coordinates().x)
    boundary_x_max = max(grid_shp["geometry"].get_coordinates().x)
    boundary_y_min = min(grid_shp["geometry"].get_coordinates().y)
    boundary_y_max = max(grid_shp["geometry"].get_coordinates().y)

    boundary_grids_edge_shp = cgdf_point.createGDF_polygons(
        lon=[[boundary_x_min, boundary_x_max, boundary_x_max, boundary_x_min]],
        lat=[[boundary_y_max, boundary_y_max, boundary_y_min, boundary_y_min]],
        crs=grid_shp.crs,
    )
    boundary_grids_edge_x_y = [
        boundary_x_min,
        boundary_y_min,
        boundary_x_max,
        boundary_y_max,
    ]

    return (
        boundary_point_center_shp,
        boundary_point_center_x_y,
        boundary_grids_edge_shp,
        boundary_grids_edge_x_y,
    )


class Grids_for_shp(Grids):
    """
    A subclass of Grids to handle grid operations for a given shapefile (gshp).

    Parameters
    ----------
    gshp : GeoDataFrame
        The input GeoDataFrame (could be basins, grids, etc.).
    cen_lons : array-like, optional
        Center longitudes for constructing grids (default is None).
    cen_lats : array-like, optional
        Center latitudes for constructing grids (default is None).
    stand_lons : array-like, optional
        Standard longitudes for constructing grids (default is None).
    stand_lats : array-like, optional
        Standard latitudes for constructing grids (default is None).
    res : float, optional
        Resolution for grid construction (default is None).
    adjust_boundary : bool, optional
        Whether to adjust the boundary by resolution (default is True).
    expand_grids_num : int, optional
        Number of grids to expand outward from the boundary (default is 0).
    boundary : tuple, optional
        The boundary coordinates (xmin, ymin, xmax, ymax) to create the grids (default is None).
    """

    def __init__(
        self,
        gshp,
        *args,
        cen_lons=None,
        cen_lats=None,
        stand_lons=None,
        stand_lats=None,
        res=None,
        adjust_boundary=True,
        geometry=None,
        crs=None,
        expand_grids_num=0,
        boundary=None,
        **kwargs,
    ):
        """
        Grids (grid_shp) for a given gshp, it can be any gpd (basins, grids...)

        res=None, one grid for this shp (boundary grid)
        cen_lons: directly construct grids based on given cen_lons (do not consider gshp boundary)
        stand_lons: a series of stand_lons, larger than gshp's boundary, construct grids based on standard grids (clip based on gshp boundary)
        adjust_boundary: adjust boundary by res (res/2)
        expand_grids_num: int, expand n grid outward

        """
        # get bound
        if boundary is None:
            shp_bounds = gshp.loc[:, "geometry"].iloc[0].bounds
        else:
            shp_bounds = boundary

        boundary_x_min = shp_bounds[0]
        boundary_x_max = shp_bounds[2]
        boundary_y_min = shp_bounds[1]
        boundary_y_max = shp_bounds[3]

        # lambda function
        grid_polygon = lambda xmin, xmax, ymin, ymax: shapely.geometry.Polygon(
            [(xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)]
        )
        grid_point = lambda x, y: shapely.geometry.Point(x, y)

        # create grid_shp
        grid_shp = gpd.GeoDataFrame()

        if res:
            # construct grids based on given cen_lons: do not consider gshp boundary
            if cen_lons is not None:  # *note: len(cen_lons) == len(cen_lats)
                grid_shp.loc[:, "geometry"] = [
                    grid_polygon(
                        cen_lons[i] - res / 2,
                        cen_lons[i] + res / 2,
                        cen_lats[i] - res / 2,
                        cen_lats[i] + res / 2,
                    )
                    for i in range(len(cen_lats))
                ]
                grid_shp.loc[:, "point_geometry"] = [
                    grid_point(cen_lons[i], cen_lats[i]) for i in range(len(cen_lats))
                ]

            # construct grids based on standard grids: clip based on gshp boundary
            elif stand_lons is not None:

                cen_lons = stand_lons[
                    np.where(
                        stand_lons - res / 2 <= boundary_x_min - res * expand_grids_num
                    )[0][-1] : np.where(
                        stand_lons + res / 2 >= boundary_x_max + res * expand_grids_num
                    )[
                        0
                    ][
                        0
                    ]
                    + 1
                ]
                cen_lats = stand_lats[
                    np.where(
                        stand_lats - res / 2 <= boundary_y_min - res * expand_grids_num
                    )[0][-1] : np.where(
                        stand_lats + res / 2 >= boundary_y_max + res * expand_grids_num
                    )[
                        0
                    ][
                        0
                    ]
                    + 1
                ]

                cen_lons, cen_lats = np.meshgrid(cen_lons, cen_lats)
                cen_lons = cen_lons.flatten()
                cen_lats = cen_lats.flatten()

                grid_shp.loc[:, "geometry"] = [
                    grid_polygon(
                        cen_lons[i] - res / 2,
                        cen_lons[i] + res / 2,
                        cen_lats[i] - res / 2,
                        cen_lats[i] + res / 2,
                    )
                    for i in range(len(cen_lats))
                ]
                grid_shp.loc[:, "point_geometry"] = [
                    grid_point(cen_lons[i], cen_lats[i]) for i in range(len(cen_lats))
                ]

            # construct grids based on boundary
            else:
                if adjust_boundary:
                    boundary_x_min = math.floor(boundary_x_min / res) * res
                    boundary_x_max = math.ceil(boundary_x_max / res) * res
                    boundary_y_min = math.floor(boundary_y_min / res) * res
                    boundary_y_max = math.ceil(boundary_y_max / res) * res

                cen_lons = np.arange(
                    (boundary_x_min + res / 2 - res * expand_grids_num),
                    (boundary_x_max + res * expand_grids_num),
                    res,
                )
                cen_lats = np.arange(
                    (boundary_y_min + res / 2 - res * expand_grids_num),
                    (boundary_y_max + res * expand_grids_num),
                    res,
                )

                # cen_lons = np.arange(math.floor((boundary_x_min + res/2) / (res/2)) * (res/2), math.ceil((boundary_x_max + res/2) / (res/2)) * (res/2), res)
                # cen_lats = np.arange(math.floor((boundary_y_min + res/2) / (res/2)) * (res/2), math.ceil((boundary_y_max + res/2) / (res/2)) * (res/2), res)

                cen_lons, cen_lats = np.meshgrid(cen_lons, cen_lats)
                cen_lons = cen_lons.flatten()
                cen_lats = cen_lats.flatten()

                grid_shp.loc[:, "geometry"] = [
                    grid_polygon(
                        cen_lons[i] - res / 2,
                        cen_lons[i] + res / 2,
                        cen_lats[i] - res / 2,
                        cen_lats[i] + res / 2,
                    )
                    for i in range(len(cen_lats))
                ]
                grid_shp.loc[:, "point_geometry"] = [
                    grid_point(cen_lons[i], cen_lats[i]) for i in range(len(cen_lats))
                ]

        # res=None, one grid for this shp (boundary grid)
        else:
            grid_shp.loc[0, "geometry"] = grid_polygon(
                boundary_x_min, boundary_x_max, boundary_y_min, boundary_y_max
            )
            grid_shp.loc[0, "point_geometry"] = grid_point(
                (boundary_x_min + boundary_x_max) / 2,
                (boundary_y_min + boundary_y_max) / 2,
            )

        grid_shp = grid_shp.set_geometry("point_geometry")
        crs = crs if crs is not None else "EPSG:4326"
        grid_shp = grid_shp.set_crs(crs)

        super().__init__(grid_shp, *args, geometry=geometry, crs=crs, **kwargs)

    def createBoundaryShp(self):
        """
        Create boundary shapefiles for the grid.

        This method uses the `createBoundaryShp` function to generate boundary shapefiles for the grid.
        It returns both the center and edge boundary shapefiles along with their coordinates.

        Returns
        -------
        tuple
            A tuple containing the following elements:
            - boundary_point_center_shp: GeoDataFrame with the center boundary shapefile.
            - boundary_point_center_x_y: List containing the minimum and maximum x, y coordinates of the center boundary.
            - boundary_grids_edge_shp: GeoDataFrame with the edge boundary shapefile.
            - boundary_grids_edge_x_y: List containing the minimum and maximum x, y coordinates of the edge boundary.
        """
        (
            boundary_point_center_shp,
            boundary_point_center_x_y,
            boundary_grids_edge_shp,
            boundary_grids_edge_x_y,
        ) = createBoundaryShp(self)
        return (
            boundary_point_center_shp,
            boundary_point_center_x_y,
            boundary_grids_edge_shp,
            boundary_grids_edge_x_y,
        )
