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
from decimal import Decimal, getcontext
import shapely
import copy

from ..geo_func.create_gdf import CreateGDF
import pandas as pd

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
    
    @property
    def _constructor(self):
        return Basins
    
    @classmethod
    def from_shapefile(cls, shapefile_path, **kwargs):
        gdf = gpd.read_file(shapefile_path)
        return cls(gdf, **kwargs)

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
    
    @property
    def _constructor(self):
        return Grids
    
    @classmethod
    def from_shapefile(cls, shapefile_path, **kwargs):
        gdf = gpd.read_file(shapefile_path)
        return cls(gdf, **kwargs)
    
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

class Grids_for_shp(Grids):
    def __init__(
        self, data=None, *args, geometry=None, crs=None, create_grid_kwargs=None, **kwargs
    ):
        if create_grid_kwargs is not None:
            grid_shp = self.create_grid_shp(**create_grid_kwargs)
            crs = crs if crs is not None else "EPSG:4326"
            super().__init__(grid_shp, *args, crs=crs, **kwargs)
        else:
            super().__init__(data, *args, geometry=geometry, crs=crs, **kwargs)
    
    @property
    def _constructor(self):
        return Grids_for_shp
    
    def create_grid_shp(
        self,
        gshp=None,
        cen_lons=None,
        cen_lats=None,
        stand_lons=None,
        stand_lats=None,
        res=None,
        adjust_boundary=True,
        crs=None,
        expand_grids_num=0,
        boundary=None,
    ):
        """
        Grids (grid_shp) for a given gshp, it can be any gpd (basins, grids...)

        res=None, one grid for this shp (boundary grid)
        cen_lons: directly construct grids based on given cen_lons (do not consider gshp boundary)
        stand_lons: a series of stand_lons, larger than gshp's boundary, construct grids based on standard grids (clip based on gshp boundary)
        adjust_boundary: adjust boundary by res (res/2)
        expand_grids_num: int, expand n grid outward

        """
        if gshp is None:
            return None
        
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
            half_res = res / 2
            res_places = len(str(res).split('.')[-1])
            half_res_places = len(str(res/2).split('.')[-1])
            
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

                cen_lons = stand_lons[np.where(stand_lons - res / 2 <= boundary_x_min - res * expand_grids_num)[0][-1] : np.where(stand_lons + res / 2 >= boundary_x_max + res * expand_grids_num)[0][0] + 1]
                cen_lats = stand_lats[np.where(stand_lats - res / 2 <= boundary_y_min - res * expand_grids_num)[0][-1] : np.where(stand_lats + res / 2 >= boundary_y_max + res * expand_grids_num)[0][0] + 1]

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
                def adjust_down(q, tol=1e-10):
                    if math.isclose(q % 1, 0.0, abs_tol=tol):
                        aligned = round(q)  # already aligned
                    else:
                        aligned = math.floor(q)  # move down
                        
                    return aligned

                def adjust_up(q, tol=1e-10):
                    if math.isclose(q % 1, 0.0, abs_tol=tol):
                        aligned = round(q)  # already aligned
                    else:
                        aligned = math.ceil(q)  # move up
                        
                    return aligned
                
                def adjust_offset(boundary_min, half_res, res, tol=1e-10):
                    q = (boundary_min + half_res) / res
                    if math.isclose(q % 1, 0.0, abs_tol=tol):
                        return boundary_min
                    else:
                        offset = ((boundary_min + half_res) % res)
                        adjusted = boundary_min - offset
                        return round(adjusted, half_res_places)
                    
                if adjust_boundary:
                    boundary_x_min = round(adjust_down(boundary_x_min / half_res) * half_res, half_res_places)
                    boundary_x_max = round(adjust_up(boundary_x_max / half_res) * half_res, half_res_places)
                    boundary_y_min = round(adjust_down(boundary_y_min / half_res) * half_res, half_res_places)
                    boundary_y_max = round(adjust_up(boundary_y_max / half_res) * half_res, half_res_places)
                    
                    # offset: make center (start from) to res
                    boundary_x_min = adjust_offset(boundary_x_min, half_res, res)
                    boundary_y_min = adjust_offset(boundary_y_min, half_res, res)
                
                n_x = (boundary_x_max - boundary_x_min) / res
                n_y = (boundary_y_max - boundary_y_min) / res
                
                n_x_fixed = adjust_up(n_x/1) + 2 * expand_grids_num
                n_y_fixed = adjust_up(n_y/1) + 2 * expand_grids_num
                
                cen_lons = boundary_x_min + half_res - expand_grids_num * res + np.arange(n_x_fixed) * res
                cen_lats = boundary_y_min + half_res - expand_grids_num * res + np.arange(n_y_fixed) * res
                
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
        
        return grid_shp

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
    