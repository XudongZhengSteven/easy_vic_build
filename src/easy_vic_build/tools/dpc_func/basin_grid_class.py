# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Data structures and helpers for basin and grid GeoDataFrames.

This module defines lightweight ``geopandas.GeoDataFrame`` subclasses used by
the data-processing workflow:

- ``Basins`` for basin polygons,
- ``Grids`` for grid polygons and center points,
- ``Grids_for_shp`` for programmatically building grid layers from boundaries.
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
    Basin polygon container.

    This class inherits from :class:`geopandas.GeoDataFrame` and keeps type
    propagation behavior for operations that return new frames.
    """
    
    @property
    def _constructor(self):
        """
        Return constructor for pandas/geopandas operations.

        Returns
        -------
        type
            ``Basins`` class.
        """
        return Basins
    
    @classmethod
    def from_shapefile(cls, shapefile_path, **kwargs):
        """
        Build a :class:`Basins` object from a shapefile path.

        Parameters
        ----------
        shapefile_path : str or path-like
            Path to a vector file readable by :func:`geopandas.read_file`.
        **kwargs : dict
            Additional keyword arguments forwarded to ``Basins(...)``.

        Returns
        -------
        Basins
            Basin GeoDataFrame instance.
        """
        gdf = gpd.read_file(shapefile_path)
        return cls(gdf, **kwargs)

class Grids(gpd.GeoDataFrame):
    """
    Grid polygon and point container.

    This class inherits from :class:`geopandas.GeoDataFrame` and is used to
    store both cell polygons (``geometry``) and cell centers
    (``point_geometry``).
    """
    
    @property
    def _constructor(self):
        """
        Return constructor for pandas/geopandas operations.

        Returns
        -------
        type
            ``Grids`` class.
        """
        return Grids
    
    @classmethod
    def from_shapefile(cls, shapefile_path, **kwargs):
        """
        Build a :class:`Grids` object from a shapefile path.

        Parameters
        ----------
        shapefile_path : str or path-like
            Path to a vector file readable by :func:`geopandas.read_file`.
        **kwargs : dict
            Additional keyword arguments forwarded to ``Grids(...)``.

        Returns
        -------
        Grids
            Grid GeoDataFrame instance.
        """
        gdf = gpd.read_file(shapefile_path)
        return cls(gdf, **kwargs)
    
    def createBoundaryShp(self):
        """
        Create center and edge boundary polygons for the grid.

        Returns
        -------
        tuple
            ``(boundary_point_center_shp, boundary_point_center_x_y,
            boundary_grids_edge_shp, boundary_grids_edge_x_y)``.
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
    """
    Grid container that can be initialized directly from grid-generation rules.
    """

    def __init__(
        self, data=None, *args, geometry=None, crs=None, create_grid_kwargs=None, **kwargs
    ):
        """
        Initialize a grid GeoDataFrame.

        Parameters
        ----------
        data : object, optional
            Existing tabular/spatial data accepted by GeoDataFrame.
        *args : tuple
            Positional arguments forwarded to ``GeoDataFrame``.
        geometry : str or array-like, optional
            Geometry column specification when ``data`` is provided.
        crs : str or CRS, optional
            Coordinate reference system. Defaults to ``"EPSG:4326"`` when
            ``create_grid_kwargs`` is used.
        create_grid_kwargs : dict, optional
            Arguments passed to :meth:`create_grid_shp`. When provided, generated
            grids are used as initialization data.
        **kwargs : dict
            Additional keyword arguments forwarded to ``GeoDataFrame``.
        """
        if create_grid_kwargs is not None:
            grid_shp = self.create_grid_shp(**create_grid_kwargs)
            crs = crs if crs is not None else "EPSG:4326"
            super().__init__(grid_shp, *args, crs=crs, **kwargs)
        else:
            super().__init__(data, *args, geometry=geometry, crs=crs, **kwargs)
    
    @property
    def _constructor(self):
        """
        Return constructor for pandas/geopandas operations.

        Returns
        -------
        type
            ``Grids_for_shp`` class.
        """
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
        Build a grid GeoDataFrame from a target geometry/boundary.

        Parameters
        ----------
        gshp : geopandas.GeoDataFrame, optional
            Target geometry container. The first row geometry is used as the
            default boundary when ``boundary`` is not provided.
        cen_lons : array-like, optional
            Grid-center longitudes used for direct grid construction.
        cen_lats : array-like, optional
            Grid-center latitudes used for direct grid construction.
        stand_lons : array-like, optional
            Standard longitude coordinates used to clip/build grids by boundary.
        stand_lats : array-like, optional
            Standard latitude coordinates used to clip/build grids by boundary.
        res : float, optional
            Grid resolution. If ``None``, only one boundary grid cell is built.
        adjust_boundary : bool, optional
            Whether to align boundaries to resolution-compatible edges.
        crs : str or CRS, optional
            Output coordinate reference system. Defaults to ``"EPSG:4326"``.
        expand_grids_num : int, optional
            Number of grid cells to expand outward beyond boundary.
        boundary : sequence of float, optional
            Explicit boundary as ``[xmin, ymin, xmax, ymax]``.

        Returns
        -------
        geopandas.GeoDataFrame or None
            Generated grid GeoDataFrame, or ``None`` when ``gshp`` is ``None``.

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

        if res is not None:
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
                    tol = 1e-10
                    boundary_x_min_o = copy.deepcopy(boundary_x_min)
                    boundary_y_min_o = copy.deepcopy(boundary_y_min)
                    
                    # boundary_x_min = round(adjust_down(boundary_x_min / half_res, tol) * half_res, half_res_places)
                    # boundary_x_max = round(adjust_up(boundary_x_max / half_res, tol) * half_res, half_res_places)
                    # boundary_y_min = round(adjust_down(boundary_y_min / half_res, tol) * half_res, half_res_places)
                    # boundary_y_max = round(adjust_up(boundary_y_max / half_res, tol) * half_res, half_res_places)
                    boundary_x_min = round(adjust_down((boundary_x_min + half_res) / half_res, tol) * half_res - half_res, half_res_places)
                    boundary_x_max = round(adjust_up((boundary_x_max + half_res) / half_res, tol) * half_res - half_res, half_res_places)
                    boundary_y_min = round(adjust_down((boundary_y_min + half_res) / half_res, tol) * half_res - half_res, half_res_places)
                    boundary_y_max = round(adjust_up((boundary_y_max + half_res) / half_res, tol) * half_res - half_res, half_res_places)
                    
                    # offset: make center (start from) to res
                    # boundary_x_min = adjust_offset(boundary_x_min, half_res, res, tol)
                    # boundary_y_min = adjust_offset(boundary_y_min, half_res, res, tol)
                    boundary_x_min = adjust_offset((boundary_x_min + half_res), half_res, res, tol) - half_res
                    boundary_y_min = adjust_offset((boundary_y_min + half_res), half_res, res, tol) - half_res
                    
                    # adjust
                    if (boundary_x_min + res - boundary_x_min_o) <= tol:
                        boundary_x_min += res
                    
                    if (boundary_y_min + res - boundary_y_min_o) <= tol:
                        boundary_y_min += res
                
                n_x = (boundary_x_max - boundary_x_min) / res
                n_y = (boundary_y_max - boundary_y_min) / res
                
                n_x_fixed = adjust_up(n_x/1, tol) + 2 * expand_grids_num
                n_y_fixed = adjust_up(n_y/1, tol) + 2 * expand_grids_num
                
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
    Create center and edge boundary polygons for a grid dataset.

    Parameters
    ----------
    grid_shp : GeoDataFrame
        Grid GeoDataFrame containing ``geometry`` and ``point_geometry``.

    Returns
    -------
    tuple
        ``(boundary_point_center_shp, boundary_point_center_x_y,
        boundary_grids_edge_shp, boundary_grids_edge_x_y)``.
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
    
