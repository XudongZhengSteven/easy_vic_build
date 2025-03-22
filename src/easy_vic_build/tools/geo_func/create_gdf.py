# code: utf-8
# author: "Xudong Zheng"
# email: z786909151@163.com

"""
Module: create_gdf

This module contains the CreateGDF class, which provides methods for creating GeoDataFrames
from geospatial data. The class supports the creation of GeoDataFrames for different types
of geometric shapes, including rectangular grids, points, and polygons. These methods are
useful for generating geospatial data structures from coordinate-based inputs, which can
then be used for spatial analysis and visualization.

Classes:
--------
    - CreateGDF: A class that provides methods to create GeoDataFrames for rectangular grids,
      points, and polygons based on input coordinates and resolution.

Dependencies:
-------------
    - os: Provides a way to interact with the operating system for file handling and directory operations.
    - pandas: Used for creating and manipulating DataFrames, which are then converted into GeoDataFrames.
    - geopandas: Extends pandas and provides support for geospatial data, allowing the creation
      of GeoDataFrames from geometric shapes.
    - shapely: Provides geometric operations, such as the creation of polygons and points, used
      in the GeoDataFrame creation process.
    - matplotlib: Used for plotting, including the visualization of geospatial data.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com
"""


import os

import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
from shapely import geometry


class CreateGDF:
    """Create GeoDataFrame (GDF) based on coordinates.

    This class provides methods to create different types of GeoDataFrames (GDFs)
    based on input coordinates (latitude and longitude), resolution, and other parameters.
    It supports creating GDFs for rectangular grids, points, and polygons.

    Attributes
    ----------
    _info : str
        An optional information string to store additional information about the instance.

    Methods
    -------
    __call__():
        Placeholder method for callable functionality (not yet implemented).
    createGDF_rectangle_central_coord(lon, lat, det, ID=None, crs="EPSG:4326"):
        Creates a GeoDataFrame for rectangular grids based on central coordinates and resolution.
    createGDF_points(lon, lat, ID=None, crs="EPSG:4326"):
        Creates a GeoDataFrame for points based on longitude and latitude.
    createGDF_polygons(lon, lat, ID=None, crs="EPSG:4326"):
        Creates a GeoDataFrame for polygons based on coordinates of their vertices.
    plot():
        Placeholder method for plotting (not yet implemented).
    """

    def __init__(self, info=""):
        """Initializes the CreateGDF instance.

        Parameters
        ----------
        info : str, optional
            An optional string to store additional information about the instance.

        """
        self._info = info

    def __call__(self):
        """Placeholder for callable functionality.

        This method is a placeholder and is not yet implemented.

        """
        pass

    def createGDF_rectangle_central_coord(
        self, lon, lat, det, ID=None, crs="EPSG:4326"
    ):
        """Creates a GeoDataFrame for rectangular grids based on the central coordinates and resolution.

        This method creates rectangular grid polygons centered on the provided longitude and latitude values,
        with the specified resolution (det).

        Parameters
        ----------
        lon : array-like
            1D array of longitude values representing the central coordinates of the grids.
        lat : array-like
            1D array of latitude values representing the central coordinates of the grids.
        det : float
            The resolution (det) that defines the size of the rectangular grid.
        ID : array-like, optional
            A sequence of IDs for the GeoDataFrame rows. If not provided, the default is the index of the DataFrame.
        crs : str, optional
            The coordinate reference system (CRS) to use. The default is "EPSG:4326".

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the rectangular grids as polygons.

        """
        gdf = pd.DataFrame(columns=["clon", "clat"])
        gdf["clon"] = lon  # central lon
        gdf["clat"] = lat  # central lat
        gdf["ID"] = gdf.index if ID is None else ID
        polygon = geometry.Polygon
        gdf["geometry"] = gdf.apply(
            lambda row: polygon(
                [
                    (row.clon - det / 2, row.clat - det / 2),
                    (row.clon + det / 2, row.clat - det / 2),
                    (row.clon + det / 2, row.clat + det / 2),
                    (row.clon - det / 2, row.clat + det / 2),
                ]
            ),
            axis=1,
        )
        gdf = gpd.GeoDataFrame(gdf, crs=crs)

        return gdf

    def createGDF_points(self, lon, lat, ID=None, crs="EPSG:4326"):
        """Creates a GeoDataFrame for points based on longitude and latitude coordinates.

        This method creates point geometries for each pair of longitude and latitude coordinates
        provided in the input arrays.

        Parameters
        ----------
        lon : array-like
            1D array of longitude values representing the points.
        lat : array-like
            1D array of latitude values representing the points.
        ID : array-like, optional
            A sequence of IDs for the GeoDataFrame rows. If not provided, the default is the index of the DataFrame.
        crs : str, optional
            The coordinate reference system (CRS) to use. The default is "EPSG:4326".

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the points as point geometries.

        """
        gdf = pd.DataFrame(columns=["lon", "lat"])
        gdf["lon"] = lon
        gdf["lat"] = lat
        gdf["ID"] = gdf.index if ID is None else ID
        point = geometry.Point
        gdf["geometry"] = gdf.apply(lambda row: point([(row.lon, row.lat)]), axis=1)
        gdf = gpd.GeoDataFrame(gdf, crs=crs)

        return gdf

    def createGDF_polygons(self, lon, lat, ID=None, crs="EPSG:4326"):
        """Creates a GeoDataFrame for polygons based on longitude and latitude coordinates of their vertices.

        This method creates polygons for each set of longitude and latitude coordinates provided
        in the input lists, where each list of coordinates defines a polygon.

        Parameters
        ----------
        lon : list of array-like
            A list of 1D arrays, where each array contains multiple longitude values defining a polygon.
            [(lon1, lon2, ...), (), (), ...]
            [(lat1, lat2, ...), (), (), ...]
            each element contain multiple points defining a polygon, the list define multiple polygons
        lat : list of array-like
            A list of 1D arrays, where each array contains multiple latitude values defining a polygon.
        ID : array-like, optional
            A sequence of IDs for the GeoDataFrame rows. If not provided, the default is the index of the DataFrame.
        crs : str, optional
            The coordinate reference system (CRS) to use. The default is "EPSG:4326".

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the polygons as polygon geometries.

        """
        gdf = pd.DataFrame()
        gdf["ID"] = gdf.index if ID is None else ID
        polygon = geometry.Polygon
        polygon_list = [polygon(zip(lon[i], lat[i])) for i in range(len(lon))]
        gdf["geometry"] = polygon_list
        gdf = gpd.GeoDataFrame(gdf, crs=crs)

        return gdf

    @staticmethod
    def plot():
        """Placeholder method for plotting.

        This method is a placeholder and is not yet implemented.

        """
        pass


def demo1():
    # read data
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
    fpath = os.path.join(__location__, "cases", "01010000.BDY")
    data = pd.read_csv(fpath, sep="  ")
    lon = data.iloc[:, 0].values
    lat = data.iloc[:, 1].values

    # CreateGDF
    creatGDF = CreateGDF()

    # createGDF_rectangle
    rectangle = creatGDF.createGDF_rectangle_central_coord(lon, lat, 0.01)
    print(rectangle)
    rectangle.plot()
    plt.show()

    # create point
    points = creatGDF.createGDF_points(lon, lat)
    print(points)
    points.plot()
    plt.show()

    # create polygon
    polygons = creatGDF.createGDF_polygons([lon], [lat])
    print(polygons)
    polygons.plot()
    plt.show()


def mopex_basin():
    # read data
    home = "F:/data/hydrometeorology/MOPEX/US_Data/Basin_Boundaries"
    fname = [p for p in os.listdir(home) if p.endswith(".BDY") or p.endswith(".bdy")]
    lon_all = []
    lat_all = []
    for n in fname:
        data = pd.read_csv(os.path.join(home, n), sep="  ")
        lon = data.iloc[:, 0].values
        lat = data.iloc[:, 1].values
        lon_all.append(lon)
        lat_all.append(lat)

    # CreateGDF
    creatGDF = CreateGDF()

    # create polygons
    polygons = creatGDF.createGDF_polygons(lon_all, lat_all, ID=fname)
    print(polygons)
    polygons.plot(aspect=1)
    plt.show()


if __name__ == "__main__":
    # demo1()
    # mopex_basin()
    pass
