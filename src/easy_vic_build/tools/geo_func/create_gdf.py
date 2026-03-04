# code: utf-8
# author: "Xudong Zheng"
# email: z786909151@163.com

"""GeoDataFrame construction helpers for common geometry types."""


import os

import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
from shapely import geometry


class CreateGDF:
    """Build GeoDataFrames from coordinate arrays.

    Attributes
    ----------
    _info : str
        Optional metadata string attached to the instance.

    Methods
    -------
    createGDF_rectangle_central_coord(lon, lat, det, ID=None, crs="EPSG:4326")
        Build rectangular-cell polygons from center coordinates.
    createGDF_points(lon, lat, ID=None, crs="EPSG:4326")
        Build point geometries from longitude/latitude pairs.
    createGDF_polygons(lon, lat, ID=None, crs="EPSG:4326")
        Build polygon geometries from per-feature vertex coordinates.
    """

    def __init__(self, info=""):
        """Initialize a :class:`CreateGDF` instance.

        Parameters
        ----------
        info : str, optional
            Optional metadata string.
        """
        self._info = info

    def __call__(self):
        """Placeholder for callable usage."""
        pass

    def createGDF_rectangle_central_coord(
        self, lon, lat, det, ID=None, crs="EPSG:4326"
    ):
        """Create rectangular polygons from center coordinates.

        Parameters
        ----------
        lon : array-like
            Longitude of cell centers.
        lat : array-like
            Latitude of cell centers.
        det : float
            Cell size (in coordinate units). A square with side ``det`` is generated
            around each center point.
        ID : array-like, optional
            Row identifiers. If ``None``, DataFrame index values are used.
        crs : str, optional
            Coordinate reference system. Default is ``"EPSG:4326"``.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with ``clon``, ``clat``, ``ID``, and polygon ``geometry``.
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
        """Create point geometries from longitude/latitude coordinates.

        Parameters
        ----------
        lon : array-like
            Point longitudes.
        lat : array-like
            Point latitudes.
        ID : array-like, optional
            Row identifiers. If ``None``, DataFrame index values are used.
        crs : str, optional
            Coordinate reference system. Default is ``"EPSG:4326"``.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with ``lon``, ``lat``, ``ID``, and point ``geometry``.
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
        """Create polygons from per-feature vertex coordinates.

        Parameters
        ----------
        lon : list of array-like
            Per-polygon longitude sequences. Each element defines one polygon.
        lat : list of array-like
            Per-polygon latitude sequences. Must align with ``lon`` by index.
        ID : array-like, optional
            Row identifiers. If ``None``, DataFrame index values are used.
        crs : str, optional
            Coordinate reference system. Default is ``"EPSG:4326"``.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with ``ID`` and polygon ``geometry``.
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
        """Placeholder for future plotting helper."""
        pass


def demo1():
    """
    Demonstrate GeoDataFrame creation from a sample boundary file.

    Returns
    -------
    None
        This function only demonstrates plotting and printing outputs.
    """
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
    """
    Demonstrate polygon creation for MOPEX basin boundary files.

    Returns
    -------
    None
        This function only demonstrates plotting and printing outputs.
    """
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
