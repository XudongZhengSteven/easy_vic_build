# code: utf-8
# author: "Xudong Zheng"
# email: z786909151@163.com
""" trans pathdata into map """
import os
import pandas as pd
import geopandas as gpd
from shapely import geometry
from matplotlib import pyplot as plt


class CreateGDF:
    """ Create GDF based on coord """

    def __init__(self, info=""):
        """ init function
        input:

        output:
        """
        self._info = info

    def __call__(self):
        pass

    def createGDF_rectangle_central_coord(self, lon, lat, det, ID=None, crs="EPSG:4326"):
        """ create GeoDataframe for each grids (rectangle) based on the central coord and det/resolution
        input:
            lon/lat: 1d array, define the central grids
            det: float, define the det
            ID: ID for the gdf row
            crs: crs, default is "EPSG:4326"
        """
        gdf = pd.DataFrame(columns=["clon", "clat"])
        gdf["clon"] = lon  # central lon
        gdf["clat"] = lat  # central lat
        gdf["ID"] = gdf.index if ID is None else ID
        polygon = geometry.Polygon
        gdf["geometry"] = gdf.apply(lambda row: polygon([(row.clon - det / 2, row.clat - det / 2),
                                                         (row.clon + det / 2,
                                                          row.clat - det / 2),
                                                         (row.clon + det / 2,
                                                          row.clat + det / 2),
                                                         (row.clon - det / 2, row.clat + det / 2)]), axis=1)
        gdf = gpd.GeoDataFrame(gdf, crs=crs)

        return gdf

    def createGDF_points(self, lon, lat, ID=None, crs="EPSG:4326"):
        """ create GeoDataframe for points based on its lat and lon
        input:
        lon/lat: 1d array, define the points
        ID: ID for the gdf row
        crs: crs, default is "EPSG:4326"
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
        """ create GeoDataframe for polygons based on lon and lat of grids consisting it
        input:
        lon/lat: list of 1d array, each element contain multiple points defining a polygon, the list define multiple polygons
            [(lon1, lon2, ...), (), (), ...]
            [(lat1, lat2, ...), (), (), ...]
        ID: ID for the gdf row
        crs: crs, default is "EPSG:4326"
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
        pass


def demo1():
    # read data
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
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
