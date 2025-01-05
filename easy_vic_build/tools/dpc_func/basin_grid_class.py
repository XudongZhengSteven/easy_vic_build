# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import geopandas as gpd
import numpy as np
import pandas as pd
import os
import shapely
import math
from ..geo_func.create_gdf import CreateGDF


class Basins(gpd.GeoDataFrame):

    def __add__(self, basins):
        pass

    def __sub__(self, basins):
        pass

    def __and__(self, basins):
        pass


class Grids(gpd.GeoDataFrame):

    def __add__(self, grids):
        pass

    def __sub__(self, grids):
        pass

    def __and__(self, grids):
        pass


class Basins_from_shapefile(Basins):
    def __init__(self, shapefile_path, data=None, *args, geometry=None, crs=None, **kwargs):
        shp_gdf = gpd.read_file(shapefile_path)
        super().__init__(shp_gdf, *args, geometry=geometry, crs=crs, **kwargs)


class HCDNBasins(Basins):
    def __init__(self, home="E:\\data\\hydrometeorology\\CAMELS", data=None, *args, geometry=None, crs=None, **kwargs):
        HCDN_shp_path = os.path.join(home, "basin_set_full_res", "HCDN_nhru_final_671.shp")
        HCDN_shp = gpd.read_file(HCDN_shp_path)
        HCDN_shp["AREA_km2"] = HCDN_shp.AREA / 1000000  # m2 -> km2
        super().__init__(HCDN_shp, *args, geometry=geometry, crs=crs, **kwargs)


class HCDNGrids(Grids):
    def __init__(self, home, *args, data=None, geometry=None, crs=None, **kwargs):
        grid_shp_label_path = os.path.join(home, "map", "grids_0_25_label.shp")
        grid_shp_label = gpd.read_file(grid_shp_label_path)
        grid_shp_path = os.path.join(home, "map", "grids_0_25.shp")
        grid_shp = gpd.read_file(grid_shp_path)
        grid_shp["point_geometry"] = grid_shp_label.geometry
        
        super().__init__(grid_shp, *args, geometry=geometry, crs=crs, **kwargs)

    def createBoundaryShp(self):
        boundary_point_center_shp, boundary_point_center_x_y, boundary_grids_edge_shp, boundary_grids_edge_x_y = createBoundaryShp(self)
        return boundary_point_center_shp, boundary_point_center_x_y, boundary_grids_edge_shp, boundary_grids_edge_x_y


def createBoundaryShp(grid_shp):
    # boundary: point center
    cgdf_point = CreateGDF()
    boundary_x_min = min(grid_shp["point_geometry"].x)
    boundary_x_max = max(grid_shp["point_geometry"].x)
    boundary_y_min = min(grid_shp["point_geometry"].y)
    boundary_y_max = max(grid_shp["point_geometry"].y)
    boundary_point_center_shp = cgdf_point.createGDF_polygons(lon=[[boundary_x_min, boundary_x_max, boundary_x_max, boundary_x_min]],
                                           lat=[[boundary_y_max, boundary_y_max, boundary_y_min, boundary_y_min]],
                                           crs=grid_shp.crs)
    boundary_point_center_x_y = [boundary_x_min, boundary_y_min, boundary_x_max, boundary_y_max]
    
    # boundary: grids edge
    boundary_x_min = min(grid_shp["geometry"].get_coordinates().x)
    boundary_x_max = max(grid_shp["geometry"].get_coordinates().x)
    boundary_y_min = min(grid_shp["geometry"].get_coordinates().y)
    boundary_y_max = max(grid_shp["geometry"].get_coordinates().y)
    
    boundary_grids_edge_shp = cgdf_point.createGDF_polygons(lon=[[boundary_x_min, boundary_x_max, boundary_x_max, boundary_x_min]],
                                           lat=[[boundary_y_max, boundary_y_max, boundary_y_min, boundary_y_min]],
                                           crs=grid_shp.crs)
    boundary_grids_edge_x_y = [boundary_x_min, boundary_y_min, boundary_x_max, boundary_y_max]
    
    return boundary_point_center_shp, boundary_point_center_x_y, boundary_grids_edge_shp, boundary_grids_edge_x_y


class Grids_for_shp(Grids):
    def __init__(self, gshp, *args,
                 cen_lons=None, cen_lats=None, stand_lons=None, stand_lats=None,
                 res=None, adjust_boundary=True, geometry=None, crs=None, expand_grids_num=0, boundary=None, **kwargs):
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
        grid_polygon = lambda xmin, xmax, ymin, ymax: shapely.geometry.Polygon([(xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)])
        grid_point = lambda x, y: shapely.geometry.Point(x, y)
        
        # create grid_shp
        grid_shp = gpd.GeoDataFrame()
        
        if res:
            # construct grids based on given cen_lons: do not consider gshp boundary
            if cen_lons is not None: # *note: len(cen_lons) == len(cen_lats)
                grid_shp.loc[:, "geometry"] = [grid_polygon(cen_lons[i]-res/2, cen_lons[i]+res/2, cen_lats[i]-res/2, cen_lats[i]+res/2) for i in range(len(cen_lats))]
                grid_shp.loc[:, "point_geometry"] = [grid_point(cen_lons[i], cen_lats[i]) for i in range(len(cen_lats))]
            
            # construct grids based on standard grids: clip based on gshp boundary
            elif stand_lons is not None:
                
                cen_lons = stand_lons[np.where(stand_lons - res/2 <= boundary_x_min - res*expand_grids_num)[0][-1]: np.where(stand_lons + res/2 >= boundary_x_max + res*expand_grids_num)[0][0] + 1]
                cen_lats = stand_lats[np.where(stand_lats - res/2 <= boundary_y_min - res*expand_grids_num)[0][-1]: np.where(stand_lats + res/2 >= boundary_y_max + res*expand_grids_num)[0][0] + 1]
                
                cen_lons, cen_lats = np.meshgrid(cen_lons, cen_lats)
                cen_lons = cen_lons.flatten()
                cen_lats = cen_lats.flatten()
                
                grid_shp.loc[:, "geometry"] = [grid_polygon(cen_lons[i]-res/2, cen_lons[i]+res/2, cen_lats[i]-res/2, cen_lats[i]+res/2) for i in range(len(cen_lats))]
                grid_shp.loc[:, "point_geometry"] = [grid_point(cen_lons[i], cen_lats[i]) for i in range(len(cen_lats))]

            # construct grids based on boundary
            else:
                if adjust_boundary:
                    boundary_x_min = math.floor(boundary_x_min / res) * res
                    boundary_x_max = math.ceil(boundary_x_max / res) * res
                    boundary_y_min = math.floor(boundary_y_min / res) * res
                    boundary_y_max = math.ceil(boundary_y_max / res) * res
                
                cen_lons = np.arange((boundary_x_min + res/2 - res*expand_grids_num), (boundary_x_max + res*expand_grids_num), res)
                cen_lats = np.arange((boundary_y_min + res/2 - res*expand_grids_num), (boundary_y_max + res*expand_grids_num), res)

                # cen_lons = np.arange(math.floor((boundary_x_min + res/2) / (res/2)) * (res/2), math.ceil((boundary_x_max + res/2) / (res/2)) * (res/2), res)
                # cen_lats = np.arange(math.floor((boundary_y_min + res/2) / (res/2)) * (res/2), math.ceil((boundary_y_max + res/2) / (res/2)) * (res/2), res)

                cen_lons, cen_lats = np.meshgrid(cen_lons, cen_lats)
                cen_lons = cen_lons.flatten()
                cen_lats = cen_lats.flatten()
                
                grid_shp.loc[:, "geometry"] = [grid_polygon(cen_lons[i]-res/2, cen_lons[i]+res/2, cen_lats[i]-res/2, cen_lats[i]+res/2) for i in range(len(cen_lats))]
                grid_shp.loc[:, "point_geometry"] = [grid_point(cen_lons[i], cen_lats[i]) for i in range(len(cen_lats))]
        
        # res=None, one grid for this shp (boundary grid)
        else:
            grid_shp.loc[0, "geometry"] = grid_polygon(boundary_x_min, boundary_x_max, boundary_y_min, boundary_y_max)
            grid_shp.loc[0, "point_geometry"] = grid_point((boundary_x_min + boundary_x_max)/2, (boundary_y_min + boundary_y_max)/2)
        
        grid_shp = grid_shp.set_geometry("point_geometry")
        crs = crs if crs is not None else "EPSG:4326"
        grid_shp = grid_shp.set_crs(crs)
        
        super().__init__(grid_shp, *args, geometry=geometry, crs=crs, **kwargs)
    
    def createBoundaryShp(self):
        boundary_point_center_shp, boundary_point_center_x_y, boundary_grids_edge_shp, boundary_grids_edge_x_y = createBoundaryShp(self)
        return boundary_point_center_shp, boundary_point_center_x_y, boundary_grids_edge_shp, boundary_grids_edge_x_y
