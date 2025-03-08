# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
from typing import Any
from .dpc_base import dataProcess_base
from .readdataIntoGrids_interface import readDataIntoGrids_API
from .readdataIntoBasins_interface import readDataIntoBasins_API
import matplotlib.pyplot as plt
from  .basin_grid_class import *
from ..utilities import *
from ..plot_func.plot_func import *

class dataProcess_VIC_level0(dataProcess_base):
    
    def __init__(self, basin_shp, grid_shp, grid_res, date_period, **kwargs):
        self.date_period = date_period
        super().__init__(basin_shp, grid_shp, grid_res, **kwargs)
    
    def __call__(self, *args: Any, readBasindata=False, readGriddata=True, readBasinAttribute=False, **kwargs: Any):
        self.read_basin_grid()
        
        if readBasindata:
            self.readDataIntoBasins()
        
        if readGriddata:
            self.readDataIntoGrids()
        
        if readBasinAttribute:
            self.readBasinAttribute()
    
    def read_basin_grid(self):
        self.createBoundaryShp()
        
    def createBoundaryShp(self):
        self.boundary_point_center_shp, self.boundary_point_center_x_y, self.boundary_grids_edge_shp, self.boundary_grids_edge_x_y = self.grid_shp.createBoundaryShp()
        
    def readDataIntoGrids(self):
        self.grid_shp = readDataIntoGrids_API.readSrtmDEMIntoGrids(self.grid_shp, grid_shp_res=self._grid_res, plot=False, save_original=False, check_search=False)
        self.grid_shp = readDataIntoGrids_API.readCONUSSoilIntoGrids(self.grid_shp, grid_shp_res=self._grid_res, plot_layer=False, save_original=False, check_search=False)
    
    def plot(self, fig=None, ax=None, grid_shp_kwargs=dict(), grid_shp_point_kwargs=dict(), basin_shp_kwargs=dict()):
        # plot grid_shp and basin_shp
        if fig is None:
            fig, ax = plt.subplots()
        
        # plot kwargs
        grid_shp_kwargs_all = {"edgecolor": "k", "alpha": 0.5, "linewidth": 0.5}
        grid_shp_kwargs_all.update(grid_shp_kwargs)
        
        grid_shp_point_kwargs_all = {"alpha": 0.5, "facecolor": "k", "markersize": 1}
        grid_shp_point_kwargs_all.update(grid_shp_point_kwargs)
        
        basin_shp_kwargs_all = {"edgecolor": "k", "alpha": 0.5, "facecolor": "b"}
        basin_shp_kwargs_all.update(basin_shp_kwargs)
        
        # plot
        self.grid_shp.boundary.plot(ax=ax, **grid_shp_kwargs_all)
        self.grid_shp["point_geometry"].plot(ax=ax, **grid_shp_point_kwargs_all)
        self.basin_shp.plot(ax=ax, **basin_shp_kwargs_all)
        
        boundary_x_y = self.boundary_grids_edge_x_y
        ax.set_xlim(boundary_x_y[0], boundary_x_y[2])
        ax.set_ylim(boundary_x_y[1], boundary_x_y[3])
        
        return fig, ax
    
    def plot_grid_column(self, column, fig=None, ax=None, grid_shp_kwargs=dict(), column_kwargs=dict(), basin_shp_kwargs=dict()):
        # plot grid_shp column
        if fig is None:
            fig, ax = plt.subplots()
        
        # plot kwargs
        grid_shp_kwargs_all = {"edgecolor": "k", "alpha": 0.5, "linewidth": 0.5}
        grid_shp_kwargs_all.update(grid_shp_kwargs)
        
        column_kwargs_all = {"cmap": "terrain", "legend": True}
        column_kwargs_all.update(column_kwargs)
        
        basin_shp_kwargs_all = {"edgecolor": "k", "alpha": 0.5, "facecolor": "b"}
        basin_shp_kwargs_all.update(basin_shp_kwargs)
        
        # plot
        self.grid_shp.boundary.plot(ax=ax, **grid_shp_kwargs_all)
        self.grid_shp.plot(column=column, ax=ax, **column_kwargs_all)
        self.basin_shp.plot(ax=ax, **basin_shp_kwargs_all)
        
        boundary_x_y = self.boundary_grids_edge_x_y
        ax.set_xlim(boundary_x_y[0], boundary_x_y[2])
        ax.set_ylim(boundary_x_y[1], boundary_x_y[3])
        
        return fig, ax
        

class dataProcess_VIC_level1(dataProcess_VIC_level0):
        
    def readDataIntoBasins(self):
        self.basin_shp = readDataIntoBasins_API.readCAMELSStreamflowIntoBasins(self.basin_shp, read_dates=self.date_period)
    
    def readBasinAttribute(self):
        self.basin_shp = readDataIntoBasins_API.readCAMELSAttributeIntoBasins(self.basin_shp, k_list=None)
        
    def readDataIntoGrids(self):
        self.grid_shp = readDataIntoGrids_API.readERA5_SoilTemperatureIntoGrids(self.grid_shp, grid_shp_res=self._grid_res, plot_layer=False, check_search=False)
        self.grid_shp = readDataIntoGrids_API.readNLDAS_annual_PIntoGrids(self.grid_shp, grid_shp_res=self._grid_res, plot=False, check_search=False)
        self.grid_shp = readDataIntoGrids_API.readUMDLandCoverIntoGrids(self.grid_shp, grid_shp_res=self._grid_res, plot=False, save_original=True, check_search=False)
        self.grid_shp = readDataIntoGrids_API.readMODISBSAIntoGrids(self.grid_shp, grid_shp_res=self._grid_res, plot_month=False, save_original=True, check_search=False)
        self.grid_shp = readDataIntoGrids_API.readMODISNDVIIntoGrids(self.grid_shp, grid_shp_res=self._grid_res, plot_month=False, save_original=True, check_search=False)
        self.grid_shp = readDataIntoGrids_API.readMODISLAIIntoGrids(self.grid_shp, grid_shp_res=self._grid_res, plot_month=False, save_original=True, check_search=False)       


class dataProcess_VIC_level2(dataProcess_VIC_level0):
    
    def readDataIntoBasins(self):
        pass
    
    def readBasinAttribute(self):
        pass
        
    def readDataIntoGrids(self):
        pass


class dataProcess_CAMELS_review(dataProcess_base):
    """_summary_: just read basin and grid of CAMELS for review

    Args:
        dataProcess_CAMELS (_type_): _description_
    """
    def __init__(self, HCDN_home, basin_shp=None, grid_shp=None, grid_res=None, **kwargs):
        super().__init__(basin_shp, grid_shp, grid_res, **kwargs)
        self._HCDN_home = HCDN_home
        self.read_basin_grid()
        
    def read_basin_grid(self):
        # read basin shp
        self.basin_shp = HCDNBasins(self._HCDN_home)
        # self.basin_shp_original = HCDNBasins(self._HCDN_home)  # backup for HCDN_shp
        
        # read grids and createBoundaryShp
        self.grid_shp = HCDNGrids(self._HCDN_home)
        self.createBoundaryShp()
        
    def createBoundaryShp(self):
        self.boundary_point_center_shp, self.boundary_point_center_x_y, self.boundary_grids_edge_shp, self.boundary_grids_edge_x_y = self.grid_shp.createBoundaryShp()
    
    def __call__(self, plot=True):
        print("=============== basin_shp ===============")
        print(self.basin_shp)
        
        print("=============== grid_shp ===============")
        print(self.grid_shp)
        
        if plot:
            self.plot()

    def plot(self):
        fig, ax = plotBackground(self.basin_shp, self.grid_shp, fig=None, ax=None)
        ax = setBoundary(ax, *self.boundary_grids_edge_x_y)
        
        return fig, ax


if __name__ == "__main__":
    # general set
    root, home = setHomePath(root="E:")

    # review
    dpc_review = dataProcess_CAMELS_review(HCDN_home=home)
    dpc_review()
    
