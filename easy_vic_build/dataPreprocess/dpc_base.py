# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from typing import Any


# ------------------------ dataProcess class ------------------------

class dataProcess_base:
    """ main/base class: dataProcess_class
    """

    def __init__(self, basin_shp, grid_shp, grid_res, **kwargs):
        self.basin_shp = basin_shp
        self.grid_shp = grid_shp
        self._grid_res = grid_res

    def __call__(self, *args: Any, **kwargs: Any):
        self.read_basin_grid()
        self.readDataIntoBasins()
        self.readDataIntoGrids()
        self.aggregate_grid_to_basins()
        self.readBasinAttribute()
        self.plot()

    def read_basin_grid(self):
        pass
        
    def readDataIntoBasins(self):
        pass

    def readDataIntoGrids(self):
        pass

    def aggregate_grid_to_basins(self):
        pass
    
    def readBasinAttribute(self):
        pass
    
    def plot(self):
        pass

