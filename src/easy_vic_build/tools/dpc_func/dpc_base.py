# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: dpc_base

This module provides a base class for processing and managing data related to basins and grids in hydrological 
and geospatial analyses. The `dataProcess_base` class serves as a template for reading basin and grid data, 
aggregating grid data to basins, and visualizing the results. It is designed to be subclassed or extended to 
accommodate specific data types or processing steps for particular hydrological modeling needs.

Class:
--------
    - dataProcess_base: A base class that implements a general workflow for processing basin and grid data, 
      including reading, aggregating, and visualizing the data. Specific methods for handling data types or 
      processing steps should be implemented in subclasses.

Class Methods:
---------------
    - __call__(self, *args: Any, **kwargs: Any): Executes the full data processing pipeline: reading basin 
      and grid data, aggregating grid data to basins, and plotting results.
    - read_basin_grid(self): Placeholder method to read basin grid data. To be extended with specific logic.
    - readDataIntoBasins(self): Placeholder method to read data into basins. To be extended with specific logic.
    - readDataIntoGrids(self): Placeholder method to read data into grids. To be extended with specific logic.
    - aggregate_grid_to_basins(self): Placeholder method to aggregate grid data to basins. To be extended with 
      specific logic.
    - readBasinAttribute(self): Placeholder method to read basin attributes. To be extended with specific logic.
    - plot(self): Placeholder method to plot the results of the data processing. To be extended with specific logic.

Usage:
------
    1. Instantiate the `dataProcess_base` class with the required basin and grid data:
        - `dp = dataProcess_base(basin_shp, grid_shp, grid_res)`
    2. Call the `__call__` method to trigger the full data processing pipeline:
        - `dp()`
    3. Implement specific logic in subclassed methods to customize data reading, aggregation, and plotting.

Example:
--------
    dp = dataProcess_base(basin_shp, grid_shp, grid_res)
    dp()

Dependencies:
-------------
    - geopandas: For handling and processing spatial data (GeoDataFrame).
    - numpy: For numerical operations and array manipulation.
    
Author:
-------
    Xudong Zheng
    Email: z786909151@163.com
    
"""


from typing import Any

class dataProcess_base:
    """
    Base class for processing data related to basins and grids in hydrological and geospatial analyses.
    
    This class serves as a template for reading basin and grid data, aggregating grid data to basins, 
    and plotting the results. It is meant to be subclassed or extended with specific implementation 
    details for particular data types or processing steps.
    
    Attributes:
    -----------
    basin_shp : GeoDataFrame
        A GeoDataFrame representing the basins, used for spatial operations.
    grid_shp : GeoDataFrame
        A GeoDataFrame representing the grids, used for spatial operations.
    _grid_res : float
        Resolution of the grid.
    
    Methods:
    --------
    __call__(self, *args: Any, **kwargs: Any):
        Executes the full data processing pipeline: reading basin and grid data, 
        aggregating data, and plotting results.
        
    read_basin_grid(self):
        Placeholder method to read basin grid data.
        
    readDataIntoBasins(self):
        Placeholder method to read data into basins.
        
    readDataIntoGrids(self):
        Placeholder method to read data into grids.
        
    aggregate_grid_to_basins(self):
        Placeholder method to aggregate grid data to basins.
        
    readBasinAttribute(self):
        Placeholder method to read basin attributes.
        
    plot(self):
        Placeholder method to plot the results.
    """

    def __init__(self, basin_shp, grid_shp, grid_res, **kwargs):
        """
        Initializes the data processing class with the provided basin and grid data.

        Parameters:
        -----------
        basin_shp : GeoDataFrame
            A GeoDataFrame containing basin geometries and attributes.
        grid_shp : GeoDataFrame
            A GeoDataFrame containing grid geometries and attributes.
        grid_res : float
            The resolution of the grid.
        kwargs : additional keyword arguments
            Other optional parameters that may be used for specific processing steps.
        """
        self.basin_shp = basin_shp
        self.grid_shp = grid_shp
        self._grid_res = grid_res

    def __call__(self, *args: Any, **kwargs: Any):
        """
        Executes the full data processing pipeline: reading basin and grid data, 
        aggregating grid data to basins, and plotting results.
        
        This method serves as the main entry point for triggering all the steps 
        involved in the data processing workflow.

        Parameters:
        -----------
        *args : Any
            Positional arguments passed to methods.
        **kwargs : Any
            Keyword arguments passed to methods.
        """
        self.read_basin_grid()
        self.readDataIntoBasins()
        self.readDataIntoGrids()
        self.aggregate_grid_to_basins()
        self.readBasinAttribute()
        self.plot()

    def read_basin_grid(self):
        """
        Reads the grid data associated with each basin.
        
        This is a placeholder method intended to be overridden or extended 
        with specific logic for reading grid data.
        """
        pass
        
    def readDataIntoBasins(self):
        """
        Reads and processes data into the basin geometries.
        
        This is a placeholder method intended to be overridden or extended 
        with specific logic for reading data into basins.
        """
        pass

    def readDataIntoGrids(self):
        """
        Reads and processes data into the grid geometries.
        
        This is a placeholder method intended to be overridden or extended 
        with specific logic for reading data into grids.
        """
        pass

    def aggregate_grid_to_basins(self):
        """
        Aggregates the grid-based data to basin-level results.
        
        This is a placeholder method intended to be overridden or extended 
        with specific logic for aggregating data from grids to basins.
        """
        pass
    
    def readBasinAttribute(self):
        """
        Reads additional attributes for each basin.
        
        This is a placeholder method intended to be overridden or extended 
        with specific logic for reading basin attributes.
        """
        pass
    
    def plot(self):
        """
        Plots the results of the data processing.
        
        This is a placeholder method intended to be overridden or extended 
        with specific logic for visualizing the processed data.
        """
        pass

