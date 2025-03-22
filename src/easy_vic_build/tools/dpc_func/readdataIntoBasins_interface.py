# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: readdataIntoBasins_interface

This module contains an API for reading and processing various types of basin-related data, including CAMELS streamflow 
and attribute data. These methods extract data from external sources and integrate them into basin shapefiles, which are 
commonly used in hydrological modeling.

Class:
--------
    - readDataIntoBasins_API: A class providing methods to read CAMELS streamflow and attribute data into basin shapefiles.

Class Methods:
---------------
    - readCAMELSStreamflowIntoBasins: Reads and processes CAMELS streamflow data into the provided basin shapefile.
    - readCAMELSAttributeIntoBasins: Reads and processes CAMELS attribute data into the provided basin shapefile.

Dependencies:
-------------
    - extractData_func: Provides the `Extract_CAMELS_Streamflow.ExtractData` and `Extract_CAMELS_Attribute.ExtractData` functions 
      for extracting CAMELS streamflow and attribute data respectively.
    - logger: A module for logging information during data extraction and processing.
    
Author:
-------
    Xudong Zheng
    Email: z786909151@163.com
    
"""


from .extractData_func import *
from ... import logger

class readDataIntoBasins_API:
    """
    API for reading and processing data into basin geometries.

    This class provides methods for reading specific data sources (e.g., CAMELS streamflow data, CAMELS attributes) 
    and processing them into basin shapefiles.

    Methods
    -------
    readCAMELSStreamflowIntoBasins(basin_shp, read_dates=None) :
        Reads and processes CAMELS streamflow data into the provided basin shapefile.
        
    readCAMELSAttributeIntoBasins(basin_shp, k_list=None) :
        Reads and processes CAMELS attribute data into the provided basin shapefile.
    """
    
    @staticmethod
    def readCAMELSStreamflowIntoBasins(basin_shp, read_dates=None):
        """
        Reads and processes CAMELS streamflow data into basin shapefile.
        
        Parameters
        ----------
        basin_shp : GeoDataFrame
            The GeoDataFrame containing the basin geometries that will be updated with streamflow data.
            
        read_dates : list of str, optional
            A list containing the start and end dates (in the format ['YYYY-MM-DD', 'YYYY-MM-DD']) 
            for filtering the data. Default is None, meaning no filtering is applied.
        
        Returns
        -------
        basin_shp : GeoDataFrame
            The updated GeoDataFrame containing the basin geometries with streamflow data.
            
        Notes
        -----
        This method utilizes the `Extract_CAMELS_Streamflow.ExtractData()` function to perform the data extraction.
        """
        logger.info(f"Reading CAMELS streamflow data for basins with dates: {read_dates}")
        # pd.date_range(start=read_dates[0], end=read_dates[1], freq="D")
        basin_shp = Extract_CAMELS_Streamflow.ExtractData(basin_shp, read_dates=read_dates)
        logger.info("CAMELS streamflow data successfully read into basins.")
        
        return basin_shp

    @staticmethod
    def readCAMELSAttributeIntoBasins(basin_shp, k_list=None):
        """
        Reads and processes CAMELS attribute data into basin shapefile.
        
        Parameters
        ----------
        basin_shp : GeoDataFrame
            The GeoDataFrame containing the basin geometries that will be updated with CAMELS attributes.
            
        k_list : list of str, optional
            A list of keys specifying which attributes to extract. Default is None, meaning all attributes are extracted.
        
        Returns
        -------
        basin_shp : GeoDataFrame
            The updated GeoDataFrame containing the basin geometries with CAMELS attributes.
            
        Notes
        -----
        This method utilizes the `Extract_CAMELS_Attribute.ExtractData()` function to perform the data extraction.
        """
        logger.info(f"Reading CAMELS attribute data with keys: {k_list}")
        basin_shp = Extract_CAMELS_Attribute.ExtractData(basin_shp, k_list=k_list)
        logger.info("CAMELS attribute data successfully read into basins.")
        
        return basin_shp
