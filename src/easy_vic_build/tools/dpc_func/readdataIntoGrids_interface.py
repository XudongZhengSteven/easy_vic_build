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
    - readDataIntoGrids_API: A class providing methods to read and process grid-based data from multiple sources.

Class Methods:
---------------
    - readSrtmDEMIntoGrids: Reads SRTM DEM data into a grid shapefile.
    - readCONUSSoilIntoGrids: Reads CONUS soil data into a grid shapefile.
    - readERA5_SoilTemperatureIntoGrids: Reads ERA5 soil temperature data into a grid shapefile.
    - readNLDAS_annual_PIntoGrids: Reads NLDAS annual precipitation data into a grid shapefile.
    - readUMDLandCoverIntoGrids: Reads UMD land cover data into a grid shapefile.
    - readMODISBSAIntoGrids: Reads MODIS BSA (burned area) data into a grid shapefile.
    - readMODISNDVIIntoGrids: Reads MODIS NDVI (Normalized Difference Vegetation Index) data into a grid shapefile.
    - readMODISLAIIntoGrids: Reads MODIS LAI (Leaf Area Index) data into a grid shapefile.

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

class readDataIntoGrids_API:
    """
    API for reading and processing grid-based data.

    This class provides methods for reading and processing various environmental datasets (e.g., SRTM DEM, ERA5, MODIS) 
    into grid shapefiles, which are often used in hydrological modeling and environmental studies.

    Methods
    -------
    readSrtmDEMIntoGrids(grid_shp, grid_shp_res=0.25, plot=False, save_original=False, check_search=False) :
        Reads and processes SRTM DEM data into the provided grid shapefile.

    readCONUSSoilIntoGrids(grid_shp, grid_shp_res=0.125, plot_layer=1, save_original=True, check_search=False) :
        Reads and processes CONUS soil data into the provided grid shapefile.

    readERA5_SoilTemperatureIntoGrids(grid_shp, grid_shp_res=0.125, plot_layer=False, check_search=False) :
        Reads and processes ERA5 soil temperature data into the provided grid shapefile.

    readNLDAS_annual_PIntoGrids(grid_shp, grid_shp_res=0.125, plot=False, check_search=False) :
        Reads and processes NLDAS annual precipitation data into the provided grid shapefile.

    readUMDLandCoverIntoGrids(grid_shp, grid_shp_res=0.125, plot=True, save_original=False, check_search=False) :
        Reads and processes UMD land cover data into the provided grid shapefile.

    readMODISBSAIntoGrids(grid_shp, grid_shp_res=0.125, plot_month=False, save_original=True, check_search=False) :
        Reads and processes MODIS BSA (burned area) data into the provided grid shapefile.

    readMODISNDVIIntoGrids(grid_shp, grid_shp_res=0.125, plot_month=False, save_original=True, check_search=False) :
        Reads and processes MODIS NDVI (Normalized Difference Vegetation Index) data into the provided grid shapefile.

    readMODISLAIIntoGrids(grid_shp, grid_shp_res=0.125, plot_month=False, save_original=True, check_search=False) :
        Reads and processes MODIS LAI (Leaf Area Index) data into the provided grid shapefile.
    """
    
    @staticmethod
    def readSrtmDEMIntoGrids(grid_shp, grid_shp_res=0.25, plot=False, save_original=False, check_search=False):
        """
        Reads and processes SRTM DEM data into the provided grid shapefile.

        Parameters
        ----------
        grid_shp : GeoDataFrame
            The GeoDataFrame containing the grid geometries to which the SRTM DEM data will be added.

        grid_shp_res : float, optional
            The spatial resolution for the grid shapefile. Default is 0.25 degrees.

        plot : bool, optional
            Whether to plot the DEM data. Default is False.

        save_original : bool, optional
            Whether to save the original data. Default is False.

        check_search : bool, optional
            Whether to perform a search check. Default is False.

        Returns
        -------
        grid_shp : GeoDataFrame
            The updated GeoDataFrame containing the grid geometries with SRTM DEM data.

        Notes
        -----
        This method utilizes the `Extract_SrtmDEM.ExtractData()` function to perform the data extraction.
        """
        logger.info(f"Reading SRTM DEM data with resolution {grid_shp_res} into grid.")
        grid_shp = Extract_SrtmDEM.ExtractData(grid_shp, grid_shp_res=grid_shp_res, plot=plot, save_original=save_original, check_search=check_search)
        logger.info("SRTM DEM data successfully read into grids.")
        
        return grid_shp

    @staticmethod
    def readCONUSSoilIntoGrids(grid_shp, grid_shp_res=0.125, plot_layer=1, save_original=True, check_search=False):
        """
        Reads and processes CONUS soil data into the provided grid shapefile.

        Parameters
        ----------
        grid_shp : GeoDataFrame
            The GeoDataFrame containing the grid geometries to which the CONUS soil data will be added.

        grid_shp_res : float, optional
            The spatial resolution for the grid shapefile. Default is 0.125 degrees.

        plot_layer : int, optional
            The specific soil layer to plot. Default is 1.

        save_original : bool, optional
            Whether to save the original data. Default is True.

        check_search : bool, optional
            Whether to perform a search check. Default is False.

        Returns
        -------
        grid_shp : GeoDataFrame
            The updated GeoDataFrame containing the grid geometries with CONUS soil data.

        Notes
        -----
        This method utilizes the `Extract_CONUS_SOIL.ExtractData()` function to perform the data extraction.
        """
        logger.info(f"Reading CONUS soil data into grid with resolution {grid_shp_res}.")
        grid_shp = Extract_CONUS_SOIL.ExtractData(grid_shp, grid_shp_res, plot_layer, save_original, check_search)
        logger.info("CONUS soil data successfully read into grids.")
        return grid_shp

    @staticmethod
    def readERA5_SoilTemperatureIntoGrids(grid_shp, grid_shp_res=0.125, plot_layer=False, check_search=False):
        """
        Reads and processes ERA5 soil temperature data into the provided grid shapefile.

        Parameters
        ----------
        grid_shp : GeoDataFrame
            The GeoDataFrame containing the grid geometries to which the ERA5 soil temperature data will be added.

        grid_shp_res : float, optional
            The spatial resolution for the grid shapefile. Default is 0.125 degrees.

        plot_layer : bool, optional
            Whether to plot the soil temperature data. Default is False.

        check_search : bool, optional
            Whether to perform a search check. Default is False.

        Returns
        -------
        grid_shp : GeoDataFrame
            The updated GeoDataFrame containing the grid geometries with ERA5 soil temperature data.

        Notes
        -----
        This method utilizes the `Extract_ERA5_SoilTemperature.ExtractData()` function to perform the data extraction.
        """
        logger.info(f"Reading ERA5 soil temperature data into grid with resolution {grid_shp_res}.")
        grid_shp = Extract_ERA5_SoilTemperature.ExtractData(grid_shp, grid_shp_res, plot_layer, check_search)
        logger.info("ERA5 soil temperature data successfully read into grids.")
        return grid_shp

    @staticmethod
    def readNLDAS_annual_PIntoGrids(grid_shp, grid_shp_res=0.125, plot=False, check_search=False):
        """
        Reads and processes NLDAS annual precipitation data into the provided grid shapefile.

        Parameters
        ----------
        grid_shp : GeoDataFrame
            The GeoDataFrame containing the grid geometries to which the NLDAS annual precipitation data will be added.

        grid_shp_res : float, optional
            The spatial resolution for the grid shapefile. Default is 0.125 degrees.

        plot : bool, optional
            Whether to plot the precipitation data. Default is False.

        check_search : bool, optional
            Whether to perform a search check. Default is False.

        Returns
        -------
        grid_shp : GeoDataFrame
            The updated GeoDataFrame containing the grid geometries with NLDAS annual precipitation data.

        Notes
        -----
        This method utilizes the `Extract_NLDAS_annual_P.ExtractData()` function to perform the data extraction.
        """
        logger.info(f"Reading NLDAS annual precipitation data into grid with resolution {grid_shp_res}.")
        grid_shp = Extract_NLDAS_annual_P.ExtractData(grid_shp, grid_shp_res, plot, check_search)
        logger.info("NLDAS annual precipitation data successfully read into grids.")
        
        return grid_shp

    @staticmethod
    def readUMDLandCoverIntoGrids(grid_shp, grid_shp_res=0.125, plot=True, save_original=False, check_search=False):
        """
        Reads and processes UMD land cover data into the provided grid shapefile.

        Parameters
        ----------
        grid_shp : GeoDataFrame
            The GeoDataFrame containing the grid geometries to which the UMD land cover data will be added.

        grid_shp_res : float, optional
            The spatial resolution for the grid shapefile. Default is 0.125 degrees.

        plot : bool, optional
            Whether to plot the land cover data. Default is True.

        save_original : bool, optional
            Whether to save the original data. Default is False.

        check_search : bool, optional
            Whether to perform a search check. Default is False.

        Returns
        -------
        grid_shp : GeoDataFrame
            The updated GeoDataFrame containing the grid geometries with UMD land cover data.

        Notes
        -----
        This method utilizes the `Extract_UMD_1km.ExtractData()` function to perform the data extraction.
        """
        logger.info(f"Reading UMD land cover data into grid with resolution {grid_shp_res}.")
        grid_shp = Extract_UMD_1km.ExtractData(grid_shp, grid_shp_res, plot, save_original, check_search)
        logger.info("UMD land cover data successfully read into grids.")
        
        return grid_shp

    @staticmethod
    def readMODISBSAIntoGrids(grid_shp, grid_shp_res=0.125, plot_month=False, save_original=True, check_search=False):
        """
        Reads and processes MODIS BSA (burned area) data into the provided grid shapefile.

        Parameters
        ----------
        grid_shp : GeoDataFrame
            The GeoDataFrame containing the grid geometries to which the MODIS BSA data will be added.

        grid_shp_res : float, optional
            The spatial resolution for the grid shapefile. Default is 0.125 degrees.

        plot_month : bool, optional
            Whether to plot the BSA data by month. Default is False.

        save_original : bool, optional
            Whether to save the original data. Default is True.

        check_search : bool, optional
            Whether to perform a search check. Default is False.

        Returns
        -------
        grid_shp : GeoDataFrame
            The updated GeoDataFrame containing the grid geometries with MODIS BSA data.

        Notes
        -----
        This method utilizes the `Extract_MODIS_BSA.ExtractData()` function to perform the data extraction.
        """
        logger.info(f"Reading MODIS BSA data into grid with resolution {grid_shp_res}.")
        grid_shp = Extract_MODIS_BSA.ExtractData(grid_shp, grid_shp_res, plot_month, save_original, check_search)
        logger.info("MODIS BSA data successfully read into grids.")
        return grid_shp

    @staticmethod
    def readMODISNDVIIntoGrids(grid_shp, grid_shp_res=0.125, plot_month=False, save_original=True, check_search=False):
        """
        Reads and processes MODIS NDVI (Normalized Difference Vegetation Index) data into the provided grid shapefile.

        Parameters
        ----------
        grid_shp : GeoDataFrame
            The GeoDataFrame containing the grid geometries to which the MODIS NDVI data will be added.

        grid_shp_res : float, optional
            The spatial resolution for the grid shapefile. Default is 0.125 degrees.

        plot_month : bool, optional
            Whether to plot the NDVI data by month. Default is False.

        save_original : bool, optional
            Whether to save the original data. Default is True.

        check_search : bool, optional
            Whether to perform a search check. Default is False.

        Returns
        -------
        grid_shp : GeoDataFrame
            The updated GeoDataFrame containing the grid geometries with MODIS NDVI data.

        Notes
        -----
        This method utilizes the `Extract_MODIS_NDVI.ExtractData()` function to perform the data extraction.
        """
        logger.info(f"Reading MODIS NDVI data into grid with resolution {grid_shp_res}.")
        grid_shp = Extract_MODIS_NDVI.ExtractData(grid_shp, grid_shp_res, plot_month, save_original, check_search)
        logger.info("MODIS NDVI data successfully read into grids.")
        return grid_shp

    @staticmethod
    def readMODISLAIIntoGrids(grid_shp, grid_shp_res=0.125, plot_month=False, save_original=True, check_search=False):
        """
        Reads and processes MODIS LAI (Leaf Area Index) data into the provided grid shapefile.

        Parameters
        ----------
        grid_shp : GeoDataFrame
            The GeoDataFrame containing the grid geometries to which the MODIS LAI data will be added.

        grid_shp_res : float, optional
            The spatial resolution for the grid shapefile. Default is 0.125 degrees.

        plot_month : bool, optional
            Whether to plot the LAI data by month. Default is False.

        save_original : bool, optional
            Whether to save the original data. Default is True.

        check_search : bool, optional
            Whether to perform a search check. Default is False.

        Returns
        -------
        grid_shp : GeoDataFrame
            The updated GeoDataFrame containing the grid geometries with MODIS LAI data.

        Notes
        -----
        This method utilizes the `Extract_MODIS_LAI.ExtractData()` function to perform the data extraction.
        """
        logger.info(f"Reading MODIS LAI data into grid with resolution {grid_shp_res}.")
        grid_shp = Extract_MODIS_LAI.ExtractData(grid_shp, grid_shp_res, plot_month, save_original, check_search)
        logger.info("MODIS LAI data successfully read into grids.")
        return grid_shp