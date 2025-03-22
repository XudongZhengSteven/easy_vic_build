# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: data_processing

This module provides various classes for processing basin and grid data in the context of hydrological
modeling using different levels of data processing. The classes extend the base functionality of data
processing and provide specific methods for reading and processing grid and basin data at different levels
(Level 0, Level 1, and Level 2) of sophistication.

Classes:
--------
    - dataProcess_VIC_level0: A base class for processing VIC model data at level 0. It handles reading
      and processing basin and grid data, including basic plotting.
    - dataProcess_VIC_level1: Extends `dataProcess_VIC_level0` with specific methods for processing
      CAMELS streamflow and attribute data, and reading additional grid datasets like ERA5, NLDAS, and MODIS.
    - dataProcess_VIC_level2: A placeholder class for future extensions of the VIC data processing.
    - dataProcess_CAMELS_review: A class for reviewing CAMELS basin and grid data, primarily for visualization
      and examination.

Dependencies:
-------------
    - matplotlib: For creating plots and visualizations.
    - .dpc_base: Base class for data processing functionality.
    - .readdataIntoGrids_interface: Interface for reading data into grid shapefiles.
    - .readdataIntoBasins_interface: Interface for reading data into basin shapefiles.
    - .basin_grid_class: Contains classes and functions for handling basin and grid data.
    - ..utilities: Utility functions for processing and handling data.
    - ..plot_func.plot_func: Plotting functions for visualizing data.
    - ...logger: For logging important events and debugging information.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com

"""

from typing import Any

import matplotlib.pyplot as plt

from ... import logger
from ..plot_func.plot_func import *
from ..utilities import *
from .basin_grid_class import *
from .dpc_base import dataProcess_base
from .readdataIntoBasins_interface import readDataIntoBasins_API
from .readdataIntoGrids_interface import readDataIntoGrids_API


class dataProcess_VIC_level0(dataProcess_base):
    """
    Base class for VIC model data processing at level 0. It provides the framework for processing
    and visualizing basin and grid data, as well as creating boundary shapefiles.
    """

    def __init__(self, basin_shp, grid_shp, grid_res, date_period, **kwargs):
        """
        Initializes the data processing class at level 0.

        Parameters:
        -----------
        basin_shp : GeoDataFrame
            A GeoDataFrame containing basin geometries and attributes.
        grid_shp : GeoDataFrame
            A GeoDataFrame containing grid geometries and attributes.
        grid_res : float
            The resolution of the grid.
        date_period : tuple
            A tuple specifying the start and end date for processing data.
        kwargs : additional keyword arguments
            Other optional parameters for additional customization.
        """
        self.date_period = date_period
        super().__init__(basin_shp, grid_shp, grid_res, **kwargs)
        logger.info("Initialized dataProcess_VIC_level0 with basin and grid data")

    def __call__(
        self,
        *args: Any,
        readBasindata=False,
        readGriddata=True,
        readBasinAttribute=False,
        **kwargs: Any,
    ):
        """
        Executes the full data processing pipeline, including reading basin and grid data,
        aggregating grid data to basins, and visualizing the results.

        Parameters:
        -----------
        readBasindata : bool, optional
            Flag to indicate whether to read basin data (default is False).
        readGriddata : bool, optional
            Flag to indicate whether to read grid data (default is True).
        readBasinAttribute : bool, optional
            Flag to indicate whether to read basin attributes (default is False).
        """
        logger.info("Starting data processing pipeline")
        self.read_basin_grid()

        if readBasindata:
            self.readDataIntoBasins()

        if readGriddata:
            self.readDataIntoGrids()

        if readBasinAttribute:
            self.readBasinAttribute()

        logger.info("Data processing completed")

    def read_basin_grid(self):
        """
        Reads and processes the basin grid data by creating boundary shapefiles.
        """
        logger.debug("Reading basin grid data")
        self.createBoundaryShp()

    def createBoundaryShp(self):
        """
        Creates boundary shapefiles for grids and basins. This includes generating shapefiles
        for the boundary points and grid edges.
        """
        logger.debug("Creating boundary shapefiles")
        (
            self.boundary_point_center_shp,
            self.boundary_point_center_x_y,
            self.boundary_grids_edge_shp,
            self.boundary_grids_edge_x_y,
        ) = self.grid_shp.createBoundaryShp()

    def readDataIntoGrids(self):
        """
        Reads and processes various grid data layers into the grid shapefile.
        This includes SRTM DEM, soil data, and other layers as per the VIC model's requirements.
        """
        logger.debug("Reading grid data into grids")
        self.grid_shp = readDataIntoGrids_API.readSrtmDEMIntoGrids(
            self.grid_shp,
            grid_shp_res=self._grid_res,
            plot=False,
            save_original=False,
            check_search=False,
        )
        self.grid_shp = readDataIntoGrids_API.readCONUSSoilIntoGrids(
            self.grid_shp,
            grid_shp_res=self._grid_res,
            plot_layer=False,
            save_original=False,
            check_search=False,
        )

    def plot(
        self,
        fig=None,
        ax=None,
        grid_shp_kwargs=dict(),
        grid_shp_point_kwargs=dict(),
        basin_shp_kwargs=dict(),
    ):
        """
        Plots the grid and basin data on a map.

        Parameters:
        -----------
        fig : matplotlib.figure.Figure, optional
            The figure to plot on (default is None).
        ax : matplotlib.axes.Axes, optional
            The axes to plot on (default is None).
        grid_shp_kwargs : dict, optional
            Additional keyword arguments for plotting grid shapes (default is an empty dictionary).
        grid_shp_point_kwargs : dict, optional
            Additional keyword arguments for plotting grid points (default is an empty dictionary).
        basin_shp_kwargs : dict, optional
            Additional keyword arguments for plotting basin shapes (default is an empty dictionary).

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes.Axes
            The axes object containing the plot.
        """
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

        logger.debug("Generated plot for grid and basin data")
        return fig, ax

    def plot_grid_column(
        self,
        column,
        fig=None,
        ax=None,
        grid_shp_kwargs=dict(),
        column_kwargs=dict(),
        basin_shp_kwargs=dict(),
    ):
        """
        Plots a specific column of grid data.

        Parameters:
        -----------
        column : str
            The column name in the grid data to be plotted.
        fig : matplotlib.figure.Figure, optional
            The figure to plot on (default is None).
        ax : matplotlib.axes.Axes, optional
            The axes to plot on (default is None).
        grid_shp_kwargs : dict, optional
            Additional keyword arguments for plotting grid shapes (default is an empty dictionary).
        column_kwargs : dict, optional
            Additional keyword arguments for plotting the grid column (default is an empty dictionary).
        basin_shp_kwargs : dict, optional
            Additional keyword arguments for plotting basin shapes (default is an empty dictionary).

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes.Axes
            The axes object containing the plot.
        """
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

        logger.debug(f"Generated plot for grid column: {column}")
        return fig, ax


class dataProcess_VIC_level1(dataProcess_VIC_level0):
    """
    Extends dataProcess_VIC_level0 for VIC model level 1 data processing, which includes handling
    additional layers of grid data and basin-specific attributes such as CAMELS streamflow.
    """

    def readDataIntoBasins(self):
        """
        Reads CAMELS streamflow data into the basin shapefile.
        """
        logger.debug("Reading CAMELS streamflow data into basins")
        self.basin_shp = readDataIntoBasins_API.readCAMELSStreamflowIntoBasins(
            self.basin_shp, read_dates=self.date_period
        )

    def readBasinAttribute(self):
        """
        Reads CAMELS basin attribute data into the basin shapefile.
        """
        logger.debug("Reading CAMELS attributes into basins")
        self.basin_shp = readDataIntoBasins_API.readCAMELSAttributeIntoBasins(
            self.basin_shp, k_list=None
        )

    def readDataIntoGrids(self):
        """
        Reads various grid data layers for VIC level 1 model processing.
        """
        logger.debug("Reading VIC level 1 grid data into grids")
        self.grid_shp = readDataIntoGrids_API.readERA5_SoilTemperatureIntoGrids(
            self.grid_shp,
            grid_shp_res=self._grid_res,
            plot_layer=False,
            check_search=False,
        )
        self.grid_shp = readDataIntoGrids_API.readNLDAS_annual_PIntoGrids(
            self.grid_shp, grid_shp_res=self._grid_res, plot=False, check_search=False
        )
        self.grid_shp = readDataIntoGrids_API.readUMDLandCoverIntoGrids(
            self.grid_shp,
            grid_shp_res=self._grid_res,
            plot=False,
            save_original=True,
            check_search=False,
        )
        self.grid_shp = readDataIntoGrids_API.readMODISBSAIntoGrids(
            self.grid_shp,
            grid_shp_res=self._grid_res,
            plot_month=False,
            save_original=True,
            check_search=False,
        )
        self.grid_shp = readDataIntoGrids_API.readMODISNDVIIntoGrids(
            self.grid_shp,
            grid_shp_res=self._grid_res,
            plot_month=False,
            save_original=True,
            check_search=False,
        )
        self.grid_shp = readDataIntoGrids_API.readMODISLAIIntoGrids(
            self.grid_shp,
            grid_shp_res=self._grid_res,
            plot_month=False,
            save_original=True,
            check_search=False,
        )


class dataProcess_VIC_level2(dataProcess_VIC_level0):
    """
    Placeholder class for level 2 VIC data processing.
    """

    def readDataIntoBasins(self):
        """
        Placeholder method for reading data into basins at level 2.
        """
        pass

    def readBasinAttribute(self):
        """
        Placeholder method for reading basin attributes at level 2.
        """
        pass

    def readDataIntoGrids(self):
        """
        Placeholder method for reading grid data into grids at level 2.
        """
        pass


class dataProcess_CAMELS_review(dataProcess_base):
    """
    Class to handle the review of CAMELS basin and grid data.

    This class reads basin and grid shapefiles for the CAMELS dataset and provides methods to visualize
    the data, including boundary information.

    Parameters
    ----------
    HCDN_home : str
        The home directory for HCDN data.
    basin_shp : GeoDataFrame, optional
        The basin shapefile to load. Default is None.
    grid_shp : GeoDataFrame, optional
        The grid shapefile to load. Default is None.
    grid_res : float, optional
        The resolution of the grid. Default is None.
    **kwargs : additional keyword arguments
        Additional parameters passed to the base class.

    Attributes
    ----------
    basin_shp : GeoDataFrame
        The basin shapefile data.
    grid_shp : GeoDataFrame
        The grid shapefile data.
    boundary_point_center_shp : GeoDataFrame
        The shapefile containing boundary points.
    boundary_point_center_x_y : tuple
        The boundary points' coordinates.
    boundary_grids_edge_shp : GeoDataFrame
        The shapefile containing the grid edges.
    boundary_grids_edge_x_y : tuple
        The grid edges' coordinates.
    """

    def __init__(
        self, HCDN_home, basin_shp=None, grid_shp=None, grid_res=None, **kwargs
    ):
        """
        Initializes the data processing class for CAMELS review.

        Parameters
        ----------
        HCDN_home : str
            The home directory for HCDN data.
        basin_shp : GeoDataFrame, optional
            The basin shapefile to load. Default is None.
        grid_shp : GeoDataFrame, optional
            The grid shapefile to load. Default is None.
        grid_res : float, optional
            The resolution of the grid. Default is None.
        **kwargs : additional keyword arguments
            Additional parameters passed to the base class.
        """
        super().__init__(basin_shp, grid_shp, grid_res, **kwargs)
        self._HCDN_home = HCDN_home
        self.read_basin_grid()
        logger.info("Initialized dataProcess_CAMELS_review")

    def read_basin_grid(self):
        """
        Reads the basin and grid shapefiles from the HCDN directory.

        This method loads the basin shapefile and grid shapefile, and creates boundary shapefiles
        for the grid.

        Attributes
        ----------
        basin_shp : GeoDataFrame
            The basin shapefile data loaded from the HCDN directory.
        grid_shp : GeoDataFrame
            The grid shapefile data loaded from the HCDN directory.
        """
        # read basin shp
        self.basin_shp = HCDNBasins(self._HCDN_home)
        # self.basin_shp_original = HCDNBasins(self._HCDN_home)  # backup for HCDN_shp

        # read grids and createBoundaryShp
        self.grid_shp = HCDNGrids(self._HCDN_home)
        self.createBoundaryShp()

    def createBoundaryShp(self):
        """
        Creates boundary shapefiles for the grid.

        This method generates shapefiles for the boundary points and grid edges, storing the
        results in the corresponding attributes.

        Attributes
        ----------
        boundary_point_center_shp : GeoDataFrame
            Shapefile containing boundary points.
        boundary_point_center_x_y : tuple
            Coordinates of the boundary points.
        boundary_grids_edge_shp : GeoDataFrame
            Shapefile containing the grid edges.
        boundary_grids_edge_x_y : tuple
            Coordinates of the grid edges.
        """
        (
            self.boundary_point_center_shp,
            self.boundary_point_center_x_y,
            self.boundary_grids_edge_shp,
            self.boundary_grids_edge_x_y,
        ) = self.grid_shp.createBoundaryShp()

    def __call__(self, plot=True):
        """
        Executes the review process and optionally plots the basin and grid data.

        Parameters
        ----------
        plot : bool, optional
            Whether to plot the data. Default is True.
        """
        logger.info("Starting to process basin_shp")
        logger.info("Basin Shapefile:\n%s", self.basin_shp)

        logger.info("Starting to process grid_shp")
        logger.info("Grid Shapefile:\n%s", self.grid_shp)

        if plot:
            self.plot()

    def plot(self):
        """
        Plots the background of the grid and basin data.

        This method generates a plot showing the basin and grid data with the boundary information.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes.Axes
            The axes object containing the plot.
        """
        fig, ax = plotBackground(self.basin_shp, self.grid_shp, fig=None, ax=None)
        ax = setBoundary(ax, *self.boundary_grids_edge_x_y)

        return fig, ax


if __name__ == "__main__":
    # general set
    root, home = setHomePath(root="E:")

    # review
    dpc_review = dataProcess_CAMELS_review(HCDN_home=home)
    dpc_review()
