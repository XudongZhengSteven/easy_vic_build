# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Build meteorological forcing NetCDF files from gridded time series."""

from ..decoractors import clock_decorator
from .createMeteForcingDataset import createMeteForcingDataset
from ..dpc_func.basin_grid_func import createStand_grids_lat_lon_from_gridshp, gridshp_index_to_grid_array_index, createEmptyArray_from_gridshp, createEmptyArray_and_assignValue_from_gridshp

from copy import deepcopy
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import os
import cftime

class buildMeteForcing_interface:
    """Interface for building yearly VIC meteorological forcing files."""
    
    def __init__(
        self,
        evb_dir,
        logger,
        dpc_VIC_level2,
        date_period_process,
        date_period_forcing,
        date_format="%Y%m%d",
        timestep="D",
        reverse_lat=True,
        stand_grids_lat_level2=None,
        stand_grids_lon_level2=None,
        rows_index_level2=None,
        cols_index_level2=None,
        file_format="NETCDF4",
    ):
        """Initialize the forcing builder.

        Parameters
        ----------
        evb_dir : object
            Directory/config object that provides output paths and filename prefix.
        logger : object
            Logger instance used for progress messages.
        dpc_VIC_level2 : object
            Data-processing container that stores level-2 basin grid data.
        date_period_process : sequence of str
            Start and end date strings for files to be generated.
        date_period_forcing : sequence of str
            Start and end date strings of available forcing time series.
        date_format : str, optional
            Date parsing format used by ``datetime.strptime``.
        timestep : str, optional
            Pandas frequency string used to build the time axis.
        reverse_lat : bool, optional
            Whether latitude is stored in reverse order.
        stand_grids_lat_level2 : array-like, optional
            Standard latitude coordinate array.
        stand_grids_lon_level2 : array-like, optional
            Standard longitude coordinate array.
        rows_index_level2 : array-like, optional
            Row indices mapping each basin grid to standard grid.
        cols_index_level2 : array-like, optional
            Column indices mapping each basin grid to standard grid.
        file_format : str, optional
            NetCDF output format.
        """
        self.evb_dir = evb_dir
        self.logger = logger
        self.dpc_VIC_level2 = dpc_VIC_level2
        self.date_period_process = date_period_process
        self.date_period_forcing = date_period_forcing
        self.date_format = date_format
        self.timestep = timestep
        
        self.reverse_lat = reverse_lat
        
        self.grid_shp_level2 = deepcopy(self.dpc_VIC_level2.get_data_from_cache("merged_grid_shp")[0])
        self.grids_num_level2 = len(self.grid_shp_level2.index)
        
        self.stand_grids_lat_level2 = stand_grids_lat_level2
        self.stand_grids_lon_level2 = stand_grids_lon_level2
        self.rows_index_level2 = rows_index_level2
        self.cols_index_level2 = cols_index_level2
        
        self.file_format = file_format
    
    @staticmethod
    def generate_cftime_dates(d):
        """Convert one pandas datetime value to ``cftime.datetime``.

        Parameters
        ----------
        d : pandas.Timestamp
            Datetime value to be converted.

        Returns
        -------
        cftime.datetime
            Converted datetime in proleptic Gregorian calendar.
        """
        year = d.year
        month = d.month if hasattr(d, 'month') else 1  # Default to January if no month field
        day = d.day if hasattr(d, 'day') else 1        # Default to 1st day if no day field
        hour = d.hour if hasattr(d, 'hour') else 0     # Default to 00:00 if no hour field
        minute = d.minute if hasattr(d, 'minute') else 0  # Default to 00 minutes
        
        # Create cftime object (using 'noleap' calendar as example, modify as needed)
        return cftime.datetime(year, month, day, hour, minute, calendar='proleptic_gregorian')
    
    @clock_decorator(print_arg_ret=False)
    def buildMeteForcing_loop_years(self):
        """Build forcing files year by year for the configured period."""
        # set_coord_map
        self.set_coord_map()
        
        # set_date_period
        self.set_date_period()
        
        # set vars names map
        self.set_vars_names_map()
        
        # loop for years to create meteforcing_dataset
        for i in tqdm(range(len(self.process_years)), colour="g", desc="loop for years to build meteforcing"):
            year = self.process_years[i]
            self.process_datetime_year = self.process_datetime_years[i]
            process_cftime_year = [self.generate_cftime_dates(d) for d in self.process_datetime_year]
            
            dst_path_year = os.path.join(self.evb_dir.MeteForcing_dir, f"{self.evb_dir.forcing_prefix}.{year}.nc")
            self.meteforcing_dataset, *_ = createMeteForcingDataset(
                dst_path_year,
                self.stand_grids_lat_level2,
                self.stand_grids_lon_level2,
                process_cftime_year,
                self.start_time,
                self.file_format,
            )
            
            # set variables
            for var_name in self.vars_names_maps.keys():
                self.set_var_value(self.vars_names_maps[var_name], var_name)
            
            self.meteforcing_dataset.close()
            
    def set_vars_names_map(self):
        """Set mapping from output variable names to source column names."""
        self.vars_names_maps = {
            "tas": "tmp_avg_C",
            "prcp": "pre_mm_per_3h",
            "pres": "prs_kPa",
            "dswrf": "swd_W_per_m2",
            "dlwrf": "lwd_W_per_m2",
            "vp": "vp_kPa",
            "wind": "wind_m_per_s"
        }
        
    def set_date_period(self):
        """Prepare datetime indices for process period and forcing period."""
        self.logger.info(f"set date_period: from {self.date_period_process[0]} to {self.date_period_process[1]}, timestep is {self.timestep}... ...")
        
        self.start_time = self.date_period_process[0]
        self.date_period_process_datetime = [
            datetime.strptime(self.date_period_process[0], self.date_format),
            datetime.strptime(self.date_period_process[1], self.date_format),
        ]
        self.process_years = list(range(self.date_period_process_datetime[0].year, self.date_period_process_datetime[1].year + 1))
        
        full_range = pd.date_range(
            start=self.date_period_process[0],
            end=self.date_period_process[1],
            freq=self.timestep
        )
        
        self.process_datetime_years = [full_range[full_range.year == year] for year in self.process_years]
        
        self.date_period_forcing_datetime = [
            datetime.strptime(self.date_period_forcing[0], self.date_format),
            datetime.strptime(self.date_period_forcing[1], self.date_format),
        ]
        
        self.forcing_datetime = pd.date_range(self.date_period_forcing_datetime[0], self.date_period_forcing_datetime[1], freq=self.timestep)
        
    def set_coord_map(self):
        """Prepare standard coordinates and index mapping for level-2 grids."""
        self.logger.debug("setting coord_map... ...")
        
        if self.stand_grids_lat_level2 is None:
            self.stand_grids_lat_level2, self.stand_grids_lon_level2 = createStand_grids_lat_lon_from_gridshp(
                self.grid_shp_level2, grid_res=None, reverse_lat=self.reverse_lat
            )

        if self.rows_index_level2 is None:
            self.rows_index_level2, self.cols_index_level2 = gridshp_index_to_grid_array_index(
                self.grid_shp_level2, self.stand_grids_lat_level2, self.stand_grids_lon_level2
            )

    def set_var_value(self, grid_shp_column, var_name):
        """Write one variable from source grid series into the output dataset.

        Parameters
        ----------
        grid_shp_column : str
            Source column name in level-2 grid shapefile data.
        var_name : str
            Target variable name in meteorological forcing dataset.
        """
        # date extraction
        start_prcess_datetime_year = self.process_datetime_year[0]
        end_prcess_datetime_year = self.process_datetime_year[-1]
        
        # start_time_index = np.where(self.forcing_datetime == start_prcess_datetime_year)[0][0]
        # end_time_index = np.where(self.forcing_datetime == end_prcess_datetime_year)[0][0]
        
        start_time_index = np.searchsorted(self.forcing_datetime, start_prcess_datetime_year)
        end_time_index = np.searchsorted(self.forcing_datetime, end_prcess_datetime_year)
        
        var_process_time = self.grid_shp_level2.apply(lambda x: x[grid_shp_column][start_time_index: end_time_index + 1], axis=1)
        
        # stack lat_lon together
        time_series_stack = np.stack(var_process_time.values)
        
        # createEmptyArray_from_gridshp: 3D
        ret_array = createEmptyArray_from_gridshp(
            self.stand_grids_lat_level2, 
            self.stand_grids_lon_level2,
            third_dim_len=len(self.process_datetime_year)
        )
        
        # assign
        idx_2d = (self.rows_index_level2, self.cols_index_level2)
        ret_array[idx_2d] = time_series_stack
        
        # set time dim first
        ret_array = np.transpose(ret_array, (2, 0, 1))
        
        # tranfer into array
        self.meteforcing_dataset.variables[var_name][:, :, :] = ret_array
        
