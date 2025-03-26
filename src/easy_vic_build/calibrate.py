# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
calibrate - A Python module for calibrating the VIC model.

This module provides functionality for calibrating hydrological models using the NSGA-II genetic algorithm approach.
It includes the implementation of the `NSGAII_VIC_SO` class for single-objective optimization and a placeholder
for the `NSGAII_VIC_MO` class for multi-objective optimization (which is not yet implemented). The module integrates
with various tools for parameter setup, evaluation metrics, and simulation, as well as visualization capabilities.

Classes:
--------
    - `NSGAII_VIC_SO`: Performs single-objective optimization using the NSGA-II genetic algorithm.
      It inherits from `NSGAII_Base` and handles the calibration process by optimizing model parameters.
    - `NSGAII_VIC_MO`: Placeholder for multi-objective optimization implementation (currently not implemented).

Usage:
------
    1. Initialize an instance of `NSGAII_VIC_SO` with the necessary parameters and configuration.
    2. Run the calibration process using the `run` method to optimize the model parameters.
    3. Retrieve the best results using the `get_best_results` method to analyze the calibration performance.
    4. Visualize the calibration results using the provided plotting functions.

Example:
--------
    basin_index = 397
    model_scale = "6km"
    date_period = ["19980101", "20071231"]

    warmup_date_period = ["19980101", "19991231"]
    calibrate_date_period = ["20000101", "20071231"]
    verify_date_period = ["20080101", "20101231"]
    case_name = f"{basin_index}_{model_scale}"

    evb_dir = Evb_dir(cases_home="/home/xdz/code/VIC_xdz/cases")
    evb_dir.builddir(case_name)
    evb_dir.vic_exe_path = "/home/xdz/code/VIC_xdz/vic_image.exe"

    dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2 = readdpc(evb_dir)

    modify_pourpoint_bool = True
    if modify_pourpoint_bool:
        pourpoint_lon = -91.8225
        pourpoint_lat = 38.3625

        modifyDomain_for_pourpoint(evb_dir, pourpoint_lon, pourpoint_lat)  # mask->1
        buildPourPointFile(evb_dir, None, names=["pourpoint"], lons=[pourpoint_lon], lats=[pourpoint_lat])

    algParams = {"popSize": 20, "maxGen": 200, "cxProb": 0.7, "mutateProb": 0.2}
    nsgaII_VIC_SO = NSGAII_VIC_SO(evb_dir, dpc_VIC_level0, dpc_VIC_level1, date_period, warmup_date_period, calibrate_date_period, verify_date_period,
                                    algParams=algParams, save_path=evb_dir.calibrate_cp_path, reverse_lat=True, parallel=False)

    calibrate_bool = False
    if calibrate_bool:
        nsgaII_VIC_SO.run()

    get_best_results_bool = True
    if get_best_results_bool:
        cali_result, verify_result = nsgaII_VIC_SO.get_best_results()

Dependencies:
-------------
    - `os`: For handling file operations and directory structures.
    - `deap`: A library for evolutionary algorithms, used for genetic operations like crossover, mutation, and selection.
    - `pandas`: For data manipulation and analysis.
    - `netCDF4`: For working with netCDF files to handle model output data.
    - `matplotlib`: For visualizing the calibration results.
    - `.tools`: Various utility and function modules for parameter setup, evaluation, and grid search.
    - `.bulid_Param`, `.build_RVIC_Param`, `.build_GlobalParam`: Modules for constructing configuration files and setting up model parameters.
    - `.tools.decoractors`: For applying the clock decorator to measure function execution time.

"""

import os
from copy import deepcopy
from datetime import datetime

import matplotlib
import pandas as pd
from deap import base, creator, tools
from netCDF4 import Dataset, num2date

from .build_GlobalParam import buildGlobalParam
from .build_RVIC_Param import (buildConvCFGFile, buildParamCFGFile,
                               buildUHBOXFile)
from .bulid_Param import (buildParam_level0, buildParam_level0_by_g,
                          buildParam_level1, scaling_level0_to_level1)
from .tools.calibrate_func.algorithm_NSGAII import NSGAII_Base
from .tools.calibrate_func.evaluate_metrics import EvaluationMetric
from .tools.calibrate_func.sampling import *
from .tools.decoractors import clock_decorator
from .tools.geo_func.search_grids import search_grids_nearest
from .tools.params_func.params_set import *
from .tools.utilities import *

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from . import logger

# plt.show(block=True)

try:
    from rvic.parameters import parameters as rvic_parameters

    HAS_RVIC = True
except:
    HAS_RVIC = False


class NSGAII_VIC_SO(NSGAII_Base):

    def __init__(
        self,
        evb_dir,
        dpc_VIC_level0,
        dpc_VIC_level1,
        date_period,
        warmup_date_period,
        calibrate_date_period,
        verify_date_period,
        rvic_OUTPUT_INTERVAL=86400,
        rvic_BASIN_FLOWDAYS=50,
        rvic_SUBSET_DAYS=10,
        rvic_uhbox_dt=3600,
        algParams={"popSize": 40, "maxGen": 250, "cxProb": 0.7, "mutateProb": 0.2},
        save_path="checkpoint.pkl",
        reverse_lat=True,
        parallel=False,
    ):
        """
        Initializes an instance of the NSGAII_VIC_SO class, which extends the NSGAII_Base class for optimization
        of VIC (Variable Infiltration Capacity) model parameters.

        This constructor sets up various configurations such as simulation period, VIC model parameters, routing
        parameters, and bounds for the optimization process. It also prepares the initial state and other key variables
        for running the VIC model and performing parameter calibration using NSGA-II.

        Parameters
        ----------
        evb_dir : `Evb_dir`
            An instance of the `Evb_dir` class, containing paths for VIC deployment.
    
        dpc_VIC_level0 : `dpc_VIC_level0`
            An instance of the `dpc_VIC_level0` class.

        dpc_VIC_level1 : `dpc_VIC_level1`
            An instance of the `dpc_VIC_level1` class.

        date_period : list of str
            The full date period for the simulation, typically in the format ["YYYYMMDD", "YYYYMMDD"].
            
        warmup_date_period : list of str
            The warm-up period for the model, typically in the format ["YYYYMMDD", "YYYYMMDD"].

        calibrate_date_period : list of str
            The calibration period for the model, typically in the format ["YYYYMMDD", "YYYYMMDD"].

        verify_date_period : list of str
            The verification period for the model, typically in the format ["YYYYMMDD", "YYYYMMDD"].

        rvic_OUTPUT_INTERVAL : int, optional
            The output interval for RVIC in seconds (default is 86400 seconds or 1 day).

        rvic_BASIN_FLOWDAYS : int, optional
            The number of flow days for RVIC (default is 50).

        rvic_SUBSET_DAYS : int, optional
            The number of subset days for RVIC (default is 10).

        rvic_uhbox_dt : int, optional
            The time step for RVIC UHBOX file in seconds (default is 3600 seconds or 1 hour).

        algParams : dict, optional
            The algorithm parameters for NSGA-II optimization, including population size, max generations,
            crossover probability, and mutation probability (default is {"popSize": 40, "maxGen": 250, "cxProb": 0.7, "mutateProb": 0.2}).

        save_path : str, optional
            The path to save the checkpoint file (default is "checkpoint.pkl"), it can be set as evb_dir.calibrate_cp_path.

        reverse_lat : bool
            Boolean flag to indicate whether to reverse latitudes (Northern Hemisphere: large -> small, set as True).
        
        parallel : bool, optional
            Whether to enable parallel processing for the VIC model (default is False).

        Returns
        -------
        None
            This constructor does not return any value. It initializes the instance of the class with the provided
            configuration and prepares the system for optimization.
        """
        logger.info(
            "Initializing NSGAII_VIC_SO instance with provided parameters... ..."
        )

        # *if parallel, uhbox_dt (rvic_OUTPUT_INTERVAL) should be same as VIC output (global param)
        # *if run with RVIC, you should modify Makefile and turn the rout_rvic, compile it
        self.evb_dir = evb_dir
        self.dpc_VIC_level0 = dpc_VIC_level0
        self.dpc_VIC_level1 = dpc_VIC_level1
        self.reverse_lat = reverse_lat
        self.rvic_OUTPUT_INTERVAL = rvic_OUTPUT_INTERVAL  # 3600, 86400
        self.rvic_BASIN_FLOWDAYS = rvic_BASIN_FLOWDAYS
        self.rvic_SUBSET_DAYS = rvic_SUBSET_DAYS
        self.rvic_uhbox_dt = rvic_uhbox_dt
        self.parallel = parallel

        logger.info(
            f"Date periods: {date_period}, {warmup_date_period}, {calibrate_date_period}, {verify_date_period}"
        )

        # period
        self.date_period = date_period
        self.warmup_date_period = warmup_date_period
        self.calibrate_date_period = calibrate_date_period
        self.verify_date_period = verify_date_period

        # clear Param
        logger.info("Clear previous parameters from the VIC model directory")
        clearParam(self.evb_dir)

        # params boundary
        self.g_boundary = g_boundary
        self.uh_params_boundary = uh_params_boundary
        self.routing_params_boundary = routing_params_boundary
        self.depths_indexes = depths_index
        self.total_boundary = [
            item
            for lst in [g_boundary, uh_params_boundary, routing_params_boundary]
            for item in lst
        ]
        self.low = [b[0] for b in self.total_boundary]
        self.up = [b[1] for b in self.total_boundary]

        # params dimension
        self.NDim = len(self.total_boundary)
        logger.info(f"Parameter space dimension: {self.NDim}")

        # params type
        self.g_types = g_types
        self.uh_params_types = uh_params_types
        self.routing_params_types = routing_params_types
        self.total_types = [
            item for lst in [g_types, uh_params_types, uh_params_types] for item in lst
        ]

        # set GlobalParam_dict
        logger.debug("Set global parameters")
        self.set_GlobalParam_dict()
        
        # get obs
        logger.debug("Load observational data")
        self.get_obs()
        
        # get sim
        self.sim_path = ""

        # initial several variable to save
        self.get_sim_searched_grids_index = None

        self.scaling_searched_grids_bool_index = None
        self.stand_grids_lat_level0 = None
        self.stand_grids_lon_level0 = None
        self.rows_index_level0 = None
        self.cols_index_level0 = None

        self.stand_grids_lat_level1 = None
        self.stand_grids_lon_level1 = None
        self.rows_index_level1 = None
        self.cols_index_level1 = None

        super().__init__(algParams, save_path)
        logger.info("Initialized")

    def set_GlobalParam_dict(self):
        """
        Set the global parameters for the VIC simulation and output them into a configuration file.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This method sets the global parameters for simulation (e.g., model steps, start and end dates)
        and defines the output variables. It then passes the parameters to the `buildGlobalParam` function
        to generate the configuration file.
        
        Another set: perhaps it can be run at hourly scale
        >>> GlobalParam_dict = {"Simulation":{"MODEL_STEPS_PER_DAY": "24",
                                        "SNOW_STEPS_PER_DAY": "24",
                                        "RUNOFF_STEPS_PER_DAY": "24",
                                        "STARTYEAR": str(warmup_date_period[0][:4]),
                                        "STARTMONTH": str(int(warmup_date_period[0][4:6])),
                                        "STARTDAY": str(int(warmup_date_period[0][6:])),
                                        "ENDYEAR": str(calibrate_date_period[1][:4]),
                                        "ENDMONTH": str(int(calibrate_date_period[1][4:6])),
                                        "ENDDAY": str(int(calibrate_date_period[1][6:])),
                                        "OUT_TIME_UNITS": "HOURS"},
                            "Output": {"AGGFREQ": "NHOURS   1"},
                            "OUTVAR1": {"OUTVAR": ["OUT_RUNOFF", "OUT_BASEFLOW", "OUT_DISCHARGE"]}
                            }
        """
        logger.debug("Setting global parameters for the simulation... ...")
        GlobalParam_dict = {
            "Simulation": {
                "MODEL_STEPS_PER_DAY": "1",
                "SNOW_STEPS_PER_DAY": "24",
                "RUNOFF_STEPS_PER_DAY": "24",
                "STARTYEAR": str(self.warmup_date_period[0][:4]),
                "STARTMONTH": str(int(self.warmup_date_period[0][4:6])),
                "STARTDAY": str(int(self.warmup_date_period[0][6:])),
                "ENDYEAR": str(self.calibrate_date_period[1][:4]),
                "ENDMONTH": str(int(self.calibrate_date_period[1][4:6])),
                "ENDDAY": str(int(self.calibrate_date_period[1][6:])),
                "OUT_TIME_UNITS": "DAYS",
            },
            "Output": {"AGGFREQ": "NDAYS   1"},
            "OUTVAR1": {"OUTVAR": ["OUT_RUNOFF", "OUT_BASEFLOW", "OUT_DISCHARGE"]},
        }

        # buildGlobalParam
        buildGlobalParam(self.evb_dir, GlobalParam_dict)
        
        logger.debug("Set the global parameters successfully")

    def get_obs(self):
        """
        Get observation.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This method get the observation to attributes.
        """
        logger.debug("Getting observation... ...")
        self.obs = self.dpc_VIC_level1.basin_shp.streamflow.iloc[0]
        date = self.obs.loc[:, "date"]
        factor_unit_feet2meter = 0.0283168
        self.obs.loc[:, "discharge(m3/s)"] = self.obs.loc[:, 4] * factor_unit_feet2meter
        self.obs.index = pd.to_datetime(date)
        
        logger.debug("Get the observation successfully")

    def get_sim(self):
        """
        Get the simulation data from the VIC output files and process it into a DataFrame.

        Parameters
        ----------
        None

        Returns
        -------
        sim_df : `pandas.DataFrame`
            A DataFrame containing the daily average discharge values. The index of the DataFrame is the
            time in datetime format, and the columns are 'time' and 'discharge(m3/s)'.

        Notes
        -----
        This method reads the VIC output data from the NetCDF file, extracts the discharge values,
        and aggregates them to daily values. The discharge values are matched to the outlet location
        based on latitude and longitude. The time is converted from numeric format to datetime.

        If parallel processing is enabled (currently not implemented), the method may perform the
        task in parallel. Otherwise, it reads the data serially from the NetCDF file.
        """
        logger.debug("Getting simulation... ...")

        # path
        nc_files = [
            fn for fn in os.listdir(self.evb_dir.VICResults_dir) if fn.endswith(".nc")
        ]

        if not nc_files:
            logger.warning("No .nc files found in the VICResults directory")
            return None

        self.sim_fn = nc_files[0]
        self.sim_path = os.path.join(self.evb_dir.VICResults_dir, self.sim_fn)
        logger.debug(f"Found simulation file: {self.sim_fn} at {self.sim_path}")

        # Initialize an empty DataFrame for simulation data
        sim_df = pd.DataFrame(columns=["time", "discharge(m3/s)"])

        # outlet lat, lon
        pourpoint_file = pd.read_csv(self.evb_dir.pourpoint_file_path)
        x, y = pourpoint_file.lons[0], pourpoint_file.lats[0]
        logger.debug(f"Outlet coordinates (lat, lon): ({y}, {x})")
        # x, y = dpc_VIC_level1.basin_shp.loc[:, "camels_topo:gauge_lon"].values[0], dpc_VIC_level1.basin_shp.loc[:, "camels_topo:gauge_lat"].values[0]

        if self.parallel:  # TODO
            logger.warning("Parallel processing not yet implemented")
            pass

        else:
            try:
                # read VIC OUTPUT file
                with Dataset(self.sim_path, "r") as dataset:
                    # get time, lat, lon
                    time = dataset.variables["time"]
                    lat = dataset.variables["lat"][:]
                    lon = dataset.variables["lon"][:]

                    # transfer time_num into date
                    time_date = num2date(
                        time[:], units=time.units, calendar=time.calendar
                    )
                    time_date = [
                        datetime(t.year, t.month, t.day, t.hour, t.second)
                        for t in time_date
                    ]

                    logger.debug(
                        f"Converted time to datetime format: {time_date[:5]}..."
                    )
                    # get outlet index
                    if self.get_sim_searched_grids_index is None:
                        searched_grids_index = search_grids_nearest(
                            [y], [x], lat, lon, search_num=1
                        )
                        searched_grid_index = searched_grids_index[0]
                        self.get_sim_searched_grids_index = searched_grid_index

                    # read data
                    out_discharge = dataset.variables["OUT_DISCHARGE"][
                        :,
                        self.get_sim_searched_grids_index[0][0],
                        self.get_sim_searched_grids_index[1][0],
                    ]

                    sim_df.loc[:, "time"] = time_date
                    sim_df.loc[:, "discharge(m3/s)"] = out_discharge
                    sim_df.index = pd.to_datetime(time_date)

                    logger.debug(f"Reading discharge data: {out_discharge[:5]}... ...")

            except Exception as e:
                logger.error(f"Error when reading the simulation file {self.sim_path}: {e}")
                return None

        # Resample the discharge data to daily averages
        sim_df = sim_df.resample("D").mean()
        logger.debug("Aggregated discharge data to daily averages")
        
        logger.debug("Get simulation successfully")

        return sim_df

    def createFitness(self):
        """
        Create a custom fitness class for the evolutionary algorithm.

        This method creates a new class `Fitness` by using the `creator.create`
        function from the DEAP library. The class `Fitness` inherits from the
        `base.Fitness` class, and the weight is set to 1.0, indicating that the
        objective is to maximize the fitness value.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This method is used to define the fitness evaluation criteria for the
        evolutionary algorithm, where higher fitness values are better.
        It uses the DEAP framework's `creator` module to define a custom fitness
        class with a specified weight.
        """
        creator.create("Fitness", base.Fitness, weights=(1.0,))

    def samplingInd(self):
        """
        Sample parameter values for an individual in the population.

        This method performs sampling for different parameter boundaries
        such as `g_boundary`, `uh_params_boundary`, and `routing_params_boundary`.
        It generates both discrete and continuous samples, combining them into
        a single set of parameter values for a new individual in the evolutionary algorithm.

        Parameters
        ----------
        None

        Returns
        -------
        creator.Individual(params_samples): `creator.Individual`
            A new individual with sampled parameter values for the evolutionary algorithm.

        Notes
        -----
        - This method first handles the sampling for `g_boundary`, selecting depth bounds
        and removing them from the list of bounds.
        - Then, it performs discrete sampling for depth parameters (`g`) and continuous
        sampling for other parameters such as `uh_params` and `routing_params`.
        - The final sampled values are combined, and the new individual is returned
        for use in the algorithm.
        """
        logger.debug("Starting parameter sampling process... ...")

        # n_samples
        n_samples = 1

        ## ----------------------- params g bounds -----------------------
        # copy
        params_g_bounds = deepcopy(self.g_boundary)

        # sampling for depths g
        depths_g_bounds = [
            params_g_bounds[di] for di in self.depths_indexes
        ]  # such as [(1, 5), (3, 8), (6, 11)], this is num, start from 1 (1~11)
        for di in sorted(self.depths_indexes, reverse=True):
            params_g_bounds.pop(di)

        ## ----------------------- RVIC params bounds -----------------------
        # uh_params={"tp": 1.4, "mu": 5.0, "m": 3.0}
        uh_params_bounds = self.uh_params_boundary

        # cfg_params={"VELOCITY": 1.5, "DIFFUSION": 800.0}
        routing_params_bounds = self.routing_params_boundary

        ## ----------------------- mixsampling -----------------------
        # discrete sampling
        depth_num_samples = sampling_CONUS_depth_num(
            n_samples, layer_ranges=depths_g_bounds
        )
        num1, num2 = depth_num_samples[0]
        insertions = list(zip(self.depths_indexes, [num1, num2]))
        logger.debug(f"Depth samples: {insertions}")

        # continuous sampling
        params_g_bounds.extend(uh_params_bounds)
        params_g_bounds.extend(routing_params_bounds)

        params_samples = sampling_Sobol(
            n_samples, len(params_g_bounds), params_g_bounds
        )[0]
        params_samples = params_samples.tolist()
        logger.debug(f"Sobol samples: {params_samples}")

        # combine samples
        logger.debug("Inserting discrete samples into continuous samples")
        for index, value in sorted(insertions, key=lambda x: x[0]):
            params_samples.insert(index, value)

        logger.debug(f"Final combined parameter samples: {params_samples}")

        return creator.Individual(params_samples)

    @clock_decorator(print_arg_ret=True)
    def run_vic(self):
        """
        Run the VIC model simulation.

        This method constructs and executes the system command to run the VIC model
        with the specified configuration. The command is executed either in parallel
        or sequentially, depending on the `parallel` attribute of the class. If running
        in parallel, MPI is used for distributed execution.

        Parameters
        ----------
        None

        Returns
        -------
        out : int
            The return code from the `os.system` command execution. A value of 0 typically
            indicates that the command was executed successfully, while any other value
            indicates an error during execution.

        Notes
        -----
        - If `parallel` is set to `True`, the VIC model is executed using MPI
        (`mpiexec`) with the number of processes specified by `self.parallel`.
        - If `parallel` is `False`, the VIC model is executed sequentially.
        - The method uses the `clock_decorator` to measure and log the execution time.
        """
        if self.parallel:
            command_run_vic = " ".join(
                [
                    f"mpiexec -np {self.parallel}",
                    self.evb_dir.vic_exe_path,
                    "-g",
                    self.evb_dir.globalParam_path,
                ]
            )
        else:
            command_run_vic = " ".join(
                [self.evb_dir.vic_exe_path, "-g", self.evb_dir.globalParam_path]
            )

        logger.info("running VIC... ...")
        logger.debug(f"VIC execution command: {command_run_vic}")
        out = os.system(command_run_vic)

        if out == 0:
            logger.debug("VIC model simulation successfully.")
        else:
            logger.error(f"VIC model simulation failed with exit code {out}, please check the VIC logs")

        return out

    @clock_decorator(print_arg_ret=True)
    def run_rvic(self, conv_cfg_file_dict):
        """
        Run the RVIC convolution process.

        This method executes the RVIC convolution using the provided configuration dictionary.
        The convolution function processes the routing of runoff and baseflow using the
        predefined routing configurations.

        Parameters
        ----------
        conv_cfg_file_dict : dict
            A dictionary containing configuration parameters for the RVIC convolution process.

        Returns
        -------
        None

        Notes
        -----
        - This method calls the `convolution` function to execute the RVIC convolution.
        - If multiple RVIC outputs are generated, they may need to be combined (currently marked as TODO).
        - The method uses the `clock_decorator` to measure and log the execution time.
        """
        logger.info("running RVIC convolution... ...")
        logger.debug(f"RVIC configuration: {conv_cfg_file_dict}")

        try:
            convolution(conv_cfg_file_dict)
            logger.info("RVIC convolution process successfully")

        except Exception as e:
            logger.error(f"RVIC convolution process failed: {e}", exc_info=True)

        # TODO: Combine RVIC output if multiple files are generated
        logger.warning("TODO: Combining multiple RVIC outputs is not yet implemented.")
        pass

    def adjust_vic_params_level0(self, params_g):
        """
        Adjust VIC parameters at level 0.

        This method updates or creates VIC parameter datasets at level 0 using the given parameters `params_g`.
        If the parameter dataset already exists, it modifies the parameters based on `params_g`. Otherwise, it
        initializes a new parameter dataset.

        Parameters
        ----------
        params_g : list
            A list of VIC model parameters used for updating or building the level 0 dataset.

        Returns
        -------
        params_dataset_level0 : `netCDF.Dataset`
            The parameter dataset for level 0.
            
        Notes
        -----
        - If `params_dataset_level0_path` exists, it updates the dataset with the given parameters.
        - If the dataset does not exist, it builds a new one using `buildParam_level0`.
        - The adjusted dataset includes the standard grids' latitude, longitude, row indices, and column indices.
        """
        logger.info("Adjusting params_dataset_level0... ...")
        logger.debug(f"Received parameters for adjustment: {params_g}")

        if os.path.exists(self.evb_dir.params_dataset_level0_path):
            logger.info(
                f"Existing params_dataset_level0 found at {self.evb_dir.params_dataset_level0_path}. Updating parameters... ..."
            )

            # read and adjust by g
            params_dataset_level0 = Dataset(
                self.evb_dir.params_dataset_level0_path, "a", format="NETCDF4"
            )
            (
                params_dataset_level0,
                stand_grids_lat,
                stand_grids_lon,
                rows_index,
                cols_index,
            ) = buildParam_level0_by_g(
                params_dataset_level0,
                params_g,
                self.dpc_VIC_level0,
                self.reverse_lat,
                self.stand_grids_lat_level0,
                self.stand_grids_lon_level0,
                self.rows_index_level0,
                self.cols_index_level0,
            )
            logger.info("Successfully updated existing params_dataset_level0")
        else:
            logger.warning(
                f"params_dataset_level0 not found at {self.evb_dir.params_dataset_level0_path}. Creating a new dataset... ..."
            )
            # build
            (
                params_dataset_level0,
                stand_grids_lat,
                stand_grids_lon,
                rows_index,
                cols_index,
            ) = buildParam_level0(
                self.evb_dir,
                params_g,
                self.dpc_VIC_level0,
                self.reverse_lat,
                self.stand_grids_lat_level0,
                self.stand_grids_lon_level0,
                self.rows_index_level0,
                self.cols_index_level0,
            )

            logger.info("Successfully created a new params_dataset_level0")

        (
            self.stand_grids_lat_level0,
            self.stand_grids_lon_level0,
            self.rows_index_level0,
            self.cols_index_level0,
        ) = (stand_grids_lat, stand_grids_lon, rows_index, cols_index)
        logger.debug("Updated VIC level 0 grid attributes")

        return params_dataset_level0

    def adjust_vic_params_level1(self, params_dataset_level0):
        """
        Adjust VIC parameters at level 1 based on level 0 parameters.

        Parameters
        ----------
        params_dataset_level0 : `netCDF.Dataset`
            The parameter dataset for level 0.

        Returns
        -------
        params_dataset_level1 : `netCDF.Dataset`
            The parameter dataset for level 1.
        """
        logger.info("Starting to adjust params_dataset_level1... ...")

        if os.path.exists(self.evb_dir.params_dataset_level1_path):
            # read
            logger.info("params_dataset_level1 file exists. Reading existing dataset... ...")
            params_dataset_level1 = Dataset(
                self.evb_dir.params_dataset_level1_path, "a", format="NETCDF4"
            )
        else:
            # build
            logger.info("params_dataset_level1 file not found. Building new dataset... ...")
            domain_dataset = readDomain(self.evb_dir)
            (
                params_dataset_level1,
                stand_grids_lat,
                stand_grids_lon,
                rows_index,
                cols_index,
            ) = buildParam_level1(
                self.evb_dir,
                self.dpc_VIC_level1,
                self.reverse_lat,
                domain_dataset,
                self.stand_grids_lat_level1,
                self.stand_grids_lon_level1,
                self.rows_index_level1,
                self.cols_index_level1,
            )
            domain_dataset.close()
            self.stand_grids_lat_level1 = stand_grids_lat
            self.stand_grids_lon_level1 = stand_grids_lon
            self.rows_index_level1 = rows_index
            self.cols_index_level1 = cols_index

        # scaling
        params_dataset_level1, searched_grids_bool_index = scaling_level0_to_level1(
            params_dataset_level0,
            params_dataset_level1,
            self.scaling_searched_grids_bool_index,
        )
        self.scaling_searched_grids_bool_index = searched_grids_bool_index

        logger.info("Adjust params_dataset_level1 successfully")

        return params_dataset_level1

    def cal_constraint_destroy(self, params_dataset_level0):
        """
        Calculate constraint violations in level 0 VIC parameters.

        Parameters
        ----------
        params_dataset_level0 : `netCDF.Dataset`
            The parameter dataset for level 0.

        Returns
        -------
        bool
            True if any constraint is violated, otherwise False.
        """
        # wp < fc
        # Wpwp_FRACT < Wcr_FRACT
        # depth_layer0 < depth_layer1
        # no nan in infilt
        # TODO check variables
        logger.info(
            "Starting to calculate constraint violations for params_dataset_level0... ..."
        )

        # Check constraints
        logger.debug("Checking wp < fc constraint... ...")
        constraint_wp_fc_destroy = np.max(
            np.array(
                params_dataset_level0.variables["wp"][:, :, :]
                > params_dataset_level0.variables["fc"][:, :, :]
            )
        )

        logger.debug("Checking Wpwp_FRACT < Wcr_FRACT constraint... ...")
        constraint_Wpwp_Wcr_FRACT_destroy = np.max(
            np.array(
                params_dataset_level0.variables["Wpwp_FRACT"][:, :, :]
                > params_dataset_level0.variables["Wcr_FRACT"][:, :, :]
            )
        )

        logger.debug("Checking depth_layer0 < depth_layer1 constraint... ...")
        constraint_depth_destroy = np.max(
            np.array(
                params_dataset_level0.variables["depth"][0, :, :]
                > params_dataset_level0.variables["depth"][1, :, :]
            )
        )
        # constraint_infilt_nan_destroy = np.sum(np.isnan(np.array(params_dataset_level0.variables["infilt"][:, :]))) > 0

        constraint_destroy = any(
            [
                constraint_wp_fc_destroy,
                constraint_Wpwp_Wcr_FRACT_destroy,
                constraint_depth_destroy,
            ]
        )
        if constraint_destroy:
            logger.warning("Constraint violation detected in params_dataset_level0")
        else:
            logger.info("No constraint violations detected")

        return constraint_destroy

    def adjust_rvic_params(self, uh_params, routing_params):
        """
        Adjust RVIC routing parameters.

        Parameters
        ----------
        uh_params : list
            List of unit hydrograph parameters [tp, mu, m].
            
        routing_params : list
            List of routing parameters [VELOCITY, DIFFUSION].

        Notes
        -----
        domain, FlowDirectionFile, PourPointFile should be already created
        """
        logger.info("Starting to adjust RVIC parameters... ...")

        # UHBOXFile adjustment
        logger.debug("Building UHBOXFile with provided unit hydrograph parameters... ...")
        uh_params_input = {
            "uh_dt": self.rvic_uhbox_dt,
            "tp": uh_params[0],
            "mu": uh_params[1],
            "m": uh_params[2],
            "max_day_range": (0, 10),
            "max_day_converged_threshold": 0.001,
        }
        uhbox_max_day = buildUHBOXFile(self.evb_dir, **uh_params_input, plot_bool=True)

        # ParamCFGFile adjust
        logger.debug("Building ParamCFGFile with routing parameters... ...")
        rvic_param_cfg_params = {
            "VELOCITY": routing_params[0],
            "DIFFUSION": routing_params[1],
            "OUTPUT_INTERVAL": self.rvic_OUTPUT_INTERVAL,
            "SUBSET_DAYS": self.rvic_SUBSET_DAYS,
            "CELL_FLOWDAYS": uhbox_max_day,
            "BASIN_FLOWDAYS": self.rvic_BASIN_FLOWDAYS,
        }
        buildParamCFGFile(self.evb_dir, **rvic_param_cfg_params)

        # Cleanup and directory setup
        logger.debug("Removing old files and creating necessary directories... ...")
        remove_and_mkdir(os.path.join(self.evb_dir.RVICParam_dir, "params"))
        remove_and_mkdir(os.path.join(self.evb_dir.RVICParam_dir, "plots"))
        remove_and_mkdir(os.path.join(self.evb_dir.RVICParam_dir, "logs"))
        inputs_fpath = [
            os.path.join(self.evb_dir.RVICParam_dir, inputs_f)
            for inputs_f in os.listdir(self.evb_dir.RVICParam_dir)
            if inputs_f.startswith("inputs") and inputs_f.endswith("tar")
        ]

        for fp in inputs_fpath:
            logger.debug(f"Removing old RVIC input file in: {fp}... ...")
            os.remove(fp)

        # build rvic_params
        logger.debug("Reading RVIC parameter configuration... ...")
        param_cfg_file_dict = read_cfg_to_dict(self.evb_dir.rvic_param_cfg_file_path)

        if HAS_RVIC:
            logger.info("Running rvic_parameters... ...")
            rvic_parameters(param_cfg_file_dict, numofproc=1)
        else:
            logger.error("RVIC module not available for calibration")
            raise ImportError("No rvic for calibrate")

        # modify rout_param_path in GlobalParam
        logger.debug("Updating GlobalParam with new routing parameters... ...")
        globalParam = GlobalParamParser()
        globalParam.load(self.evb_dir.globalParam_path)
        self.rout_param_path = os.path.join(
            self.evb_dir.rout_param_dir, os.listdir(self.evb_dir.rout_param_dir)[0]
        )
        globalParam.set("Routing", "ROUT_PARAM", self.rout_param_path)

        # Write updated GlobalParam
        logger.debug("Writing updated GlobalParam file... ...")
        with open(self.evb_dir.globalParam_path, "w") as f:
            globalParam.write(f)

        logger.info("Adjusting RVIC parameters successfully")

    def adjust_rvic_conv_params(self):
        """
        Adjust RVIC convolution parameters.

        Returns
        -------
        dict
            Dictionary containing the parsed RVIC convolution configuration.
        """
        # TODO DATL_LIQ_FLDS, OUT_RUNOFF, OUT_BASEFLOW might be run individually

        logger.info("Starting to adjust RVIC convolution parameters... ...")

        # build rvic_conv_cfg_params, construct RUN_STARTDATE from date_period
        logger.debug("Formatting RUN_STARTDATE from date_period... ...")
        RUN_STARTDATE = f"{self.date_period[0][:4]}-{self.date_period[0][4:6]}-{self.date_period[0][6:]}-00"

        # Build RVIC convolution configuration
        logger.debug("Building RVIC convolution configuration file... ...")
        rvic_conv_cfg_params = {
            "RUN_STARTDATE": RUN_STARTDATE,
            "DATL_FILE": self.sim_fn,
            "PARAM_FILE_PATH": self.rout_param_path,
        }

        buildConvCFGFile(self.evb_dir, **rvic_conv_cfg_params)

        # Read and return configuration dictionary
        logger.debug("Reading RVIC convolution configuration file... ...")
        conv_cfg_file_dict = read_cfg_to_dict(self.evb_dir.rvic_conv_cfg_file_path)

        logger.info("Adjusting convolution parameter adjustment successfully")

        return conv_cfg_file_dict

    def evaluate(self, ind):
        """
        Evaluate the fitness of an individual based on VIC and RVIC simulations.

        Parameters
        ----------
        ind : list
            List of parameter values including VIC, UH, and routing parameters.

        Returns
        -------
        tuple
            A tuple containing the fitness value.
        """
        logger.info("Starting evaluate individual... ...")

        # Extract parameter groups
        params_g = ind[:-5]
        uh_params = ind[-5:-2]
        routing_params = ind[-2:]

        # Convert parameter types
        params_g = [self.g_types[i](params_g[i]) for i in range(len(params_g))]
        uh_params = [
            self.uh_params_types[i](uh_params[i]) for i in range(len(uh_params))
        ]
        routing_params = [
            self.routing_params_types[i](routing_params[i])
            for i in range(len(routing_params))
        ]

        # =============== adjust vic params based on ind ===============
        # adjust params_dataset_level0 based on params_g
        logger.info("Adjusting params_dataset_level0")
        params_dataset_level0 = self.adjust_vic_params_level0(params_g)

        # Check for constraint violations
        logger.info("Checking parameter constraints")
        constraint_destroy = self.cal_constraint_destroy(params_dataset_level0)
        logger.info(
            f"Constraint violation: {constraint_destroy}, true means invalid params, set fitness = -9999.0"
        )

        if constraint_destroy:
            logger.warning("Invalid parameters detected. Assigning fitness = -9999.0")
            return (-9999.0,)

        # Adjust params_dataset_level1 based on params_dataset_level0
        logger.info("Adjusting params_dataset_level1")
        params_dataset_level1 = self.adjust_vic_params_level1(params_dataset_level0)

        # close
        params_dataset_level0.close()
        params_dataset_level1.close()

        # Adjust RVIC parameters
        logger.info("Adjusting RVIC parameters")
        self.adjust_rvic_params(uh_params, routing_params)

        # Run VIC simulation
        logger.info("Running VIC simulation")
        remove_files(self.evb_dir.VICResults_dir)
        remove_and_mkdir(self.evb_dir.VICLog_dir)
        out_vic = self.run_vic()
        # self.sim_fn = [fn for fn in os.listdir(self.evb_dir.VICResults_dir) if fn.endswith(".nc")][0]
        # self.sim_path = os.path.join(self.evb_dir.VICResults_dir, self.sim_fn)

        # =============== run rvic offline ===============
        if self.parallel:
            # clear RVICConv_dir
            remove_and_mkdir(os.path.join(self.evb_dir.RVICConv_dir))

            # build cfg file
            conv_cfg_file_dict = self.adjust_rvic_conv_params()

            # run
            out_rvic = self.run_rvic(conv_cfg_file_dict)

        # Evaluate performance
        logger.info("Evaluating model performance")
        sim = self.get_sim()

        sim_cali = sim.loc[
            self.calibrate_date_period[0] : self.calibrate_date_period[1],
            "discharge(m3/s)",
        ]
        obs_cali = self.obs.loc[
            self.calibrate_date_period[0] : self.calibrate_date_period[1],
            "discharge(m3/s)",
        ]

        evaluation_metric = EvaluationMetric(sim_cali, obs_cali)
        fitness = evaluation_metric.KGE()
        # fitness = evaluation_metric.KGE_m()

        # plot discharge
        logger.info("Generating discharge plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sim_cali, "r-", label=f"sim({round(fitness, 2)})", linewidth=0.5)
        ax.plot(obs_cali, "k-", label="obs", linewidth=1)
        ax.set_xlabel("date")
        ax.set_ylabel("discharge m3/s")
        ax.legend()
        # plt.show(block=True)
        fig.savefig(
            os.path.join(self.evb_dir.VICResults_fig_dir, "evaluate_discharge.tiff")
        )

        # Ensure fitness is valid
        if np.isnan(fitness):
            logger.warning(
                "Fitness calculation resulted in NaN. Assigning fitness = -9999.0"
            )
            fitness = -9999.0

        logger.info(f"Evaluation completed. Fitness: {fitness}")

        return (fitness,)

    def simulate(self, ind, GlobalParam_dict):
        """
        Simulate the VIC model with given parameters.

        Parameters
        ----------
        ind : list
            List of parameter values including VIC, UH, and routing parameters.
        
        GlobalParam_dict : dict
            Dictionary containing global parameter settings for VIC.

        Returns
        -------
        sim : DataFrame
            Simulation results.
        """
        logger.info("Starting VIC simulation... ...")

        # buildGlobalParam
        buildGlobalParam(self.evb_dir, GlobalParam_dict)

        # =============== get ind ===============
        params_g = ind[:-5]
        uh_params = ind[-5:-2]
        routing_params = ind[-2:]

        # type params
        params_g = [self.g_types[i](params_g[i]) for i in range(len(params_g))]
        uh_params = [
            self.uh_params_types[i](uh_params[i]) for i in range(len(uh_params))
        ]
        routing_params = [
            self.routing_params_types[i](routing_params[i])
            for i in range(len(routing_params))
        ]

        # =============== adjust vic params based on ind ===============
        # adjust params_dataset_level0 based on params_g
        logger.info("Adjusting params_dataset_level0... ...")
        params_dataset_level0 = self.adjust_vic_params_level0(params_g)

        # adjust params_dataset_level1 based on params_dataset_level0
        logger.info("Adjusting params_dataset_level1... ...")
        params_dataset_level1 = self.adjust_vic_params_level1(params_dataset_level0)

        # close
        params_dataset_level0.close()
        params_dataset_level1.close()

        # =============== adjust rvic params based on ind ===============
        logger.info("Adjusting RVIC parameters... ...")
        self.adjust_rvic_params(uh_params, routing_params)

        # =============== run vic ===============
        logger.info("Running VIC simulation... ...")
        remove_files(self.evb_dir.VICResults_dir)
        remove_and_mkdir(self.evb_dir.VICLog_dir)
        out_vic = self.run_vic()

        # get simulation
        logger.info("Retrieving simulation results... ...")
        sim = self.get_sim()

        logger.info("VIC simulation successfully")

        return sim

    def get_best_results(self):
        """
        Retrieve the best simulation results from the last optimization step.

        Returns
        -------
        tuple
            Calibration and verification results as DataFrames.
        """
        logger.info(
            "Starting to retrieve best results from optimization history... ..."
        )
        # get front
        front = self.history[-1][1][0][0]

        # get fitness
        logger.info(f"Current best fitness: {front.fitness.values}")

        # GlobalParam_dict
        GlobalParam_dict = {
            "Simulation": {
                "MODEL_STEPS_PER_DAY": "1",
                "SNOW_STEPS_PER_DAY": "24",
                "RUNOFF_STEPS_PER_DAY": "24",
                "STARTYEAR": str(self.warmup_date_period[0][:4]),
                "STARTMONTH": str(int(self.warmup_date_period[0][4:6])),
                "STARTDAY": str(int(self.warmup_date_period[0][6:])),
                "ENDYEAR": str(self.verify_date_period[1][:4]),
                "ENDMONTH": str(int(self.verify_date_period[1][4:6])),
                "ENDDAY": str(int(self.verify_date_period[1][6:])),
                "OUT_TIME_UNITS": "DAYS",
            },
            "Output": {"AGGFREQ": "NDAYS   1"},
            "OUTVAR1": {
                "OUTVAR": [
                    "OUT_RUNOFF",
                    "OUT_BASEFLOW",
                    "OUT_DISCHARGE",
                    "OUT_SOIL_MOIST",
                    "OUT_EVAP",
                ]
            },
        }

        # simulate
        logger.info("Running simulation with best parameters... ...")
        sim = self.simulate(front, GlobalParam_dict)

        # get result
        logger.info("Extracting calibration and verification results... ...")
        sim_cali = sim.loc[
            self.calibrate_date_period[0] : self.calibrate_date_period[1],
            "discharge(m3/s)",
        ]
        obs_cali = self.obs.loc[
            self.calibrate_date_period[0] : self.calibrate_date_period[1],
            "discharge(m3/s)",
        ]

        sim_verify = sim.loc[
            self.verify_date_period[0] : self.verify_date_period[1], "discharge(m3/s)"
        ]
        obs_verify = self.obs.loc[
            self.verify_date_period[0] : self.verify_date_period[1], "discharge(m3/s)"
        ]

        cali_result = pd.concat([sim_cali, obs_cali], axis=1)
        cali_result.columns = ["sim_cali discharge(m3/s)", "obs_cali discharge(m3/s)"]

        verify_result = pd.concat([sim_verify, obs_verify], axis=1)
        verify_result.columns = [
            "sim_verify discharge(m3/s)",
            "obs_verify discharge(m3/s)",
        ]

        cali_result.to_csv(os.path.join(self.evb_dir.VICResults_dir, "cali_result.csv"))
        verify_result.to_csv(
            os.path.join(self.evb_dir.VICResults_dir, "verify_result.csv")
        )

        logger.info(f"Best results extraction successfully, saved to {self.evb_dir.VICResults_dir}, cali_result.csv and verify_result.csv")

        return cali_result, verify_result

    @staticmethod
    def operatorMate(parent1, parent2, low, up):
        """
        Perform Simulated Binary Crossover (SBX) between two parents.

        Parameters
        ----------
        parent1 : Individual
            The first parent individual.
            
        parent2 : Individual
            The second parent individual.
            
        low : array-like
            The lower bounds for the crossover.
            
        up : array-like
            The upper bounds for the crossover.

        Returns
        -------
        tuple
            The two offspring produced by the crossover.
        """
        logger.debug("Performing crossover between two parents... ...")
        return tools.cxSimulatedBinaryBounded(
            parent1, parent2, eta=20.0, low=low, up=up
        )

    @staticmethod
    def operatorMutate(ind, low, up, NDim):
        """
        Perform Polynomial Mutation on an individual.

        Parameters
        ----------
        ind : Individual
            The individual to mutate.
        low : array-like
            The lower bounds for the mutation.
        up : array-like
            The upper bounds for the mutation.
        NDim : int
            The number of dimensions of the individual.

        Returns
        -------
        Individual
            The mutated individual.
        """
        logger.debug("Performing mutation on individual... ...")
        return tools.mutPolynomialBounded(ind, eta=20.0, low=low, up=up, indpb=1 / NDim)

    @staticmethod
    def operatorSelect(population, popSize):
        """
        Perform NSGA-II selection on the population.

        Parameters
        ----------
        population : list of Individual
            The current population.
        popSize : int
            The size of the selected population.

        Returns
        -------
        list of Individual
            The selected individuals.
        """
        logger.debug("Performing selection on the population... ...")
        return tools.selNSGA2(population, popSize)

    def apply_genetic_operators(self, offspring):
        """
        Apply genetic operators (crossover and mutation) to the offspring.

        Parameters
        ----------
        offspring : list of Individual
            The offspring to which the genetic operators will be applied.

        Notes
        -----
        This method first performs crossover with a probability defined by
        `self.toolbox.cxProb`. Afterward, it applies mutation with a probability
        defined by `self.toolbox.mutateProb`.
        """
        logger.info("Applying genetic operators to offspring... ...")

        # it can be implemented by algorithms.varAnd
        # crossover
        logger.debug("Starting crossover operation... ...")
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= self.toolbox.cxProb:
                logger.debug(f"Crossover between {child1} and {child2}")
                self.toolbox.mate(child1, child2, self.low, self.up)
                del child1.fitness.values
                del child2.fitness.values

        # mutate
        logger.debug("Starting mutation operation... ...")
        for mutant in offspring:
            if random.random() <= self.toolbox.mutateProb:
                logger.debug(f"Mutation applied to {mutant}")
                self.toolbox.mutate(mutant, self.low, self.up, self.NDim)
                del mutant.fitness.values

        logger.info("Applying genetic operators to offspring successfully")


class NSGAII_VIC_MO(NSGAII_VIC_SO):

    def createFitness(self):
        creator.create("Fitness", base.Fitness, weights=(-1.0,))

    def samplingInd(self):
        return super().samplingInd()

    def evaluate(self, ind):
        return super().evaluate(ind)

    def evaluatePop(self, population):
        return super().evaluatePop(population)

    def operatorMate(self):
        return super().operatorMate()

    def operatorMutate(self):
        return super().operatorMutate()

    def operatorSelect(self):
        return super().operatorSelect()
