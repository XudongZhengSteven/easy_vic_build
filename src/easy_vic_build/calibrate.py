# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Calibration workflows for VIC/RVIC based on NSGA-II search.

Public classes
--------------
``NSGAII_VIC_SO``
    Single-objective calibration using KGE as fitness.
``NSGAII_VIC_MO``
    Minimal multi-objective subclass placeholder.
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
from .tools.routing_func.create_uh import createGUH
from .tools.calibrate_func.algorithm_NSGAII import NSGAII_Base
from .tools.calibrate_func.evaluate_metrics import EvaluationMetric
from .tools.calibrate_func.sampling import *
from .tools.decoractors import clock_decorator
from .tools.geo_func.search_grids import search_grids_nearest
from .tools.params_func.build_Param_interface import buildParam_level0_interface, buildParam_level1_interface
from .tools.params_func.params_set import *
from .tools.params_func.TransferFunction import TF_VIC
from .build_Param import scaling_level0_to_level1
from .build_RVIC_Param import buildRVICParam_basic, buildRVICParam
from .tools.dpc_func.extractData_func.Extract_CONUS_SOIL import CONUS_soillayerresampler
from .tools.utilities import *

import matplotlib.pyplot as plt

from . import logger

try:
    from rvic.parameters import parameters as rvic_parameters
    from rvic.convolution import convolution

    HAS_RVIC = True
except:
    HAS_RVIC = False


class NSGAII_VIC_SO(NSGAII_Base):
    """Single-objective NSGA-II calibrator for VIC/RVIC workflows."""

    def __init__(
        self,
        evb_dir,
        dpc_VIC_level0,
        dpc_VIC_level1,
        dpc_VIC_level3,
        date_period,
        warmup_date_period,
        calibrate_date_period,
        verify_date_period,
        domain_dataset=None,
        snaped_outlet_lons=None,
        snaped_outlet_lats=None,
        snaped_outlet_names=None,
        buildParam_level0_interface_class=buildParam_level0_interface,
        buildParam_level1_interface_class=buildParam_level1_interface,
        soillayerresampler=CONUS_soillayerresampler,
        TF_VIC_class=TF_VIC,
        nlayer_list=[1, 2, 3],
        rvic_OUTPUT_INTERVAL=86400,
        rvic_BASIN_FLOWDAYS=50,
        rvic_SUBSET_DAYS=10,
        rvic_uhbox_dt=3600,
        algParams={"popSize": 40, "maxGen": 250, "cxProb": 0.7, "mutateProb": 0.2},
        save_path="checkpoint.pkl",
        reverse_lat=True,
        parallel=False,
    ):
        """Initialize calibration state and algorithm settings.

        Parameters
        ----------
        evb_dir : Evb_dir
            Case directory manager and runtime paths.
        dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level3 : object
            DPC instances used for parameter updates and observations.
        date_period, warmup_date_period, calibrate_date_period, verify_date_period : list
            Simulation and evaluation date windows (``YYYYMMDD`` strings).
        domain_dataset : netCDF4.Dataset, optional
            Domain dataset. If ``None``, it is loaded from disk.
        snaped_outlet_lons, snaped_outlet_lats, snaped_outlet_names : list, optional
            Optional outlet definitions for RVIC routing.
        buildParam_level0_interface_class, buildParam_level1_interface_class : type, optional
            Interface classes for parameter generation.
        soillayerresampler : object, optional
            Soil-layer resampler for level-0 parameter build.
        TF_VIC_class : type, optional
            Transfer-function class.
        nlayer_list : list, optional
            Soil-layer indices used by scaling.
        rvic_OUTPUT_INTERVAL, rvic_BASIN_FLOWDAYS, rvic_SUBSET_DAYS, rvic_uhbox_dt : int, optional
            RVIC runtime/config defaults.
        algParams : dict, optional
            NSGA-II control parameters.
        save_path : str, optional
            Checkpoint path used by ``NSGAII_Base``.
        reverse_lat : bool, optional
            Whether latitude axis is arranged north-to-south.
        parallel : bool or int, optional
            VIC parallel execution flag/process count.
        """
        logger.info(
            "Initializing NSGAII_VIC_SO instance with provided parameters... ..."
        )

        # *if parallel, uhbox_dt (rvic_OUTPUT_INTERVAL) should be same as VIC output (global param)
        # *if run with RVIC, you should modify Makefile and turn the rout_rvic, compile it
        self.evb_dir = evb_dir
        self.dpc_VIC_level0 = dpc_VIC_level0
        self.dpc_VIC_level1 = dpc_VIC_level1
        self.dpc_VIC_level3 = dpc_VIC_level3
        self.basin_shp = dpc_VIC_level3.get_data_from_cache("basin_shp")[0]
        self.domain_dataset = domain_dataset if domain_dataset is not None else readDomain(evb_dir)
        self.snaped_outlet_lons = snaped_outlet_lons
        self.snaped_outlet_lats = snaped_outlet_lats
        self.snaped_outlet_names = snaped_outlet_names
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
        
        # buildParam set
        self.buildParam_level0_interface_class = buildParam_level0_interface_class
        self.buildParam_level1_interface_class = buildParam_level1_interface_class
        self.soillayerresampler = soillayerresampler
        self.TF_VIC_class = TF_VIC_class
        self.nlayer_list = nlayer_list if nlayer_list is not None else [1, 2, 3]
        
        # param dict set
        self.paramManager = ParamManager(params)
        
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
        """Build and write the default VIC global-parameter configuration."""
        logger.debug("Setting global parameters for the simulation... ...")
        GlobalParam_dict = {
            "Simulation": {
                "MODEL_STEPS_PER_DAY": "1",
                "SNOW_STEPS_PER_DAY": "24",
                "RUNOFF_STEPS_PER_DAY": "24",
                "STARTYEAR": str(self.warmup_date_period[0][:4]),
                "STARTMONTH": str(int(self.warmup_date_period[0][4:6])),
                "STARTDAY": str(int(self.warmup_date_period[0][6:8])),
                "ENDYEAR": str(self.calibrate_date_period[1][:4]),
                "ENDMONTH": str(int(self.calibrate_date_period[1][4:6])),
                "ENDDAY": str(int(self.calibrate_date_period[1][6:8])),
                "OUT_TIME_UNITS": "DAYS",
            },
            "Output": {"AGGFREQ": "NDAYS   1"},
            "OUTVAR1": {"OUTVAR": ["OUT_RUNOFF", "OUT_BASEFLOW", "OUT_DISCHARGE"]},
        }

        # buildGlobalParam
        buildGlobalParam(self.evb_dir, GlobalParam_dict)
        
        logger.debug("Set the global parameters successfully")

    def get_obs(self):
        """Load observed streamflow and convert discharge to ``m3/s``."""
        logger.debug("Getting observation... ...")
        basin_shp_with_streamflow = self.dpc_VIC_level3.get_data_from_cache("streamflow")[0]
        self.obs = basin_shp_with_streamflow.streamflow
        date = self.obs.loc[:, "date"]
        factor_unit_feet2meter = 0.0283168
        self.obs.loc[:, "discharge(m3/s)"] = self.obs.loc[:, 4] * factor_unit_feet2meter  # TODO check
        self.obs.index = pd.to_datetime(date)
        
        logger.debug("Get the observation successfully")

    def get_sim(self):
        """Read simulated discharge at pour point and resample to daily mean."""
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
        """Register single-objective fitness in DEAP creator."""
        creator.create("Fitness", base.Fitness, weights=(1.0,))

    def samplingInd(self):
        """Sample one individual from parameter bounds via LHS."""
        logger.debug("Starting parameter sampling process... ...")

        # n_samples
        n_samples = 1

        # get bounds
        bounds = self.paramManager.vector_bounds()
        
        # sample
        params_samples = sampling_LHS_2(n_samples, bounds)

        return creator.Individual(params_samples)

    @clock_decorator(print_arg_ret=True)
    def run_vic(self):
        """Execute VIC using the current global parameter file."""
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
        """Run RVIC convolution with a loaded configuration dictionary."""
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

    def adjust_vic_params_level0(self, g_params):
        """Update or build level-0 parameters from a candidate vector."""
        logger.info("Adjusting params_dataset_level0... ...")
        logger.debug(f"Received parameters for adjustment: {g_params}")

        buildParam_level0_interface_instance = self.buildParam_level0_interface_class(
            self.evb_dir,
            logger,
            self.dpc_VIC_level0,
            g_params,
            self.soillayerresampler,
            self.TF_VIC_class,
            self.reverse_lat,
            self.stand_grids_lat_level0,
            self.stand_grids_lon_level0,
            self.rows_index_level0,
            self.cols_index_level0
        )
        
        if os.path.exists(self.evb_dir.params_dataset_level0_path):
            logger.info(f"Existing params_dataset_level0 found at {self.evb_dir.params_dataset_level0_path}. Updating parameters... ...")

            # read and adjust by g
            params_dataset_level0 = Dataset(self.evb_dir.params_dataset_level0_path, "a", format="NETCDF4")
            buildParam_level0_interface_instance.params_dataset_level0 = params_dataset_level0
            buildParam_level0_interface_instance.buildParam_level0_by_g_tf()
            
            logger.info("Successfully updated existing params_dataset_level0")
        else:
            logger.info(f"params_dataset_level0 not found at {self.evb_dir.params_dataset_level0_path}. Creating a new dataset... ...")
            
            # build
            buildParam_level0_interface_instance.buildParam_level0_basic()
            buildParam_level0_interface_instance.buildParam_level0_by_g_tf()
            params_dataset_level0 = buildParam_level0_interface_instance.params_dataset_level0

            logger.info("Successfully created a new params_dataset_level0")

        # save these attributes to increase speed
        self.stand_grids_lat_level0 = buildParam_level0_interface_instance.stand_grids_lat_level0
        self.stand_grids_lon_level0 = buildParam_level0_interface_instance.stand_grids_lon_level0
        self.rows_index_level0 = buildParam_level0_interface_instance.rows_index_level0
        self.cols_index_level0 = buildParam_level0_interface_instance.cols_index_level0

        return params_dataset_level0

    def adjust_vic_params_level1(self, params_dataset_level0):
        """Update/build level-1 parameters and apply level-0 to level-1 scaling."""
        logger.info("Starting to adjust params_dataset_level1... ...")
        
        buildParam_level1_interface_instance = self.buildParam_level1_interface_class(
            self.evb_dir,
            logger,
            self.dpc_VIC_level1,
            self.TF_VIC_class,
            self.reverse_lat,
            self.domain_dataset,
            self.stand_grids_lat_level1,
            self.stand_grids_lon_level1,
            self.rows_index_level1,
            self.cols_index_level1
        )
        
        if os.path.exists(self.evb_dir.params_dataset_level1_path):
            # read
            logger.info("params_dataset_level1 file exists. Reading existing dataset... ...")
            params_dataset_level1 = Dataset(self.evb_dir.params_dataset_level1_path, "a", format="NETCDF4")
            
        else:
            # build
            logger.info("params_dataset_level1 file not found. Building new dataset... ...")
            buildParam_level1_interface_instance.buildParam_level1_basic()
            buildParam_level1_interface_instance.buildParam_level1_by_tf()
            params_dataset_level1 = buildParam_level1_interface_instance.params_dataset_level1
            
            logger.info("Successfully created a new params_dataset_level1")
            
            # save these attributes to increase speed
            self.stand_grids_lat_level1 = buildParam_level1_interface_instance.stand_grids_lat_level1
            self.stand_grids_lon_level1 = buildParam_level1_interface_instance.stand_grids_lon_level1
            self.rows_index_level1 = buildParam_level1_interface_instance.rows_index_level1
            self.cols_index_level1 = buildParam_level1_interface_instance.cols_index_level1

        # scaling
        params_dataset_level1, searched_grids_bool_index = scaling_level0_to_level1(
            params_dataset_level0,
            params_dataset_level1,
            self.scaling_searched_grids_bool_index,
            self.nlayer_list,
        )
        
        self.scaling_searched_grids_bool_index = searched_grids_bool_index

        logger.info("Adjust params_dataset_level1 successfully")

        return params_dataset_level1

    def cal_constraint_destroy(self, params_dataset_level0):
        """Check hard physical constraints on level-0 parameters."""
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
            logger.warning(f"Constraint violation detected in params_dataset_level0: constraint_destroy({constraint_destroy})")
        else:
            logger.info("No constraint violations detected")

        return constraint_destroy

    def adjust_rvic_params(self, guh_params, rvic_params):
        """Rebuild RVIC routing parameters from candidate UH/routing values."""
        logger.info("Starting to adjust RVIC parameters... ...")
        
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
        buildRVICParam(
            self.evb_dir,
            self.domain_dataset,
            ppf_kwargs={
                "names": self.snaped_outlet_names,
                "lons": self.snaped_outlet_lons,
                "lats": self.snaped_outlet_lats,
            },
            
            uh_params={
                "createUH_func": createGUH,
                "uh_dt": self.rvic_uhbox_dt,
                "tp": guh_params["tp"]["optimal"][0],
                "mu": guh_params["mu"]["optimal"][0],
                "m": guh_params["m"]["optimal"][0],
                "plot_bool": True,
                "max_day": None,
                "max_day_range": (0, 10),
                "max_day_converged_threshold": 0.001
            },
            
            cfg_params={
                "VELOCITY": rvic_params["VELOCITY"]["optimal"][0],
                "DIFFUSION": rvic_params["DIFFUSION"]["optimal"][0],
                "OUTPUT_INTERVAL": self.rvic_OUTPUT_INTERVAL,
                "SUBSET_DAYS": self.rvic_SUBSET_DAYS,
                "CELL_FLOWDAYS": None,
                "BASIN_FLOWDAYS": self.rvic_BASIN_FLOWDAYS,
            }
        )

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
        """Build RVIC convolution config and return it as a dictionary."""
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
        """Evaluate one individual and return fitness tuple."""
        logger.info("Starting evaluate individual... ...")

        # format dtype
        ind_format = [t(v) for v, t in zip(ind, self.paramManager.vector_types())]
        
        # Extract parameter groups
        param_dict = self.paramManager.to_dict(vector=ind_format, field="optimal")
        
        g_params = param_dict["g_params"]
        guh_params = param_dict["guh_params"]
        rvic_params = param_dict["rvic_params"]
        
        # =============== adjust vic params based on ind ===============
        # adjust params_dataset_level0 based on g_params
        logger.info("Adjusting params_dataset_level0")
        params_dataset_level0 = self.adjust_vic_params_level0(g_params)

        # Check for constraint violations
        logger.info("Checking parameter constraints")
        constraint_destroy = self.cal_constraint_destroy(params_dataset_level0)
        logger.info(f"Constraint violation: {constraint_destroy}, true means invalid params, set fitness = -9999.0")

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
        self.adjust_rvic_params(guh_params, rvic_params)

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
        """Run full VIC+RVIC simulation for one individual."""
        logger.info("Starting VIC-RVIC simulation... ...")

        # buildGlobalParam
        buildGlobalParam(self.evb_dir, GlobalParam_dict)

        # =============== get ind ===============
        # format dtype
        ind_format = [t(v) for v, t in zip(ind, self.paramManager.vector_types())]
        
        # Extract parameter groups
        param_dict = self.paramManager.to_dict(vector=ind_format, field="optimal")
        
        g_params = param_dict["g_params"]
        guh_params = param_dict["guh_params"]
        rvic_params = param_dict["rvic_params"]

        # =============== adjust vic params based on ind ===============
        # adjust params_dataset_level0 based on g_params
        logger.info("Adjusting params_dataset_level0... ...")
        params_dataset_level0 = self.adjust_vic_params_level0(g_params)

        # adjust params_dataset_level1 based on params_dataset_level0
        logger.info("Adjusting params_dataset_level1... ...")
        params_dataset_level1 = self.adjust_vic_params_level1(params_dataset_level0)

        # close
        params_dataset_level0.close()
        params_dataset_level1.close()

        # =============== adjust rvic params based on ind ===============
        logger.info("Adjusting RVIC parameters... ...")
        self.adjust_rvic_params(guh_params, rvic_params)

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

    def simulate_vic(self, ind, GlobalParam_dict):
        """Run VIC-only simulation for one individual."""
        logger.info("Starting VIC simulation... ...")
        
        # buildGlobalParam
        buildGlobalParam(self.evb_dir, GlobalParam_dict)

        # =============== get ind ===============
        # format dtype
        ind_format = [t(v) for v, t in zip(ind, self.paramManager.vector_types())]
        
        # Extract parameter groups
        param_dict = self.paramManager.to_dict(vector=ind_format, field="optimal")
        
        g_params = param_dict["g_params"]
        
        # =============== adjust vic params based on ind ===============
        # adjust params_dataset_level0 based on g_params
        logger.info("Adjusting params_dataset_level0... ...")
        params_dataset_level0 = self.adjust_vic_params_level0(g_params)

        # adjust params_dataset_level1 based on params_dataset_level0
        logger.info("Adjusting params_dataset_level1... ...")
        params_dataset_level1 = self.adjust_vic_params_level1(params_dataset_level0)

        # close
        params_dataset_level0.close()
        params_dataset_level1.close()
        
        # =============== run vic ===============
        logger.info("Running VIC simulation... ...")
        remove_files(self.evb_dir.VICResults_dir)
        remove_and_mkdir(self.evb_dir.VICLog_dir)
        out_vic = self.run_vic()
        
        logger.info("VIC simulation successfully")
        
        return out_vic
        

    def simulate_rvic(self, ind, GlobalParam_dict):
        """Prepare RVIC inputs for one individual (execution path is incomplete)."""
        logger.info("Starting RVIC simulation... ...")
        
        # buildGlobalParam
        buildGlobalParam(self.evb_dir, GlobalParam_dict)
        
        # =============== get ind ===============
        # format dtype
        ind_format = [t(v) for v, t in zip(ind, self.paramManager.vector_types())]
        
        # Extract parameter groups
        param_dict = self.paramManager.to_dict(vector=ind_format, field="optimal")
        
        guh_params = param_dict["guh_params"]
        rvic_params = param_dict["rvic_params"]
        
        # =============== adjust rvic params based on ind ===============
        logger.info("Adjusting RVIC parameters... ...")
        self.adjust_rvic_params(guh_params, rvic_params)
        
        # =============== run rvic ===============
        # build cfg file
        conv_cfg_file_dict = self.adjust_rvic_conv_params()
        
        pass
    
    def get_best_results(self):
        """Run best individual and export calibration/verification time series."""
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
                "STARTDAY": str(int(self.warmup_date_period[0][6:8])),
                "ENDYEAR": str(self.verify_date_period[1][:4]),
                "ENDMONTH": str(int(self.verify_date_period[1][4:6])),
                "ENDDAY": str(int(self.verify_date_period[1][6:8])),
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
        logger.debug("Performing crossover between two parents... ...")
        return tools.cxSimulatedBinaryBounded(
            parent1, parent2, eta=20.0, low=low, up=up
        )

    @staticmethod
    def operatorMutate(ind, low, up, NDim):
        logger.debug("Performing mutation on individual... ...")
        return tools.mutPolynomialBounded(ind, eta=20.0, low=low, up=up, indpb=1 / NDim)

    @staticmethod
    def operatorSelect(population, popSize):
        logger.debug("Performing selection on the population... ...")
        return tools.selNSGA2(population, popSize)

    def apply_genetic_operators(self, offspring):
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
    """Minimal multi-objective subclass placeholder."""

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
