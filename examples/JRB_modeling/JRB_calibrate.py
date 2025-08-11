# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from general_info import *
from JRB_build_evb_dir import build_modeling_dir
from JRB_build_dpc import dataProcess_VIC_level0_JRB, dataProcess_VIC_level1_JRB, dataProcess_VIC_level2_CDMet_JRB, dataProcess_VIC_level3_JRB, dataProcess_VIC_level1_GLEAM_JRB

from easy_vic_build.tools.dpc_func.basin_grid_func import createEmptyArray_from_gridshp, createStand_grids_lat_lon_from_gridshp, gridshp_index_to_grid_array_index

from JRB_build_Param import buildParam_level0_interface_JRB, buildParam_level1_interface_JRB

import os
from copy import deepcopy
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from deap import base, creator, tools
from netCDF4 import Dataset, num2date

from easy_vic_build.build_GlobalParam import buildGlobalParam
from easy_vic_build.tools.routing_func.create_uh import createGUH
from easy_vic_build.tools.calibrate_func.algorithm_NSGAII import NSGAII_Base
from easy_vic_build.tools.calibrate_func.evaluate_metrics import EvaluationMetric
from easy_vic_build.tools.calibrate_func.sampling import *
from easy_vic_build.tools.decoractors import clock_decorator
from easy_vic_build.tools.params_func.build_Param_interface import buildParam_level0_interface, buildParam_level1_interface
from easy_vic_build.tools.params_func.params_set import *
from easy_vic_build.tools.params_func.TransferFunction import TF_VIC
from easy_vic_build.build_Param import scaling_level0_to_level1
from easy_vic_build.build_RVIC_Param import buildRVICParam
from JRB_extractData_func.Extract_SoilGrids1km import SoilGrids_soillayerresampler
from easy_vic_build.tools.utilities import *
from easy_vic_build import logger

try:
    from rvic.parameters import parameters as rvic_parameters

    HAS_RVIC = True
except:
    HAS_RVIC = False

g_params["soil_layers_breakpoints"] = {
    "default": [1, 5],
    "boundary": [[1, 2], [2, 5]],
    "type": int,
    "optimal": [None, None]
}

# "soil_layers_breakpoints": {
#     "default": [3, 9],  # soil layer breakpoints, original layers -> modeling layers, note exclusive
#     "boundary": [[1, 3], [3, 9]],
#     "type": int,
#     "optimal": [None, None],
# },

class NSGAII_VIC_MO(NSGAII_Base):

    def __init__(
        self,
        evb_dir,
        dpc_VIC_level0,
        dpc_VIC_level1,
        dpc_VIC_level3,
        dpc_VIC_level1_GLEAM,
        warmup_date_period,
        calibrate_date_period,
        verify_date_period,
        timestep,
        timestep_evaluate,
        GlobalParam_dict,
        domain_dataset=None,
        snaped_outlet_lons=None,
        snaped_outlet_lats=None,
        snaped_outlet_names=None,
        buildParam_level0_interface_class=buildParam_level0_interface_JRB,
        buildParam_level1_interface_class=buildParam_level1_interface_JRB,
        soillayerresampler=SoilGrids_soillayerresampler,
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
        logger.info(
            "Initializing NSGAII_VIC_SO instance with provided parameters... ..."
        )

        # *if parallel, uhbox_dt (rvic_OUTPUT_INTERVAL) should be same as VIC output (global param)
        # *if run with RVIC, you should modify Makefile and turn the rout_rvic, compile it
        self.evb_dir = evb_dir
        self.dpc_VIC_level0 = dpc_VIC_level0
        self.dpc_VIC_level1 = dpc_VIC_level1
        self.dpc_VIC_level3 = dpc_VIC_level3
        self.dpc_VIC_level1_GLEAM = dpc_VIC_level1_GLEAM
        
        self.basin_shp = dpc_VIC_level3.get_data_from_cache("basin_shp")[0]
        self.grid_shp_level0 = dpc_VIC_level0.get_data_from_cache("grid_shp")[0]
        self.grid_shp_level1 = dpc_VIC_level1.get_data_from_cache("grid_shp")[0]
        
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
            f"Date periods: {warmup_date_period}, {calibrate_date_period}, {verify_date_period}"
        )

        # period
        self.warmup_date_period = warmup_date_period
        self.calibrate_date_period = calibrate_date_period
        self.verify_date_period = verify_date_period
        self.set_date(timestep, timestep_evaluate)

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
        self.low = [b[0] for b in self.paramManager.vector_bounds()]
        self.up = [b[1] for b in self.paramManager.vector_bounds()]
        
        # set GlobalParam_dict
        logger.debug("Set global parameters")
        self.set_GlobalParam_dict(GlobalParam_dict)

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
        
        # set coord map
        self.set_coord_map()

        # get obs
        logger.debug("Load observational data")
        self.get_obs()
        
        # get sim
        self.sim_path = ""

        super().__init__(algParams, save_path)
        logger.info("Initialized")
    
    def set_date(self, timestep, timestep_evaluate):
        logger.debug("setting date... ...")
        self.warmup_date = pd.date_range(warmup_date_period[0], warmup_date_period[-1], freq=timestep)
        self.warmup_date_eval = pd.date_range(warmup_date_period[0], warmup_date_period[-1], freq=timestep_evaluate)

        self.calibrate_date = pd.date_range(calibrate_date_period[0], calibrate_date_period[-1], freq=timestep)
        self.calibrate_date_eval = pd.date_range(calibrate_date_period[0], calibrate_date_period[-1], freq=timestep_evaluate)

        self.verify_date = pd.date_range(verify_date_period[0], verify_date_period[-1], freq=timestep)
        self.verify_date_eval = pd.date_range(verify_date_period[0], verify_date_period[-1], freq=timestep_evaluate)


    def set_coord_map(self):
        logger.debug("setting coord_map... ...")
        
        # level0
        if self.stand_grids_lat_level0 is None:
            self.stand_grids_lat_level0, self.stand_grids_lon_level0 = createStand_grids_lat_lon_from_gridshp(
                self.grid_shp_level0, grid_res=None, reverse_lat=self.reverse_lat
            )

        if self.rows_index_level0 is None:
            self.rows_index_level0, self.cols_index_level0 = gridshp_index_to_grid_array_index(
                self.grid_shp_level0, self.stand_grids_lat_level0, self.stand_grids_lon_level0
            )
        
        # level1
        if self.stand_grids_lat_level1 is None:
            self.stand_grids_lat_level1, self.stand_grids_lon_level1 = createStand_grids_lat_lon_from_gridshp(
                self.grid_shp_level1, grid_res=None, reverse_lat=self.reverse_lat
            )

        if self.rows_index_level1 is None:
            self.rows_index_level1, self.cols_index_level1 = gridshp_index_to_grid_array_index(
                self.grid_shp_level1, self.stand_grids_lat_level1, self.stand_grids_lon_level1
            )
        
    
    def set_GlobalParam_dict(self, GlobalParam_dict):
        logger.debug("Setting global parameters for the simulation... ...")

        # buildGlobalParam
        buildGlobalParam(self.evb_dir, GlobalParam_dict)
        
        logger.debug("Set the global parameters successfully")     
        
        
    def get_obs(self, type="calibration"):
        logger.debug("Getting observation... ...")
        
        # set obs
        self.obs = {}

        n_warmup = len(self.warmup_date_eval)
        n_calibration = len(self.calibrate_date_eval)
        n_verification = len(self.verify_date_eval)

        # read streamflow
        basin_shp_with_streamflow = self.dpc_VIC_level3.get_data_from_cache("streamflow")[0]
        streamflow = basin_shp_with_streamflow.loc[0, "stationdata_streamflow"]
        self.obs["streamflow(m3/s)"] = streamflow.iloc[n_warmup:n_warmup+n_calibration]
        
        # read GLEAM: SMtotal
        grid_shp_with_GLEAM_sm = self.dpc_VIC_level1_GLEAM.get_data_from_cache("GLEAM_sm")[0]
        GLEAM_SMtotal = createEmptyArray_from_gridshp(
            self.stand_grids_lat_level1,
            self.stand_grids_lon_level1,
            third_dim_len=grid_shp_with_GLEAM_sm["SMtotal(mm)"][0].shape[0]  # only daily
        )
        
        idx_2d = (self.rows_index_level1, self.cols_index_level1)
        GLEAM_SMtotal[idx_2d] = np.stack(grid_shp_with_GLEAM_sm["SMtotal(mm)"].values)
        self.obs["GLEAM_SMtotal(mm)"] = np.transpose(GLEAM_SMtotal[:, :, n_warmup:n_warmup+n_calibration], (2, 0, 1))

        # read GLEAM: E
        grid_shp_with_GLEAM_E = self.dpc_VIC_level1_GLEAM.get_data_from_cache("GLEAM_E")[0]
        GLEAM_E = createEmptyArray_from_gridshp(
            self.stand_grids_lat_level1,
            self.stand_grids_lon_level1,
            third_dim_len=grid_shp_with_GLEAM_E["E(mm)"][0].shape[0]  # only daily
        )
        
        idx_2d = (self.rows_index_level1, self.cols_index_level1)
        GLEAM_E[idx_2d] = np.stack(grid_shp_with_GLEAM_E["E(mm)"].values)
        self.obs["GLEAM_E(mm)"] = np.transpose(GLEAM_E[:, :, n_warmup:n_warmup+n_calibration], (2, 0, 1))
        
        logger.debug("Get the observation successfully")
        
    def get_sim(self, type="calibration"):
        logger.debug("Getting simulation... ...")
        
        # set sim, calibration period
        n_warmup = len(self.warmup_date_eval)
        n_calibration = len(self.calibrate_date_eval)
        n_verification = len(self.verify_date_eval)

        self.sim = {}

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

        # read
        with Dataset(self.sim_path, "r") as sim_dataset:
            # lon, lat
            sim_lon = sim_dataset["lon"][:]
            sim_lat = sim_dataset["lat"][:]
            sim_time = sim_dataset["time"]

            sim_time = num2date(sim_time[:], sim_time.units, sim_time.calendar)
            sim_time = sim_time[n_warmup:]
            logger.info(f"get simluation between {sim_time[0].year, sim_time[0].month, sim_time[0].day}-{sim_time[-1].year, sim_time[-1].month, sim_time[-1].day}")

            # streamflow: OUT_DISCHARGE
            outlet_index = np.where(sim_lat==self.snaped_outlet_lats[0])[0][0], np.where(sim_lon==self.snaped_outlet_lons[0])[0][0]  # lat, lon
            self.sim["streamflow(m3/s)"] = sim_dataset.variables["OUT_DISCHARGE"][n_warmup:, outlet_index[0], outlet_index[1]]

            # baseflow: OUT_BASEFLOW

            # sm: OUT_SOIL_MOIST
            self.sim["SMtotal(mm)"] = np.nansum(sim_dataset.variables["OUT_SOIL_MOIST"][n_warmup:, :, :, :], axis=1)

            # E: OUT_EVAP
            self.sim["E(mm)"] = sim_dataset.variables["OUT_EVAP"][n_warmup:, :, :]

        logger.debug("Get the simulation successfully")
        
    def createFitness(self):
        # KEG, ESS_ET (EOF_similarity_score), ESS_SM
        creator.create("Fitness", base.Fitness, weights=(1.0, 1.0, 1.0))

    def samplingInd(self):
        logger.debug("Starting parameter sampling process... ...")

        # n_samples
        n_samples = 1

        # get bounds
        bounds = self.paramManager.vector_bounds()
        
        # sample
        params_samples = sampling_LHS_2(n_samples, bounds)
        params_samples = params_samples.flatten().tolist()

        return creator.Individual(params_samples)

    @clock_decorator(print_arg_ret=True)
    def run_vic(self):
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

    def adjust_vic_params_level0(self, g_params):
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
            buildParam_level0_interface_instance.set_coord_map()
            buildParam_level0_interface_instance.params_dataset_level0 = params_dataset_level0
            buildParam_level0_interface_instance.set_dims()
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
        # wp < fc
        # Wpwp_FRACT < Wcr_FRACT
        # depth_layer0 < depth_layer1
        # no nan in infilt
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
        
    def evaluate(self, ind):
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
            return (-9999.0, -9999.0, -9999.0)

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
        # Evaluate performance
        logger.info("Evaluating model performance")
        self.get_sim()

        try:
            # discharge
            sim_streamflow = self.sim["streamflow(m3/s)"].filled(0)
            obs_streamflow = self.obs["streamflow(m3/s)"].values.flatten()
            fitness_streamflow = EvaluationMetric(sim_streamflow, obs_streamflow).KGE_m()

            # SMtotal
            #ESS_SMsurf = EvaluationMetric(self.sim["SMsurf(mm)"], self.obs["GLEAM_SMsurf(mm)"]).spatialPCC(mask=self.sim["SMsurf(mm)"].mask[0, :, :])
            #ESS_SMsurf_timemedian = np.nanmedian(ESS_SMsurf)
            # KGE_SMtotal = EvaluationMetric(sim_SMtotal, self.obs["GLEAM_SMsurf(mm)"]).spatialPCC(mask=self.sim["SMsurf(mm)"].mask[0, :, :])
            # total_depth = self.TF_VIC_class.total_depth(self.soillayerresampler.orig_total, *g_params["total_depths"]["optimal"])
            sim_SMtotal = np.nanmean(self.sim["SMtotal(mm)"], axis=(1, 2))
            obs_SMtotal = np.nanmean(self.obs["GLEAM_SMtotal(mm)"], axis=(1, 2))
            fitness_SMtotal = EvaluationMetric(sim_SMtotal, obs_SMtotal).R()[0]
            
            # E
            # ESS_E = EvaluationMetric(self.sim["E(mm)"], self.obs["GLEAM_E(mm)"]).spatialPCC(mask=self.sim["E(mm)"].mask[0, :, :])
            # ESS_E_timemedian = np.nanmedian(ESS_E)
            sim_E = np.nanmean(self.sim["E(mm)"], axis=(1, 2))
            obs_E = np.nanmean(self.obs["GLEAM_E(mm)"], axis=(1, 2))
            fitness_E = EvaluationMetric(sim_E, obs_E).KGE_m()

            # combine
            fitness = (fitness_streamflow, fitness_SMtotal, fitness_E)  # ESS_SMtotal_timemedian

            # plot discharge
            logger.info("Generating simulation plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.calibrate_date_eval, sim_streamflow, "r-", label=f"sim({round(fitness_streamflow, 2)})", linewidth=0.5)
            ax.plot(self.calibrate_date_eval, obs_streamflow, "k-", label="obs", linewidth=1)
            ax.set_xlabel("date")
            ax.set_ylabel("discharge m3/s")
            ax.legend()
            fig.savefig(
                os.path.join(self.evb_dir.VICResults_fig_dir, "evaluate_discharge.tiff")
            )

            # plot SM
            # fig, axes = plt.subplots(1, 2)
            # im1 = axes[0].imshow(np.nanmedian(self.sim["SMsurf(mm)"], axis=0))
            # im2 = axes[1].imshow(np.ma.array(np.nanmedian(self.obs["GLEAM_SMsurf(mm)"], axis=0), mask=self.sim["SMsurf(mm)"].mask[0, :, :]))
            # axes[0].set_title(f"Simulated SMsurf(mm), sim({round(fitness[1], 2)})")
            # axes[1].set_title("Observed GLEAM_SMsurf(mm)")
            # cbar1 = fig.colorbar(im1, ax=axes[0], shrink=0.8, orientation="horizontal")
            # cbar2 = fig.colorbar(im2, ax=axes[1], shrink=0.8, orientation="horizontal")
            # cbar1.set_label('SM mm')
            # cbar2.set_label('SM mm')
            # fig.savefig(os.path.join(self.evb_dir.VICResults_fig_dir, "evaluate_SMsurf.tiff"))
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.calibrate_date_eval, (sim_SMtotal - np.nanmean(sim_SMtotal)) / np.std(sim_SMtotal), "r-", label=f"sim({round(fitness_SMtotal, 2)})", linewidth=0.5)
            ax.plot(self.calibrate_date_eval, (obs_SMtotal - np.nanmean(obs_SMtotal)) / np.std(obs_SMtotal), "k-", label="obs", linewidth=1)
            ax.set_xlabel("date")
            ax.set_ylabel("Normalized SM")
            ax.legend()
            fig.savefig(
                os.path.join(self.evb_dir.VICResults_fig_dir, "evaluate_SMtotal.tiff")
            )
            
            # import matplotlib
            # import matplotlib.pyplot as plt
            # matplotlib.use("TkAgg")

            # fig, axes = plt.subplots(1, 2)
            # index=100
            # im1 = axes[0].imshow(self.sim["SMtotal(mm)"][index, :, :])
            # im2 = axes[1].imshow(np.ma.array(self.obs["GLEAM_SMtotal(mm)"][index, :, :], mask=self.sim["SMtotal(mm)"].mask[0, :, :]))
            # axes[0].set_title(f"Simulated SMtotal(mm), sim({round(fitness[1], 2)})")
            # axes[1].set_title("Observed GLEAM_SMtotal(mm)")
            # cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.8, orientation="horizontal")
            # cbar.set_label('SM mm')

            # plot E
            # fig, axes = plt.subplots(1, 2)
            # im1 = axes[0].imshow(np.nanmedian(self.sim["E(mm)"], axis=0))
            # im2 = axes[1].imshow(np.ma.array(np.nanmedian(self.obs["GLEAM_E(mm)"], axis=0), mask=self.sim["E(mm)"].mask[0, :, :]))
            # axes[0].set_title(f"Simulated E(mm), sim({round(fitness[2], 2)})")
            # axes[1].set_title("Observed GLEAM_E(mm)")
            # cbar1 = fig.colorbar(im1, ax=axes[0], shrink=0.8, orientation="horizontal")
            # cbar2 = fig.colorbar(im2, ax=axes[1], shrink=0.8, orientation="horizontal")
            # cbar1.set_label('E mm')
            # cbar2.set_label('E mm')
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.calibrate_date_eval, sim_E, "r-", label=f"sim({round(fitness_E, 2)})", linewidth=0.5)
            ax.plot(self.calibrate_date_eval, obs_E, "k-", label="obs", linewidth=1)
            ax.set_xlabel("date")
            ax.set_ylabel("E mm")
            ax.legend()
            fig.savefig(
                os.path.join(self.evb_dir.VICResults_fig_dir, "evaluate_E.tiff")
            )

            # fig, axes = plt.subplots(1, 2)
            # index=100
            # im1 = axes[0].imshow(self.sim["E(mm)"][index, :, :])
            # im2 = axes[1].imshow(np.ma.array(self.obs["GLEAM_E(mm)"][index, :, :], mask=self.sim["E(mm)"].mask[0, :, :]))
            # axes[0].set_title(f"Simulated E(mm), sim({round(fitness[1], 2)})")
            # axes[1].set_title("Observed GLEAM_E(mm)")
            # cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.8, orientation="horizontal")
            # cbar.set_label('E mm')

            # fig.savefig(
            #     os.path.join(self.evb_dir.VICResults_fig_dir, "evaluate_E.tiff")
            # )

            # Ensure fitness is valid
            if sum(np.isnan(fitness)):
                logger.warning(
                    "Fitness calculation resulted in NaN. Assigning fitness = -9999.0"
                )
                fitness = (-9999.0, -9999.0, -9999.0)

        except:
            fitness = (-9999.0, -9999.0, -9999.0)

        logger.info(f"Evaluation completed. Fitness: {fitness}")

        return fitness

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
                self.toolbox.mutate(mutant, self.low, self.up, len(self.low))
                del mutant.fitness.values

        logger.info("Applying genetic operators to offspring successfully")
    

def calibrate_JRB():
    # build evb
    evb_dir_modeling = build_modeling_dir(subname=f"{station_name}_{model_scale}")

    evb_dir_modeling.vic_exe_path = "/home/xudong/VIC/vic_image.exe"

    # read dpc
    dpc_VIC_level0 = dataProcess_VIC_level0_JRB(evb_dir_modeling._dpc_VIC_level0_path)
    dpc_VIC_level1 = dataProcess_VIC_level1_JRB(evb_dir_modeling._dpc_VIC_level1_path)
    dpc_VIC_level1_GLEAM = dataProcess_VIC_level1_GLEAM_JRB(os.path.join(evb_dir_modeling.dpcFile_dir, "dpc_VIC_level1_GLEAM.pkl"))
    dpc_VIC_level3 = dataProcess_VIC_level3_JRB(evb_dir_modeling._dpc_VIC_level3_path)

    # read domain
    domain_dataset = readDomain(evb_dir_modeling)
    
    # read snaped outlet gdf
    station_id = basin_outlets_reference_i_map[station_name]
    snaped_outlet_gdf = gpd.read_file(os.path.join(
        evb_dir_modeling.Hydroanalysis_dir,
        "wbw_working_directory_level1",
        f"snaped_outlet_with_reference_{station_id}.shp"
    ))
    
    # nsgaII set
    algParams = {"popSize": 20, "maxGen": 200, "cxProb": 0.7, "mutateProb": 0.2}
    # algParams = {"popSize": 5, "maxGen": 200, "cxProb": 0.7, "mutateProb": 0.2}
    
    # GlobalParam_dict
    GlobalParam_dict = {
            "Simulation": {
                "MODEL_STEPS_PER_DAY": "1",
                "SNOW_STEPS_PER_DAY": "8",
                "RUNOFF_STEPS_PER_DAY": "8",
                "STARTYEAR": str(warmup_date_period[0][:4]),
                "STARTMONTH": str(int(warmup_date_period[0][4:6])),
                "STARTDAY": str(int(warmup_date_period[0][6:8])),
                "ENDYEAR": str(calibrate_date_period[1][:4]),
                "ENDMONTH": str(int(calibrate_date_period[1][4:6])),
                "ENDDAY": str(int(calibrate_date_period[1][6:8])),
                "OUT_TIME_UNITS": "DAYS",
            },
            "Output": {"AGGFREQ": "NDAYS   1"},
            "OUTVAR1": {"OUTVAR": ["OUT_DISCHARGE", "OUT_SOIL_MOIST", "OUT_EVAP", "OUT_RUNOFF"]},  # "OUT_BASEFLOW", "OUT_RUNOFF", "OUT_SOIL_MOIST"
        }

    nsgaII_VIC_MO = NSGAII_VIC_MO(
        evb_dir_modeling,
        dpc_VIC_level0,
        dpc_VIC_level1,
        dpc_VIC_level3,
        dpc_VIC_level1_GLEAM,
        warmup_date_period,
        calibrate_date_period,
        verify_date_period,
        timestep,
        timestep_evaluate,
        GlobalParam_dict,
        domain_dataset,
        [snaped_outlet_gdf.geometry.x.values[0]],
        [snaped_outlet_gdf.geometry.y.values[0]],
        [station_name],
        buildParam_level0_interface_JRB,
        buildParam_level1_interface_JRB,
        SoilGrids_soillayerresampler,
        TF_VIC,
        [1, 2, 3],
        rvic_OUTPUT_INTERVAL=3600*24,
        rvic_BASIN_FLOWDAYS=50,
        rvic_SUBSET_DAYS=10,
        rvic_uhbox_dt=60,
        algParams=algParams,
        save_path=evb_dir_modeling.calibrate_cp_path,
        reverse_lat=True,
        parallel=False,
    )
    
    # calibrate
    calibrate_bool = True
    if calibrate_bool:
        nsgaII_VIC_MO.run()


if __name__ == "__main__":
    calibrate_JRB()
    
    
