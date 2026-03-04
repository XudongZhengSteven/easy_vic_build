# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
from deap import base, creator, tools
from netCDF4 import Dataset, num2date

from general_info import *
import json

from HRB_build_dpc import dataProcess_VIC_level0_HRB, dataProcess_VIC_level1_HRB, dataProcess_VIC_level3_HRB
from HRB_build_Param import buildParam_level0_interface_HRB, buildParam_level1_interface_HRB, buildParam_level0_interface_Nijssen_spatially_uniform_HRB
from HRB_extractData_func.Extract_SoilGrids1km import SoilGrids_soillayerresampler

from easy_vic_build.bulid_Domain import remap_level1_to_level0_mask
from easy_vic_build.build_Param import scaling_level0_to_level1
from easy_vic_build.build_GlobalParam import buildGlobalParam
from easy_vic_build.build_RVIC_Param import buildConvCFGFile, buildRVICParam

from easy_vic_build.tools.dpc_func.basin_grid_func import createStand_grids_lat_lon_from_gridshp, gridshp_index_to_grid_array_index, retriveArray_to_gridshp_values_list

from easy_vic_build.tools.calibrate_func.algorithm_NSGAII import NSGAII_Base
from easy_vic_build.tools.calibrate_func.evaluate_metrics import EvaluationMetric, SignatureEvaluationMetric
from easy_vic_build.tools.calibrate_func.sampling import *
from easy_vic_build.tools.calibrate_func.algorithm_CMA_ES import CMA_ES_Base

from easy_vic_build.tools.params_func.params_set import *
from easy_vic_build.tools.params_func.TransferFunction import TF_VIC

from easy_vic_build.tools.decoractors import clock_decorator
from easy_vic_build.tools.routing_func.create_uh import createGUH
from easy_vic_build.tools.nested_basin_func.nested_basin_func import cal_unique_mask_nested_basin
from easy_vic_build.tools.utilities import *
try:
    from rvic.parameters import parameters as rvic_parameters
    from rvic.convolution import convolution

    HAS_RVIC = True
except:
    HAS_RVIC = False

# soil layer set
def set_g_params_soilGrids_layer(g_params):
    g_params["soil_layers_breakpoints"] = {
        "default": [1, 5],
        "boundary": [[1, 3], [2, 5]],
        "type": int,
        "optimal": [None, None],
        "free": True,
    }
    return g_params

# build_basin_hierarchy
def build_basin_hierarchy(evb_dir_hydroanalysis, evb_dir_modeling):
    # build basin_hierarchy
    dpc_VIC_level0 = dataProcess_VIC_level0_HRB(evb_dir_modeling._dpc_VIC_level0_path)
    dpc_VIC_level1 = dataProcess_VIC_level1_HRB(evb_dir_modeling._dpc_VIC_level1_path)
    grid_shp_level0 = dpc_VIC_level0.get_data_from_cache("grid_shp")[0]
    grid_shp_level1 = dpc_VIC_level1.get_data_from_cache("grid_shp")[0]
    
    stand_grids_lat_level0, stand_grids_lon_level0 = createStand_grids_lat_lon_from_gridshp(
        grid_shp_level0, grid_res=None, reverse_lat=reverse_lat
    )
    
    stand_grids_lat_level1, stand_grids_lon_level1 = createStand_grids_lat_lon_from_gridshp(
        grid_shp_level1, grid_res=None, reverse_lat=reverse_lat
    )
    
    # read basins
    nested_basins_shp = gpd.read_file(os.path.join(evb_dir_hydroanalysis.Hydroanalysis_dir, "wbw_working_directory_level0", "basins_vector_outlets_with_reference.shp"))
    basin_shps = {
        "hanzhong": nested_basins_shp.iloc[0:1, :],
        "yangxian": nested_basins_shp.iloc[1:2, :],
        "lianghekou": nested_basins_shp.iloc[2:3, :],
        "shiquan": nested_basins_shp.iloc[3:4, :],
        "youshui": nested_basins_shp.iloc[4:5, :],
    }
    
    # enforce_unique_masks
    unique_masks_level1 = cal_unique_mask_nested_basin(
        station_names,
        grid_shp_level1,
        basin_shps,
        main_basin_shp=basin_shps["shiquan"],
        plot=False
    )
    
    # get subbasin_masks_level0
    masks_level0 = {}
    for station_name in station_names:
        lon_level1 = stand_grids_lon_level1
        lat_level1 = stand_grids_lat_level1
        lon_level0 = stand_grids_lon_level0
        lat_level0 = stand_grids_lat_level0
        mask_level1 = unique_masks_level1[station_name]
        
        mask_level0 = remap_level1_to_level0_mask(
            lon_level1, lat_level1, mask_level1,
            lon_level0, lat_level0
        )
        
        masks_level0[station_name] = mask_level0
        
    basin_hierarchy = {
        "station_names": station_names,
        "subbasin_masks_level0": masks_level0,
    }

    return basin_hierarchy


def get_inds_NSGAII(evb_dir_modeling, ind_num=5, max_num=30, percentile=None, cp_path=None):
    cp_path = cp_path if cp_path is not None else evb_dir_modeling.calibrate_cp_path
    with open(cp_path, "rb") as f:
        state = pickle.load(f)
        history = state["history"]
    
    history = history[:max_num]
    all_first_front = []
    all_best_ind = []
    all_best_ind_fitness = []

    for h in history:
        first_front = h["first_front"]
        first_front_fitness = [ind.fitness for ind in first_front]
        
        best_ind_index = np.argmax([fitness.values[-1] for fitness in first_front_fitness])  # best_ind, based on the fitness[-1] (shiquan max)
        best_ind = first_front[best_ind_index]
        best_ind_fitness = best_ind.fitness

        all_first_front.append(first_front)
        all_best_ind.append(best_ind)
        all_best_ind_fitness.append(best_ind_fitness.values[-1])

    ind_num = ind_num if ind_num is not None else int(len(all_best_ind_fitness)*percentile)

    first_num_best_ind_index = sorted(range(len(all_best_ind_fitness)), key=lambda i: all_best_ind_fitness[i], reverse=True)[:ind_num]
    first_num_best_ind = [all_best_ind[i] for i in first_num_best_ind_index]
    first_num_best_ind_fitness = [all_best_ind_fitness[i] for i in first_num_best_ind_index]

    first_num_data = {
        "first_num_best_ind_index": first_num_best_ind_index,
        "first_num_best_ind": first_num_best_ind,
        "first_num_best_ind_fitness": first_num_best_ind_fitness
    }

    return first_num_data


def get_inds_CMA_ES(evb_dir_modeling, ind_num=5, max_num=30, percentile=None, cp_path=None):
    cp_path = cp_path if cp_path is not None else evb_dir_modeling.calibrate_cp_path
    with open(cp_path, "rb") as f:
        state = pickle.load(f)
        history = state["history"]

    history = history[:max_num]
    all_pop = []
    all_pop_fitness = []
    all_best_ind = []
    all_best_ind_fitness = []

    for h in history:
        pop = h["population"]
        best_ind = h["best_ind"]
        
        pop_fitness = [ind.fitness.values[0] for ind in pop]
        best_ind_fitness = best_ind.fitness.values[0]

        all_pop.append(pop)
        all_pop_fitness.append(pop_fitness)
        all_best_ind.append(best_ind)
        all_best_ind_fitness.append(best_ind_fitness)

    all_data = {
        "all_pop": all_pop,
        "all_pop_fitness": all_pop_fitness,
        "all_best_ind": all_best_ind,
        "all_best_ind_fitness": all_best_ind_fitness
    }

    ind_num = ind_num if ind_num is not None else int(len(all_best_ind_fitness)*percentile)
    first_num_best_ind_index = sorted(range(len(all_best_ind_fitness)), key=lambda i: all_best_ind_fitness[i], reverse=True)[:ind_num]
    first_num_best_ind = [all_best_ind[i] for i in first_num_best_ind_index]
    first_num_best_ind_fitness = [all_best_ind_fitness[i] for i in first_num_best_ind_index]

    first_num_data = {
        "first_num_best_ind_index": first_num_best_ind_index,
        "first_num_best_ind": first_num_best_ind,
        "first_num_best_ind_fitness": first_num_best_ind_fitness
    }

    return first_num_data


def copy_result_to_case_dir(evb_dir_modeling, case_dir, save_type="cali", suffix=1):
    suffix_all = save_type + str(suffix)

    shutil.copy(os.path.join(evb_dir_modeling.ParamFile_dir, "params_level0.nc"), os.path.join(case_dir, f"params_level0_{suffix_all}.nc"))
    logger.info(f'mv {os.path.join(evb_dir_modeling.ParamFile_dir, "params_level0.nc")} to {os.path.join(case_dir, f"params_level0_{suffix_all}.nc")}')

    shutil.copy(os.path.join(evb_dir_modeling.ParamFile_dir, "params_level1.nc"), os.path.join(case_dir, f"params_level1_{suffix_all}.nc"))
    logger.info(f'mv {os.path.join(evb_dir_modeling.ParamFile_dir, "params_level1.nc")} to {os.path.join(case_dir, f"params_level1_{suffix_all}.nc")}')

    VIC_Result_fn = [fn for fn in os.listdir(evb_dir_modeling.VICResults_dir) if fn.endswith(".nc")][0]
    shutil.copy(os.path.join(evb_dir_modeling.VICResults_dir, VIC_Result_fn), os.path.join(case_dir, VIC_Result_fn.replace(".nc", f"_{suffix_all}.nc")))
    logger.info(f'mv {os.path.join(evb_dir_modeling.VICResults_dir, VIC_Result_fn)} to {os.path.join(case_dir, VIC_Result_fn.replace(".nc", f"_{suffix_all}.nc"))}')
    
    VIC_Result_fig_fn = [fn for fn in os.listdir(os.path.join(evb_dir_modeling.VICResults_dir, "Figs")) if fn.endswith(".tiff")][0]
    shutil.copy(os.path.join(evb_dir_modeling.VICResults_dir, "Figs", VIC_Result_fig_fn), os.path.join(case_dir, VIC_Result_fig_fn.replace(".tiff", f"_{suffix_all}.tiff")))
    logger.info(f'mv {os.path.join(evb_dir_modeling.VICResults_dir, "Figs", VIC_Result_fig_fn)} to {os.path.join(case_dir, VIC_Result_fig_fn.replace(".tiff", f"_{suffix_all}.tiff"))}')

    RVIC_Result_fn = [fn for fn in os.listdir(os.path.join(evb_dir_modeling.RVICConv_dir, "hist")) if fn.endswith(".nc")][0]
    shutil.copy(os.path.join(evb_dir_modeling.RVICConv_dir, "hist", RVIC_Result_fn), os.path.join(case_dir, RVIC_Result_fn.replace(".nc", f"_{suffix_all}.nc")))
    logger.info(f'mv {os.path.join(evb_dir_modeling.RVICConv_dir, "hist", RVIC_Result_fn)} to {os.path.join(case_dir, RVIC_Result_fn.replace(".nc", f"_{suffix_all}.nc"))}')


class Calibrate_VIC_HRB:
    # -----------------------------
    #  general setting
    # ----------------------------- 
    def set_date(self, timestep, timestep_evaluate):
        logger.debug("setting date... ...")
        self.warmup_date = pd.date_range(warmup_date_period[0], warmup_date_period[-1], freq=timestep)
        self.warmup_date_eval = pd.date_range(warmup_date_period[0], warmup_date_period[-1], freq=timestep_evaluate)

        self.calibrate_date = pd.date_range(calibrate_date_period[0], calibrate_date_period[-1], freq=timestep)
        self.calibrate_date_eval = pd.date_range(calibrate_date_period[0], calibrate_date_period[-1], freq=timestep_evaluate)

        self.verify_date = pd.date_range(verify_date_period[0], verify_date_period[-1], freq=timestep)
        self.verify_date_eval = pd.date_range(verify_date_period[0], verify_date_period[-1], freq=timestep_evaluate)

        self.total_date = pd.date_range(calibrate_date_period[0], verify_date_period[-1], freq=timestep)
        self.total_date_eval = pd.date_range(calibrate_date_period[0], verify_date_period[-1], freq=timestep_evaluate)

        self.plot_date_dict = {
            "calibration": self.calibrate_date_eval,
            "validation": self.verify_date_eval,
            "total": self.total_date_eval
        }

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

    # -----------------------------
    #  get obs and sim
    # ----------------------------- 
    def get_obs(self, get_type="calibration", station_name=None):
        logger.debug("Getting observation... ...")
        
        # set set date index
        n_warmup = len(self.warmup_date_eval)
        n_calibration = len(self.calibrate_date_eval)
        n_validation = len(self.verify_date_eval)

        if get_type == "calibration":
            start_index = n_warmup
            end_index = n_warmup + n_calibration
        elif get_type == "validation":
            start_index = n_warmup + n_calibration
            end_index = n_warmup + n_calibration + n_validation
        elif get_type == "total":
            start_index = n_warmup
            end_index = n_warmup + n_calibration + n_validation
        else:
            logger.error("input right get_type")

        # set obs
        self.obs = {}
        
        # read streamflow
        basin_shp_with_streamflow = self.dpc_VIC_level3.get_data_from_cache("streamflow")[0]
        self.gauge_info = self.dpc_VIC_level3.get_data_from_cache("gauge_info")[0]
        
        # get all stations
        self.snaped_outlet_names = [self.gauge_info[key]["station_name"] for key in station_name]
        self.snaped_outlet_lons = [self.gauge_info[key]["gauge_coord(lon, lat)_level1"][0] for key in station_name]
        self.snaped_outlet_lats = [self.gauge_info[key]["gauge_coord(lon, lat)_level1"][1] for key in station_name]
        
        for name in station_name:
            self.obs[f"streamflow(m3/s)_{name}"] = basin_shp_with_streamflow[f"stationdata_streamflow_{name}"][0].iloc[start_index:end_index]
        
        # if len(station_name) > 1: # get all stations
        #     self.snaped_outlet_names = [self.gauge_info[key]["station_name"] for key in station_name]
        #     self.snaped_outlet_lons = [self.gauge_info[key]["gauge_coord(lon, lat)_level1"][0] for key in station_name]
        #     self.snaped_outlet_lats = [self.gauge_info[key]["gauge_coord(lon, lat)_level1"][1] for key in station_name]

        #     for name in station_name:
        #         self.obs[f"streamflow(m3/s)_{name}"] = basin_shp_with_streamflow[f"stationdata_streamflow_{name}"][0].iloc[start_index:end_index]
        
        # else:
        #     name = station_name[0]
        #     self.snaped_outlet_names = [self.gauge_info[name]["station_name"]]
        #     self.snaped_outlet_lons = [self.gauge_info[name]["gauge_coord(lon, lat)_level1"][0]]
        #     self.snaped_outlet_lats = [self.gauge_info[name]["gauge_coord(lon, lat)_level1"][1]]
        #     self.obs[f"streamflow(m3/s)_{name}"] = basin_shp_with_streamflow[f"stationdata_streamflow_{name}"][0].iloc[start_index:end_index]

        logger.info("readed streamflow obs at stations: " + ", ".join(self.snaped_outlet_names))
        
        logger.debug("Get the observation successfully")

    def get_sim_vic(self, get_type="calibration", station_name=None):
        logger.debug("Getting simulation (vic)... ...")
        
        # set sim, calibration period
        n_warmup = len(self.warmup_date_eval)
        n_calibration = len(self.calibrate_date_eval)
        n_validation = len(self.verify_date_eval)

        if get_type == "calibration":
            start_index = n_warmup
            end_index = n_warmup + n_calibration
        elif get_type == "validation":
            start_index = n_warmup + n_calibration
            end_index = n_warmup + n_calibration + n_validation
        elif get_type == "total":
            start_index = n_warmup
            end_index = n_warmup + n_calibration + n_validation
        else:
            logger.error("input right get_type")

        # path
        nc_files = [
            fn for fn in os.listdir(self.evb_dir.VICResults_dir) if fn.endswith(".nc")
        ]
        
        if not nc_files:
            logger.warning("No .nc files found in the VICResults directory")
            return None
        
        self.sim_fn_vic = nc_files[0]
        self.sim_path_vic = os.path.join(self.evb_dir.VICResults_dir, self.sim_fn_vic)
        logger.debug(f"Found simulation file: {self.sim_fn_vic} at {self.sim_path_vic}")

        # read
        with Dataset(self.sim_path_vic, "r") as sim_dataset:
            # lon, lat
            sim_lon = sim_dataset["lon"][:]
            sim_lat = sim_dataset["lat"][:]
            sim_time = sim_dataset["time"]

            sim_time = num2date(sim_time[:], sim_time.units, sim_time.calendar)
            sim_time = sim_time[start_index:end_index]
            logger.info(f"get simluation between {sim_time[0].year, sim_time[0].month, sim_time[0].day}-{sim_time[-1].year, sim_time[-1].month, sim_time[-1].day}")

            # EA: OUT_EVAP
            self.sim[f"EA(mm)"] = sim_dataset.variables["OUT_EVAP"][start_index:end_index, :, :]

            # SM: OUT_SOIL_MOIST
            self.sim[f"SM(mm)"] = sim_dataset.variables["OUT_SOIL_MOIST"][start_index:end_index, :, :, :]

        logger.debug("Get the simulation successfully")

    def get_sim_rvic(self, get_type="calibration", station_name=None):
        logger.debug("Getting simulation (rvic)... ...")
        
        # set date index
        n_warmup = len(self.warmup_date_eval)
        n_calibration = len(self.calibrate_date_eval)
        n_validation = len(self.verify_date_eval)

        if get_type == "calibration":
            start_index = n_warmup
            end_index = n_warmup + n_calibration
        elif get_type == "validation":
            start_index = n_warmup + n_calibration
            end_index = n_warmup + n_calibration + n_validation
        elif get_type == "total":
            start_index = n_warmup
            end_index = n_warmup + n_calibration + n_validation
        else:
            logger.error("input right get_type")

        # path
        nc_files = [
            fn for fn in os.listdir(os.path.join(self.evb_dir.RVICConv_dir, "hist")) if fn.endswith(".nc")
        ]

        if not nc_files:
            logger.warning("No .nc files found in the RVICConv_dir directory")
            return None
    
        self.sim_fn_rvic = nc_files[0]
        self.sim_path_rvic = os.path.join(self.evb_dir.RVICConv_dir, "hist", self.sim_fn_rvic)
        logger.debug(f"Found simulation file: {self.sim_fn_rvic} at {self.sim_path_rvic}")

        # read
        with Dataset(self.sim_path_rvic, "r") as sim_dataset:
            # lon, lat
            sim_time = sim_dataset["time"]

            sim_time = num2date(sim_time[:], sim_time.units, sim_time.calendar)
            sim_time = sim_time[start_index:end_index]
            logger.info(f"get simluation between {sim_time[0].year, sim_time[0].month, sim_time[0].day}-{sim_time[-1].year, sim_time[-1].month, sim_time[-1].day}")

            # streamflow: OUT_DISCHARGE
            if len(station_name) > 1: # get all
                for i, name in enumerate(station_name):
                    self.sim[f"streamflow(m3/s)_{name}"] = sim_dataset.variables["streamflow"][start_index:end_index, i]
            else:
                name = station_name[0]
                self.sim[f"streamflow(m3/s)_{name}"] = sim_dataset.variables["streamflow"][start_index:end_index, 0]

        logger.debug("Get the simulation successfully")

    def get_sim_coupled(self, get_type="calibration", station_name=None):
        logger.debug("Getting simulation... ...")
        
        # set sim, calibration period
        n_warmup = len(self.warmup_date_eval)
        n_calibration = len(self.calibrate_date_eval)
        n_verification = len(self.verify_date_eval)

        if get_type == "calibration":
            start_index = n_warmup
        elif get_type == "validation":
            start_index = n_warmup + n_calibration
        elif get_type == "total":
            start_index = 0
        else:
            logger.error("input right get_type")

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
            sim_time = sim_time[start_index:]
            logger.info(f"get simluation between {sim_time[0].year, sim_time[0].month, sim_time[0].day}-{sim_time[-1].year, sim_time[-1].month, sim_time[-1].day}")

            # streamflow: OUT_DISCHARGE
            if len(station_name) > 1: # get all
                for i, name in enumerate(station_name):
                    outlet_index = (np.where(sim_lat==self.snaped_outlet_lats[i])[0][0], np.where(sim_lon==self.snaped_outlet_lons[i])[0][0])
                    self.sim[f"streamflow(m3/s)_{name}"] = sim_dataset.variables["OUT_DISCHARGE"][start_index:, outlet_index[0], outlet_index[1]]
            else:
                name = station_name[0]
                outlet_index = (np.where(sim_lat==self.snaped_outlet_lats[0])[0][0], np.where(sim_lon==self.snaped_outlet_lons[0])[0][0])
                self.sim[f"streamflow(m3/s)_{name}"] = sim_dataset.variables["OUT_DISCHARGE"][start_index:, outlet_index[0], outlet_index[1]]
            
        logger.debug("Get the simulation successfully")

    # -----------------------------
    #  run vic, rvic
    # ----------------------------- 
    @clock_decorator(print_arg_ret=True)
    def run_vic(self):
        if self.parallel_vic is not None:
            command_run_vic = " ".join(
                [
                    f"mpiexec -np {self.parallel_vic}",
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
        logger.info("running RVIC convolution... ...")
        logger.debug(f"RVIC configuration: {conv_cfg_file_dict}")

        try:
            convolution(conv_cfg_file_dict)
            logger.info("RVIC convolution process successfully")

        except Exception as e:
            logger.error(f"RVIC convolution process failed: {e}", exc_info=True)
            
        return 0
    
    # -----------------------------
    #  adjust parameter from ind
    # ----------------------------- 
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
            self.cols_index_level0,
            self.basin_hierarchy
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
            elev_scaling=self.elev_scaling,
        )
        
        self.scaling_searched_grids_bool_index = searched_grids_bool_index

        # save slope
        if self.slope is None:
            self.slope = params_dataset_level1["slope"][:, :]

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
        
        # input path
        inputs_fpath = [
            os.path.join(self.evb_dir.RVICParam_dir, inputs_f)
            for inputs_f in os.listdir(self.evb_dir.RVICParam_dir)
            if inputs_f.startswith("inputs") and inputs_f.endswith("tar")
        ]

        for fp in inputs_fpath:
            logger.debug(f"Removing old RVIC input file in: {fp}... ...")
            os.remove(fp)
        
        # build rvic_params
        if self.rvic_spatial:
            cfg_params={
                # spatial estimation
                "VELOCITY": "velocity",
                "DIFFUSION": "diffusion",
                "OUTPUT_INTERVAL": self.rvic_OUTPUT_INTERVAL,
                "SUBSET_DAYS": self.rvic_SUBSET_DAYS,
                "CELL_FLOWDAYS": self.rvic_CELL_FLOWDAYS,
                "BASIN_FLOWDAYS": self.rvic_BASIN_FLOWDAYS,
                "CONSTRAIN_FRACTIONS": self.CONSTRAIN_FRACTIONS,
            }

            if self.slope is None:
                with Dataset(self.evb_dir.params_dataset_level1_path) as params_dataset_level1:
                    self.slope = params_dataset_level1["slope"][:, :]

            fd_params={
                # spatial estimation
                "g_velocity": rvic_params["VELOCITY"]["optimal"],
                "g_diffusion": rvic_params["DIFFUSION"]["optimal"],
                "slope": self.slope,
                "TF_VIC_class": self.TF_VIC_class
            }
        else:
            cfg_params={
                # constant estimation
                "VELOCITY": rvic_params["VELOCITY"]["optimal"][0],
                "DIFFUSION": rvic_params["DIFFUSION"]["optimal"][0],
                "OUTPUT_INTERVAL": self.rvic_OUTPUT_INTERVAL,
                "SUBSET_DAYS": self.rvic_SUBSET_DAYS,
                "CELL_FLOWDAYS": self.rvic_CELL_FLOWDAYS,
                "BASIN_FLOWDAYS": self.rvic_BASIN_FLOWDAYS,
                "CONSTRAIN_FRACTIONS": self.CONSTRAIN_FRACTIONS,
            }

            fd_params={
                # constant estimation
                "g_velocity": None,
                "g_diffusion": None,
                "slope": None,
                "TF_VIC_class": self.TF_VIC_class
            }

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
            cfg_params=cfg_params,
            fd_params=fd_params,
            numofproc=self.parallel_rvic_param if self.parallel_rvic_param is not None else 1,
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
        logger.info("Starting to adjust RVIC convolution parameters... ...")

        # Cleanup and directory setup
        logger.debug("Removing old files and creating necessary directories... ...")
        remove_and_mkdir(os.path.join(self.evb_dir.RVICConv_dir, "hist"))
        remove_and_mkdir(os.path.join(self.evb_dir.RVICConv_dir, "logs"))
        remove_and_mkdir(os.path.join(self.evb_dir.RVICConv_dir, "restarts"))
        
        # forcing/input path
        sim_fn = [
            fn for fn in os.listdir(self.evb_dir.VICResults_dir) if fn.endswith(".nc")
        ][0]

        with Dataset(os.path.join(self.evb_dir.VICResults_dir, sim_fn), "r") as sim_dataset:
            hist_length = len(sim_dataset["time"][:])

        # build rvic_conv_cfg_params, construct RUN_STARTDATE from date_period
        logger.debug("Formatting RUN_STARTDATE from date_period... ...")
        RUN_STARTDATE = f"{self.warmup_date_period[0][:4]}-{self.warmup_date_period[0][4:6]}-{self.warmup_date_period[0][6:8]}-00"

        # Build RVIC convolution configuration
        logger.debug("Building RVIC convolution configuration file... ...")
        
        rvic_conv_cfg_params = {
            "RUN_STARTDATE": RUN_STARTDATE,
            "DATL_FILE": sim_fn,
            "PARAM_FILE_PATH": self.rout_param_path,
            "RVICHIST_MFILT": hist_length + 10,
        }

        buildConvCFGFile(self.evb_dir, **rvic_conv_cfg_params)

        # Read and return configuration dictionary
        logger.debug("Reading RVIC convolution configuration file... ...")
        conv_cfg_file_dict = read_cfg_to_dict(self.evb_dir.rvic_conv_cfg_file_path)

        logger.info("Adjusting convolution parameter adjustment successfully")

        return conv_cfg_file_dict
    
    # -----------------------------
    #  simulate based on ind
    # -----------------------------
    def simulate_vic(self, ind):
        logger.info("Starting VIC simulation... ...")
        # =============== get ind ===============
        # format dtype
        ind_format = self.paramManager.format_vector(ind, get_free=True)

        # Extract parameter groups
        param_dict = self.paramManager.to_dict(vector=ind_format, field="optimal", get_free=True)
        
        g_params = param_dict["g_params"]
        
        # =============== adjust vic params based on ind ===============
        # adjust params_dataset_level0 based on g_params
        logger.info("Adjusting params_dataset_level0... ...")
        params_dataset_level0 = self.adjust_vic_params_level0(g_params)

        # Check for constraint violations
        logger.info("Checking parameter constraints")
        constraint_destroy = self.cal_constraint_destroy(params_dataset_level0)
        logger.info(f"Constraint violation: {constraint_destroy}, true means invalid params, set fitness = -9999.0")
        
        if constraint_destroy:
            logger.warning("Invalid parameters detected. Assigning fitness = -9999.0")
            params_dataset_level0.close()
            return False
        
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

        logger.info(f"Current gen: {self.current_generation}, max gen: {self.maxGen}")
        self.run_vic()
        
        logger.info("VIC simulation successfully")
        
        return True
    
    def simulate_rvic(self, ind):
        logger.info("Starting RVIC simulation... ...")
        # =============== get ind ===============
        # format dtype
        ind_format = self.paramManager.format_vector(ind, get_free=True)
        
        # Extract parameter groups
        param_dict = self.paramManager.to_dict(vector=ind_format, field="optimal", get_free=True)
        
        guh_params = param_dict["guh_params"]
        rvic_params = param_dict["rvic_params"]
        
        # =============== adjust rvic params based on ind ===============
        logger.info("Adjusting RVIC parameters... ...")
        self.adjust_rvic_params(guh_params, rvic_params)
        
        # build cfg file
        conv_cfg_file_dict = self.adjust_rvic_conv_params()
        
        # =============== run rvic ===============
        self.run_rvic(conv_cfg_file_dict)
        
        logger.info("RVIC simulation successfully")
        
        return True
    
    def simulate_coupled(self, ind):
        # format dtype
        ind_format = self.paramManager.format_vector(ind, get_free=True)
        
        # Extract parameter groups
        param_dict = self.paramManager.to_dict(vector=ind_format, field="optimal", get_free=True)
        
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
            params_dataset_level0.close()
            return False

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

        logger.info(f"Current gen: {self.current_generation}, max gen: {self.maxGen}")
        self.run_vic()

        return True
    
    def evaluate(self, ind, get_type="calibration", get_obs_bool=False):
        logger.info("Starting evaluate individual... ...")

        # get obs
        if get_obs_bool:
            self.get_obs(get_type=get_type, station_name=self.station_names)
        
        # --- simulate ----
        logger.info("Starting simulating... ...")

        # offline sim
        sim_right_vic = self.simulate_vic(ind)
        if sim_right_vic:
            self.simulate_rvic(ind)
        else:
            return self.worst_fitness
        
        # couple sim
        # sim_right = self.simulate_coupled(ind)
        # if not sim_right:
        #     return self.worst_fitness
        # self.simulate_rvic(ind)

        # --- evaluate ----
        logger.info("Starting evaluating... ...")

        # offline get sim
        # self.get_sim_vic(station_name=self.station_names_cali)
        self.get_sim_rvic(get_type=get_type, station_name=self.station_names)

        # couple get sim
        # self.get_sim_coupled(get_type=get_type, station_name=self.station_names_cali)

        # set plot type
        plot_date = self.plot_date_dict[get_type]

        try:
            fitness_list = []

            # discharge
            fig, axes = plt.subplots(len(self.station_names), 1, figsize=(12, 7), sharex=True)

            for i, name in enumerate(self.station_names):
                sim_streamflow = self.sim[f"streamflow(m3/s)_{name}"].filled(0)
                obs_streamflow = self.obs[f"streamflow(m3/s)_{name}"].values.flatten()

                # KGE
                fitness_streamflow_KGE = EvaluationMetric(sim_streamflow, obs_streamflow).KGE_m()

                # # high flow
                # fitness_streamflow_FHVBias = SignatureEvaluationMetric(sim_streamflow, obs_streamflow).FHVBias(q_high=0.98)

                # # low flow
                # fitness_streamflow_FLVBias = SignatureEvaluationMetric(sim_streamflow, obs_streamflow).FLVBias(q_low=0.3)
                
                # plot discharge
                axes[i].plot(plot_date, obs_streamflow, "k-", label=f"obs({name})", linewidth=1)
                axes[i].plot(plot_date, sim_streamflow, "r-", label=f"sim", linewidth=0.5)

                fitness_str =  (
                    f"KGE = {fitness_streamflow_KGE:.2f}\n"
                    # f"FHVBias = {fitness_streamflow_FHVBias:.2f}\n"
                    # f"FLVBias = {fitness_streamflow_FLVBias:.2f}"
                )

                axes[i].text(
                    0.05, 0.95, fitness_str,
                    transform=axes[i].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                )

                if name == self.snaped_outlet_names[-1]:
                    axes[i].set_xlabel("date")
                    axes[i].set_ylabel("discharge m3/s")

                axes[i].legend(loc="upper right")

                if name in self.station_names_cali:
                    fitness_list.extend([fitness_streamflow_KGE])  # , abs(fitness_streamflow_FHVBias), abs(fitness_streamflow_FLVBias)

            fig.savefig(os.path.join(self.evb_dir.VICResults_fig_dir, "evaluate_discharge.tiff"))
            
            # Ensure fitness is valid
            fitness = tuple(fitness_list)

            # avoid nan
            if sum(np.isnan(fitness)):
                logger.warning(
                    "Fitness calculation resulted in NaN. Assigning fitness = -9999.0"
                )
                fitness = self.worst_fitness

        except:
            fitness = self.worst_fitness
        
        logger.info(f"Evaluation completed. Fitness: {fitness}")

        # best fitness
        self.save_best_fitness_in_log(fitness)

        return fitness

    def save_best_fitness_in_log(self, fitness):
        pass


class NSGAII_VIC_HRB(Calibrate_VIC_HRB, NSGAII_Base):

    def __init__(
        self,
        evb_dir,
        warmup_date_period,
        calibrate_date_period,
        verify_date_period,
        timestep,
        timestep_evaluate,
        GlobalParam_dict,
        param_temp=params_minimal_rvic_spatial,
        buildParam_level0_interface_class=buildParam_level0_interface_HRB,
        buildParam_level1_interface_class=buildParam_level1_interface_HRB,
        soillayerresampler=SoilGrids_soillayerresampler,
        TF_VIC_class=TF_VIC,
        nlayer_list=[1, 2, 3],
        rvic_OUTPUT_INTERVAL=86400,
        rvic_BASIN_FLOWDAYS=50,
        rvic_SUBSET_DAYS=10,
        rvic_CELL_FLOWDAYS=1,
        rvic_uhbox_dt=3600,
        algParams={"popSize": 40, "maxGen": 250, "cxProb": 0.7, "mutateProb": 0.2},
        reverse_lat=True,
        parallel_vic=None,
        parallel_rvic_param=None,
        rvic_spatial=False,
        station_names=None,
        station_names_cali=None,
        elev_scaling="Arithmetic_min",
        basin_hierarchy=None,
        CONSTRAIN_FRACTIONS=True,  # nested basin: False
    ):
        # *if parallel, uhbox_dt (rvic_OUTPUT_INTERVAL) should be same as VIC output (global param)
        logger.info(
            "Initializing NSGAII_VIC instance with provided parameters... ..."
        )

        # station
        self.station_names = station_names
        self.station_number = len(self.station_names)
        self.station_names_cali = station_names_cali
        self.station_number_cali = len(self.station_names_cali)

        if self.station_number_cali == 1:
            logger.warning("station numer is 1, this is not suitable for multi-objective calibration")

        # evb set
        self.evb_dir = evb_dir

        # best_fitness, worst_fitness, weights set
        self.best_fitness = tuple([1.0] * self.station_number_cali)
        self.worst_fitness = tuple([-9999.0] * self.station_number_cali)
        self.weights = tuple([1.0] * self.station_number_cali)

        self.cali_best_fitness = self.worst_fitness

        # read
        self.dpc_VIC_level0 = dataProcess_VIC_level0_HRB(self.evb_dir.dpc_VIC_level0_path)
        self.dpc_VIC_level1 = dataProcess_VIC_level1_HRB(self.evb_dir.dpc_VIC_level1_path)
        self.dpc_VIC_level3 = dataProcess_VIC_level3_HRB(self.evb_dir.dpc_VIC_level3_path)
        
        self.grid_shp_level0 = self.dpc_VIC_level0.get_data_from_cache("grid_shp")[0]
        self.grid_shp_level1 = self.dpc_VIC_level1.get_data_from_cache("grid_shp")[0]
        
        self.reverse_lat = reverse_lat
        self.rvic_OUTPUT_INTERVAL = rvic_OUTPUT_INTERVAL  # 3600, 86400
        self.rvic_BASIN_FLOWDAYS = rvic_BASIN_FLOWDAYS
        self.rvic_SUBSET_DAYS = rvic_SUBSET_DAYS
        self.rvic_CELL_FLOWDAYS = rvic_CELL_FLOWDAYS
        self.rvic_uhbox_dt = rvic_uhbox_dt
        self.parallel_vic = parallel_vic
        self.parallel_rvic_param = parallel_rvic_param
        self.rvic_spatial = rvic_spatial
        self.elev_scaling = elev_scaling
        self.basin_hierarchy = basin_hierarchy
        self.CONSTRAIN_FRACTIONS = CONSTRAIN_FRACTIONS

        logger.info(
            f"Date periods: {warmup_date_period}, {calibrate_date_period}, {verify_date_period}"
        )

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

        self.sim = {}
        self.obs = {}

        # period
        self.warmup_date_period = warmup_date_period
        self.calibrate_date_period = calibrate_date_period
        self.verify_date_period = verify_date_period
        self.set_date(timestep, timestep_evaluate)

        # set coord map
        self.set_coord_map()

        # read domain
        self.domain_dataset = readDomain(self.evb_dir)
        # self.mask = self.domain_dataset.variables["mask"][:, :]
        # self.mask_1D = retriveArray_to_gridshp_values_list(
        #     self.mask,
        #     self.rows_index_level1,
        #     self.cols_index_level1,
        # )

        # clear Param
        logger.info("Clear previous parameters from the VIC model directory")
        clearParam(self.evb_dir)
        
        # buildParam set
        self.buildParam_level0_interface_class = buildParam_level0_interface_class
        self.buildParam_level1_interface_class = buildParam_level1_interface_class
        self.soillayerresampler = soillayerresampler
        self.TF_VIC_class = TF_VIC_class
        self.nlayer_list = nlayer_list if nlayer_list is not None else [1, 2, 3]
        self.slope = None
        
        # param dict set
        self.paramManager = ParamManager(param_temp)
        self.low = [b[0] for b in self.paramManager.vector_bounds(get_free=True)]
        self.up = [b[1] for b in self.paramManager.vector_bounds(get_free=True)]
        self.NDim = len(self.low)
        
        # set GlobalParam_dict
        self.GlobalParam_dict = GlobalParam_dict
        self.set_GlobalParam_dict(GlobalParam_dict)

        # get obs
        logger.debug("Load observational data")
        self.get_obs(get_type="calibration", station_name=self.station_names)
        
        # sim path
        self.sim_path = ""

        # init algorithm
        NSGAII_Base.__init__(self, algParams, self.evb_dir.calibrate_cp_path)
        logger.info("Initialized")
    
    def save_best_fitness_in_log(self, fitness):
        if sum(w * f for w, f in zip(self.weights, fitness)) / sum(self.weights) > \
        sum(w * f for w, f in zip(self.weights, self.cali_best_fitness)) / sum(self.weights) + 1e-6:
            self.cali_best_fitness = fitness
        
        logger.info(f"best fitness: {self.cali_best_fitness}")

    def createFitness(self):
        creator.create("Fitness", base.Fitness, weights=self.weights)

    def samplingLHS(self):
        logger.debug("Starting parameter sampling process... ...")

        # get bounds
        bounds = self.paramManager.vector_bounds(get_free=True)
        
        # sample
        params_samples = sampling_LHS_2(self.popSize, bounds)  # shape (popSize, n_params)

        return params_samples
    
    def registerPop(self):
        """Registers the population initialization function with the toolbox."""
        self.lhs_index = 0
        self.lhs_samples = self.samplingLHS()
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        
    def samplingInd(self):
        sample = self.lhs_samples[self.lhs_index]
        self.lhs_index += 1
        
        if self.lhs_index > len(self.lhs_samples):
            raise IndexError("LHS samples exhausted!")
        
        return creator.Individual(sample.tolist())
    
    def get_eta(self):
        # define eta based on current generation
        max_eta = 20.0
        min_eta = 5.0
        eta = max_eta - (self.current_generation / self.maxGen) * (max_eta - min_eta)
        
        return eta
    
    def operatorMate(self, parent1, parent2):
        logger.debug("Performing crossover between two parents... ...")
        return tools.cxSimulatedBinaryBounded(
            parent1, parent2, eta=self.get_eta(), low=self.low, up=self.up
        )
        
    def operatorMutate(self, ind):
        logger.debug("Performing mutation on individual... ...")
        return tools.mutPolynomialBounded(ind, eta=self.get_eta(), low=self.low, up=self.up, indpb=1 / self.NDim)

    def operatorSelect(self, population, popSize):
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
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # mutate
        logger.debug("Starting mutation operation... ...")
        for mutant in offspring:
            if random.random() <= self.toolbox.mutateProb:
                logger.debug(f"Mutation applied to {mutant}")
                self.toolbox.mutate(mutant)
                del mutant.fitness.values

        logger.info("Applying genetic operators to offspring successfully")


class CMA_ES_VIC_HRB(Calibrate_VIC_HRB, CMA_ES_Base):

    def __init__(
        self,
        evb_dir,
        warmup_date_period,
        calibrate_date_period,
        verify_date_period,
        timestep,
        timestep_evaluate,
        GlobalParam_dict,
        param_temp=params_minimal_rvic_spatial,
        buildParam_level0_interface_class=buildParam_level0_interface_HRB,
        buildParam_level1_interface_class=buildParam_level1_interface_HRB,
        soillayerresampler=SoilGrids_soillayerresampler,
        TF_VIC_class=TF_VIC,
        nlayer_list=[1, 2, 3],
        rvic_OUTPUT_INTERVAL=86400,
        rvic_BASIN_FLOWDAYS=50,
        rvic_SUBSET_DAYS=10,
        rvic_CELL_FLOWDAYS=1,
        rvic_uhbox_dt=3600,
        algParams={"popSize":20, "maxGen": 250, "sigma": 0.5},
        reverse_lat=True,
        parallel_vic=None,
        parallel_rvic_param=None,
        rvic_spatial=False,
        station_names=None,
        station_names_cali=None,
        elev_scaling="Arithmetic_min",
        basin_hierarchy=None,
        CONSTRAIN_FRACTIONS=True,
    ):
        # this algorithm is designed for one object calibration
        logger.info(
            "Initializing CMA_ES instance with provided parameters... ..."
        )
        
        # station
        self.station_names = station_names
        self.station_number = len(self.station_names)
        self.station_names_cali = station_names_cali
        self.station_number_cali = len(self.station_names_cali)
        assert self.station_number_cali == 1, "station_name must be 1 for single objection calibration"

        # evb set
        self.evb_dir = evb_dir

        # best_fitness, worst_fitness, weights set
        self.best_fitness = (1.0, )
        self.worst_fitness = (-9999.0, )
        self.weights = (1.0, )

        self.cali_best_fitness = self.worst_fitness

        # path set for station
        # self.evb_dir.domainFile_path = self.evb_dir.domainFile_path.replace(".nc", f"_{self.station_names_cali[0]}.nc")
        # self.evb_dir.params_dataset_level0_path = self.evb_dir.params_dataset_level0_path.replace(".nc", f"_{self.station_names_cali[0]}.nc")
        # self.evb_dir.params_dataset_level1_path = self.evb_dir.params_dataset_level1_path.replace(".nc", f"_{self.station_names_cali[0]}.nc")
        # self.evb_dir.calibrate_cp_path = self.evb_dir.calibrate_cp_path.replace(".pkl", f"_{self.station_names_cali[0]}.pkl")
        
        # read
        self.dpc_VIC_level0 = dataProcess_VIC_level0_HRB(self.evb_dir.dpc_VIC_level0_path)
        self.dpc_VIC_level1 = dataProcess_VIC_level1_HRB(self.evb_dir.dpc_VIC_level1_path)
        self.dpc_VIC_level3 = dataProcess_VIC_level3_HRB(self.evb_dir.dpc_VIC_level3_path)

        self.grid_shp_level0 = self.dpc_VIC_level0.get_data_from_cache("grid_shp")[0]
        self.grid_shp_level1 = self.dpc_VIC_level1.get_data_from_cache("grid_shp")[0]

        self.reverse_lat = reverse_lat
        self.rvic_OUTPUT_INTERVAL = rvic_OUTPUT_INTERVAL  # 3600, 86400
        self.rvic_BASIN_FLOWDAYS = rvic_BASIN_FLOWDAYS
        self.rvic_SUBSET_DAYS = rvic_SUBSET_DAYS
        self.rvic_CELL_FLOWDAYS = rvic_CELL_FLOWDAYS
        self.rvic_uhbox_dt = rvic_uhbox_dt
        self.parallel_vic = parallel_vic
        self.parallel_rvic_param = parallel_rvic_param
        self.rvic_spatial = rvic_spatial
        self.elev_scaling = elev_scaling
        self.basin_hierarchy = basin_hierarchy
        self.CONSTRAIN_FRACTIONS = CONSTRAIN_FRACTIONS

        logger.info(
            f"Date periods: {warmup_date_period}, {calibrate_date_period}, {verify_date_period}"
        )

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

        self.sim = {}
        self.obs = {}

        # period
        self.warmup_date_period = warmup_date_period
        self.calibrate_date_period = calibrate_date_period
        self.verify_date_period = verify_date_period
        self.set_date(timestep, timestep_evaluate)

        # set coord map
        self.set_coord_map()

        # read domain
        self.domain_dataset = readDomain(self.evb_dir)
        # self.mask = self.domain_dataset.variables["mask"][:, :]
        # self.mask_1D = retriveArray_to_gridshp_values_list(
        #     self.mask,
        #     self.rows_index_level1,
        #     self.cols_index_level1,
        # )

        # clear Param
        logger.info("Clear previous parameters from the VIC model directory")
        clearParam(self.evb_dir)

        # buildParam set
        self.buildParam_level0_interface_class = buildParam_level0_interface_class
        self.buildParam_level1_interface_class = buildParam_level1_interface_class
        self.soillayerresampler = soillayerresampler
        self.TF_VIC_class = TF_VIC_class
        self.nlayer_list = nlayer_list if nlayer_list is not None else [1, 2, 3]
        self.slope = None

        # param dict set
        self.paramManager = ParamManager(param_temp)
        self.low = [b[0] for b in self.paramManager.vector_bounds(get_free=True)]
        self.up = [b[1] for b in self.paramManager.vector_bounds(get_free=True)]
        self.NDim = len(self.low)
        
        # set GlobalParam_dict
        self.GlobalParam_dict = GlobalParam_dict
        self.set_GlobalParam_dict(GlobalParam_dict)

        # get obs
        logger.debug("Load observational data")
        self.get_obs(get_type="calibration", station_name=self.station_names)
        
        # sim path
        self.sim_path = ""

        # init algorithm
        algParams["dim"] = self.NDim
        CMA_ES_Base.__init__(self, algParams, self.evb_dir.calibrate_cp_path)
        logger.info("Initialized")
    
    def save_best_fitness_in_log(self, fitness):
        # best fitness
        if sum(fitness) > sum(self.cali_best_fitness):
            self.cali_best_fitness = fitness
        
        logger.info(f"best fitness: {self.cali_best_fitness}")

    def createFitness(self):
        creator.create("Fitness", base.Fitness, weights=self.weights)

    def registerGenerate(self):
        def generate_with_bounds():
            inds = self.strategy.generate(creator.Individual)

            inds_clipped = []
            for ind in inds:
                ind_clipped = [max(min(x, up), low) for x, low, up in zip(ind, self.low, self.up)]
                inds_clipped.append(creator.Individual(ind_clipped))  # wrapper to keep fitness

            return inds_clipped

        self.toolbox.register("generate", generate_with_bounds)


def case_set(case_name):
    if case_name == "case1":
        # case1: multi-gauges, calibrate VIC param (uniform), soil depths (uniform), RVIC params (uniform)
        logger.info("case1: multi-gauges, calibrate VIC param (uniform), soil depths (uniform), RVIC params (uniform)")
        case_name = "case1"
        params_case = deepcopy(params_minimal)
        params_case["g_params"] = g_params_Nijssen_spatially_uniform_minimal
        params_case["g_params"] = set_g_params_soilGrids_layer(params_case["g_params"])
        rvic_spatial_case = False
        evb_dir_modeling.calibrate_cp_path = evb_dir_modeling.calibrate_cp_path.replace(".pkl", f"_case1.pkl")
        basin_hierarchy = None
        buildParam_level0_interface_class = buildParam_level0_interface_Nijssen_spatially_uniform_HRB
        Baseflow_scheme = "NIJSSEN2001"
        cali_method = "NSGAII"

    elif case_name == "case2":
        # case2: multi-gauges, calibrate VIC param (distributed), soil depths (uniform), RVIC params (uniform)
        logger.info("case2: multi-gauges, calibrate VIC param (distributed), soil depths (uniform), RVIC params (uniform)")
        case_name = "case2"
        params_case = deepcopy(params_minimal)
        params_case["g_params"] = set_g_params_soilGrids_layer(params_case["g_params"])
        rvic_spatial_case = False
        evb_dir_modeling.calibrate_cp_path = evb_dir_modeling.calibrate_cp_path.replace(".pkl", f"_case2.pkl")
        basin_hierarchy = None
        buildParam_level0_interface_class = buildParam_level0_interface_HRB
        Baseflow_scheme = "NIJSSEN2001"
        cali_method = "NSGAII"

    elif case_name == "case3":
        # case3: multi-gauges, calibrate VIC param (distributed), soil depths (uniform), RVIC params (fixed)
        logger.info("case3: multi-gauges, calibrate VIC param (distributed), soil depths (uniform), RVIC params (fixed)")
        case_name = "case3"
        params_case = deepcopy(params_minimal)
        params_case["g_params"] = set_g_params_soilGrids_layer(params_case["g_params"])
        params_case["rvic_params"]["VELOCITY"]["free"] = False
        params_case["rvic_params"]["DIFFUSION"]["free"] = False
        params_case["guh_params"]["tp"]["free"] = False
        params_case["guh_params"]["mu"]["free"] = False
        params_case["guh_params"]["m"]["free"] = False
        rvic_spatial_case = False
        buildParam_level0_interface_class = buildParam_level0_interface_HRB
        evb_dir_modeling.calibrate_cp_path = evb_dir_modeling.calibrate_cp_path.replace(".pkl", f"_case3.pkl")
        Baseflow_scheme = "NIJSSEN2001"
        cali_method = "NSGAII"
        basin_hierarchy = None

    elif case_name == "case4":
        # case4: multi-gauges, calibrate VIC param (distributed), soil depths (uniform), RVIC params (distributed)
        logger.info("case4: multi-gauges, calibrate VIC param (distributed), soil depths (uniform), RVIC params (distributed)")
        case_name = "case4"
        params_case = deepcopy(params_minimal)
        params_case["rvic_params"] = rvic_params_spatial
        params_case["g_params"] = set_g_params_soilGrids_layer(params_case["g_params"])
        rvic_spatial_case = True
        buildParam_level0_interface_class = buildParam_level0_interface_HRB
        evb_dir_modeling.calibrate_cp_path = evb_dir_modeling.calibrate_cp_path.replace(".pkl", f"_case4.pkl")
        Baseflow_scheme = "NIJSSEN2001"
        cali_method = "NSGAII"
        basin_hierarchy = None

    elif case_name == "case5":
        # case5: multi-gauges, calibrate VIC param (distributed), soil depths (distributed), RVIC params (distributed)
        logger.info("case5: multi-gauges, calibrate VIC param (distributed), soil depths (distributed), RVIC params (distributed)")
        case_name = "case5"
        params_case = deepcopy(params_minimal)
        params_case["g_params"] = set_g_params_soilGrids_layer(params_case["g_params"])
        params_case["g_params"] = expand_station_wise_params(params_case["g_params"], station_num=len(station_names))
        params_case["rvic_params"] = rvic_params_spatial
        rvic_spatial_case = True
        buildParam_level0_interface_class = buildParam_level0_interface_HRB
        evb_dir_modeling.calibrate_cp_path = evb_dir_modeling.calibrate_cp_path.replace(".pkl", f"_case5.pkl")
        basin_hierarchy = build_basin_hierarchy(evb_dir_hydroanalysis, evb_dir_modeling)
        Baseflow_scheme = "NIJSSEN2001"
        cali_method = "NSGAII"

    elif case_name == "case6":
        # case6: single-gauge, calibrate VIC param (uniform), soil depths (uniform), RVIC params (fixed)
        logger.info("case6: single-gauge, calibrate VIC param (uniform), soil depths (uniform), RVIC params (fixed)")
        case_name = "case6"
        params_case = deepcopy(params_minimal)
        params_case["g_params"] = g_params_Nijssen_spatially_uniform_minimal
        params_case["g_params"] = set_g_params_soilGrids_layer(params_case["g_params"])
        params_case["rvic_params"]["VELOCITY"]["free"] = False
        params_case["rvic_params"]["DIFFUSION"]["free"] = False
        params_case["guh_params"]["tp"]["free"] = False
        params_case["guh_params"]["mu"]["free"] = False
        params_case["guh_params"]["m"]["free"] = False
        rvic_spatial_case = False
        evb_dir_modeling.calibrate_cp_path = evb_dir_modeling.calibrate_cp_path.replace(".pkl", f"_case6.pkl")
        basin_hierarchy = None
        buildParam_level0_interface_class = buildParam_level0_interface_Nijssen_spatially_uniform_HRB
        Baseflow_scheme = "NIJSSEN2001"
        cali_method = "CMA_ES"

    elif case_name == "case7":
        # case7: single-gauge, calibrate VIC param (uniform), soil depths (uniform), RVIC params (uniform)
        logger.info("case7: single-gauge, calibrate VIC param (uniform), soil depths (uniform), RVIC params (uniform)")
        case_name = "case7"
        params_case = deepcopy(params_minimal)
        params_case["g_params"] = g_params_Nijssen_spatially_uniform_minimal
        params_case["g_params"] = set_g_params_soilGrids_layer(params_case["g_params"])
        rvic_spatial_case = False
        evb_dir_modeling.calibrate_cp_path = evb_dir_modeling.calibrate_cp_path.replace(".pkl", f"_case7.pkl")
        basin_hierarchy = None
        buildParam_level0_interface_class = buildParam_level0_interface_Nijssen_spatially_uniform_HRB
        Baseflow_scheme = "NIJSSEN2001"
        cali_method = "CMA_ES"

    elif case_name == "case8":
        # case8: single-gauge, calibrate VIC param (distributed), soil depths (distributed), RVIC params (distributed)
        logger.info("case8: single-gauge, calibrate VIC param (distributed), soil depths (distributed), RVIC params (distributed)")
        case_name = "case8"
        params_case = deepcopy(params_minimal)
        params_case["g_params"] = set_g_params_soilGrids_layer(params_case["g_params"])
        params_case["g_params"] = expand_station_wise_params(params_case["g_params"], station_num=len(station_names))
        params_case["rvic_params"] = rvic_params_spatial
        rvic_spatial_case = True
        evb_dir_modeling.calibrate_cp_path = evb_dir_modeling.calibrate_cp_path.replace(".pkl", f"_case8.pkl")
        basin_hierarchy = build_basin_hierarchy(evb_dir_hydroanalysis, evb_dir_modeling)
        Baseflow_scheme = "NIJSSEN2001"
        buildParam_level0_interface_class = buildParam_level0_interface_HRB
        cali_method = "CMA_ES"
    
    return params_case, rvic_spatial_case, basin_hierarchy, buildParam_level0_interface_class, Baseflow_scheme, cali_method

def calibrate_HRB(evb_dir_modeling):
    # *if run with RVIC, you should modify Makefile and turn the rout_rvic, compile it
    # evb_dir_modeling.vic_exe_path = "/home/xudong/VIC/vic_image.exe"
    evb_dir_modeling.vic_exe_path = "/home/xudong/VIC/vic_image_stub_rout.exe"
    
    # nsgaII set
    algParams_NSGAII = {"popSize": 20, "maxGen": 30, "cxProb": 0.9, "mutateProb": 0.15}
    
    # CMA_ES set
    algParams_CMA_ES = {"popSize":20, "maxGen": 30, "sigma": 0.5}
    
    case_name = "case5"
    params_case, rvic_spatial_case, basin_hierarchy, buildParam_level0_interface_class, Baseflow_scheme, cali_method = case_set(case_name)

    # load calibration set
    param_temp = params_case
    rvic_spatial = rvic_spatial_case
    basin_hierarchy = basin_hierarchy  # basin_hierarchy
    Baseflow_scheme = Baseflow_scheme  # "NIJSSEN2001"  #  Baseflow_scheme
    params_cali = ParamManager(param_temp)
    logger.info(f"calibrate param: {params_cali.vector_names(get_free=True)}")
    cali_method = cali_method # "NSGAII" # CMA_ES
    CONSTRAIN_FRACTIONS = False  # nested basin
    
    # GlobalParam_dict
    GlobalParam_dict = {
            "Simulation": {
                "MODEL_STEPS_PER_DAY": "1",
                "SNOW_STEPS_PER_DAY": "8",
                "RUNOFF_STEPS_PER_DAY": "8",
                "STARTYEAR": str(warmup_date_period[0][:4]),
                "STARTMONTH": str(int(warmup_date_period[0][4:6])),
                "STARTDAY": str(int(warmup_date_period[0][6:8])),
                "ENDYEAR": str(calibrate_date_period[1][:4]),  # calibrate_date_period
                "ENDMONTH": str(int(calibrate_date_period[1][4:6])),  # calibrate_date_period
                "ENDDAY": str(int(calibrate_date_period[1][6:8])),  # calibrate_date_period
                "OUT_TIME_UNITS": "DAYS",
            },
            # "Precipitation": {"CORRPREC": "TRUE"},  # correct PREC
            "Output": {"AGGFREQ": "NDAYS   1"},
            "OUTVAR1": {"OUTVAR": ["OUT_RUNOFF", "OUT_BASEFLOW"]},  # "OUT_BASEFLOW", "OUT_RUNOFF", "OUT_SOIL_MOIST" , "OUT_EVAP", "OUT_RUNOFF", "OUT_DISCHARGE"
            "Param": {"BASEFLOW": Baseflow_scheme},  # NIJSSEN2001, ARNO
        }
    
    # NSGAII
    if cali_method == "NSGAII":
        nsgaII_VIC_HRB = NSGAII_VIC_HRB(
            evb_dir_modeling,
            warmup_date_period,
            calibrate_date_period,
            verify_date_period,
            timestep,
            timestep_evaluate,
            GlobalParam_dict,
            param_temp=param_temp,
            buildParam_level0_interface_class=buildParam_level0_interface_class,
            buildParam_level1_interface_class=buildParam_level1_interface_HRB,
            soillayerresampler=SoilGrids_soillayerresampler,
            TF_VIC_class=TF_VIC,
            nlayer_list=[1, 2, 3],
            rvic_OUTPUT_INTERVAL=86400,
            rvic_BASIN_FLOWDAYS=50,
            rvic_SUBSET_DAYS=20,
            rvic_CELL_FLOWDAYS=1,
            rvic_uhbox_dt=60,
            algParams=algParams_NSGAII,
            reverse_lat=True,
            parallel_vic=None,
            parallel_rvic_param=None,
            rvic_spatial=rvic_spatial,
            station_names=station_names,
            station_names_cali=station_names,
            elev_scaling="Arithmetic_min",
            basin_hierarchy=basin_hierarchy,
            CONSTRAIN_FRACTIONS=CONSTRAIN_FRACTIONS,
        )

        cali_algorithm = nsgaII_VIC_HRB

    # CMA_ES
    if cali_method == "CMA_ES":
        cmaes_VIC_HRB = CMA_ES_VIC_HRB(
            evb_dir_modeling,
            warmup_date_period,
            calibrate_date_period,
            verify_date_period,
            timestep,
            timestep_evaluate,
            GlobalParam_dict,
            param_temp=param_temp,
            buildParam_level0_interface_class=buildParam_level0_interface_class,
            buildParam_level1_interface_class=buildParam_level1_interface_HRB,
            soillayerresampler=SoilGrids_soillayerresampler,
            TF_VIC_class=TF_VIC,
            nlayer_list=[1, 2, 3],
            rvic_OUTPUT_INTERVAL=86400,
            rvic_BASIN_FLOWDAYS=50,
            rvic_SUBSET_DAYS=20,
            rvic_CELL_FLOWDAYS=1,
            rvic_uhbox_dt=60,
            algParams=algParams_CMA_ES,
            reverse_lat=True,
            parallel_vic=None,
            parallel_rvic_param=None,
            rvic_spatial=rvic_spatial,
            station_names=station_names,
            station_names_cali=["shiquan"],
            elev_scaling="Arithmetic_min",
            basin_hierarchy=basin_hierarchy,
            CONSTRAIN_FRACTIONS=CONSTRAIN_FRACTIONS,
        )

        cali_algorithm = cmaes_VIC_HRB

    # calibrate
    calibrate_NSGAII_bool = False
    if calibrate_NSGAII_bool:
        nsgaII_VIC_HRB.run(
            plot_progress=True,
            plot_dir=evb_dir_modeling.CalibrateVIC_dir,
            names_plot=station_names,
        )
    
    calibrate_CMA_ES_bool = False
    if calibrate_CMA_ES_bool:
        cmaes_VIC_HRB.run(
            plot_progress=True,
            plot_dir=evb_dir_modeling.CalibrateVIC_dir,
            plot_ylim=(-0.5, 1.0),
        )

    # simulation
    simulate_bool = True
    if simulate_bool:
        # mkdir
        case_dir = os.path.join(evb_dir_modeling.CalibrateVIC_dir, case_name)
        check_and_mkdir(case_dir)

        # clear
        remove_files(evb_dir_modeling.VICResults_dir)
        remove_and_mkdir(evb_dir_modeling.VICLog_dir)

        # build GlobalParam: total period (modify end date)
        GlobalParam_dict_all_period = deepcopy(GlobalParam_dict)
        GlobalParam_dict_all_period["Simulation"].update(
            {
                "ENDYEAR": str(verify_date_period[1][:4]),
                "ENDMONTH": str(int(verify_date_period[1][4:6])),
                "ENDDAY": str(int(verify_date_period[1][6:8])),
            }
        )

        GlobalParam_dict_all_period["OUTVAR1"].update(
            {
                "OUTVAR": ["OUT_RUNOFF", "OUT_BASEFLOW", "OUT_SOIL_MOIST", "OUT_EVAP"]
            }   
        )
        cali_algorithm.set_GlobalParam_dict(GlobalParam_dict_all_period)

        # sim
        ind_num = 5
        max_num = 30

        if cali_method == "NSGAII":
            # get ind
            first_num_data = get_inds_NSGAII(evb_dir_modeling, ind_num=ind_num, max_num=max_num)
            
            for i, ind in enumerate(first_num_data["first_num_best_ind"]):
                # sim cali
                fitness_cali = cali_algorithm.evaluate(ind, get_type="calibration", get_obs_bool=True)
                copy_result_to_case_dir(evb_dir_modeling, case_dir, save_type="cali", suffix=i)

                # sim vali
                fitness_vali = cali_algorithm.evaluate(ind, get_type="validation", get_obs_bool=True)
                copy_result_to_case_dir(evb_dir_modeling, case_dir, save_type="vali", suffix=i)

        elif cali_method == "CMA_ES":
            # get ind
            first_num_data = get_inds_CMA_ES(evb_dir_modeling, ind_num=ind_num, max_num=max_num)

            # calibrated_params_case_all = {}
            for i, ind in enumerate(first_num_data["first_num_best_ind"]):
                # params
                # calibrated_params_case = params_case_pm.to_dict(ind, field="optimal", get_free=True)
                # calibrated_params_case_all[i] = calibrated_params_case

                # sim cali
                fitness_cali = cali_algorithm.evaluate(ind, get_type="calibration", get_obs_bool=True)
                copy_result_to_case_dir(evb_dir_modeling, case_dir, save_type="cali", suffix=i)

                # sim vali
                fitness_vali = cali_algorithm.evaluate(ind, get_type="validation", get_obs_bool=True)
                copy_result_to_case_dir(evb_dir_modeling, case_dir, save_type="vali", suffix=i)


if __name__ == "__main__":
    calibrate_HRB(evb_dir_modeling)
    
    
