# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from .tools.calibrate_func.algorithm_NSGAII import NSGAII_Base
from .tools.calibrate_func.sampling import *
from .tools.params_func.params_set import *
from .tools.calibrate_func.evaluate_metrics import EvaluationMetric
import os
from deap import creator, base, tools, algorithms
from copy import deepcopy
from .bulid_Param import buildParam_level0, buildParam_level1, scaling_level0_to_level1, buildParam_level0_by_g
from .build_RVIC_Param import buildUHBOXFile, buildParamCFGFile, buildConvCFGFile
from .build_GlobalParam import buildGlobalParam
from netCDF4 import Dataset, num2date
import pandas as pd
from .tools.geo_func.search_grids import search_grids_nearest
from .tools.utilities import *
from copy import deepcopy
from datetime import datetime
from .tools.decoractors import clock_decorator
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from rvic.parameters import parameters
from rvic.convolution import convolution
import math
# plt.show(block=True)


class NSGAII_VIC_SO(NSGAII_Base):
    
    def __init__(self, evb_dir, dpc_VIC_level0, dpc_VIC_level1, date_period, calibrate_date_period,
                 rvic_OUTPUT_INTERVAL=3600, rvic_BASIN_FLOWDAYS=50, rvic_SUBSET_DAYS=10, rvic_uhbox_dt=60,
                 algParams={"popSize": 40, "maxGen": 250, "cxProb": 0.7, "mutateProb": 0.2},
                 save_path="checkpoint.pkl", reverse_lat=True, parallel=False):
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
        
        # period
        self.date_period = date_period
        self.calibrate_date_period = calibrate_date_period
        
        # clear Param
        clearParam(self.evb_dir)
        
        # params boundary
        self.g_boundary = g_boundary
        self.uh_params_boundary = uh_params_boundary
        self.routing_params_boundary = routing_params_boundary
        self.depths_indexes = depths_index
        self.total_boundary = [item for lst in [g_boundary, uh_params_boundary, routing_params_boundary] for item in lst]
        self.low = [b[0] for b in self.total_boundary]
        self.up = [b[1] for b in self.total_boundary]
        
        # params dimension
        self.NDim = len(self.total_boundary)
        
        # params type
        self.g_types = g_types
        self.uh_params_types = uh_params_types
        self.routing_params_types = routing_params_types
        self.total_types = [item for lst in [g_types, uh_params_types, uh_params_types] for item in lst]

        # get obs
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
    
    def get_obs(self):
        self.obs = self.dpc_VIC_level1.basin_shp.streamflow.iloc[0]
        date = self.obs.loc[:, "date"]
        factor_unit_feet2meter = 0.0283168
        self.obs.loc[:, "discharge(m3/s)"] = self.obs.loc[:, 4] * factor_unit_feet2meter
        self.obs.index = pd.to_datetime(date)
    
    def get_sim(self):
        # sim_df
        sim_df = pd.DataFrame(columns=["time", "discharge(m3/s)"])
        
        # outlet lat, lon
        pourpoint_file = pd.read_csv(self.evb_dir.pourpoint_file_path)
        x, y = pourpoint_file.lons[0], pourpoint_file.lats[0]
        # x, y = dpc_VIC_level1.basin_shp.loc[:, "camels_topo:gauge_lon"].values[0], dpc_VIC_level1.basin_shp.loc[:, "camels_topo:gauge_lat"].values[0]

        if self.parallel: # TODO
            pass
            
        else:
            # read VIC OUTPUT file
            with Dataset(self.sim_path, "r") as dataset:
                # get time, lat, lon
                time = dataset.variables["time"]
                lat = dataset.variables["lat"][:]
                lon = dataset.variables["lon"][:]
                
                # transfer time_num into date
                time_date = num2date(time[:], units=time.units, calendar=time.calendar)
                time_date = [datetime(t.year, t.month, t.day, t.hour, t.second) for t in time_date]
                
                # get outlet index
                if self.get_sim_searched_grids_index is None:
                    searched_grids_index = search_grids_nearest([y], [x], lat, lon, search_num=1)
                    searched_grid_index = searched_grids_index[0]
                    self.get_sim_searched_grids_index = searched_grid_index
                
                # read data
                out_discharge = dataset.variables["OUT_DISCHARGE"][:, self.get_sim_searched_grids_index[0][0], self.get_sim_searched_grids_index[1][0]]
                
                sim_df.loc[:, "time"] = time_date
                sim_df.loc[:, "discharge(m3/s)"] = out_discharge
                sim_df.index = pd.to_datetime(time_date)
        
        # aggregate
        sim_df = sim_df.resample("D").mean()
        
        return sim_df
    
    def createFitness(self):
        creator.create("Fitness", base.Fitness, weights=(1.0, ))

    def samplingInd(self):
        # n_samples
        n_samples = 1
        
        ## ----------------------- params g bounds -----------------------
        # copy
        params_g_bounds = deepcopy(self.g_boundary)
        
        # sampling for depths g
        depths_g_bounds = [params_g_bounds[di] for di in self.depths_indexes]  # such as [(1, 5), (3, 8), (6, 11)], this is num, start from 1 (1~11)
        for di in sorted(self.depths_indexes, reverse=True):
            params_g_bounds.pop(di)
        
        ## ----------------------- RVIC params bounds -----------------------
        # uh_params={"tp": 1.4, "mu": 5.0, "m": 3.0}
        uh_params_bounds = self.uh_params_boundary
        
        # cfg_params={"VELOCITY": 1.5, "DIFFUSION": 800.0}
        routing_params_bounds = self.routing_params_boundary
        
        ## ----------------------- mixsampling -----------------------
        # discrete sampling
        depth_num_samples = sampling_CONUS_depth_num(n_samples, layer_ranges=depths_g_bounds)
        num1, num2 = depth_num_samples[0]
        insertions = list(zip(self.depths_indexes, [num1, num2]))
        
        # continuous sampling
        params_g_bounds.extend(uh_params_bounds)
        params_g_bounds.extend(routing_params_bounds)
        
        params_samples = sampling_Sobol(n_samples, len(params_g_bounds), params_g_bounds)[0]
        params_samples = params_samples.tolist()
        
        # combine samples
        for index, value in sorted(insertions, key=lambda x: x[0]):
            params_samples.insert(index, value)
        
        return creator.Individual(params_samples)

    @clock_decorator(print_arg_ret=True)
    def run_vic(self):
        if self.parallel:
            command_run_vic = " ".join([f"mpiexec -np {self.parallel}", self.evb_dir.vic_exe_path, "-g", self.evb_dir.globalParam_path])
        else:
            command_run_vic = " ".join([self.evb_dir.vic_exe_path, "-g", self.evb_dir.globalParam_path])
            
        print("============= running VIC =============")
        out = os.system(command_run_vic)
        return out
    
    @clock_decorator(print_arg_ret=True)
    def run_rvic(self, conv_cfg_file_dict):
        print("============= running RVIC convolution =============")
        convolution(conv_cfg_file_dict)
        # TODO combine RVIC output (if multiple)
        pass
    
    def adjust_vic_params_level0(self, params_g):
        if os.path.exists(self.evb_dir.params_dataset_level0_path):
            # read and adjust by g
            params_dataset_level0 = Dataset(self.evb_dir.params_dataset_level0_path, "a", format="NETCDF4")
            params_dataset_level0, stand_grids_lat, stand_grids_lon, rows_index, cols_index = buildParam_level0_by_g(params_dataset_level0, params_g, self.dpc_VIC_level0, self.reverse_lat,
                                                                                                                     self.stand_grids_lat_level0, self.stand_grids_lon_level0,
                                                                                                                     self.rows_index_level0, self.cols_index_level0)
        else:
            # build
            params_dataset_level0, stand_grids_lat, stand_grids_lon, rows_index, cols_index = buildParam_level0(self.evb_dir, params_g, self.dpc_VIC_level0, self.reverse_lat,
                                                                                                                self.stand_grids_lat_level0, self.stand_grids_lon_level0,
                                                                                                                self.rows_index_level0, self.cols_index_level0)
        
        self.stand_grids_lat_level0, self.stand_grids_lon_level0, self.rows_index_level0, self.cols_index_level0 = stand_grids_lat, stand_grids_lon, rows_index, cols_index
        
        return params_dataset_level0

    def adjust_vic_params_level1(self, params_dataset_level0):
        if os.path.exists(self.evb_dir.params_dataset_level1_path):
            # read
            params_dataset_level1 = Dataset(self.evb_dir.params_dataset_level1_path, "a", format="NETCDF4")
        else:
            # build
            domain_dataset = readDomain(self.evb_dir)
            params_dataset_level1, stand_grids_lat, stand_grids_lon, rows_index, cols_index = buildParam_level1(self.evb_dir, self.dpc_VIC_level1, self.reverse_lat, domain_dataset,
                                                                                                                self.stand_grids_lat_level1, self.stand_grids_lon_level1,
                                                                                                                self.rows_index_level1, self.cols_index_level1)
            domain_dataset.close()
            self.stand_grids_lat_level1, self.stand_grids_lon_level1, self.rows_index_level1, self.cols_index_level1 = stand_grids_lat, stand_grids_lon, rows_index, cols_index
        
        # scaling
        params_dataset_level1, searched_grids_bool_index = scaling_level0_to_level1(params_dataset_level0, params_dataset_level1, self.scaling_searched_grids_bool_index)
        self.scaling_searched_grids_bool_index = searched_grids_bool_index
        
        return params_dataset_level1
        
    def cal_constraint_destroy(self, params_dataset_level0):
        # wp < fc
        # Wpwp_FRACT < Wcr_FRACT
        # depth_layer0 < depth_layer1
        # no nan in infilt
        # TODO check variables
        constraint_wp_fc_destroy = np.max(np.array(params_dataset_level0.variables["wp"][:, :, :] > params_dataset_level0.variables["fc"][:, :, :]))
        constraint_Wpwp_Wcr_FRACT_destroy = np.max(np.array(params_dataset_level0.variables["Wpwp_FRACT"][:, :, :] > params_dataset_level0.variables["Wcr_FRACT"][:, :, :]))
        constraint_depth_destroy = np.max(np.array(params_dataset_level0.variables["depth"][0, :, :] > params_dataset_level0.variables["depth"][1, :, :]))
        # constraint_infilt_nan_destroy = np.sum(np.isnan(np.array(params_dataset_level0.variables["infilt"][:, :]))) > 0
        
        constraint_destroy = any([constraint_wp_fc_destroy, constraint_Wpwp_Wcr_FRACT_destroy, constraint_depth_destroy])
        return constraint_destroy
    
    def adjust_rvic_params(self, uh_params, routing_params):
        # domain, FlowDirectionFile, PourPointFile should be already created
        
        # adjust UHBOXFile
        uh_params_input = {"uh_dt": self.rvic_uhbox_dt, "tp": uh_params[0], "mu": uh_params[1], "m": uh_params[2], "max_day_range": (0, 10), "max_day_converged_threshold": 0.001}
        uhbox_max_day = buildUHBOXFile(self.evb_dir, **uh_params_input, plot_bool=True)
        
        # adjust ParamCFGFile
        rvic_param_cfg_params = {"VELOCITY": routing_params[0], "DIFFUSION": routing_params[1],
                                 "OUTPUT_INTERVAL": self.rvic_OUTPUT_INTERVAL,
                                 "SUBSET_DAYS": self.rvic_SUBSET_DAYS,
                                 "CELL_FLOWDAYS": uhbox_max_day,
                                 "BASIN_FLOWDAYS": self.rvic_BASIN_FLOWDAYS}
        buildParamCFGFile(self.evb_dir, **rvic_param_cfg_params)
        
        # remove files
        remove_and_mkdir(os.path.join(self.evb_dir.RVICParam_dir, "params"))
        remove_and_mkdir(os.path.join(self.evb_dir.RVICParam_dir, "plots"))
        remove_and_mkdir(os.path.join(self.evb_dir.RVICParam_dir, "logs"))
        inputs_fpath = [os.path.join(self.evb_dir.RVICParam_dir, inputs_f) for inputs_f in os.listdir(self.evb_dir.RVICParam_dir) if inputs_f.startswith("inputs") and inputs_f.endswith("tar")]
        if len(inputs_fpath) > 0:
            [os.remove(fp) for fp in inputs_fpath]

        # build rvic_params
        param_cfg_file_dict = read_cfg_to_dict(self.evb_dir.rvic_param_cfg_file_path)
        parameters(param_cfg_file_dict, numofproc=1)
        
        # modify rout_param_path in GlobalParam
        globalParam = GlobalParamParser()
        globalParam.load(self.evb_dir.globalParam_path)
        self.rout_param_path = os.path.join(self.evb_dir.rout_param_dir, os.listdir(self.evb_dir.rout_param_dir)[0])
        globalParam.set("Routing", "ROUT_PARAM", self.rout_param_path)
        
        # write
        with open(self.evb_dir.globalParam_path, "w") as f:
            globalParam.write(f)
    
    def adjust_rvic_conv_params(self):
        # TODO DATL_LIQ_FLDS, OUT_RUNOFF, OUT_BASEFLOW might be run individually
        # build rvic_conv_cfg_params
        RUN_STARTDATE = f"{self.date_period[0][:4]}-{self.date_period[0][4:6]}-{self.date_period[0][6:]}-00"
        rvic_conv_cfg_params = {"RUN_STARTDATE": RUN_STARTDATE, "DATL_FILE": self.sim_fn, "PARAM_FILE_PATH": self.rout_param_path}
        buildConvCFGFile(self.evb_dir, **rvic_conv_cfg_params)
        
        conv_cfg_file_dict = read_cfg_to_dict(self.evb_dir.rvic_conv_cfg_file_path)
        return conv_cfg_file_dict
        
    def evaluate(self, ind):
        # =============== get ind ===============
        params_g = ind[:-5]
        uh_params = ind[-5:-2]
        routing_params = ind[-2:]
        
        # type params
        params_g = [self.g_types[i](params_g[i]) for i in range(len(params_g))]
        uh_params = [self.uh_params_types[i](uh_params[i]) for i in range(len(uh_params))]
        routing_params = [self.routing_params_types[i](routing_params[i]) for i in range(len(routing_params))]
        
        # =============== adjust vic params based on ind ===============
        # adjust params_dataset_level0 based on params_g
        print("============= building params_level0 =============")
        params_dataset_level0 = self.adjust_vic_params_level0(params_g)

        # constraint to make sure the params are all valid
        print("============= cal_constraint_destroy =============")
        constraint_destroy = self.cal_constraint_destroy(params_dataset_level0)
        print(f"constraint_destroy: {constraint_destroy}, true means invalid params, set fitness = -9999.0")
        
        # if constraint_destroy, the params is not valid, vic will report mistake, we dont run it and return -9999.0
        if constraint_destroy:
            fitness = -9999.0
        else:
            # adjust params_dataset_level1 based on params_dataset_level0
            print("============= building params_level1 =============")
            params_dataset_level1 = self.adjust_vic_params_level1(params_dataset_level0)
            
            # close
            params_dataset_level0.close()
            params_dataset_level1.close()
            
            # =============== adjust rvic params based on ind ===============
            print("============= building rvic params =============")
            self.adjust_rvic_params(uh_params, routing_params)
            
            # =============== run vic ===============
            # clear VICResults_dir and VICLog_dir
            remove_files(self.evb_dir.VICResults_dir)
            remove_and_mkdir(self.evb_dir.VICLog_dir)
            
            # run
            out_vic = self.run_vic()
            self.sim_fn = [fn for fn in os.listdir(self.evb_dir.VICResults_dir) if fn.endswith(".nc")][0]
            self.sim_path = os.path.join(self.evb_dir.VICResults_dir, self.sim_fn)
            
            # =============== run rvic offline ===============
            if self.parallel:
                # clear RVICConv_dir
                remove_and_mkdir(os.path.join(self.evb_dir.RVICConv_dir))
                
                # build cfg file
                conv_cfg_file_dict = self.adjust_rvic_conv_params()
                
                # run
                out_rvic = self.run_rvic(conv_cfg_file_dict)
            
            # =============== evaluate ===============
            print("============= evaluating =============")
            # get obs: alreay got
            # get sim
            sim = self.get_sim()
            
            # clip sim and obs during the calibrate_date_period
            sim_cali = sim.loc[self.calibrate_date_period[0]: self.calibrate_date_period[1], "discharge(m3/s)"]
            obs_cali = self.obs.loc[self.calibrate_date_period[0]: self.calibrate_date_period[1], "discharge(m3/s)"]
            
            # evaluate
            evaluation_metric = EvaluationMetric(sim_cali, obs_cali)
            fitness = evaluation_metric.KGE()
            # fitness = evaluation_metric.KGE_m()

            # plot discharge
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(sim_cali, "r-", label=f"sim({round(fitness, 2)})", linewidth=0.5)
            ax.plot(obs_cali, "k-", label="obs", linewidth=1)
            ax.set_xlabel("date")
            ax.set_ylabel("discharge m3/s")
            ax.legend()
            # plt.show(block=True)
            fig.savefig(os.path.join(self.evb_dir.VICResults_fig_dir, "evaluate_discharge.tiff"))
        
        print("fitness:", fitness)
        fitness = -9999.0 if np.isnan(fitness) else fitness
        
        return (fitness, )
    
    def get_best_results(self):
        # get front
        front = self.history[-1][1][0][0]
        
        # get fitness
        print(f"current fitness: {front.fitness.values}")
        
        # sim
        fitness_cal_again = self.evaluate(front)
    
    @staticmethod
    def operatorMate(parent1, parent2, low, up):
        return tools.cxSimulatedBinaryBounded(parent1, parent2, eta=20.0, low=low, up=up)
    
    @staticmethod
    def operatorMutate(ind, low, up, NDim):
        return tools.mutPolynomialBounded(ind, eta=20.0, low=low, up=up, indpb=1/NDim)
    
    @staticmethod
    def operatorSelect(population, popSize):
        return tools.selNSGA2(population, popSize)
    
    def apply_genetic_operators(self, offspring):
        # it can be implemented by algorithms.varAnd
        # crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= self.toolbox.cxProb:
                self.toolbox.mate(child1, child2, self.low, self.up)
                del child1.fitness.values
                del child2.fitness.values
        
        # mutate
        for mutant in offspring:
            if random.random() <= self.toolbox.mutateProb:
                self.toolbox.mutate(mutant, self.low, self.up, self.NDim)
                del mutant.fitness.values
                
class NSGAII_VIC_MO(NSGAII_VIC_SO):
    
    def createFitness(self):
        creator.create("Fitness", base.Fitness, weights=(-1.0, ))

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


def calibrate_VIC_SO(evb_dir):
    # calibrate
    algParams = algParams={"popSize": 40, "maxGen": 250, "cxProb": 0.7, "mutateProb": 0.2}
    save_path = os.path.join(evb_dir.CalibrateVIC_dir, "checkpoint.pkl")
    nsgaII_VIC = NSGAII_VIC_SO(algParams, save_path)
    pass
