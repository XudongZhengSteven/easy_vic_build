# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from .tools.calibrate_func.algorithm_NSGAII import NSGAII_Base
from .tools.calibrate_func.sampling import *
from .tools.calibrate_func.params_set import *
from .tools.calibrate_func.evaluate_metrics import EvaluationMetric
import os
from deap import creator, base, tools, algorithms
from copy import deepcopy
from .bulid_Param import buildParam_level0, scaling_level0_to_level1
from .build_RVIC_Param import buildUHBOXFile, buildParamCFGFile
from netCDF4 import Dataset, num2date
import pandas as pd
from .tools.geo_func.search_grids import search_grids_nearest


class NSGAII_VIC_SO(NSGAII_Base):
    
    def __init__(self, dpc_VIC_level0, dpc_VIC_level1, evb_dir, date_period, calibrate_date_period,
                 algParams={"popSize": 40, "maxGen": 250, "cxProb": 0.7, "mutateProb": 0.2},
                 save_path="checkpoint.pkl"):
        super().__init__(algParams, save_path)
        self.evb_dir = evb_dir
        self.dpc_VIC_level0 = dpc_VIC_level0
        self.dpc_VIC_level1 = dpc_VIC_level1
        
        # period
        self.date_period = date_period
        self.calibrate_date_period = calibrate_date_period
        
        # params boundary
        self.g_boundary = g_boundary
        self.uh_params_boundary = uh_params_boundary
        self.routing_params_boundary = routing_params_boundary
        
        # get obs
        self.get_obs()
    
    def get_obs(self):
        self.obs = self.dpc_VIC_level1.basin_shp.streamflow.iloc[0]
        date = self.obs.loc[:, "date"]
        self.obs.index = pd.to_datetime(date)
    
    def get_sim(self):
        # sim_df
        sim_df = pd.Dataframe(columns=["time", "OUT_DISCHARGE"])
        
        # outlet lat, lon
        x, y = self.dpc_VIC_level1.basin_shp.loc[:, "camels_topo:gauge_lon"].values[0], self.dpc_VIC_level1.basin_shp.loc[:, "camels_topo:gauge_lat"].values[0]      
        
        # read VIC OUTPUT file
        sim_path = os.path.join(self.VICResults_dir, os.listdir(self.evb_dir.VICResults_dir)[0])
        with Dataset(sim_path, "r") as dataset:
            # get time, lat, lon
            time = dataset.variables["time"]
            lat = dataset.variables["lat"][:]
            lon = dataset.variables["lon"][:]
            
            # transfer time_num into date
            time_date = num2date(time[:], units=time.units, calendar=time.calendar)
            
            # get outlet index
            searched_grids_index = search_grids_nearest([y], [x], lat, lon, search_num=1)
            searched_grid_index = searched_grids_index[0]
            
            # read data
            out_discharge = dataset.variables["OUT_DISCHARGE"][:, searched_grid_index[0][0], searched_grid_index[1][0]]
            
            sim_df.loc[:, "time"] = time_date
            sim_df.loc[:, "OUT_DISCHARGE"] = out_discharge
            sim_df.index = pd.to_datetime(time_date)
        
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
        depths_indexes = [1, 2]
        depths_g_bounds = [params_g_bounds.pop(di) for di in depths_indexes]  # [(0, 3), (3, 8), (8, 11)]
        
        ## ----------------------- RVIC params bounds -----------------------
        # uh_params={"tp": 1.4, "mu": 5.0, "m": 3.0}
        uh_params_bounds = self.uh_params_boundary
        
        # cfg_params={"VELOCITY": 1.5, "DIFFUSION": 800.0}
        routing_params_bounds = self.routing_params_boundary
        
        ## ----------------------- mixsampling -----------------------
        # discrete sampling
        depths_g_samples = sampling_COUNU_Soil(n_samples, layer_ranges=depths_g_bounds)
        
        # continuous sampling
        params_g_bounds.extend(uh_params_bounds)
        params_g_bounds.extend(routing_params_bounds)
        
        params_samples = sampling_Sobol(n_samples, len(params_g_bounds), params_g_bounds)
        
        # combine samples
        params_samples.insert(depths_indexes[0], depths_g_samples[0]["Layer1 percentile"])
        params_samples.insert(depths_indexes[1], depths_g_samples[0]["Layer2 percentile"])
        
        return creator.Individual(params_samples)

    def evaluate(self, ind):
        # =============== get ind ===============
        params_g = ind[:-5]
        uh_params = ind[-5:-2]
        routing_params = ind[-2:]
        
        # type params: not need now      
        
        # =============== adjust vic params based on ind ===============
        # adjust params_dataset_level0
        params_dataset_level0 = buildParam_level0(params_g, self.dpc_VIC_level0, self.evb_dir, reverse_lat=True)
        
        # read params_dataset_level1
        if not hasattr(self, "params_dataset_level1"):
            self.params_dataset_level1 = Dataset(self.evb_dir.params_dataset_level1_path, "a", format="NETCDF4")
        
        # scaling
        self.params_dataset_level1, searched_grids_index = scaling_level0_to_level1(params_dataset_level0, self.params_dataset_level1)
        
        # =============== adjust rvic params based on ind ===============       
        # adjust UHBOXFile
        uh_params = {"tp": uh_params[0], "mu": uh_params[1], "m": uh_params[2]}
        buildUHBOXFile(self.evb_dir, **uh_params, plot_bool=False)
        
        # adjust ParamCFGFile
        cfg_params = {"VELOCITY": routing_params[0], "DIFFUSION": routing_params[1], "OUTPUT_INTERVAL": 86400}
        buildParamCFGFile(self.evb_dir, **cfg_params)
        
        # =============== run vic ===============
        command_run_vic = " ".join([self.evb_dir.vic_exe_path, "-g", self.evb_dir.globalParam_path])
        os.system(command_run_vic)
        
        # =============== evaluate ===============
        # get obs: alreay got
        # get sim
        sim = self.get_sim()
        
        # clip sim and obs during the calibrate_date_period
        sim_cali = sim.loc[self.calibrate_date_period[0], self.calibrate_date_period[1]]
        obs_cali = self.obs.loc[self.calibrate_date_period[0], self.calibrate_date_period[1]]
        
        # evaluate
        evaluation_metric = EvaluationMetric(sim_cali, obs_cali)
        fitness = evaluation_metric.KGE()
        
        return (fitness, )
    
    def operatorMate(self, parent1, parent2):
        return tools.cxSimulatedBinaryBounded(parent1, parent2, )
    
    def operatorMutate(self, ind):
        return tools.
    
    def operatorSelect(self, population):
        return tools.selNSGA2(population, self.popSize)


class NSGAII_VIC_MO(NSGAII_Base):
    
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
