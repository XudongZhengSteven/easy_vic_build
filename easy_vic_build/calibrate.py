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


class NSGAII_VIC_SO(NSGAII_Base):
    
    def __init__(self, dpc_VIC_level1, evb_dir, algParams=..., save_path="checkpoint.pkl"):
        super().__init__(algParams, save_path)
        self.evb_dir = evb_dir
        self.dpc_VIC_level1 = dpc_VIC_level1
        
        # params boundary
        self.g_boundary = g_boundary
        self.uh_params_boundary = uh_params_boundary
        self.routing_params_boundary = routing_params_boundary
        
        # get obs
        self.get_obs()
    
    def get_obs(self):
        self.obs_streamflow = self.dpc_VIC_level1
    
    def get_sim(self):
        os.path.join(self.VICResults_dir, "")
    
    def createFitness(self):
        creator.create("Fitness", base.Fitness, weights=(-1.0, ))

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
        # get ind
        
        # type params
        
        # build params
        
        # build RVIC params
        buildParam_level0(g_list, dpc_VIC_level0, evb_dir, reverse_lat=True)
        
        # run VIC
        vic_path = os.path.join()
        globalParam_path = os.path.join(self.evb_dir.GlobalParam_dir, "global_param.txt")
        os.system("./vic.exe -g ")
        
        # get sims
        
        
        # get obs
        # self.obs_streamflow()
        
        # evaluate
        evaluation_metric = EvaluationMetric()
        
        
        return super().evaluate(ind)
    
    def evaluatePop(self, population):
        return super().evaluatePop(population)
    
    def operatorMate(self):
        return super().operatorMate()
    
    def operatorMutate(self):
        return super().operatorMutate()
    
    def operatorSelect(self):
        return super().operatorSelect()


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
