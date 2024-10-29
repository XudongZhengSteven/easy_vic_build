# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import numpy as np
from scipy.stats import pearsonr

class EvaluationMetric:

    def __init__(self, sim, obs):
        self.sim = np.array(sim)
        self.obs = np.array(obs)

    def MSE(self):
        mse = (sum((self.sim - self.obs) ** 2) / len(self.sim))
        return mse
    
    def RMSE(self):
        """ RMSE """
        rmse = (sum((self.sim - self.obs) ** 2) / len(self.sim)) ** 0.5
        return rmse

    def RRMSE(self):
        """ RRMSE """
        rrmse = (sum((self.sim - self.obs) ** 2)) ** 0.5 / len(self.sim) / self.obs.mean()
        return rrmse

    def R(self, confidence: float = 0.95):
        """ R: Pearson Correlation Coefficient """
        r, p_value = pearsonr(self.sim, self.obs)
        # or np.corrcoef(self.sim, self.obs)[0, 1]
        significance = 0
        if p_value < 1 - confidence:
            if r > 0:
                significance = 1
            elif r < 0:
                significance = -1
        
        return r, p_value, significance
    
    def R2(self):
        """ R2 of Linear Fit """
        r = np.corrcoef(self.sim, self.obs)[0, 1]
        r2 = r ** 2
        
        return r2
    
    def NSE(self):
        """ Nash coefficient of efficiency """
        nse = 1 - sum((self.obs - self.sim) ** 2) / sum((self.obs - self.sim.mean()) ** 2)
        return nse

    def Bias(self):
        """ Bias """
        bias = (self.obs - self.sim).mean()
        return bias

    def PBias(self):
        pbias = sum(self.obs - self.sim) / sum(self.obs) * 100
        return pbias
    
    def KGE(self):
        r = np.corrcoef(self.sim, self.obs)[0, 1]
        beta = np.mean(self.sim) / np.mean(self.obs)
        gamma = np.std(self.sim) / np.std(self.obs)
        
        kge = 1 - ((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2) ** 0.5
        return kge

    def KGE_m(self):
        r = np.corrcoef(self.sim, self.obs)[0, 1]
        beta = np.mean(self.sim) / np.mean(self.obs)
        gamma = (np.std(self.sim) / np.mean(self.sim)) / (np.std(self.obs) / np.mean(self.obs))
        
        kge = 1 - ((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2) ** 0.5
        return kge


