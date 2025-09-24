# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: evaluate_metrics

This module provides a set of evaluation metrics for assessing the performance of
simulated and observed data. The metrics implemented include commonly used statistical
and performance measures in model validation, such as Mean Squared Error (MSE), Root
Mean Squared Error (RMSE), Pearson Correlation Coefficient (R), Nash-Sutcliffe Efficiency
(NSE), Bias, Percent Bias (PBias), and Kling-Gupta Efficiency (KGE). These metrics help
to quantify the accuracy and reliability of the simulation model by comparing its
output with the observed data.

Class:
--------
    EvaluationMetric: A class for evaluating simulated and observed data using various statistical metrics.

Class Methods:
---------------
    - MSE: Computes the Mean Squared Error (MSE) between the simulated and observed values.
    - RMSE: Computes the Root Mean Squared Error (RMSE) between the simulated and observed values.
    - RRMSE: Computes the Relative Root Mean Squared Error (RRMSE) between the simulated and observed values.
    - R: Computes the Pearson Correlation Coefficient (R) and its significance based on a given confidence level.
    - R2: Computes the R-squared (R²) value of the linear fit between the simulated and observed values.
    - NSE: Computes the Nash-Sutcliffe Efficiency (NSE) coefficient between the simulated and observed values.
    - Bias: Computes the bias between the simulated and observed values.
    - PBias: Computes the Percent Bias (PBias) between the simulated and observed values.
    - KGE: Computes the Kling-Gupta Efficiency (KGE) metric between the simulated and observed values.
    - KGE_m: Computes the modified Kling-Gupta Efficiency (KGE-m) metric between the simulated and observed values.

Usage:
------
    1. Instantiate the `EvaluationMetric` class with simulated and observed data.
    2. Call the relevant method to compute the desired evaluation metric:
        - `MSE()` for Mean Squared Error.
        - `RMSE()` for Root Mean Squared Error.
        - `R()` for Pearson Correlation Coefficient.
        - `NSE()` for Nash-Sutcliffe Efficiency.
    3. Use the returned metric values to evaluate the model's performance.

Example:
--------
    sim_data = [1.0, 2.0, 3.0]
    obs_data = [1.2, 2.1, 2.9]

    eval_metric = EvaluationMetric(sim_data, obs_data)
    mse = eval_metric.MSE()
    print(f"Mean Squared Error: {mse}")

Dependencies:
-------------
    - numpy: For numerical operations on arrays.
    - scipy: For computing the Pearson correlation coefficient.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com
"""

import numpy as np
from scipy.stats import pearsonr
from eofs.standard import Eof
from scipy.spatial.distance import cosine


class EvaluationMetric:
    """
    A class for evaluating the performance of simulated and observed data
    using various statistical metrics.

    Attributes
    ----------
    sim : numpy.ndarray
        Simulated values.
    obs : numpy.ndarray
        Observed values.

    Methods
    -------
    MSE()
        Computes the Mean Squared Error (MSE) between the simulated and observed values.
    RMSE()
        Computes the Root Mean Squared Error (RMSE) between the simulated and observed values.
    RRMSE()
        Computes the Relative Root Mean Squared Error (RRMSE) between the simulated and observed values.
    R(confidence=0.95)
        Computes the Pearson correlation coefficient (R) and its significance based on a given confidence level.
    R2()
        Computes the R-squared (R²) value of the linear fit between the simulated and observed values.
    NSE()
        Computes the Nash-Sutcliffe Efficiency (NSE) coefficient between the simulated and observed values.
    Bias()
        Computes the bias between the simulated and observed values.
    PBias()
        Computes the Percent Bias (PBias) between the simulated and observed values.
    KGE()
        Computes the Kling-Gupta Efficiency (KGE) metric between the simulated and observed values.
    KGE_m()
        Computes the modified Kling-Gupta Efficiency (KGE-m) metric between the simulated and observed values.
    """

    def __init__(self, sim, obs):
        """
        Initializes the EvaluationMetric class with simulated and observed values.

        Parameters
        ----------
        sim : array-like
            Simulated values.
        obs : array-like
            Observed values.
        """
        self.sim = np.array(sim)
        self.obs = np.array(obs)

    def MSE(self):
        """
        Computes the Mean Squared Error (MSE) between the simulated and observed values.

        Returns
        -------
        float
            The calculated MSE.
        """
        mse = sum((self.sim - self.obs) ** 2) / len(self.sim)
        return mse

    def RMSE(self):
        """
        Computes the Root Mean Squared Error (RMSE) between the simulated and observed values.

        Returns
        -------
        float
            The calculated RMSE.
        """
        rmse = (sum((self.sim - self.obs) ** 2) / len(self.sim)) ** 0.5
        return rmse

    def RRMSE(self):
        """
        Computes the Relative Root Mean Squared Error (RRMSE) between the simulated and observed values.

        Returns
        -------
        float
            The calculated RRMSE.
        """
        rrmse = (
            (sum((self.sim - self.obs) ** 2)) ** 0.5 / len(self.sim) / self.obs.mean()
        )
        return rrmse

    def R(self, confidence: float = 0.95):
        """
        Computes the Pearson correlation coefficient (R) and its significance.

        Parameters
        ----------
        confidence : float, optional
            The confidence level to determine the significance, by default 0.95.

        Returns
        -------
        tuple
            A tuple containing the correlation coefficient (r), p-value, and significance:
            - r : float
                The Pearson correlation coefficient.
            - p_value : float
                The p-value corresponding to the correlation coefficient.
            - significance : int
                A value indicating the significance of the correlation:
                1 for positive correlation, -1 for negative, and 0 for no significant correlation.
        """
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
        """
        Computes the R-squared (R²) value of the linear fit between the simulated and observed values.

        Returns
        -------
        float
            The calculated R² value.
        """
        r = np.corrcoef(self.sim, self.obs)[0, 1]
        r2 = r**2

        return r2

    def NSE(self):
        """
        Computes the Nash-Sutcliffe Efficiency (NSE) coefficient.

        The NSE measures how well the simulated values match the observed values,
        with higher values indicating better performance.

        Returns
        -------
        float
            The calculated NSE value.
        """
        nse = 1 - sum((self.obs - self.sim) ** 2) / sum(
            (self.obs - self.sim.mean()) ** 2
        )
        return nse

    def Bias(self):
        """
        Computes the bias between the simulated and observed values.

        The bias is the mean difference between the observed and simulated values.

        Returns
        -------
        float
            The calculated bias.
        """
        bias = (self.obs - self.sim).mean()
        return bias

    def PBias(self):
        """
        Computes the Percent Bias (PBias) between the simulated and observed values.

        Returns
        -------
        float
            The calculated PBias.
        """
        pbias = sum(self.obs - self.sim) / sum(self.obs) * 100
        return pbias

    def KGE(self):
        """
        Computes the Kling-Gupta Efficiency (KGE) metric between the simulated and observed values.

        The KGE metric is based on the correlation coefficient (r), the ratio of means (beta),
        and the ratio of standard deviations (gamma).

        Returns
        -------
        float
            The calculated KGE value.
        """
        r = np.corrcoef(self.sim, self.obs)[0, 1]
        beta = np.mean(self.sim) / np.mean(self.obs)
        gamma = np.std(self.sim) / np.std(self.obs)

        kge = 1 - ((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2) ** 0.5
        return kge

    def KGE_m(self):
        """
        Computes the modified Kling-Gupta Efficiency (KGE-m) metric between the simulated and observed values.

        The KGE-m metric is similar to KGE but adjusts the gamma term to account for
        the relative standard deviations of the observed and simulated values.

        Returns
        -------
        float
            The calculated KGE-m value.
        """
        r = np.corrcoef(self.sim, self.obs)[0, 1]
        beta = np.mean(self.sim) / np.mean(self.obs)
        gamma = (np.std(self.sim) / np.mean(self.sim)) / (
            np.std(self.obs) / np.mean(self.obs)
        )

        kge = 1 - ((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2) ** 0.5
        return kge
    
    def ESS(self, lats=None, n_modes=None, remove_mean=True, mask=None):
        assert self.obs.shape == self.sim.shape, "sim and obs must have identical dimensions"
        
        ntime, nlat, nlon = self.sim.shape
        
        # reshape
        sim_2d = self.sim.reshape(ntime, nlat * nlon)
        obs_2d = self.obs.reshape(ntime, nlat * nlon)
        
        # mask
        if mask is not None:
            assert mask.shape == (nlat, nlon), "mask shape must match spatial dimensions"
            flat_mask = mask.flatten()
            sim_2d = sim_2d[:, flat_mask]
            obs_2d = obs_2d[:, flat_mask]
            
        # combine sim and obs array
        combine_2d = np.concatenate([sim_2d, obs_2d], axis=0)
        
        # Compute anomalies by removing the time-mean
        if remove_mean:
            combine_2d -= np.mean(combine_2d, axis=0)
        
        # latitude weights are applied before the computation of EOFs
        if lats is not None:
            coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
            wgts = np.sqrt(coslat)[..., np.newaxis]
            
            if mask is not None:
                # Also apply mask to weights
                wgts = wgts.flatten()[flat_mask]
        else:
            wgts = None
        
        # solver
        solver = Eof(combine_2d, weights=wgts)
        
        # get values
        pcs = solver.pcs(npcs=n_modes)
        eigvals = solver.eigenvalues()[:n_modes]
        varfrac = solver.varianceFraction(neigs=n_modes)
        
        loadings = pcs * np.sqrt(eigvals)  # shape: (2*ntime, n_modes)
        
        # get ess
        obs_loadings = loadings[:ntime]        # shape: (ntime, n_modes)
        sim_loadings = loadings[ntime:]  # shape: (ntime, n_modes)
        
        diffs = np.abs(obs_loadings - sim_loadings)  # shape: (n_days, n_modes)
        
        ess = np.sum(diffs * varfrac[np.newaxis, :], axis=1)  # shape: (ntime,)
        
        return ess

    def spatialPCC(self, mask=None):
        assert self.obs.shape == self.sim.shape, "sim and obs must have identical dimensions"        

        ntime, nlat, nlon = self.sim.shape       

        # reshape
        sim_2d = self.sim.reshape(ntime, nlat * nlon)
        obs_2d = self.obs.reshape(ntime, nlat * nlon)        

        # mask
        if mask is not None:
            assert mask.shape == (nlat, nlon), "mask shape must match spatial dimensions"
            flat_mask = mask.flatten()
            sim_2d = sim_2d[:, ~flat_mask]
            obs_2d = obs_2d[:, ~flat_mask]
            
        pcc_array = np.full(ntime, np.nan)

        for t in range(ntime):
            sim_vec = sim_2d[t, :]
            obs_vec = obs_2d[t, :]

            valid_idx = (~np.isnan(sim_vec)) & (~np.isnan(obs_vec))

            if np.sum(valid_idx) > 1:
                pcc = np.corrcoef(sim_vec[valid_idx], obs_vec[valid_idx])[0, 1]
                pcc_array[t] = pcc
            else:
                pcc_array[t] = np.nan

        return pcc_array


class CategoricalEvaluationMetric:
    
    def __init__(self, sim, obs):
        """
        Initializes the EvaluationMetric class with simulated and observed values.

        Parameters
        ----------
        sim : array-like
            Simulated values.
        obs : array-like
            Observed values.
        """
        self.sim = np.array(sim)
        self.obs = np.array(obs)
        self.sim_bin = None
        self.obs_bin = None
        self.H = self.M = self.F = self.CN = None
    
    def binarize(self, threshold=0.0):
        """
        Convert continuous values to binary events based on threshold
        Values > threshold are considered 'event occurred' (1), else 'no event' (0).
        
        Parameters
        ----------
        threshold : float
            Threshold for event occurrence.
        """
        self.sim_bin = (self.sim > threshold).astype(int)
        self.obs_bin = (self.obs > threshold).astype(int)
        self._compute_contingency()

    def _compute_contingency(self):
        """Compute Hit, Miss, False Alarm, Correct Negative counts """
        self.H = np.sum((self.sim_bin == 1) & (self.obs_bin == 1))
        self.M = np.sum((self.sim_bin == 0) & (self.obs_bin == 1))
        self.F = np.sum((self.sim_bin == 1) & (self.obs_bin == 0))
        self.CN = np.sum((self.sim_bin == 0) & (self.obs_bin == 0))
        
    def POD(self):
        """ POD (Probability of Detection) """
        return self.H / (self.H + self.M) if (self.H + self.M) > 0 else None

    def FAR(self):
        """ FAR (False Alarm Ratio) """
        return self.F / (self.H + self.F) if (self.H + self.F) > 0 else None

    def CSI(self):
        """ CSI (Critical Success Index) """
        return self.H / (self.H + self.M + self.F) if (self.H + self.M + self.F) > 0 else None

    def HSS(self):
        """ HSS (Heidke Skill Score) """
        num = 2 * (self.H * self.CN - self.M * self.F)
        den = (self.H + self.M) * (self.M + self.CN) + (self.H + self.F) * (self.F + self.CN)
        return num / den if den > 0 else None

    def ETS(self):
        """ ETS (Equitable Threat Score) """
        H_random = (self.H + self.M) * (self.H + self.F) / (self.H + self.M + self.F + self.CN)
        den = self.H + self.M + self.F - H_random
        return (self.H - H_random) / den if den > 0 else None
        

class SignatureEvaluationMetric:
    
    def __init__(self, sim, obs):
        """
        Initializes the EvaluationMetric class with simulated and observed values.

        Parameters
        ----------
        sim : array-like
            Simulated values.
        obs : array-like
            Observed values.
        """
        self.sim = np.array(sim)
        self.obs = np.array(obs)
        
    def FHVBias(self, q_high=0.9):
        """
        Flood High Volume error (relative).

        Parameters
        ----------
        q_high : float, optional (default=0.9)
            Quantile threshold to define high flows (e.g., top 10%).

        Returns
        -------
        float
            Relative error in high flows.
        """
        thresh = np.quantile(self.obs, q_high)
        mask = self.obs >= thresh

        sim_high = self.sim[mask]
        obs_high = self.obs[mask]

        return ((sim_high - obs_high).sum()) / obs_high.sum()
    
    def FLVBias(self, q_low=0.1):
        """
        Flood Low Volume error (relative).

        Parameters
        ----------
        q_low : float, optional (default=0.1)
            Quantile threshold to define low flows (e.g., bottom 10%).

        Returns
        -------
        float
            Relative error in low flows.
        """
        thresh = np.quantile(self.obs, q_low)
        mask = self.obs <= thresh

        sim_low = self.sim[mask]
        obs_low = self.obs[mask]

        return ((sim_low - obs_low).sum()) / obs_low.sum()
    
    def FMSPBias(self, mid_low=0.2, mid_high=0.7):
        """
        Flow Magnitude Skewness (FMS).
        Relative error in the slope of the flow duration curve
        (between Q20 and Q70, i.e. mid-range flows).

        Returns
        -------
        float
        """
        q20_obs, q70_obs = np.quantile(self.obs, [mid_low, mid_high])
        q20_sim, q70_sim = np.quantile(self.sim, [mid_low, mid_high])

        slope_obs = (q20_obs - q70_obs) / (mid_high - mid_low)
        slope_sim = (q20_sim - q70_sim) / (mid_high - mid_low)

        fms = 100 * (slope_sim - slope_obs) / slope_obs
        return fms
    
    def CVBias(self):
        """
        variability
        Coefficient of Variation, relative error of CV
        CV = std / mean
        """
        cv_obs = np.std(self.obs) / np.mean(self.obs)
        cv_sim = np.std(self.sim) / np.mean(self.sim)
        return (cv_sim - cv_obs) / cv_obs
    
def create_test_data(seed=42):
    np.random.seed(seed)
    
    T, H, W = 10, 25, 25

    # Construct a base spatial pattern (e.g., high values concentrated in upper-right)
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    X, Y = np.meshgrid(x, y)
    
    # Gaussian pattern centered at (0.8, 0.2)
    base_pattern = np.exp(-((X - 0.8)**2 + (Y - 0.2)**2) / 0.05)
    
    # Create time-dependent scaling (simulating seasonal or dynamic variation)
    time_scaling = np.sin(np.linspace(0, 2*np.pi, T)) + 1.5  # shifted sine wave to ensure all values ≥ 0

    # Construct data1: reference dataset with clean spatial-temporal structure
    data1 = np.array([time_scaling[t] * base_pattern for t in range(T)])  # shape: (T, H, W)

    # Construct data2: similar to data1 but with additional noise and a slight spatial bias
    noise_level = 0.2
    
    # Random spatial noise added to each time step
    spatial_noise = np.random.normal(0, noise_level, size=(T, H, W))
    
    # Small systematic bias in spatial structure (simulating model error)
    mode_shift = 0.03 * np.random.randn(H, W)

    # Combine to generate simulated dataset
    data2 = data1 + spatial_noise + mode_shift
    
    return data1, data2


def create_test_data2():
    n_time = 30
    n_lat = 25
    n_lon = 25

    lat_grid, lon_grid = np.meshgrid(np.linspace(-1, 1, n_lat), np.linspace(-1, 1, n_lon))
    base_pattern = np.exp(-4 * (lat_grid**2 + lon_grid**2))

    obs_data = np.zeros((n_time, n_lat, n_lon))
    sim_data = np.zeros((n_time, n_lat, n_lon))

    for t in range(n_time):
        noise_level = 0.0
        if t < 5:
            noise_level = 0.05
        elif t < 25:
            noise_level = 0.3
        else:
            noise_level = 0.5

        obs_data[t] = base_pattern + np.random.normal(0, 0.02, size=base_pattern.shape)
        sim_data[t] = base_pattern + np.random.normal(0, noise_level, size=base_pattern.shape)
    
    return obs_data, sim_data


if __name__ == "__main__":
    data1, data2 = create_test_data2()
    
    EM = EvaluationMetric(data1, data2)
    
    ess = EM.ESS()
    