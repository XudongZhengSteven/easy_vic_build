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
