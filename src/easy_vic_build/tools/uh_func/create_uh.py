# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: create_uh

This module provides functions for generating Unit Hydrographs (UH) using different methodologies, including 
the General Unit Hydrograph (gUH) and Nash Unit Hydrograph (NashUH). These functions allow for the creation of 
dimensionless unit hydrographs based on specified time parameters and other relevant inputs. The module includes 
utilities for plotting and visualizing the generated unit hydrographs, and it also includes functionality for 
determining the maximum day for the UH based on convergence criteria.

Functions:
----------
    - get_max_day: Determines the maximum day for UH calculation based on the convergence of the hydrograph.
    - createGUH: Generates a General Unit Hydrograph using an analytical expression based on time parameters.
    - createNashUH: Generates a Nash Unit Hydrograph using a gamma distribution with specified parameters.

Dependencies:
-------------
    - numpy: Used for numerical operations, particularly with arrays and time-related calculations.
    - pandas: For creating and manipulating data structures, especially DataFrames for output.
    - matplotlib.pyplot: Used for creating static visualizations of the unit hydrographs.
    - scipy.stats.gamma: Provides tools for generating the gamma distribution used in the NashUH calculation.
    - os: For interacting with the file system, including saving plot images.
    
Author:
-------
    Xudong Zheng
    Email: zhengxd@sehemodel.club
"""

from scipy.optimize import minimize
from scipy.stats import gamma
import math
import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt
import os


def get_max_day(UH_func_frozen_param, max_day_range=(0, 10), max_day_converged_threshold=0.001):
    """
    Calculate the maximum day (in days) at which the unit hydrograph (UH) function converges below a specified threshold.

    This function automatically calculates the maximum day for the UH function, starting with an initial 
    guess for the top time and iterating to find the day at which the function falls below the convergence threshold.

    Parameters
    ----------
    UH_func_frozen_param : function
        A function representing the unit hydrograph (UH) at a given time in hours. 
        The function should take a single argument `t` (time in hours) and return the UH value.
    
    max_day_range : tuple, optional
        The range of maximum days (in days) to consider for the search. 
        Defaults to (0, 10), i.e., searching within 0 to 10 days.

    max_day_converged_threshold : float, optional
        The threshold below which the UH function is considered converged. 
        Defaults to 0.001.

    Returns
    -------
    max_day : int
        The maximum day (rounded up to the nearest whole day) at which the UH function converges below 
        the specified threshold.

    Notes
    -----
    The function uses a binary search approach to find the time when the UH function value first falls 
    below the specified threshold, with the search range defined by `max_day_range`. The result is 
    returned in days (rounded up).
    
    """
    # UH_func_frozen_param: only receive t (hours)
    # auto-calculate the max_day for the uhbox
    
    # get top_t
    max_t_range_top_t = [0, max_day_range[1]*24]
    neg_func = lambda t: -1 * UH_func_frozen_param(t)
    result_top_t = minimize(neg_func, [1])
    top_t = result_top_t.x[0]
    
    # get max_day
    max_t_range = [top_t, max_day_range[1]*24]
    
    def UH_func_target(t):
        ret = 0 if UH_func_frozen_param(t) < max_day_converged_threshold else UH_func_frozen_param(t) - max_day_converged_threshold
        ret = ret if not np.isnan(ret) else np.inf
        return ret
    
    def find_first_below_threshold(func, max_t_range, max_day_converged_threshold, tol=1e-5):
        left = max_t_range[0]
        right = max_t_range[1]
        
        while right - left > tol:
            mid = (left + right) / 2
            mid_value = func(mid)
            
            if np.isnan(mid_value):
                right = mid
            elif mid_value <= max_day_converged_threshold:
                right = mid
            else:
                left = mid
        
        return (left + right) / 2
    
    result = find_first_below_threshold(UH_func_frozen_param, max_t_range, max_day_converged_threshold, tol=1e-5)
    
    max_day = math.ceil(result / 24)
    
    return max_day


def createGUH(evb_dir, uh_dt=3600, tp=1.4, mu=5.0, m=3.0, plot_bool=False, max_day=None, max_day_range=(0, 10), max_day_converged_threshold=0.001):
    """
    Create a General Unit Hydrograph (gUH) and optionally plot and save the results.

    This function calculates the General Unit Hydrograph (gUH) using parameters defined in the SCS UH method,
    including time steps, and generates a UHBOX file for hydrological analysis. The function also provides an 
    option to plot the gUH functions and save the plots as images.

    Parameters
    ----------
    evb_dir : object
        The directory object where the results, including plots, will be saved.
        
    uh_dt : int, optional
        The time step (in seconds) for the UH calculation. Default is 3600 seconds (1 hour).
        
    tp : float, optional
        The time parameter for the General Unit Hydrograph (gUH) model in hours. Default is 1.4 hours.
        
    mu : float, optional
        The scaling factor for the gUH model. Default is 5.0, based on SCS UH.
        
    m : float, optional
        A parameter for the gUH model. Should be greater than 1. Default is 3.0, based on SCS UH.
        
    plot_bool : bool, optional
        If True, the function will plot the gUH and save the plot. Default is False.
        
    max_day : int, optional
        The maximum day for the UH calculation. If None, the function will calculate it using `get_max_day`.
        
    max_day_range : tuple, optional
        The range of maximum days (in days) to consider for the search. Default is (0, 10).
        
    max_day_converged_threshold : float, optional
        The threshold below which the UH function is considered converged. Default is 0.001.

    Returns
    -------
    max_day : int
        The maximum day (rounded up to the nearest whole day) at which the UH function converges below 
        the specified threshold.

    UHBOX_file : pandas.DataFrame
        A DataFrame containing the time and corresponding General Unit Hydrograph (gUH) values.

    Notes
    -----
    The function uses the method from Guo (2022) to calculate the general and analytic unit hydrograph (gUH).
    It also supports plotting the results for visualization purposes. The calculated UHBOX file is saved as 
    a DataFrame containing the time series and corresponding gUH values.
    
    """
    # general UH
    # default uh_dt=3600 (hours)
    # dimensionless by tp (hours)
    # ====================== build UHBOXFile ======================
    # tp (hourly, 0~2.5h), mu (default 5.0, based on SCS UH), m (should > 1, default 3.0, based on SCS UH)
    # general UH function
    # Guo, J. (2022), General and Analytic Unit Hydrograph and Its Applications, Journal of Hydrologic Engineering, 27.
    gUH_xt = lambda t: np.exp(mu*(t/tp - 1))
    gUH_gt = lambda t: 1 - (1 + m*gUH_xt(t)) ** (-1/m)
    gUH_st = lambda t: 1 - gUH_gt(t)
    
    gUH_iuh = lambda t: mu/tp * gUH_xt(t) * (1 + m*gUH_xt(t)) ** (-(1+1/m))
    det_t = 1
    gUH_uh = lambda t: (gUH_gt(t)[1:] - gUH_gt(t)[:-1]) / det_t
    
    # t
    if max_day is None:
        UH_func_frozen_param = gUH_iuh
        max_day = get_max_day(UH_func_frozen_param, max_day_range, max_day_converged_threshold)
    
    # day_range = (0, max_day)
    t_step = uh_dt
    t_start = 0
    t_end = max_day * 24 * 3600 + t_step
    t_s = np.arange(t_start, t_end, t_step)  # s, det is uh_dt
    t_hour = t_s / 3600  # uh_dt -> hours, to input into UH (tp is hour, to make it dimensionless)
    
    # UH
    gUH_gt_ret = gUH_gt(t_hour)
    gUH_st_ret = gUH_st(t_hour)
    # gUH_uh_ret = gUH_uh(t_hour)
    gUH_iuh_ret = gUH_iuh(t_hour)
    
    # plot
    if plot_bool:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].plot(t_hour, gUH_iuh_ret, "-k", label="gUH_iuh", linewidth=1, alpha=1)
        # ax[0].plot(t_interval, gUH_uh_ret, "--k", label="gUH_uh", linewidth=3, alpha=0.5)

        ax[0].set_xlabel("time/hours")
        ax[0].set_ylabel("gUH (dimensionless)")
        ax[0].set_ylim(ymin=0)
        ax[0].set_xlim(xmin=0, xmax=t_hour[-1])
        ax[0].legend()
        
        ax[1].plot(t_hour, gUH_gt_ret, "-r", label="gUH_gt", linewidth=1)
        ax[1].plot(t_hour, gUH_st_ret, "-b", label="gUH_st", linewidth=1)
        
        ax[1].set_xlabel("time/hours")
        ax[1].set_ylabel("st, gt")
        ax[1].set_ylim(ymin=0)
        ax[1].set_xlim(xmin=0, xmax=t_hour[-1])
        ax[1].legend()
        
        if evb_dir is not None:
            fig.savefig(os.path.join(evb_dir.RVICParam_dir, "UHBOX.tiff"))
    
    # df
    UHBOX_file = pd.DataFrame(columns=["time", "UHBOX"])
    UHBOX_file.time = t_s
    UHBOX_file.UHBOX = gUH_iuh_ret
    UHBOX_file["UHBOX"] = UHBOX_file["UHBOX"].fillna(0)
    
    return max_day, UHBOX_file


def createNashUH(evb_dir, uh_dt=3600, n=None, K=None, tp=1.4, qp=0.15, plot_bool=False, max_day=None, max_day_range=(0, 10), max_day_converged_threshold=0.001):
    """
    Create a Nash Unit Hydrograph (UH) using a gamma distribution and optionally plot and save the results.

    This function calculates the Nash Unit Hydrograph (NashUH) based on the Nash gamma distribution. The function 
    also provides an option to plot the NashUH function and save the plots as images.

    Parameters
    ----------
    evb_dir : object
        The directory object where the results, including plots, will be saved.
        
    uh_dt : int, optional
        The time step (in seconds) for the UH calculation. Default is 3600 seconds (1 hour).
        
    n : float, optional
        The shape parameter for the gamma distribution. If None, it is calculated based on `beta` and `tp`. 
        Default is None.
        
    K : float, optional
        The scale parameter for the gamma distribution. If None, it is calculated based on `n` and `tp`. 
        Default is None.
        
    tp : float, optional
        The time parameter for the Nash UH model in hours. Default is 1.4 hours.
        
    qp : float, optional
        A parameter for the Nash UH model. Default is 0.15.
        
    plot_bool : bool, optional
        If True, the function will plot the Nash UH and save the plot. Default is False.
        
    max_day : int, optional
        The maximum day for the UH calculation. If None, the function will calculate it using `get_max_day`.
        
    max_day_range : tuple, optional
        The range of maximum days (in days) to consider for the search. Default is (0, 10).
        
    max_day_converged_threshold : float, optional
        The threshold below which the UH function is considered converged. Default is 0.001.

    Returns
    -------
    max_day : int
        The maximum day (rounded up to the nearest whole day) at which the UH function converges below 
        the specified threshold.

    UHBOX_file : pandas.DataFrame
        A DataFrame containing the time and corresponding Nash Unit Hydrograph (NashUH) values.

    Notes
    -----
    The function uses a gamma distribution to model the Nash Unit Hydrograph (NashUH) and calculates the 
    maximum day for which the UH function converges. It also supports plotting the results for visualization 
    purposes. The calculated UHBOX file is saved as a DataFrame containing the time series and corresponding 
    NashUH values.
    
    """
    # Nash Gamma UH
    # dimensionless by tp (hours)
    # Roy, A., and R. Thomas (2016), A Comparative Study on the Derivation of Unit Hydrograph for Bharathapuzha River Basin, Procedia Technology, 24, 62-69.
    # K < 20
    
    # cal N and K based on tp and qp
    beta = qp * tp  # beta should > 0.01
    if n is None:
        n = 6.29 * (beta**1.998) + 1.157 if beta > 0.35 else 5.53 * (beta**1.75) + 1.04
    
    if K is None:
        K = tp / (n - 1)
    
    # gamma distribution
    rv = gamma(a=n, scale=K)
    NashUH_iuh = lambda t: rv.pdf(t)
    
    # t
    if max_day is None:
        UH_func_frozen_param = NashUH_iuh
        max_day = get_max_day(UH_func_frozen_param, max_day_range, max_day_converged_threshold)
    
    # day_range = (0, max_day)
    t_step = uh_dt
    t_start = 0
    t_end = max_day * 24 * 3600 + t_step
    t_s = np.arange(t_start, t_end, t_step)  # s, det is uh_dt
    t_hour = t_s / 3600  # uh_dt -> hours, to input into UH (tp is hour, to make it dimensionless)
    
    # UH
    NashUH_iuh_ret = NashUH_iuh(t_hour)
    
    # plot
    if plot_bool:
        fig, ax = plt.subplots()
        ax.plot(t_hour, NashUH_iuh_ret, "-k", label="NashUH_iuh", linewidth=1, alpha=1)
        ax.set_xlabel("time/hours")
        ax.set_ylabel("gUH (dimensionless)")
        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=0, xmax=24*max_day-1)
        ax.legend()
        
        if evb_dir is not None:
            fig.savefig(os.path.join(evb_dir.RVICParam_dir, "UHBOX.tiff"))
    
    # df
    UHBOX_file = pd.DataFrame(columns=["time", "UHBOX"])
    UHBOX_file.time = t_s * 3600  # Convert to s
    UHBOX_file.UHBOX = NashUH_iuh_ret
    UHBOX_file["UHBOX"] = UHBOX_file["UHBOX"].fillna(0)
    
    return max_day, UHBOX_file