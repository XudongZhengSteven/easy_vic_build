# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
from scipy.optimize import minimize
from scipy.stats import gamma
import math
import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt
import os


def get_max_day(UH_func_frozen_param, max_day_range=(0, 10), max_day_converged_threshold=0.001):
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