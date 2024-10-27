# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from copy import copy
from datetime import datetime
import copy


def readForcingDaymet(hru_id_list=None):
    # hru_id_list: [hru_id ...] to extract specific stations, or read all stations, note that hru_id is int type
    # general set
    forcingDaymet_dir = "E:/data/hydrometeorology/CAMELS/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/daymet"

    fns = []
    fpaths = []
    forcingDaymet = []
    forcingDaymetGaugeAttributes = []

    # loop read all fns
    for dir in os.listdir(forcingDaymet_dir):
        fns.extend([fn for fn in os.listdir(os.path.join(forcingDaymet_dir, dir)) if fn.endswith(".txt")])
        fpaths.extend([os.path.join(forcingDaymet_dir, dir, fn)
                      for fn in os.listdir(os.path.join(forcingDaymet_dir, dir)) if fn.endswith(".txt")])

    # extract fn/fns based on hru_id
    if hru_id_list is not None:
        fns = [fn for fn in fns if int(fn[:fn.find("_")]) in hru_id_list]
        fpaths = [fpath for fpath in fpaths if int(fpath[fpath.rfind("\\") + 1:fpath.rfind("lump_cida")-1]) in hru_id_list]
    
    for i in range(len(fns)):
        fn = fns[i]
        fpath = fpaths[i]
        forcingDaymet.append(pd.read_csv(fpath, sep="\s+", skiprows=3))
        GaugeAttributes_ = pd.read_csv(fpath, header=None, nrows=3).values
        forcingDaymetGaugeAttributes.append({"latitude": GaugeAttributes_[0][0],
                                             "elevation": GaugeAttributes_[1][0],
                                             "basinArea": GaugeAttributes_[2][0],
                                             "gauge_id": int(fn[:8])})

    return fns, fpaths, forcingDaymet, forcingDaymetGaugeAttributes


def ExtractForcingDaymet(forcingDaymet, forcingDaymetGaugeAttributes, gauge_id, read_dates, plot=False):
    """_summary_

    Args:
        forcingDaymet (_type_): _description_
        forcingDaymetGaugeAttributes (_type_): _description_
        gauge_id (_type_): _description_
        extract_dates (_type_): _description_, #!note should be set to avoid missing value
            e.g. extract_dates = pd.date_range("19800101", "20141231", freq="D")
        plot (bool, optional): _description_. Defaults to False.
    """
    # read as df
    forcingDaymet_gauge_id = np.array([s["gauge_id"] for s in forcingDaymetGaugeAttributes])
    gauge_index = np.where(forcingDaymet_gauge_id == gauge_id)
    forcingDaymet_gauge = copy.copy(forcingDaymet[gauge_index[0][0]])
    
    # create datetimeIndex
    dates = list(map(lambda i: datetime(*i), zip(forcingDaymet_gauge.loc[:, 'Year'], forcingDaymet_gauge.loc[:, 'Mnth'],
                                                 forcingDaymet_gauge.loc[:, 'Day'])))
    dates = pd.to_datetime(dates)
    forcingDaymet_gauge.index = dates
    
    if read_dates is not None:
        date_period_index_bool = (dates >= datetime.strptime(read_dates[0], "%Y%m%d")) & (dates <= datetime.strptime(read_dates[1], "%Y%m%d"))
    else:
        date_period_index_bool = np.full(len(forcingDaymet_gauge.index), fill_value=True)
    
    # extract
    try:
        prcp = forcingDaymet_gauge.loc[date_period_index_bool, "prcp(mm/day)"]
        srad = forcingDaymet_gauge.loc[date_period_index_bool, "srad(W/m2)"]
        swe = forcingDaymet_gauge.loc[date_period_index_bool, "swe(mm)"]
        tmax = forcingDaymet_gauge.loc[date_period_index_bool, "tmax(C)"]
        tmin = forcingDaymet_gauge.loc[date_period_index_bool, "tmin(C)"]
        vp = forcingDaymet_gauge.loc[date_period_index_bool, "vp(Pa)"]
        
        forcingDaymet_gauge_set = {"prcp(mm/day)": prcp,
                                "srad(W/m2)": srad,
                                "swe(mm)": swe,
                                "tmax(C)": tmax,
                                "tmin(C)": tmin,
                                "vp(Pa)": vp}
    
        # plot
        fig_all = []
        if plot:
            for k in forcingDaymet_gauge_set:
                fig, ax = plt.subplots()
                forcingDaymet_gauge_set[k].plot(ax=ax, label=k)
                plt.legend()
                fig_all.append(fig)
                
    except ValueError(f"extract_period not suitable for gauge_id: {gauge_id}, return None"):
        forcingDaymet_gauge_set = None
        fig_all = None
            
    return forcingDaymet_gauge_set, fig_all


def ExtractData(basinShp, read_dates=None, read_keys=None):
    """
    params:
        read_dates: pd.date_range("19800101", "20141231", freq="D"), should be set to avoid missing value
        read_keys: ["prcp(mm/day)"]  # "prcp(mm/day)" "srad(W/m2)" "dayl(s)" "swe(mm)" "tmax(C)" "tmin(C)" "vp(Pa)"
    """
    # get data
    fns, fpaths, forcingDaymet, forcingDaymetGaugeAttributes = readForcingDaymet()
    
    # set read_keys
    read_keys = read_keys if read_keys is not None else ["prcp(mm/day)", "srad(W/m2)", "dayl(s)", "swe(mm)", "tmax(C)", "tmin(C)", "vp(Pa)"]
    
    extract_lists = [[] for i in range(len(read_keys))]
    for i in basinShp.index:
        basinShp_i = basinShp.loc[i, :]
        hru_id = basinShp_i.hru_id
        for j in range(len(read_keys)):
            key = read_keys[j]
            extract_list = extract_lists[j]
            forcingDaymet_basin_set, _ = ExtractForcingDaymet(forcingDaymet, forcingDaymetGaugeAttributes, hru_id, read_dates)
            extract_list.append(forcingDaymet_basin_set[key])

    for j in range(len(read_keys)):
        key = read_keys[j]
        extract_list = extract_lists[j]
        basinShp[key] = extract_list

    return basinShp
    
    