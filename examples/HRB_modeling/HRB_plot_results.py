# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import os
import pandas as pd
import numpy as np
import pickle
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import LinearLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from shapely.geometry import box
import geopandas as gpd
from scipy.stats import ttest_ind
import itertools

from easy_vic_build.tools.plot_func.plot_evaluation import taylor_diagram
from easy_vic_build.tools.calibrate_func.evaluate_metrics import EvaluationMetric, SignatureEvaluationMetric
from easy_vic_build.tools.utilities import readdpc
from easy_vic_build.tools.dpc_func.basin_grid_func import createArray_from_gridshp, createStand_grids_lat_lon_from_gridshp
from easy_vic_build.tools.plot_func.plot_utilities import set_boundary, set_xyticks, get_NDVI_cmap, get_colorbar, get_UMD_LULC_cmap
from easy_vic_build.tools.params_func.params_set import *

from general_info import *
from HRB_build_dpc import dataProcess_VIC_level0_HRB, dataProcess_VIC_level1_HRB, dataProcess_VIC_level3_HRB
from HRB_calibrate_backup2 import case_set, set_g_params_soilGrids_layer, get_inds_CMA_ES, get_inds_NSGAII

plt.rcParams['font.family']='Arial'
plt.rcParams['font.size']=12

def set_date(timestep, timestep_evaluate):
    warmup_date = pd.date_range(warmup_date_period[0], warmup_date_period[-1], freq=timestep)
    warmup_date_eval = pd.date_range(warmup_date_period[0], warmup_date_period[-1], freq=timestep_evaluate)

    calibrate_date = pd.date_range(calibrate_date_period[0], calibrate_date_period[-1], freq=timestep)
    calibrate_date_eval = pd.date_range(calibrate_date_period[0], calibrate_date_period[-1], freq=timestep_evaluate)

    verify_date = pd.date_range(verify_date_period[0], verify_date_period[-1], freq=timestep)
    verify_date_eval = pd.date_range(verify_date_period[0], verify_date_period[-1], freq=timestep_evaluate)

    total_date = pd.date_range(calibrate_date_period[0], verify_date_period[-1], freq=timestep)
    total_date_eval = pd.date_range(calibrate_date_period[0], verify_date_period[-1], freq=timestep_evaluate)

    plot_date_dict = {
        "calibration": calibrate_date_eval,
        "validation": verify_date_eval,
        "total": total_date_eval
    }
    
    return warmup_date, warmup_date_eval, calibrate_date, calibrate_date_eval, verify_date, verify_date_eval, total_date, total_date_eval, plot_date_dict

def get_obs(evb_dir_modeling, get_type="calibration", station_names=None):
    # read dpc_VIC_level3
    dpc_VIC_level3 = dataProcess_VIC_level3_HRB(evb_dir_modeling.dpc_VIC_level3_path)
    
    # set date index
    n_warmup = len(warmup_date_eval)
    n_calibration = len(calibrate_date_eval)
    n_validation = len(verify_date_eval)

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
    obs = {}
    
    # read streamflow
    basin_shp_with_streamflow = dpc_VIC_level3.get_data_from_cache("streamflow")[0]
    gauge_info = dpc_VIC_level3.get_data_from_cache("gauge_info")[0]
    
    # get all stations
    snaped_outlet_names = [gauge_info[key]["station_name"] for key in station_names]
    snaped_outlet_lons = [gauge_info[key]["gauge_coord(lon, lat)_level1"][0] for key in station_names]
    snaped_outlet_lats = [gauge_info[key]["gauge_coord(lon, lat)_level1"][1] for key in station_names]
    
    for name in station_names:
        obs[f"streamflow(m3/s)_{name}"] = basin_shp_with_streamflow[f"stationdata_streamflow_{name}"][0].iloc[start_index:end_index]
    
    return obs


def get_sim_rvic(evb_dir_modeling, case_name, ind_num, station_names=None):
    # set date index
    n_warmup = len(warmup_date_eval)
    n_calibration = len(calibrate_date_eval)
    n_validation = len(verify_date_eval)

    # read
    case_home = os.path.join(evb_dir_modeling.CalibrateVIC_dir, "case_40_best_inds", case_name + "_40")
    sim = {}
    for num in range(ind_num):
        sim[num] = {"cali": {}, "vali": {}}
        
        cali_nc_fn = f"HRB_shiquan_6km.rvic.h0a.2019-01-01_cali{num}.nc"
        vali_nc_fn = f"HRB_shiquan_6km.rvic.h0a.2019-01-01_vali{num}.nc"
        
        cali_nc_fp = os.path.join(case_home, cali_nc_fn)
        vali_nc_fp =  os.path.join(case_home, vali_nc_fn)
        
        # read cali
        with Dataset(cali_nc_fp, "r") as sim_dataset:
            # set date index
            start_index = n_warmup
            end_index = n_warmup + n_calibration
            
            # lon, lat
            sim_time = sim_dataset["time"]
            sim_time = num2date(sim_time[:], sim_time.units, sim_time.calendar)
            sim_time = sim_time[start_index:end_index]
            
            # streamflow
            for i, name in enumerate(station_names):
                sim[num]["cali"][f"streamflow(m3/s)_{name}"] = sim_dataset.variables["streamflow"][start_index:end_index, i]
        
        # read vali
        with Dataset(vali_nc_fp, "r") as sim_dataset:
            # set date index
            start_index = n_warmup + n_calibration
            end_index = n_warmup + n_calibration + n_validation
            
            # lon, lat
            sim_time = sim_dataset["time"]
            sim_time = num2date(sim_time[:], sim_time.units, sim_time.calendar)
            sim_time = sim_time[start_index:end_index]
            
            # streamflow
            for i, name in enumerate(station_names):
                sim[num]["vali"][f"streamflow(m3/s)_{name}"] = sim_dataset.variables["streamflow"][start_index:end_index, i]
                
    return sim


def get_sim_vic(evb_dir_modeling, case_name, ind_num):
    # set date index
    n_warmup = len(warmup_date_eval)
    n_calibration = len(calibrate_date_eval)
    n_validation = len(verify_date_eval)
    
    # read
    case_home = os.path.join(evb_dir_modeling.CalibrateVIC_dir, "case_40_best_inds", case_name + "_40")
    sim = {}
    for num in range(ind_num):
        sim[num] = {"cali": {}, "vali": {}}
        
        cali_nc_fn = f"fluxes.2003-01-01_cali{num}.nc"
        vali_nc_fn = f"fluxes.2003-01-01_vali{num}.nc"
        
        cali_nc_fp = os.path.join(case_home, cali_nc_fn)
        vali_nc_fp =  os.path.join(case_home, vali_nc_fn)
        
        # read cali
        with Dataset(cali_nc_fp, "r") as sim_dataset:
            # set date index
            start_index = n_warmup
            end_index = n_warmup + n_calibration
            
            # lon, lat
            sim_time = sim_dataset["time"]
            sim_time = num2date(sim_time[:], sim_time.units, sim_time.calendar)
            sim_time = sim_time[start_index:end_index]
            
            sim[num]["cali"][f"OUT_BASEFLOW"] = sim_dataset.variables["OUT_BASEFLOW"][start_index:end_index, :, :]
            sim[num]["cali"][f"OUT_RUNOFF"] = sim_dataset.variables["OUT_RUNOFF"][start_index:end_index, :, :]
            sim[num]["cali"][f"OUT_EVAP"] = sim_dataset.variables["OUT_EVAP"][start_index:end_index, :, :]
            sim[num]["cali"][f"OUT_SOIL_MOIST"] = sim_dataset.variables["OUT_SOIL_MOIST"][start_index:end_index, :, :, :]
        
        # read vali
        with Dataset(vali_nc_fp, "r") as sim_dataset:
            # set date index
            start_index = n_warmup + n_calibration
            end_index = n_warmup + n_calibration + n_validation
            
            # lon, lat
            sim_time = sim_dataset["time"]
            sim_time = num2date(sim_time[:], sim_time.units, sim_time.calendar)
            sim_time = sim_time[start_index:end_index]
            
            sim[num]["vali"][f"OUT_BASEFLOW"] = sim_dataset.variables["OUT_BASEFLOW"][start_index:end_index, :, :]
            sim[num]["vali"][f"OUT_RUNOFF"] = sim_dataset.variables["OUT_RUNOFF"][start_index:end_index, :, :]
            sim[num]["vali"][f"OUT_EVAP"] = sim_dataset.variables["OUT_EVAP"][start_index:end_index, :, :]
            sim[num]["vali"][f"OUT_SOIL_MOIST"] = sim_dataset.variables["OUT_SOIL_MOIST"][start_index:end_index, :, :, :]
            
    return sim


def get_obs_sim_all_inds(ind_num=5, case_num=8):
    # get obs
    obs_cali = get_obs(evb_dir_modeling, get_type="calibration", station_names=station_names)
    obs_vali = get_obs(evb_dir_modeling, get_type="validation", station_names=station_names)
    
    obs_cali_all_stations = {}
    obs_vali_all_stations = {}
    sim_cali_all_stations = {}
    sim_vali_all_stations = {}
    
    # get all stations
    for name in station_names:
        # obs
        obs_cali_station = obs_cali[f"streamflow(m3/s)_{name}"].values.flatten()
        obs_vali_station = obs_vali[f"streamflow(m3/s)_{name}"].values.flatten()

        obs_cali_all_stations[name] = obs_cali_station
        obs_vali_all_stations[name] = obs_vali_station
        
        # sim
        sim_cali_case = {}
        sim_vali_case = {}
        
        for case_n in range(1, case_num+1):
            case_name = f"case{case_n}"
            
            sim = get_sim_rvic(evb_dir_modeling, case_name, ind_num, station_names=station_names)
            
            sim_cali_streamflow_all_ind = []
            sim_vali_streamflow_all_ind = []
            for num in range(ind_num):
                sim_ind = sim[num]
                
                sim_cali_streamflow = sim_ind["cali"][f"streamflow(m3/s)_{name}"].filled(0)
                sim_vali_streamflow = sim_ind["vali"][f"streamflow(m3/s)_{name}"].filled(0)
                
                sim_cali_streamflow_all_ind.append(sim_cali_streamflow)
                sim_vali_streamflow_all_ind.append(sim_vali_streamflow)
            
            sim_cali_streamflow_all_ind = np.mean(np.array(sim_cali_streamflow_all_ind), axis=0)
            sim_vali_streamflow_all_ind = np.mean(np.array(sim_vali_streamflow_all_ind), axis=0)
        
            sim_cali_case[case_name] = sim_cali_streamflow_all_ind
            sim_vali_case[case_name] = sim_vali_streamflow_all_ind
        
        sim_cali_all_stations[name] = sim_cali_case
        sim_vali_all_stations[name] = sim_vali_case
        
    return obs_cali_all_stations, obs_vali_all_stations, sim_cali_all_stations, sim_vali_all_stations


def get_params(evb_dir_modeling, case_name, ind_num, period="vali"):
    # read
    case_home = os.path.join(evb_dir_modeling.CalibrateVIC_dir, "case_40_best_inds", case_name + "_40")
    params = {}
    for num in range(ind_num):
        params[num] = {"level0": {}, "level1": {}}
        params_level0_nc_fn = f"params_level0_{period}{num}.nc"
        params_level1_nc_fn = f"params_level1_{period}{num}.nc"
        
        with Dataset(os.path.join(case_home, params_level0_nc_fn), "r") as params_level0_dataset:
            params[num]["level0"]["depth"] = params_level0_dataset["depth"][:, :, :]
            
        with Dataset(os.path.join(case_home, params_level1_nc_fn), "r") as params_level1_dataset:
            params[num]["level1"]["depth"] = params_level1_dataset["depth"][:, :, :]
    
    return params
    
def get_calibration_cps(evb_dir_modeling, case_num=8, ind_num=5, max_num=40):
    cps_home = os.path.join(os.path.join(evb_dir_modeling.CalibrateVIC_dir, "calibration_cps"))
    
    case_num_NSGAII = list(range(1, 6))
    case_num_CMAES = list(range(6, 9))
    
    cp_cases = {}
    
    for case_n in range(1, case_num+1):
        case_name = f"case{case_n}"
        suffix = "_40" if case_n != 5 else "_90"
        case_cp_fp = os.path.join(cps_home, case_name+suffix, f"calibrate_cp_case{case_n}.pkl")

        if case_n in case_num_NSGAII:
            with open(case_cp_fp, "rb") as f:
                state = pickle.load(f)
                history = state["history"]
                history = history[:max_num]
            
            all_pop = []
            all_combined_pop = []
            all_first_front = []
            all_best_ind = []
            all_best_ind_fitness = []
            
            for h in history:
                pop = h["population"]
                combined_pop = h["combined_population"]
                first_front = h["first_front"]
                first_front_fitness = [ind.fitness for ind in first_front]
                
                best_ind_index = np.argmax([fitness.values[-1] for fitness in first_front_fitness])  # best_ind, based on the fitness[-1] (shiquan max)
                best_ind = first_front[best_ind_index]
                best_ind_fitness = best_ind.fitness

                all_pop.append(pop)
                all_combined_pop.append(combined_pop)
                all_first_front.append(first_front)
                all_best_ind.append(best_ind)
                all_best_ind_fitness.append(best_ind_fitness.values[-1])
            
            first_num_best_ind_index = sorted(range(len(all_best_ind_fitness)), key=lambda i: all_best_ind_fitness[i], reverse=True)[:ind_num]
            
            cp_cases[case_name] = {
                "all_pop": all_pop,
                "all_combined_pop": all_combined_pop,
                "all_first_front": all_first_front,
                "all_best_ind": all_best_ind, 
                "first_num_best_ind_index": first_num_best_ind_index
            }
            
        elif case_n in case_num_CMAES:
            with open(case_cp_fp, "rb") as f:
                state = pickle.load(f)
                history = state["history"]
                history = history[:max_num]
                
            all_pop = []
            all_best_ind = []
            all_best_ind = []
            all_best_ind_fitness = []

            for h in history:
                pop = h["population"]
                best_ind = h["best_ind"]

                pop_fitness = [ind.fitness.values[0] for ind in pop]
                best_ind_fitness = best_ind.fitness.values[0]
                
                all_pop.append(pop)
                all_best_ind.append(best_ind)
                all_best_ind.append(best_ind)
                all_best_ind_fitness.append(best_ind_fitness)

            first_num_best_ind_index = sorted(range(len(all_best_ind_fitness)), key=lambda i: all_best_ind_fitness[i], reverse=True)[:ind_num]

            cp_cases[case_name] = {
                "all_pop": all_pop,
                "all_best_ind": all_best_ind,
                "first_num_best_ind_index": first_num_best_ind_index,
            }
    
    return cp_cases
    

def plot_performance_table(ind_num=5, case_num=8):
    # get obs
    obs_cali = get_obs(evb_dir_modeling, get_type="calibration", station_names=station_names)
    obs_vali = get_obs(evb_dir_modeling, get_type="validation", station_names=station_names)
    
    performance_table_cali_arrays = {}
    performance_table_vali_arrays = {}
    for num in range(ind_num):
        performance_table_cali_array = np.zeros((case_num, len(station_names)))
        performance_table_vali_array = np.zeros((case_num, len(station_names)))
        performance_table_cali_arrays[num] = performance_table_cali_array
        performance_table_vali_arrays[num] = performance_table_vali_array
    
    # loop for cases
    for case_n in range(1, case_num+1):
        case_name = f"case{case_n}"
        
        # get sim
        sim = get_sim_rvic(evb_dir_modeling, case_name, ind_num, station_names=station_names)
        
        # evaluate
        for name in station_names:
            for num in range(ind_num):
                sim_ind = sim[num]

                sim_cali_streamflow = sim_ind["cali"][f"streamflow(m3/s)_{name}"].filled(0)
                sim_vali_streamflow = sim_ind["vali"][f"streamflow(m3/s)_{name}"].filled(0)

                obs_cali_streamflow = obs_cali[f"streamflow(m3/s)_{name}"].values.flatten()
                obs_vali_streamflow = obs_vali[f"streamflow(m3/s)_{name}"].values.flatten()
                
                KGE_m_cali = EvaluationMetric(sim_cali_streamflow, obs_cali_streamflow).KGE_m()
                KGE_m_vali = EvaluationMetric(sim_vali_streamflow, obs_vali_streamflow).KGE_m()
                
                performance_table_cali_arrays[num][case_n-1, station_names.index(name)] = KGE_m_cali
                performance_table_vali_arrays[num][case_n-1, station_names.index(name)] = KGE_m_vali
    
    # arrays combine
    performance_table_cali_arrays_combined = [performance_table_cali_arrays[key] for key in performance_table_cali_arrays.keys()]
    performance_table_vali_arrays_combined = [performance_table_vali_arrays[key] for key in performance_table_vali_arrays.keys()]
    
    # save
    performance_table_cali_mean = pd.DataFrame(np.mean(performance_table_cali_arrays_combined, axis=0), index=[f"case{n+1}" for n in range(case_num)], columns=station_names)
    performance_table_vali_mean = pd.DataFrame(np.mean(performance_table_vali_arrays_combined, axis=0), index=[f"case{n+1}" for n in range(case_num)], columns=station_names)
    
    performance_table_cali_max = pd.DataFrame(np.max(performance_table_cali_arrays_combined, axis=0), index=[f"case{n+1}" for n in range(case_num)], columns=station_names)
    performance_table_vali_max = pd.DataFrame(np.max(performance_table_vali_arrays_combined, axis=0), index=[f"case{n+1}" for n in range(case_num)], columns=station_names)
    
    performance_table_cali_min = pd.DataFrame(np.min(performance_table_cali_arrays_combined, axis=0), index=[f"case{n+1}" for n in range(case_num)], columns=station_names)
    performance_table_vali_min = pd.DataFrame(np.min(performance_table_vali_arrays_combined, axis=0), index=[f"case{n+1}" for n in range(case_num)], columns=station_names)
    
    output_path = os.path.join(evb_dir_modeling.CalibrateVIC_dir, "performance_tables.xlsx")
    output_format_path = os.path.join(evb_dir_modeling.CalibrateVIC_dir, "performance_tables_format.xlsx")

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        performance_table_cali_mean.to_excel(writer, sheet_name='cali_mean')
        performance_table_vali_mean.to_excel(writer, sheet_name='vali_mean')
        performance_table_cali_max.to_excel(writer, sheet_name='cali_max')
        performance_table_vali_max.to_excel(writer, sheet_name='vali_max')
        performance_table_cali_min.to_excel(writer, sheet_name='cali_min')
        performance_table_vali_min.to_excel(writer, sheet_name='vali_min')


def plot_taylor(ind_num=5, case_num=8):    
    # get obs
    obs_cali = get_obs(evb_dir_modeling, get_type="calibration", station_names=station_names)
    obs_vali = get_obs(evb_dir_modeling, get_type="validation", station_names=station_names)

    for name in station_names:
        obs_cali_station = obs_cali[f"streamflow(m3/s)_{name}"].values.flatten()
        obs_vali_station = obs_vali[f"streamflow(m3/s)_{name}"].values.flatten()

        models_cali = []
        models_vali = []
        models_names = []
        # loop for cases
        for case_n in range(1, case_num+1):
            case_name = f"case{case_n}"
            
            sim = get_sim_rvic(evb_dir_modeling, case_name, ind_num, station_names=station_names)
            
            sim_cali_streamflow_all_ind = []
            sim_vali_streamflow_all_ind = []
            for num in range(ind_num):
                sim_ind = sim[num]
                
                sim_cali_streamflow = sim_ind["cali"][f"streamflow(m3/s)_{name}"].filled(0)
                sim_vali_streamflow = sim_ind["vali"][f"streamflow(m3/s)_{name}"].filled(0)
                
                sim_cali_streamflow_all_ind.append(sim_cali_streamflow)
                sim_vali_streamflow_all_ind.append(sim_vali_streamflow)
            
            sim_cali_streamflow_all_ind = np.mean(np.array(sim_cali_streamflow_all_ind), axis=0)
            sim_vali_streamflow_all_ind = np.mean(np.array(sim_vali_streamflow_all_ind), axis=0)
            
            models_cali.append(sim_cali_streamflow_all_ind)
            models_vali.append(sim_vali_streamflow_all_ind)
            models_names.append(case_n)
        
        # general set
        cali_names_ha = ["left", "right", "left", "left", "left", "left", "left", "left"]
        cali_names_va = ["bottom", "top", "bottom", "top", "top", "bottom", "top", "top"]
        verify_names_ha = ["left", "right", "left", "left", "left", "left", "left", "left"]
        verify_names_va = ["bottom", "top", "bottom", "bottom", "bottom", "bottom", "top", "top"]
        models_colors = [
            "#1f77b4",  # muted blue
            "#ff7f0e",  # orange
            "#2ca02c",  # green
            "#d62728",  # red
            "#9467bd",  # purple
            "#8c564b",  # brown
            "#e377c2",  # pink
            "#7f7f7f",  # gray
        ]
        model_markers = ["o", "o", "o", "o", "o", "^", "^", "^"]
        
        # plot
        fig_taylor = plt.figure(figsize=(12, 6))
        fig_taylor.subplots_adjust(left=0.08, right=0.92, bottom=0.01, top=0.9, wspace=0.3)
        ax1 = fig_taylor.add_subplot(121, projection='polar')
        ax2 = fig_taylor.add_subplot(122, projection='polar')
        
        fig_taylor, ax1 = taylor_diagram(
            obs_cali_station, models_cali, models_names,
            cali_names_ha, cali_names_va,
            model_colors=models_colors, model_markers=model_markers,
            title="(a) Calibration",
            fig=fig_taylor,
            ax=ax1,
            add_text=False
        )
        fig_taylor, ax2 = taylor_diagram(
            obs_vali_station, models_vali, models_names,
            verify_names_ha, verify_names_va,
            model_colors=models_colors, model_markers=model_markers,
            title="(b) Verification",
            fig=fig_taylor,
            ax=ax2,
            add_text=False
        )
    

def plot_metrics_evaluation(ind_num=5, case_num=8):
    # get all
    obs_cali_all_stations, obs_vali_all_stations, sim_cali_all_stations, sim_vali_all_stations = get_obs_sim_all_inds(ind_num, case_num)
    
    records = []
    for name in station_names:
        obs_cali_station = obs_cali_all_stations[name]
        obs_vali_station = obs_vali_all_stations[name]
        sim_cali_station = sim_cali_all_stations[name]
        sim_vali_station = sim_vali_all_stations[name]
        
        for case_n in range(1, case_num+1):
            case_name = f"case{case_n}"
            sim_cali_station_case = sim_cali_station[case_name]
            sim_vali_station_case = sim_vali_station[case_name]
            
            # metrics
            em_cali = EvaluationMetric(sim_cali_station_case, obs_cali_station)
            em_vali = EvaluationMetric(sim_vali_station_case, obs_vali_station)
            
            sm_cali = SignatureEvaluationMetric(sim_cali_station_case, obs_cali_station)
            sm_vali = SignatureEvaluationMetric(sim_vali_station_case, obs_vali_station)
            
            # cali
            records.append({
                "station": name,
                "case": case_name,
                "stage": "cali",
                "metric": "PBias",
                "value": em_cali.PBias(),
            })
            
            records.append({
                "station": name,
                "case": case_name,
                "stage": "cali",
                "metric": "BiasFHV",
                "value": sm_cali.BiasFHV(q_high=0.02),
            })
            
            records.append({
                "station": name,
                "case": case_name,
                "stage": "cali",
                "metric": "BiasFHV_1",
                "value": sm_cali.BiasFHV(q_high=0.01),
            })
            
            records.append({
                "station": name,
                "case": case_name,
                "stage": "cali",
                "metric": "BiasFLV",
                "value": sm_cali.BiasFLV(q_low=0.7),
            })
            
            records.append({
                "station": name,
                "case": case_name,
                "stage": "cali",
                "metric": "BiasFMS",
                "value": sm_cali.BiasFMS(p1=0.2, p2=0.7),
            })
            
            records.append({
                "station": name,
                "case": case_name,
                "stage": "cali",
                "metric": "BiasFMM",
                "value": sm_cali.BiasFMM(),
            })
            
            records.append({
                "station": name,
                "case": case_name,
                "stage": "cali",
                "metric": "fdc",
                "value": sm_cali.get_fdc(),
            })
            
            records.append({
                "station": name,
                "case": case_name,
                "stage": "cali",
                "metric": "KGE_m_r",
                "value": em_cali.KGE_m(components=True)[1],
            })
                        
            records.append({
                "station": name,
                "case": case_name,
                "stage": "cali",
                "metric": "KGE_m_beta",
                "value": em_cali.KGE_m(components=True)[2],
            })
            
            records.append({
                "station": name,
                "case": case_name,
                "stage": "cali",
                "metric": "KGE_m_gamma",
                "value": em_cali.KGE_m(components=True)[3],
            })
            
            # vali
            records.append({
                "station": name,
                "case": case_name,
                "stage": "vali",
                "metric": "PBias",
                "value": em_vali.PBias(),
            })
            
            records.append({
                "station": name,
                "case": case_name,
                "stage": "vali",
                "metric": "BiasFHV",
                "value": sm_vali.BiasFHV(q_high=0.02),
            })
            
            records.append({
                "station": name,
                "case": case_name,
                "stage": "vali",
                "metric": "BiasFHV_1",
                "value": sm_vali.BiasFHV(q_high=0.01),
            })
            
            records.append({
                "station": name,
                "case": case_name,
                "stage": "vali",
                "metric": "BiasFLV",
                "value": sm_vali.BiasFLV(q_low=0.7),
            })
            
            records.append({
                "station": name,
                "case": case_name,
                "stage": "vali",
                "metric": "BiasFMS",
                "value": sm_vali.BiasFMS(p1=0.2, p2=0.7),
            })
            
            records.append({
                "station": name,
                "case": case_name,
                "stage": "vali",
                "metric": "BiasFMM",
                "value": sm_vali.BiasFMM(),
            })
            
            records.append({
                "station": name,
                "case": case_name,
                "stage": "vali",
                "metric": "fdc",
                "value": sm_vali.get_fdc(),
            })
            
            records.append({
                "station": name,
                "case": case_name,
                "stage": "vali",
                "metric": "KGE_m_r",
                "value": em_cali.KGE_m(components=True)[1],
            })
                        
            records.append({
                "station": name,
                "case": case_name,
                "stage": "vali",
                "metric": "KGE_m_beta",
                "value": em_cali.KGE_m(components=True)[2],
            })
            
            records.append({
                "station": name,
                "case": case_name,
                "stage": "vali",
                "metric": "KGE_m_gamma",
                "value": em_cali.KGE_m(components=True)[3],
            })
    
    df_metrics = pd.DataFrame.from_records(records)
    
    # save
    output_path = os.path.join(evb_dir_modeling.CalibrateVIC_dir, "metrics_tables.xlsx")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_metrics.to_excel(writer)
    
    # df_metrics.groupby(["case", "stage", "metric"])["value"].median()
    # df_pbias = df_metrics.query("metric == 'PBias' & stage == 'vali'")
    # df_metrics.groupby(["case", "stage", "metric"])["value"].mean()
    
    return df_metrics


def plot_fdc(ind_num=5, case_num=8):
    # get df_metrics
    df_metrics = plot_metrics_evaluation(ind_num=ind_num, case_num=case_num)
    
    # plot fdc: case1, 5, 7, 8
    df_fdc = df_metrics.query("station == 'shiquan' & stage == 'vali' & metric == 'fdc'")
    p_obs = df_fdc.iloc[0, -1]["obs"]['p']
    q_obs = df_fdc.iloc[0, -1]["obs"]['q']
    
    fig, ax = plt.subplots(gridspec_kw={"left": 0.13, "right": 0.95, "bottom": 0.11, "top": 0.95, "wspace": 0.2})
    ax.plot(p_obs, q_obs, color="gray", label="Observed", alpha=0.5, linewidth=3)
    case_names = ["case1", "case5", "case7", "case8"]
    case_names_plots = ["Case 1", "Case 5", "Case 7", "Case 8"]
    # case_colors = ["red", "blue", "red", "blue"]
    case_colors = ["red", "blue", "darkorange", "m"]
    case_linestyles = ["-", "-", "--", "--"]
    
    for case_name, case_name_plot, case_color, case_linestyle in zip(case_names, case_names_plots, case_colors, case_linestyles):
        row = df_fdc.loc[df_fdc["case"] == case_name, "value"].iloc[0]
        p_sim = row["sim"]["p"]
        q_sim = row["sim"]["q"]
        ax.plot(p_sim, q_sim, color=case_color, label=case_name_plot, linewidth=1, linestyle=case_linestyle)
        
    xs = [0.02, 0.2, 0.7]
    for x in xs:
        ax.axvline(x, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.text(
            x+0.01,
            plt.ylim()[1],
            f"{x:.2f}",
            rotation=90,
            va="top",
            ha="left",
            fontsize=9,
            color="gray",
            weight="bold",
        )
        
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_ylabel("Streamflow (m$^3$/s)")
    ax.set_xlabel("Flow exceedance probability [-]")
    
    plt.legend(loc='upper right')
    fig.savefig(os.path.join(evb_dir_modeling.CalibrateVIC_dir, "case_40_best_inds", "Exp1_case15_case78_compare_fdc.tiff"), dpi=300)
    plt.show(block=True)
    
    
    # # sub fig
    # fig, ax = plt.subplots(gridspec_kw={"left": 0.13, "right": 0.95, "bottom": 0.11, "top": 0.95, "wspace": 0.2})
    # ax.plot(p_obs, q_obs, color="gray", label="Observed", alpha=0.5, linewidth=3)
    # case_names = ["case1", "case5", "case7", "case8"]
    # case_names_plots = ["Case 1", "Case 5", "Case 7", "Case 8"]
    # # case_colors = ["red", "blue", "red", "blue"]
    # case_colors = ["red", "blue", "darkorange", "m"]
    # case_linestyles = ["-", "-", "--", "--"]
    
    # for case_name, case_name_plot, case_color, case_linestyle in zip(case_names, case_names_plots, case_colors, case_linestyles):
    #     row = df_fdc.loc[df_fdc["case"] == case_name, "value"].iloc[0]
    #     p_sim = row["sim"]["p"]
    #     q_sim = row["sim"]["q"]
    #     ax.plot(p_sim, q_sim, color=case_color, label=case_name_plot, linewidth=1, linestyle=case_linestyle)
        
    # xs = [0.02, 0.2, 0.7]
    # for x in xs:
    #     ax.axvline(x, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        
    # ax.set_yscale("log")
    # ax.set_xlim(0, 1)
    # ax.set_ylabel("Streamflow (m$^3$/s)")
    # ax.set_xlabel("Flow exceedance probability [-]")
    # plt.show(block=True)
    
    
def plot_scatter_comparison(ind_num=5, case_num=8):
    # get all
    obs_cali_all_stations, obs_vali_all_stations, sim_cali_all_stations, sim_vali_all_stations = get_obs_sim_all_inds(ind_num, case_num)
    
    # plot shiquan station
    obs_cali_shiquan = obs_cali_all_stations["shiquan"]
    obs_vali_shiquan = obs_vali_all_stations["shiquan"]
    sim_cali_shiquan = sim_cali_all_stations["shiquan"]
    sim_vali_shiquan = sim_vali_all_stations["shiquan"]
    
    # get cases
    sim_cali_shiquan_case1 = sim_cali_shiquan["case1"]
    sim_cali_shiquan_case5 = sim_cali_shiquan["case5"]
    sim_cali_shiquan_case7 = sim_cali_shiquan["case7"]
    sim_cali_shiquan_case8 = sim_cali_shiquan["case8"]
    
    sim_vali_shiquan_case1 = sim_vali_shiquan["case1"]
    sim_vali_shiquan_case5 = sim_vali_shiquan["case5"]
    sim_vali_shiquan_case7 = sim_vali_shiquan["case7"]
    sim_vali_shiquan_case8 = sim_vali_shiquan["case8"]
    
    obs_total = obs_vali_shiquan
    
    model_names_left = ["Case 1", "Case 5"]
    models_total_left = [sim_vali_shiquan_case1, sim_vali_shiquan_case5]
    
    model_names_right = ["Case 7", "Case 8"]
    models_total_right = [sim_vali_shiquan_case7, sim_vali_shiquan_case8]
    model_colors = ["red", "navy"]
    
    # model_names = ["case1", "case5", "case7", "case8"]
    # models_total = [sim_vali_shiquan_case1, sim_vali_shiquan_case5, sim_vali_shiquan_case7, sim_vali_shiquan_case8]
    # model_colors = ["red", "green", "blue", "purple"]
    
    # plot
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={"left": 0.1, "right": 0.95, "bottom": 0.15, "top": 0.9, "wspace": 0.2})
    
    # set lim
    xylim_total_left = (min(np.min(obs_total), min([min(model) for model in models_total_left])), max(np.max(obs_total), max([max(model) for model in models_total_left])))
    xylim_total_right = (min(np.min(obs_total), min([min(model) for model in models_total_right])), max(np.max(obs_total), max([max(model) for model in models_total_right])))
    axes[0].set_xlim(xylim_total_left)
    axes[0].set_ylim(xylim_total_left)
    axes[1].set_xlim(xylim_total_right)
    axes[1].set_ylim(xylim_total_right)
    
    axes[0].plot(np.arange(axes[0].get_xlim()[0], axes[0].get_xlim()[1], 1), np.arange(axes[0].get_xlim()[0], axes[0].get_xlim()[1], 1), "grey", alpha=0.5, linestyle="--", linewidth=1)
    axes[1].plot(np.arange(axes[1].get_xlim()[0], axes[1].get_xlim()[1], 1), np.arange(axes[1].get_xlim()[0], axes[1].get_xlim()[1], 1), "grey", alpha=0.5, linestyle="--", linewidth=1)
    
    alpha = 0.8
    linewidth = 1
    s = 10
    
    for i, (model_total, model_name, model_color) in enumerate(zip(models_total_left, model_names_left, model_colors)):        
        axes[0].scatter(obs_total, model_total, facecolors='none', edgecolor=model_color, s=s, linewidth=linewidth, label=None, alpha=alpha, zorder=2)

        p_total = np.polyfit(obs_total, model_total, deg=1, rcond=None, full=False, w=None, cov=False)
        axes[0].plot(np.arange(axes[0].get_xlim()[0], axes[0].get_xlim()[1], 1), np.polyval(p_total, np.arange(axes[0].get_xlim()[0], axes[0].get_xlim()[1], 1)), color=model_color, linestyle="-", linewidth=1, label=f"{model_name}: y = {p_total[0]:.2f}x + {p_total[1]:.2f}", zorder=3)
    
    for i, (model_total, model_name, model_color) in enumerate(zip(models_total_right, model_names_right, model_colors)):        
        axes[1].scatter(obs_total, model_total, facecolors='none', edgecolor=model_color, s=s, linewidth=linewidth, label=None, alpha=alpha, zorder=2)

        p_total = np.polyfit(obs_total, model_total, deg=1, rcond=None, full=False, w=None, cov=False)
        axes[1].plot(np.arange(axes[0].get_xlim()[0], axes[0].get_xlim()[1], 1), np.polyval(p_total, np.arange(axes[0].get_xlim()[0], axes[0].get_xlim()[1], 1)), color=model_color, linestyle="-", linewidth=1, label=f"{model_name}: y = {p_total[0]:.2f}x + {p_total[1]:.2f}", zorder=3)
        
    axes[0].set_ylabel("Simulated streamflow (m$^3$/s)")
    [ax.set_xlabel("Observed streamflow (m$^3$/s)") for ax in axes]  
    
    axes[0].set_title("Multi-gauge Calibration")
    axes[1].set_title("Single-gauge Calibration")
    
    axes[0].legend(loc="upper right", prop={'size': 10, 'family': 'Arial'})
    axes[1].legend(loc="upper right", prop={'size': 10, 'family': 'Arial'})
    
    axes[0].annotate("(a)", xy=(0.02, 0.9), xycoords='axes fraction', fontsize=14, fontweight='bold')
    axes[1].annotate("(b)", xy=(0.02, 0.9), xycoords='axes fraction', fontsize=14, fontweight='bold')
    
    fig.savefig(os.path.join(evb_dir_modeling.CalibrateVIC_dir, "case_40_best_inds", "Exp1_case15_case78_compare_scatter.tiff"), dpi=300)
    plt.show(block=True)


def plot_water_balance_comparison_pie(ind_num=5, case_num=8):
    # loop for cases
    sim_all_cases = {}
    for case_n in range(1, case_num+1):
        case_name = f"case{case_n}"
        
        # get sim
        sim = get_sim_vic(evb_dir_modeling, case_name, ind_num)
        sim_all_cases[case_name] = sim
    
    period = "vali"
    
    # plot pie_plot
    cases_names = ["case1", "case5", "case7", "case8"]
    cases_names_plot = ["Case 1", "Case 5", "Case 7", "Case 8"]
    ratio_list_cases = {}
    
    for case_name in cases_names:
        sim_case = sim_all_cases[case_name]
        
        # OUT_BASEFLOW
        sim_case_baseflow = np.array([sim_case[key][period]["OUT_BASEFLOW"] for key in sim_case.keys()])  # [inds_num, time, lat, lon]
        fill_value_ = sim_case[list(sim_case.keys())[0]][period]["OUT_BASEFLOW"].fill_value
        sim_case_baseflow[sim_case_baseflow == fill_value_] = np.nan
        sim_case_baseflow = np.nanmean(sim_case_baseflow, axis=0)  # [time, lat, lon]
        sim_case_baseflow = np.nanmean(sim_case_baseflow, axis=(1, 2))  # [time, ]
        sim_case_baseflow_det = (sim_case_baseflow[1:] + sim_case_baseflow[:-1]) / 2
        sim_case_baseflow_det_sum = np.sum(sim_case_baseflow_det)
        
        # OUT_RUNOFF
        sim_case_runoff = np.array([sim_case[key][period]["OUT_RUNOFF"] for key in sim_case.keys()])  # [inds_num, time, lat, lon]
        fill_value_ = sim_case[list(sim_case.keys())[0]][period]["OUT_RUNOFF"].fill_value
        sim_case_runoff[sim_case_runoff == fill_value_] = np.nan
        sim_case_runoff = np.nanmean(sim_case_runoff, axis=0)  # [time, lat, lon]
        sim_case_runoff = np.nanmean(sim_case_runoff, axis=(1, 2))  # [time, ]
        sim_case_runoff_det = (sim_case_runoff[1:] + sim_case_runoff[:-1]) / 2
        sim_case_runoff_det_sum = np.sum(sim_case_runoff_det)
        
        # OUT_EVAP
        sim_case_evap = np.array([sim_case[key][period]["OUT_EVAP"] for key in sim_case.keys()])  # [inds_num, time, lat, lon]
        fill_value_ = sim_case[list(sim_case.keys())[0]][period]["OUT_EVAP"].fill_value
        sim_case_evap[sim_case_evap == fill_value_] = np.nan
        sim_case_evap = np.nanmean(sim_case_evap, axis=0)  # [time, lat, lon]
        sim_case_evap = np.nanmean(sim_case_evap, axis=(1, 2))  # [time, ]
        sim_case_evap_det = (sim_case_evap[1:] + sim_case_evap[:-1]) / 2
        sim_case_evap_det_sum = np.sum(sim_case_evap_det)
        
        # OUT_SOIL_MOIST
        sim_case_sm = np.array([sim_case[key][period]["OUT_SOIL_MOIST"] for key in sim_case.keys()])  # [inds_num, time, layers, lat, lon]
        fill_value_ = sim_case[list(sim_case.keys())[0]][period]["OUT_SOIL_MOIST"].fill_value
        sim_case_sm[sim_case_sm == fill_value_] = np.nan
        sim_case_sm = np.nanmean(sim_case_sm, axis=0)  # [time, layers, lat, lon]
        sim_case_sm = np.nansum(sim_case_sm, axis=1)  # [time, lat, lon]
        sim_case_sm = np.nanmean(sim_case_sm, axis=(1, 2))  # [time, ]
        sim_case_sm_det = sim_case_sm[1:] - sim_case_sm[:-1]
        sim_case_sm_det_sum = np.sum(np.abs(sim_case_sm_det))
        
        sum_all = sim_case_baseflow_det_sum + sim_case_runoff_det_sum + sim_case_evap_det_sum + sim_case_sm_det_sum
        
        ratio_list = {
            "runoff": sim_case_runoff_det_sum/sum_all,
            "baseflow": sim_case_baseflow_det_sum/sum_all,
            "evap": sim_case_evap_det_sum/sum_all,
            "sm": sim_case_sm_det_sum/sum_all,
        }
        
        ratio_list_cases[case_name] = ratio_list
        # plt.plot(sim_case_baseflow, label="baseflow")
        # plt.plot(sim_case_runoff, label="runoff")
        # plt.plot(sim_case_evap, label="evap")
        # plt.legend()
        # plt.show(block=True)
        
        # plt.plot(sim_case_baseflow_det, label="baseflow_det")
        # plt.plot(sim_case_runoff_det, label="runoff_det")
        # plt.plot(sim_case_evap_det, label="evap_det")
        # plt.plot(sim_case_sm_det, label="sm_det")
        # plt.legend()
        # plt.show(block=True)
    
    # plot pie
    color_map = {
        "runoff":   "#3B5BA5", #"#2F4FB3", #"#3B5BA5", # "#1736FF",
        "baseflow": "#9C0C0A",
        "evap":     "#63BD00",
        "sm":       "gray", # "#6E858C",
    }
    
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    
    for i, case_name in enumerate(cases_names):
        ratio = ratio_list_cases[case_name]
        
        labels = list(ratio.keys())
        sizes = list(ratio.values())
        colors = [color_map[label] for label in labels]
        axes[i].pie(
            sizes,
            autopct="%.1f%%",
            startangle=90,
            colors=colors,
            wedgeprops=dict(
                edgecolor="k",
                linewidth=1.2
            ),
            textprops=dict(
                fontsize=15
            )
        )
        axes[i].set_title(cases_names_plot[i], fontsize=18)
        axes[i].axis("equal")
    
    plt.tight_layout()
    fig.savefig(os.path.join(evb_dir_modeling.CalibrateVIC_dir, "case_40_best_inds", "Exp1_case15_case78_compare_pieplot.tiff"), dpi=300)
    plt.show(block=True)
    

def plot_spatial_pattern(ind_num=5, case_num=8):
    # loop for cases
    sim_all_cases = {}
    for case_n in range(1, case_num+1):
        case_name = f"case{case_n}"
        
        # get sim
        sim = get_sim_vic(evb_dir_modeling, case_name, ind_num)
        sim_all_cases[case_name] = sim
    
    period = "vali"
    
    # plot spatial
    cases_names_plot = ["case1", "case5", "case7", "case8"]
    sim_cases_baseflow = {}
    sim_cases_runoff = {}
    sim_cases_evap = {}
    sim_cases_sm = {}
    
    for case_name in cases_names_plot:
        sim_case = sim_all_cases[case_name]
        
        # OUT_BASEFLOW
        sim_case_baseflow = np.array([sim_case[key][period]["OUT_BASEFLOW"] for key in sim_case.keys()])  # [inds_num, time, lat, lon]
        fill_value_ = sim_case[list(sim_case.keys())[0]][period]["OUT_BASEFLOW"].fill_value
        sim_case_baseflow[sim_case_baseflow == fill_value_] = np.nan
        sim_case_baseflow = np.nanmean(sim_case_baseflow, axis=0)  # [time, lat, lon]
        sim_cases_baseflow[case_name] = sim_case_baseflow
        
        # OUT_RUNOFF
        sim_case_runoff = np.array([sim_case[key][period]["OUT_RUNOFF"] for key in sim_case.keys()])  # [inds_num, time, lat, lon]
        fill_value_ = sim_case[list(sim_case.keys())[0]][period]["OUT_RUNOFF"].fill_value
        sim_case_runoff[sim_case_runoff == fill_value_] = np.nan
        sim_case_runoff = np.nanmean(sim_case_runoff, axis=0)  # [time, lat, lon]
        sim_cases_runoff[case_name] = sim_case_runoff
        
        # OUT_EVAP
        sim_case_evap = np.array([sim_case[key][period]["OUT_EVAP"] for key in sim_case.keys()])  # [inds_num, time, lat, lon]
        fill_value_ = sim_case[list(sim_case.keys())[0]][period]["OUT_EVAP"].fill_value
        sim_case_evap[sim_case_evap == fill_value_] = np.nan
        sim_case_evap = np.nanmean(sim_case_evap, axis=0)  # [time, lat, lon]
        sim_cases_evap[case_name] = sim_case_evap
        
        # OUT_SOIL_MOIST
        sim_case_sm = np.array([sim_case[key][period]["OUT_SOIL_MOIST"] for key in sim_case.keys()])  # [inds_num, time, layers, lat, lon]
        fill_value_ = sim_case[list(sim_case.keys())[0]][period]["OUT_SOIL_MOIST"].fill_value
        sim_case_sm[sim_case_sm == fill_value_] = np.nan
        sim_case_sm = np.nanmean(sim_case_sm, axis=0)  # [time, layers, lat, lon]
        sim_cases_sm[case_name] = sim_case_sm
    
    # compare surface runoff and baseflow
    dpc_VIC_level1 = readdpc(evb_dir_modeling.dpc_VIC_level1_path, dataProcess_VIC_level1_HRB)
    grid_shp_level1 = dpc_VIC_level1.get_data_from_cache("grid_shp")[0]
    basin_shp_level1 = dpc_VIC_level1.get_data_from_cache("basin_shp")[0]
    stand_grids_lat, stand_grids_lon = createStand_grids_lat_lon_from_gridshp(
        grid_shp_level1, reverse_lat
    )
    
    xmin = np.min(stand_grids_lon)
    xmax = np.max(stand_grids_lon)
    ymin = np.min(stand_grids_lat)
    ymax = np.max(stand_grids_lat)
    extent = [xmin, xmax, ymin, ymax]
    
    cases_left = ["case5", "case1"]
    cases_right = ["case8", "case7"]

    sim_runoff_diff_left = sim_cases_runoff[cases_left[0]] - sim_cases_runoff[cases_left[1]]
    sim_runoff_diff_left = np.nanmean(sim_runoff_diff_left, axis=0)
    
    sim_runoff_diff_right = sim_cases_runoff[cases_right[0]] - sim_cases_runoff[cases_right[1]]
    sim_runoff_diff_right = np.nanmean(sim_runoff_diff_right, axis=0)
    
    sim_baseflow_diff_left = sim_cases_baseflow[cases_left[0]] - sim_cases_baseflow[cases_left[1]]
    sim_baseflow_diff_left = np.nanmean(sim_baseflow_diff_left, axis=0)
    
    sim_baseflow_diff_right = sim_cases_baseflow[cases_right[0]] - sim_cases_baseflow[cases_right[1]]
    sim_baseflow_diff_right = np.nanmean(sim_baseflow_diff_right, axis=0)
    
    # plot
    fig, axes = plt.subplots(2, 3, figsize=(9, 7), gridspec_kw={
        "width_ratios": [1, 1, 0.05],
        "wspace": 0.03,
        "hspace": 0.03,
        "left":0.08,
        "right": 0.9,
        "bottom": 0.05,
        "top": 0.93
    })
    
    # norm
    vmax_runoff = np.nanmax(
        np.abs([sim_runoff_diff_left, sim_runoff_diff_right])
    )
    
    norm_runoff = TwoSlopeNorm(
        vmin=-vmax_runoff,
        vcenter=0.0,
        vmax=vmax_runoff
    )
    
    vmax_baseflow = np.nanmax(
        np.abs([sim_baseflow_diff_left, sim_baseflow_diff_right])
    )
    
    norm_baseflow = TwoSlopeNorm(
        vmin=-vmax_baseflow,
        vcenter=0.0,
        vmax=vmax_baseflow
    )

    cmap = "RdBu"
    im_runoff_l = axes[0, 0].imshow(sim_runoff_diff_left, extent=extent, cmap=cmap, norm=norm_runoff)
    im_runoff_r = axes[0, 1].imshow(sim_runoff_diff_right, extent=extent, cmap=cmap, norm=norm_runoff)
    im_baseflow_l = axes[1, 0].imshow(sim_baseflow_diff_left, extent=extent, cmap=cmap, norm=norm_baseflow)
    im_baseflow_r = axes[1, 1].imshow(sim_baseflow_diff_right, extent=extent, cmap=cmap, norm=norm_baseflow)
    
    # set boundary and xyticks
    for ax in axes[:, :2].flatten():
        basin_shp_level1.plot(ax=ax, facecolor="none", edgecolor="k", linewidth=3.0)
        bbox = box(xmin, ymin, xmax, ymax)
        
        basin_fixed = basin_shp_level1.copy()
        basin_fixed["geometry"] = basin_fixed.geometry.buffer(0)
        basin_union = basin_fixed.geometry.unary_union
        outside = bbox.difference(basin_union)

        outside_gdf = gpd.GeoDataFrame(
            geometry=[outside],
            crs=basin_shp_level1.crs
        )

        outside_gdf.plot(
            ax=ax,
            facecolor="w",
            edgecolor="none",
            zorder=10
        )
        # ax.set_axisbelow(True)
        set_boundary(ax, [xmin, ymin, xmax, ymax])
        set_xyticks(ax, x_locator_interval=0.5, y_locator_interval=0.5, yticks_rotation=90)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_zorder(20)
    
    # annotation
    axes[0, 0].annotate("(a)", xy=(0.02, 0.9), xycoords='axes fraction', fontsize=15, fontweight='bold', zorder=15)
    axes[0, 1].annotate("(b)", xy=(0.02, 0.9), xycoords='axes fraction', fontsize=15, fontweight='bold', zorder=15)
    axes[1, 0].annotate("(c)", xy=(0.02, 0.9), xycoords='axes fraction', fontsize=15, fontweight='bold', zorder=15)
    axes[1, 1].annotate("(d)", xy=(0.02, 0.9), xycoords='axes fraction', fontsize=15, fontweight='bold', zorder=15)
    axes[0, 0].set_title("Case 5 - Case 1", fontsize=15, pad=10)
    axes[0, 1].set_title("Case 8 - Case 7", fontsize=15, pad=10)
    # axes[0, 0].set_ylabel("Surface runoff", fontsize=15, labelpad=10)
    # axes[1, 0].set_ylabel("Baseflow", fontsize=15, labelpad=10)
    
    # colorbar
    cbar_runoff = fig.colorbar(
        im_runoff_l,
        cax=axes[0, 2],
        orientation="vertical"
    )
    cbar_runoff.set_label("Surface runoff difference (mm)")

    cbar_baseflow = fig.colorbar(
        im_baseflow_l,
        cax=axes[1, 2],
        orientation="vertical"
    )
    cbar_baseflow.set_label("Baseflow difference (mm)")
    
    for ax in axes[:, 2]:
        ax.tick_params(labelsize=10)
    
    axes[0, 0].tick_params(axis="x", bottom=False, labelbottom=False)
    axes[0, 1].tick_params(axis="x", bottom=False, labelbottom=False)
    axes[0, 1].tick_params(axis="y", left=False, labelleft=False)
    axes[1, 1].tick_params(axis="y", left=False, labelleft=False)
    
    fig.savefig(os.path.join(evb_dir_modeling.CalibrateVIC_dir, "case_40_best_inds", "Exp1_case15_case78_compare_spatial_pattern.tiff"), dpi=300)
    plt.show(block=True)

def plot_parameter_identifiability_NSGAII_case1(cp_cases, case_name="case1"):
    cp_case = cp_cases[case_name]
    all_pop = cp_case["all_pop"]
    all_pop_flatten = [y for x in all_pop for y in x]
    all_combined_pop = cp_case["all_combined_pop"]
    all_combined_pop_flatten = [y for x in all_combined_pop for y in x]
    all_first_front = cp_case["all_first_front"]
    all_best_ind = cp_case["all_best_ind"]
    first_num_best_ind_index = cp_case["first_num_best_ind_index"]
    best_ind = all_best_ind[first_num_best_ind_index[0]]
    
    # param case
    # logger.info("case1: multi-gauges, calibrate VIC param (uniform), soil depths (uniform), RVIC params (uniform)")
    # params_case = deepcopy(params_minimal)
    # params_case["g_params"] = g_params_Nijssen_spatially_uniform_minimal
    # params_case["g_params"] = set_g_params_soilGrids_layer(params_case["g_params"])
    # pm = ParamManager(params_case)
    # free_param_names = pm.vector_names(get_free=True)
    free_param_names_plot = [
        r"$g_{10}$",
        r"$z_{1}$",
        r"$z_{2}$",
        r"$b$",
        r"$D_{1}$",
        r"$D_{2}$",
        r"$D_{3}$",
        r"$tp$",
        r"$\mu$",
        r"$m$",
        r"$v$",
        r"$D$"
    ]
    n_free_param = len(free_param_names_plot)
    
    fig, axes = plt.subplots(
        n_free_param, n_free_param,
        figsize=(15, 15),
        gridspec_kw={
            "wspace": 0.2,
            "hspace": 0.2,
            "left": 0.07,
            "right": 0.95,
            "bottom": 0.05,
            "top": 0.99,
        }
    )
    
    kde_mappable = None
    for i in range(n_free_param):
        for j in range(n_free_param):
            ax = axes[i, j]

            all_inds_j = [ind[j] for ind in all_pop_flatten]
            all_inds_i = [ind[i] for ind in all_pop_flatten]
            
            # void
            if j > i:
                ax.axis("off")
                continue

            # Histogram
            if i == j:
                ax.hist(
                    all_inds_i,
                    bins=30,
                    color="lightblue",
                    edgecolor="k",
                    density=True,
                    linewidth=0.5,
                )
                ax.axvline(best_ind[i], color="red", lw=1.5)
            
            # params
            else:
                x = all_inds_j
                y = all_inds_i
                
                bw = 1.5 if len(x) < 500 else 1.2
                xy = np.vstack([x, y])
                kde = gaussian_kde(xy, bw_method=bw)
                xmin, xmax = min(x), max(x)
                ymin, ymax = min(y), max(y)
                
                xx, yy = np.mgrid[
                    xmin:xmax:120j,
                    ymin:ymax:120j
                ]
                
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                zz = zz / zz.max()
                norm = Normalize(vmin=0, vmax=1)
                levels = np.linspace(0, 1, 11)

                cs = ax.contourf(
                    xx, yy, zz,
                    levels=levels,
                    norm=norm,
                    cmap="viridis",
                    alpha=0.9,
                    vmin=0,
                    vmax=1,
                )
                
                if kde_mappable is None:
                    kde_mappable = cs
                
                ax.scatter(
                    best_ind[j],
                    best_ind[i],
                    c="red",
                    edgecolors="w",
                    s=30,
                    marker="o",
                    zorder=10
                )

            # axis
            if i == n_free_param - 1:
                ax.set_xlabel(free_param_names_plot[j], fontsize=16)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=2))
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(free_param_names_plot[i], fontsize=16)
                ax.yaxis.set_label_coords(-0.64, 0.5)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            else:
                ax.set_yticklabels([])

            ax.tick_params(direction="in", labelsize=14)
    
    cbar = fig.colorbar(
        kde_mappable,
        ax=axes,
        fraction=0.025,
        pad=0.02
    )
    cbar.set_label("Normalized kernel density", fontsize=16)
    fig.savefig(os.path.join(evb_dir_modeling.CalibrateVIC_dir, "calibration_cps", f"Exp1_case15_case78_compare_parameter_identifiability_{case_name}.tiff"), dpi=300)
    plt.show(block=True)
    

def plot_parameter_identifiability_NSGAII_case5(cp_cases, case_name="case5"):
    cp_case = cp_cases[case_name]
    all_pop = cp_case["all_pop"]
    all_pop_flatten = [y for x in all_pop for y in x]
    all_combined_pop = cp_case["all_combined_pop"]
    all_combined_pop_flatten = [y for x in all_combined_pop for y in x]
    all_first_front = cp_case["all_first_front"]
    all_best_ind = cp_case["all_best_ind"]
    first_num_best_ind_index = cp_case["first_num_best_ind_index"]
    best_ind = all_best_ind[first_num_best_ind_index[0]]
    
    # param case
    # case5: multi-gauges, calibrate VIC param (distributed), soil depths (distributed), RVIC params (distributed)
    # logger.info("case5: multi-gauges, calibrate VIC param (distributed), soil depths (distributed), RVIC params (distributed)")
    # params_case = deepcopy(params_minimal)
    # params_case["g_params"] = set_g_params_soilGrids_layer(params_case["g_params"])
    # params_case["g_params"] = expand_station_wise_params(params_case["g_params"], station_num=len(station_names))
    # params_case["rvic_params"] = rvic_params_spatial
    # pm = ParamManager(params_case)
    # free_param_names = pm.vector_names(get_free=True)
    
    free_param_names_plot = [
        r"$g_{1}$",
        r"$g_{2}$",
        r"$g_{3}$",
        r"$g_{4}$",
        r"$g_{5}$",
        r"$g_{10,1}$",
        r"$g_{10,2}$",
        r"$g_{10,3}$",
        r"$g_{10,4}$",
        r"$g_{10,5}$",
        r"$z_{1,1}$",
        r"$z_{2,1}$",
        r"$z_{1,2}$",
        r"$z_{2,2}$",
        r"$z_{1,3}$",
        r"$z_{2,3}$",
        r"$z_{1,4}$",
        r"$z_{2,4}$",
        r"$z_{1,5}$",
        r"$z_{2,5}$",
        r"$tp$",
        r"$\mu$",
        r"$m$",
        r"$g_{6}$",
        r"$g_{7}$",
        r"$g_{8}$",
        r"$g_{9}$",
    ]
    
    n_free_param = len(free_param_names_plot)
    
    fig, axes = plt.subplots(
        n_free_param, n_free_param,
        # figsize=(2 * n_free_param, 2 * n_free_param),
        figsize=(22, 20),
        # sharex='col',
        # sharey='row'
        gridspec_kw={
            "wspace": 0.25,
            "hspace": 0.25,
            "left": 0.03,
            "right": 0.96,
            "bottom": 0.03,
            "top": 0.99,
        }
    )
    
    kde_mappable = None
    for i in range(n_free_param):
        for j in range(n_free_param):
            ax = axes[i, j]

            all_inds_j = [ind[j] for ind in all_pop_flatten]
            all_inds_i = [ind[i] for ind in all_pop_flatten]
            
            # void
            if j > i:
                ax.axis("off")
                continue

            # Histogram
            if i == j:
                ax.hist(
                    all_inds_i,
                    bins=30,
                    color="lightblue",
                    edgecolor="k",
                    density=True,
                    linewidth=0.5,
                )
                ax.axvline(best_ind[i], color="red", lw=1.5)
            
            # params
            else:
                x = all_inds_j
                y = all_inds_i
                
                bw = 1.5 if len(x) < 500 else 1.2
                xy = np.vstack([x, y])
                kde = gaussian_kde(xy, bw_method=bw)
                xmin, xmax = min(x), max(x)
                ymin, ymax = min(y), max(y)
                
                xx, yy = np.mgrid[
                    xmin:xmax:120j,
                    ymin:ymax:120j
                ]
                
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                zz = zz / zz.max()
                norm = Normalize(vmin=0, vmax=1)
                levels = np.linspace(0, 1, 11)

                cs = ax.contourf(
                    xx, yy, zz,
                    levels=levels,
                    norm=norm,
                    cmap="viridis",
                    alpha=0.9,
                    vmin=0,
                    vmax=1,
                )
                
                if kde_mappable is None:
                    kde_mappable = cs
                
                ax.scatter(
                    best_ind[j],
                    best_ind[i],
                    c="red",
                    edgecolors="w",
                    s=30,
                    marker="o",
                    zorder=10
                )

            # axis
            if i == n_free_param - 1:
                ax.set_xlabel(free_param_names_plot[j], fontsize=16)
                # ax.xaxis.set_major_locator(MaxNLocator(nbins=2))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=2, prune='both'))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(free_param_names_plot[i], fontsize=16)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
                ax.yaxis.set_label_coords(-0.8, 0.5)
            else:
                ax.set_yticklabels([])

            ax.tick_params(direction="in", labelsize=12)
    
    cbar = fig.colorbar(
        kde_mappable,
        ax=axes,
        fraction=0.025,
        pad=0.02
    )
    cbar.set_label("Normalized kernel density", fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    fig.savefig(os.path.join(evb_dir_modeling.CalibrateVIC_dir, "calibration_cps", f"Exp1_case15_case78_compare_parameter_identifiability_{case_name}.tiff"), dpi=300)
    plt.show(block=True)


def plot_parameter_identifiability_CMAES_case7(cp_cases, case_name="case7"):
    cp_case = cp_cases[case_name]
    all_pop = cp_case["all_pop"]
    all_pop_flatten = [y for x in all_pop for y in x]
    all_best_ind = cp_case["all_best_ind"]
    first_num_best_ind_index = cp_case["first_num_best_ind_index"]
    best_ind = all_best_ind[first_num_best_ind_index[0]]
    
    # param case
    # case7: single-gauge, calibrate VIC param (uniform), soil depths (uniform), RVIC params (uniform)
    # logger.info("case7: single-gauge, calibrate VIC param (uniform), soil depths (uniform), RVIC params (uniform)")
    # params_case = deepcopy(params_minimal)
    # params_case["g_params"] = g_params_Nijssen_spatially_uniform_minimal
    # params_case["g_params"] = set_g_params_soilGrids_layer(params_case["g_params"])
    # pm = ParamManager(params_case)
    # free_param_names = pm.vector_names(get_free=True)
    
    free_param_names_plot = [
        r"$g_{10}$",
        r"$z_{1}$",
        r"$z_{2}$",
        r"$b$",
        r"$D_{1}$",
        r"$D_{2}$",
        r"$D_{3}$",
        r"$tp$",
        r"$\mu$",
        r"$m$",
        r"$v$",
        r"$D$"
    ]
    
    n_free_param = len(free_param_names_plot)
    
    fig, axes = plt.subplots(
        n_free_param, n_free_param,
        figsize=(15, 15),
        gridspec_kw={
            "wspace": 0.2,
            "hspace": 0.2,
            "left": 0.07,
            "right": 0.95,
            "bottom": 0.05,
            "top": 0.99,
        }
    )
    
    kde_mappable = None
    for i in range(n_free_param):
        for j in range(n_free_param):
            ax = axes[i, j]

            all_inds_j = [ind[j] for ind in all_pop_flatten]
            all_inds_i = [ind[i] for ind in all_pop_flatten]
            
            # void
            if j > i:
                ax.axis("off")
                continue

            # Histogram
            if i == j:
                ax.hist(
                    all_inds_i,
                    bins=30,
                    color="lightblue",
                    edgecolor="k",
                    density=True,
                    linewidth=0.5,
                )
                ax.axvline(best_ind[i], color="red", lw=1.5)
            
            # params
            else:
                x = all_inds_j
                y = all_inds_i
                
                bw = 1.5 if len(x) < 500 else 1.2
                xy = np.vstack([x, y])
                kde = gaussian_kde(xy, bw_method=bw)
                xmin, xmax = min(x), max(x)
                ymin, ymax = min(y), max(y)
                
                xx, yy = np.mgrid[
                    xmin:xmax:120j,
                    ymin:ymax:120j
                ]
                
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                zz = zz / zz.max()
                norm = Normalize(vmin=0, vmax=1)
                levels = np.linspace(0, 1, 11)

                cs = ax.contourf(
                    xx, yy, zz,
                    levels=levels,
                    norm=norm,
                    cmap="viridis",
                    alpha=0.9,
                    vmin=0,
                    vmax=1,
                )
                
                if kde_mappable is None:
                    kde_mappable = cs
                
                ax.scatter(
                    best_ind[j],
                    best_ind[i],
                    c="red",
                    edgecolors="w",
                    s=30,
                    marker="o",
                    zorder=10
                )

            # axis
            if i == n_free_param - 1:
                ax.set_xlabel(free_param_names_plot[j], fontsize=16)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=2))
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(free_param_names_plot[i], fontsize=16)
                ax.yaxis.set_label_coords(-0.64, 0.5)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            else:
                ax.set_yticklabels([])

            ax.tick_params(direction="in", labelsize=14)
    
    cbar = fig.colorbar(
        kde_mappable,
        ax=axes,
        fraction=0.025,
        pad=0.02
    )
    cbar.set_label("Normalized kernel density", fontsize=14)
    fig.savefig(os.path.join(evb_dir_modeling.CalibrateVIC_dir, "calibration_cps", f"Exp1_case15_case78_compare_parameter_identifiability_{case_name}.tiff"), dpi=300)
    plt.show(block=True)
    
    
def plot_parameter_identifiability_CMAES_case8(cp_cases, case_name="case8"):
    cp_case = cp_cases[case_name]
    all_pop = cp_case["all_pop"]
    all_pop_flatten = [y for x in all_pop for y in x]
    all_best_ind = cp_case["all_best_ind"]
    first_num_best_ind_index = cp_case["first_num_best_ind_index"]
    best_ind = all_best_ind[first_num_best_ind_index[0]]
    
    # param case
    # case8: single-gauge, calibrate VIC param (distributed), soil depths (distributed), RVIC params (distributed)
    # logger.info("case8: single-gauge, calibrate VIC param (distributed), soil depths (distributed), RVIC params (distributed)")
    # case_name = "case8"
    # params_case = deepcopy(params_minimal)
    # params_case["g_params"] = set_g_params_soilGrids_layer(params_case["g_params"])
    # params_case["g_params"] = expand_station_wise_params(params_case["g_params"], station_num=len(station_names))
    # params_case["rvic_params"] = rvic_params_spatial
    # pm = ParamManager(params_case)
    # free_param_names = pm.vector_names(get_free=True)
    
    free_param_names_plot = [
        r"$g_{1}$",
        r"$g_{2}$",
        r"$g_{3}$",
        r"$g_{4}$",
        r"$g_{5}$",
        r"$g_{10,1}$",
        r"$g_{10,2}$",
        r"$g_{10,3}$",
        r"$g_{10,4}$",
        r"$g_{10,5}$",
        r"$z_{1,1}$",
        r"$z_{2,1}$",
        r"$z_{1,2}$",
        r"$z_{2,2}$",
        r"$z_{1,3}$",
        r"$z_{2,3}$",
        r"$z_{1,4}$",
        r"$z_{2,4}$",
        r"$z_{1,5}$",
        r"$z_{2,5}$",
        r"$tp$",
        r"$\mu$",
        r"$m$",
        r"$g_{6}$",
        r"$g_{7}$",
        r"$g_{8}$",
        r"$g_{9}$",
    ]
    
    n_free_param = len(free_param_names_plot)
    
    fig, axes = plt.subplots(
        n_free_param, n_free_param,
        # figsize=(2 * n_free_param, 2 * n_free_param),
        figsize=(22, 20),
        # sharex='col',
        # sharey='row'
        gridspec_kw={
            "wspace": 0.25,
            "hspace": 0.25,
            "left": 0.03,
            "right": 0.96,
            "bottom": 0.03,
            "top": 0.99,
        }
    )
    
    kde_mappable = None
    for i in range(n_free_param):
        for j in range(n_free_param):
            ax = axes[i, j]

            all_inds_j = [ind[j] for ind in all_pop_flatten]
            all_inds_i = [ind[i] for ind in all_pop_flatten]
            
            # void
            if j > i:
                ax.axis("off")
                continue

            # Histogram
            if i == j:
                ax.hist(
                    all_inds_i,
                    bins=30,
                    color="lightblue",
                    edgecolor="k",
                    density=True,
                    linewidth=0.5,
                )
                ax.axvline(best_ind[i], color="red", lw=1.5)
            
            # params
            else:
                x = all_inds_j
                y = all_inds_i
                
                bw = 1.5 if len(x) < 500 else 1.2
                xy = np.vstack([x, y])
                kde = gaussian_kde(xy, bw_method=bw)
                xmin, xmax = min(x), max(x)
                ymin, ymax = min(y), max(y)
                
                xx, yy = np.mgrid[
                    xmin:xmax:120j,
                    ymin:ymax:120j
                ]
                
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                zz = zz / zz.max()
                norm = Normalize(vmin=0, vmax=1)
                levels = np.linspace(0, 1, 11)

                cs = ax.contourf(
                    xx, yy, zz,
                    levels=levels,
                    norm=norm,
                    cmap="viridis",
                    alpha=0.9,
                    vmin=0,
                    vmax=1,
                )
                
                if kde_mappable is None:
                    kde_mappable = cs
                
                ax.scatter(
                    best_ind[j],
                    best_ind[i],
                    c="red",
                    edgecolors="w",
                    s=30,
                    marker="o",
                    zorder=10
                )

            # axis
            if i == n_free_param - 1:
                ax.set_xlabel(free_param_names_plot[j], fontsize=16)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=2, prune='both'))
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(free_param_names_plot[i], fontsize=16)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax.yaxis.set_label_coords(-0.8, 0.5)
            else:
                ax.set_yticklabels([])

            ax.tick_params(direction="in", labelsize=12)
    
    cbar = fig.colorbar(
        kde_mappable,
        ax=axes,
        fraction=0.025,
        pad=0.02
    )
    cbar.set_label("Normalized kernel density", fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    fig.savefig(os.path.join(evb_dir_modeling.CalibrateVIC_dir, "calibration_cps", f"Exp1_case15_case78_compare_parameter_identifiability_{case_name}.tiff"), dpi=300)
    plt.show(block=True)


def plot_parameter_identifiability():
    cp_cases = get_calibration_cps(evb_dir_modeling, case_num=8)
    
    plot_parameter_identifiability_NSGAII_case1(cp_cases)
    plot_parameter_identifiability_CMAES_case7(cp_cases)
    
    plot_parameter_identifiability_NSGAII_case5(cp_cases)
    plot_parameter_identifiability_CMAES_case8(cp_cases)


def plot_calibration_process(case_name="case1"):
    cp_cases = get_calibration_cps(evb_dir_modeling, case_num=8)
    
    # case1
    # case_name = "case1"
    cp_case = cp_cases[case_name]
    
    all_pop = cp_case["all_pop"]
    all_pop_flatten = [y for x in all_pop for y in x]
    all_pop_flatten_fitness = np.array([ind.fitness.values for ind in all_pop_flatten])
    all_pop_flatten_fitness_clean = all_pop_flatten_fitness[~np.any(all_pop_flatten_fitness == -9999.0, axis=1)]
    
    all_first_front = cp_case["all_first_front"]
    all_first_front_flatten = [y for x in all_first_front for y in x]
    all_first_front_flatten_fitness = np.array([ind.fitness.values for ind in all_first_front_flatten])
    all_first_front_flatten_fitness_clean = all_first_front_flatten_fitness[~np.any(all_first_front_flatten_fitness == -9999.0, axis=1)]
    
    all_vals_clean = np.vstack([all_pop_flatten_fitness_clean, all_first_front_flatten_fitness_clean])
    
    last_first_front = all_first_front[-1]
    last_first_front_fitness = np.array([ind.fitness.values for ind in last_first_front])
    last_first_front_fitness_clean = last_first_front_fitness[~np.any(last_first_front_fitness == -9999.0, axis=1)]
    
    first_first_front = all_first_front[0]
    first_first_front_fitness = np.array([ind.fitness.values for ind in first_first_front])
    first_first_front_fitness_clean = first_first_front_fitness[~np.any(first_first_front_fitness == -9999.0, axis=1)]
    
    all_best_ind = cp_case["all_best_ind"]
    
    # plot
    names_plot = ['Hanzhong', 'Yangxian', 'Youshui', 'Lianghekou', 'Shiquan']
    n_obj = len(names_plot)
    fig, axes = plt.subplots(nrows=n_obj, ncols=n_obj, figsize=(14, 8),
                        gridspec_kw={"wspace": 0.3, "hspace": 0.3,
                                    "left":0.08, "right": 0.98,
                                    "bottom": 0.08, "top": 0.98}
                        )
    for i in range(n_obj):
        for j in range(n_obj):
            ax = axes[i, j]
            
            # void
            if j > i:
                ax.axis("off")
                continue

            # Histogram
            if i == j:
                ax.hist(
                    all_pop_flatten_fitness_clean[:, i],
                    bins=30,
                    color="lightblue",
                    edgecolor="k",
                    density=True,
                    linewidth=0.5,
                )
                ax.set_xlim((0, 1))
                ax.xaxis.set_label_coords(0.5, -0.3)
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=2, prune='lower'))
                
                ax.yaxis.set_label_coords(-0.3, 0.5)
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
                
            else:
                ax.scatter(all_pop_flatten_fitness_clean[:, j], all_pop_flatten_fitness_clean[:, i], color='gray', alpha=0.5, s=2, zorder=5)
                ax.scatter(first_first_front_fitness_clean[:, j], first_first_front_fitness_clean[:, i], alpha=0.5, color='blue', s=8, zorder=10)
                ax.scatter(last_first_front_fitness_clean[:, j], last_first_front_fitness_clean[:, i], alpha=1, color='red', s=8, zorder=10)
                
                # set ticks
                x_min, x_max = np.percentile(all_vals_clean[:, j], [2, 100])
                y_min, y_max = np.percentile(all_vals_clean[:, i], [2, 100])
                pad_x = (x_max - x_min) * 0.05
                pad_y = (y_max - y_min) * 0.05

                ax.set_xlim((x_min-pad_x, x_max+pad_x))
                ax.set_ylim((y_min-pad_y, y_max+pad_y))
                
                ax.xaxis.set_label_coords(0.5, -0.3)
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=2))
                
                ax.yaxis.set_label_coords(-0.3, 0.5)
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
                
            # set axis labels
            if i == n_obj - 1:
                ax.set_xlabel(names_plot[j], fontdict={'weight':'bold'}, fontsize=16)
            if j == 0:
                ax.set_ylabel(names_plot[i], fontdict={'weight':'bold'}, fontsize=16)
                
            ax.tick_params(direction="in", labelsize=14)
            
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(evb_dir_modeling.CalibrateVIC_dir, "calibration_cps", f"Exp1_case15_case78_calibration_process_{case_name}.tiff"), dpi=300)
    plt.show(block=True)


def plot_soil_layer_depths_compare_temporalpattern(ind_num=5, ind=None):
    case_names = ["case4", "case5"]
    case_names_plot = ["Case 4", "Case 5"]
        
    # get data
    sim_all_cases = {}
    for case_name in case_names:
        # get sim
        sim = get_sim_vic(evb_dir_modeling, case_name, ind_num)
        sim_all_cases[case_name] = sim
    
    # load and preprocess data
    period = "vali"
    
    sim_all_cases_baseflow = {}
    sim_all_cases_runoff = {}
    sim_all_cases_evap = {}
    sim_all_cases_sm = {}
    for case_name in case_names:
        sim_case = sim_all_cases[case_name]
        
        # OUT_BASEFLOW
        sim_case_baseflow = np.array([sim_case[key][period]["OUT_BASEFLOW"] for key in sim_case.keys()])  # [inds_num, time, lat, lon]
        fill_value_ = sim_case[list(sim_case.keys())[0]][period]["OUT_BASEFLOW"].fill_value
        sim_case_baseflow[sim_case_baseflow == fill_value_] = np.nan
        if ind is not None:
            sim_case_baseflow = sim_case_baseflow[ind, :, :, :]
        else:
            sim_case_baseflow = np.nanmean(sim_case_baseflow, axis=0)  # [time, lat, lon]
        
        # OUT_RUNOFF
        sim_case_runoff = np.array([sim_case[key][period]["OUT_RUNOFF"] for key in sim_case.keys()])  # [inds_num, time, lat, lon]
        fill_value_ = sim_case[list(sim_case.keys())[0]][period]["OUT_RUNOFF"].fill_value
        sim_case_runoff[sim_case_runoff == fill_value_] = np.nan
        if ind is not None:
            sim_case_runoff = sim_case_runoff[ind, :, :, :]
        else:
            sim_case_runoff = np.nanmean(sim_case_runoff, axis=0)  # [time, lat, lon]
        
        # OUT_EVAP
        sim_case_evap = np.array([sim_case[key][period]["OUT_EVAP"] for key in sim_case.keys()])  # [inds_num, time, lat, lon]
        fill_value_ = sim_case[list(sim_case.keys())[0]][period]["OUT_EVAP"].fill_value
        sim_case_evap[sim_case_evap == fill_value_] = np.nan
        if ind is not None:
            sim_case_evap = sim_case_evap[ind, :, :, :]
        else:
            sim_case_evap = np.nanmean(sim_case_evap, axis=0)  # [time, lat, lon]
        
        # OUT_SOIL_MOIST
        sim_case_sm = np.array([sim_case[key][period]["OUT_SOIL_MOIST"] for key in sim_case.keys()])  # [inds_num, time, layers, lat, lon]
        fill_value_ = sim_case[list(sim_case.keys())[0]][period]["OUT_SOIL_MOIST"].fill_value
        sim_case_sm[sim_case_sm == fill_value_] = np.nan
        if ind is not None:
            sim_case_sm = sim_case_sm[ind, :, :, :, :]
        else:
            sim_case_sm = np.nanmean(sim_case_sm, axis=0)  # [time, layers, lat, lon]
        
        sim_all_cases_baseflow[case_name] = sim_case_baseflow
        sim_all_cases_runoff[case_name] = sim_case_runoff
        sim_all_cases_evap[case_name] = sim_case_evap
        sim_all_cases_sm[case_name] = sim_case_sm
    
    # plot
    fig, axes = plt.subplots(
        2, 3,
        figsize=(16, 8),
        gridspec_kw={
            "wspace": 0.2,
            "hspace": 0.1,
            "left":0.05,
            "right": 0.99,
            "bottom": 0.08,
            "top": 0.98
        },
        sharex=True
    )
    
    # fig, axes = plt.subplots(
    #     2, 3,
    #     figsize=(14, 8),
    #     gridspec_kw={
    #         "wspace": 0.13,
    #         "hspace": 0.13,
    #         "left":0.03,
    #         "right": 0.98,
    #         "bottom": 0.08,
    #         "top": 0.95
    #     },
    #     sharex=True
    # )
    
    axes = axes.flatten()
    
    colors = {
        "case4": "deepskyblue", #"dodgerblue", #"limegreen",
        "case5": "r",
    }
    
    colors_line = {
        "case4": "dodgerblue", #"green",
        "case5": "r",
    }
    
    alpha = 1
    markersize = 20
    ax_in_list = []
    ax_annotation = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    ax_subtitles = [
        r"$R$ (mm d$^{-1}$)",                    # Surface runoff
        r"$Q_b$ (mm d$^{-1}$)",                  # Baseflow
        # r"$ET$ (mm d$^{-1}$)",                   # Evapotranspiration
        r"$Q_t$ (mm d$^{-1}$)",                   # total runoff
        r"$SM_1$ (mm)",                          # Soil moisture layer 1
        r"$SM_2$ (mm)",                          # Soil moisture layer 2
        r"$SM_3$ (mm)"                           # Soil moisture layer 3
    ]
    
    for ax in axes:
        ax_in = inset_axes(
            ax,
            width="15%",
            height="15%",
            loc="upper left",
            borderpad=0.8
        )
        ax_in.set_xlim(1, 12)
        ax_in.set_xticks([1, 6, 12])
        ax_in.set_xticklabels(["Jan", "Jun", "Dec"], fontsize=10)
        ax_in.tick_params(axis="y", labelsize=8)
        ax_in.grid(alpha=0.3)
        # ax_in.patch.set_alpha(0.6)
        ax_in.set_yticks([])
        for spine in ax_in.spines.values():
            spine.set_linewidth(0.8)

        ax_in_list.append(ax_in)
    
    for i, case_name in enumerate(case_names):
        color = colors[case_name]
        
        sim_all_case_baseflow = sim_all_cases_baseflow[case_name]
        sim_all_case_runoff = sim_all_cases_runoff[case_name]
        sim_all_case_evap = sim_all_cases_evap[case_name]
        sim_all_case_sm = sim_all_cases_sm[case_name]
        
        sim_all_case_baseflow = np.nanmean(sim_all_case_baseflow, axis=(1, 2))
        sim_all_case_runoff = np.nanmean(sim_all_case_runoff, axis=(1, 2))
        sim_all_case_evap = np.nanmean(sim_all_case_evap, axis=(1, 2))
        sim_all_case_sm_layer1 = np.nanmean(sim_all_case_sm, axis=(2, 3))[:, 0]
        sim_all_case_sm_layer2 = np.nanmean(sim_all_case_sm, axis=(2, 3))[:, 1]
        sim_all_case_sm_layer3 = np.nanmean(sim_all_case_sm, axis=(2, 3))[:, 2]
        
        sim_all_case_baseflow = pd.DataFrame(sim_all_case_baseflow, index=verify_date_eval)
        sim_all_case_runoff = pd.DataFrame(sim_all_case_runoff, index=verify_date_eval)
        sim_all_case_evap = pd.DataFrame(sim_all_case_evap, index=verify_date_eval)
        sim_all_case_sm_layer1 = pd.DataFrame(sim_all_case_sm_layer1, index=verify_date_eval)
        sim_all_case_sm_layer2 = pd.DataFrame(sim_all_case_sm_layer2, index=verify_date_eval)
        sim_all_case_sm_layer3 = pd.DataFrame(sim_all_case_sm_layer3, index=verify_date_eval)
        
        group_by_md = [verify_date_eval.month, verify_date_eval.day]
        sim_all_case_baseflow_md = sim_all_case_baseflow.groupby(group_by_md).mean().values.flatten()
        sim_all_case_runoff_md = sim_all_case_runoff.groupby(group_by_md).mean().values.flatten()
        sim_all_case_evap_md = sim_all_case_evap.groupby(group_by_md).mean().values.flatten()
        sim_all_case_sm_layer1_md = sim_all_case_sm_layer1.groupby(group_by_md).mean().values.flatten()
        sim_all_case_sm_layer2_md = sim_all_case_sm_layer2.groupby(group_by_md).mean().values.flatten()
        sim_all_case_sm_layer3_md = sim_all_case_sm_layer3.groupby(group_by_md).mean().values.flatten()
        
        group_by_m = verify_date_eval.month
        sim_all_case_baseflow_m = sim_all_case_baseflow.groupby(group_by_m).mean().values.flatten()
        sim_all_case_runoff_m = sim_all_case_runoff.groupby(group_by_m).mean().values.flatten()
        sim_all_case_evap_m = sim_all_case_evap.groupby(group_by_m).mean().values.flatten()
        sim_all_case_sm_layer1_m = sim_all_case_sm_layer1.groupby(group_by_m).mean().values.flatten()
        sim_all_case_sm_layer2_m = sim_all_case_sm_layer2.groupby(group_by_m).mean().values.flatten()
        sim_all_case_sm_layer3_m = sim_all_case_sm_layer3.groupby(group_by_m).mean().values.flatten()
        
        sim_all_case_all_var_md = [
            sim_all_case_runoff_md,
            sim_all_case_baseflow_md,
            # sim_all_case_evap_md,
            sim_all_case_baseflow_md+sim_all_case_runoff_md,
            sim_all_case_sm_layer1_md,
            sim_all_case_sm_layer2_md,
            sim_all_case_sm_layer3_md
        ]
        
        sim_all_case_all_var_m = [
            sim_all_case_runoff_m,
            sim_all_case_baseflow_m,
            sim_all_case_baseflow_m+sim_all_case_runoff_m,
            # sim_all_case_evap_m,
            sim_all_case_sm_layer1_m,
            sim_all_case_sm_layer2_m,
            sim_all_case_sm_layer3_m
        ]
        
        # plot
        for j, ax in enumerate(axes):            
            ax.scatter(
                np.arange(1, len(sim_all_case_all_var_md[j])+1),
                sim_all_case_all_var_md[j],
                color=color,
                label=case_names_plot[i],
                marker="o",
                alpha=alpha,
                s=23,
                edgecolor="k",
                linewidth=0.7,
            )
            
            ax.set_xlim(1, 366)
            ax.annotate(ax_annotation[j], xy=(0.92, 0.9), xycoords='axes fraction', fontsize=15, fontweight='bold')
            # ax.set_title(ax_subtitles[j])
            ax.set_ylabel(ax_subtitles[j])
            ax.yaxis.set_label_coords(-0.11, 0.5)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            
            ax_in_list[j].plot(
                np.arange(1, 13),
                sim_all_case_all_var_m[j],
                color=colors_line[case_name],
                linewidth=0.8,
                marker="",
                markersize=3,
                linestyle="-",
            )
            
            # if j == 2:
            #     ax.legend()
            
        fig.text(0.5, 0.015, "Day of year", ha='center', fontsize=15)
        
    # plt.legend()
    # plt.show(block=True)
    fig.savefig(os.path.join(evb_dir_modeling.CalibrateVIC_dir, "case_40_best_inds", f"Exp3_case45_soil_layer_depths_compare_temporalpattern_ind{ind}.tiff"), dpi=300)
    plt.show(block=True)
    

def plot_soil_layer_depths_compare_spatialpattern(ind_num=5, ind=None):
    case_names = ["case4", "case5"]
    case_names_plot = ["Case 4", "Case 5"]
    
    # get data
    sim_all_cases = {}
    for case_name in case_names:
        # get sim
        sim = get_sim_vic(evb_dir_modeling, case_name, ind_num)
        sim_all_cases[case_name] = sim
    
    dpc_VIC_level1 = readdpc(evb_dir_modeling.dpc_VIC_level1_path, dataProcess_VIC_level1_HRB)
    grid_shp_level1 = dpc_VIC_level1.get_data_from_cache("grid_shp")[0]
    basin_shp_level1 = dpc_VIC_level1.get_data_from_cache("basin_shp")[0]
    stand_grids_lat, stand_grids_lon = createStand_grids_lat_lon_from_gridshp(
        grid_shp_level1, reverse_lat
    )
    xmin = np.min(stand_grids_lon)
    xmax = np.max(stand_grids_lon)
    ymin = np.min(stand_grids_lat)
    ymax = np.max(stand_grids_lat)
    extent = [xmin, xmax, ymin, ymax]
    
    # load and preprocess data
    period = "vali"
    
    sim_all_cases_baseflow = {}
    sim_all_cases_runoff = {}
    sim_all_cases_evap = {}
    sim_all_cases_sm_layer1 = {}
    sim_all_cases_sm_layer2 = {}
    sim_all_cases_sm_layer3 = {}
    for case_name in case_names:
        sim_case = sim_all_cases[case_name]
        
        # OUT_BASEFLOW
        sim_case_baseflow = np.array([sim_case[key][period]["OUT_BASEFLOW"] for key in sim_case.keys()])  # [inds_num, time, lat, lon]
        fill_value_ = sim_case[list(sim_case.keys())[0]][period]["OUT_BASEFLOW"].fill_value
        sim_case_baseflow[sim_case_baseflow == fill_value_] = np.nan
        if ind is not None:
            sim_case_baseflow = sim_case_baseflow[ind, :, :, :]  # [time, lat, lon]
        else:
            sim_case_baseflow = np.nanmean(sim_case_baseflow, axis=0)  # [time, lat, lon]
        sim_case_baseflow = np.nanmean(sim_case_baseflow, axis=0)
        
        # OUT_RUNOFF
        sim_case_runoff = np.array([sim_case[key][period]["OUT_RUNOFF"] for key in sim_case.keys()])  # [inds_num, time, lat, lon]
        fill_value_ = sim_case[list(sim_case.keys())[0]][period]["OUT_RUNOFF"].fill_value
        sim_case_runoff[sim_case_runoff == fill_value_] = np.nan
        if ind is not None:
            sim_case_runoff = sim_case_runoff[ind, :, :, :]  # [time, lat, lon]
        else:   
            sim_case_runoff = np.nanmean(sim_case_runoff, axis=0)  # [time, lat, lon]
        sim_case_runoff = np.nanmean(sim_case_runoff, axis=0)
        
        # OUT_EVAP
        sim_case_evap = np.array([sim_case[key][period]["OUT_EVAP"] for key in sim_case.keys()])  # [inds_num, time, lat, lon]
        fill_value_ = sim_case[list(sim_case.keys())[0]][period]["OUT_EVAP"].fill_value
        sim_case_evap[sim_case_evap == fill_value_] = np.nan
        if ind is not None:
            sim_case_evap = sim_case_evap[ind, :, :, :]  # [time, lat, lon]
        else:
            sim_case_evap = np.nanmean(sim_case_evap, axis=0)  # [time, lat, lon]
        sim_case_evap = np.nanmean(sim_case_evap, axis=0)
        
        # OUT_SOIL_MOIST
        sim_case_sm = np.array([sim_case[key][period]["OUT_SOIL_MOIST"] for key in sim_case.keys()])  # [inds_num, time, layers, lat, lon]
        fill_value_ = sim_case[list(sim_case.keys())[0]][period]["OUT_SOIL_MOIST"].fill_value
        sim_case_sm[sim_case_sm == fill_value_] = np.nan
        if ind is not None:
            sim_case_sm = sim_case_sm[ind, :, :, :, :]  # [time, layers, lat, lon]
        else:
            sim_case_sm = np.nanmean(sim_case_sm, axis=0)  # [time, layers, lat, lon]
        sim_case_sm = np.nanmean(sim_case_sm, axis=0)  # [layers, lat, lon]
        
        sim_all_cases_baseflow[case_name] = sim_case_baseflow
        sim_all_cases_runoff[case_name] = sim_case_runoff
        sim_all_cases_evap[case_name] = sim_case_evap
        sim_all_cases_sm_layer1[case_name] = sim_case_sm[0, :, :]
        sim_all_cases_sm_layer2[case_name] = sim_case_sm[1, :, :]
        sim_all_cases_sm_layer3[case_name] = sim_case_sm[2, :, :]
    
    case_left = "case5"
    case_right = "case4"
    sim_all_cases_baseflow_diff = sim_all_cases_baseflow[case_left] - sim_all_cases_baseflow[case_right]
    sim_all_cases_runoff_diff = sim_all_cases_runoff[case_left] - sim_all_cases_runoff[case_right]
    # sim_all_cases_evap_diff = sim_all_cases_evap[case_left] - sim_all_cases_evap[case_right]
    sim_all_cases_total_runoff_diff = sim_all_cases_runoff_diff + sim_all_cases_baseflow_diff
    sim_all_cases_sm_layer1_diff = sim_all_cases_sm_layer1[case_left] - sim_all_cases_sm_layer1[case_right]
    sim_all_cases_sm_layer2_diff = sim_all_cases_sm_layer2[case_left] - sim_all_cases_sm_layer2[case_right]
    sim_all_cases_sm_layer3_diff = sim_all_cases_sm_layer3[case_left] - sim_all_cases_sm_layer3[case_right]    
    
    diff_all_var = [
        sim_all_cases_runoff_diff,
        sim_all_cases_baseflow_diff,
        sim_all_cases_total_runoff_diff,
        # sim_all_cases_evap_diff,
        sim_all_cases_sm_layer1_diff,
        sim_all_cases_sm_layer2_diff,
        sim_all_cases_sm_layer3_diff
    ]
    
    params_all_cases = {
        "level0": {},
        "level1": {}
    }
    
    for case_name in case_names:
        params_case = get_params(evb_dir_modeling, case_name, ind_num, period="vali")
        params_case_level0_depth = np.array([params_case[key]["level0"]["depth"] for key in params_case.keys()])  # [inds_num, nlayer, lat, lon]
        fill_value_ = params_case[list(params_case.keys())[0]]["level0"]["depth"].fill_value
        params_case_level0_depth[params_case_level0_depth == fill_value_] = np.nan
        if ind is not None:
            params_case_level0_depth = params_case_level0_depth[ind, :, :, :]  # [nlayer, lat, lon]
        else:
            params_case_level0_depth = np.nanmean(params_case_level0_depth, axis=0)  # [nlayer, lat, lon]
        
        params_case_level1_depth = np.array([params_case[key]["level1"]["depth"] for key in params_case.keys()])
        fill_value_ = params_case[list(params_case.keys())[0]]["level1"]["depth"].fill_value
        params_case_level1_depth[params_case_level1_depth == fill_value_] = np.nan
        if ind is not None:
            params_case_level1_depth = params_case_level1_depth[ind, :, :, :]  # [nlayer, lat, lon]
        else:
            params_case_level1_depth = np.nanmean(params_case_level1_depth, axis=0)  # [nlayer, lat, lon]
        
        params_all_cases["level0"][case_name] = params_case_level0_depth
        params_all_cases["level1"][case_name] = params_case_level1_depth
    
    diff_depth_level1_layer1 = (params_all_cases["level1"][case_left][0, :, :] - params_all_cases["level1"][case_right][0, :, :]) * 1000
    diff_depth_level1_layer2 = (params_all_cases["level1"][case_left][1, :, :] - params_all_cases["level1"][case_right][1, :, :]) * 1000
    diff_depth_level1_layer3 = (params_all_cases["level1"][case_left][2, :, :] - params_all_cases["level1"][case_right][2, :, :]) * 1000
    
    diff_all_var.extend([diff_depth_level1_layer1, diff_depth_level1_layer2, diff_depth_level1_layer3])
    
    # plot
    fig, axes = plt.subplots(
        3, 3,
        figsize=(13, 11),
        gridspec_kw={
            "wspace": 0.03,
            "hspace": 0.1,
            "left":0.03,
            "right": 0.90,
            "bottom": 0.03,
            "top": 0.97
        },
        sharex=True,
        sharey=True
    )
    
    axes = axes.flatten()
    ax_annotation = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]
    # ax_subtitles = [
    #     r"$R$ (mm d$^{-1}$)",                    # Surface runoff
    #     r"$Q_b$ (mm d$^{-1}$)",                  # Baseflow
    #     # r"$ET$ (mm d$^{-1}$)",                   # Evapotranspiration
    #     r"$Q_t$ (mm d$^{-1}$)",                   # total runoff
    #     r"$SM_1$ (mm)",                          # Soil moisture layer 1
    #     r"$SM_2$ (mm)",                          # Soil moisture layer 2
    #     r"$SM_3$ (mm)",                           # Soil moisture layer 3
    #     r"$\Delta d_1$ (mm)",
    #     r"$\Delta d_2$ (mm)",
    #     r"$\Delta d_3$ (mm)",
    # ]
    
    ax_subtitles = [
        r"$\Delta R$",                    # Surface runoff
        r"$\Delta Q_b$",                  # Baseflow
        # r"$ET$ (mm d$^{-1}$)",                   # Evapotranspiration
        r"$\Delta Q_t$",                   # total runoff
        r"$\Delta SM_1$",                          # Soil moisture layer 1
        r"$\Delta SM_2$",                          # Soil moisture layer 2
        r"$\Delta SM_3$",                           # Soil moisture layer 3
        r"$\Delta h_1$",
        r"$\Delta h_2$",
        r"$\Delta h_3$",
    ]
    
    cmap1 = "RdBu_r"
    cmap2 = "coolwarm"
    cmap3 = "bwr"
    cmap = [
        cmap1,
        cmap1,
        cmap1,
        cmap2,
        cmap2,
        cmap2,
        cmap3,
        cmap3,
        cmap3,
    ]
    
    absmax_1 = np.nanmax(np.abs(diff_all_var[0:3]))
    absmax_2 = np.nanmax(np.abs(diff_all_var[3:6]))
    absmax_3 = np.nanmax(np.abs(diff_all_var[6:]))
    
    absmax_1 -= absmax_1 * 0.2
    absmax_2 -= absmax_2 * 0.2
    absmax_3 -= absmax_3 * 0.2
    
    norm1 = mpl.colors.Normalize(vmin=-absmax_1, vmax=absmax_1)
    norm2 = mpl.colors.Normalize(vmin=-absmax_2, vmax=absmax_2)
    norm3 = mpl.colors.Normalize(vmin=-absmax_3, vmax=absmax_3)
    
    sm1 = mpl.cm.ScalarMappable(norm=norm1, cmap=cmap1)
    sm2 = mpl.cm.ScalarMappable(norm=norm2, cmap=cmap2)
    sm3 = mpl.cm.ScalarMappable(norm=norm3, cmap=cmap3)
    
    for j, ax in enumerate(axes):
        group = j // 3
        
        if group == 0:
            vmin, vmax = -absmax_1, absmax_1
        elif group == 1:
            vmin, vmax = -absmax_2, absmax_2
        else:
            vmin, vmax = -absmax_3, absmax_3
            
        ax.imshow(
            diff_all_var[j],
            extent=extent,
            cmap=cmap[j],
            vmin=vmin,
            vmax=vmax
        )
        
        basin_shp_level1.plot(ax=ax, facecolor="none", edgecolor="k", linewidth=3.0)
        bbox = box(xmin, ymin, xmax, ymax)
        
        basin_fixed = basin_shp_level1.copy()
        basin_fixed["geometry"] = basin_fixed.geometry.buffer(0)
        basin_union = basin_fixed.geometry.unary_union
        outside = bbox.difference(basin_union)

        outside_gdf = gpd.GeoDataFrame(
            geometry=[outside],
            crs=basin_shp_level1.crs
        )

        outside_gdf.plot(
            ax=ax,
            facecolor="w",
            edgecolor="none",
            zorder=10
        )
        
        ax.set_title(ax_subtitles[j], fontsize=17, pad=6)
        ax.annotate(ax_annotation[j], xy=(0.02, 0.9), xycoords='axes fraction', fontsize=15, fontweight='bold', zorder=15)
        set_boundary(ax, [xmin, ymin, xmax, ymax])
        set_xyticks(ax, x_locator_interval=0.5, y_locator_interval=0.5, yticks_rotation=90)
        ax.tick_params(labelsize=13)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_zorder(20)
        
    nrows = 3
    ncols = 3
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i * ncols + j]
            if i < nrows - 1:
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
            if j > 0:
                ax.tick_params(axis="y", left=False, labelleft=False)
    
    cax1 = fig.add_axes([0.92, 0.69, 0.015, 0.26])  # top group
    cax2 = fig.add_axes([0.92, 0.37, 0.015, 0.26])  # middle group
    cax3 = fig.add_axes([0.92, 0.05, 0.015, 0.26])  # bottom group
    cb1 = fig.colorbar(sm1, cax=cax1)
    cb2 = fig.colorbar(sm2, cax=cax2)
    cb3 = fig.colorbar(sm3, cax=cax3)

    cb1.set_label("Difference (mm d$^{-1}$)", fontsize=15)
    cb2.set_label("Difference (mm)", fontsize=15)
    cb3.set_label("Difference (mm)", fontsize=15)
    cb1.ax.yaxis.set_label_coords(3.4, 0.5)
    cb2.ax.yaxis.set_label_coords(3.4, 0.5)
    cb3.ax.yaxis.set_label_coords(3.4, 0.5)
    
    cb1.ax.tick_params(labelsize=10)
    cb2.ax.tick_params(labelsize=10)
    cb3.ax.tick_params(labelsize=10)
    
    fig.savefig(os.path.join(evb_dir_modeling.CalibrateVIC_dir, "case_40_best_inds", f"Exp3_case45_soil_layer_depths_compare_spatialpattern_ind{ind}.tiff"), dpi=300)
    plt.show(block=True)


def plot_soil_layer_depths_compare_params_flows(ind_num=5, ind=0):
    cp_cases = get_calibration_cps(evb_dir_modeling, ind_num=ind_num, max_num=40)
    cp_ind = cp_cases["case4"]["all_best_ind"][cp_cases["case4"]["first_num_best_ind_index"][ind]]
    params_case = deepcopy(params_minimal)
    params_case["rvic_params"] = rvic_params_spatial
    params_case["g_params"] = set_g_params_soilGrids_layer(params_case["g_params"])
    pm = ParamManager(params_case)
    ind_format = pm.format_vector(cp_ind, get_free=True)
    param_dict = pm.to_dict(vector=ind_format, field="optimal", get_free=True)
    param_dict["g_params"]["total_depths"]
    param_dict["g_params"]["soil_layers_breakpoints"]
    
    cp_ind = cp_cases["case5"]["all_best_ind"][cp_cases["case5"]["first_num_best_ind_index"][ind]]
    params_case = deepcopy(params_minimal)
    params_case["g_params"] = set_g_params_soilGrids_layer(params_case["g_params"])
    params_case["g_params"] = expand_station_wise_params(params_case["g_params"], station_num=len(station_names))
    params_case["rvic_params"] = rvic_params_spatial
    pm = ParamManager(params_case)
    ind_format = pm.format_vector(cp_ind, get_free=True)
    param_dict = pm.to_dict(vector=ind_format, field="optimal", get_free=True)
    
    g10 = []
    z1 = []
    z2 = []
    for i in range(5):
        g10.append(param_dict["g_params"][f"total_depths_{i}"]["optimal"][0])
        z1.append(param_dict["g_params"][f"soil_layers_breakpoints_{i}"]["optimal"][0])
        z2.append(param_dict["g_params"][f"soil_layers_breakpoints_{i}"]["optimal"][1])
        
    # plot flows
    station_name = "lianghekou"
    obs_youshui = get_obs(evb_dir_modeling, get_type="validation", station_names=station_names)[f"streamflow(m3/s)_{station_name}"]
    sim_youshui_case4 = get_sim_rvic(evb_dir_modeling, case_name="case4", ind_num=ind_num, station_names=station_names)[ind]["vali"][f"streamflow(m3/s)_{station_name}"]
    sim_youshui_case5 = get_sim_rvic(evb_dir_modeling, case_name="case5", ind_num=ind_num, station_names=station_names)[ind]["vali"][f"streamflow(m3/s)_{station_name}"]
    
    # plt.plot(obs_youshui, sim_youshui_case4, "b.")
    plt.plot(range(len(obs_youshui)), obs_youshui, "k-")
    plt.plot(range(len(obs_youshui)), sim_youshui_case4, "b-")
    plt.plot(range(len(obs_youshui)), sim_youshui_case5, "r.")
    plt.show(block=True)
    
        
def plot_RVIC_parameters_compare_res():
    home = "case_rvic_params_compare_case4_runoff_base"
    obs = get_obs(evb_dir_modeling, get_type="validation", station_names=station_names)
    
    case_names = ["case3", "case2", "case4"]  # increasing complexity
    n_warmup = len(warmup_date_eval)
    n_calibration = len(calibrate_date_eval)
    n_validation = len(verify_date_eval)
    
    # read
    sim = {
        "case3": {},
        "case2": {},
        "case4": {},
    }
    for case_name in case_names:
        case_home = os.path.join(evb_dir_modeling.CalibrateVIC_dir, home, case_name)
        
        rvic_fn = "HRB_shiquan_6km.rvic.h0a.2019-01-01_cali0.nc"
        with Dataset(os.path.join(case_home, rvic_fn), "r") as sim_dataset:
            # set date index
            start_index = n_warmup + n_calibration
            end_index = n_warmup + n_calibration + n_validation
            
            # lon, lat
            sim_time = sim_dataset["time"]
            sim_time = num2date(sim_time[:], sim_time.units, sim_time.calendar)
            sim_time = sim_time[start_index:end_index]
            
            # streamflow
            for i, name in enumerate(station_names):
                sim[case_name][f"streamflow(m3/s)_{name}"] = sim_dataset.variables["streamflow"][start_index:end_index, i]
    
    # plot
    # station_name = station_names[-1]
    # plt.plot(verify_date_eval, obs[f"streamflow(m3/s)_{station_name}"], "k")
    # plt.plot(verify_date_eval, sim["case3"][f"streamflow(m3/s)_{station_name}"], "g-", markersize=5)
    # plt.plot(verify_date_eval, sim["case2"][f"streamflow(m3/s)_{station_name}"], "b-", markersize=5)
    # plt.plot(verify_date_eval, sim["case4"][f"streamflow(m3/s)_{station_name}"], "r-", markersize=5)
    # plt.show(block=True)
    
    for station_name in station_names:
        print(station_name)
        print("case3: KGE_m", EvaluationMetric(sim["case3"][f"streamflow(m3/s)_{station_name}"].filled(0), obs[f"streamflow(m3/s)_{station_name}"].values.flatten()).KGE_m())
        print("case2: KGE_m", EvaluationMetric(sim["case2"][f"streamflow(m3/s)_{station_name}"].filled(0), obs[f"streamflow(m3/s)_{station_name}"].values.flatten()).KGE_m())
        print("case4: KGE_m", EvaluationMetric(sim["case4"][f"streamflow(m3/s)_{station_name}"].filled(0), obs[f"streamflow(m3/s)_{station_name}"].values.flatten()).KGE_m())
    
    for station_name in station_names:
        for case_name in case_names:
            sim[case_name][f"streamflow(m3/s)_{station_name}"] = sim[case_name][f"streamflow(m3/s)_{station_name}"].filled(0)
        
        obs[f"streamflow(m3/s)_{station_name}"] = obs[f"streamflow(m3/s)_{station_name}"].values.flatten()
            
    case_pairs = [
        ("case2", "case3"),
        ("case2", "case4"),
        ("case3", "case4"),
    ]

    n_station = 5
    n_pair = 3

    title_map = {
        ("case2", "case3"):
            r"$\Delta Q_{\mathrm{Case 2}}$ vs $\Delta Q_{\mathrm{Case 3}}$",
        ("case2", "case4"):
            r"$\Delta Q_{\mathrm{Case 2}}$ vs $\Delta Q_{\mathrm{Case 4}}$",
        ("case3", "case4"):
            r"$\Delta Q_{\mathrm{Case 3}}$ vs $\Delta Q_{\mathrm{Case 4}}$",
    }
    
    def quad_scatter(ax, x, y, obs_q, pct_h=0.98, pct_low=0.3):
        """
        Four-quadrant scatter with top/bottom percent highlighted
        """
        # r = np.sqrt(x**2 + y**2)

        # q_low  = np.quantile(r, pct_low)
        # q_high = np.quantile(r, pct_h)

        # idx_low  = r <= q_low
        # idx_high = r >= q_high
        # idx_mid  = (~idx_low) & (~idx_high)
        q_low  = np.quantile(obs_q, pct_low)
        q_high = np.quantile(obs_q, pct_h)

        idx_low  = obs_q <= q_low
        idx_high = obs_q >= q_high
        idx_mid  = (~idx_low) & (~idx_high)

        s = 10
        ax.scatter(x[idx_mid],  y[idx_mid],  s=s, c="lightgrey", alpha=0.9)
        ax.scatter(x[idx_high], y[idx_high], s=s, c="red", alpha=0.9)
        ax.scatter(x[idx_low],  y[idx_low],  s=s, c="blue", alpha=0.9)

        # 4-quadrant lines
        ax.axhline(0, color="k", lw=0.8, alpha=0.8)
        ax.axvline(0, color="k", lw=0.8, alpha=0.8)
        
        lim = np.max(np.abs(np.r_[x, y]))
        lim -= lim * 0.3
        ax.plot([-lim, lim], [-lim, lim],
                linestyle="--", color="k", lw=0.8, zorder=1)

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        # ax.set_aspect("equal", adjustable="box")
        # ax.set_aspect("equal", adjustable="box")
    
    fig, axes = plt.subplots(
        n_pair, n_station,
        figsize=(14, 8),
        gridspec_kw={
            "wspace": 0.08,
            "hspace": 0.08,
            "left":0.1,
            "right": 0.99,
            "bottom": 0.08,
            "top": 0.95
        },
        sharex="col",
        sharey="row",
    )
    station_names_plot = [
        "Hanzhong",
        "Yangxian",
        "Youshui",
        "Lianghekou",
        "Shiquan"
    ]
    for i, station_name in enumerate(station_names[:n_station]):
        obs_q = obs[f"streamflow(m3/s)_{station_name}"]

        for j, (caseA, caseB) in enumerate(case_pairs):
            simA = sim[caseA][f"streamflow(m3/s)_{station_name}"]
            simB = sim[caseB][f"streamflow(m3/s)_{station_name}"]

            x = simA - obs_q
            y = simB - obs_q

            ax = axes[j, i]
            quad_scatter(ax, x, y, obs_q)

            if j == 0:
                ax.set_title(station_names_plot[i], fontsize=14)

            if i == 0:
                ax.annotate(
                    title_map[(caseA, caseB)],
                    xy=(-0.28, 0.5),
                    xycoords="axes fraction",
                    ha="right",
                    va="center",
                    fontsize=14,
                    rotation=90
                )
                
    fig.supxlabel("ΔQ (case A - obs)", fontsize=14)
    fig.supylabel("ΔQ (case B - obs)", fontsize=14)
    # plt.show(block=True)
    fig.savefig(os.path.join(evb_dir_modeling.CalibrateVIC_dir, home, f"Exp3_case45_RVIC_parameters_compare_res_ind0.tiff"), dpi=300)
    
    # fig, axes = plt.subplots(
    #     n_station, n_pair,
    #     figsize=(10, 14),
    #     gridspec_kw={
    #         "wspace": 0.1,
    #         "hspace": 0.15,
    #         "left":0.13,
    #         "right": 0.95,
    #         "bottom": 0.05,
    #         "top": 0.96
    #     },
    #     sharey="row",
    # )

    # for i, station_name in enumerate(station_names[:n_station]):
    #     obs_q = obs[f"streamflow(m3/s)_{station_name}"]

    #     for j, (caseA, caseB) in enumerate(case_pairs):
    #         simA = sim[caseA][f"streamflow(m3/s)_{station_name}"]
    #         simB = sim[caseB][f"streamflow(m3/s)_{station_name}"]

    #         x = simA - obs_q
    #         y = simB - obs_q

    #         ax = axes[i, j]
    #         quad_scatter(ax, x, y, obs_q)

    #         # titles & labels
    #         if i == 0:
    #             # ax.set_title(f"{caseA} - obs vs {caseB} - obs", fontsize=11)
    #             ax.set_title(
    #                 title_map[(caseA, caseB)],
    #                 fontsize=14
    #             )
    #         if j == 0:
    #             ax.set_ylabel(station_name, fontsize=14)
    #             ax.yaxis.set_label_coords(-0.25, 0.5)

    # # global labels
    # fig.supxlabel("ΔQ (case A - obs)", fontsize=14)
    # fig.supylabel("ΔQ (case B - obs)", fontsize=14)

    # # plt.tight_layout()
    # fig.savefig(os.path.join(evb_dir_modeling.CalibrateVIC_dir, "case_rvic_params_compare_case3_runoff_base", f"Exp3_case45_RVIC_parameters_compare_res_ind0.tiff"), dpi=300)
    plt.show(block=True)


def plot_performance_statistical_evaluation(percentile=0.1):
    cps_home = os.path.join(os.path.join(evb_dir_modeling.CalibrateVIC_dir, "calibration_cps"))
    case_num_NSGAII = list(range(1, 6))
    case_num_CMAES = list(range(6, 9))
    
    case_name_NSGAII = [f"case{num}" for num in case_num_NSGAII]
    case_name_CMAES = [f"case{num}" for num in case_num_CMAES]
    max_num = 40
    
    first_num_best_ind_fitness_all = {}
    
    for case_name in case_name_NSGAII:
        suffix = "_40" if case_name != "case5" else "_90"
        cp_path_dir = os.path.join(cps_home, case_name + suffix)
        cp_path_fn = [fn for fn in os.listdir(cp_path_dir) if fn.endswith(".pkl")][0]
        cp_path = os.path.join(cp_path_dir, cp_path_fn)
        # first_num_data = get_inds_NSGAII(evb_dir_modeling, ind_num=None, max_num=max_num, percentile=percentile, cp_path=cp_path)
        first_num_data = get_inds_NSGAII(evb_dir_modeling, ind_num=40, max_num=max_num, percentile=None, cp_path=cp_path)
        first_num_best_ind_fitness_all[case_name] = first_num_data["first_num_best_ind_fitness"]
        
    for case_name in case_name_CMAES:
        suffix = "_40" if case_name != "case5" else "_90"
        cp_path_dir = os.path.join(cps_home, case_name + suffix)
        cp_path_fn = [fn for fn in os.listdir(cp_path_dir) if fn.endswith(".pkl")][0]
        cp_path = os.path.join(cp_path_dir, cp_path_fn)
        # first_num_data = get_inds_CMA_ES(evb_dir_modeling, ind_num=None, max_num=max_num, percentile=percentile, cp_path=cp_path)
        first_num_data = get_inds_CMA_ES(evb_dir_modeling, ind_num=40, max_num=max_num, percentile=None, cp_path=cp_path)
        first_num_best_ind_fitness_all[case_name] = first_num_data["first_num_best_ind_fitness"]
    
    cases = list(first_num_best_ind_fitness_all.keys())
    df_ttest = pd.DataFrame(index=cases, columns=cases, dtype=object)
    for case1, case2 in itertools.product(cases, repeat=2):
        x = first_num_best_ind_fitness_all[case1]
        y = first_num_best_ind_fitness_all[case2]
        
        if case1 == case2:
            df_ttest.loc[case1, case2] = None
        else:
            t_stat, p_val = ttest_ind(x, y, equal_var=False)  # Welch t-test
            
            if p_val < 0.05:
                sig = '*'
            elif p_val < 0.01:
                sig = '**'
            else:
                sig = ''
            
            df_ttest.loc[case1, case2] = f"{t_stat:.3f}{sig}" # f"t={t_stat:.3f}, p={p_val:.3f}{sig}"

    print(df_ttest)
    
    output_path = os.path.join(evb_dir_modeling.CalibrateVIC_dir, "metrics_ttest.xlsx")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_ttest.to_excel(writer)


def plot_high_flow_low_flow_compare_Exp_1():
    home = "case_rvic_params_compare_case1578_runoff_base"
    obs = get_obs(evb_dir_modeling, get_type="validation", station_names=station_names)
    
    case_names = ["case1", "case5", "case7", "case8"]
    n_warmup = len(warmup_date_eval)
    n_calibration = len(calibrate_date_eval)
    n_validation = len(verify_date_eval)
    
    # read
    sim = {
        "case1": {},
        "case5": {},
        "case7": {},
        "case8": {}
    }
    for case_name in case_names:
        case_home = os.path.join(evb_dir_modeling.CalibrateVIC_dir, home, case_name)
        
        rvic_fn = "HRB_shiquan_6km.rvic.h0a.2019-01-01_cali0.nc"
        with Dataset(os.path.join(case_home, rvic_fn), "r") as sim_dataset:
            # set date index
            start_index = n_warmup + n_calibration
            end_index = n_warmup + n_calibration + n_validation
            
            # lon, lat
            sim_time = sim_dataset["time"]
            sim_time = num2date(sim_time[:], sim_time.units, sim_time.calendar)
            sim_time = sim_time[start_index:end_index]
            
            # streamflow
            for i, name in enumerate(station_names):
                sim[case_name][f"streamflow(m3/s)_{name}"] = sim_dataset.variables["streamflow"][start_index:end_index, i]
    
    for station_name in station_names:
        print(station_name)
        print("case1: KGE_m", EvaluationMetric(sim["case1"][f"streamflow(m3/s)_{station_name}"].filled(0), obs[f"streamflow(m3/s)_{station_name}"].values.flatten()).KGE_m())
        print("case5: KGE_m", EvaluationMetric(sim["case5"][f"streamflow(m3/s)_{station_name}"].filled(0), obs[f"streamflow(m3/s)_{station_name}"].values.flatten()).KGE_m())
        print("case7: KGE_m", EvaluationMetric(sim["case7"][f"streamflow(m3/s)_{station_name}"].filled(0), obs[f"streamflow(m3/s)_{station_name}"].values.flatten()).KGE_m())
        print("case8: KGE_m", EvaluationMetric(sim["case8"][f"streamflow(m3/s)_{station_name}"].filled(0), obs[f"streamflow(m3/s)_{station_name}"].values.flatten()).KGE_m())
        
    for station_name in station_names:
        for case_name in case_names:
            sim[case_name][f"streamflow(m3/s)_{station_name}"] = sim[case_name][f"streamflow(m3/s)_{station_name}"].filled(0)
        
        obs[f"streamflow(m3/s)_{station_name}"] = obs[f"streamflow(m3/s)_{station_name}"].values.flatten()
            
    case_pairs = [
        ("case1", "case5"),
        ("case1", "case7"),
        ("case1", "case8"),
        ("case5", "case7"),
        ("case5", "case8"),
        ("case7", "case8"),
    ]

    n_station = 5
    n_pair = len(case_pairs)

    title_map = {
        (c1, c2): rf"$\Delta Q_{{\mathrm{{Case {c1[-1]}}}}}$ vs $\Delta Q_{{\mathrm{{Case {c2[-1]}}}}}$"
        for c1, c2 in case_pairs
    }
    
    def quad_scatter(ax, x, y, obs_q, pct_h=0.98, pct_low=0.3):
        """
        Four-quadrant scatter with top/bottom percent highlighted
        """
        # r = np.sqrt(x**2 + y**2)

        # q_low  = np.quantile(r, pct_low)
        # q_high = np.quantile(r, pct_h)

        # idx_low  = r <= q_low
        # idx_high = r >= q_high
        # idx_mid  = (~idx_low) & (~idx_high)
        q_low  = np.quantile(obs_q, pct_low)
        q_high = np.quantile(obs_q, pct_h)

        idx_low  = obs_q <= q_low
        idx_high = obs_q >= q_high
        idx_mid  = (~idx_low) & (~idx_high)

        s = 10
        ax.scatter(x[idx_mid],  y[idx_mid],  s=s, c="lightgrey", alpha=0.9)
        ax.scatter(x[idx_high], y[idx_high], s=s, c="red", alpha=0.9)
        ax.scatter(x[idx_low],  y[idx_low],  s=s, c="blue", alpha=0.9)

        # 4-quadrant lines
        ax.axhline(0, color="k", lw=0.8, alpha=0.8)
        ax.axvline(0, color="k", lw=0.8, alpha=0.8)
        
        lim = np.max(np.abs(np.r_[x, y]))
        lim -= lim * 0.3
        ax.plot([-lim, lim], [-lim, lim],
                linestyle="--", color="k", lw=0.8, zorder=1)

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        # ax.set_aspect("equal", adjustable="box")
        # ax.set_aspect("equal", adjustable="box")
    
    fig, axes = plt.subplots(
        n_pair, n_station,
        figsize=(10, 14),
        gridspec_kw={
            "wspace": 0.08,
            "hspace": 0.08,
            "left":0.1,
            "right": 0.99,
            "bottom": 0.08,
            "top": 0.95
        },
        sharex="col",
        sharey="row",
    )
    station_names_plot = [
        "Hanzhong",
        "Yangxian",
        "Youshui",
        "Lianghekou",
        "Shiquan"
    ]
    for i, station_name in enumerate(station_names[:n_station]):
        obs_q = obs[f"streamflow(m3/s)_{station_name}"]

        for j, (caseA, caseB) in enumerate(case_pairs):
            simA = sim[caseA][f"streamflow(m3/s)_{station_name}"]
            simB = sim[caseB][f"streamflow(m3/s)_{station_name}"]

            x = simA - obs_q
            y = simB - obs_q

            ax = axes[j, i]
            quad_scatter(ax, x, y, obs_q)

            if j == 0:
                ax.set_title(station_names_plot[i], fontsize=14)

            if i == 0:
                ax.annotate(
                    title_map[(caseA, caseB)],
                    xy=(-0.28, 0.5),
                    xycoords="axes fraction",
                    ha="right",
                    va="center",
                    fontsize=14,
                    rotation=90
                )
                
    fig.supxlabel("ΔQ (case A - obs)", fontsize=14)
    fig.supylabel("ΔQ (case B - obs)", fontsize=14)
    plt.show(block=True)
    # fig.savefig(os.path.join(evb_dir_modeling.CalibrateVIC_dir, home, f"Exp1_1578_RVIC_parameters_compare_res_ind0.tiff"), dpi=300)
    
    def plot_shiquan_high_low_sim_beauty(
        sim, obs,
        case_names=("case1", "case5", "case7", "case8"),
        station_name="shiquan",
        pct_high=0.98,
        pct_low=0.30,
    ):
        """
        Plot high/low flow at Shiquan station with each case as colored scatter points
        and median markers (like paper-style).
        """
        # colors = ["red", "blue", "darkorange", "m"]
        colors = [
        (1.0, 0.3, 0.3, 0.7),
        (0.3, 0.5, 1.0, 0.7),
        (1.0, 0.6, 0.2, 0.7),
        (0.7, 0.3, 0.7, 0.7) 
    ]

        obs_q = obs[f"streamflow(m3/s)_{station_name}"]

        q_high = np.quantile(obs_q, pct_high)
        q_low = np.quantile(obs_q, pct_low)

        idx_high = obs_q >= q_high
        idx_low = obs_q <= q_low

        # Prepare data
        data_high = [sim[c][f"streamflow(m3/s)_{station_name}"][idx_high] for c in case_names]
        data_low  = [sim[c][f"streamflow(m3/s)_{station_name}"][idx_low]  for c in case_names]

        def plot_box_scatter(ax, data, title):
            for i, d in enumerate(data):
                x_center = i + 1
                
                # scatter points with jitter
                x = np.random.normal(x_center, 0.05, size=len(d))
                ax.scatter(x, d, color=colors[i], edgecolor="k", s=20, zorder=2, alpha=0.3)
                
                # median line
                median = np.median(d)
                ax.plot([i+0.8, i+1.2], [median, median], color='r', lw=2, zorder=3, linestyle='--')  # [i+0.6, i+1.4]

                # Whisker: min -> median -> max vertical line
                d_min = np.min(d)
                d_max = np.max(d)
                median = np.median(d)
                gap_fraction = 0.05

                lower_end = median - (d_max - d_min) * (gap_fraction)
                upper_start = median + (d_max - d_min) * (gap_fraction)
                
                ax.vlines(x_center, d_min, lower_end, color='k', lw=2, zorder=1)
                ax.vlines(x_center, upper_start, d_max, color='k', lw=2, zorder=1)
                        
            # Add vertical separation lines between every two cases
            for i in range(1, len(case_names)+1):
                ax.axvline(i+0.5, color='gray', linestyle='--', lw=1, alpha=0.5, zorder=1)

            # Labels and title
            ax.set_xticks(range(1, len(case_names)+1))
            ax.set_xticklabels(case_names, fontsize=20)
            # ax.set_ylabel("Simulated Streamflow (m³/s)", fontsize=12)
            ax.set_title(title, fontsize=20)
            ax.set_xlim(0.5, len(case_names)+0.5)

        # High-flow plot
        fig_high_flow, ax = plt.subplots(figsize=(6,2))
        plot_box_scatter(ax, data_high, "High Flow")
        plt.tight_layout()

        # Low-flow plot
        fig_low_flow, ax = plt.subplots(figsize=(6,2))
        plot_box_scatter(ax, data_low, "Low Flow")
        plt.tight_layout()
        return fig_high_flow, fig_low_flow
    
    fig_high_flow, fig_low_flow = plot_shiquan_high_low_sim_beauty(sim, obs)
    fig_high_flow.savefig(os.path.join(evb_dir_modeling.CalibrateVIC_dir, "case_40_best_inds", f"Exp1_1578_compare_highflow.tiff"), dpi=300)
    fig_low_flow.savefig(os.path.join(evb_dir_modeling.CalibrateVIC_dir, "case_40_best_inds", f"Exp1_1578_compare_lowflow.tiff"), dpi=300)
    
    plt.show(block=True)

if __name__ == "__main__":
    # set date
    warmup_date, warmup_date_eval, calibrate_date, calibrate_date_eval, verify_date, verify_date_eval, total_date, total_date_eval, plot_date_dict = set_date(timestep, timestep_evaluate)
    
    # plot results
    # plot_performance_table()
    # plot_taylor()
    # plot_metrics_evaluation()
    # plot_scatter_comparison()
    # plot_fdc()
    # plot_water_balance_comparison_pie()
    # plot_spatial_pattern()
    # plot_parameter_identifiability()
    # plot_calibration_process(case_name="case1")
    # plot_calibration_process(case_name="case5")
    # plot_soil_layer_depths_compare_temporalpattern(ind=0)
    # plot_soil_layer_depths_compare_spatialpattern(ind=0)
    # plot_soil_layer_depths_compare_params_flows(ind=0)
    # plot_RVIC_parameters_compare_res()
    # plot_performance_statistical_evaluation()
    plot_high_flow_low_flow_compare_Exp_1()