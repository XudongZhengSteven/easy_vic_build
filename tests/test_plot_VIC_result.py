# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from easy_vic_build import Evb_dir
from easy_vic_build.tools.utilities import *
from easy_vic_build.tools.calibrate_func.evaluate_metrics import *
from easy_vic_build.tools.plot_func.plot_func import *
from netCDF4 import Dataset
from netCDF4 import num2date
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredText

plt.rcParams['font.family']='Arial'
plt.rcParams['font.size']=12
plt.rcParams['font.weight']='normal'

def read_VIC_result(basin_index=397, model_scale="12km"):
    case_name = f"{basin_index}_{model_scale}"
    
    # set evb_dir
    evb_dir = Evb_dir(cases_home="./examples")
    evb_dir.builddir(case_name)
    
    # read VIC result
    cali_result = pd.read_csv(os.path.join(evb_dir.VICResults_dir, "cali_result.csv"), index_col=0)
    verify_result = pd.read_csv(os.path.join(evb_dir.VICResults_dir, "verify_result.csv"), index_col=0)
    simulated_dataset = Dataset(os.path.join(evb_dir.VICResults_dir, [fn for fn in os.listdir(evb_dir.VICResults_dir) if fn.endswith(".nc")][0]))    
    
    return evb_dir, cali_result, verify_result, simulated_dataset


def test_plot_VIC_performance(evb_dir, cali_result, verify_result):
    fig, _ = plot_VIC_performance(cali_result, verify_result)
    fig.savefig(os.path.join(evb_dir.VICResults_dir, "test_plot_VIC_performance.tiff"), dpi=300)


def test_plot_VIC_streamflow_transferability(evb_dirs, cali_results, verify_results, model_names, model_colors,
                                             cali_names_ha, cali_names_va, verify_names_ha, verify_names_va):
    
    # simulation comparison: streamflow
    # Talyor diagram
    obs_cali = cali_result_12km["obs_cali discharge(m3/s)"].values
    obs_verify = verify_result_12km["obs_verify discharge(m3/s)"].values
    obs_total = np.concatenate([obs_cali, obs_verify])
    models_cali = [cali_result["sim_cali discharge(m3/s)"].values for cali_result in cali_results]
    models_verify = [verify_result["sim_verify discharge(m3/s)"].values for verify_result in verify_results]
    models_total = [np.concatenate([models_cali[i], models_verify[i]]) for i in range(len(models_cali))]
    
    fig_taylor = plt.figure(figsize=(12, 6))
    fig_taylor.subplots_adjust(left=0.08, right=0.92, bottom=0.01, top=0.9, wspace=0.3)
    ax1 = fig_taylor.add_subplot(121, projection='polar')
    ax2 = fig_taylor.add_subplot(122, projection='polar')
    
    fig_taylor, ax1 = taylor_diagram(obs_cali, models_cali, model_names, cali_names_ha, cali_names_va, model_colors=model_colors, title="(a) Calibration", fig=fig_taylor, ax=ax1)
    fig_taylor, ax2 = taylor_diagram(obs_verify, models_verify, model_names, verify_names_ha, verify_names_va, model_colors=model_colors, title="(b) Verification", fig=fig_taylor, ax=ax2)
    
    # highflow lowflow comparison
    fig_streamflow_scatter, _ = plot_multimodel_comparison_scatter(obs_total, models_total, model_names, model_colors=model_colors)
    
    # TODO seasonal comparison
    # date_total = np.concatenate([cali_result_12km.index, verify_result_12km.index])
    # obs_total_df = pd.DataFrame(obs_total, index=date_total, columns=["obs_total discharge(m3/s)"])
    # models_total_df = [pd.DataFrame(models_total[i], index=date_total, columns=[f"sim_total discharge(m3/s)_{model_names[i].strip()}"]) for i in range(len(models_total))]
    # all_df = pd.concat([obs_total_df] + models_total_df, axis=1)
    # months = list(range(1, 13))
    
    # save
    fig_taylor.savefig(os.path.join(evb_dirs[0]._cases_dir, "test_plot_VIC_transferability_taylor.tiff"), dpi=300)
    fig_streamflow_scatter.savefig(os.path.join(evb_dirs[0]._cases_dir, "test_plot_VIC_transferability_streamflow_scatter.tiff"), dpi=300)
    

def test_plot_distributed_simulation_comparison(evb_dirs, cali_results, verify_results, simulated_datasets, model_names, model_colors):
    # set period
    event_period = ["20010201", "20010331"]  # ... days
    rising_period = ["20010222", "20010226"]  # 5 days
    recession_period = ["20010309", "20010313"]  # 5 days
    
    # event_period = ["20020505", "20020626"]  # ... days
    # rising_period = ["20020507", "20020511"]  # 5 days
    # recession_period = ["20020518", "20020522"]  # 5 days
    # recession_period = ["20020520", "20020524"]  # 5 days
    
    # get prec
    MeteForcing_year_dataset = Dataset(os.path.join(evb_dirs[0].MeteForcing_dir, f"{evb_dirs[0]._forcing_prefix}.{event_period[0][:4]}{evb_dirs[0]._MeteForcing_src_suffix}"))
    start_date = pd.Timestamp(f"{event_period[0]}0000")
    end_date = pd.Timestamp(f"{event_period[1]}2300")
    MeteForcing_times = MeteForcing_year_dataset.variables["time"]
    datasets_dates = num2date(MeteForcing_times[:], units=MeteForcing_times.units, calendar=MeteForcing_times.calendar)
    datasets_datetime_index = pd.to_datetime([date.strftime('%Y-%m-%d %H:%M:%S') for date in datasets_dates])
    index_start = datasets_datetime_index.get_loc(start_date)
    index_end = datasets_datetime_index.get_loc(end_date)
    
    MeteForcing_timeseries = np.nanmean(MeteForcing_year_dataset.variables["prcp"][index_start:index_end+1, :, :], axis=1)
    MeteForcing_timeseries = np.nanmean(MeteForcing_timeseries, axis=1)
    MeteForcing_df = pd.DataFrame(MeteForcing_timeseries, index=datasets_datetime_index[index_start:index_end+1], columns=["prcp mm"])
    MeteForcing_df = MeteForcing_df.resample("D").sum().fillna(0)
    fig_multimodel_comparison_distributed_OUTPUT = plot_multimodel_comparison_distributed_OUTPUT(cali_results, verify_results, simulated_datasets, MeteForcing_df,
                                                                                                 model_names, model_colors,
                                                                                                 event_period, rising_period, recession_period)
    plt.show()
    fig_multimodel_comparison_distributed_OUTPUT.savefig(os.path.join(evb_dirs[0]._cases_dir, "test_plot_distributed_simulation_comparison.tiff"), dpi=300)

def test_plot_params(evb_dir, params_dataset_level0, params_dataset_level1):
    fig_params_level0, axes = plot_params(params_dataset_level0)
    fig_params_level1, axes = plot_params(params_dataset_level1)
    
    fig_params_level0.savefig(os.path.join(evb_dir.ParamFile_dir, "fig_params_level0.tiff"), dpi=300)
    fig_params_level1.savefig(os.path.join(evb_dir.ParamFile_dir, "fig_params_level1.tiff"), dpi=300)
    
    
if __name__ == "__main__":
    # ------------- plot VIC performance -------------
    # evb_dir, cali_result, verify_result, simulated_dataset = read_VIC_result(basin_index=397, model_scale="12km")
    # test_plot_VIC_performance(evb_dir, cali_result, verify_result)
    # simulated_dataset.close()
    
    # ------------- plot VIC transferability -------------
    evb_dir_12km, cali_result_12km, verify_result_12km, simulated_dataset_12km = read_VIC_result(basin_index=397, model_scale="12km")
    evb_dir_8km, cali_result_8km, verify_result_8km, simulated_dataset_8km = read_VIC_result(basin_index=397, model_scale="8km_transferability")
    evb_dir_6km, cali_result_6km, verify_result_6km, simulated_dataset_6km = read_VIC_result(basin_index=397, model_scale="6km_transferability")
    
    params_dataset_level0_12km, params_dataset_level1_12km = readParam(evb_dir_12km)
    params_dataset_level0_8km, params_dataset_level1_8km = readParam(evb_dir_8km)
    params_dataset_level0_6km, params_dataset_level1_6km = readParam(evb_dir_6km)
    
    evb_dirs = [evb_dir_12km, evb_dir_8km, evb_dir_6km]
    
    cali_results = [cali_result_12km, cali_result_8km, cali_result_6km]
    verify_results = [verify_result_12km, verify_result_8km, verify_result_6km]
    
    simulated_datasets = [simulated_dataset_12km, simulated_dataset_8km, simulated_dataset_6km]
    params_dataset_level0_sets = [params_dataset_level0_12km, params_dataset_level0_8km, params_dataset_level0_6km]
    params_dataset_level1_sets = [params_dataset_level1_12km, params_dataset_level1_8km, params_dataset_level1_6km]
    model_names = ["12km ", "8km ", "6km "]
    model_colors = ["red", "blue", "green"]
    cali_names_ha = ["left", "right", "left"]  # {'center', 'right', 'left'}
    cali_names_va = ["bottom", "top", "bottom"]  # {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
    verify_names_ha = ["left", "right", "left"]
    verify_names_va = ["bottom", "top", "bottom"]
    
    # streamflow comparison
    # test_plot_VIC_streamflow_transferability(evb_dirs, cali_results, verify_results, model_names, model_colors,
    #                                          cali_names_ha, cali_names_va, verify_names_ha, verify_names_va)
    
    # distributed comparison
    test_plot_distributed_simulation_comparison(evb_dirs, cali_results, verify_results, simulated_datasets, model_names, model_colors)
    
    # params comparison
    # test_plot_params(evb_dir_12km, params_dataset_level0_12km, params_dataset_level1_12km)
    
    # close
    # [simulated_datasets.close() for simulated_datasets in simulated_datasets]
    # [params.close() for params in params_dataset_level0_sets]
    # [params.close() for params in params_dataset_level1_sets]
    