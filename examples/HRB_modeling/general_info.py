import numpy as np
import pandas as pd
from easy_vic_build import logger
from easy_vic_build.tools.nested_basin_func.nested_basin_func import get_all_upstreams, get_topo_order
from HRB_build_evb_dir import build_modeling_dir

# general info
scalemap = {
    "3km": 0.025,
    "6km": 0.055,
    "8km": 0.072,
    "12km": 0.11,
    "1_32_deg": 1/32,  # ~3.5km
    "1_20_deg": 1/20,  # ~5.5km, 0.05deg
    "1_16_deg": 1/16,  # ~7km, 0.0625deg
    "1_14_deg": 1/14,  # ~8km, 0.071deg
    "1_12_deg": 1/12,  # ~9.2km, 0.083deg
    
    # used scales
    "1_10_deg": 1/10,  # ~11km, 0.1deg (CMFD scale)
    "1_8_deg": 1/8,  # ~13.8km, 0.125deg, maybe the prefer scale
    "1_6_deg": 1/6, # ~18.5km, 0.166deg
    "1_5_deg": 1/5, # ~22.2km, 0.2deg
    "1_4_deg": 1/4,  # ~27km, 0.25deg
    "1_2_deg": 1/2,  # ~55km, 0.5deg
    "1_deg": 1, # ~111km, 1deg
    "1_grid": None,
}

# set stations
station_name = "shiquan"  # control total basin
station_names = ["hanzhong", "yangxian", "youshui", "lianghekou", "shiquan"]  # "wuhou", 

station_coords = {
    # "wuhou": (33.146500, 106.616666),
    "hanzhong": (33.049000, 107.023315),
    "yangxian": (33.218708, 107.536583),
    "youshui": (33.267975, 107.766781),
    "lianghekou": (33.26325, 108.06896),
    "shiquan": (33.038635, 108.240737),
}  # modified

nest_upstream_map = {
    "hanzhong": [],
    "yangxian": ["hanzhong"],
    "youshui": [],
    "lianghekou": [],
    "shiquan": ["hanzhong", "yangxian", "youshui", "lianghekou"],
}

topo_station_order = get_topo_order(station_names, nest_upstream_map)

station_coords_df = pd.DataFrame({
    "station_name": station_names,
    "lat": [station_coords[name][0] for name in station_names],
    "lon": [station_coords[name][1] for name in station_names],
})

boundary = [105.6, 32, 109, 34.8]

basin_outlets_reference_i_map = {
    "hanzhong": 0, "yangxian": 1,
    "youshui": 2, "lianghekou": 3, "shiquan": 4,
}


# set configuration
model_scale = "6km"
timestep = "3h"
timestep_evaluate = "D"

date_period=["20030101 00:00:00", "20181231 21:00:00"]
warmup_date_period = ["20030101 00:00:00", "20041231 21:00:00"]
calibrate_date_period = ["20050101 00:00:00", "20141231 21:00:00"]
verify_date_period = ["20150101 00:00:00", "20181231 21:00:00"]

date = pd.date_range(date_period[0], date_period[1], freq=timestep)
warmup_date = pd.date_range(warmup_date_period[0], warmup_date_period[1], freq=timestep)
calibrate_date = pd.date_range(calibrate_date_period[0], calibrate_date_period[1], freq=timestep)
verify_date = pd.date_range(verify_date_period[0], verify_date_period[1], freq=timestep)

date_evaluate = pd.date_range(date_period[0], date_period[1], freq=timestep_evaluate)
warmup_date_evaluate = pd.date_range(warmup_date_period[0], warmup_date_period[1], freq=timestep_evaluate)
calibrate_date_evaluate = pd.date_range(calibrate_date_period[0], calibrate_date_period[1], freq=timestep_evaluate)
verify_date_evaluate = pd.date_range(verify_date_period[0], verify_date_period[1], freq=timestep_evaluate)

reverse_lat=True

# set scale level
grid_res_level0= 0.00833
grid_res_level1=scalemap[model_scale]
grid_res_level2=scalemap[model_scale]  # to be consistent with level1

# build evb
try:
    sub_name = "hydroanalysis"
    evb_dir_hydroanalysis = build_modeling_dir(subname=sub_name)
    logger.info(f"create evb_dir: {sub_name}")
except:
    logger.info("hydroanalysis dir exists")
    
try:
    sub_name = f"{station_name}_{model_scale}"
    evb_dir_modeling = build_modeling_dir(subname=sub_name)
    logger.info(f"create evb_dir: {sub_name}")
except:
    logger.info("modeling dir exists")
    
# attach logger
logger = evb_dir_modeling.attach_logger_file(logger)

# model configuration summary
logger.info(
    "=== Model configuration summary ===\n"
    "Spatial configuration:\n"
    "  - Model scale            : %s\n"
    "  - Grid resolution level0 : %.5f°\n"
    "  - Grid resolution level1 : %.5f°\n"
    "  - Grid resolution level2 : %.5f° (consistent with level1)\n"
    "Temporal configuration:\n"
    "  - Simulation timestep       : %s\n"
    "  - Evaluation timestep       : %s\n"
    "  - Simulation period         : %s to %s (%d steps)\n"
    "  - Warm-up period            : %s to %s (%d / %d steps)\n"
    "  - Calibration period        : %s to %s (%d / %d steps)\n"
    "  - Verification period       : %s to %s (%d / %d steps)\n"
    "Other configuration:\n"
    "  - Reverse latitude          : %s",
    # Spatial
    model_scale,
    grid_res_level0,
    grid_res_level1,
    grid_res_level2,
    
    # Temporal
    timestep,
    timestep_evaluate,
    date_period[0], date_period[1], len(date),
    warmup_date_period[0], warmup_date_period[1], len(warmup_date), len(warmup_date_evaluate),
    calibrate_date_period[0], calibrate_date_period[1], len(calibrate_date), len(calibrate_date_evaluate),
    verify_date_period[0], verify_date_period[1], len(verify_date), len(verify_date_evaluate),
    
    # Other
    reverse_lat
)