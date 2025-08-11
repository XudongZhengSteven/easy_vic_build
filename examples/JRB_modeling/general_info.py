# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import numpy as np
import pandas as pd

# general info
scalemap = {
    "3km": 0.025,
    "6km": 0.055,
    "8km": 0.072,
    "12km": 0.11,
    "1_32_deg": 1/32,  # ~3.5km
    "1_16_deg": 1/16,  # ~7km
    "1_8_deg": 1/8,  # ~13.8km, maybe the prefer scale
    "1_6_deg": 1/6, # ~18.5km
    "1_4_deg": 1/4,  # ~27km
    "1_2_deg": 1/2,  # ~55km
    "1_grid": None,
}

basin_outlets_reference_i_map = {
    "Zhangjiashan": 1, "Yangjiapin": 0, "Qingyang": 2,
    "2021042103p": 4, "2022092592p": 3,
}

stationdata_fname_map = {
    "Zhangjiashan": "stationdata_Zhangjiashan_daily_1960_2020.txt",
    "Yangjiapin": "stationdata_Yangjiapin_daily_1956_2020.txt",
    "Qingyang": "stationdata_Qinyang_daily_1956_2020.txt",
    "2022092592p": "stationdata_2022092592p_Intermittent_2020_2022.txt",
    "2021042103": "stationdata_2021042103p_Intermittent_2016_2024.txt",
}

# set configuration
station_name = "Zhangjiashan"
model_scale = "1_8_deg"
timestep = "3h"
timestep_evaluate = "D"

date_period=["20080101 00:00:00", "20181231 21:00:00"]
warmup_date_period = ["20080101 00:00:00", "20091231 21:00:00"]
calibrate_date_period = ["20100101 00:00:00", "20151231 21:00:00"]
verify_date_period = ["20160101 00:00:00", "20181231 21:00:00"]

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
