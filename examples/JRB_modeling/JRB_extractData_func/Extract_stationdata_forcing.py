# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
import pandas as pd


def filter_stations(date_period):
    JRB_meteStations_path = "F:\\research\\Research\\JRB_scaling\\data\\map\\JRB_meteStations.csv"
    CMA_meteStations_path = "F:\\research\\Research\\JRB_scaling\\data\\forcing\\CMA_climate_daily_dataset_v3\\SURF_CLI_CHN_MUL_DAY_V3.0\\documents\\SURF_CLI_CHN_MUL_DAY_STATION.txt"
    
    JRB_meteStations_df = pd.read_csv(JRB_meteStations_path, sep=",")
    CMA_meteStations_df = pd.read_csv(CMA_meteStations_path, sep="\t")
    
    for i in JRB_meteStations_df.index:
        stationid = int(JRB_meteStations_df.loc[i, "stationid"])
        station_valid_bool = stationid in CMA_meteStations_df.loc[:, "区站号"].to_list()
        JRB_meteStations_df.loc[i, "station_valid_bool"] = station_valid_bool
        JRB_meteStations_df.loc[i, "start_date"] = int(CMA_meteStations_df.loc[CMA_meteStations_df["区站号"] == stationid, "开始年月"].values[0])
        JRB_meteStations_df.loc[i, "end_date"] = int(CMA_meteStations_df.loc[CMA_meteStations_df["区站号"] == stationid, "结束年月"].values[0])
        
    # not valid
    JRB_meteStations_df_notvalid = JRB_meteStations_df.loc[JRB_meteStations_df["station_valid_bool"] == False, :]
    
    print(f"------ not valid station id: contained in CMA data ({len(JRB_meteStations_df_notvalid)}) -------")
    print(JRB_meteStations_df_notvalid)
    
    # valid
    JRB_meteStations_df = JRB_meteStations_df.loc[JRB_meteStations_df["station_valid_bool"] == True, :]
    JRB_meteStations_df = JRB_meteStations_df.reset_index(drop=True)
    
    # period filter
    period_start = pd.to_datetime(date_period[0], format="%Y%m")
    period_end = pd.to_datetime(date_period[1], format="%Y%m")
    
    for i in JRB_meteStations_df.index:
        start_date = pd.to_datetime(str(int(JRB_meteStations_df.loc[i, "start_date"])), format="%Y%m")
        end_date = pd.to_datetime(str(int(JRB_meteStations_df.loc[i, "end_date"])), format="%Y%m")
        
        station_date_valid_bool = (start_date <= period_start) & (end_date >= period_end)
        JRB_meteStations_df.loc[i, "date_valid_bool"] = station_date_valid_bool
    
    
    JRB_meteStations_df_notvalid = JRB_meteStations_df.loc[JRB_meteStations_df["date_valid_bool"] == False, :]
    print(f"------ not valid station id: period ({len(JRB_meteStations_df_notvalid)}) -------")
    print(JRB_meteStations_df_notvalid)
    
    JRB_meteStations_df = JRB_meteStations_df.loc[JRB_meteStations_df["date_valid_bool"] == True, :]
    JRB_meteStations_df = JRB_meteStations_df.reset_index(drop=True)
    
    # save
    JRB_meteStations_df.to_csv(f"F:\\research\\Research\\JRB_scaling\\data\\map\\JRB_meteStations_combined_CMA_cover_{date_period[0]}_{date_period[1]}.csv", sep=",", index=False)
    
if __name__ == "__main__":
    filter_stations(date_period=[198001, 201212])

