# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
import pandas as pd
import matplotlib.pyplot as plt

def ExtractData(basin_shp, date_period=None, fname="stationdata_Zhangjiashan_daily_1960_2020.txt", plot=False):
    # general set
    data_dir = "F:\\research\\Research\\JRB_scaling\\data\\streamflow"
    fpath = os.path.join(data_dir, fname)
    
    # read
    data_df = pd.read_csv(fpath, sep="\t")
    
    # set dateindex
    data_df.index = pd.to_datetime(data_df.date, format="%Y/%m/%d")
    
    # extract for read_dates
    if date_period is not None:
        extracted_data_df = data_df.loc[date_period[0]: date_period[1]]
    else:
        extracted_data_df = data_df
    
    # drop
    extracted_data_df = extracted_data_df.drop("date", axis=1)
    
    # save
    extracted_data_df = extracted_data_df.astype("float")
    basin_shp["stationdata_streamflow"] = [extracted_data_df]
    
    # plot
    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(extracted_data_df.index, extracted_data_df["Q(m3/s)"].to_list(), label="Streamflow")
        ax.set_xlabel("Date")
        ax.set_ylabel("Streamflow (m3/s)")
        ax.set_title("Streamflow Data")
        ax.set_xlim([extracted_data_df.index[0], extracted_data_df.index[-1]])
        ax.set_ylim([0, extracted_data_df["Q(m3/s)"].max() * 1.2])
        ax.legend()
        plt.show(block=True)
    
    return basin_shp
    