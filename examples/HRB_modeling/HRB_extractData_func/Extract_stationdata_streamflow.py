# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
import pandas as pd
import matplotlib.pyplot as plt

def ExtractData(basin_shp, station_names, date_period=None, plot=False):
    # general set
    data_dir = "F:\\research\\Research\\ModelingUncertainty_hanjiang\\data\\streamflow"
    
    # read
    data_dict = {}
    for station_name in station_names:
        data_dict[station_name] = pd.read_csv(os.path.join(data_dir, f"{station_name}.txt"), sep="\t")
    
    # process
    for key in data_dict.keys():
        # set dateindex
        data_df = data_dict[key]
        data_df.index = pd.to_datetime(data_df.Date, format="%Y/%m/%d")
    
        # extract for read_dates
        if date_period is not None:
            data_df = data_df.loc[date_period[0]: date_period[1]]
        
        # drop
        data_df = data_df.drop("Date", axis=1)
    
        # save
        data_df = data_df.astype("float")
        data_dict[key] = data_df
        
        basin_shp[f"stationdata_streamflow_{key}"] = [data_df]
    
    # plot
    if plot:
        fig, axes = plt.subplots(len(station_names), 1, figsize=(10, 5), sharex=True)
        
        for key, ax in zip(data_dict.keys(), axes):
            data_df = basin_shp[f"stationdata_streamflow_{key}"][0]
            ax.plot(data_df.index, data_df.values, label=key, linewidth=1)

            ax.set_xlim([data_df.index[0], data_df.index[-1]])
            ax.set_ylim([0, None])
            
            if ax == axes[-1]:
                ax.set_xlabel("Date")
            ax.set_ylabel("Streamflow (m3/s)")
            ax.legend(loc="upper right")
        plt.show(block=True)
    
    return basin_shp
    