# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
from .extractData_func import *

def selectBasinremovingStreamflowMissing(basin_shp, date_period=["19980101", "20101231"]):
    # get remove streamflow missing
    streamflows_dict_original, streamflows_dict_removed_missing = Extract_CAMELS_Streamflow.getremoveStreamflowMissing(date_period)
    remove_num = len(streamflows_dict_original["usgs_streamflows"]) - len(streamflows_dict_removed_missing["usgs_streamflows"])
    print(f"remove Basin based on StreamflowMissing: remove {remove_num} files")
    
    # get ids removed missing
    streamflow_ids_removed_missing = streamflows_dict_removed_missing["streamflow_ids"]
    index_removed_missing = [id in streamflow_ids_removed_missing for id in basin_shp.hru_id.values]
    
    # remove
    basin_shp = basin_shp.iloc[index_removed_missing, :]

    print(f"remain {len(basin_shp)}")
    return basin_shp


def selectBasinBasedOnArea(basin_shp, min_area, max_area):
    print(f"select Basin based on Area: {min_area} - {max_area}")
    basin_shp = basin_shp.loc[(basin_shp.loc[:, "AREA_km2"] >= min_area) & (basin_shp.loc[:, "AREA_km2"] <= max_area), :]

    print(f"remain {len(basin_shp)}")
    return basin_shp


def selectBasinStreamflowWithZero(basin_shp, usgs_streamflow, streamflow_id, zeros_min_num=100):
    """ select basin with zero streamflow (many low flows) """
    # loop for each basin
    selected_id = []
    print(f"select Basin based on StreamflowWithZero, zeros_min_num is {zeros_min_num}")
    for i in range(len(usgs_streamflow)):
        usgs_streamflow_ = usgs_streamflow[i]
        streamflow = usgs_streamflow_.iloc[:, 4].values
        if sum(streamflow == 0) > zeros_min_num:  # find basin with zero streamflow
            selected_id.append(streamflow_id[i])
            print(f"nums of zero value: {sum(streamflow == 0)}")
            # plt.plot(streamflow)
            # plt.ylim(bottom=0)
            # plt.show()

    selected_index = [id in selected_id for id in basin_shp.hru_id.values]
    basin_shp = basin_shp.iloc[selected_index, :]

    print(f"remain {len(basin_shp)}")
    return basin_shp


def selectBasinBasedOnAridity(basin_shp, aridity):
    pass


def selectBasinBasedOnElevSlope(basin_shp, elev_slope):
    pass
