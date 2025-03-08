# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
from .extractData_func import *

# ------------------------ read data into basins ------------------------
class readDataIntoBasins_API:
    def readCAMELSStreamflowIntoBasins(basin_shp, read_dates=None):
        # pd.date_range(start=read_dates[0], end=read_dates[1], freq="D")
        basin_shp = Extract_CAMELS_Streamflow.ExtractData(basin_shp, read_dates=read_dates)
        
        return basin_shp

    def readCAMELSAttributeIntoBasins(basin_shp, k_list=None):
        basin_shp = Extract_CAMELS_Attribute.ExtractData(basin_shp, k_list=k_list)
        return basin_shp
