# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import os
from netCDF4 import Dataset
from tqdm import *


class create_nc:
    """ create nc class """

    def __init__(self):
        pass

    def __call__(self, nc_path, dimensions, variables, var_value, return_dataset=False):
        """ call function
        input:
            nc_path: the path of created nc file
            dimensions: dict, {"dimname": size}, general contains lon, lat, time, time_bound. If a dimension is
                    unlimitied, set size None or 0
            variables: dict, {"varname": kwargs}, general contains dimension variables and data variables
                    kwargs is a dict, which could contain below:
                        datatype, dimensions=(), zlib=False, complevel=4, shuffle=True, fletcher32=False, contiguous=False,
                        chunksizes=None, endian='native', least_significant_digit=None, fill_value=None
            var_value: dict, {"varname": values}, general is a np.ma.array()
            return_dataset: bool, whether return dataset to set attributes or close file

        output:
            nc_path file
        """
        dataset = Dataset(nc_path, "w")

        # create dimension
        for key in dimensions:
            dataset.createDimension(dimname=key, size=dimensions[key])

        # create dimension variable
        for key in variables:
            dataset.createVariable(key, **variables[key])

        # Assign values to variables
        for key in var_value:
            dataset.variables[key][:] = var_value[key]

        if return_dataset:
            return dataset
        else:
            dataset.close()

    def copy_garrtibutefunc(self, dst_dataset_path, src_dataset_path):
        """ copy global attributes of src_dataset to dst_dataset """
        with Dataset(src_dataset_path, "r") as src_dataset:
            with Dataset(dst_dataset_path, "a") as dst_dataset:
                for key in src_dataset.ncattrs():
                    print(f"set global attribute {key}")
                    dst_dataset.setncattr(key, src_dataset.getncattr(key))

    def copy_vattributefunc(self, dst_dataset_path, src_dataset_path):
        """ copy variable attributes of src_dataset to dst_dataset """
        with Dataset(src_dataset_path, "r") as src_dataset:
            with Dataset(dst_dataset_path, "a") as dst_dataset:
                # loop for each variable in dst_dataset
                for key in dst_dataset.variables:
                    print(f"set variable attribute for {key}")

                    # get variables attributes from src_dataset
                    ncattr_dict = dict(((key_nacttr, src_dataset.variables[key].getncattr(key_nacttr)) for key_nacttr
                                        in src_dataset.variables[key].ncattrs()))

                    # loop for setting each attributes
                    for key_nacttr in ncattr_dict.keys():
                        if key_nacttr != "_FillValue":  # "_FillValue" should be set when create variable
                            print(f"    set variable attributes {key_nacttr}")
                            dst_dataset.variables[key].setncattr(key_nacttr, ncattr_dict[key_nacttr])


def copy_vattributefunc(src_var, dst_var):
    ncattr_dict = dict(((key_nacttr, src_var.getncattr(key_nacttr)) for key_nacttr in src_var.ncattrs()))
    
    for key_nacttr in ncattr_dict.keys():
        if key_nacttr != "_FillValue":  # "_FillValue" should be set when create variable
            dst_var.setncattr(key_nacttr, ncattr_dict[key_nacttr])


def copy_garrtibutefunc(src_dataset, dst_dataset):
    """ copy global attributes of src_dataset to dst_dataset """
    for key in src_dataset.ncattrs():
        dst_dataset.setncattr(key, src_dataset.getncattr(key))


def demos1():
    import os
    from netCDF4 import Dataset
    import numpy as np
    import pandas as pd
    import time as t

    # general set
    home = 'H:/research/flash_drough/GLDAS_Noah'
    data_path = os.path.join(home, 'SoilMoi0_100cm_inst_19480101_20141231_Pentad_muldis_SmPercentile.npy')
    out_path = os.path.join(home, 'SoilMoi0_100cm_inst_19480101_20141231_Pentad_muldis_SmPercentile.nc4')
    refernce_path = "H:/GIT/Python/yanxiang_1_2/gldas/GLDAS_NOAH025_M.A194801.020.nc4"
    coord_path = "H:/research/flash_drough/coord.txt"
    det = 0.25

    # load data
    data = np.load(data_path)
    reference = Dataset(refernce_path, mode='r')
    coord = pd.read_csv(coord_path, sep=",")
    mask_value = reference.missing_value

    # create full array
    data_lon = coord.lon.values
    data_lat = coord.lat.values
    extent = [min(data_lon), max(data_lon), min(data_lat), max(data_lat)]
    array_data = np.full((data.shape[0],
                          int((extent[3] - extent[2]) / det + 1),
                          int((extent[1] - extent[0]) / det + 1)),
                         fill_value=mask_value, dtype='float32')

    # array_data_lon/lat is the center point
    array_data_lon = np.linspace(extent[0], extent[1],
                                 num=int((extent[1] - extent[0]) / det + 1))
    array_data_lat = np.linspace(extent[2], extent[3],
                                 num=int((extent[3] - extent[2]) / det + 1))

    # cal coord index
    lat_index = []
    lon_index = []
    for i in range(len(coord)):
        lat_index.append(np.where(array_data_lat == coord["lat"][i])[0][0])
        lon_index.append(np.where(array_data_lon == coord["lon"][i])[0][0])

    # put the data into the full array based on index
    for i in range(data.shape[0]):
        for j in range(data.shape[1] - 1):
            array_data[i, lat_index[j], lon_index[j]] = data[i, j + 1]

    # mask array
    mask = array_data == mask_value
    masked_array_data = np.ma.masked_array(array_data, mask=mask, fill_value=mask_value)

    # time
    time = data[:, 0]
    time = time.flatten()

    # set dimensions and varaibels
    cn = create_nc()
    dimensions = {"lon": array_data_lon.size,
                  "lat": array_data_lat.size,
                  "time": data.shape[0],
                  "bnds": reference.dimensions["bnds"].size
                  }

    variables = {"time_bnds": {"datatype": "f4", "dimensions": ("time", "bnds")},
                 "lon": {"datatype": "f4", "dimensions": ("lon",)},
                 "lat": {"datatype": "f4", "dimensions": ("lat",)},
                 "time": {"datatype": "f4", "dimensions": ("time",)},
                 "SM_percentile": {"datatype": "f8", "dimensions": ("time", "lat", "lon")}
                 }

    # set var_value
    var_value = {"time_bnds": reference.variables["time_bnds"][:],
                 "lon": array_data_lon,
                 "lat": array_data_lat,
                 "time": time,
                 "SM_percentile": masked_array_data,
                 }

    # cn -> create and set dataset
    dataset = cn(nc_path=out_path, dimensions=dimensions, variables=variables, var_value=var_value, return_dataset=True)

    # set global attributes
    dataset.history = f"created on date: {t.ctime(t.time())}"
    dataset.title = "Soil moisture percentile calculated by GLDAS 2.0 dataset on pentad scale used for CDFDI framework"
    dataset.missing_value = mask_value
    dataset.DX = 0.25
    dataset.DY = 0.25

    # set variables attributes
    dataset.variables["lat"].units = "degrees_north"
    dataset.variables["lon"].units = "degrees_east"
    dataset.variables["time"].format = "%Y%m%d"
    dataset.variables["SM_percentile"].units = "percentile / pentad"
    dataset.variables["SM_percentile"].cal_method = "Calculated by fitting probability distribution of soil moisture"

    # close
    dataset.close()
    reference.close()

def demo_combineTRMM_P_add_time_dim():
    # general
    src_home = "E:/data/hydrometeorology/TRMM_P/TRMM_3B42/data"
    dst_home = "E:/data/hydrometeorology/TRMM_P/TRMM_3B42"
    suffix = ".nc4"
    
    src_names = [n for n in os.listdir(src_home) if n.endswith(suffix)]
    src_paths = [os.path.join(src_home, n) for n in src_names]
    
    # set time
    date_period = pd.date_range("19980101", "20191230", freq="1D")
    date_period_str = list(date_period.strftime("%Y%m%d"))
    from datetime import datetime
    date_period_datetime = [datetime.strptime(d, "%Y%m%d") for d in date_period_str]
    
    # create dataset
    dst_path = os.path.join(dst_home, f"3B42_Daily_combined{suffix}")
    dst_dataset = Dataset(dst_path, "w")
    
    # create time dimension
    dst_dataset.createDimension("time", len(src_names))
    dst_times = dst_dataset.createVariable("time", "f8", ("time", ))
    dst_times.units = "days since 1998-01-01"
    dst_times.calendar = "gregorian"
    from netCDF4 import date2num
    dst_times[:] = date2num(date_period_datetime, units=dst_times.units, calendar=dst_times.calendar)
    
    # read one file to copy
    with Dataset(src_paths[0], "r") as src_dataset:
        src_lon = src_dataset.variables["lon"][:]
        src_lat = src_dataset.variables["lat"][:]
        
        # create lon/lat dimension
        dst_dataset.createDimension("lon", len(src_lon))
        dst_dataset.createDimension("lat", len(src_lat))
        
        # create variables
        dst_lon = dst_dataset.createVariable("lon", "f4", ("lon", ))
        dst_lat = dst_dataset.createVariable("lat", "f4", ("lat", ))
        dst_precipitation = dst_dataset.createVariable("precipitation", "f4", ("time", "lon", "lat", ), fill_value=src_dataset.variables["precipitation"].getncattr("_FillValue"))
        dst_precipitation_cnt = dst_dataset.createVariable("precipitation_cnt", "i2", ("time", "lon", "lat", ))
        dst_IRprecipitation = dst_dataset.createVariable("IRprecipitation", "f4", ("time", "lon", "lat", ), fill_value=src_dataset.variables["IRprecipitation"].getncattr("_FillValue"))
        dst_IRprecipitation_cnt = dst_dataset.createVariable("IRprecipitation_cnt", "i2", ("time", "lon", "lat", ))
        dst_HQprecipitation = dst_dataset.createVariable("HQprecipitation", "f4", ("time", "lon", "lat", ), fill_value=src_dataset.variables["HQprecipitation"].getncattr("_FillValue"))
        dst_HQprecipitation_cnt = dst_dataset.createVariable("HQprecipitation_cnt", "i2", ("time", "lon", "lat", ))
        dst_randomError = dst_dataset.createVariable("randomError", "f4", ("time", "lon", "lat", ), fill_value=src_dataset.variables["randomError"].getncattr("_FillValue"))
        dst_randomError_cnt = dst_dataset.createVariable("randomError_cnt", "i2", ("time", "lon", "lat", ))
        
        # set/copy lon/lat variables
        dst_lon[:] = src_dataset.variables["lon"][:]
        dst_lat[:] = src_dataset.variables["lat"][:]
        
        # copy variable attr
        copy_vattributefunc(src_dataset.variables["lon"], dst_lon)
        copy_vattributefunc(src_dataset.variables["lat"], dst_lat)
        copy_vattributefunc(src_dataset.variables["precipitation"], dst_precipitation)
        copy_vattributefunc(src_dataset.variables["precipitation_cnt"], dst_precipitation_cnt)
        copy_vattributefunc(src_dataset.variables["IRprecipitation"], dst_IRprecipitation)
        copy_vattributefunc(src_dataset.variables["IRprecipitation_cnt"], dst_IRprecipitation_cnt)
        copy_vattributefunc(src_dataset.variables["HQprecipitation"], dst_HQprecipitation)
        copy_vattributefunc(src_dataset.variables["HQprecipitation_cnt"], dst_HQprecipitation_cnt)
        copy_vattributefunc(src_dataset.variables["randomError"], dst_randomError)
        copy_vattributefunc(src_dataset.variables["randomError_cnt"], dst_randomError_cnt)
        
        # add time in variable attr
        add_time_attr = "time " + src_dataset.variables["precipitation"].getncattr("coordinates")
        dst_precipitation.setncattr("coordinates", add_time_attr)
        dst_precipitation_cnt.setncattr("coordinates", add_time_attr)
        dst_IRprecipitation.setncattr("coordinates", add_time_attr)
        dst_IRprecipitation_cnt.setncattr("coordinates", add_time_attr)
        dst_HQprecipitation.setncattr("coordinates", add_time_attr)
        dst_HQprecipitation_cnt.setncattr("coordinates", add_time_attr)
        dst_randomError.setncattr("coordinates", add_time_attr)
        dst_randomError_cnt.setncattr("coordinates", add_time_attr)
        
        # copy dataset attribute
        dst_dataset.FileHeader = src_dataset.FileHeader
        dst_dataset.InputPointer = src_dataset.InputPointer
        dst_dataset.title = src_dataset.title
        dst_dataset.ProductionTime = "2023-09-28"
        dst_dataset.history = src_dataset.history
        dst_dataset.description = f"combine daily TRMM P into one file, by XudongZheng, 28/09/2023T{time.ctime(time.time())}"
    
    # loop for file to read data
    for i in tqdm(range(len(date_period_str)), desc="loop for file to combine", colour="green"):
        src_path = os.path.join(src_home, f"3B42_Daily.{date_period_str[i]}.7.nc4{suffix}")
        with Dataset(src_path, "r") as src_dataset:
            dst_precipitation[i, :, :] = src_dataset.variables["precipitation"][:]
            dst_precipitation_cnt[i, :, :] = src_dataset.variables["precipitation_cnt"][:]
            dst_IRprecipitation[i, :, :] = src_dataset.variables["IRprecipitation"][:]
            dst_IRprecipitation_cnt[i, :, :] = src_dataset.variables["IRprecipitation_cnt"][:]
            dst_HQprecipitation[i, :, :] = src_dataset.variables["HQprecipitation"][:]
            dst_HQprecipitation_cnt[i, :, :] = src_dataset.variables["HQprecipitation_cnt"][:]
            dst_randomError[i, :, :] = src_dataset.variables["randomError"][:]
            dst_randomError_cnt[i, :, :] = src_dataset.variables["randomError_cnt"][:]

    dst_dataset.close()

def demo_combineGlobalSnow_SWE_add_time_dim():
    # general
    src_home = "E:/data/hydrometeorology/globalsnow/archive_v3.0/L3A_daily_SWE/NetCDF4"
    dst_home = "E:/data/hydrometeorology/globalsnow/archive_v3.0/L3A_daily_SWE"
    suffix = ".nc"
    
    src_names = [n for n in os.listdir(src_home) if n.endswith(suffix)]
    src_paths = [os.path.join(src_home, n) for n in src_names]
    
    # set time
    str_date = [n[:n.find("_")] for n in src_names]
    from datetime import datetime
    date_period_datetime = [datetime.strptime(d, "%Y%m%d") for d in str_date]
    
    # create dataset
    dst_path = os.path.join(dst_home, f"GlobalSnow_swe_northern_hemisphere_swe_0.25grid_combined{suffix}")
    dst_dataset = Dataset(dst_path, "w")
    
    # create time dimension
    dst_dataset.createDimension("time", len(src_names))
    dst_times = dst_dataset.createVariable("time", "f8", ("time", ))
    dst_times.units = "days since 1979-01-06"
    dst_times.calendar = "gregorian"
    from netCDF4 import date2num
    dst_times[:] = date2num(date_period_datetime, units=dst_times.units, calendar=dst_times.calendar)
    
    # read one file to copy
    with Dataset(src_paths[0], "r") as src_dataset:
        src_x = src_dataset.variables["x"][:]
        src_y = src_dataset.variables["y"][:]
        
        # create lon/lat dimensionpip
        dst_dataset.createDimension("x", len(src_x))
        dst_dataset.createDimension("y", len(src_y))
        
        # create variables
        dst_x = dst_dataset.createVariable("x", "f8", ("x", ))
        dst_y = dst_dataset.createVariable("y", "f8", ("y", ))
        dst_swe = dst_dataset.createVariable("swe", "i4", ("time", "y", "x", ),
                                             fill_value=src_dataset.variables["swe"].getncattr("_FillValue"))
        dst_crs = dst_dataset.createVariable("crs", "S1", )
        
        # set/copy lon/lat variables
        dst_x[:] = src_dataset.variables["x"][:]
        dst_y[:] = src_dataset.variables["y"][:]
        
        # copy variable attr
        copy_vattributefunc(src_dataset.variables["x"], dst_x)
        copy_vattributefunc(src_dataset.variables["y"], dst_y)
        copy_vattributefunc(src_dataset.variables["swe"], dst_swe)
        copy_vattributefunc(src_dataset.variables["crs"], dst_crs)
        
        # copy dataset attribute
        copy_garrtibutefunc(src_dataset, dst_dataset)
        dst_dataset.description = f"combine GlobalSnow SWE into one file, by XudongZheng, 08/10/2023T{time.ctime(time.time())}"
    
    # loop for file to read data
    for i in tqdm(range(len(src_paths)), desc="loop for file to combine", colour="green"):
        src_path = src_paths[i]
        with Dataset(src_path, "r") as src_dataset:
            dst_swe[i, :, :] = src_dataset.variables["swe"][:]

    dst_dataset.close()
    
    
if __name__ == "__main__":
    home = "F:/Yanxiang/Python/yanxiang_episode4"
    nc_path = os.path.join(home, "GLDAS_NOAH025_M.A194801.020.nc4")
    out_path = os.path.join("GLDAS_NOAH025_M.A194801.020_Rainf.nc4")
    shp_path = os.path.join(home, "GIS/fr.shp")
    f = Dataset(nc_path, mode="r")

    cn = create_nc()
    cn(nc_path=out_path,
       dimensions={"lon": f.dimensions["lon"].size,
                   "lat": f.dimensions["lat"].size,
                   "time": f.dimensions["time"].size,
                   "bnds": f.dimensions["bnds"].size
                   },
        variables={"time_bnds": {"datatype": "f4", "dimensions": ("time", "bnds")},
                   "lon": {"datatype": "f4", "dimensions": ("lon",)},
                   "lat": {"datatype": "f4", "dimensions": ("lat",)},
                   "time": {"datatype": "f4", "dimensions": ("time",)},
                   "Rainf_f_tavg": {"datatype": "f8", "dimensions": ("time", "lat", "lon")}
                   },
        var_value={"time_bnds": f.variables["time_bnds"][:],
                   "lon": f.variables["lon"][:],
                   "lat": f.variables["lat"][:],
                   "time": f.variables["time"][:],
                   "Rainf_f_tavg": f.variables["Rainf_f_tavg"][:],
                   }
        )

    f.close()

    # set attribute
    cn.copy_garrtibutefunc(out_path, nc_path)
    cn.copy_vattributefunc(out_path, nc_path)