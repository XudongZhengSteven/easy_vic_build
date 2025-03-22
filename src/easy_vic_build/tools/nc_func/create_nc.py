# code: utf-8
# author: "Xudong Zheng"
# email: Z786909151@163.com

"""
Module: create_nc

This module provides functionality for creating and managing NetCDF files. It includes a class `create_nc`
that offers methods to create a NetCDF file with specified dimensions, variables, and values. The module also
provides methods for copying global and variable attributes between NetCDF files, enabling data management
and file manipulation for scientific computing.

Class:
----------
    - create_nc: A class for creating and managing NetCDF files. It supports:
        - Creating a NetCDF file with specified dimensions, variables, and values.
        - Copying global attributes from one NetCDF file to another.
        - Copying variable attributes from one NetCDF file to another.

Functions:
----------
    - copy_vattributefunc: Copies variable attributes from one variable to another.
    - copy_garrtibutefunc: Copies global attributes from one NetCDF file to another.

Dependencies:
-------------
    - netCDF4: Used for reading and writing NetCDF files.
    - numpy: Used for handling arrays and numerical operations.
    - tqdm: Used for progress bars in the terminal.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com
"""


import os

from netCDF4 import Dataset
from tqdm import *

from ... import logger


class create_nc:
    """
    A class for creating and managing NetCDF files.

    This class provides methods to create NetCDF files by specifying dimensions,
    variables, and their values. Additionally, it includes functions for copying
    global and variable attributes from one NetCDF file to another.

    Methods:
    --------
    __call__(self, nc_path, dimensions, variables, var_value, return_dataset=False):
        Creates a NetCDF file at the specified path with the given dimensions and variables.

    copy_garrtibutefunc(self, dst_dataset_path, src_dataset_path):
        Copies global attributes from the source NetCDF file to the destination NetCDF file.

    copy_vattributefunc(self, dst_dataset_path, src_dataset_path):
        Copies variable attributes from the source NetCDF file to the destination NetCDF file.
    """

    def __init__(self):
        """Initializes the create_nc class."""
        pass

    def __call__(self, nc_path, dimensions, variables, var_value, return_dataset=False):
        """
        Creates a NetCDF file with the specified dimensions, variables, and values.

        Parameters:
        -----------
        nc_path : str
            The path where the created NetCDF file will be saved.
        dimensions : dict
            A dictionary where keys are dimension names and values are the sizes of the dimensions.
            The dimensions typically include "lon", "lat", "time", etc. If a dimension is unlimited, set its size to None or 0.
        variables : dict
            A dictionary where keys are variable names, and values are dictionaries of keyword arguments.
            The keyword arguments can include:
            - datatype : str, Data type of the variable (e.g., 'float32')
            - dimensions : tuple, Tuple of dimension names
            - zlib : bool, Whether to use compression (default is False)
            - complevel : int, Compression level (default is 4)
            - shuffle : bool, Whether to use shuffle filter (default is True)
            - fletcher32 : bool, Whether to use Fletcher32 checksum (default is False)
            - contiguous : bool, Whether to store data contiguously (default is False)
            - chunksizes : tuple, Chunk sizes for data storage (default is None)
            - endian : str, Byte order (default is 'native')
            - least_significant_digit : int, Precision of floating point data (default is None)
            - fill_value : int/float, Value used to fill missing data (default is None)
        var_value : dict
            A dictionary where keys are variable names and values are their corresponding values.
            The values should typically be `np.ma.array` or other compatible data types.
        return_dataset : bool, optional
            If True, returns the NetCDF dataset object for further manipulation. If False (default), closes the file after creation.

        Returns:
        --------
        None or Dataset
            Returns the dataset object if `return_dataset=True`, otherwise the function returns None.
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
        """
        Copies global attributes from the source NetCDF file to the destination NetCDF file.

        Parameters:
        -----------
        dst_dataset_path : str
            Path to the destination NetCDF file.
        src_dataset_path : str
            Path to the source NetCDF file.

        Returns:
        --------
        None
        """
        with Dataset(src_dataset_path, "r") as src_dataset:
            with Dataset(dst_dataset_path, "a") as dst_dataset:
                for key in src_dataset.ncattrs():
                    logger.info(f"set global attribute {key}")
                    dst_dataset.setncattr(key, src_dataset.getncattr(key))

    def copy_vattributefunc(self, dst_dataset_path, src_dataset_path):
        """
        Copies variable attributes from the source NetCDF file to the destination NetCDF file.

        Parameters:
        -----------
        dst_dataset_path : str
            Path to the destination NetCDF file.
        src_dataset_path : str
            Path to the source NetCDF file.

        Returns:
        --------
        None
        """
        with Dataset(src_dataset_path, "r") as src_dataset:
            with Dataset(dst_dataset_path, "a") as dst_dataset:
                # loop for each variable in dst_dataset
                for key in dst_dataset.variables:
                    logger.info(f"set variable attribute for {key}")

                    # get variables attributes from src_dataset
                    ncattr_dict = dict(
                        (
                            (
                                key_nacttr,
                                src_dataset.variables[key].getncattr(key_nacttr),
                            )
                            for key_nacttr in src_dataset.variables[key].ncattrs()
                        )
                    )

                    # loop for setting each attributes
                    for key_nacttr in ncattr_dict.keys():
                        if (
                            key_nacttr != "_FillValue"
                        ):  # "_FillValue" should be set when create variable
                            logger.info(f"    set variable attributes {key_nacttr}")
                            dst_dataset.variables[key].setncattr(
                                key_nacttr, ncattr_dict[key_nacttr]
                            )


def copy_vattributefunc(src_var, dst_var):
    """
    Copies variable attributes from the source variable to the destination variable.

    Parameters:
    -----------
    src_var : netCDF4.Variable
        The source variable from which attributes will be copied.
    dst_var : netCDF4.Variable
        The destination variable to which attributes will be copied.

    Returns:
    --------
    None
    """
    ncattr_dict = dict(
        (
            (key_nacttr, src_var.getncattr(key_nacttr))
            for key_nacttr in src_var.ncattrs()
        )
    )

    for key_nacttr in ncattr_dict.keys():
        if (
            key_nacttr != "_FillValue"
        ):  # "_FillValue" should be set when create variable
            dst_var.setncattr(key_nacttr, ncattr_dict[key_nacttr])


def copy_garrtibutefunc(src_dataset, dst_dataset):
    """
    Copies global attributes from the source NetCDF dataset to the destination NetCDF dataset.

    Parameters:
    -----------
    src_dataset : netCDF4.Dataset
        The source dataset from which global attributes will be copied.
    dst_dataset : netCDF4.Dataset
        The destination dataset to which global attributes will be copied.

    Returns:
    --------
    None
    """
    for key in src_dataset.ncattrs():
        dst_dataset.setncattr(key, src_dataset.getncattr(key))


def demos1():
    import os
    import time as t

    import numpy as np
    import pandas as pd
    from netCDF4 import Dataset

    # general set
    home = "H:/research/flash_drough/GLDAS_Noah"
    data_path = os.path.join(
        home, "SoilMoi0_100cm_inst_19480101_20141231_Pentad_muldis_SmPercentile.npy"
    )
    out_path = os.path.join(
        home, "SoilMoi0_100cm_inst_19480101_20141231_Pentad_muldis_SmPercentile.nc4"
    )
    refernce_path = "H:/GIT/Python/yanxiang_1_2/gldas/GLDAS_NOAH025_M.A194801.020.nc4"
    coord_path = "H:/research/flash_drough/coord.txt"
    det = 0.25

    # load data
    data = np.load(data_path)
    reference = Dataset(refernce_path, mode="r")
    coord = pd.read_csv(coord_path, sep=",")
    mask_value = reference.missing_value

    # create full array
    data_lon = coord.lon.values
    data_lat = coord.lat.values
    extent = [min(data_lon), max(data_lon), min(data_lat), max(data_lat)]
    array_data = np.full(
        (
            data.shape[0],
            int((extent[3] - extent[2]) / det + 1),
            int((extent[1] - extent[0]) / det + 1),
        ),
        fill_value=mask_value,
        dtype="float32",
    )

    # array_data_lon/lat is the center point
    array_data_lon = np.linspace(
        extent[0], extent[1], num=int((extent[1] - extent[0]) / det + 1)
    )
    array_data_lat = np.linspace(
        extent[2], extent[3], num=int((extent[3] - extent[2]) / det + 1)
    )

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
    dimensions = {
        "lon": array_data_lon.size,
        "lat": array_data_lat.size,
        "time": data.shape[0],
        "bnds": reference.dimensions["bnds"].size,
    }

    variables = {
        "time_bnds": {"datatype": "f4", "dimensions": ("time", "bnds")},
        "lon": {"datatype": "f4", "dimensions": ("lon",)},
        "lat": {"datatype": "f4", "dimensions": ("lat",)},
        "time": {"datatype": "f4", "dimensions": ("time",)},
        "SM_percentile": {"datatype": "f8", "dimensions": ("time", "lat", "lon")},
    }

    # set var_value
    var_value = {
        "time_bnds": reference.variables["time_bnds"][:],
        "lon": array_data_lon,
        "lat": array_data_lat,
        "time": time,
        "SM_percentile": masked_array_data,
    }

    # cn -> create and set dataset
    dataset = cn(
        nc_path=out_path,
        dimensions=dimensions,
        variables=variables,
        var_value=var_value,
        return_dataset=True,
    )

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
    dataset.variables["SM_percentile"].cal_method = (
        "Calculated by fitting probability distribution of soil moisture"
    )

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
    dst_times = dst_dataset.createVariable("time", "f8", ("time",))
    dst_times.units = "days since 1998-01-01"
    dst_times.calendar = "gregorian"
    from netCDF4 import date2num

    dst_times[:] = date2num(
        date_period_datetime, units=dst_times.units, calendar=dst_times.calendar
    )

    # read one file to copy
    with Dataset(src_paths[0], "r") as src_dataset:
        src_lon = src_dataset.variables["lon"][:]
        src_lat = src_dataset.variables["lat"][:]

        # create lon/lat dimension
        dst_dataset.createDimension("lon", len(src_lon))
        dst_dataset.createDimension("lat", len(src_lat))

        # create variables
        dst_lon = dst_dataset.createVariable("lon", "f4", ("lon",))
        dst_lat = dst_dataset.createVariable("lat", "f4", ("lat",))
        dst_precipitation = dst_dataset.createVariable(
            "precipitation",
            "f4",
            (
                "time",
                "lon",
                "lat",
            ),
            fill_value=src_dataset.variables["precipitation"].getncattr("_FillValue"),
        )
        dst_precipitation_cnt = dst_dataset.createVariable(
            "precipitation_cnt",
            "i2",
            (
                "time",
                "lon",
                "lat",
            ),
        )
        dst_IRprecipitation = dst_dataset.createVariable(
            "IRprecipitation",
            "f4",
            (
                "time",
                "lon",
                "lat",
            ),
            fill_value=src_dataset.variables["IRprecipitation"].getncattr("_FillValue"),
        )
        dst_IRprecipitation_cnt = dst_dataset.createVariable(
            "IRprecipitation_cnt",
            "i2",
            (
                "time",
                "lon",
                "lat",
            ),
        )
        dst_HQprecipitation = dst_dataset.createVariable(
            "HQprecipitation",
            "f4",
            (
                "time",
                "lon",
                "lat",
            ),
            fill_value=src_dataset.variables["HQprecipitation"].getncattr("_FillValue"),
        )
        dst_HQprecipitation_cnt = dst_dataset.createVariable(
            "HQprecipitation_cnt",
            "i2",
            (
                "time",
                "lon",
                "lat",
            ),
        )
        dst_randomError = dst_dataset.createVariable(
            "randomError",
            "f4",
            (
                "time",
                "lon",
                "lat",
            ),
            fill_value=src_dataset.variables["randomError"].getncattr("_FillValue"),
        )
        dst_randomError_cnt = dst_dataset.createVariable(
            "randomError_cnt",
            "i2",
            (
                "time",
                "lon",
                "lat",
            ),
        )

        # set/copy lon/lat variables
        dst_lon[:] = src_dataset.variables["lon"][:]
        dst_lat[:] = src_dataset.variables["lat"][:]

        # copy variable attr
        copy_vattributefunc(src_dataset.variables["lon"], dst_lon)
        copy_vattributefunc(src_dataset.variables["lat"], dst_lat)
        copy_vattributefunc(src_dataset.variables["precipitation"], dst_precipitation)
        copy_vattributefunc(
            src_dataset.variables["precipitation_cnt"], dst_precipitation_cnt
        )
        copy_vattributefunc(
            src_dataset.variables["IRprecipitation"], dst_IRprecipitation
        )
        copy_vattributefunc(
            src_dataset.variables["IRprecipitation_cnt"], dst_IRprecipitation_cnt
        )
        copy_vattributefunc(
            src_dataset.variables["HQprecipitation"], dst_HQprecipitation
        )
        copy_vattributefunc(
            src_dataset.variables["HQprecipitation_cnt"], dst_HQprecipitation_cnt
        )
        copy_vattributefunc(src_dataset.variables["randomError"], dst_randomError)
        copy_vattributefunc(
            src_dataset.variables["randomError_cnt"], dst_randomError_cnt
        )

        # add time in variable attr
        add_time_attr = "time " + src_dataset.variables["precipitation"].getncattr(
            "coordinates"
        )
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
    for i in tqdm(
        range(len(date_period_str)), desc="loop for file to combine", colour="green"
    ):
        src_path = os.path.join(
            src_home, f"3B42_Daily.{date_period_str[i]}.7.nc4{suffix}"
        )
        with Dataset(src_path, "r") as src_dataset:
            dst_precipitation[i, :, :] = src_dataset.variables["precipitation"][:]
            dst_precipitation_cnt[i, :, :] = src_dataset.variables["precipitation_cnt"][
                :
            ]
            dst_IRprecipitation[i, :, :] = src_dataset.variables["IRprecipitation"][:]
            dst_IRprecipitation_cnt[i, :, :] = src_dataset.variables[
                "IRprecipitation_cnt"
            ][:]
            dst_HQprecipitation[i, :, :] = src_dataset.variables["HQprecipitation"][:]
            dst_HQprecipitation_cnt[i, :, :] = src_dataset.variables[
                "HQprecipitation_cnt"
            ][:]
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
    str_date = [n[: n.find("_")] for n in src_names]
    from datetime import datetime

    date_period_datetime = [datetime.strptime(d, "%Y%m%d") for d in str_date]

    # create dataset
    dst_path = os.path.join(
        dst_home, f"GlobalSnow_swe_northern_hemisphere_swe_0.25grid_combined{suffix}"
    )
    dst_dataset = Dataset(dst_path, "w")

    # create time dimension
    dst_dataset.createDimension("time", len(src_names))
    dst_times = dst_dataset.createVariable("time", "f8", ("time",))
    dst_times.units = "days since 1979-01-06"
    dst_times.calendar = "gregorian"
    from netCDF4 import date2num

    dst_times[:] = date2num(
        date_period_datetime, units=dst_times.units, calendar=dst_times.calendar
    )

    # read one file to copy
    with Dataset(src_paths[0], "r") as src_dataset:
        src_x = src_dataset.variables["x"][:]
        src_y = src_dataset.variables["y"][:]

        # create lon/lat dimensionpip
        dst_dataset.createDimension("x", len(src_x))
        dst_dataset.createDimension("y", len(src_y))

        # create variables
        dst_x = dst_dataset.createVariable("x", "f8", ("x",))
        dst_y = dst_dataset.createVariable("y", "f8", ("y",))
        dst_swe = dst_dataset.createVariable(
            "swe",
            "i4",
            (
                "time",
                "y",
                "x",
            ),
            fill_value=src_dataset.variables["swe"].getncattr("_FillValue"),
        )
        dst_crs = dst_dataset.createVariable(
            "crs",
            "S1",
        )

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
    for i in tqdm(
        range(len(src_paths)), desc="loop for file to combine", colour="green"
    ):
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
    cn(
        nc_path=out_path,
        dimensions={
            "lon": f.dimensions["lon"].size,
            "lat": f.dimensions["lat"].size,
            "time": f.dimensions["time"].size,
            "bnds": f.dimensions["bnds"].size,
        },
        variables={
            "time_bnds": {"datatype": "f4", "dimensions": ("time", "bnds")},
            "lon": {"datatype": "f4", "dimensions": ("lon",)},
            "lat": {"datatype": "f4", "dimensions": ("lat",)},
            "time": {"datatype": "f4", "dimensions": ("time",)},
            "Rainf_f_tavg": {"datatype": "f8", "dimensions": ("time", "lat", "lon")},
        },
        var_value={
            "time_bnds": f.variables["time_bnds"][:],
            "lon": f.variables["lon"][:],
            "lat": f.variables["lat"][:],
            "time": f.variables["time"][:],
            "Rainf_f_tavg": f.variables["Rainf_f_tavg"][:],
        },
    )

    f.close()

    # set attribute
    cn.copy_garrtibutefunc(out_path, nc_path)
    cn.copy_vattributefunc(out_path, nc_path)
