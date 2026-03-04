# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Create VIC meteorological forcing NetCDF datasets."""

from netCDF4 import Dataset
import cftime
import numpy as np


def createMeteForcingDataset(dst_path, lat_list, lon_list, time_datetime, start_time="19970101", format="NETCDF4"):
    """Create a meteorological forcing dataset with VIC-required variables.

    Parameters
    ----------
    dst_path : str
        Output NetCDF file path.
    lat_list : sequence of float
        Latitude coordinate values.
    lon_list : sequence of float
        Longitude coordinate values.
    time_datetime : sequence of datetime-like
        Datetime sequence used to build the ``time`` dimension.
    start_time : str, optional
        Time origin in ``YYYYMMDD`` format for the ``time`` units attribute.
    format : str, optional
        NetCDF file format passed to ``netCDF4.Dataset``.

    Returns
    -------
    tuple
        ``(meteforcing_dataset, time_v, lats, lons, lat_v, lon_v)``.
    """
    # create dataset
    meteforcing_dataset = Dataset(dst_path, "w", format=format)

    # ===================== define dimensions =====================
    time_dim = meteforcing_dataset.createDimension("time", len(time_datetime))
    lat_dim = meteforcing_dataset.createDimension("lat", len(lat_list))
    lon_dim = meteforcing_dataset.createDimension("lon", len(lon_list))
    
    # ===================== define dimension variables ======================
    time_v = meteforcing_dataset.createVariable("time", int, ("time",))
    lat_v = meteforcing_dataset.createVariable("lat", "f8", ("lat",))  # 1D array
    lon_v = meteforcing_dataset.createVariable("lon", "f8", ("lon",))  # 1D array
    lats = meteforcing_dataset.createVariable("lats", "f8", ("lat", "lon",),)  # 2D array
    lons = meteforcing_dataset.createVariable("lons", "f8", ("lat", "lon",),)  # 2D array
    
    # ====================== assign attribute for dimension variables ======================
    time_v.calendar = "proleptic_gregorian"
    time_v.units = f"hours since {start_time[:4]}-{start_time[4:6]}-{start_time[6:8]} 00:00:00"

    lat_v.units = "degrees_north"
    lat_v.long_name = "latitude of grid cell center"
    lat_v.standard_name = "latitude"
    lat_v.axis = "Y"

    lon_v.units = "degrees_east"
    lon_v.long_name = "longitude of grid cell center"
    lon_v.standard_name = "longitude"
    lon_v.axis = "X"

    lats.long_name = "lats 2D"
    lats.description = "Latitude of grid cell 2D"
    lats.units = "degrees"

    lons.long_name = "lons 2D"
    lons.description = "longitude of grid cell 2D"
    lons.units = "degrees"
    
    # ====================== assign values for dimension variables ======================
    meteforcing_dataset.variables["time"][:] = cftime.date2num(time_datetime, units=time_v.units, calendar=time_v.calendar)
    meteforcing_dataset.variables["lat"][:] = np.array(lat_list)  # 1D array
    meteforcing_dataset.variables["lon"][:] = np.array(lon_list)  # 1D array
    grid_array_lons, grid_array_lats = np.meshgrid(
        meteforcing_dataset.variables["lon"][:], meteforcing_dataset.variables["lat"][:]
    )  # 2D array
    meteforcing_dataset.variables["lons"][:, :] = grid_array_lons  # 2D array
    meteforcing_dataset.variables["lats"][:, :] = grid_array_lats  # 2D array
    
    # ====================== define variables ======================
    tas = meteforcing_dataset.createVariable("tas", "f4", ("time", "lat", "lon",),)
    prcp = meteforcing_dataset.createVariable("prcp", "f4", ("time", "lat", "lon",),)
    pres = meteforcing_dataset.createVariable("pres", "f4", ("time", "lat", "lon",),)
    dswrf = meteforcing_dataset.createVariable("dswrf", "f4", ("time", "lat", "lon",),)
    dlwrf = meteforcing_dataset.createVariable("dlwrf", "f4", ("time", "lat", "lon",),)
    vp = meteforcing_dataset.createVariable("vp", "f4", ("time", "lat", "lon",),)
    wind = meteforcing_dataset.createVariable("wind", "f4", ("time", "lat", "lon",),)
    
    # ====================== assign attribute for variables ======================
    tas.long_name = "AIR_TEMP"
    tas.description = "Average air temperature"
    tas.units = "C"

    prcp.long_name = "PREC"
    prcp.description = "Total precipitation (rain and snow)"
    prcp.units = "mm"

    pres.long_name = "PRESSURE"
    pres.description = "Atmospheric pressure"
    pres.units = "kPa"

    dswrf.long_name = "SWDOWN"
    dswrf.description = "Incoming shortwave radiation"
    dswrf.units = "W/m2"

    dlwrf.long_name = "LWDOWN"
    dlwrf.description = "Incoming longwave radiation"
    dlwrf.units = "W/m2"

    vp.long_name = "VP"
    vp.description = "Vapor pressure"
    vp.units = "kPa"

    wind.long_name = "WIND"
    wind.description = "Wind speed"
    wind.units = "m/s"
    
    # ====================== assign attribute for dataset ======================
    meteforcing_dataset.title = "VIC5 image meteForcing dataset"
    meteforcing_dataset.note = ("meteForcing dataset generated by XudongZheng, zhengxd@sehemodel.club")
    meteforcing_dataset.Conventions = "CF-1.6"
    
    return meteforcing_dataset, time_v, lats, lons, lat_v, lon_v
    
    
    
    
