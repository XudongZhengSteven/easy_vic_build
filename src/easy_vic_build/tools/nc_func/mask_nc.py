# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com

"""
Module: mask_nc

This module provides functionality for applying spatial masking to variables in NetCDF files. 
It includes a class `mask_nc` that allows for masking based on geographic regions or shapefiles, 
and then saves the masked data into a new NetCDF file. This module is particularly useful for 
spatially filtered data processing in atmospheric and hydrological models that rely on NetCDF format.

Class:
------
    - mask_nc: A class that handles the application of spatial masks to NetCDF variables, 
      using either a shapefile or defined geographic regions (latitudes and longitudes).

Class Methods:
--------------
    - __init__: Initializes the mask_nc class.
    - __call__: Applies the mask to the specified variables from the source NetCDF file and writes 
      the results to the destination NetCDF file.
    - get_masked_val: Retrieves the masked values by applying the mask based on a shapefile or geographic region.

Dependencies:
-------------
    - netCDF4: For reading and writing NetCDF files.
    - regionmask: Optional, used for masking based on shapefiles or geographic regions.
    - geopandas: Used for reading shapefiles that define the mask.
    - numpy: Provides array manipulation and mathematical operations for applying the mask.
    - warnings: Used to suppress warnings related to missing packages.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com
"""


from netCDF4 import Dataset
try:
    import regionmask
    HAS_regionmask = True
except:
    HAS_regionmask = False

import geopandas as gp
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")


class mask_nc:
    """ 
    A class for applying spatial masks to variables in NetCDF files. 
    This class allows the user to apply a mask based on either a shapefile or 
    geographic regions (latitude and longitude), and writes the masked data 
    into a new NetCDF file.
    """

    def __init__(self):
        """
        Initializes the mask_nc class. 
        This class does not require any parameters for initialization.
        """
        pass

    def __call__(self, src_path, dst_path, mask_valname, mask_shp=None, mask_region=None, lon_valname="lon",
                 lat_valname="lat"):
        """
        Apply the mask to variables in a NetCDF file and write the results to a new NetCDF file.

        Parameters
        ----------
        src_path : str
            The file path of the source NetCDF file to read data from.
        dst_path : str
            The file path of the destination NetCDF file to write the masked data to.
        mask_valname : list of str
            A list of variable names in the source NetCDF file that will be masked.
        mask_shp : str or None, optional
            The path to a shapefile to use for masking (default is None).
        mask_region : list of float or None, optional
            A list of four floats representing the geographic region to mask in the form
            [lat_min, lat_max, lon_min, lon_max] (default is None).
        lon_valname : str, optional
            The name of the longitude variable in the NetCDF file (default is "lon").
        lat_valname : str, optional
            The name of the latitude variable in the NetCDF file (default is "lat").

        Raises
        ------
        ValueError
            If neither `mask_shp` nor `mask_region` is provided for masking.
        ImportError
            If the `regionmask` package is not available.
        """

        with Dataset(src_path, mode="r") as src_dataset:
            with Dataset(dst_path, mode="w") as dst_dataset:
                # lon/lat
                lon = src_dataset.variables[lon_valname][:]
                lat = src_dataset.variables[lat_valname][:]

                # create dimensions
                for key in src_dataset.dimensions:
                    dst_dataset.createDimension(key, size=src_dataset.dimensions[key].size)

                # create variables and copy variables attributes
                for key in src_dataset.variables:
                    # get variables attributes from src_dataset
                    ncattr_dict = dict(((key_nacttr, src_dataset.variables[key].getncattr(key_nacttr)) for key_nacttr
                                        in src_dataset.variables[key].ncattrs()))

                    # fill_value
                    fill_value = ncattr_dict["_FillValue"] if "_FillValue" in ncattr_dict.keys() else None

                    # createVariable
                    dst_dataset.createVariable(key,
                                               datatype=src_dataset.variables[key].dtype,
                                               dimensions=src_dataset.variables[key].dimensions,
                                               fill_value=fill_value)

                    # set/copy variables attributes
                    for key_nacttr in ncattr_dict.keys():
                        if key_nacttr != "_FillValue":
                            dst_dataset.variables[key].setncattr(key_nacttr, ncattr_dict[key_nacttr])

                # Assign values to variables: set mask
                for key in src_dataset.variables:
                    if key in mask_valname:
                        masked_val = self.get_masked_val(lon, lat, src_dataset.variables[key], mask_shp, mask_region,
                                                         lon_valname, lat_valname, src_dataset)
                        dst_dataset.variables[key][:] = masked_val
                    else:
                        dst_dataset.variables[key][:] = src_dataset.variables[key][:]

                # global attributes copy
                for key in src_dataset.ncattrs():
                    dst_dataset.setncattr(key, src_dataset.getncattr(key))

                # set auto_mask
                dst_dataset.set_auto_mask(True)

    def get_masked_val(self, lon, lat, src_val, mask_shp, mask_region, lon_valname, lat_valname, src_dataset):
        """
        Retrieve the masked values based on the provided mask or geographic region.

        Parameters
        ----------
        lon : ndarray
            The longitude values in the NetCDF file.
        lat : ndarray
            The latitude values in the NetCDF file.
        src_val : ndarray
            The variable data that will be masked.
        mask_shp : str or None
            The path to the shapefile to use for masking (default is None).
        mask_region : list of float or None
            The geographic region to mask in the form [lat_min, lat_max, lon_min, lon_max] (default is None).
        lon_valname : str
            The name of the longitude variable in the NetCDF file.
        lat_valname : str
            The name of the latitude variable in the NetCDF file.
        src_dataset : Dataset
            The source NetCDF dataset object.

        Returns
        -------
        np.ma.array
            The masked data with a mask applied based on the specified criteria.

        Raises
        ------
        ValueError
            If neither `mask_shp` nor `mask_region` is provided.
        ImportError
            If the `regionmask` package is not available.
        """
        if HAS_regionmask:
            if mask_shp is not None:
                mask = regionmask.mask_geopandas(gp.read_file(mask_shp), lon, lat, wrap_lon=False)
                # wrap_lon=False if lon not in range(0, 180)
                dim = 0
                for key in src_val.dimensions:
                    if key != lon_valname and key != lat_valname:
                        mask = np.repeat(np.expand_dims(mask, axis=dim), len(src_dataset.variables[key]), axis=dim)
                    dim += 1

                masked_val = np.ma.array(src_val[:], mask=mask)

            elif mask_region is not None:
                lat_min_index = np.where(src_dataset.variables[lat_valname][:] <= mask_region[0])[0][-1]
                lat_max_index = np.where(src_dataset.variables[lat_valname][:] >= mask_region[1])[0][0]

                lon_min_index = np.where(src_dataset.variables[lon_valname][:] <= mask_region[2])[0][-1]
                lon_max_index = np.where(src_dataset.variables[lon_valname][:] >= mask_region[3])[0][0]

                mask = np.ones_like(src_val[:])  # all mask
                slice_lat = slice(lat_min_index, lat_max_index + 1, 1)
                slice_lon = slice(lon_min_index, lon_max_index + 1, 1)
                slice_all = []
                for key in src_val.dimensions:
                    if key == lat_valname:
                        slice_all.append(slice_lat)
                    elif key == lon_valname:
                        slice_all.append(slice_lon)
                    else:
                        slice_all.append(slice(0, src_dataset.dimensions[key].size))

                mask[slice_all] = 0

                masked_val = np.ma.array(src_val[:], mask=mask)

            else:
                raise ValueError("input mask area into mask_shp or mask_region")
        else:
            raise ImportError("do not have regionmask package to support mask_nc module")
        
        return masked_val


if __name__ == "__main__":
    home = "F:/Yanxiang/Python/yanxiang_episode4"
    nc_path = os.path.join(home, "GLDAS_NOAH025_M.A194801.020_Rainf.nc4")
    out_path1 = os.path.join("out_maskregion.nc4")
    out_path2 = os.path.join("out_maskshp.nc4")
    shp_path = os.path.join(home, "GIS/fr_proj.shp")
    f = Dataset(nc_path, mode="r")

    mc = mask_nc()
    mc(nc_path, out_path1, mask_valname="Rainf_f_tavg", mask_shp=None, mask_region=[3, 53, 73, 135])
    mc(nc_path, out_path2, mask_valname="Rainf_f_tavg", mask_shp=shp_path, mask_region=None)

    f.close()