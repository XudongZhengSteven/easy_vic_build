# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
utilities - A Python module providing a set of utility functions.

This module provides a set of utility functions for reading and processing various data
related to the VIC model and hydrometeorological datasets. It includes functions for reading
basin and parameter files, processing DPC data, and handling configuration files. These
functions facilitate the extraction, manipulation, and storage of data for use in VIC model
simulations and related analysis.

Functions:
----------
    - `check_and_mkdir`: Checks if a directory exists, and creates it if not.
    - `remove_and_mkdir`: Removes a directory and recreates it.
    - `remove_files`: Removes files from the specified path.
    - `setHomePath`: Sets the home directory path.
    - `exportToCsv`: Exports data to a CSV file.
    - `checkGaugeBasin`: Checks if the gauge basin exists in the dataset.
    - `readHCDNBasins`: Reads and returns the basin shapefile from the HCDN dataset.
    - `read_one_HCDN_basin_shp`: Retrieves data for a specific basin from the HCDN dataset.
    - `readdpc`: Loads and returns the serialized DPC data for the VIC model at three levels.
    - `readDomain`: Reads the domain configuration from a NETCDF file.
    - `readParam`: Reads parameter datasets at two levels (level0 and level1) from NETCDF files.
    - `clearParam`: Deletes parameter datasets at level0 and level1.
    - `readRVICParam`: Reads and returns configuration data related to flow direction, pourpoints,
      unit hydrograph, and other parameters from various files.
    - `read_cfg_to_dict`: Converts the configuration file into a dictionary for easy access.
    - `readGlobalParam`: Loads global parameters using the GlobalParamParser.
    - `readCalibrateCp`: Reads and returns the state of the calibration process from a pickle file.
    - `readBasinMap`: Reads the basin map shapefile containing stream data.
    - `read_NLDAS_annual_prec`: Reads and processes the annual precipitation data from NLDAS.
    - `read_globalParam_reference`: Reads the reference global parameter dataset.
    - `read_rvic_param_cfg_file_reference`: Reads the reference configuration file for RVIC parameters.
    - `read_rvic_conv_cfg_file_reference`: Reads the reference configuration file for RVIC convergence.
    - `read_veg_type_attributes_umd`: Reads vegetation type attributes from UMD data.
    - `read_NLDAS_Veg_monthly`: Reads and processes the monthly vegetation data from NLDAS.
    - `read_veg_param_json`: Reads vegetation parameters from a JSON file.
    - `readHCDNGrids`: Reads the grid data from the HCDN dataset.

Example:
--------
    check_and_mkdir("E:\\data\\new_directory")

    HCDN_shp_all = readHCDNBasins(home="E:\\data\\hydrometeorology\\CAMELS")
    basin_index = 213
    basin_shp_all, basin_shp = read_one_HCDN_basin_shp(basin_index)

    evb_dir = Evb_dir(cases_home="./examples")
    params_dataset_level0, params_dataset_level1 = readParam(evb_dir)

    clearParam(evb_dir)

    state = readCalibrateCp(evb_dir)

Dependencies:
-------------
    - `pickle`: For serializing and deserializing DPC and calibration state data.
    - `geopandas`: For reading and processing shapefiles.
    - `pandas`: For reading CSV files (e.g., pourpoint and UHbox data).
    - `netCDF4`: For reading NETCDF files (domain and parameter datasets).
    - `ConfigParser`: For reading and processing configuration files.
    - `matplotlib`: For plotting, if required in some data processing functions.

"""

import io
import json
import os
import pickle
import pkgutil
import shutil
from configparser import ConfigParser

import geopandas as gpd
import pandas as pd
from netCDF4 import Dataset
from tqdm import *

from .dpc_func.basin_grid_class import HCDNBasins
from .geo_func.search_grids import *
from .params_func.GlobalParamParser import GlobalParamParser
from .params_func.params_set import *

## ------------------------ general utilities ------------------------

top_package = __package__.split(".")[0]


def check_and_mkdir(dir):
    """
    Checks if a directory exists, and if not, creates it.

    Parameters
    ----------
    dir : str
        The directory path to check and create.
    """
    if not os.path.exists(dir):
        os.mkdir(os.path.abspath(dir))


def remove_and_mkdir(dir):
    """
    Removes a directory and creates a new one.

    Parameters
    ----------
    dir : str
        The directory path to remove and create.
    """
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(os.path.abspath(dir))


def remove_files(dir):
    """
    Removes all files in a directory, but not the subdirectories.

    Parameters
    ----------
    dir : str
        The directory from which files will be removed.
    """
    # do not remove subdir
    for f in os.listdir(dir):
        fp = os.path.join(dir, f)
        if os.path.isfile(fp):
            os.remove(fp)


def setHomePath(root="E:"):
    """
    Sets the home path for the CAMELS dataset.

    Parameters
    ----------
    root : str, optional
        The root directory for the CAMELS dataset, default is "E:".

    Returns
    -------
    root: str
        The root directory path.
        
    home: str
        The home directory path.
    """
    home = f"{root}/data/hydrometeorology/CAMELS"
    return root, home


def exportToCsv(basin_shp, fpath_dir):
    """
    Exports data from the provided basin shapefile to CSV files.

    Parameters
    ----------
    basin_shp : `geopandas.GeoDataFrame`
        The basin shapefile containing the required data.
        
    fpath_dir : str
        The directory path to save the exported CSV files.
    """
    columns = basin_shp.columns.to_list()
    streamflow = basin_shp.loc[:, columns.pop(11)]
    prep = basin_shp.loc[:, columns.pop(11)]
    gleam_e_daily = basin_shp.loc[:, columns.pop(-1)]

    columns.remove("geometry")
    # columns.remove("intersects_grids")

    csv_df = basin_shp.loc[:, columns]

    # save
    csv_df.to_csv(os.path.join(fpath_dir, "basin_shp.csv"))

    # save streamflow
    if not os.path.exists(os.path.join(fpath_dir, "streamflow")):
        os.mkdir(os.path.join(fpath_dir, "streamflow"))
    for i in tqdm(
        streamflow.index, desc="loop for basins to save streamflow", colour="green"
    ):
        streamflow[i].to_csv(os.path.join(fpath_dir, "streamflow", f"{i}.csv"))

    # save prep
    if not os.path.exists(os.path.join(fpath_dir, "prep")):
        os.mkdir(os.path.join(fpath_dir, "prep"))
    for i in tqdm(prep.index, desc="loop for basins to save prep", colour="green"):
        prep[i].to_csv(os.path.join(fpath_dir, "prep", f"{i}.csv"))

    # save gleam_e_daily
    if not os.path.exists(os.path.join(fpath_dir, "gleam_e_daily")):
        os.mkdir(os.path.join(fpath_dir, "gleam_e_daily"))
    for i in tqdm(
        gleam_e_daily.index,
        desc="loop for basins to save gleam_e_daily",
        colour="green",
    ):
        gleam_e_daily[i].to_csv(os.path.join(fpath_dir, "gleam_e_daily", f"{i}.csv"))


def checkGaugeBasin(
    basinShp, usgs_streamflow, BasinAttribute, forcingDaymetGaugeAttributes
):
    """
    Compares the basin shapefile data with the USGS streamflow, BasinAttribute,
    and forcingDaymet gauge attributes to check for mismatches.

    Parameters
    ----------
    basinShp : `geopandas.GeoDataFrame`
        The basin shapefile containing basin IDs.
        
    usgs_streamflow : list
        A list of USGS streamflow data.
        
    BasinAttribute : `pandas.DataFrame`
        A dataframe containing basin attributes.
        
    forcingDaymetGaugeAttributes : list
        A list of forcing Daymet gauge attributes.
    """
    id_basin_shp = set(basinShp.hru_id.values)
    id_usgs_streamflow = set(
        [usgs_streamflow[i].iloc[0, 0] for i in range(len(usgs_streamflow))]
    )
    id_BasinAttribute = set(BasinAttribute["camels_clim"].gauge_id.values)
    id_forcingDaymet = set(
        [
            forcingDaymetGaugeAttributes[i]["gauge_id"]
            for i in range(len(forcingDaymetGaugeAttributes))
        ]
    )
    print(
        f"id_HCDN_shp: {len(id_basin_shp)}, id_usgs_streamflow: {len(id_usgs_streamflow)}, id_BasinAttribute: {len(id_BasinAttribute)}, id_forcingDatmet: {len(id_forcingDaymet)}"
    )
    print(f"id_usgs_streamflow - id_HCDN_shp: {id_usgs_streamflow - id_basin_shp}")
    print(f"id_BasinAttribute - id_HCDN_shp: {id_BasinAttribute - id_basin_shp}")
    print(f"id_forcingDatmet - id_HCDN_shp: {id_forcingDaymet - id_basin_shp}")
    # result
    # id_usgs_streamflow - id_HCDN_shp: {9535100, 6775500, 6846500}
    # id_forcingDatmet - id_HCDN_shp: {6846500, 6775500, 3448942, 1150900, 2081113, 9535100}


## ------------------------ read function ------------------------


def read_NLDAS_annual_prec():
    """
    Reads the NLDAS annual precipitation data.

    Returns
    -------
    data_annual_P: `np.ndarray`
        The annual precipitation data.
        
    annual_P_lon: `np.ndarray`
        The longitude data.
        
    annual_P_lat: `np.ndarray`
        The latitude data.
    """
    with io.BytesIO(pkgutil.get_data(top_package, "data/NLDAS_annual_prec.npy")) as f:
        data_annual_P = np.load(f)

    with io.BytesIO(pkgutil.get_data(top_package, "data/annual_prec_lon.txt")) as f:
        annual_P_lon = np.loadtxt(f)

    with io.BytesIO(pkgutil.get_data(top_package, "data/annual_prec_lat.txt")) as f:
        annual_P_lat = np.loadtxt(f)

    return data_annual_P, annual_P_lon, annual_P_lat


def read_globalParam_reference():
    """
    Reads the global parameter reference file.

    Returns
    -------
    globalParam: `GlobalParamParser`
        The global parameter object.
    """
    with io.StringIO(
        pkgutil.get_data(top_package, "data/global_param_reference.txt").decode("utf-8")
    ) as f:
        globalParam = GlobalParamParser()
        globalParam.load(f)
    return globalParam


def read_rvic_param_cfg_file_reference():
    """
    Reads the RVIC parameter configuration reference file.

    Returns
    -------
    cfg_file: `ConfigParser`
        The configuration file parser object.
    """
    with io.StringIO(
        pkgutil.get_data(top_package, "data/rvic.parameters.reference.cfg").decode(
            "utf-8"
        )
    ) as f:
        cfg_file = ConfigParser()
        cfg_file.optionxform = str
        cfg_file.read_file(f)

    return cfg_file


def read_rvic_conv_cfg_file_reference():
    """
    Reads the RVIC convolution configuration reference file.

    Returns
    -------
    cfg_file: `ConfigParser`
        The configuration file parser object.
    """
    with io.StringIO(
        pkgutil.get_data(top_package, "data/rvic.convolution.reference.cfg").decode(
            "utf-8"
        )
    ) as f:
        cfg_file = ConfigParser()
        cfg_file.optionxform = str
        cfg_file.read_file(f)

    return cfg_file


def read_veg_type_attributes_umd():
    """
    Reads vegetation type attributes from UMD.

    Returns
    -------
    veg_params_json: dict
        A dictionary of vegetation parameters.
    """
    with io.BytesIO(
        pkgutil.get_data(top_package, "data/veg_type_attributes_umd.json")
    ) as f:
        veg_params_json = json.load(f)
        veg_params_json = veg_params_json["classAttributes"]
        veg_keys = [int(v["class"]) for v in veg_params_json]
        veg_params = [v["properties"] for v in veg_params_json]
        veg_params_json = dict(zip(veg_keys, veg_params))
    return veg_params_json


def read_NLDAS_Veg_monthly():
    """
    Reads monthly vegetation data from NLDAS.

    Returns
    -------
    NLDAS_Veg_monthly_veg_rough: `pandas.DataFrame`
        A dataframe containing vegetation roughness data.
        
    NLDAS_Veg_monthly_veg_displacement: `pandas.DataFrame`
        A dataframe containing vegetation displacement data.
    """
    with io.BytesIO(pkgutil.get_data(top_package, "data/NLDAS_Veg_monthly.xlsx")) as f:
        NLDAS_Veg_monthly_veg_rough = pd.read_excel(f, sheet_name=0, skiprows=2)
        NLDAS_Veg_monthly_veg_displacement = pd.read_excel(f, sheet_name=1, skiprows=2)

    return NLDAS_Veg_monthly_veg_rough, NLDAS_Veg_monthly_veg_displacement


def read_veg_param_json():
    """
    Reads updated vegetation attributes from a JSON file.

    Returns
    -------
    veg_params_json: dict
        A dictionary of updated vegetation parameters.
    """
    with io.BytesIO(
        pkgutil.get_data(top_package, "data/veg_type_attributes_umd_updated.json")
    ) as f:
        veg_params_json = json.load(f)
    return veg_params_json


def readHCDNGrids(home="E:\\data\\hydrometeorology\\CAMELS"):
    """
    Reads HCDN grid data.

    Parameters
    ----------
    home : str, optional
        The directory containing the grid data, default is "E:\\data\\hydrometeorology\\CAMELS".

    Returns
    -------
    grid_shp: `geopandas.GeoDataFrame`
        The grid shapefile data with added point geometry.
    """
    grid_shp_label_path = os.path.join(home, "map", "grids_0_25_label.shp")
    grid_shp_label = gpd.read_file(grid_shp_label_path)
    print(grid_shp_label)

    grid_shp_path = os.path.join(home, "map", "grids_0_25.shp")
    grid_shp = gpd.read_file(grid_shp_path)
    print(grid_shp)

    # combine grid_shp_lable into grid_shp
    grid_shp["point_geometry"] = grid_shp_label.geometry

    return grid_shp


def readHCDNBasins(home="E:\\data\\hydrometeorology\\CAMELS"):
    """
    Reads the HCDN basins shapefile and adds an area column in km².

    Parameters
    ----------
    home : str, optional
        The root directory where the CAMELS dataset is stored, by default "E:\\data\\hydrometeorology\\CAMELS".

    Returns
    -------
    HCDN_shp: `geopandas.GeoDataFrame`
        The HCDN shapefile with an added column for basin area in km².
    """
    # read data: HCDN
    HCDN_shp_path = os.path.join(home, "basin_set_full_res", "HCDN_nhru_final_671.shp")
    HCDN_shp = gpd.read_file(HCDN_shp_path)
    HCDN_shp["AREA_km2"] = HCDN_shp.AREA / 1000000  # m2 -> km2
    print(HCDN_shp)
    return HCDN_shp


def read_one_HCDN_basin_shp(basin_index, home="E:\\data\\hydrometeorology\\CAMELS"):
    """
    Reads a specific basin shapefile from the HCDN dataset.

    Parameters
    ----------
    basin_index : int
        The index of the basin to read from the HCDN shapefile.
        
    home : str, optional
        The root directory where the CAMELS dataset is stored, by default "E:\\data\\hydrometeorology\\CAMELS".

    Returns
    -------
    basin_shp_all: `geopandas.GeoDataFrame`
        The full HCDN shapefile GeoDataFrame.
        
    basin_shp: `geopandas.GeoDataFrame`
        The specific basin's GeoDataFrame.
    """
    basin_shp_all = HCDNBasins(home)
    basin_shp = basin_shp_all.loc[basin_index:basin_index, :]
    return basin_shp_all, basin_shp


def readdpc(evb_dir):
    """
    Reads the dpc files from disk.

    Parameters
    ----------
    evb_dir : `Evb_dir`
        An instance of the `Evb_dir` class, containing paths for VIC deployment.

    Returns
    -------
    dpc_VIC_level0: `dpc_VIC_level0`
        An instance of the `dpc_VIC_level0` class to process data at level 0 of the VIC model.
    
    dpc_VIC_level1: `dpc_VIC_level1`
        An instance of the `dpc_VIC_level1` class to process data at level 0 of the VIC model.
    
    dpc_VIC_level2: `dpc_VIC_level2`
        An instance of the `dpc_VIC_level2` class to process data at level 0 of the VIC model.
    """
    # read
    with open(evb_dir.dpc_VIC_level0_path, "rb") as f:
        dpc_VIC_level0 = pickle.load(f)

    with open(evb_dir.dpc_VIC_level1_path, "rb") as f:
        dpc_VIC_level1 = pickle.load(f)

    with open(evb_dir.dpc_VIC_level2_path, "rb") as f:
        dpc_VIC_level2 = pickle.load(f)

    return dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2


def readDomain(evb_dir):
    """
    Reads the domain data from a NetCDF file.

    Parameters
    ----------
    evb_dir : `Evb_dir`
        An instance of the `Evb_dir` class, containing paths for VIC deployment.

    Returns
    -------
    domain_dataset: `netCDF.Dataset`
        The domain dataset loaded from the NetCDF file.
    """
    # read
    domain_dataset = Dataset(evb_dir.domainFile_path, "r", format="NETCDF4")

    return domain_dataset


def readParam(evb_dir, mode="r"):
    """
    Reads parameter data from NetCDF files at levels 0 and 1.

    Parameters
    ----------
    evb_dir : `Evb_dir`
        An instance of the `Evb_dir` class, containing paths for VIC deployment.
        
    mode : str, optional
        The mode to open the files, by default "r".

    Returns
    -------
    params_dataset_level0: `netCDF4.Dataset`
        The parameter dataset at level 0.
        
    params_dataset_level1: `netCDF4.Dataset`
        The parameter dataset at level 1.
    """
    # read
    params_dataset_level0 = Dataset(
        evb_dir.params_dataset_level0_path, mode, format="NETCDF4"
    )
    params_dataset_level1 = Dataset(
        evb_dir.params_dataset_level1_path, mode, format="NETCDF4"
    )

    return params_dataset_level0, params_dataset_level1


def clearParam(evb_dir):
    """
    Deletes the parameter NetCDF files from disk.

    Parameters
    ----------
    evb_dir : `Evb_dir`
        An instance of the `Evb_dir` class, containing paths for VIC deployment.
    """
    if os.path.isfile(evb_dir.params_dataset_level0_path):
        os.remove(evb_dir.params_dataset_level0_path)

    if os.path.isfile(evb_dir.params_dataset_level1_path):
        os.remove(evb_dir.params_dataset_level1_path)


def readRVICParam(evb_dir):
    """
    Reads the RVIC parameter files from disk.

    Parameters
    ----------
    evb_dir : `Evb_dir`
        An instance of the `Evb_dir` class, containing paths for VIC deployment.

    Returns
    -------
    flow_direction_dataset: `netCDF4.Dataset`
        The flow direction dataset.
        
    pourpoint_file: `pandas.DataFrame`
        The pourpoint file data.
        
    uhbox_file: `pandas.DataFrame`
        The unit hydrograph box file data.
        
    cfg_file: `ConfigParser`
        The configuration file parser object.
    """
    # read
    flow_direction_dataset = Dataset(evb_dir.flow_direction_file_path, "r", "NETCDF4")
    pourpoint_file = pd.read_csv(evb_dir.pourpoint_file_path)
    uhbox_file = pd.read_csv(evb_dir.uhbox_file_path)
    cfg_file = ConfigParser()
    cfg_file.read(evb_dir.cfg_file_path)

    return flow_direction_dataset, pourpoint_file, uhbox_file, cfg_file


def read_cfg_to_dict(cfg_file_path):
    """
    Reads a configuration file and converts it to a dictionary.

    Parameters
    ----------
    cfg_file_path : str
        The path to the configuration file.

    Returns
    -------
    cfg_file_dict: dict
        A dictionary containing the configuration parameters.
    """
    cfg_file = ConfigParser()
    cfg_file.optionxform = str
    cfg_file.read(cfg_file_path)
    cfg_file_dict = {
        section: dict(cfg_file.items(section)) for section in cfg_file.sections()
    }

    return cfg_file_dict


def readGlobalParam(evb_dir):
    """
    Reads the global parameters from a configuration file.

    Parameters
    ----------
    evb_dir : `Evb_dir`
        An instance of the `Evb_dir` class, containing paths for VIC deployment.

    Returns
    -------
    globalParam: `GlobalParamParser`
        A global parameter object loaded from the specified file.
    """
    globalParam = GlobalParamParser()
    globalParam.load(evb_dir.globalParam_path)

    return globalParam


def readCalibrateCp(evb_dir):
    """
    Reads the calibration control points from a pickle file.

    Parameters
    ----------
    evb_dir : `Evb_dir`
        An instance of the `Evb_dir` class, containing paths for VIC deployment.

    Returns
    -------
    state: dict
        A dictionary containing the state information from the calibration file. The keys of the dictionary are:
            - current_generation
            - initial_population
            - population
            - history
            - front_fitness
    """
    with open(evb_dir.calibrate_cp_path, "rb") as f:
        state = pickle.load(f)
        # current_generation = state["current_generation"]
        # initial_population = state["initial_population"]
        # population = state["population"]
        # history = state["history"]
        # front_fitness = [history[i][1][0][0].fitness.values for i in range(len(history))]
        # plt.plot(front_fitness)
    return state


def readBasinMap(evb_dir):
    """
    Reads the basin map shapefile.

    Parameters
    ----------
    evb_dir : `Evb_dir`
        An instance of the `Evb_dir` class, containing paths for VIC deployment.

    Returns
    -------
    stream_gdf: `geopandas.GeoDataFrame`
        The stream basin shapefile data.
    """
    stream_gdf = gpd.read_file(
        os.path.join(evb_dir.BasinMap_dir, "stream_raster_shp_clip.shp")
    )
    return stream_gdf
