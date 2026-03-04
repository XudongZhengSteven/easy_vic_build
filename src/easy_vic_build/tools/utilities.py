# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""General utility functions for path handling, resource loading, dataset I/O, and so on"""

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

from .dpc_func.basin_grid_class import Basins
from .geo_func.search_grids import *
from .params_func.GlobalParamParser import GlobalParamParser
from .params_func.params_set import *

## ------------------------ general utilities ------------------------

top_package = __package__.split(".")[0]


def check_and_mkdir(dir):
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    dir : str
        The directory path to check and create.
    """
    if not os.path.exists(dir):
        os.mkdir(os.path.abspath(dir))


def remove_and_mkdir(dir):
    """
    Remove a directory and recreate it as an empty directory.

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
    Remove files in a directory while keeping subdirectories.

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
    Read HCDN basin polygons and add ``AREA_km2``.

    Parameters
    ----------
    home : str, optional
        The root directory where the CAMELS dataset is stored, by default "E:\\data\\hydrometeorology\\CAMELS".

    Returns
    -------
    HCDN_shp : geopandas.GeoDataFrame
        Basin shapefile with area converted from ``m2`` to ``km2``.
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
    HCDN_shp_path = os.path.join(
        home, "basin_set_full_res", "HCDN_nhru_final_671.shp"
    )
    basin_shp_all = Basins.from_shapefile(HCDN_shp_path)
    basin_shp_all["AREA_km2"] = basin_shp_all.AREA / 1000000  # m2 -> km2
    basin_shp = basin_shp_all.loc[basin_index:basin_index, :]
    return basin_shp_all, basin_shp


def readdpc(
    dpc_fpath,
    dpc_class,
):
    """
    Instantiate a DPC object from a serialized path.

    Parameters
    ----------
    dpc_fpath : str
        Path to DPC serialized data.
    dpc_class : type
        DPC class constructor that accepts ``dpc_fpath``.

    Returns
    -------
    object
        DPC instance created by ``dpc_class``.
    """
    # read
    dpc_instance = dpc_class(dpc_fpath)

    return dpc_instance


def readDomain(evb_dir, name=None):
    """
    Reads the domain data from a NetCDF file.

    Parameters
    ----------
    evb_dir : Evb_dir
        Case directory manager.
    name : str, optional
        Optional suffix; if provided, read ``domain_<name>.nc``.

    Returns
    -------
    domain_dataset : netCDF4.Dataset
        The domain dataset loaded from the NetCDF file.
    """
    # read
    if name is not None:
        domain_path = evb_dir.domainFile_path.replace(".nc", f"_{name}.nc")
    else:
        domain_path = evb_dir.domainFile_path
        
    domain_dataset = Dataset(domain_path, "r", format="NETCDF4")

    return domain_dataset


def readParam(evb_dir, mode="r"):
    """
    Reads parameter data from NetCDF files at levels 0 and 1.

    Parameters
    ----------
    evb_dir : Evb_dir
        Case directory manager.
        
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
    evb_dir : Evb_dir
        Case directory manager.
    """
    if os.path.isfile(evb_dir.params_dataset_level0_path):
        os.remove(evb_dir.params_dataset_level0_path)

    if os.path.isfile(evb_dir.params_dataset_level1_path):
        os.remove(evb_dir.params_dataset_level1_path)
        
    if os.path.isfile(os.path.join(evb_dir.RVICParam_dir, "flow_direction_file.nc")):
        os.remove(os.path.join(evb_dir.RVICParam_dir, "flow_direction_file.nc"))


def readRVICParam(evb_dir):
    """
    Reads the RVIC parameter files from disk.

    Parameters
    ----------
    evb_dir : Evb_dir
        Case directory manager. Also expects ``evb_dir.cfg_file_path``.

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
    evb_dir : Evb_dir
        Case directory manager.

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
    evb_dir : Evb_dir
        Case directory manager.

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
    evb_dir : Evb_dir
        Case directory manager.

    Returns
    -------
    stream_gdf: `geopandas.GeoDataFrame`
        The stream basin shapefile data.
    """
    stream_gdf = gpd.read_file(
        os.path.join(evb_dir.BasinMap_dir, "stream_raster_shp_clip.shp")
    )
    return stream_gdf
