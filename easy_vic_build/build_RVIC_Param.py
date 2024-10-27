# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import os
import numpy as np
import pandas as pd
import rasterio.transform
import rasterio
from rasterio import CRS
import shutil
from copy import deepcopy
from .tools.params_func.createParametersDataset import createFlowDirectionFile
from .tools.utilities import check_and_mkdir, remove_and_mkdir
import matplotlib.pyplot as plt
from configparser import ConfigParser


def buildRVICParam(dpc_VIC_level1, evb_dir, params_dataset_level1, domain_dataset, reverse_lat=True, stream_acc_threshold=100.0,
                   ppf_kwargs=dict(), uh_params={"tp": 1.4, "mu": 5.0, "m": 3.0}, uh_plot_bool=False,
                   cfg_params={"VELOCITY": 1.5, "DIFFUSION": 800.0, "OUTPUT_INTERVAL": 86400}):
    # set dir
    RVICParam_dir = evb_dir.RVICParam_dir
    param_cfg_file_path = os.path.join(RVICParam_dir, "rvic.parameters.cfg")
    
    # cp domain.nc to RVICParam_dir
    copy_domain(evb_dir)
    
    # buildFlowDirectionFile
    buildFlowDirectionFile(evb_dir, params_dataset_level1, domain_dataset, reverse_lat, stream_acc_threshold)
    
    # buildPourPointFile
    buildPourPointFile(dpc_VIC_level1, evb_dir, **ppf_kwargs)
    
    # buildUHBOXFile
    buildUHBOXFile(evb_dir, **uh_params, plot_bool=uh_plot_bool)
    
    # buildParamCFGFile
    buildParamCFGFile(evb_dir, **cfg_params)
    
    # build rvic parameters
    from rvic.parameters import parameters
    parameters.parameters(param_cfg_file_path, np=1)


def copy_domain(evb_dir):
    # set dir
    domainFile_path = os.path.join(evb_dir.DomainFile_dir, "domain.nc")
    RVICParam_dir = evb_dir.RVICParam_dir
    
    # cp domain.nc to RVICParam_dir
    shutil.copy(domainFile_path, os.path.join(RVICParam_dir, "domain.nc"))


def buildFlowDirectionFile(evb_dir, params_dataset_level1, domain_dataset, reverse_lat=True, stream_acc_threshold=100.0):
    # ====================== set dir and path ======================
    # set dir
    RVICParam_dir = evb_dir.RVICParam_dir
    
    # arcpy related path
    arcpy_python_path = evb_dir.arcpy_python_path
    arcpy_python_script_path = os.path.join(evb_dir.__package_dir__, "arcpy_scripts\\build_flowdirection_arcpy.py")
    
    arcpy_workspace_dir = os.path.join(RVICParam_dir, "arcpy_workspace")
    remove_and_mkdir(arcpy_workspace_dir)
    workspace_dir = arcpy_workspace_dir
    
    # set path
    flow_direction_file_path = os.path.join(RVICParam_dir, "flow_direction_file.nc")
    dem_level1_tif_path = os.path.join(RVICParam_dir, "dem_level1.tif")
    flow_direction_path = os.path.join(RVICParam_dir, "flow_direction.tif")
    flow_acc_path = os.path.join(RVICParam_dir, "flow_acc.tif")
    flow_distance_path = os.path.join(RVICParam_dir, "flow_distance.tif")
    
    # ====================== read ======================
    params_lat = params_dataset_level1.variables["lat"][:]
    params_lon = params_dataset_level1.variables["lon"][:]
    params_elev = params_dataset_level1.variables["elev"][:, :]
    params_mask = params_dataset_level1.variables["run_cell"][:, :]
    x_length_array = domain_dataset.variables["x_length"][:, :]
    y_length_array = domain_dataset.variables["y_length"][:, :]
    
    # ====================== create and save dem_level1.tif ======================
    ulx = min(params_lon)
    uly = max(params_lat)
    xres = round((max(params_lon) - min(params_lon)) / (len(params_lon) - 1), 6)
    yres = round((max(params_lat) - min(params_lat)) / (len(params_lat) - 1), 6)
    if reverse_lat:
        transform = rasterio.transform.from_origin(ulx-xres/2, uly+yres/2, xres, yres)
    else:
        transform = rasterio.transform.from_origin(ulx-xres/2, uly-yres/2, xres, yres)
    
    if not os.path.exists(dem_level1_tif_path):
        with rasterio.open(dem_level1_tif_path, 'w', driver='GTiff',
                           height=params_elev.shape[0],
                           width=params_elev.shape[1],
                           count=1,
                           dtype=params_elev.dtype,
                           crs=CRS.from_string("EPSG:4326"),
                           transform=transform,
                           ) as dst:
            dst.write(params_elev, 1)
    
    # ====================== build flow drection based on arcpy ======================
    out = buildFlowDirection_arcpy(arcpy_workspace_dir, dem_level1_tif_path, arcpy_python_path, arcpy_python_script_path, stream_acc_threshold)
    # TODO implement Whitebox Workflows to build Flowdirection File

    # cp data from workspace to RVICParam_dir
    shutil.copy(os.path.join(workspace_dir, "flow_direction.tif"), flow_direction_path)
    shutil.copy(os.path.join(workspace_dir, "flow_acc.tif"), flow_acc_path)
    
    # read
    with rasterio.open(flow_direction_path, 'r', driver='GTiff') as dataset:
        flow_direction_array = dataset.read(1)
        
    with rasterio.open(flow_acc_path, 'r', driver='GTiff') as dataset:
        flow_acc_array = dataset.read(1)
    
    # ====================== cal flow distance and save it ======================
    flow_direction_distance_map = {"zonal": [64, 4], "meridional": [1, 16], "diagonal": [32, 128, 8, 2]}
    flow_distance_func_map = {"zonal": lambda x_length, y_length: y_length,
                            "meridional": lambda x_length, y_length: x_length,
                            "diagonal": lambda x_length, y_length: (x_length**2 + y_length**2)**0.5}
    
    def flow_distance_funcion(flow_direction, x_length, y_length):
        for k in flow_direction_distance_map:
            if flow_direction in flow_direction_distance_map[k]:
                distance_type = k
                break
        
        flow_distance_func = flow_distance_func_map[distance_type]
        return flow_distance_func(x_length, y_length)

    flow_distance_funcion_vect = np.vectorize(flow_distance_funcion)
    flow_distance_array = flow_distance_funcion_vect(flow_direction_array, x_length_array, y_length_array)
    
    # save as tif file, transform same as dem
    with rasterio.open(flow_distance_path, 'w', driver='GTiff',
                    height=flow_distance_array.shape[0],
                    width=flow_distance_array.shape[1],
                    count=1,
                    dtype=flow_distance_array.dtype,
                    crs=CRS.from_string("EPSG:4326"),
                    transform=transform,
                    ) as dst:
        dst.write(flow_distance_array, 1)
    
    # read
    with rasterio.open(flow_distance_path, 'r', driver='GTiff') as dataset:
        flow_distance_array = dataset.read(1)
    
    # ====================== combine them into a nc file ======================
    # create nc file
    flow_direction_dataset = createFlowDirectionFile(flow_direction_file_path, params_lat, params_lon)
    
    # change type
    params_mask_array = deepcopy(params_mask)
    params_mask_array = params_mask_array.astype(int)
    flow_direction_array = flow_direction_array.astype(int)
    flow_distance_array = flow_distance_array.astype(float)
    flow_acc_array = flow_acc_array.astype(float)
    
    # mask
    params_mask_array[params_mask==0] = int(-9999)
    flow_direction_array[params_mask==0] = int(-9999)
    flow_distance_array[params_mask==0] = float(-9999.0)
    flow_acc_array[params_mask==0] = float(-9999.0)
    
    # assign values
    flow_direction_dataset.variables["lat"][:] = np.array(params_lat)
    flow_direction_dataset.variables["lon"][:] = np.array(params_lon)
    flow_direction_dataset.variables["Basin_ID"][:, :] = np.array(params_mask_array)
    flow_direction_dataset.variables["Flow_Direction"][:, :] = np.array(flow_direction_array)
    flow_direction_dataset.variables["Flow_Distance"][:, :] = np.array(flow_distance_array)
    flow_direction_dataset.variables["Source_Area"][:, :] = np.array(flow_acc_array)
    
    flow_direction_dataset.close()


def buildFlowDirection_arcpy(workspace_path, dem_tiff_path, arcpy_python_path, arcpy_python_script_path, stream_acc_threshold):
    # build flow direction based on arcpy
    stream_acc_threshold = str(stream_acc_threshold) #* set this threshold each time
    filled_dem_file_path = os.path.join(workspace_path, "filled_dem")
    flow_direction_file_path = os.path.join(workspace_path, "flow_direction")
    flow_acc_file_path = os.path.join(workspace_path, "flow_acc")
    stream_acc_file_path = os.path.join(workspace_path, "stream_acc")
    stream_link_file_path = os.path.join(workspace_path, "stream_link")
    stream_feature_file_path = "stream_feature"
    command_arcpy = " ".join([arcpy_python_script_path, workspace_path, stream_acc_threshold, dem_tiff_path, filled_dem_file_path,
                              flow_direction_file_path, flow_acc_file_path, stream_acc_file_path, stream_link_file_path,
                              stream_feature_file_path])
    
    # conduct arcpy file
    out = os.system(f'{arcpy_python_path} {command_arcpy}')
    return out

def buildFlowDirection_wbw(params_lat, params_lon, params_mask, params_elev,
                           x_length_array, y_length_array, dem_tiff_path, domain_area, transform,
                           dst_home, dpc_VIC_level1, reverse_lat=True,
                           stream_acc_threshold=100.0):
    # not yet implemented
    # build flow direction based on wbw
    # read and save weights (area ** 0.5) file
    weights = domain_area ** 0.5
    weights_tiff_path = os.path.join(dst_home, "weights.tif")
    if not os.path.exists(weights_tiff_path):
        with rasterio.open(weights_tiff_path, 'w', driver='GTiff',
                        height=params_elev.shape[0],
                        width=params_elev.shape[1],
                        count=1,
                        dtype=params_elev.dtype,
                        crs=CRS.from_string("EPSG:4326"),
                        transform=transform,
                        ) as dst:
            dst.write(weights, 1)
    
    # env set
    from whitebox_workflows import WbEnvironment, show
    wbe = WbEnvironment()
    workspace_path = os.path.join(dst_home, "buildFlowDirectionFile_wbw")
    if os.path.exists(workspace_path):
        shutil.rmtree(workspace_path)
    os.mkdir(workspace_path)
    
    wbe.working_directory = workspace_path
    
    # read dem
    dem = wbe.read_raster(dem_tiff_path)
    # show(dem, colorbar_kwargs={'label': 'Elevation (m)'})
    
    # read weights
    weights = wbe.read_raster(weights_tiff_path)
    # show(weights, colorbar_kwargs={'label': 'Weights (m)'})
    
    # fill depressions
    filled_dem = wbe.breach_depressions_least_cost(dem)
    filled_dem = wbe.fill_depressions(filled_dem)
    wbe.write_raster(filled_dem, 'filled_dem.tif')
    # show(filled_dem, colorbar_kwargs={'label': 'Elevation (m)'})
    # show(filled_dem - dem, colorbar_kwargs={'label': 'fill (m)'})

    # flow direction #! it different with ArcGIS
    flow_direction = wbe.d8_pointer(filled_dem)
    wbe.write_raster(flow_direction, 'flow_direction.tif')
    # show(flow_direction, colorbar_kwargs={'label': 'flow direction (D8)'})
    
    # flow accumulation
    flow_acc = wbe.fd8_flow_accum(filled_dem, "cells", convergence_threshold= float('inf'), log_transform=False)
    wbe.write_raster(flow_acc, 'flow_acc.tif')
    # show(flow_acc, colorbar_kwargs={'label': 'flow acc (number)'}, vmin=200)
    
    # stream raster
    stream_raster = wbe.extract_streams(flow_acc, threshold=100.0)
    wbe.write_raster(stream_raster, 'stream_raster.tif')
    # show(stream_raster, colorbar_kwargs={'label': 'stream raster (1, bool)'})
    
    # # stream vector
    # stream_vector = wbe.raster_streams_to_vector(stream_raster, flow_direction)
    # stream_vector, tmp1, tmp2, tmp3 = wbe.vector_stream_network_analysis(stream_vector, filled_dem) # We only want the streams output
    # wbe.write_vector(stream_vector, 'stream_vector.shp')
    # show(stream_vector, colorbar_kwargs={'label': 'stream vector(1, bool)'})
    
    # flow distance
    flow_distance = wbe.downslope_flowpath_length(flow_direction, weights=weights)
    # show(flow_distance, colorbar_kwargs={'label': 'flow distance'})
    wbe.write_raster(flow_distance, 'flow_distance.tif')


def buildPourPointFile(dpc_VIC_level1, evb_dir, names=None, lons=None, lats=None):
    #* dpc_VIC_level1.basin_shp should contain "camels_topo" attributes
    # ====================== set dir and path ======================
    RVICParam_dir = evb_dir.RVICParam_dir
    pourpoint_file_path = os.path.join(RVICParam_dir, "pour_points.csv")
    
    # ====================== build PourPointFile ======================
    # df
    pourpoint_file = pd.DataFrame(columns=["lons", "lats", "names"])
    
    if dpc_VIC_level1 is not None:
        x, y = dpc_VIC_level1.basin_shp.loc[:, "camels_topo:gauge_lon"].values[0], dpc_VIC_level1.basin_shp.loc[:, "camels_topo:gauge_lat"].values[0]
        pourpoint_file.lons = [x]
        pourpoint_file.lats = [y]
        pourpoint_file.names = [dpc_VIC_level1.basin_shp.loc[:, "camels_topo:gauge_id"].values[0]]
    else:
        pourpoint_file.lons = lons
        pourpoint_file.lats = lats
        pourpoint_file.names = names
    
    pourpoint_file.to_csv(pourpoint_file_path, header=True, index=False)


def buildUHBOXFile(evb_dir, tp=1.4, mu=5.0, m=3.0, plot_bool=False):
    # ====================== set dir and path ======================
    RVICParam_dir = evb_dir.RVICParam_dir
    uhbox_file_path = os.path.join(RVICParam_dir, "UHBOX.csv")
    
    # ====================== build UHBOXFile ======================
    # tp (hourly, 0~2.5h), mu (default 5.0, based on SCS UH), m (should > 1, default 3.0, based on SCS UH)
    # general UH function
    # Guo, J. (2022), General and Analytic Unit Hydrograph and Its Applications, Journal of Hydrologic Engineering, 27.
    gUH_xt = lambda t, tp, mu: np.exp(mu*(t/tp - 1))
    gUH_gt = lambda t, m, tp, mu: 1 - (1 + m*gUH_xt(t, tp, mu)) ** (-1/m)
    gUH_st = lambda t, m, tp, mu: 1 - gUH_gt(t, m, tp, mu)
    
    gUH_iuh = lambda t, m, tp, mu: mu/tp * gUH_xt(t, tp, mu) * (1 + m*gUH_xt(t, tp, mu)) ** (-(1+1/m))
    gUH_uh = lambda t, m, tp, mu, det_t: (gUH_gt(t, m, tp, mu)[1:] - gUH_gt(t, m, tp, mu)[:-1]) / det_t
    
    # t
    t = np.arange(0, 48)
    t_interval = np.arange(0.5, 47.5)
    
    # UH
    gUH_gt_ret = gUH_gt(t, m, tp, mu)
    gUH_st_ret = gUH_st(t, m, tp, mu)
    gUH_uh_ret = gUH_uh(t, m, tp, mu, det_t=1)
    gUH_iuh_ret = gUH_iuh(t, m, tp, mu)
    
    # plot
    if plot_bool:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].plot(t, gUH_iuh_ret, "-k", label="gUH_iuh", linewidth=1, alpha=1)
        ax[0].plot(t_interval, gUH_uh_ret, "--k", label="gUH_uh", linewidth=3, alpha=0.5)

        ax[0].set_xlabel("time/hours")
        ax[0].set_ylabel("gUH (dimensionless)")
        ax[0].set_ylim(ymin=0)
        ax[0].set_xlim(xmin=0, xmax=47)
        ax[0].legend()
        
        ax[1].plot(t, gUH_gt_ret, "-r", label="gUH_gt", linewidth=1)
        ax[1].plot(t, gUH_st_ret, "-b", label="gUH_st", linewidth=1)
        
        ax[1].set_xlabel("time/hours")
        ax[1].set_ylabel("st, gt")
        ax[1].set_ylim(ymin=0)
        ax[1].set_xlim(xmin=0, xmax=47)
        ax[1].legend()
        
        fig.savefig(os.path.join(RVICParam_dir, "UHBOX.tiff"))
    
    # df
    UHBOX_file = pd.DataFrame(columns=["time", "UHBOX"])
    UHBOX_file.time = t * 3600  # Convert to s
    UHBOX_file.UHBOX = gUH_iuh_ret
    
    UHBOX_file.to_csv(uhbox_file_path, header=True, index=False)


def buildParamCFGFile(evb_dir, VELOCITY=1.5, DIFFUSION=800.0, OUTPUT_INTERVAL=86400):
    # ====================== set dir and path ======================
    case_name = evb_dir._case_name
    RVICParam_dir = evb_dir.RVICParam_dir
    param_cfg_file_path = os.path.join(RVICParam_dir, "rvic.parameters.cfg")
    param_cfg_file_reference_path = os.path.join(evb_dir.__data_dir__, "rvic.parameters.reference.cfg")
    rvic_temp_dir = os.path.join(RVICParam_dir, "temp")
    check_and_mkdir(rvic_temp_dir)
    
    pourpoint_file_path = os.path.join(RVICParam_dir, "pour_points.csv")
    uhbox_file_path = os.path.join(RVICParam_dir, "UHBOX.csv")
    
    flow_direction_file_path = os.path.join(RVICParam_dir, "flow_direction_file.nc")
    domainFile_path = os.path.join(RVICParam_dir, "domain.nc")
    
    # ====================== build CFGFile ======================
    # read reference cfg
    param_cfg_file = ConfigParser()
    param_cfg_file.read(param_cfg_file_reference_path)
    
    # set cfg
    param_cfg_file.set("OPTIONS", 'CASEID', case_name)
    param_cfg_file.set("OPTIONS", 'CASE_DIR', RVICParam_dir)
    param_cfg_file.set("OPTIONS", 'TEMP_DIR', rvic_temp_dir)
    param_cfg_file.set("POUR_POINTS", 'FILE_NAME', pourpoint_file_path)
    param_cfg_file.set("UH_BOX", 'FILE_NAME', uhbox_file_path)
    param_cfg_file.set("ROUTING", 'FILE_NAME', flow_direction_file_path)
    param_cfg_file.set("ROUTING", 'VELOCITY', str(VELOCITY))
    param_cfg_file.set("ROUTING", 'DIFFUSION', str(DIFFUSION))
    param_cfg_file.set("ROUTING", 'OUTPUT_INTERVAL', str(OUTPUT_INTERVAL))
    param_cfg_file.set("DOMAIN", 'FILE_NAME', domainFile_path)
    
    # write cfg
    with open(param_cfg_file_path, 'w') as configfile:
        param_cfg_file.write(configfile)
        
def buildConvCFGFile():
    pass
