# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
from JRB_build_evb_dir import build_modeling_dir
from easy_vic_build.tools.hydroanalysis_func.mosaic_dem import merge_dems
from easy_vic_build.build_hydroanalysis import buildHydroanalysis_level0, buildHydroanalysis_level1
from easy_vic_build.tools.utilities import *
from JRB_build_dpc import dataProcess_VIC_level0_JRB, dataProcess_VIC_level1_JRB, dataProcess_VIC_level3_JRB
from general_info import *
from easy_vic_build.tools.hydroanalysis_func.hydroanalysis_wbw.set_workenv import setWorkenv
import matplotlib.pyplot as plt
pd.set_option('display.width',None)


def mosaic_dem_JRB():
    input_dir = "F:\\research\\Research\\JRB_scaling\\data\\DEM\\ASTGTM2_originalData"
    output_file = "F:\\research\\Research\\JRB_scaling\\data\\DEM\\ASTGTM2_mosaic.tif"
    
    merge_dems(input_dir, suffix=".tif", 
               output_file=output_file, 
               srcSRS="EPSG:4326", dstSRS="EPSG:4326")

def read_stations_coords_JRB():
    stations_coords_home = "F:\\research\\Research\\JRB_scaling\\data"
    hydroStation_coord_path = os.path.join(stations_coords_home, "hydroStation_coord.txt")
    hydroStation_coord_EchoHAT_path = os.path.join(stations_coords_home, "hydroStation_coord_EchoHAT.txt")
    meteStation_coord_path = os.path.join(stations_coords_home, "meteStation_coord.txt")

    hydroStation_coord = pd.read_csv(hydroStation_coord_path, sep="\t")
    hydroStation_coord_EchoHAT = pd.read_csv(hydroStation_coord_EchoHAT_path, sep="\t")
    meteStation_coord = pd.read_csv(meteStation_coord_path, sep="\t")
    
    # combine hydroStation_coord and hydroStation_coord_EchoHAT
    hydroStation_coord_combined = pd.concat([hydroStation_coord, hydroStation_coord_EchoHAT], axis=0)
    
    
    return hydroStation_coord, hydroStation_coord_EchoHAT, hydroStation_coord_combined, meteStation_coord


def hydroanalysis_level0_JRB(evb_dir_hydroanalysis_level0):
    # read stations coord
    hydroStation_coord, hydroStation_coord_EchoHAT, hydroStation_coord_combined, meteStation_coord = read_stations_coords_JRB()
    
    # hydroanalysis
    buildHydroanalysis_level0(
        evb_dir_hydroanalysis_level0,
        dem_level0_path="F:\\research\\Research\\JRB_scaling\\data\\DEM\\ASTGTM2_mosaic.tif",
        flow_direction_pkg="wbw",
        stream_acc_threshold=None, # cal using calculate_streamnetwork_threshold
        calculate_streamnetwork_threshold_kwargs={
            "method": "drainage_area",
            "drainage_area_km2": 0.01,
        },
        d8_streamnetwork_kwargs={
          "snap_dist": 0.001,
        },
        snap_outlet_to_stream_kwargs={
            "snap_dist": 30.0,
        },
        crs_str="EPSG:4326",
        esri_pointer=True,
        outlets_with_reference_coords=[hydroStation_coord_combined.lon.to_list(), hydroStation_coord_combined.lat.to_list()]
    )

def clip_streamflow_raster_level0(evb_dir_hydroanalysis_level0, station_name, model_scale):
    # build evb
    evb_dir_modeling = build_modeling_dir(subname=f"{station_name}_{model_scale}")
    
    # dpc_VIC_level3
    dpc_VIC_level3 = readdpc(evb_dir_modeling.dpc_VIC_level3_path, dataProcess_VIC_level3_JRB)
    
    # set wbe
    wbw_working_directory = os.path.join(evb_dir_hydroanalysis_level0.Hydroanalysis_dir, "wbw_working_directory_level0")
    wbe = setWorkenv(wbw_working_directory)
    
    # save basin shp
    basin_shp = dpc_VIC_level3.get_data_from_cache("basin_shp")[0]
    basin_shp.to_file(os.path.join(wbw_working_directory, "basin_shp.shp"))
    
    # clip stream raster
    stream_raster = wbe.read_raster("stream_raster.tif")
    basin_shp_vector = wbe.read_vector("basin_shp.shp")
    stream_raster_clip = wbe.clip_raster_to_polygon(stream_raster, basin_shp_vector)
    
    wbe.write_raster(stream_raster_clip, "stream_raster_clip.tif")
    
    stream_raster_clip_vector = wbe.raster_to_vector_polygons(stream_raster_clip)
    wbe.write_vector(stream_raster_clip_vector, "stream_raster_clip_vector.shp")
    
    stream_raster_clip_vector_lines = wbe.polygons_to_lines(stream_raster_clip_vector)
    wbe.write_vector(stream_raster_clip_vector_lines, "stream_raster_clip_vector_lines.shp")
    
    
def hydroanalysis_level1_JRB(station_name, model_scale, reverse_lat):
    # read stations coord
    hydroStation_coord, hydroStation_coord_EchoHAT, hydroStation_coord_combined, meteStation_coord = read_stations_coords_JRB()
    
    # build evb
    evb_dir_modeling = build_modeling_dir(subname=f"{station_name}_{model_scale}")
    
    # read params
    params_dataset_level0, params_dataset_level1 = readParam(evb_dir_modeling)
    
    # read domain
    domain_dataset = readDomain(evb_dir_modeling)
    
    # build hydroanalysis level1
    buildHydroanalysis_level1(
        evb_dir_modeling,
        params_dataset_level1,
        domain_dataset,
        reverse_lat,
        stream_acc_threshold=10,
        flow_direction_pkg="wbw",
        crs_str="EPSG:4326",
        d8_streamnetwork_kwargs={
          "snap_dist": 0.001,
        },
        snap_outlet_to_stream_kwargs={
            "snap_dist": 30.0,
        },
        outlets_with_reference_coords=[hydroStation_coord_combined.lon.to_list(), hydroStation_coord_combined.lat.to_list()]
    )
    
    # close
    domain_dataset.close()
    params_dataset_level0.close()
    params_dataset_level1.close()
    
    
if __name__ == "__main__":
    # create dem
    # mosaic_dem_JRB()
    
    # hydroanalysis for level0
    evb_dir_hydroanalysis_level0 = build_modeling_dir(subname="hydroanalysis")
    hydroanalysis_level0_JRB(evb_dir_hydroanalysis_level0)
    
    # hydroanalysis for level1
    # hydroanalysis_level1_JRB(station_name, model_scale, reverse_lat)
    
    # clip
    clip_streamflow_raster_level0(evb_dir_hydroanalysis_level0, station_name, model_scale)
    