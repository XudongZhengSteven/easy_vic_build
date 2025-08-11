# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from easy_vic_build.build_RVIC_Param import *
from easy_vic_build.Evb_dir_class import Evb_dir
from easy_vic_build.tools.utilities import readdpc, readParam, readDomain
from easy_vic_build.build_hydroanalysis import buildHydroanalysis_level0, buildHydroanalysis_level1
from easy_vic_build.tools.dpc_func.dpc_subclass import dataProcess_VIC_level3
from easy_vic_build.tools.hydroanalysis_func.create_dem import create_dem_from_params
from easy_vic_build.tools.hydroanalysis_func.hydroanalysis_wbw.set_workenv import setWorkenv
from general_info import *


"""
general information:

basin set
106(10_100_km_humid); 240(10_100_km_semi_humid); 648(10_100_km_semi_arid); 
213(100_1000_km_humid); 38(100_1000_km_semi_humid); 670(10_100_km_semi_arid);
397(1000_larger_km_humid); 636(1000_larger_km_semi_humid); 580(1000_larger_km_semi_arid) 

grid_res_level0=1km(0.00833)
grid_res_level1=3km(0.025), 6km(0.055), 8km(0.072), 12km(0.11)

"""

def test():
    # general set
    case_name = f"{basin_index}_{model_scale}"
    
    # build dir
    evb_dir = Evb_dir(cases_home="./examples")
    evb_dir.builddir(case_name)
    
    # read params
    params_dataset_level0, params_dataset_level1 = readParam(evb_dir)
    
    # read domain
    domain_dataset = readDomain(evb_dir)
    
    # read dpc
    dpc_VIC_level3 = readdpc(evb_dir.dpc_VIC_level3_path, dataProcess_VIC_level3)
    basin_attribute = dpc_VIC_level3.get_data_from_cache("basin_attribute")[0]
    
    # build hydroanalysis level1
    buildHydroanalysis_level1(
        evb_dir,
        params_dataset_level1,
        domain_dataset,
        reverse_lat,
        stream_acc_threshold=None,
        crs_str="EPSG:4326",
        d8_streamnetwork_kwargs={
          "snap_dist": 0.001,
        },
        snap_outlet_to_stream_kwargs={
            "snap_dist": 30.0,
        },
        outlets_with_reference_coords=[[basin_attribute["camels_topo:gauge_lon"].values[0]], [basin_attribute["camels_topo:gauge_lat"].values[0]]]
    )
    
    # close
    domain_dataset.close()
    params_dataset_level0.close()
    params_dataset_level1.close()
    
    
def test_buildHydroanalysis_level0():
    # general set
    case_name = f"{basin_index}_hydroanalysis"
    
    # build dir
    evb_dir = Evb_dir(cases_home="./examples")
    evb_dir.builddir(case_name)

    #* cp dpc_level3, params_dataset_level0
    
    # read dpc
    dpc_VIC_level3 = readdpc(evb_dir.dpc_VIC_level3_path, dataProcess_VIC_level3)
    basin_attribute = dpc_VIC_level3.get_data_from_cache("basin_attribute")[0]
    basin_shp = dpc_VIC_level3.get_data_from_cache("basin_shp")[0]
    
    # read params
    params_dataset_level0, params_dataset_level1 = readParam(evb_dir)

    # create_dem
    dem_level0_path = os.path.join(evb_dir.Hydroanalysis_dir, "dem_level0.tif")
    transform = create_dem_from_params(
        params_dataset_level0,
        dem_level0_path,
        crs_str="EPSG:4326",
        reverse_lat=reverse_lat,
    )
    
    # hydroanalysis
    buildHydroanalysis_level0(
        evb_dir,
        dem_level0_path,
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
        outlets_with_reference_coords=[[basin_attribute["camels_topo:gauge_lon"].values[0]], [basin_attribute["camels_topo:gauge_lat"].values[0]]]
    )

    # set wbe
    wbw_working_directory = os.path.join(evb_dir.Hydroanalysis_dir, "wbw_working_directory_level0")
    wbe = setWorkenv(wbw_working_directory)
    
    # save basin shp
    basin_shp.to_file(os.path.join(wbw_working_directory, "basin_shp.shp"))
    
    # clip stream raster
    stream_raster = wbe.read_raster("stream_raster.tif")
    basin_shp_vector = wbe.read_vector("basin_shp.shp")
    stream_raster_clip = wbe.clip_raster_to_polygon(stream_raster, basin_shp_vector)
    
    wbe.write_raster(stream_raster_clip, "stream_raster_clip.tif")
    
    stream_raster_clip_vector = wbe.raster_to_vector_polygons(stream_raster_clip)
    wbe.write_vector(stream_raster_clip_vector, "stream_raster_clip_vector.shp")
    
if __name__ == "__main__":
    # test_buildHydroanalysis_level0()
    test()
    