# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from easy_vic_build.tools.hydroanalysis_func.mosaic_dem import merge_dems
from easy_vic_build.tools.utilities import readDomain
from easy_vic_build.build_hydroanalysis import buildHydroanalysis_level0, buildHydroanalysis_level1, buildRivernetwork_level1
from easy_vic_build.tools.geo_func.clip import clip_tiff
from easy_vic_build.tools.utilities import *
from general_info import *
from HRB_build_dpc import dataProcess_VIC_level3_HRB
from copy import deepcopy

def mosaic_dem():
    input_dir = "F:\\research\\Research\\ModelingUncertainty_hanjiang\data\\DEM\\ASTGTM2_originalData"
    output_file = "F:\\research\\Research\\ModelingUncertainty_hanjiang\\data\\DEM\\ASTGTM2_mosaic.tif"
    
    merge_dems(input_dir, suffix=".tif", 
               output_file=output_file, 
               srcSRS="EPSG:4326", dstSRS="EPSG:4326")
    
def clip_dem():
    input_file = "F:\\research\\Research\\ModelingUncertainty_hanjiang\\data\\DEM\\ASTGTM2_mosaic.tif"
    output_file = "F:\\research\\Research\\ModelingUncertainty_hanjiang\\data\\DEM\\ASTGTM2_mosaic_clip.tif"
    clip_tiff(
        input_file,
        output_file,
        boundary,
    )
    
#* you can also get dem from the param file, see buildHydroanalysis_level1

def hydroanalysis_level0_HRB(evb_dir_hydroanalysis_level0):
    # hydroanalysis
    buildHydroanalysis_level0(
        evb_dir_hydroanalysis_level0,
        dem_level0_path="F:\\research\\Research\\ModelingUncertainty_hanjiang\\data\\DEM\\ASTGTM2_mosaic_clip.tif",
        flow_direction_pkg="wbw",
        stream_acc_threshold=100000,  #None, # cal using calculate_streamnetwork_threshold
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
        outlets_with_reference_coords=[station_coords_df.lon.to_list(), station_coords_df.lat.to_list()],
        filldem_kwargs={
            "add_perturbation": False,
            "fill_depressions_bool": True,
            "max_dist": 500,
            "flat_increment": 0.001,
        }
    )
    

def clip_dem_for_basin_shp():
    input_file="F:\\research\\Research\\ModelingUncertainty_hanjiang\\data\\DEM\\ASTGTM2_mosaic_clip.tif"
    output_file = "F:\\research\\Research\\ModelingUncertainty_hanjiang\\data\\DEM\\ASTGTM2_mosaic_clip_basin.tif"
    shp_path = os.path.join(evb_dir_hydroanalysis.Hydroanalysis_dir, f"wbw_working_directory_level0\\basin_vector_outlet_with_reference_{basin_outlets_reference_i_map[station_name]}.shp")
    
    clip_tiff(
        input_file,
        output_file,
        shp_path=shp_path,
    )
    
    
def hydroanalysis_level1_HRB(evb_dir_modeling, reverse_lat):
    # read stations coord
    dpc_VIC_level3 = dataProcess_VIC_level3_HRB(
            load_path=evb_dir_modeling._dpc_VIC_level3_path,
            reset_on_load_failure=False,
    )
    
    gauge_info = dpc_VIC_level3.get_data_from_cache("gauge_info")[0]
    outlets_with_reference_coords = [
        [gauge_info[name]["gauge_coord(lon, lat)_level0"][0] for name in station_names],
        [gauge_info[name]["gauge_coord(lon, lat)_level0"][1] for name in station_names]
    ]
    
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
        stream_acc_threshold=2,
        flow_direction_pkg="wbw",
        crs_str="EPSG:4326",
        d8_streamnetwork_kwargs={
          "snap_dist": 0.001,
        },
        snap_outlet_to_stream_kwargs={
            "snap_dist": 30.0,
        },
        outlets_with_reference_coords=outlets_with_reference_coords,
        filldem_kwargs={
            "add_perturbation": True,
            "fill_depressions_bool": True,
        }
    )
    
    # close
    domain_dataset.close()
    params_dataset_level0.close()
    params_dataset_level1.close()
    
    
def get_outlets_nodes(evb_dir_modeling):
    dpc_VIC_level3 = dataProcess_VIC_level3_HRB(
            load_path=evb_dir_modeling._dpc_VIC_level3_path,
            reset_on_load_failure=False,
    )
    
    gauge_info = dpc_VIC_level3.get_data_from_cache("gauge_info")[0]
    
    domain_dataset = readDomain(evb_dir_modeling)
    lon = domain_dataset.variables["lon"][:]
    lat = domain_dataset.variables["lat"][:]
    
    # find outlet
    outlets = []
    for name in station_names:
        lon_index = np.argmin(np.abs(lon - gauge_info[name]["gauge_coord(lon, lat)_level1"][0]))
        lat_index = np.argmin(np.abs(lat - gauge_info[name]["gauge_coord(lon, lat)_level1"][1]))
        node_name = f"cell_{int(lat_index)}_{int(lon_index)}"
        outlets.append(node_name)
    
    return outlets, domain_dataset
    
def buildRivernetwork_level1_HRB(evb_dir_modeling, threshold=None):
    outlets, domain_dataset = get_outlets_nodes(evb_dir_modeling)
    
    river_network = buildRivernetwork_level1(
        evb_dir_modeling,
        threshold=threshold,
        domain_dataset=domain_dataset,
        plot_bool=True,
        labeled_nodes=outlets,
    )
    
    save_path_river_network = os.path.join(evb_dir_modeling.Hydroanalysis_dir, "river_network_graph.pkl")
    save_path_river_network_full = os.path.join(evb_dir_modeling.Hydroanalysis_dir, "river_network_graph_full.pkl")
    save_path_river_network_connected = os.path.join(evb_dir_modeling.Hydroanalysis_dir, "river_network_graph_connected.pkl")
    
    with open(save_path_river_network, "wb") as f:
        pickle.dump(river_network["river_network_graph"], f)
    
    with open(save_path_river_network_full, "wb") as f:
        pickle.dump(river_network["river_network_graph_full"], f)
    
    with open(save_path_river_network_connected, "wb") as f:
        pickle.dump(river_network["river_network_graph_connected"], f)

    river_network["figs"]["fig_river_network"].savefig(os.path.join(evb_dir_modeling.Hydroanalysis_dir, "fig_river_network.tiff"))
    river_network["figs"]["fig_river_network_full"].savefig(os.path.join(evb_dir_modeling.Hydroanalysis_dir, "fig_river_network_full.tiff"))
    river_network["figs"]["fig_river_network_connected"].savefig(os.path.join(evb_dir_modeling.Hydroanalysis_dir, "fig_river_network_connected.tiff"))

def move_outlets():
    dpc_VIC_level3 = dataProcess_VIC_level3_HRB(
            load_path=evb_dir_modeling._dpc_VIC_level3_path,
            reset_on_load_failure=False,
    )
    
    gauge_info = dpc_VIC_level3.get_data_from_cache("gauge_info")[0]
    
    domain_dataset = readDomain(evb_dir_modeling)
    lon = domain_dataset.variables["lon"][:]
    lat = domain_dataset.variables["lat"][:]
    
    outlets = []
    outlets_coord_index = []
    for name in station_names:
        lon_index = np.argmin(np.abs(lon - gauge_info[name]["gauge_coord(lon, lat)_level1"][0]))
        lat_index = np.argmin(np.abs(lat - gauge_info[name]["gauge_coord(lon, lat)_level1"][1]))
        node_name = f"cell_{int(lat_index)}_{int(lon_index)}"
        outlets.append(node_name)
        outlets_coord_index.append((lon_index, lat_index))
    
    outlets_coord_index_moved = [
        (outlets_coord_index[0][0], outlets_coord_index[0][1]-1),
        (outlets_coord_index[1][0], outlets_coord_index[1][1]),
        (outlets_coord_index[2][0], outlets_coord_index[2][1]),
        (outlets_coord_index[3][0], outlets_coord_index[3][1]),
        (outlets_coord_index[4][0], outlets_coord_index[4][1]),
    ]

    gauge_info_moved = deepcopy(gauge_info)
    for name in station_names:
        lon_index = outlets_coord_index_moved[basin_outlets_reference_i_map[name]][0]
        lat_index = outlets_coord_index_moved[basin_outlets_reference_i_map[name]][1]
        gauge_info_moved[name]["gauge_coord(lon, lat)_level1"] = [lon[lon_index], lat[lat_index]]
        print(f"{name}: {gauge_info[name]['gauge_coord(lon, lat)_level1']} -> {gauge_info_moved[name]['gauge_coord(lon, lat)_level1']}")
    
    # save to dpc_VIC_level3
    dpc_VIC_level3.clear_data_from_cache(
        save_names=["gauge_info"],
        step_name="load_gauge_info"
    )
    
    dpc_VIC_level3.save_data_to_cache(
        save_name="gauge_info",
        data=gauge_info_moved,
        data_level="basin_level",
        step_name="load_gauge_info",
    )
    
    dpc_VIC_level3.save_state(evb_dir_modeling._dpc_VIC_level3_path)
    

if __name__ == "__main__":
    # mosaic_dem()
    # clip_dem()
    hydroanalysis_level0_HRB(evb_dir_hydroanalysis)
    # hydroanalysis_level1_HRB(evb_dir_modeling, reverse_lat)
    # buildRivernetwork_level1_HRB(evb_dir_modeling, threshold=2)
    # move_outlets()
    # clip_dem_for_basin_shp()
    