# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
from ..utilities import *
from .create_dem import create_dem_from_params
from .hydroanalysis_wbw import hydroanalysis_wbw
import geopandas as gpd

def hydroanalysis_for_basin(evb_dir):
    # read dpc
    dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2 = readdpc(evb_dir)
    
    # read params
    params_dataset_level0, params_dataset_level1 = readParam(evb_dir, mode="r")
    
    # hydroanalysis for level0
    transform = create_dem_from_params(params_dataset_level0, os.path.join(evb_dir.BasinMap_dir, "dem_level0.tif"), crs_str="EPSG:4326", reverse_lat=True)
    hydroanalysis_wbw(evb_dir.BasinMap_dir, os.path.join(evb_dir.BasinMap_dir, "dem_level0.tif"), create_stream=True)
    
    # clip stream gdf within basin shp
    stream_gdf = gpd.read_file(os.path.join(evb_dir.BasinMap_dir, "stream_raster_shp.shp"))
    stream_gdf_clip = gpd.overlay(stream_gdf, dpc_VIC_level0.basin_shp.loc[:, "geometry": "geometry"], how="intersection")
    stream_gdf_clip.to_file(os.path.join(evb_dir.BasinMap_dir, "stream_raster_shp_clip.shp"))
    
    # close params
    params_dataset_level0.close()
    params_dataset_level1.close()