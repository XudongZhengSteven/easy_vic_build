# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import os
from whitebox_workflows import WbEnvironment, show
from ..geo_func.format_conversion import *


def hydroanalysis_wbw(workspace_dir, dem_level1_tif_path, create_stream=False):
    # build flow direction based on wbw
    
    # env set
    wbe = WbEnvironment()
    wbe.working_directory = workspace_dir
    
    # read dem
    dem = wbe.read_raster(dem_level1_tif_path)
    # show(dem, colorbar_kwargs={'label': 'Elevation (m)'})
    
    # fill depressions
    filled_dem = wbe.breach_depressions_least_cost(dem)
    # filled_dem = wbe.fill_depressions(filled_dem)
    wbe.write_raster(filled_dem, 'filled_dem.tif')
    # show(filled_dem, colorbar_kwargs={'label': 'Elevation (m)'})
    # show(filled_dem - dem, colorbar_kwargs={'label': 'fill (m)'})

    # flow direction
    flow_direction = wbe.d8_pointer(filled_dem, esri_pointer=True)
    wbe.write_raster(flow_direction, 'flow_direction.tif')
    # show(flow_direction, colorbar_kwargs={'label': 'flow direction (D8)'})
    
    # flow accumulation
    flow_acc = wbe.d8_flow_accum(flow_direction, out_type="cells", log_transform=False, input_is_pointer=True, esri_pntr=True)
    wbe.write_raster(flow_acc, 'flow_acc.tif')
    # show(flow_acc, colorbar_kwargs={'label': 'flow acc (number)'}, vmin=200)
    
    if create_stream:
        # stream raster
        stream_raster = wbe.extract_streams(flow_acc, threshold=100.0)
        wbe.write_raster(stream_raster, 'stream_raster.tif')
        # show(stream_raster, colorbar_kwargs={'label': 'stream raster (1, bool)'})
        
        # stream raster to shp
        stream_raster_shp_gdf = raster_to_shp(os.path.join(workspace_dir, 'stream_raster.tif'), os.path.join(workspace_dir, 'stream_raster_shp.shp'))
        
        # stream vector
        stream_vector = wbe.raster_streams_to_vector(stream_raster, flow_direction)
        stream_vector, tmp1, tmp2, tmp3 = wbe.vector_stream_network_analysis(stream_vector, filled_dem) # We only want the streams output
        wbe.write_vector(stream_vector, 'stream_vector.shp')
        # show(stream_vector, colorbar_kwargs={'label': 'stream vector(1, bool)'})
    
    return True
