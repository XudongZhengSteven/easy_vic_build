# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import arcpy
from arcpy import env
from arcpy.sa import *
import sys

def main(workspace_path, stream_acc_threshold,
         dem_file_name, filled_dem_file_path,
         flow_direction_file_path,
         flow_acc_file_path, stream_raster_file_path,
         stream_link_file_path, stream_vector_file_path):
    # Set environment settings
    env.workspace = workspace_path
    
    # fill dem
    filled_dem = Fill(dem_file_name)
    filled_dem.save(filled_dem_file_path)
    filled_dem.save(filled_dem_file_path + ".tif")
    
    # cal flow direction
    flow_direction = FlowDirection(filled_dem, "FORCE")
    flow_direction.save(flow_direction_file_path + ".tif")

    # cal flow acc
    flow_acc = FlowAccumulation(flow_direction, "", "INTEGER")
    flow_acc.save(flow_acc_file_path)
    flow_acc.save(flow_acc_file_path + ".tif")
    
    # stream_acc based on a threshold
    raster_expression = 'Con("flow_acc">%f, True)' % stream_acc_threshold
    stream_acc = arcpy.gp.RasterCalculator_sa(raster_expression, stream_raster_file_path + ".tif")
    # stream_acc.save(stream_acc_file_path)
    # stream_acc.save()
    
    # stream link
    stream_link = StreamLink(stream_acc, flow_direction)
    stream_link.save(stream_link_file_path)
    stream_link.save(stream_link_file_path + ".tif")
    
    # stream to feature
    StreamToFeature(stream_acc, flow_direction, stream_vector_file_path, "NO_SIMPLIFY")


if __name__ == "__main__":
    workspace_path = sys.argv[1]
    stream_acc_threshold = float(sys.argv[2])
    dem_file_name = sys.argv[3]
    filled_dem_file_path = sys.argv[4]
    flow_direction_file_path = sys.argv[5]
    flow_acc_file_path = sys.argv[6]
    stream_raster_file_path = sys.argv[7]
    stream_link_file_path = sys.argv[8]
    stream_vector_file_path = sys.argv[9]
    main(workspace_path, stream_acc_threshold, dem_file_name, filled_dem_file_path,
         flow_direction_file_path, flow_acc_file_path, stream_raster_file_path,
         stream_link_file_path, stream_vector_file_path)
