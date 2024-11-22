# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import os

def hydroanalysis_arcpy(workspace_path, dem_tiff_path, arcpy_python_path, arcpy_python_script_path, stream_acc_threshold):
    # hydroanalysis based on arcpy
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