# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import rasterio
from rasterio import CRS
import numpy as np


def create_flow_distance(flow_distance_path, flow_direction_array, x_length_array, y_length_array, transform, crs_str="EPSG:4326"):
    flow_direction_distance_map = {"zonal": [64, 4], "meridional": [1, 16], "diagonal": [32, 128, 8, 2], "noflow": [0]}
    flow_distance_func_map = {"zonal": lambda x_length, y_length: y_length,
                            "meridional": lambda x_length, y_length: x_length,
                            "diagonal": lambda x_length, y_length: (x_length**2 + y_length**2)**0.5,
                            "noflow": lambda x_length, y_length: 0}
    
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
                    crs=CRS.from_string(crs_str),
                    transform=transform,
                    ) as dst:
        dst.write(flow_distance_array, 1)
        