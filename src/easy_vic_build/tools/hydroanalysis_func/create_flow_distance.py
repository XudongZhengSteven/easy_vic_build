# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Create flow-distance rasters from D8 flow-direction grids."""


import numpy as np
import rasterio
from rasterio import CRS


def create_flow_distance(
    flow_distance_path,
    flow_direction_array,
    x_length_array,
    y_length_array,
    transform,
    crs_str="EPSG:4326",
):
    """Calculate flow distance per grid cell and write it as a GeoTIFF file.

    Parameters
    ----------
    flow_distance_path : str
        Output file path for flow-distance raster.
    flow_direction_array : numpy.ndarray
        D8 flow-direction code array.
    x_length_array : numpy.ndarray
        Grid-cell length in the x direction.
    y_length_array : numpy.ndarray
        Grid-cell length in the y direction.
    transform : affine.Affine
        Affine transform used when writing the output raster.
    crs_str : str, optional
        CRS string for the output raster.

    Returns
    -------
    None
        The function writes output to ``flow_distance_path``.
    """
    flow_direction_distance_map = {
        "zonal": [64, 4],
        "meridional": [1, 16],
        "diagonal": [32, 128, 8, 2],
        "edge": [0],
    }
    flow_distance_func_map = {
        "zonal": lambda x_length, y_length: y_length,
        "meridional": lambda x_length, y_length: x_length,
        "diagonal": lambda x_length, y_length: (x_length**2 + y_length**2) ** 0.5,
        "edge": lambda x_length, y_length: (x_length**2 + y_length**2) ** 0.5,
    }

    def flow_distance_funcion(flow_direction, x_length, y_length):
        """Map one flow-direction code to a flow-distance value.

        Parameters
        ----------
        flow_direction : int
            D8 flow-direction code.
        x_length : float
            Grid-cell length in the x direction.
        y_length : float
            Grid-cell length in the y direction.

        Returns
        -------
        float
            Flow distance for one grid cell.
        """
        for k in flow_direction_distance_map:
            if flow_direction in flow_direction_distance_map[k]:
                distance_type = k
                break

        flow_distance_func = flow_distance_func_map[distance_type]
        return flow_distance_func(x_length, y_length)

    flow_distance_funcion_vect = np.vectorize(flow_distance_funcion)
    flow_distance_array = flow_distance_funcion_vect(
        flow_direction_array, x_length_array, y_length_array
    )

    # save as tif file, transform same as dem
    with rasterio.open(
        flow_distance_path,
        "w",
        driver="GTiff",
        height=flow_distance_array.shape[0],
        width=flow_distance_array.shape[1],
        count=1,
        dtype=flow_distance_array.dtype,
        crs=CRS.from_string(crs_str),
        transform=transform,
    ) as dst:
        dst.write(flow_distance_array, 1)
