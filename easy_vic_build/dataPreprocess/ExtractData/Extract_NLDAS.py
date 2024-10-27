# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import os
from tqdm import *
import numpy as np
from netCDF4 import Dataset


def clipNLDAS(dir_name, xmin, xmax, ymin, ymax):
    # nco
    from nco import Nco
    from nco.custom import Limit, LimitSingle
    
    # general
    home = "E:/data/hydrometeorology/NLDAS/NLDAS2_Primary_Forcing_Data_subset_0.125"
    suffix = ".nc4"
    
    src_fnames = np.array([n for n in os.listdir(os.path.join(home, "data")) if n.endswith(suffix)], dtype=str)
    src_paths = np.array([os.path.join(home, "data", n) for n in src_fnames], dtype=str)

    # make dir
    dir_path = os.path.join(home, dir_name)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    
    # loop for clip files
    nco = Nco()
    opt = [Limit("lon", xmin, xmax), Limit("lat", ymin, ymax)]
    for i in tqdm(range(len(src_fnames))):
        src_fname = src_fnames[i]
        src_path = src_paths[i]
        dst_fname = src_fname
        dst_path = os.path.join(dir_path, dst_fname)
        
        nco.ncks(input=src_path, output=dst_path, options=opt)


# def clip_basin(basin_index):
#     # get NLDAS_lon/lat
#     NLDAS_lon = np.loadtxt("F:/research/ScaleEffect/code/VICmodel/NLDAS_0125_lon.txt")
#     NLDAS_lat = np.loadtxt("F:/research/ScaleEffect/code/VICmodel/NLDAS_0125_lat.txt")

#     # 1998-2010
#     date_period = ["19980101", "20101231"]
#     root, home = setHomePath(root="E:")
#     subdir = "WaterBalanceAnalysis"
    
#     # dpc_VIC
#     dpc_VIC = dataProcess_CAMELS_VIC(home, subdir, date_period)
#     dpc_VIC(basin_index, res=0.125, stand_lon=NLDAS_lon, stand_lat=NLDAS_lat)
    
#     # plot
#     # dpc_VIC.plot()
    
#     # set clip params
#     dir_name = f"data_{dpc_VIC.basin_shp.index[0]}_{dpc_VIC.basin_shp.loc[dpc_VIC.basin_shp.index[0], "hru_id"]}"
#     xmin = dpc_VIC.grid_shp["point_geometry"].x.min()
#     xmax = dpc_VIC.grid_shp["point_geometry"].x.max()
#     ymin = dpc_VIC.grid_shp["point_geometry"].y.min()
#     ymax = dpc_VIC.grid_shp["point_geometry"].y.max()
    
#     # clipNLDAS
#     clipNLDAS(dir_name, xmin, xmax, ymin, ymax)
    

def get_grids_coord_from_NLDAS(read_from_file=True):
    if not read_from_file:
        with Dataset("F:/research/ScaleEffect/code/VICmodel/NLDAS_FORA0125_H.A19980101.0000.002.grb.SUB.nc4", "r") as dataset:
            NLDAS_lon = np.array(dataset.variables["lon"][:])
            NLDAS_lat = np.array(dataset.variables["lat"][:])

        np.savetxt("F:/research/ScaleEffect/code/VICmodel/NLDAS_0125_lon.txt", NLDAS_lon)
        np.savetxt("F:/research/ScaleEffect/code/VICmodel/NLDAS_0125_lat.txt", NLDAS_lat)
    
    # load
    else:
        NLDAS_lon = np.loadtxt("F:/research/ScaleEffect/code/VICmodel/NLDAS_0125_lon.txt")
        NLDAS_lat = np.loadtxt("F:/research/ScaleEffect/code/VICmodel/NLDAS_0125_lat.txt")
    
    return NLDAS_lon, NLDAS_lat


if __name__ == "__main__":
    # clip_basin(359)
    pass