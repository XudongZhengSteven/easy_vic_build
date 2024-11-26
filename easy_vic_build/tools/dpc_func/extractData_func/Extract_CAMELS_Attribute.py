# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import pandas as pd
import os
from copy import deepcopy

def readBasinAttribute_CAMELS(home="E:\\data\\hydrometeorology\\CAMELS", dir_camels_attribute="camels_attributes_v2.0"):
    camels_clim = pd.read_csv(os.path.join(home, dir_camels_attribute, "camels_clim.txt"), sep=";")
    camels_geol = pd.read_csv(os.path.join(home, dir_camels_attribute, "camels_geol.txt"), sep=";")
    camels_hydro = pd.read_csv(os.path.join(home, dir_camels_attribute, "camels_hydro.txt"), sep=";")
    camels_soil = pd.read_csv(os.path.join(home, dir_camels_attribute, "camels_soil.txt"), sep=";")
    camels_topo = pd.read_csv(os.path.join(home, dir_camels_attribute, "camels_topo.txt"), sep=";")
    camels_vege = pd.read_csv(os.path.join(home, dir_camels_attribute, "camels_vege.txt"), sep=";")

    BasinAttribute = {"camels_clim": camels_clim,
                      "camels_geol": camels_geol,
                      "camels_hydro": camels_hydro,
                      "camels_soil": camels_soil,
                      "camels_topo": camels_topo,
                      "camels_vege": camels_vege}

    return BasinAttribute


def ExtractData(basin_shp, k_list=None):
    """ k_list: camels_clim, camels_geol, camels_hydro, camels_soil, camels_topo, camels_vege or None """
    # read BasinAttribute
    BasinAttributes = readBasinAttribute_CAMELS()

    # chose key to read
    k_list = k_list if k_list is not None else [k for k in BasinAttributes.keys()]

    for k in k_list:
        # BasinAttribute
        id_basinAttribute = "gauge_id"
        
        # get BasinAttribute for k
        BasinAttribute = deepcopy(BasinAttributes[k])
        all_columns = BasinAttribute.columns
        
        # set columns ("camels_clim: ...")
        prefix = k + ":"
        all_columns = [prefix + c for c in all_columns]
        BasinAttribute.columns = all_columns
        id_basinAttribute = prefix + id_basinAttribute
        
        # basin_shp id
        id_basin_shp = "hru_id"

        # merge
        basin_shp = basin_shp.merge(BasinAttribute, left_on=id_basin_shp, right_on=id_basinAttribute, how="left")

    return basin_shp