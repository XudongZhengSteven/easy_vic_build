# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
import pandas as pd
import json
from easy_vic_build import Evb_dir
from ..utilities import read_veg_type_attributes_umd, read_NLDAS_Veg_monthly

def prepare_veg_param_json(veg_param_json_path, veg_param_json_updated_path, NLDAS_Veg_monthly_path):
    # update the veg_type_attributes_umd.json by the NLDAS_Veg_monthly.xlsx
    veg_params_json = read_veg_type_attributes_umd()
    # read json
    # with open(veg_param_json_path, 'r') as f:
    #     veg_params_json = json.load(f)
    #     veg_params_json = veg_params_json["classAttributes"]
    #     veg_keys = [int(v["class"]) for v in veg_params_json]
    #     veg_params = [v["properties"] for v in veg_params_json]
    #     veg_params_json = dict(zip(veg_keys, veg_params))
    
    # read NLDAS_Veg_monthly
    veg_class_list = list(range(14))
    month_list = list(range(1, 13))
    
    # veg_rough
    NLDAS_Veg_monthly_veg_rough, NLDAS_Veg_monthly_veg_displacement = read_NLDAS_Veg_monthly()
    # NLDAS_Veg_monthly_veg_rough = pd.read_excel(NLDAS_Veg_monthly_path, sheet_name=0, skiprows=2)
    NLDAS_Veg_monthly_veg_rough = NLDAS_Veg_monthly_veg_rough.iloc[:, 1:]
    NLDAS_Veg_monthly_veg_rough.index = veg_class_list
    NLDAS_Veg_monthly_veg_rough.columns = month_list
    
    # displacement
    # NLDAS_Veg_monthly_veg_displacement = pd.read_excel(NLDAS_Veg_monthly_path, sheet_name=1, skiprows=2)
    NLDAS_Veg_monthly_veg_displacement = NLDAS_Veg_monthly_veg_displacement.iloc[:, 1:]
    NLDAS_Veg_monthly_veg_displacement.index = veg_class_list
    NLDAS_Veg_monthly_veg_displacement.columns = month_list
    
    # json
    for i in veg_class_list:
        for j in month_list:
            veg_params_json[i].update({f"veg_rough_month_{j}": NLDAS_Veg_monthly_veg_rough.loc[i, j]})
            veg_params_json[i].update({f"veg_displacement_month_{j}": NLDAS_Veg_monthly_veg_displacement.loc[i, j]})
    
    # save
    with open(veg_param_json_updated_path, 'w') as f:
        json.dump(veg_params_json, f)
    

if __name__ == "__main__":
    veg_param_json_path = os.path.join(Evb_dir.__data_dir__, "veg_type_attributes_umd.json")
    veg_param_json_updated_path = os.path.join(Evb_dir.__data_dir__, "veg_type_attributes_umd_updated.json")
    NLDAS_Veg_monthly_path = os.path.join(Evb_dir.__data_dir__, "NLDAS_Veg_monthly.xlsx")
    prepare_veg_param_json(veg_param_json_path, veg_param_json_updated_path, NLDAS_Veg_monthly_path)
