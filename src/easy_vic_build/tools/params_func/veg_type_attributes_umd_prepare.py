# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: veg_type_attributes_umd_prepare

This module defines the functionality to update the vegetation parameter JSON file by integrating 
vegetation roughness and displacement data from the `NLDAS_Veg_monthly.xlsx` file. It reads and processes 
vegetation attributes from the UMD dataset, updates these attributes with monthly data, and then saves 
the updated parameters into a new JSON file.

Functions:
----------
    - prepare_veg_param_json: Reads the original vegetation parameter JSON, updates it with 
      data from the NLDAS_Veg_monthly file, and saves the updated data to a new file.

Dependencies:
-------------
    - json: Used for reading and writing JSON data.
    - easy_vic_build: Provides access to the `Evb_dir` module.
    - utilities: Provides helper functions like `read_veg_type_attributes_umd` and `read_NLDAS_Veg_monthly`.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com
"""

import os
import json
from ..utilities import read_veg_type_attributes_umd, read_NLDAS_Veg_monthly

def prepare_veg_param_json(veg_param_json_path, veg_param_json_updated_path, NLDAS_Veg_monthly_path):
    """
    Update the vegetation parameters in a JSON file using data from the NLDAS_Veg_monthly Excel file.

    Parameters
    ----------
    veg_param_json_path : str
        The file path to the original vegetation parameters JSON file.
    veg_param_json_updated_path : str
        The file path where the updated vegetation parameters JSON will be saved.
    NLDAS_Veg_monthly_path : str
        The file path to the NLDAS_Veg_monthly Excel file containing vegetation roughness and displacement data.

    Returns
    -------
    None
        This function updates the original JSON file with vegetation roughness and displacement data
        from the NLDAS_Veg_monthly file and saves it to the specified updated file path.
    """
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
    # veg_param_json_path = os.path.join(Evb_dir.__data_dir__, "veg_type_attributes_umd.json")
    # veg_param_json_updated_path = os.path.join(Evb_dir.__data_dir__, "veg_type_attributes_umd_updated.json")
    # NLDAS_Veg_monthly_path = os.path.join(Evb_dir.__data_dir__, "NLDAS_Veg_monthly.xlsx")
    # prepare_veg_param_json(veg_param_json_path, veg_param_json_updated_path, NLDAS_Veg_monthly_path)
    pass