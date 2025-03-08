# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import os
from .tools.params_func.GlobalParamParser import GlobalParamParser
from .tools.utilities import read_globalParam_reference
import re

def buildGlobalParam(evb_dir, GlobalParam_dict):
    ## ====================== set dir and path ======================
    # get rout_param
    try:
        rout_param_path = os.path.join(evb_dir.rout_param_dir, os.listdir(evb_dir.rout_param_dir)[0])
    except:
        rout_param_path = ""
    
    ## ====================== build GlobalParam ======================
    # read GlobalParam_reference parser
    globalParam = read_globalParam_reference()
    # globalParam = GlobalParamParser()
    # globalParam.load(evb_dir.globalParam_reference_path)
    
    # set default param (dir and path)
    globalParam.set("Forcing", "FORCING1", os.path.join(evb_dir.MeteForcing_dir, f"{evb_dir.forcing_prefix}."))
    globalParam.set("Domain", "DOMAIN", evb_dir.domainFile_path)
    globalParam.set("Param", "PARAMETERS", evb_dir.params_dataset_level1_path)
    globalParam.set("Output", "LOG_DIR", evb_dir.VICLog_dir + "/")
    globalParam.set("Output", "RESULT_DIR", evb_dir.VICResults_dir)
    globalParam.set("Routing", "ROUT_PARAM", rout_param_path)
    
    # set based on GlobalParam_dict (override the default param)
    for section_name in GlobalParam_dict.keys():
        if re.match(r'^(FORCE_TYPE|DOMAIN_TYPE|OUTVAR\d*)$', section_name):
            # replace section
            section_dict = GlobalParam_dict[section_name]
            globalParam.set_section_values(section_name, section_dict)
            
        else:
            section_dict = GlobalParam_dict[section_name]
            for key, value in section_dict.items():
                globalParam.set(section_name, key, value)

    # save
    with open(evb_dir.globalParam_path, "w") as f:
        globalParam.write(f)
    

