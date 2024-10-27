# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import os
from .tools.params_func.GlobalParamParser import GlobalParamParser
import re

def buildGlobalParam(evb_dir, GlobalParam_dict):
    ## ====================== set dir and path ======================
    # set path
    GlobalParam_dir = evb_dir.GlobalParam_dir
    globalParam_path = os.path.join(GlobalParam_dir, "global_param.txt")
    GlobalParam_reference_path = os.path.join(evb_dir.__data_dir__, "global_param_reference.txt")

    ## ====================== build GlobalParam ======================
    # read GlobalParam_reference parser
    globalParam = GlobalParamParser()
    globalParam.load(GlobalParam_reference_path)
    
    # set default param (dir and path)
    globalParam.set("Forcing", "FORCING1", os.path.join(evb_dir.MeteForcing_dir, "forcings."))
    globalParam.set("Domain", "DOMAIN", os.path.join(evb_dir.DomainFile_dir, "domain.nc"))
    globalParam.set("Param", "PAREMETERS", os.path.join(evb_dir.ParamFile_dir, "params_dataset_level1.nc"))
    globalParam.set("Output", "LOG_DIR", evb_dir.VICLog_dir)
    globalParam.set("Output", "RESULT_DIR", evb_dir.VICResults_dir)
    
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
    with open(globalParam_path, "w") as f:
        globalParam.write(f)
    

