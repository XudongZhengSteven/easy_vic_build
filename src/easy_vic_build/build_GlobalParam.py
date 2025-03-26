# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
build_GlobalParam - A Python module for building global parameter file.

This module provides functions for constructing and modifying the global parameter file
used in the VIC (Variable Infiltration Capacity) model. The main function `buildGlobalParam`
sets up the global parameter configuration by reading an existing reference file and updating
it with values from a provided dictionary.

Functions:
----------
    - `buildGlobalParam`: Constructs the global parameter file for VIC simulations by reading
      a reference file and applying configurations from the provided dictionary.
      It sets default values and overrides them based on the dictionary content.

Usage:
------
    1. Call the `buildGlobalParam` to build global parameter file with specificed parameters.

Example:
--------
    basin_index = 213
    model_scale = "6km"
    case_name = f"{basin_index}_{model_scale}"
    date_period = ["19980101", "19981231"]

    evb_dir = Evb_dir("./examples")
    evb_dir.builddir(case_name)

    GlobalParam_dict = {"Simulation":{"MODEL_STEPS_PER_DAY": "1",
                                      "SNOW_STEPS_PER_DAY": "24",
                                      "RUNOFF_STEPS_PER_DAY": "24",
                                      "STARTYEAR": str(date_period[0][:4]),
                                      "STARTMONTH": str(int(date_period[0][4:6])),
                                      "STARTDAY": str(int(date_period[0][4:6])),
                                      "ENDYEAR": str(date_period[1][:4]),
                                      "ENDMONTH": str(int(date_period[1][4:6])),
                                      "ENDDAY": str(int(date_period[1][4:6])),
                                      "OUT_TIME_UNITS": "DAYS"},
                        "Output": {"AGGFREQ": "NDAYS   1"},
                        "OUTVAR1": {"OUTVAR": ["OUT_RUNOFF", "OUT_BASEFLOW", "OUT_DISCHARGE"]}
                        }

    buildGlobalParam(evb_dir, GlobalParam_dict)

Dependencies:
-------------
    - `os`: For file and directory operations.
    - `re`: For regular expressions to match section names.
    - `.tools.utilities`: For utilities like `read_globalParam_reference` to load reference files.

"""

import os
import re

from . import logger
from .tools.utilities import read_globalParam_reference


def buildGlobalParam(evb_dir, GlobalParam_dict):
    """
    Build the global parameter configuration file for VIC simulations.

    This function reads a reference global parameter file, sets default values for various
    sections (such as forcing, domain, parameters, output, and routing), and then updates
    the sections with values from the provided `GlobalParam_dict`. The updated configuration
    is saved to the specified file path.

    Parameters:
    -----------
    evb_dir : `Evb_dir`
        An instance of the `Evb_dir` class, containing paths for VIC deployment.
        
    GlobalParam_dict : dict
        A dictionary containing parameters to override the default values in the global configuration.

    Returns:
    --------
    None
    """
    logger.info("Starting to generate global parameter file... ...")
    ## ====================== set dir and path ======================
    # get rout_param
    try:
        rout_param_path = os.path.join(
            evb_dir.rout_param_dir, os.listdir(evb_dir.rout_param_dir)[0]
        )
    except Exception as e:
        rout_param_path = ""
        logger.warning(f"Routing parameter path could not be determined: {e}")

    ## ====================== build GlobalParam ======================
    # read GlobalParam_reference parser
    globalParam = read_globalParam_reference()
    # globalParam = GlobalParamParser()
    # globalParam.load(evb_dir.globalParam_reference_path)

    # set default param (dir and path)
    globalParam.set(
        "Forcing",
        "FORCING1",
        os.path.join(evb_dir.MeteForcing_dir, f"{evb_dir.forcing_prefix}."),
    )
    globalParam.set("Domain", "DOMAIN", evb_dir.domainFile_path)
    globalParam.set("Param", "PARAMETERS", evb_dir.params_dataset_level1_path)
    globalParam.set("Output", "LOG_DIR", evb_dir.VICLog_dir + "/")
    globalParam.set("Output", "RESULT_DIR", evb_dir.VICResults_dir)
    globalParam.set("Routing", "ROUT_PARAM", rout_param_path)

    # set based on GlobalParam_dict (override the default param)
    for section_name in GlobalParam_dict.keys():
        if re.match(r"^(FORCE_TYPE|DOMAIN_TYPE|OUTVAR\d*)$", section_name):
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

    logger.info(
        f"Building global parameter file successfully, saved to {evb_dir.globalParam_path}"
    )
