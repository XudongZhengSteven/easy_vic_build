# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import sys
sys.path.append("../easy_vic_build")
from easy_vic_build import Evb_dir
from easy_vic_build.build_GlobalParam import buildGlobalParam
from easy_vic_build.calibrate import *
from easy_vic_build.tools.utilities import *


if __name__ == "__main__":
    # make sure you have already prepare the basic params/information for running vic
    basin_index = 397
    date_period = ["19980101", "20011231"]
    warmup_date_period = ["19980101", "19981231"]
    calibrate_date_period = ["19990101", "20011231"]
    case_name = "397_3km"
    grid_res_level1 = 0.025
    
    # read evb_dir
    evb_dir = Evb_dir(cases_home="/home/xdz/code/VIC_xdz/cases")
    evb_dir.builddir(case_name)
    evb_dir.vic_exe_path = "/home/xdz/code/VIC_xdz/vic_image.exe"
    
    # read dpc_VIC
    dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2 = readdpc(evb_dir)
    
    # read domain
    domain_dataset = readDomain(evb_dir)
    
    # read param
    params_dataset_level0, params_dataset_level1 = readParam(evb_dir)
    
    # set GlobalParam_dict
    GlobalParam_dict = {"Simulation":{"MODEL_STEPS_PER_DAY": "24",
                                    "SNOW_STEPS_PER_DAY": "24",
                                    "RUNOFF_STEPS_PER_DAY": "24",
                                    "STARTYEAR": str(date_period[0][:4]),
                                    "STARTMONTH": str(int(date_period[0][4:6])),
                                    "STARTDAY": str(int(date_period[0][6:])),
                                    "ENDYEAR": str(date_period[1][:4]),
                                    "ENDMONTH": str(int(date_period[1][4:6])),
                                    "ENDDAY": str(int(date_period[1][6:])),
                                    "OUT_TIME_UNITS": "DAYS"},
                        "OUTVAR1": {"OUTVAR": ["OUT_RUNOFF", "OUT_BASEFLOW", "OUT_PET", "OUT_DISCHARGE"]}
                        }
    
    # buildGlobalParam
    buildGlobalParam(evb_dir, GlobalParam_dict)
    
    # calibrate
    algParams = {"popSize": 40, "maxGen": 250, "cxProb": 0.7, "mutateProb": 0.2}
    nsgaII_VIC_SO = NSGAII_VIC_SO(dpc_VIC_level1, evb_dir, algParams, evb_dir.calibrate_cp_path)
    nsgaII_VIC_SO.run()
    
    
    
    