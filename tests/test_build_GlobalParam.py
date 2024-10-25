# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import sys
sys.path.append("../easy_vic_build")
from easy_vic_build import Evb_dir
from easy_vic_build.build_GlobalParam import buildGlobalParam


if __name__ == "__main__":
    basin_index = 106
    date_period = ["19980101", "20101231"]
    case_name = "106_3km"
    
    # build dir
    evb_dir = Evb_dir()
    evb_dir.builddir(case_name)

    # set GlobalParam_dict
    GlobalParam_dict = {"Simulation":{"MODEL_STEPS_PER_DAY": "24",
                                      "SNOW_STEPS_PER_DAY": "24",
                                      "RUNOFF_STEPS_PER_DAY": "24",
                                      "STARTYEAR": str(date_period[0][:4]),
                                      "STARTMONTH": str(int(date_period[0][4:6])),
                                      "STARTDAY": str(int(date_period[0][4:6])),
                                      "ENDYEAR": str(date_period[1][:4]),
                                      "ENDMONTH": str(int(date_period[1][4:6])),
                                      "ENDDAY": str(int(date_period[1][4:6])),
                                      "OUT_TIME_UNITS": "DAYS"},
                        "OUTVAR1": {"OUTVAR": ["OUT_RUNOFF", "OUT_BASEFLOW", "OUT_PET"]}
                        }
    
    # buildGlobalParam
    buildGlobalParam(evb_dir, GlobalParam_dict)
    
    