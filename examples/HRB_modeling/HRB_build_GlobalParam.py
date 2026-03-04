# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
from general_info import *
from easy_vic_build.build_GlobalParam import buildGlobalParam

def HRB_build_GlobalParam(evb_dir_modeling):
    # set GlobalParam_dict
    GlobalParam_dict = {"Simulation":{"MODEL_STEPS_PER_DAY": "1",
                                      "SNOW_STEPS_PER_DAY": "24",
                                      "RUNOFF_STEPS_PER_DAY": "24",
                                      "STARTYEAR": str(date_period[0][:4]),
                                      "STARTMONTH": str(int(date_period[0][4:6])),
                                      "STARTDAY": str(int(date_period[0][6:8])),
                                      "ENDYEAR": str(date_period[1][:4]),
                                      "ENDMONTH": str(int(date_period[1][4:6])),
                                      "ENDDAY": str(int(date_period[1][6:8])),
                                      "OUT_TIME_UNITS": "DAYS"},
                        "Output": {"AGGFREQ": "NDAYS   1"},
                        "OUTVAR1": {"OUTVAR": ["OUT_RUNOFF", "OUT_BASEFLOW", "OUT_DISCHARGE"]},
                        "Param": {"BASEFLOW": "NIJSSEN2001"}  # "ARNO", "NIJSSEN2001"
                        }

    # buildGlobalParam
    buildGlobalParam(evb_dir_modeling, GlobalParam_dict)

if __name__ == "__main__":
    HRB_build_GlobalParam(evb_dir_modeling)
    