# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from easy_vic_build import Evb_dir
from easy_vic_build.build_GlobalParam import buildGlobalParam

"""
general information:

basin set
106(10_100_km_humid); 240(10_100_km_semi_humid); 648(10_100_km_semi_arid); 
213(100_1000_km_humid); 38(100_1000_km_semi_humid); 670(10_100_km_semi_arid);
397(1000_larger_km_humid); 636(1000_larger_km_semi_humid); 580(1000_larger_km_semi_arid) 

grid_res_level0=1km(0.00833)
grid_res_level1=3km(0.025), 6km(0.055), 8km(0.072), 12km(0.11)

""" 

scalemap = {"3km": 0.025, "6km": 0.055, "8km": 0.072, "12km": 0.11}

if __name__ == "__main__":
    basin_index = 213
    model_scale = "6km"
    case_name = f"{basin_index}_{model_scale}"
    date_period = ["19980101", "19981231"]
    
    # build dir
    evb_dir = Evb_dir("./examples")
    evb_dir.builddir(case_name)

    # set GlobalParam_dict
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
    
    # buildGlobalParam
    buildGlobalParam(evb_dir, GlobalParam_dict)
    
    