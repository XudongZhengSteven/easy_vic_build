# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import sys
sys.path.append("../easy_vic_build")
from easy_vic_build import Evb_dir
from easy_vic_build.build_GlobalParam import buildGlobalParam
from easy_vic_build.calibrate import *
from easy_vic_build.tools.utilities import *
from easy_vic_build.build_RVIC_Param import buildFlowDirectionFile, buildPourPointFile, buildUHBOXFile, buildParamCFGFile, modifyRVICParam_for_pourpoint
from easy_vic_build.bulid_Param import buildParam_level0, buildParam_level1
from easy_vic_build.bulid_Domain import *
import warnings
warnings.filterwarnings("ignore")

"""
general information:

basin set
106(10_100_km_humid); 240(10_100_km_semi_humid); 648(10_100_km_semi_arid); 
213(100_1000_km_humid); 38(100_1000_km_semi_humid); 670(10_100_km_semi_arid);
397(1000_larger_km_humid); 636(1000_larger_km_semi_humid); 580(1000_larger_km_semi_arid) 

grid_res_level0=1km(0.00833)
grid_res_level1=3km(0.025), 6km(0.055), 8km(0.072), 12km(0.11)

""" 

if __name__ == "__main__":
    # make sure you have already prepare the basic params/information for running vic
    basin_index = 397
    model_scale = "12km"
    date_period = ["19980101", "20071231"]
    warmup_date_period = ["19980101", "19991231"]
    calibrate_date_period = ["20000101", "20071231"]
    case_name = f"{basin_index}_{model_scale}"
    grid_res_level1 = 0.11
    
    # set evb_dir
    evb_dir = Evb_dir(cases_home="/home/xdz/code/VIC_xdz/cases")
    evb_dir.builddir(case_name)
    evb_dir.vic_exe_path = "/home/xdz/code/VIC_xdz/vic_image.exe"
    
    # read
    dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2 = readdpc(evb_dir)
    
    # modify PourPoint File to match FlowAcc
    # modify: may need to modify the domain.mask based on the acc, and rebuild the rvic parameter
    # modify: may need to modify the pourpoint and flow direction based on the acc
    modify_pourpoint_bool = True
    if modify_pourpoint_bool:
        pourpoint_lon = -91.795
        pourpoint_lat = 38.445
        pourpoint_direction_code = 128
        
        modifyDomain_for_pourpoint(evb_dir, pourpoint_lon, pourpoint_lat)  # mask->1
        buildPourPointFile(evb_dir, None, names=["pourpoint"], lons=[pourpoint_lon], lats=[pourpoint_lat])
        
        # params_dataset_level0, params_dataset_level1 = readParam(evb_dir)
        # domain_dataset = readDomain(evb_dir)
        # modifyRVICParam_for_pourpoint(evb_dir, pourpoint_lon, pourpoint_lat, pourpoint_direction_code, params_dataset_level1, domain_dataset,
        #                               reverse_lat=True, stream_acc_threshold=100.0, flow_direction_pkg="wbw", crs_str="EPSG:4326")
        # params_dataset_level0.close()
        # params_dataset_level1.close()
        # domain_dataset.close()
    
    # set GlobalParam_dict
    GlobalParam_dict = {"Simulation":{"MODEL_STEPS_PER_DAY": "1",
                                    "SNOW_STEPS_PER_DAY": "24",
                                    "RUNOFF_STEPS_PER_DAY": "24",
                                    "STARTYEAR": str(date_period[0][:4]),
                                    "STARTMONTH": str(int(date_period[0][4:6])),
                                    "STARTDAY": str(int(date_period[0][6:])),
                                    "ENDYEAR": str(date_period[1][:4]),
                                    "ENDMONTH": str(int(date_period[1][4:6])),
                                    "ENDDAY": str(int(date_period[1][6:])),
                                    "OUT_TIME_UNITS": "DAYS"},
                        "Output": {"AGGFREQ": "NDAYS   1"},
                        "OUTVAR1": {"OUTVAR": ["OUT_RUNOFF", "OUT_BASEFLOW", "OUT_DISCHARGE"]}
                        }
    
    # perhaps it can be run at hourly scale
    # GlobalParam_dict = {"Simulation":{"MODEL_STEPS_PER_DAY": "24",
    #                                 "SNOW_STEPS_PER_DAY": "24",
    #                                 "RUNOFF_STEPS_PER_DAY": "24",
    #                                 "STARTYEAR": str(date_period[0][:4]),
    #                                 "STARTMONTH": str(int(date_period[0][4:6])),
    #                                 "STARTDAY": str(int(date_period[0][6:])),
    #                                 "ENDYEAR": str(date_period[1][:4]),
    #                                 "ENDMONTH": str(int(date_period[1][4:6])),
    #                                 "ENDDAY": str(int(date_period[1][6:])),
    #                                 "OUT_TIME_UNITS": "HOURS"},
    #                     "Output": {"AGGFREQ": "NHOURS   1"},
    #                     "OUTVAR1": {"OUTVAR": ["OUT_RUNOFF", "OUT_BASEFLOW", "OUT_DISCHARGE"]}
    #                     }
    
    # buildGlobalParam
    buildGlobalParam(evb_dir, GlobalParam_dict)
    
    # nsgaII set
    algParams = {"popSize": 20, "maxGen": 200, "cxProb": 0.7, "mutateProb": 0.2}
    nsgaII_VIC_SO = NSGAII_VIC_SO(evb_dir, dpc_VIC_level0, dpc_VIC_level1, date_period, calibrate_date_period,
                                    algParams=algParams, save_path=evb_dir.calibrate_cp_path, reverse_lat=True, parallel=False)
    
    # calibrate
    calibrate_bool = True
    if calibrate_bool:
        nsgaII_VIC_SO.run()
    
    # get best results
    get_best_results_bool = False
    if get_best_results_bool:
        nsgaII_VIC_SO.get_best_results()
    
    # read cp
    # state = readCalibrateCp(evb_dir)
    # state2 = {'current_generation': 247,
    #     'population': state["history"][247][0],
    #     'initial_population': state["initial_population"],
    #     'history': state["history"][:247]
    #     }
    
    # with open(path, 'wb') as f:
    #     pickle.dump(state2, f)
    