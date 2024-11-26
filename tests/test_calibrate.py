# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import sys
sys.path.append("../easy_vic_build")
from easy_vic_build import Evb_dir
from easy_vic_build.build_GlobalParam import buildGlobalParam
from easy_vic_build.calibrate import *
from easy_vic_build.tools.utilities import *
from easy_vic_build.build_RVIC_Param import buildFlowDirectionFile, buildPourPointFile, buildUHBOXFile, buildParamCFGFile
from easy_vic_build.bulid_Param import buildParam_level0, buildParam_level1
from easy_vic_build.bulid_Domain import buildDomain

"""
general information:

basin set
106(10_100_km_humid); 240(10_100_km_semi_humid); 648(10_100_km_semi_arid); 
213(100_1000_km_humid); 38(100_1000_km_semi_humid); 670(10_100_km_semi_arid);
397(1000_larger_km_humid); 636(1000_larger_km_semi_humid); 580(1000_larger_km_semi_arid) 

grid_res_level0=1km(0.00833)
grid_res_level1=3km(0.025), 6km(0.055), 12km(0.11)

""" 

if __name__ == "__main__":
    # make sure you have already prepare the basic params/information for running vic
    basin_index = 397
    date_period = ["19980101", "20031231"]
    warmup_date_period = ["19980101", "19981231"]
    calibrate_date_period = ["19990101", "20031231"]
    case_name = "397_12km"
    grid_res_level1 = 0.11
    
    # set evb_dir
    evb_dir = Evb_dir(cases_home="/home/xdz/code/VIC_xdz/cases")
    evb_dir.builddir(case_name)
    evb_dir.vic_exe_path = "/home/xdz/code/VIC_xdz/vic_image.exe"
    
    # read and build bool
    read_dpc_bool = True
    read_domain_dataset_bool = True
    read_params_bool = False
    
    build_domain_dataset_bool = False
    build_param_bool = False
    buildRVIC_Param_bool = False
    buildGlobalParam_bool = True
    
    modify_PourPointFile_bool = False
    
    # read
    if read_dpc_bool:
        dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2 = readdpc(evb_dir)
    
    if read_domain_dataset_bool:
        domain_dataset = readDomain(evb_dir)
    
    # build domain
    if build_domain_dataset_bool:
        buildDomain(dpc_VIC_level1, evb_dir, reverse_lat=True)
    
    # build params
    if build_param_bool:
        # build params_level0 with default params
        params_dataset_level0 = buildParam_level0(default_g_list, dpc_VIC_level0, evb_dir, reverse_lat=True)
        
        # build params_level1
        params_dataset_level1 = buildParam_level1(dpc_VIC_level1, evb_dir, reverse_lat=True, domain_dataset=domain_dataset)
        
        # scaling_level0_to_level1
        params_dataset_level1, searched_grids_index = scaling_level0_to_level1(params_dataset_level0, params_dataset_level1)
        
        params_dataset_level0.close()
        params_dataset_level1.close()
        
    # read params_dataset_level1
    if read_params_bool:
        params_dataset_level0, params_dataset_level1 = readParam(evb_dir)
    
    # build RVIC_Param
    if buildRVIC_Param_bool:
        # buildFlowDirectionFile
        buildFlowDirectionFile(evb_dir, params_dataset_level1, domain_dataset, reverse_lat=True, stream_acc_threshold=100.0, flow_direction_pkg="wbw")
        
        # buildPourPointFile
        buildPourPointFile(dpc_VIC_level1, evb_dir)
        
        # modify PourPoint File to match FlowAcc
        if modify_PourPointFile_bool:
            buildPourPointFile(None, evb_dir, names=["pourpoint"], lons=[-91.905], lats=[38.335])
        
        # buildUHBOXFile
        uh_params = {"tp": 1.4, "mu": 5.0, "m": 3.0}
        buildUHBOXFile(evb_dir, **uh_params, plot_bool=True)
        
        # buildParamCFGFile
        cfg_params = {"VELOCITY": 1.5, "DIFFUSION": 800.0, "OUTPUT_INTERVAL": 86400}
        buildParamCFGFile(evb_dir, **cfg_params)
        
        from rvic.parameters import parameters
        
        RVICParam_dir = evb_dir.RVICParam_dir
        param_cfg_file_path = os.path.join(RVICParam_dir, "rvic.parameters.cfg")
        param_cfg_file = ConfigParser()
        param_cfg_file.optionxform = str
        param_cfg_file.read(param_cfg_file_path)
        
        param_cfg_file_dict = {section: dict(param_cfg_file.items(section)) for section in param_cfg_file.sections()}
        parameters(param_cfg_file_dict, numofproc=1)
    
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
    
    # buildGlobalParam
    if buildGlobalParam_bool:
        buildGlobalParam(evb_dir, GlobalParam_dict)
    
    # calibrate
    calibrate_bool = True
    if calibrate_bool:
        algParams = {"popSize": 20, "maxGen": 500, "cxProb": 0.7, "mutateProb": 0.2}
        nsgaII_VIC_SO = NSGAII_VIC_SO(dpc_VIC_level0, dpc_VIC_level1, evb_dir, date_period, calibrate_date_period,
                                      algParams=algParams, save_path=evb_dir.calibrate_cp_path, reverse_lat=True, parallel=False)
        nsgaII_VIC_SO.run()
    
    # close
    if read_domain_dataset_bool:
        domain_dataset.close()
        
    if read_params_bool:
        params_dataset_level0.close()
        params_dataset_level1.close()
    
    # read cp
    # state = readCalibrateCp(evb_dir)
    