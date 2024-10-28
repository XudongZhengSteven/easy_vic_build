# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import sys
sys.path.append("../easy_vic_build")
from easy_vic_build import Evb_dir
from easy_vic_build.bulid_Domain import buildDomain
from easy_vic_build.build_dpc import readdpc, readParam, readDomain
from easy_vic_build.build_dpc import builddpc
from easy_vic_build.build_GlobalParam import buildGlobalParam
from easy_vic_build.build_MeteForcing_nco import buildMeteForcingnco
from easy_vic_build.bulid_Param import buildParam_level0, buildParam_level1
from easy_vic_build.bulid_Param import get_default_g_list, scaling_level0_to_level1
from easy_vic_build.build_RVIC_Param import copy_domain, buildFlowDirectionFile, buildPourPointFile, buildUHBOXFile, buildParamCFGFile

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
    basin_index = 397
    date_period = ["19980101", "20101231"]
    case_name = "397_3km"
    grid_res_level1 = 0.025
    
    # ============================ set bool ============================
    build_dir_bool = True
    build_dpc_bool = False
    build_domain_bool = False
    build_param_bool = False
    build_meteforcing_bool = False
    build_rvic_param_bool = False
    build_global_param_bool = True
    # ============================ build dir ============================
    if build_dir_bool:
        evb_dir = Evb_dir()
        evb_dir.builddir(case_name)
        
        # set arcpy_python_path
        evb_dir.arcpy_python_path = "C:\\Python27\\ArcGIS10.5\\python.exe"
        
        # set MeteForcing_src_dir and MeteForcing_src_suffix
        evb_dir.MeteForcing_src_dir = "E:\\data\\hydrometeorology\\NLDAS\\NLDAS2_Primary_Forcing_Data_subset_0.125\\data"
        evb_dir.MeteForcing_src_suffix = ".nc4"
        
        # set linux_share_temp_dir
        evb_dir.linux_share_temp_dir = "F:\\Linux\\C_VirtualBox_Share\\temp"
    
    # ============================ build dpc ============================
    if build_dpc_bool:
        dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2 = builddpc(evb_dir, basin_index, date_period,
                                                                grid_res_level0=0.00833, grid_res_level1=grid_res_level1, grid_res_level2=0.125,
                                                                dpc_VIC_level0_call_kwargs={"readGriddata": True},
                                                                dpc_VIC_level1_call_kwargs={"readBasindata": True, "readGriddata": True, "readBasinAttribute": True})
        
    # read dpc_VIC
    dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2 = readdpc(evb_dir)
    
    # ============================ build domain ============================
    if build_domain_bool:
        buildDomain(dpc_VIC_level1, evb_dir, reverse_lat=True)
    
    # read domain
    domain_dataset = readDomain(evb_dir)
    
    # ============================ build Param ============================
    if build_param_bool:
        # build params_level0 with default params
        default_g_list, g_boundary =  get_default_g_list()
        params_dataset_level0 = buildParam_level0(default_g_list, dpc_VIC_level0, evb_dir, reverse_lat=True)
        
        # build params_level1
        params_dataset_level1 = buildParam_level1(dpc_VIC_level1, evb_dir, reverse_lat=True, domain_dataset=domain_dataset)
        
        # scaling_level0_to_level1
        params_dataset_level1, searched_grids_index = scaling_level0_to_level1(params_dataset_level0, params_dataset_level1)
        
        params_dataset_level0.close()
        params_dataset_level1.close()
    
    # read param
    params_dataset_level0, params_dataset_level1 = readParam(evb_dir)
    
    # ============================ build MeteForcing ============================
    if build_meteforcing_bool:
        # build MeteForcingnco: step=1
        buildMeteForcingnco(dpc_VIC_level1, evb_dir, date_period,
                            step=build_meteforcing_bool, reverse_lat=True, check_search=False,
                            year_re_exp=r"\d{4}.nc4")
        
        # go to linux, run combineYearly.py
        # build MeteForcingnco: step=2
        # buildMeteForcingnco(dpc_VIC_level1, evb_dir, date_period,
        #                     step=2, reverse_lat=True, check_search=False,
        #                     year_re_exp=r"\d{4}.nc4")
        
        # build MeteForcingnco: step=3
        # buildMeteForcingnco(dpc_VIC_level1, evb_dir, date_period,
        #                     step=3, reverse_lat=True, check_search=False,
        #                     year_re_exp=r"\d{4}.nc4")
    
    # ============================ build RVIC_Param ============================
    if build_rvic_param_bool:
        # cp domain
        copy_domain(evb_dir)
        
        # buildFlowDirectionFile
        buildFlowDirectionFile(evb_dir, params_dataset_level1, domain_dataset, reverse_lat=True, stream_acc_threshold=100.0)
        
        # buildPourPointFile
        buildPourPointFile(dpc_VIC_level1, evb_dir)
    
        # buildUHBOXFile
        uh_params = {"tp": 1.4, "mu": 5.0, "m": 3.0}
        buildUHBOXFile(evb_dir, **uh_params, plot_bool=True)
        
        # buildParamCFGFile
        cfg_params = cfg_params={"VELOCITY": 1.5, "DIFFUSION": 800.0, "OUTPUT_INTERVAL": 86400}
        buildParamCFGFile(evb_dir, **cfg_params)
    
        # build RVIC_Param
        # buildRVICParam(dpc_VIC_level1, evb_dir, params_dataset_level1, domain_dataset, reverse_lat=True, stream_acc_threshold=100.0,
        #             ppf_kwargs=dict(), uh_params={"tp": 1.4, "mu": 5.0, "m": 3.0}, uh_plot_bool=True,
        #             cfg_params={"VELOCITY": 1.5, "DIFFUSION": 800.0, "OUTPUT_INTERVAL": 86400, "RVIC_input_name": "fluxes.nc"})
    
    # ============================ build GlobalParam ============================
    if build_global_param_bool:
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
                            "OUTVAR1": {"OUTVAR": ["OUT_RUNOFF", "OUT_BASEFLOW", "OUT_PET", "OUT_DISCHARGE"]}
                            }
        
        # buildGlobalParam
        buildGlobalParam(evb_dir, GlobalParam_dict)
    
    # ============================ close ============================
    domain_dataset.close()
    params_dataset_level0.close()
    params_dataset_level1.close()
    