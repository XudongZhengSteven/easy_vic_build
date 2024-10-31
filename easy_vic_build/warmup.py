# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import os
from .tools.params_func.GlobalParamParser import GlobalParamParser


def warmup_VIC(evb_dir, warmup_period):
    # this is only useful is you just warm up the model and not to run it
    # in generl, you can run the mode across the total date_period, and ignore the warm-up period when you calibrate and evaluate
    ## ====================== set dir and path ======================
    # set path
    GlobalParam_dir = evb_dir.GlobalParam_dir
    globalParam_path = os.path.join(GlobalParam_dir, "global_param.txt")
    
    ## ====================== set Global param ======================
    #* note: make sure you have already a globalparam file, modify on built globalparam file
    # read global param
    globalParam = GlobalParamParser()
    globalParam.load(globalParam_path)
    
    # update date period
    globalParam.set("Simulation", "STARTYEAR", str(warmup_period[0][:4]))
    globalParam.set("Simulation", "STARTMONTH", str(warmup_period[0][4:6]))
    globalParam.set("Simulation", "STARTDAY", str(warmup_period[0][6:]))
    globalParam.set("Simulation", "ENDYEAR", str(warmup_period[1][:4]))
    globalParam.set("Simulation", "ENDMONTH", str(warmup_period[1][4:6]))
    globalParam.set("Simulation", "ENDDAY", str(warmup_period[1][6:]))

    # set [State Files], the last day of the warmup_period will be saved as states
    globalParam.set("State Files", "STATENAME", os.path.join(evb_dir.VICStates_dir, "states."))
    globalParam.set("State Files", "STATEYEAR", str(warmup_period[1][:4]))
    globalParam.set("State Files", "STATEMONTH", str(warmup_period[1][4:6]))
    globalParam.set("State Files", "STATEDAY", str(warmup_period[1][6:]))
    globalParam.set("State Files", "STATESEC", str(86400))
    globalParam.set("State Files", "STATE_FORMAT", "NETCDF4")
    
    # write
    with open(globalParam_path, "w") as f:
        globalParam.write(f)
    
    ## ====================== run vic and save state ======================
    command_run_vic = " ".join([evb_dir.vic_exe_path, "-g", globalParam_path])
    os.system(command_run_vic)