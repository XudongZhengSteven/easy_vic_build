# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Run a VIC warm-up period and write a model state file."""

import os

from . import logger
from .tools.params_func.GlobalParamParser import GlobalParamParser


def warmup_VIC(evb_dir, warmup_period):
    """
    Run VIC for a warm-up period and store the end state.

    Parameters
    ----------
    evb_dir : Evb_dir
        Case directory manager. Requires ``evb_dir.vic_exe_path`` and an existing
        global parameter file at ``evb_dir.globalParam_path``.
    warmup_period : list of str
        Two dates ``[start, end]`` formatted as ``YYYYMMDD``.

    Returns
    -------
    None
    """
    # this is only useful is you just warm up the model and not to run it
    # in generl, you can run the mode across the total date_period, and ignore the warm-up period when you calibrate and evaluate

    logger.info("Loading global parameter file and preparing for warmup... ...")
    ## ====================== set Global param ======================
    # * note: make sure you have already a globalparam file, modify on built globalparam file
    # read global param
    globalParam = GlobalParamParser()
    globalParam.load(evb_dir.globalParam_path)

    # update date period
    logger.info(f"Setting simulation period: {warmup_period[0]} to {warmup_period[1]}")
    globalParam.set("Simulation", "STARTYEAR", str(warmup_period[0][:4]))
    globalParam.set("Simulation", "STARTMONTH", str(warmup_period[0][4:6]))
    globalParam.set("Simulation", "STARTDAY", str(warmup_period[0][6:8]))
    globalParam.set("Simulation", "ENDYEAR", str(warmup_period[1][:4]))
    globalParam.set("Simulation", "ENDMONTH", str(warmup_period[1][4:6]))
    globalParam.set("Simulation", "ENDDAY", str(warmup_period[1][6:8]))

    # set [State Files], the last day of the warmup_period will be saved as states
    logger.info(
        f"Setting state file save parameters for the last day of the warmup period: {warmup_period[1]}... ..."
    )
    globalParam.set(
        "State Files", "STATENAME", os.path.join(evb_dir.VICStates_dir, "states.")
    )
    globalParam.set("State Files", "STATEYEAR", str(warmup_period[1][:4]))
    globalParam.set("State Files", "STATEMONTH", str(warmup_period[1][4:6]))
    globalParam.set("State Files", "STATEDAY", str(warmup_period[1][6:8]))
    globalParam.set("State Files", "STATESEC", str(86400))
    globalParam.set("State Files", "STATE_FORMAT", "NETCDF4")

    # write
    with open(evb_dir.globalParam_path, "w") as f:
        globalParam.write(f)

    ## ====================== run vic and save state ======================
    logger.info("Running VIC model for the warmup period... ...")
    command_run_vic = " ".join([evb_dir.vic_exe_path, "-g", evb_dir.globalParam_path])
    try:
        os.system(command_run_vic)
        logger.info("VIC model run successfully")
    except Exception as e:
        logger.error(f"Failed to run VIC model: {e}")
        raise e

    logger.info(f"warmup successfully, state files have been saved to {evb_dir.VICStates_dir}")
