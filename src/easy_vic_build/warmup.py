# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: warmup

This module provides functionality for warming up the VIC (Variable Infiltration Capacity) model
by setting up the simulation period, modifying the global parameter file, and running the model for the warmup period.
The function `warmup_VIC` adjusts the start and end dates for the simulation, saves the model state,
and executes the VIC model with the updated parameters.

Functions:
----------
    - warmup_VIC: Adjusts the global parameters for the warmup period, runs the VIC model,
      and saves the model state after the warmup.

Usage:
------
    To use this module, provide an `evb_dir` instance containing paths to necessary directories and files,
    as well as a `warmup_period` which specifies the start and end dates of the warmup period.

    The `warmup_VIC` function will modify the global parameter file and execute the VIC model using
    the provided warmup period, saving the model state at the end of the warmup period.

Example:
--------
    evb_dir = Evb_dir("./rvic_example")
    evb_dir.builddir("case_study")

    warmup_period = ["19980101", "19981231"]
    warmup_VIC(evb_dir, warmup_period)

Dependencies:
-------------
    - os: For file and directory operations and running shell commands.
    - .tools.params_func.GlobalParamParser: For reading and writing the global parameter configuration file.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com
"""

import os

from . import logger
from .tools.params_func.GlobalParamParser import GlobalParamParser


def warmup_VIC(evb_dir, warmup_period):
    """
    Perform a warmup of the VIC model by modifying the global parameter file
    and running the model for the specified warmup period.

    The warmup period is defined by the `warmup_period` parameter, which should be a list
    containing the start and end dates as strings in the format "YYYYMMDD".

    Parameters:
    -----------
    evb_dir : object
        The directory object that contains the paths to required directories and files,
        including the path to the global parameter file and VIC executable.

    warmup_period : list of str
        A list containing the start and end dates of the warmup period,
        in the format ["YYYYMMDD", "YYYYMMDD"].

    Returns:
    --------
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
    globalParam.set("Simulation", "STARTDAY", str(warmup_period[0][6:]))
    globalParam.set("Simulation", "ENDYEAR", str(warmup_period[1][:4]))
    globalParam.set("Simulation", "ENDMONTH", str(warmup_period[1][4:6]))
    globalParam.set("Simulation", "ENDDAY", str(warmup_period[1][6:]))

    # set [State Files], the last day of the warmup_period will be saved as states
    logger.info(
        f"Setting state file save parameters for the last day of the warmup period: {warmup_period[1]}"
    )
    globalParam.set(
        "State Files", "STATENAME", os.path.join(evb_dir.VICStates_dir, "states.")
    )
    globalParam.set("State Files", "STATEYEAR", str(warmup_period[1][:4]))
    globalParam.set("State Files", "STATEMONTH", str(warmup_period[1][4:6]))
    globalParam.set("State Files", "STATEDAY", str(warmup_period[1][6:]))
    globalParam.set("State Files", "STATESEC", str(86400))
    globalParam.set("State Files", "STATE_FORMAT", "NETCDF4")

    # write
    with open(evb_dir.globalParam_path, "w") as f:
        globalParam.write(f)

    ## ====================== run vic and save state ======================
    logger.info("Running VIC model for the warmup period")
    command_run_vic = " ".join([evb_dir.vic_exe_path, "-g", evb_dir.globalParam_path])
    try:
        os.system(command_run_vic)
        logger.info("VIC model run completed successfully.")
    except Exception as e:
        logger.error(f"Failed to run VIC model: {e}")
        raise e

    logger.info(f"warmup complete successfully, state saved to {evb_dir.VICStates_dir}")
