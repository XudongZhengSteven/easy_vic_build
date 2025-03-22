# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

""" 
Package: easy_vic_build

An Open-Source Python Framework for Scalable Deployment and Advanced Applications of VIC Model
This package provides tools for configuring, preparing, and calibrating the VIC model efficiently.
It supports automation, preprocessing, and postprocessing workflows.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com

"""
# import
import logging

# Default logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def setup_logger(log_level=None, log_format=None, log_to_file=None, log_file=None):
    """
    Allow users to modify the logger configuration. If no parameters are passed, 
    the default logger configuration remains unchanged.

    Parameters:
    -----------
    log_level : int, optional
        The logging level to set. Default is None (no change).
    
    log_format : str, optional
        The log format to set. Default is None (no change).
    
    log_to_file : bool, optional
        Whether to log to a file. Default is None (no change).
    
    log_file : str, optional
        If logging to a file, specify the file path. Default is None (no change).

    Returns:
    --------
    None
        The logger is updated in place based on the provided parameters.
        
    Example:
    --------
    # User wants to modify the logger setup. To use this, the user would call `setup_logger` with their desired configuration:
    setup_logger(log_level=logging.DEBUG, log_format="%(asctime)s - %(levelname)s - %(message)s", log_to_file=True, log_file="custom_log.log")
    """
    
    # If user provides a new log level, update it
    if log_level is not None:
        logger.setLevel(log_level)
    
    # If user provides a new format, update it
    if log_format is not None:
        formatter = logging.Formatter(log_format)
        for handler in logger.handlers:
            handler.setFormatter(formatter)

    # If user provides options to log to file, add a file handler
    if log_to_file is not None and log_to_file:
        if log_file is None:
            log_file = "evb.log"  # Default log file name
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

# Default logger configuration (user doesn't need to call anything)
logger.debug("Default logger setup with INFO level.")

# Test to ensure default logging is working
logger.info("This is an info message with the default setup.")

# Log the configuration details
logger.info("--------------- EVB Configuration ---------------")

try:
    import nco
    HAS_NCO = True
    logger.info("NCO: Using MeteForcing with nco")
    from . import build_MeteForcing_nco
except:
    HAS_NCO = False
    logger.warning("NCO: Using MeteForcing without nco")   
    from . import build_MeteForcing

try:
    from rvic.parameters import parameters as rvic_parameters
    logger.info("RVIC: You have rvic!")
    HAS_RVIC = True
except:
    logger.warning("RVIC: No RVIC detected, but easy_vic_build is still usable.")
    HAS_RVIC = False

logger.info("-------------------------------------------------")

# import
from . import tools
from . import build_dpc, build_GlobalParam, build_hydroanalysis, build_RVIC_Param, bulid_Domain, bulid_Param, calibrate, warmup

# Define the package's public API and version
__all__ = ["build_dpc", "build_GlobalParam", "build_hydroanalysis", "build_RVIC_Param", "bulid_Domain", "bulid_Param", "calibrate", "warmup", "tools", "build_MeteForcing_nco", "build_MeteForcing", "logger"]
__version__ = "0.1.0"
__author__ = "Xudong Zheng"
__email__ = "z786909151@163.com"