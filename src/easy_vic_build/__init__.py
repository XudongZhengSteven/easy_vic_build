# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
easy_vic_build - A Python package for easily building VIC model.

This package provides an open-source Python framework for scalable deployment and advanced applications
of the VIC model. It streamlines the process of configuring, preparing, and calibrating the VIC model,
and supports automation, preprocessing, and postprocessing workflows. The package is designed to
improve efficiency and reduce the complexity of VIC model deployment.

Submodules:
-----------
    - `tools`: A subpackage containing utility modules for supporting VIC model deployment.
    - `build_dpc`: Module for building data processing class at three levels.
    - `build_GlobalParam`: Module for building the global parameter file.
    - `build_hydroanalysis`: Module for performing hydroanalysis tasks.
    - `build_MeteForcing_nco`: Module for building meteorological forcing files with nco.
    - `build_MeteForcing`: Module for building meteorological forcing files without nco.
    - `build_RVIC_Param`: Module for building RVIC parameter files.
    - `build_Domain`: Module for building domain files.
    - `build_Param`: Module for building VIC parameter files.
    - `calibrate`: Module for calibrating VIC model.
    - `Evb_dir_class`: Module containing Evb_dir class for managing path and directory.
    - `warmup`: Module for warmuping VIC model.

Usage:
------
    1. Build DPC (`build_dpc`)
    2. Build Domain (`build_Domain`)
    3. Build Parameters (`build_Param`)
    4. Perform Hydroanalysis (`build_hydroanalysis`)
    5. Build Meteorological Forcing (`build_MeteForcing`) or (`build_MeteForcing_nco`)
    6. Build RVIC Parameters (`build_RVIC_Param`)
    7. Build Global Parameters (`build_GlobalParam`)
    8. Calibrate the Model (`calibrate`)
    9. Plot Basin Map (`plot_Basin_map`), note that you must first run `hydroanalysis_for_basin`
    10. Plot VIC Results (`plot_VIC_result`)

Version: 0.1.0
Author: Xudong Zheng
License: MIT

"""

# import
from .Logger import logger, setup_logger
from . import (build_dpc, build_GlobalParam, build_hydroanalysis,
               build_RVIC_Param, bulid_Domain, bulid_Param, calibrate, tools,
               warmup)

# Log the configuration details
logger.info("---------------------- EVB Configuration ----------------------")

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

    logger.info("RVIC: RVIC package has been imported.")
    HAS_RVIC = True
except:
    logger.warning("RVIC: No RVIC detected, but easy_vic_build is still usable.")
    HAS_RVIC = False

logger.info("---------------------------------------------------------------")

# Define the package's public API and version
__all__ = [
    "build_dpc",
    "build_GlobalParam",
    "build_hydroanalysis",
    "build_RVIC_Param",
    "bulid_Domain",
    "bulid_Param",
    "calibrate",
    "warmup",
    "tools",
    "build_MeteForcing_nco",
    "build_MeteForcing",
    "logger",
    "setup_logger",
]

__version__ = "0.1.0"
__author__ = "Xudong Zheng"
__email__ = "z786909151@163.com"
