# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Top-level package for ``easy_vic_build``.

This module exposes commonly used builders and utilities and performs optional
feature detection at import time:

- If ``nco`` is available, ``build_MeteForcing_nco`` is imported.
- Otherwise, fallback ``build_MeteForcing`` is imported.
- If ``rvic`` is available, ``HAS_RVIC`` is set to ``True``.
"""

# import
from .Logger import logger, setup_logger
from . import (build_GlobalParam, build_hydroanalysis,
               build_RVIC_Param, bulid_Domain, build_Param,
               calibrate, tools, warmup)

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
    "build_GlobalParam",
    "build_hydroanalysis",
    "build_RVIC_Param",
    "bulid_Domain",
    "build_Param",
    "calibrate",
    "warmup",
    "tools",
    "build_MeteForcing",
    "logger",
    "setup_logger",
]

__version__ = "0.1.0"
__author__ = "Xudong Zheng"
__email__ = "z786909151@163.com"
