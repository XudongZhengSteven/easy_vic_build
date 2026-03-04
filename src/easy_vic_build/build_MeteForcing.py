# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Build VIC meteorological forcing files from level-2 DPC data.

Public function
---------------
``buildMeteForcing``
    Create forcing files for a date range through a pluggable interface class.
"""

from tqdm import *

from .tools.decoractors import clock_decorator

from . import logger
from .tools.mete_func.build_MeteForcing_interface import buildMeteForcing_interface

@clock_decorator(print_arg_ret=False)
def buildMeteForcing(
    evb_dir,
    dpc_VIC_level2,
    date_period_process,
    date_period_forcing,
    buildMeteForcing_interface: buildMeteForcing_interface,
    date_format="%Y%m%d %H:%M:%S",
    timestep="D",
    reverse_lat=True,
    stand_grids_lat_level2=None,
    stand_grids_lon_level2=None,
    rows_index_level2=None,
    cols_index_level2=None,
    file_format="NETCDF4",
):
    """Build meteorological forcing files for one case.

    Parameters
    ----------
    evb_dir : Evb_dir
        Case directory manager.
    dpc_VIC_level2 : object
        DPC instance used by the forcing interface.
    date_period_process : list
        Source data period used by the interface.
    date_period_forcing : list
        Output forcing period.
    buildMeteForcing_interface : type
        Interface class implementing ``buildMeteForcing_loop_years``.
    date_format : str, optional
        Datetime parsing/format string used in the interface.
    timestep : str, optional
        Output timestep tag (for example ``"D"``).
    reverse_lat : bool, optional
        Whether latitude axis is arranged north-to-south.
    stand_grids_lat_level2, stand_grids_lon_level2, rows_index_level2, cols_index_level2 : optional
        Precomputed grid metadata for repeated runs.
    file_format : str, optional
        NetCDF output format used by the interface.

    Returns
    -------
    object
        The created forcing-interface instance.
    """
    logger.info("Starting to building meteForcing... ...")
    
    buildMeteForcing_interface_instance = buildMeteForcing_interface(
        evb_dir,
        logger,
        dpc_VIC_level2,
        date_period_process,
        date_period_forcing,
        date_format,
        timestep,
        reverse_lat,
        stand_grids_lat_level2,
        stand_grids_lon_level2,
        rows_index_level2,
        cols_index_level2,
        file_format,
    )
    
    buildMeteForcing_interface_instance.buildMeteForcing_loop_years()
    logger.info(f"Building meteForcing successfully, meteForcing files has been saved to {evb_dir.MeteForcing_dir}")

    return buildMeteForcing_interface_instance
