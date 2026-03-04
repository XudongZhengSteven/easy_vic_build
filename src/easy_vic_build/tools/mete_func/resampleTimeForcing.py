# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
"""Module ``easy_vic_build.tools.mete_func.resampleTimeForcing``."""

import os
from ... import logger
from ..utilities import remove_and_mkdir
from tqdm import *
import xarray as xr


def resampleTimeForcing(evb_dir, dst_time_hours=24):
    """
    Resample the meteorological forcing data to a different time step.

    Parameters
    ----------
    evb_dir : `Evb_dir`
        An instance of the `Evb_dir` class, containing paths for VIC deployment.
        
    dst_time_hours : int, optional
        The target time step in hours for resampling (default is 24 hours).

    Returns
    -------
    None
        The function saves the resampled data as NetCDF files in the destination directory.
    """
    logger.info(
        f"Starting to resample meteorological forcing files, dst_time_hours: {dst_time_hours}... ..."
    )

    suffix = ".nc"
    MeteForcing_dir = evb_dir.MeteForcing_dir
    resample_dir = os.path.join(MeteForcing_dir, "resample_forcing")
    remove_and_mkdir(resample_dir)
    logger.debug(f"Resample directory created at: {resample_dir}")

    dst_home = resample_dir
    src_home = MeteForcing_dir
    src_names = [n for n in os.listdir(src_home) if n.endswith(suffix)]

    # resample_factor = int(dst_time_hours / src_time_hours)

    ## ====================== loop for forcing resample time ======================
    for i in tqdm(
        range(len(src_names)), desc="loop for forcing resample time", colour="green"
    ):
        # general
        src_name = src_names[i]
        src_path = os.path.join(src_home, src_name)
        dst_path = os.path.join(dst_home, src_name)

        logger.info(f"Processing {src_name} from {src_path}")

        # Open source dataset
        try:
            src_dataset = xr.open_dataset(src_path)
            logger.debug(f"Opened source dataset: {src_path}")
        except Exception as e:
            logger.error(f"Error opening source dataset {src_path}: {e}")
            continue

        # resample data
        try:
            dst_dataset = xr.Dataset(
                {
                    "dlwrf": src_dataset["dlwrf"]
                    .resample(time=f"{dst_time_hours}H")
                    .mean(skipna=True),
                    "dswrf": src_dataset["dswrf"]
                    .resample(time=f"{dst_time_hours}H")
                    .mean(skipna=True),
                    "prcp": src_dataset["prcp"]
                    .resample(time=f"{dst_time_hours}H")
                    .sum(skipna=True),
                    "pres": src_dataset["pres"]
                    .resample(time=f"{dst_time_hours}H")
                    .mean(skipna=True),
                    "tas": src_dataset["tas"]
                    .resample(time=f"{dst_time_hours}H")
                    .mean(skipna=True),
                    "vp": src_dataset["vp"]
                    .resample(time=f"{dst_time_hours}H")
                    .mean(skipna=True),
                    "wind": src_dataset["wind"]
                    .resample(time=f"{dst_time_hours}H")
                    .mean(skipna=True),
                }
            )

            dst_dataset["lats"] = src_dataset["lats"]
            dst_dataset["lons"] = src_dataset["lons"]

            dst_dataset.to_netcdf(dst_path)
            logger.info(f"Resampled data saved to: {dst_path}")


        except Exception as e:
            logger.error(f"Error during resampling or saving {src_name}: {e}")
            continue

        finally:
            # close
            src_dataset.close()
            dst_dataset.close()

    logger.info("Resample meteorological forcing files successfully")