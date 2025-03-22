"""
Subpackage: extractData_func

A Subpackage of easy_vic_build.tools.dpc_func

This subpackage contains a collection of modules that provide functions for extracting data.

Modules:
--------
    - Extract_CAMELS_ForcingDaymet: Extracts Daymet forcing data for CAMELS datasets.
    - Extract_CONUS_SOIL: Extracts soil data for the CONUS region.
    - Extract_ERA5_SM: Extracts soil moisture data from the ERA5 dataset.
    - Extract_ERA5_SoilTemperature: Extracts soil temperature data from the ERA5 dataset.
    - Extract_GLDAS: Extracts GLDAS data for global land data analysis.
    - Extract_GLEAM_E: Extracts evapotranspiration data from the GLEAM dataset.
    - Extract_GlobalSnow_SWE: Extracts snow water equivalent (SWE) data from global snow datasets.
    - Extract_HWSD: Extracts soil data from the HWSD (Harmonized World Soil Database) dataset.
    - Extract_MODIS_BSA: Extracts bare soil fraction data from MODIS.
    - Extract_MODIS_LAI: Extracts leaf area index (LAI) data from MODIS.
    - Extract_MODIS_NDVI: Extracts normalized difference vegetation index (NDVI) data from MODIS.
    - Extract_NLDAS_annual_P: Extracts annual precipitation data from NLDAS.
    - Extract_NLDAS: Extracts general NLDAS forcing data.
    - Extract_SrtmDEM: Extracts Digital Elevation Model (DEM) data from SRTM.
    - Extract_TRMM_P: Extracts precipitation data from TRMM.
    - Extract_UMD_1km: Extracts land cover data from the UMD dataset (1km resolution).
    - Extract_UMDLandCover: Extracts land cover data from UMD's global land cover map.
    - Extract_CAMELS_Streamflow: Extracts streamflow data from the CAMELS dataset.
    - Extract_CAMELS_Attribute: Extracts attribute data for CAMELS basins.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com

"""

# # Importing submodules for ease of access
from . import (Extract_CAMELS_Attribute, Extract_CAMELS_ForcingDaymet,
               Extract_CAMELS_Streamflow, Extract_CONUS_SOIL, Extract_ERA5_SM,
               Extract_ERA5_SoilTemperature, Extract_GLDAS, Extract_GLEAM_E,
               Extract_GlobalSnow_SWE, Extract_HWSD, Extract_MODIS_BSA,
               Extract_MODIS_LAI, Extract_MODIS_NDVI, Extract_NLDAS,
               Extract_NLDAS_annual_P, Extract_SrtmDEM, Extract_TRMM_P,
               Extract_UMD_1km, Extract_UMDLandCover)

# Define the package's public API and version
__all__ = [
    "Extract_CAMELS_ForcingDaymet",
    "Extract_CONUS_SOIL",
    "Extract_ERA5_SM",
    "Extract_ERA5_SoilTemperature",
    "Extract_GLDAS",
    "Extract_GLEAM_E",
    "Extract_GlobalSnow_SWE",
    "Extract_HWSD",
    "Extract_MODIS_BSA",
    "Extract_MODIS_LAI",
    "Extract_MODIS_NDVI",
    "Extract_NLDAS_annual_P",
    "Extract_NLDAS",
    "Extract_SrtmDEM",
    "Extract_TRMM_P",
    "Extract_UMD_1km",
    "Extract_UMDLandCover",
    "Extract_CAMELS_Streamflow",
    "Extract_CAMELS_Attribute",
]
