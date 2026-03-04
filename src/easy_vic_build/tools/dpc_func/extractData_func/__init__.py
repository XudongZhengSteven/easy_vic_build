"""Data-source extraction modules used by DPC workflows.

This subpackage exposes extractors for forcing, terrain, soil, vegetation, and
observation products used to populate basin/grid datasets.
"""

# # Importing submodules for ease of access
from . import (Extract_CAMELS_Attribute, Extract_CAMELS_ForcingDaymet,
               Extract_CAMELS_Streamflow, Extract_CONUS_SOIL, Extract_ERA5_SM,
               Extract_ERA5_SoilTemperature, Extract_GLDAS, Extract_GLEAM_E,
               Extract_GlobalSnow_SWE, Extract_HWSD, Extract_MODIS_BSA,
               Extract_MODIS_LAI, Extract_MODIS_NDVI, Extract_NLDAS,
               Extract_NLDAS_annual_P, Extract_SrtmDEM, Extract_TRMM_P,
               Extract_UMD_1km, Extract_UMDLandCover, Extract_NLDAS_forcing)

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
    "Extract_NLDAS_forcing"
]
