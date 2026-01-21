# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from typing import Any

import matplotlib.pyplot as plt

from ... import logger
from ..utilities import *
from ..decoractors import processing_step
from .basin_grid_class import *
from .dpc_base import dataProcess_base
from .extractData_func import *
from copy import deepcopy

class dataProcess_VIC_level0(dataProcess_base):
    
    @processing_step(
        step_name="load_dem",
        save_names="dem", 
        data_level="grid_level",
        deps=["load_basin_shp", "load_grid_shp"]
    )
    def load_dem(self):
        grid_shp = deepcopy(self.loaddata_kwargs["grid_shp"])
        grid_res = self.loaddata_kwargs["grid_res"]
        
        logger.info(f"Loading SRTM DEM data with resolution {grid_res} into grid... ...")
        
        grid_shp_with_dem = Extract_SrtmDEM.ExtractData(
            grid_shp,
            grid_res,
            plot=False,
            save_original=False,
            check_search=True,
        )
        
        logger.info("SRTM DEM data successfully loaded into grids")
        
        ret = {"dem": grid_shp_with_dem}
        
        return ret
    
    
    @processing_step(
        step_name="load_soil",
        save_names="soil",
        data_level="grid_level",
        deps=["load_basin_shp", "load_grid_shp"]
    )
    def load_soil(self):
        grid_shp = deepcopy(self.loaddata_kwargs["grid_shp"])
        grid_res = self.loaddata_kwargs["grid_res"]
        
        logger.info(f"Loading CONUS soil data into grids with resolution {grid_res}... ...")
        
        grid_shp_with_soil = Extract_CONUS_SOIL.ExtractData(
            grid_shp,
            grid_res,
            plot_layer=False,
            save_original=False,
            check_search=True,
        )
    
        logger.info("CONUS soil data successfully loaded into grids")
        
        ret = {"soil": grid_shp_with_soil}
        return ret
    
    
class dataProcess_VIC_level1(dataProcess_base):
    
    @processing_step(
        step_name="load_st",
        save_names="st",
        data_level="grid_level",
        deps=["load_basin_shp", "load_grid_shp"]
    )
    def load_st(self):
        grid_shp = deepcopy(self.loaddata_kwargs["grid_shp"])
        grid_res = self.loaddata_kwargs["grid_res"]
        search_method = self.loaddata_kwargs["search_method_st"]
        
        logger.info(
            f"Loading ERA5 soil temperature data into grid with resolution {grid_res}... ..."
        )
        
        grid_shp_with_st = Extract_ERA5_SoilTemperature.ExtractData(
            grid_shp,
            grid_res,
            plot_layer=False,
            check_search=True,
            search_method=search_method,
        )
        
        logger.info("ERA5 soil temperature data successfully loaded into grids")
        
        ret = {"st": grid_shp_with_st}
        
        return ret
    
    @processing_step(
        step_name="load_annual_P",
        save_names="annual_P",
        data_level="grid_level",
        deps=["load_basin_shp", "load_grid_shp"]
    )
    def load_annual_P(self):
        grid_shp = deepcopy(self.loaddata_kwargs["grid_shp"])
        grid_res = self.loaddata_kwargs["grid_res"]
        search_method = self.loaddata_kwargs["search_method_annual_P"]
        
        logger.info(f"Loading NLDAS annual precipitation into grid with resolution {grid_res}... ...")
        
        grid_shp_with_annual_P = Extract_NLDAS_annual_P.ExtractData(
            grid_shp,
            grid_res,
            plot=False,
            check_search=True,
            search_method=search_method,
        )
        
        logger.info("NLDAS annual precipitation data successfully loaded into grids")
        
        ret = {"annual_P": grid_shp_with_annual_P}
        
        return ret
    
    @processing_step(
        step_name="load_lulc",
        save_names="lulc",
        data_level="grid_level",
        deps=["load_basin_shp", "load_grid_shp"]
    )
    def load_lulc(self):
        grid_shp = deepcopy(self.loaddata_kwargs["grid_shp"])
        grid_res = self.loaddata_kwargs["grid_res"]
        
        logger.info(f"Loading UMD land cover data into grid with resolution {grid_res}")
        
        grid_shp_with_lulc = Extract_UMD_1km.ExtractData(
            grid_shp,
            grid_res,
            plot=False,
            save_original=True,
            check_search=True,
        )
        
        logger.info("UMD land cover data successfully loaded into grids.")
        
        ret = {"lulc": grid_shp_with_lulc}
        
        return ret
    
    @processing_step(
        step_name="load_bsa",
        save_names="bsa",
        data_level="grid_level",
        deps=["load_basin_shp", "load_grid_shp", "load_lulc"]
    )
    def load_bsa(self):
        grid_shp = deepcopy(self.loaddata_kwargs["grid_shp"])
        grid_res = self.loaddata_kwargs["grid_res"]
        
        logger.info(f"Loading MODIS BSA data into grid with resolution {grid_res}... ...")
        
        grid_shp_with_bsa = Extract_MODIS_BSA.ExtractData(
            grid_shp,
            grid_res,
            plot_month=False,
            save_original=True,
            check_search=True,
        )
        
        logger.info("MODIS BSA data successfully loaded into grids")
        
        ret = {"bsa": grid_shp_with_bsa}
        
        return ret
    
    @processing_step(
        step_name="load_ndvi",
        save_names="ndvi",
        data_level="grid_level",
        deps=["load_basin_shp", "load_grid_shp", "load_lulc"]
    )
    def load_ndvi(self):
        grid_shp = deepcopy(self.loaddata_kwargs["grid_shp"])
        grid_res = self.loaddata_kwargs["grid_res"]
        
        logger.info(f"Loading MODIS NDVI data into grid with resolution {grid_res}... ...")
        
        grid_shp_with_ndvi = Extract_MODIS_NDVI.ExtractData(
            grid_shp,
            grid_res,
            plot_month=False,
            save_original=True,
            check_search=True,
        )
        
        logger.info("MODIS NDVI data successfully loaded into grids")
        
        ret = {"ndvi": grid_shp_with_ndvi}
        
        return ret
    
    @processing_step(
        step_name="load_lai",
        save_names="lai",
        data_level="grid_level",
        deps=["load_basin_shp", "load_grid_shp", "load_lulc"]
    )
    def load_lai(self):
        grid_shp = deepcopy(self.loaddata_kwargs["grid_shp"])
        grid_res = self.loaddata_kwargs["grid_res"]
        
        logger.info(f"Loading MODIS LAI data into grid with resolution {grid_res}... ...")
        
        grid_shp_with_lai = Extract_MODIS_LAI.ExtractData(
            grid_shp,
            grid_res,
            plot_month=False,
            save_original=True,
            check_search=True,
        )
        
        logger.info("MODIS LAI data successfully loaded into grids")
        
        ret = {"lai": grid_shp_with_lai}
        
        return ret
    
class dataProcess_VIC_level2(dataProcess_base):

    @processing_step(
        step_name="load_NLDAS_forcing",
        save_names="NLDAS_forcing",
        data_level="grid_level",
        deps=["load_basin_shp", "load_grid_shp"]
    )
    def load_forcing(self):
        grid_shp = deepcopy(self.loaddata_kwargs["grid_shp"])
        grid_res = self.loaddata_kwargs["grid_res"]
        date_period = self.loaddata_kwargs["date_period"]
        search_method = self.loaddata_kwargs["search_method"]
        N_PROCESS = self.loaddata_kwargs.get("N_PROCESS", 8)
        CHUNK_SIZE = self.loaddata_kwargs.get("CHUNK_SIZE", 60)
        
        logger.info(
            f"Loading forcing data into grid with resolution {grid_res}... ..."
        )
        
        grid_shp_with_NLDAS_forcing = Extract_NLDAS_forcing.ExtractData(
            grid_shp,
            date_period,
            search_method=search_method,
            grid_shp_res=grid_res,
            plot=True,
            check_search=True,
            N_PROCESS=N_PROCESS,
            CHUNK_SIZE=CHUNK_SIZE,
        )
        
        logger.info("NLDAS forcing data successfully loaded into grids")
        
        ret = {"NLDAS_forcing": grid_shp_with_NLDAS_forcing}
        
        return ret
        
class dataProcess_VIC_level3(dataProcess_base):
    def load_grid_shp(self):
        pass
    
    @processing_step(
        step_name="load_streamflow",
        save_names="streamflow",
        data_level="basin_level",
        deps=["load_basin_shp"]
    )
    def load_streamflow(self):
        basin_shp = self.loaddata_kwargs["basin_shp"]
        date_period = self.loaddata_kwargs["date_period"]
        
        logger.info(f"Loading streamflow data for basin {basin_shp.hru_id.values[0]} with dates: {date_period}... ...")
        
        basin_shp_with_streamflow = Extract_CAMELS_Streamflow.ExtractData(
            basin_shp,
            read_dates=date_period
        )
        
        logger.info("CAMELS streamflow data successfully loaded into basins")
        
        ret = {"streamflow": basin_shp_with_streamflow}
    
        return ret
        
    @processing_step(
        step_name="load_basin_attribute",
        save_names="basin_attribute",
        data_level="basin_level",
        deps=["load_basin_shp"]
    )
    def load_basin_attribute(self):
        basin_shp = self.loaddata_kwargs["basin_shp"]
        k_list = self.loaddata_kwargs["k_list"]
        
        logger.info(f"Loading basin_attribute data for basin {basin_shp.hru_id.values[0]}... ...")
        
        basin_shp_with_basin_attribute = Extract_CAMELS_Attribute.ExtractData(
            basin_shp,
            k_list=k_list,
        )
            
        logger.info("CAMELS basin_attribute data successfully loaded into basins")
        
        ret = {"basin_attribute": basin_shp_with_basin_attribute}
    
        return ret
        
        
        
        