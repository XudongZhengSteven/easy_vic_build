# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
import geopandas as gpd
from JRB_build_evb_dir import build_modeling_dir

from easy_vic_build.tools.dpc_func.basin_grid_class import Basins, Grids
from easy_vic_build.tools.dpc_func.basin_grid_func import build_grid_shp
from easy_vic_build.tools.dpc_func.dpc_base import dataProcess_base
from easy_vic_build.tools.decoractors import processing_step
from easy_vic_build.tools.dpc_func.extractData_func import Extract_UMD_1km

from JRB_extractData_func import Extract_ASTGTM2DEM, Extract_SoilGrids1km, Extract_ERA5_SoilTemperature, Extract_Annual_P, Extract_stationdata_streamflow
from JRB_extractData_func import Extract_MODIS_BSA, Extract_MODIS_NDVI, Extract_MODIS_LAI, Extract_CMADSV1_forcing, Extract_CDMet_forcing, Extract_CMFD_forcing
from JRB_extractData_func import Extract_GLEAM_SM, Extract_GLEAM_ET

from easy_vic_build import logger
from general_info import *
import matplotlib.pyplot as plt

from copy import deepcopy


## --------------------------- dataProcess_VIC ---------------------------
class dataProcess_VIC_level0_JRB(dataProcess_base):
    
    @processing_step(
        step_name="load_dem",
        save_names="dem", 
        data_level="grid_level",
        deps=["load_basin_shp", "load_grid_shp"]
    )
    def load_dem(self):
        grid_shp = deepcopy(self.loaddata_kwargs["grid_shp"])
        grid_res = self.loaddata_kwargs["grid_res"]
        
        logger.info(f"Loading ASTGTM2 DEM data into grids with resolution {grid_res}... ...")
        
        grid_shp_with_dem = Extract_ASTGTM2DEM.ExtractData(
            grid_shp,
            grid_res,
            plot=False,
            save_original=False,
            check_search=False,
        )
        
        logger.info("ASTGTM2 DEM data successfully loaded into grids")
        
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
        
        logger.info(f"Loading SoilGrids1km data into grids with resolution {grid_res}... ...")
        
        grid_shp_with_soil = Extract_SoilGrids1km.ExtractData(
            grid_shp,
            grid_res,
            plot_layer=False,
            check_search=False,
        )
    
        logger.info("SoilGrids1km data successfully loaded into grids")
        
        ret = {"soil": grid_shp_with_soil}
        return ret


class dataProcess_VIC_level1_JRB(dataProcess_base):
    
    @processing_step(
        step_name="load_st",
        save_names="st",
        data_level="grid_level",
        deps=["load_basin_shp", "load_grid_shp"]
    )
    def load_st(self):
        grid_shp = deepcopy(self.loaddata_kwargs["grid_shp"])
        grid_res = self.loaddata_kwargs["grid_res"]
        date_period = self.loaddata_kwargs["date_period"]
        search_method = self.loaddata_kwargs["search_method_st"]
        
        logger.info(
            f"Loading ERA5 soil temperature data into grid with resolution {grid_res}... ..."
        )
                
        logger.info(f"search method for ST: {search_method} (grid_res:{grid_res}, source ST_res: 0.1)")
        
        grid_shp_with_st = Extract_ERA5_SoilTemperature.ExtractData(
            grid_shp,
            grid_res,
            date_period,
            search_method,
            check_search=False,
            plot=False,
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
        date_period = self.loaddata_kwargs["date_period"]
        evb_dir = self.loaddata_kwargs["evb_dir"]
        reverse_lat = self.loaddata_kwargs["reverse_lat"]
        
        logger.info(f"Loading annual precipitation data into grid with resolution {grid_res}... ...")
        
        grid_shp_with_annual_P = Extract_Annual_P.ExtractData(
            grid_shp,
            evb_dir,
            date_period,
            plot=True,
            reverse_lat=reverse_lat
        )
        
        logger.info("Annual precipitation data successfully loaded into grids")
        
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
            check_search=False,
        )
        
        logger.info("UMD land cover data successfully loaded into grids")
        
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
            check_search=False,
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
            check_search=False,
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
            check_search=False,
        )
        
        logger.info("MODIS LAI data successfully loaded into grids")
        
        ret = {"lai": grid_shp_with_lai}
        
        return ret
    
    
class dataProcess_VIC_level1_GLEAM_JRB(dataProcess_base):
    @processing_step(
        step_name="load_sm",
        save_names="GLEAM_sm",
        data_level="grid_level",
        deps=["load_basin_shp", "load_grid_shp"]
    )
    def load_sm(self):
        grid_shp = deepcopy(self.loaddata_kwargs["grid_shp"])
        grid_res = self.loaddata_kwargs["grid_res"]
        date_period = self.loaddata_kwargs["date_period"]
        search_method = self.loaddata_kwargs["search_method"]
        
        logger.info(f"search method for GLEAM: {search_method} (grid_res:{grid_res}, source res: 0.25 deg)")
        
        grid_shp_with_GLEAM_sm = Extract_GLEAM_SM.ExtractData(
            grid_shp,
            grid_res,
            date_period,
            search_method,
            plot=False,
            check_search=True,
        )
        
        logger.info("GLEAM data successfully loaded into grids")
        
        ret = {"GLEAM_sm": grid_shp_with_GLEAM_sm}
        
        return ret

    @processing_step(
        step_name="load_E",
        save_names="GLEAM_E",
        data_level="grid_level",
        deps=["load_basin_shp", "load_grid_shp"]
    )
    def load_E(self):
        grid_shp = deepcopy(self.loaddata_kwargs["grid_shp"])
        grid_res = self.loaddata_kwargs["grid_res"]
        date_period = self.loaddata_kwargs["date_period"]
        search_method = self.loaddata_kwargs["search_method"]
        
        logger.info(f"search method for GLEAM: {search_method} (grid_res:{grid_res}, source res: 0.25 deg)")
        
        grid_shp_with_GLEAM_E = Extract_GLEAM_ET.ExtractData(
            grid_shp,
            grid_res,
            date_period,
            search_method,
            plot=False,
            check_search=True,
        )
        
        logger.info("GLEAM data successfully loaded into grids")
        
        ret = {"GLEAM_E": grid_shp_with_GLEAM_E}
        
        return ret


class dataProcess_VIC_level2_CMADSV1_JRB(dataProcess_base):
    
    @processing_step(
        step_name="load_cmadsv1_forcing",
        save_names="cmadsv1_forcing",
        data_level="grid_level",
        deps=["load_basin_shp", "load_grid_shp"]
    )
    def load_cmadsv1_forcing(self):
        grid_shp = deepcopy(self.loaddata_kwargs["grid_shp"])
        grid_res = self.loaddata_kwargs["grid_res"]
        date_period = self.loaddata_kwargs["date_period"]
        reverse_lat = self.loaddata_kwargs["reverse_lat"]
        search_method = self.loaddata_kwargs["search_method"]
        
        logger.info(
            f"Loading CMADS forcing data into grid with resolution {grid_res}... ..."
        )
        
        logger.info(f"search method for CMADSV1: {search_method} (grid_res:{grid_res}, source res: 1/3 degree)")
        
        grid_shp_with_cmadsv1_forcing = Extract_CMADSV1_forcing.ExtractData(
            grid_shp,
            grid_res,
            date_period,
            search_method,
            plot=False,
            check_search=False,
            reverse_lat=reverse_lat,
            time_UTC=12,
            elevation=1368,
        )
        
        logger.info("CMADS forcing data successfully loaded into grids")
        
        ret = {"cmadsv1_forcing": grid_shp_with_cmadsv1_forcing}
        
        return ret


class dataProcess_VIC_level2_CMFD_JRB(dataProcess_base):
    
    @processing_step(
        step_name="load_cmfd_forcing",
        save_names="cmfd_forcing",
        data_level="grid_level",
        deps=["load_basin_shp", "load_grid_shp"]
    )
    def load_cmfd_forcing(self):
        grid_shp = deepcopy(self.loaddata_kwargs["grid_shp"])
        grid_res = self.loaddata_kwargs["grid_res"]
        date_period = self.loaddata_kwargs["date_period"]
        search_method = self.loaddata_kwargs["search_method"]
        
        logger.info(
            f"Loading CMFD forcing data into grid with resolution {grid_res}... ..."
        )
                
        logger.info(f"search method for CMFD: {search_method} (grid_res:{grid_res}, source res: 0.1 degree)")
        
        grid_shp_with_cmfd_forcing = Extract_CMFD_forcing.ExtractData(
            grid_shp,
            grid_res,
            date_period,
            search_method,
            plot=False,
            check_search=False
        )
        
        logger.info("CMFD forcing data successfully loaded into grids")
        
        ret = {"cmfd_forcing": grid_shp_with_cmfd_forcing}
        
        return ret
        

class dataProcess_VIC_level2_CDMet_JRB(dataProcess_base):

    @processing_step(
        step_name="load_cdmet_forcing",
        save_names="cdmet_forcing",
        data_level="grid_level",
        deps=["load_basin_shp", "load_grid_shp"]
    )
    def load_cdmet_forcing(self):
        grid_shp = deepcopy(self.loaddata_kwargs["grid_shp"])
        grid_res = self.loaddata_kwargs["grid_res"]
        date_period = self.loaddata_kwargs["date_period"]
        search_method = self.loaddata_kwargs["search_method"]
                
        logger.info(f"search method for CDMet: {search_method} (grid_res:{grid_res}, source res: 4km (~=0.036 degree))")
        
        grid_shp_with_cdmet_forcing = Extract_CDMet_forcing.ExtractData(
            grid_shp,
            grid_res,
            date_period,
            search_method,
            plot=False,
            check_search=False
        )
        
        logger.info("CDMet forcing data successfully loaded into grids")
        
        ret = {"cdmet_forcing": grid_shp_with_cdmet_forcing}
        
        return ret
    

class dataProcess_VIC_level3_JRB(dataProcess_base):
    def load_grid_shp(self):
        # This method removes the registration in the parent class
        pass
    
    @processing_step(
        step_name="load_streamflow",
        save_names=["streamflow", "gauge_info"],
        data_level="basin_level",
        deps=["load_basin_shp"]
    )
    def load_streamflow(self):
        basin_shp = self.loaddata_kwargs["basin_shp"]
        date_period = self.loaddata_kwargs["date_period"]
        station_name = self.loaddata_kwargs["station_name"]
        
        logger.info(f"Loading streamflow data for basin {station_name} with dates: {date_period}... ...")
        stationdata_fname = stationdata_fname_map[station_name]
        
        basin_shp_with_streamflow = Extract_stationdata_streamflow.ExtractData(
            basin_shp,
            date_period,
            stationdata_fname,
            plot=False,
        )
        
        # read gauge information
        station_id = basin_outlets_reference_i_map[station_name]
        snaped_gauge = gpd.read_file(os.path.join(
            evb_dir_hydroanalysis.Hydroanalysis_dir,
            "wbw_working_directory_level0",
            f"snaped_outlet_with_reference_{station_id}.shp"
        ))
        
        gauge_coord = [snaped_gauge.geometry.x.values[0], snaped_gauge.geometry.y.values[0]]  # lon, lat
        gauge_info = {
            "station_name": station_name,
            "station_id": station_id,
            "gauge_coord(lon, lat)": gauge_coord
        }
        
        logger.info("Streamflow data successfully loaded into basins")
        
        ret = {"streamflow": basin_shp_with_streamflow, "gauge_info": gauge_info}
        
        return ret


def build_basin_shp(evb_dir_hydroanalysis):
    basin_shp_path = os.path.join(evb_dir_hydroanalysis.Hydroanalysis_dir, f"wbw_working_directory_level0\\basin_vector_outlet_with_reference_{basin_outlets_reference_i_map[station_name]}.shp")
    basin_shp = Basins.from_shapefile(basin_shp_path)
    return basin_shp


def build_dpc_VIC_JRB(evb_dir_hydroanalysis, station_name, model_scale, date_period, reverse_lat=True):
    # build evb
    evb_dir_modeling = build_modeling_dir(subname=f"{station_name}_{model_scale}")
    
    # read shpfile and get basin_shp (Basins)
    basin_shp = build_basin_shp(evb_dir_hydroanalysis)
    
    # build grid_shp
    grid_shp_level0, grid_shp_level1, grid_shp_level2, grid_shp_level3 = build_grid_shp(
        basin_shp,
        grid_res_level0,
        grid_res_level1,
        grid_res_level2,
        expand_grids_num=1,
        plot=True
    )
    
    # build dpc level0
    build_dpc_VIC_level0 = False
    if build_dpc_VIC_level0:
        dpc_VIC_level0 = dataProcess_VIC_level0_JRB(
            load_path=evb_dir_modeling._dpc_VIC_level0_path,
            reset_on_load_failure=True,
        )
        
        dpc_VIC_level0.loaddata_pipeline(
            save_path=evb_dir_modeling._dpc_VIC_level0_path,
            loaddata_kwargs={
                "basin_shp": basin_shp,
                "grid_shp": grid_shp_level0,
                "grid_res": grid_res_level0,
            }
        )
        
        dpc_VIC_level0.plot()
        dpc_VIC_level0.save_state(evb_dir_modeling._dpc_VIC_level0_path)
    
    # build dpc level2
    build_dpc_VIC_level2 = False
    if build_dpc_VIC_level2:
        dpc_VIC_level2_CMADSV1 = dataProcess_VIC_level2_CMADSV1_JRB(
            load_path=evb_dir_modeling._dpc_VIC_level2_path.replace(".pkl", "_CMADSV1.pkl"),
            reset_on_load_failure=True,
        )
        
        dpc_VIC_level2_CMADSV1.loaddata_pipeline(
            save_path=evb_dir_modeling._dpc_VIC_level2_path.replace(".pkl", "_CMADSV1.pkl"),
            loaddata_kwargs={
                "basin_shp": basin_shp,
                "grid_shp": grid_shp_level2,
                "grid_res": grid_res_level2,
                "date_period": date_period,
                "reverse_lat": reverse_lat,
                "search_method": "nearest", # src: 1/3 deg ~= 37km, radius_rectangle_reverse
            }
        )
        
        dpc_VIC_level2_CMADSV1.save_state(evb_dir_modeling._dpc_VIC_level2_path.replace(".pkl", "_CMADSV1.pkl"))
        
        dpc_VIC_level2_CMFD = dataProcess_VIC_level2_CMFD_JRB(
            load_path=evb_dir_modeling._dpc_VIC_level2_path.replace(".pkl", "_CMFD.pkl"),
            reset_on_load_failure=True,
        )
        
        dpc_VIC_level2_CMFD.loaddata_pipeline(
            save_path=evb_dir_modeling._dpc_VIC_level2_path.replace(".pkl", "_CMFD.pkl"),
            loaddata_kwargs={
                "basin_shp": basin_shp,
                "grid_shp": grid_shp_level2,
                "grid_res": grid_res_level2,
                "date_period": date_period,
                "search_method": "radius_rectangle", # src: 0.1 deg ~= 11km, nearest
            }
        )
        
        dpc_VIC_level2_CMFD.save_state(evb_dir_modeling._dpc_VIC_level2_path.replace(".pkl", "_CMFD.pkl"))
        
        dpc_VIC_level2_CDMet = dataProcess_VIC_level2_CDMet_JRB(
            load_path=evb_dir_modeling._dpc_VIC_level2_path.replace(".pkl", "_CDMet.pkl"),
            reset_on_load_failure=True,
        )
        
        dpc_VIC_level2_CDMet.loaddata_pipeline(
            save_path=evb_dir_modeling._dpc_VIC_level2_path.replace(".pkl", "_CDMet.pkl"),
            loaddata_kwargs={
                "basin_shp": basin_shp,
                "grid_shp": grid_shp_level2,
                "grid_res": grid_res_level2,
                "date_period": date_period,
                "search_method": "radius_rectangle", # src: 4km ~= 0.036 deg
            }
        )
        
        dpc_VIC_level2_CDMet.save_state(evb_dir_modeling._dpc_VIC_level2_path.replace(".pkl", "_CDMet.pkl"))
    
        # TODO dpc_VIC_level2_Insitu_CMA
        
    # build dpc level1
    build_dpc_VIC_level1 = False
    if build_dpc_VIC_level1:
        dpc_VIC_level1 = dataProcess_VIC_level1_JRB(
            load_path=evb_dir_modeling._dpc_VIC_level1_path,
            reset_on_load_failure=True,
        )
        
        dpc_VIC_level1.loaddata_pipeline(
            save_path=evb_dir_modeling._dpc_VIC_level1_path,
            loaddata_kwargs={
                "basin_shp": basin_shp,
                "grid_shp": grid_shp_level1,
                "grid_res": grid_res_level1,
                "date_period": date_period,
                "evb_dir": evb_dir_modeling,
                "reverse_lat": reverse_lat,
                "search_method_st": "radius_rectangle", # src: 0.1 deg ~= 11km, nearest
            }
        )
        
        dpc_VIC_level1.plot()
        dpc_VIC_level1.save_state(evb_dir_modeling._dpc_VIC_level1_path)
    
    # build dpc level3
    build_dpc_VIC_level3 = False
    if build_dpc_VIC_level3:
        dpc_VIC_level3 = dataProcess_VIC_level3_JRB(
            load_path=evb_dir_modeling._dpc_VIC_level3_path,
            reset_on_load_failure=True,
        )

        dpc_VIC_level3.loaddata_pipeline(
            save_path=evb_dir_modeling._dpc_VIC_level3_path,
            loaddata_kwargs={
                "basin_shp": basin_shp,
                "date_period": date_period,
                "station_name": station_name,
            }
        )

        dpc_VIC_level3.save_state(evb_dir_modeling._dpc_VIC_level3_path)
        
    # build dpc GLEAM
    build_dpc_VIC_GLEAM = True
    if build_dpc_VIC_GLEAM:
        dpc_VIC_GLEAM = dataProcess_VIC_level1_GLEAM_JRB(
            load_path=os.path.join(evb_dir_modeling.dpcFile_dir, "dpc_VIC_level1_GLEAM.pkl"),
            reset_on_load_failure=True,
        )
        
        dpc_VIC_GLEAM.loaddata_pipeline(
            save_path=os.path.join(evb_dir_modeling.dpcFile_dir, "dpc_VIC_level1_GLEAM.pkl"),
            loaddata_kwargs={
                "basin_shp": basin_shp,
                "grid_shp": grid_shp_level1,
                "grid_res": grid_res_level1,
                "date_period": date_period,
                "search_method": "radius_rectangle_reverse", # src: 0.25 deg ~= 28km, radius_rectangle_reverse
            }
        )
        

if __name__ == "__main__":
    # build hydroanalysis evb_dir for read basin_shp
    evb_dir_hydroanalysis = build_modeling_dir(subname="hydroanalysis")
    
    # build dpc
    build_dpc_VIC_JRB(evb_dir_hydroanalysis, station_name, model_scale, date_period)
    