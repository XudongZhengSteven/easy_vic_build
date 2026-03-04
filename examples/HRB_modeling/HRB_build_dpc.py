# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
import geopandas as gpd

from easy_vic_build.tools.dpc_func.basin_grid_class import Basins, Grids
from easy_vic_build.tools.dpc_func.basin_grid_func import build_grid_shp, createGridForBasin, shift_grids
from easy_vic_build.tools.dpc_func.dpc_base import dataProcess_base
from easy_vic_build.tools.decoractors import processing_step
from easy_vic_build.tools.dpc_func.extractData_func import Extract_UMD_1km

from HRB_extractData_func import Extract_ASTGTM2DEM, Extract_SoilGrids1km, Extract_CMFD_forcing, Extract_stationdata_streamflow
from HRB_extractData_func import Extract_ERA5_SoilTemperature, Extract_Annual_P, Extract_MODIS_BSA, Extract_MODIS_NDVI, Extract_MODIS_LAI

from general_info import *
import matplotlib.pyplot as plt

from copy import deepcopy

class dataProcess_VIC_level0_HRB(dataProcess_base):
    
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


class dataProcess_VIC_level2_CMFD_HRB(dataProcess_base):
    
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
        
        
class dataProcess_VIC_level1_HRB(dataProcess_base):
    
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
            check_search=True,
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
            check_search=True,
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


class dataProcess_VIC_level3_HRB(dataProcess_base):
    def load_grid_shp(self):
        # This method removes the registration in the parent class
        pass

    @processing_step(
        step_name="load_basin_shp",
        save_names=["basin_shp", "basin_shps"],
        data_level="basin_level",
        deps=None,
    )
    def load_basin_shp(self):
        loaded_basin_shp = deepcopy(self.loaddata_kwargs["basin_shp"])
        loaded_basin_shps = deepcopy(self.loaddata_kwargs["basin_shps"])
        ret = {"basin_shp": loaded_basin_shp, "basin_shps": loaded_basin_shps}
        return ret
    
    @processing_step(
        step_name="load_streamflow",
        save_names=["streamflow"],
        data_level="basin_level",
        deps=["load_basin_shp"]
    )
    def load_streamflow(self):
        basin_shp = self.loaddata_kwargs["basin_shp"]
        date_period = self.loaddata_kwargs["date_period"]
        station_names = self.loaddata_kwargs["station_names"]
        
        logger.info(f"Loading streamflow data for basin with dates: {date_period}... ...")
        
        basin_shp_with_streamflow = Extract_stationdata_streamflow.ExtractData(
            basin_shp,
            station_names,
            date_period,
            plot=False,
        )
        
        logger.info("Streamflow data successfully loaded into basins")
        
        ret = {"streamflow": basin_shp_with_streamflow}
        
        return ret
    
    @processing_step(
        step_name="load_gauge_info",
        save_names=["gauge_info"],
        data_level="basin_level",
        deps=["load_basin_shp"]
    )
    def load_gauge_info(self):
        basin_shp = self.loaddata_kwargs["basin_shp"]
        date_period = self.loaddata_kwargs["date_period"]
        station_names = self.loaddata_kwargs["station_names"]
        load_level1 = self.loaddata_kwargs["load_level1"]
        
        logger.info(f"Loading gauge info for basin with dates: {date_period}... ...")
        
        gauge_info = {}
        
        for station_name in station_names:
            station_id = basin_outlets_reference_i_map[station_name]
            
            snaped_gauge_level0 = gpd.read_file(os.path.join(
                evb_dir_hydroanalysis.Hydroanalysis_dir,
                "wbw_working_directory_level0",
                f"snaped_outlet_with_reference_{station_id}.shp"
            ))  # level 0
            
            gauge_coord_level0 = [snaped_gauge_level0.geometry.x.values[0], snaped_gauge_level0.geometry.y.values[0]]  # lon, lat
            
            gauge_info[station_name] = {
                "station_name": station_name,
                "station_id": station_id,
                "gauge_coord(lon, lat)_level0": gauge_coord_level0
            }
                            
            if load_level1:
                snaped_gauge_level1 = gpd.read_file(os.path.join(
                    evb_dir_modeling.Hydroanalysis_dir,
                    "wbw_working_directory_level1",
                    f"snaped_outlet_with_reference_{station_id}.shp"
                ))  # level 1 needed for rvic
    
                gauge_coord_level1 = [snaped_gauge_level1.geometry.x.values[0], snaped_gauge_level1.geometry.y.values[0]]  # lon, lat
            
                gauge_info[station_name].update({
                    "gauge_coord(lon, lat)_level1": gauge_coord_level1
                })
        
        ret = {"gauge_info": gauge_info}
        return ret
    
    
def build_basin_shp_JRB(evb_dir_hydroanalysis, plot_bool=False):
    basin_shps = {}
    for station_name in station_names:
        basin_shp_path = os.path.join(evb_dir_hydroanalysis.Hydroanalysis_dir, f"wbw_working_directory_level0\\basin_vector_outlet_with_reference_{basin_outlets_reference_i_map[station_name]}.shp")
        basin_shp = Basins.from_shapefile(basin_shp_path)
        basin_shps[station_name] = basin_shp
    
    if plot_bool:
        fig, ax = plt.subplots()
        facecolors = dict(zip(station_names, ["g", "b", "r"]))
        for station_name in station_names:
            basin_shps[station_name].plot(ax=ax, facecolor=facecolors[station_name])
        plt.show(block=True)
    
    return basin_shps

def build_dpc_VIC_HRB(evb_dir_hydroanalysis, evb_dir_modeling, date_period, reverse_lat=True):
    # read shpfile and get basin_shp (Basins)
    basin_shps = build_basin_shp_JRB(evb_dir_hydroanalysis)
    
    # build grid_shp
    grid_shp_level0, grid_shp_level1, grid_shp_level2, grid_shp_level3 = build_grid_shp(
        basin_shps[station_name],
        grid_res_level0,
        grid_res_level1,
        grid_res_level2,
        expand_grids_num=1,
        plot=True,
    )
    
    # grid_shp save
    grid_shp_save = False
    if grid_shp_save:
        grid_shp_level0_grid = grid_shp_level0.loc[:, "geometry"]
        grid_shp_level0_center = grid_shp_level0.loc[:, "point_geometry"]
        grid_shp_level0_grid.to_file(os.path.join(evb_dir_modeling.dpcFile_dir, "grid_shp", "grid_shp_level0_grid.shp"), driver="ESRI Shapefile", encoding="utf-8")
        grid_shp_level0_center.to_file(os.path.join(evb_dir_modeling.dpcFile_dir, "grid_shp", "grid_shp_level0_center.shp"), driver="ESRI Shapefile", encoding="utf-8")
        
        grid_shp_level1_grid = grid_shp_level1.loc[:, "geometry"]
        grid_shp_level1_center = grid_shp_level1.loc[:, "point_geometry"]
        grid_shp_level1_grid.to_file(os.path.join(evb_dir_modeling.dpcFile_dir, "grid_shp", "grid_shp_level1_grid.shp"), driver="ESRI Shapefile", encoding="utf-8")
        grid_shp_level1_center.to_file(os.path.join(evb_dir_modeling.dpcFile_dir, "grid_shp", "grid_shp_level1_center.shp"), driver="ESRI Shapefile", encoding="utf-8")
    
    # build dpc base
    build_dpc_base = False
    if build_dpc_base:
        dpc_VIC_base = dataProcess_base(
            load_path=os.path.join(evb_dir_modeling.dpcFile_dir, "dpc_VIC_base.pkl"),
            reset_on_load_failure=True,
        )
        
        dpc_VIC_base.loaddata_pipeline(
            save_path=os.path.join(evb_dir_modeling.dpcFile_dir, "dpc_VIC_base.pkl"),
            loaddata_kwargs={
                "basin_shp": basin_shps[station_name],
                "grid_shp": grid_shp_level1,
                "grid_res": grid_res_level1,
            }
        )
        
        dpc_VIC_base.save_state(os.path.join(evb_dir_modeling.dpcFile_dir, "dpc_VIC_base.pkl"))
    
    # build dpc level0
    build_dpc_VIC_level0 = False
    if build_dpc_VIC_level0:
        dpc_VIC_level0 = dataProcess_VIC_level0_HRB(
            load_path=evb_dir_modeling._dpc_VIC_level0_path,
            reset_on_load_failure=True,
        )
        
        dpc_VIC_level0.loaddata_pipeline(
            save_path=evb_dir_modeling._dpc_VIC_level0_path,
            loaddata_kwargs={
                "basin_shp": basin_shps[station_name],
                "grid_shp": grid_shp_level0,
                "grid_res": grid_res_level0,
            }
        )
        
        dpc_VIC_level0.plot()
        dpc_VIC_level0.save_state(evb_dir_modeling._dpc_VIC_level0_path)
    
    # build level2
    build_dpc_VIC_level2_CMFD = False
    if build_dpc_VIC_level2_CMFD:
        dpc_VIC_level2_CMFD = dataProcess_VIC_level2_CMFD_HRB(
            load_path=evb_dir_modeling._dpc_VIC_level2_path.replace(".pkl", "_CMFD.pkl"),
            reset_on_load_failure=True,
        )
        
        dpc_VIC_level2_CMFD.loaddata_pipeline(
            save_path=evb_dir_modeling._dpc_VIC_level2_path.replace(".pkl", "_CMFD.pkl"),
            loaddata_kwargs={
                "basin_shp": basin_shps[station_name],
                "grid_shp": grid_shp_level2,
                "grid_res": grid_res_level2,
                "date_period": date_period,
                "search_method": "radius_rectangle_reverse", # src: 0.1 deg ~= 11km, nearest, "radius_rectangle"
            }
        )
        
        dpc_VIC_level2_CMFD.save_state(evb_dir_modeling._dpc_VIC_level2_path.replace(".pkl", "_CMFD.pkl"))
    
    # build dpc level1
    build_dpc_VIC_level1 = False
    if build_dpc_VIC_level1:
        dpc_VIC_level1 = dataProcess_VIC_level1_HRB(
            load_path=evb_dir_modeling._dpc_VIC_level1_path,
            reset_on_load_failure=True,
        )
        
        dpc_VIC_level1.loaddata_pipeline(
            save_path=evb_dir_modeling._dpc_VIC_level1_path,
            loaddata_kwargs={
                "basin_shp": basin_shps[station_name],
                "grid_shp": grid_shp_level1,
                "grid_res": grid_res_level1,
                "date_period": date_period,
                "evb_dir": evb_dir_modeling,
                "reverse_lat": reverse_lat,
                "search_method_st": "radius_rectangle_reverse", # src: 0.1 deg ~= 11km, nearest, radius_rectangle
            }
        )
        
        dpc_VIC_level1.plot()
        dpc_VIC_level1.save_state(evb_dir_modeling._dpc_VIC_level1_path)
    
    # build dpc level3
    build_dpc_VIC_level3 = False
    if build_dpc_VIC_level3:
        dpc_VIC_level3 = dataProcess_VIC_level3_HRB(
            load_path=evb_dir_modeling._dpc_VIC_level3_path,
            reset_on_load_failure=True,
        )

        dpc_VIC_level3.loaddata_pipeline(
            save_path=evb_dir_modeling._dpc_VIC_level3_path,
            loaddata_kwargs={
                "basin_shp": basin_shps[station_name],
                "basin_shps": basin_shps,
                "date_period": date_period,
                "station_names": station_names,
                "load_level1": False
            }
        )
        
        dpc_VIC_level3.save_state(evb_dir_modeling._dpc_VIC_level3_path)
    
    # build dpc level3 for loading level1 hydroanalysis results
    build_dpc_VIC_level3_load_level1 = True
    if build_dpc_VIC_level3_load_level1:
        dpc_VIC_level3 = dataProcess_VIC_level3_HRB(
            load_path=evb_dir_modeling._dpc_VIC_level3_path,
            reset_on_load_failure=True,
        )
        
        try:
            dpc_VIC_level3.clear_data_from_cache(
                save_names=["gauge_info"],
                step_name="load_gauge_info"
            )
        except:
            pass
        
        dpc_VIC_level3.loaddata_pipeline(
            save_path=evb_dir_modeling._dpc_VIC_level3_path,
            loaddata_kwargs={
                "basin_shp": basin_shps[station_name],
                "basin_shps": basin_shps,
                "date_period": date_period,
                "station_names": station_names,
                "load_level1": True
            }
        )
        
        dpc_VIC_level3.save_state(evb_dir_modeling._dpc_VIC_level3_path)
        

if __name__ == "__main__":
    # build dpc
    build_dpc_VIC_HRB(evb_dir_hydroanalysis, evb_dir_modeling, date_period)
    
