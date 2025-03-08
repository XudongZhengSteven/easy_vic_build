# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
from .extractData_func import *

# ------------------------ read data into grids ------------------------
class readDataIntoGrids_API:
    def readSrtmDEMIntoGrids(grid_shp, grid_shp_res=0.25, plot=False, save_original=False, check_search=False):
        grid_shp = Extract_SrtmDEM.ExtractData(grid_shp, grid_shp_res=grid_shp_res, plot=plot, save_original=save_original, check_search=check_search)
        return grid_shp

    def readCONUSSoilIntoGrids(grid_shp, grid_shp_res=0.125, plot_layer=1, save_original=True, check_search=False):
        grid_shp = Extract_CONUS_SOIL.ExtractData(grid_shp, grid_shp_res, plot_layer, save_original, check_search)
        return grid_shp

    def readERA5_SoilTemperatureIntoGrids(grid_shp, grid_shp_res=0.125, plot_layer=False, check_search=False):
        grid_shp = Extract_ERA5_SoilTemperature.ExtractData(grid_shp, grid_shp_res, plot_layer, check_search)
        return grid_shp

    def readNLDAS_annual_PIntoGrids(grid_shp, grid_shp_res=0.125, plot=False, check_search=False):
        grid_shp = Extract_NLDAS_annual_P.ExtractData(grid_shp, grid_shp_res, plot, check_search)
        return grid_shp

    def readUMDLandCoverIntoGrids(grid_shp, grid_shp_res=0.125, plot=True, save_original=False, check_search=False):
        grid_shp = Extract_UMD_1km.ExtractData(grid_shp, grid_shp_res, plot, save_original, check_search)
        return grid_shp

    def readMODISBSAIntoGrids(grid_shp, grid_shp_res=0.125, plot_month=False, save_original=True, check_search=False):
        grid_shp = Extract_MODIS_BSA.ExtractData(grid_shp, grid_shp_res, plot_month, save_original, check_search)
        return grid_shp

    def readMODISNDVIIntoGrids(grid_shp, grid_shp_res=0.125, plot_month=False, save_original=True, check_search=False):
        grid_shp = Extract_MODIS_NDVI.ExtractData(grid_shp, grid_shp_res, plot_month, save_original, check_search)
        return grid_shp

    def readMODISLAIIntoGrids(grid_shp, grid_shp_res=0.125, plot_month=False, save_original=True, check_search=False):
        grid_shp = Extract_MODIS_LAI.ExtractData(grid_shp, grid_shp_res, plot_month, save_original, check_search)
        return grid_shp