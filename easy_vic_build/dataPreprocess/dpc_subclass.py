# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
from typing import Any
from .dpc_base import dataProcess_base
from .readdataIntoGrids_interface import *
from .readdataIntoBasin_interface import *
import matplotlib.pyplot as plt
from  .basin_grid import *
from ..tools.utilities import *

class dataProcess_VIC_level0(dataProcess_base):
    
    def __init__(self, basin_shp, grid_shp, grid_res, date_period, **kwargs):
        self.date_period = date_period
        super().__init__(basin_shp, grid_shp, grid_res, **kwargs)
    
    def __call__(self, *args: Any, readBasindata=False, readGriddata=True, readBasinAttribute=False, **kwargs: Any):
        self.read_basin_grid()
        
        if readBasindata:
            self.readDataIntoBasins()
        
        if readGriddata:
            self.readDataIntoGrids()
        
        if readBasinAttribute:
            self.readBasinAttribute()
    
    def read_basin_grid(self):
        self.createBoundaryShp()
        
    def createBoundaryShp(self):
        self.boundary_point_center_shp, self.boundary_point_center_x_y, self.boundary_grids_edge_shp, self.boundary_grids_edge_x_y = self.grid_shp.createBoundaryShp()
        
    def readDataIntoGrids(self):
        self.grid_shp = readSrtmDEMIntoGrids(self.grid_shp, grid_shp_res=self._grid_res, plot=False, save_original=False, check_search=False)
        self.grid_shp = readCONUSSoilIntoGrids(self.grid_shp, grid_shp_res=self._grid_res, plot_layer=False, save_original=False, check_search=False)
    
    def plot(self, fig=None, ax=None, grid_shp_kwargs=dict(), grid_shp_point_kwargs=dict(), basin_shp_kwargs=dict()):
        # plot grid_shp and basin_shp
        if fig is None:
            fig, ax = plt.subplots()
        
        # plot kwargs
        grid_shp_kwargs_all = {"edgecolor": "k", "alpha": 0.5, "linewidth": 0.5}
        grid_shp_kwargs_all.update(grid_shp_kwargs)
        
        grid_shp_point_kwargs_all = {"alpha": 0.5, "facecolor": "k", "markersize": 1}
        grid_shp_point_kwargs_all.update(grid_shp_point_kwargs)
        
        basin_shp_kwargs_all = {"edgecolor": "k", "alpha": 0.5, "facecolor": "b"}
        basin_shp_kwargs_all.update(basin_shp_kwargs)
        
        # plot
        self.grid_shp.boundary.plot(ax=ax, **grid_shp_kwargs_all)
        self.grid_shp["point_geometry"].plot(ax=ax, **grid_shp_point_kwargs_all)
        self.basin_shp.plot(ax=ax, **basin_shp_kwargs_all)
        
        boundary_x_y = self.boundary_grids_edge_x_y
        ax.set_xlim(boundary_x_y[0], boundary_x_y[1])
        ax.set_ylim(boundary_x_y[2], boundary_x_y[3])
        
        return fig, ax
    
    def plot_grid(self, column, fig=None, ax=None, grid_shp_kwargs=dict(), column_kwargs=dict(), basin_shp_kwargs=dict()):
        # plot grid_shp column
        if fig is None:
            fig, ax = plt.subplots()
        
        # plot kwargs
        grid_shp_kwargs_all = {"edgecolor": "k", "alpha": 0.5, "linewidth": 0.5}
        grid_shp_kwargs_all.update(grid_shp_kwargs)
        
        column_kwargs_all = {"cmap": "terrain", "legend": True}
        column_kwargs_all.update(column_kwargs)
        
        basin_shp_kwargs_all = {"edgecolor": "k", "alpha": 0.5, "facecolor": "b"}
        basin_shp_kwargs_all.update(basin_shp_kwargs)
        
        # plot
        self.grid_shp.boundary.plot(ax=ax, **grid_shp_kwargs_all)
        self.grid_shp.plot(column=column, ax=ax, **column_kwargs_all)
        self.basin_shp.plot(ax=ax, **basin_shp_kwargs_all)
        
        boundary_x_y = self.boundary_grids_edge_x_y
        ax.set_xlim(boundary_x_y[0], boundary_x_y[1])
        ax.set_ylim(boundary_x_y[2], boundary_x_y[3])
        
        return fig, ax
        
        

class dataProcess_VIC_level1(dataProcess_VIC_level0):
        
    def readDataIntoBasins(self):
        self.basin_shp = readCAMELSStreamflowIntoBasins(self.basin_shp, read_dates=self.date_period)
    
    def readBasinAttribute(self):
        self.basin_shp = readCAMELSAttributeIntoBasins(self.basin_shp, k_list=None)
        
    def readDataIntoGrids(self):
        self.grid_shp = readERA5_SoilTemperatureIntoGrids(self.grid_shp, grid_shp_res=self._grid_res, plot_layer=False, check_search=False)
        self.grid_shp = readNLDAS_annual_PIntoGrids(self.grid_shp, grid_shp_res=self._grid_res, plot=False, check_search=False)
        self.grid_shp = readUMDLandCoverIntoGrids(self.grid_shp, grid_shp_res=self._grid_res, plot=False, save_original=True, check_search=False)
        self.grid_shp = readMODISBSAIntoGrids(self.grid_shp, grid_shp_res=self._grid_res, plot_month=False, save_original=True, check_search=False)
        self.grid_shp = readMODISNDVIIntoGrids(self.grid_shp, grid_shp_res=self._grid_res, plot_month=False, save_original=True, check_search=False)
        self.grid_shp = readMODISLAIIntoGrids(self.grid_shp, grid_shp_res=self._grid_res, plot_month=False, save_original=True, check_search=False)       


class dataProcess_VIC_level2(dataProcess_VIC_level0):
    
    def readDataIntoBasins(self):
        pass
    
    def readBasinAttribute(self):
        pass
        
    def readDataIntoGrids(self):
        pass


class dataProcess_CAMELS_review(dataProcess_base):
    """_summary_: just read basin and grid of CAMELS for review

    Args:
        dataProcess_CAMELS (_type_): _description_
    """
    def __init__(self, HCDN_home, basin_shp=None, grid_shp=None, grid_res=None, **kwargs):
        super().__init__(basin_shp, grid_shp, grid_res, **kwargs)
        self._HCDN_home = HCDN_home
        self.read_basin_grid()
        
    def read_basin_grid(self):
        # read basin shp
        self.basin_shp = HCDNBasins(self._HCDN_home)
        # self.basin_shp_original = HCDNBasins(self._HCDN_home)  # backup for HCDN_shp
        
        # read grids and createBoundaryShp
        self.grid_shp = HCDNGrids(self._HCDN_home)
        self.createBoundaryShp()
        
    def createBoundaryShp(self):
        self.boundary_point_center_shp, self.boundary_point_center_x_y, self.boundary_grids_edge_shp, self.boundary_grids_edge_x_y = self.grid_shp.createBoundaryShp()
    
    def __call__(self, plot=True):
        print("=============== basin_shp ===============")
        print(self.basin_shp)
        
        print("=============== grid_shp ===============")
        print(self.grid_shp)
        
        if plot:
            self.plot()

    def plot(self):
        fig, ax = plotBackground(self.basin_shp, self.grid_shp, fig=None, ax=None)
        ax = setBoundary(ax, *self.boundary_grids_edge_x_y)
        
        return fig, ax
    

# class dataProcess_CAMELS:
#     """ main/base class: dataProcess_CAMELS

#     function structure:
#         self.__init__(): set path
#         self.bool_set(): a set of bool_params, a switch controlling enablement of the functions in the class
#         self.__call__(): workflow of this class
#             self.read_from_exist_file(): read variable from exist filem see save()
#             self.read_basin_grid(): read basin shp and grid shp, create boundary_shp and boudary_x_y
#             self.readDataIntoBasins(): read data into basins
#             self.selectBasins(): select basin from original basin, all based on self.basin_shp
#             self.intersectsGridsWithBasins(): intersect grids with basins to get intersects_grids
#             self.readDataIntoGrids(): read data into grids, normally, read into intersects_grids
#             self.combineIntersectsGridsWithBasins(): combine intersects_grids with basins to save the intersects_grids with data into basin_shp
#             self.aggregate_grid_to_basins(): aggregate basin_shp["intersects_grids"][...] into basin_shp[...]
#             self.save(): save data into files

#     variable structure:
#         private params: _params, params controlling initial settings
#         main variable: 
#             self.basin_shp: GeoDataframe, index is the basins id, columns contains "intersects_grids" "AREA" "AREA_km2" "lon_cen" "lat_cen"
#             self.basin_shp_original: GeoDataframe, the backup of basin_shp (the basin_shp is modified in the process)
#             self.grid_shp: GeoDataframe, rectangular grid, index is the grids id
#             self.intersects_grids: GeoDataframe, same as grid_shp but intersected with basin_shp, thus is can be treated as grid_shp (e.g., you can read data into it)
#             self.boundary_shp: GeoDataframe, rectangular shp corresponding the boundary, based on grid_shp
#             self.boundary_x_y: list, [boundary_x_min, boundary_x_max, boundary_y_min, boundary_y_max]
#         other:
#             self.fns_streamflow: list of str, file name corresponding to the usgs_streamflow
#             self.streamflow_id: list of int, id corresponding to the usgs_streamflow
#             self.fpaths_streamflow: list of str, file path corresponding to the usgs_streamflow 
#             self.usgs_streamflow: list of pd.Dataframe, usgs streamflow data
#             self.remove_files_Missing: list of set, attributes corresponding to the removed files
#                 set: {"fn": fn, "fpath": fpath, "usgs_streamflow": usgs_streamflow_, "reason": reason}

#     save file structure:
#         same as variable structure, all files are saved in .pkl format
#             basin_shp.pkl
#             basin_shp_original.pkl
#             grid_shp.pkl
#             intersects_grids.pkl
#             boundary_shp.pkl
#             boundary_x_y.pkl
#             remove_files_Missing.pkl

#     how to use:
#         # general use
#         subclass = class()  # specific your design, use subclass overriding any part you want
#         instance = subclass()
#         instance.bool_set()  # a switch controling enablement of the functions in the class, you can remove it and set instance function specifically
#         instance()

#         # append additional data into existing files
#         (1) read_from_exist_file
#         (2) use the instance.basin_shp/grid_shp and instance.function to read data again
#         (3) self.save() again

#     """

#     def __init__(self, home=None, subdir=None, date_period=None) -> None:
#         self._home = home
#         self._subdir = subdir
#         self._date_period = date_period

#     def __call__(self, read_from_exist_file_bool=False, *args: Any, **kwds: Any) -> Any:
#         if read_from_exist_file_bool:
#             self.read_from_exist_file()
#         else:
#             self.read_basin_grid()
#             self.readDataIntoBasins()
#             self.selectBasins()
#             self.intersectsGridsWithBasins()
#             self.readDataIntoGrids()
#             self.combineIntersectsGridsWithBasins()
#             self.aggregate_grid_to_basins()
#             self.save()

#         if self._readBasinAttribute_bool:
#             self.BasinAttribute = readBasinAttribute(self._home)  # 671

#         if self._plot_bool:
#             self.plot()

#     def read_from_exist_file(self):
#         # read data from file
#         self.basin_shp = pd.read_pickle(os.path.join(
#             self._home, "dataPreprocess_CAMELS", self._subdir, "basin_shp.pkl"))
#         self.basin_shp_original = pd.read_pickle(os.path.join(
#             self._home, "dataPreprocess_CAMELS", self._subdir, "basin_shp_original.pkl"))
#         self.grid_shp = pd.read_pickle(os.path.join(self._home, "dataPreprocess_CAMELS", self._subdir, "grid_shp.pkl"))
#         self.intersects_grids = pd.read_pickle(os.path.join(
#             self._home, "dataPreprocess_CAMELS", self._subdir, "intersects_grids.pkl"))
#         self.boundary_shp = pd.read_pickle(os.path.join(
#             self._home, "dataPreprocess_CAMELS", self._subdir, "boundary_shp.pkl"))
#         with open(os.path.join(self._home, "dataPreprocess_CAMELS", self._subdir, "boundary_x_y.pkl"), "rb") as f:
#             self.boundary_x_y = pickle.load(f)
#         if self._removeStreamflowMissing_bool:
#             with open(os.path.join(self._home, "dataPreprocess_CAMELS", self._subdir, "remove_files_Missing.pkl"), "rb") as f:
#                 self.remove_files_Missing = pickle.load(f)

#     def read_basin_grid(self):
#         # read basin shp
#         self.basin_shp = HCDNBasins(self._home)
#         self.basin_shp_original = HCDNBasins(self._home)  # backup for HCDN_shp

#         # read grids and createBoundaryShp
#         self.grid_shp = HCDNGrids(self._home)
#         # boundary_x_y = [boundary_x_min, boundary_x_max, boundary_y_min, boundary_y_max]
#         boundary_point_center_shp, boundary_point_center_x_y, boundary_grids_edge_shp, boundary_grids_edge_x_y = self.grid_shp.createBoundaryShp()

#     def readDataIntoBasins(self):
#         # read streamflow into basins
#         self.readStreamflowIntoBasins()
#         self.readForcingDaymetIntoBasins()

#     def readStreamflowIntoBasins(self):
#         self.fns_streamflow, self.fpaths_streamflow, self.usgs_streamflow, self.streamflow_id = readStreamflow(
#             self._home)  # 674

#         # read streamflow into basins
#         self.basin_shp = readStreamflowIntoBasins(self.basin_shp, self.streamflow_id, self.usgs_streamflow, self._date_period)

#     def readForcingDaymetIntoBasins(self):
#         fns_forcingDaymet, fpaths_forcingDaymet, forcingDaymet, forcingDaymetGaugeAttributes = readForcingDaymet(
#             self._home)  # 677

#         # read forcingDaymet (multi-variables) into basins
#         read_dates = pd.date_range(self._date_period[0], self._date_period[1], freq="D")
#         read_keys = ["prcp(mm/day)"]  # "prcp(mm/day)" "srad(W/m2)" "dayl(s)" "swe(mm)" "tmax(C)" "tmin(C)" "vp(Pa)"
#         self.basin_shp = readForcingDaymetIntoBasins(
#             forcingDaymet, forcingDaymetGaugeAttributes, self.basin_shp, read_dates, read_keys)

#     def selectBasins(self):
#         self.removeStreamflowMissing()
#         self.basin_shp = selectBasinBasedOnStreamflowWithZero(
#             self.basin_shp, self.usgs_streamflow, self.streamflow_id, zeros_min_num=100)  # 552 -> 103

#     def removeStreamflowMissing(self):
#         # remove streamflow when Missing
#         self.fns_streamflow, self.fpaths_streamflow, self.usgs_streamflow, self.streamflow_id, self.remove_files_Missing = removeStreamflowMissing(
#             self.fns_streamflow, self.fpaths_streamflow, self.usgs_streamflow, date_period=self._date_period)  # 674 - 122 = 552

#         # remove basins with streamflowMissing
#         self.basin_shp = removeBasinBasedOnStreamflowMissing(self.basin_shp, self.streamflow_id)  # 671 - 122 = 552

#     def intersectsGridsWithBasins(self):
#         self.basin_shp, self.intersects_grids = intersectGridsWithBasins(self.grid_shp, self.basin_shp)

#     def readDataIntoGrids(self):
#         self.intersects_grids = readSrtmDEMIntoGrids(self.intersects_grids, self._plot_bool)
#         self.intersects_grids = readUMDLandCoverIntoGrids(self.intersects_grids)
#         self.intersects_grids = readHWSDSoilDataIntoGirds(self.intersects_grids, self.boundary_shp, self._plot_bool)
#         self.intersects_grids = readGLEAMEDailyIntoGrids(
#             self.intersects_grids, period=list(range(1980, 2011)), var_name="E")

#     def combineIntersectsGridsWithBasins(self):
#         self.basin_shp, self.intersects_grids = intersectGridsWithBasins(self.intersects_grids, self.basin_shp)

#     def aggregate_grid_to_basins(self):
#         self.basin_shp = aggregate_GLEAMEDaily(self.basin_shp)

#     def save(self):
#         self.basin_shp.to_pickle(os.path.join(self._home, "dataPreprocess_CAMELS", self._subdir, "basin_shp.pkl"))
#         self.basin_shp_original.to_pickle(os.path.join(
#             self._home, "dataPreprocess_CAMELS", self._subdir, "basin_shp_original.pkl"))
#         self.grid_shp.to_pickle(os.path.join(self._home, "dataPreprocess_CAMELS", self._subdir, "grid_shp.pkl"))
#         self.intersects_grids.to_pickle(os.path.join(
#             self._home, "dataPreprocess_CAMELS", self._subdir, "intersects_grids.pkl"))
#         self.boundary_shp.to_pickle(os.path.join(self._home, "dataPreprocess_CAMELS", self._subdir, "boundary_shp.pkl"))

#         with open(os.path.join(self._home, "dataPreprocess_CAMELS", self._subdir, "boundary_x_y.pkl"), "wb") as f:
#             pickle.dump(self.boundary_x_y, f)

#         if self._removeStreamflowMissing_bool:
#             with open(os.path.join(self._home, "dataPreprocess_CAMELS", self._subdir, "remove_files_Missing.pkl"), "wb") as f:
#                 pickle.dump(self.remove_files_Missing, f)

#     def plot(self):
#         fig, ax = plotBackground(self.basin_shp_original, self.grid_shp, fig=None, ax=None)
#         plot_kwgs1 = {"facecolor": "none", "alpha": 0.2, "edgecolor": "r"}
#         plot_kwgs2 = {"facecolor": "none", "alpha": 0.5, "edgecolor": "r", "markersize": 0.5}
#         fig, ax = plotGrids(self.intersects_grids, None, fig, ax, plot_kwgs1, plot_kwgs2)
#         ax = setBoundary(ax, *self.boundary_x_y)

#         if self._readUMDLandCoverIntoGrids_bool:
#             plotLandCover(self.basin_shp_original, self.basin_shp, self.grid_shp,
#                           self.intersects_grids, *self.boundary_x_y)
#         if self._readHWSDSoilDataIntoGirds_bool:
#             plotHWSDSoilData(self.basin_shp_original, self.basin_shp, self.grid_shp,
#                              self.intersects_grids, *self.boundary_x_y)
#         if self._readSrtmDEMIntoGrids_bool:
#             plotStrmDEM(self.basin_shp_original, self.basin_shp, self.grid_shp,
#                         self.intersects_grids, *self.boundary_x_y)
#         plt.show()


# class dataProcess_CAMELS_read_basin_grid_removeStreamflowMissing(dataProcess_CAMELS):
#     """_summary_: read basin and grid of CAMELS, and remove streamflow missing
#     baseline for dataProcess_CAMELS: 
#         read basin grid: create basin and grid shp
#         select basins: removeStreamflowMissing(), 671 -> 652
#         read data into basins: readStreamflowIntoBasins(), readBasinAttributeIntoBasins()
#         #* read data into grids: no read grids data, leave it for further customization
    
#     remove basins:
#     01118300
#     01121000
#     01187300
#     01510000
#     02125000
#     02202600
#     03281100
#     03300400
#     05062500
#     06154410
#     06291500
#     07290650
#     07295000
#     08079600
#     09497800
#     10173450
#     12383500
#     12388400
    
#     Args:
#         dataProcess_CAMELS (_type_): _description_
#     """
#     def __init__(self, home, subdir, date_period) -> None:
#         self._date_period = date_period
#         super().__init__(home, subdir)

#     def __call__(self, plot=True) -> Any:
#         self.read_basin_grid()
#         self.readDataIntoBasins()
#         self.selectBasins()
#         self.intersectsGridsWithBasins()
#         self.combineIntersectsGridsWithBasins()

#         if plot:
#             self.plot()
        
#     def readDataIntoBasins(self):
#         self.readStreamflowIntoBasins()
#         self.readBasinAttributeIntoBasins()
#         # self.readForcingDaymetIntoBasins()
    
#     def readStreamflowIntoBasins(self):
#         self.fns_streamflow, self.fpaths_streamflow, self.usgs_streamflow, self.streamflow_id = readStreamflow(
#             self._home)  # 674

#         # read streamflow into basins
#         self.basin_shp = readStreamflowIntoBasins(self.basin_shp, self.streamflow_id, self.usgs_streamflow, self._date_period)
    
#     def readForcingDaymetIntoBasins(self):
#         fns_forcingDaymet, fpaths_forcingDaymet, forcingDaymet, forcingDaymetGaugeAttributes = readForcingDaymet(
#             self._home)  # 677
        
#         # read forcingDaymet (multi-variables) into basins
#         read_dates = pd.date_range(self._date_period[0], self._date_period[1], freq="D")
#         read_keys = ["prcp(mm/day)", "swe(mm)", "tmax(C)", "tmin(C)"]  # "prcp(mm/day)" "srad(W/m2)" "dayl(s)" "swe(mm)" "tmax(C)" "tmin(C)" "vp(Pa)"
#         self.basin_shp = readForcingDaymetIntoBasins(
#             forcingDaymet, forcingDaymetGaugeAttributes, self.basin_shp, read_dates, read_keys)
        
#     def readBasinAttributeIntoBasins(self):
#         BasinAttributes = readBasinAttribute(self._home)
#         for key in BasinAttributes.keys():
#             self.basin_shp = readBasinAttributeIntoBasins(BasinAttributes[key], self.basin_shp, prefix=key+":")
    
#     def selectBasins(self):
#         self.removeStreamflowMissing()  # 671 -> 652
    
#     def removeStreamflowMissing(self):
#         # remove streamflow when Missing
#         self.fns_streamflow, self.fpaths_streamflow, self.usgs_streamflow, self.streamflow_id, self.remove_files_Missing = removeStreamflowMissing(
#             self.fns_streamflow, self.fpaths_streamflow, self.usgs_streamflow, date_period=self._date_period)  # 671 -> 652

#         # remove basins with streamflowMissing
#         self.basin_shp = removeBasinBasedOnStreamflowMissing(self.basin_shp, self.streamflow_id)

#     def plot(self):
#         fig, ax = plotBackground(self.basin_shp_original, self.grid_shp, fig=None, ax=None)
#         plot_kwgs1 = {"facecolor": "none", "alpha": 0.2, "edgecolor": "r"}
#         plot_kwgs2 = {"facecolor": "none", "alpha": 0.5, "edgecolor": "r", "markersize": 0.5}
#         fig, ax = plotGrids(self.intersects_grids, None, fig, ax, plot_kwgs1, plot_kwgs2)
#         ax = setBoundary(ax, *self.boundary_x_y)
#         plt.show()


# class dataProcess_CAMELS_WaterBalanceAnalysis(dataProcess_CAMELS):
#     """_summary_: WaterBalanceAnalysis detS = P - E -R, WaterClosureResidual = P - E - R - detS
#     read grids data for water balance analysis: based on baseline data from dataProcess_CAMELS_read_basin_grid_removeStreamflowMissing
#     read grids data into basins and aggregate it
    
#     Args:
#         dataProcess_CAMELS (_type_): _description_
#     """
#     def __init__(self, home, subdir, date_period) -> None:
#         self._date_period = date_period
#         super().__init__(home, subdir)

#     def __call__(self, dpc_base=None, iindex_basin_shp=0) -> Any:
#         if dpc_base:
#             self.basin_shp = dpc_base.basin_shp.iloc[iindex_basin_shp: iindex_basin_shp + 1, :]
#             self.basin_shp_original = dpc_base.basin_shp_original
#             self.grid_shp = dpc_base.grid_shp
#             self.boundary_shp = dpc_base.boundary_shp
#             self.boundary_x_y = dpc_base.boundary_x_y
#             self.intersects_grids = dpc_base.intersects_grids
#         else:
#             self.read_basin_grid()
            
#         self.intersectsGridsWithBasins()
#         self.readDataIntoGrids()
#         self.combineIntersectsGridsWithBasins()
#         self.aggregate_grid_to_basins()

#     def readDataIntoGrids(self):
#         # read TRMM_P
#         self.intersects_grids = readTRMMPIntoGrids(
#             self.intersects_grids, period=self._date_period,
#             var_name="precipitation")
        
#         # read GLEAME_Daily
#         self.intersects_grids = readGLEAMEDailyIntoGrids(
#             self.intersects_grids, period=self._date_period, var_name="E")
        
#         # read GLEAME_Daily_Ep
#         self.intersects_grids = readGLEAMEDailyIntoGrids(
#             self.intersects_grids, period=self._date_period, var_name="Ep")

#         # read ERA5_SM
#         self.intersects_grids = readERA5_SMIntoGrids(self.intersects_grids, period=self._date_period, var_name="1")
#         self.intersects_grids = readERA5_SMIntoGrids(self.intersects_grids, period=self._date_period, var_name="2")
#         self.intersects_grids = readERA5_SMIntoGrids(self.intersects_grids, period=self._date_period, var_name="3")
#         self.intersects_grids = readERA5_SMIntoGrids(self.intersects_grids, period=self._date_period, var_name="4")

#         # read GlobalSnow_SWE
#         self.intersects_grids = readGlobalSnow_SWEIntoGrids(self.intersects_grids, period=self._date_period, var_name="swe")
        
#         # read GLDAS_CanopInt
#         self.intersects_grids = readGLDAS_CanopIntIntoGrids(self.intersects_grids, period=self._date_period, var_name="CanopInt_tavg")

#     def aggregate_grid_to_basins(self):
#         # aggregate TRMM_P
#         self.basin_shp = aggregate_TRMM_P(self.basin_shp)
        
#         # aggregate GLEAME_Daily
#         self.basin_shp = aggregate_GLEAMEDaily(self.basin_shp)
#         self.basin_shp = aggregate_GLEAMEpDaily(self.basin_shp)
        
#         # aggregate ERA5_SM
#         self.basin_shp = aggregate_ERA5_SM(self.basin_shp, aggregate_column="swvl1")
#         self.basin_shp = aggregate_ERA5_SM(self.basin_shp, aggregate_column="swvl2")
#         self.basin_shp = aggregate_ERA5_SM(self.basin_shp, aggregate_column="swvl3")
#         self.basin_shp = aggregate_ERA5_SM(self.basin_shp, aggregate_column="swvl4")
        
#         # aggregate GlobalSnow_SWE
#         self.basin_shp = aggregate_GlobalSnow_SWE(self.basin_shp, aggregate_column="swe")
        
#         # aggregate GLDAS_CanopInt
#         self.basin_shp = aggregate_GLDAS_CanopInt(self.basin_shp, aggregate_column="CanopInt_tavg")


# class dataProcess_CAMELS_Malan_Basins_with_Zeros(dataProcess_CAMELS):
    
#     def __call__(self, read_from_exist_file_bool=False, *args: Any, **kwds: Any) -> Any:
#         self._removeStreamflowMissing_bool = True
#         if read_from_exist_file_bool:
#             self.read_from_exist_file()
#         else:
#             self.read_basin_grid()
#             self.readDataIntoBasins()
#             self.selectBasins()
#             self.intersectsGridsWithBasins()
#             self.readDataIntoGrids()
#             self.combineIntersectsGridsWithBasins()
#             self.aggregate_grid_to_basins()
#             self.save()

#     def readDataIntoBasins(self):
#         self.readStreamflowIntoBasins()
        
#     def readDataIntoGrids(self):
#         # read GLEAME_Daily
#         self.intersects_grids = readGLEAMEDailyIntoGrids(
#             self.intersects_grids, period=self._date_period, var_name="E")
        
#     def readStreamflowIntoBasins(self):
#         self.fns_streamflow, self.fpaths_streamflow, self.usgs_streamflow, self.streamflow_id = readStreamflow(
#             self._home)  # 674

#         # read streamflow into basins
#         self.basin_shp = readStreamflowIntoBasins(self.basin_shp, self.streamflow_id, self.usgs_streamflow, self._date_period)
    
#     def selectBasins(self):
#         self.removeStreamflowMissing()
#         self.basin_shp = selectBasinBasedOnStreamflowWithZero(
#             self.basin_shp, self.usgs_streamflow, self.streamflow_id, zeros_min_num=100)  # 552 -> 103
    
#     def aggregate_grid_to_basins(self):
#         # aggregate GLEAME_Daily
#         self.basin_shp = aggregate_GLEAMEDaily(self.basin_shp)
        



# # ------------------------ patch ------------------------
# class dataProcess_CAMELS_WaterBalanceAnalysis_patch_Ep(dataProcess_CAMELS):
#     """_summary_: WaterBalanceAnalysis detS = P - E -R, WaterClosureResidual = P - E - R - detS

#     Args:
#         dataProcess_CAMELS (_type_): _description_
#     """
#     def __init__(self, home, subdir, date_period) -> None:
#         self._date_period = date_period
#         super().__init__(home, subdir)

#     def __call__(self, dpc_base=None, iindex_basin_shp=0) -> Any:
#         if dpc_base:
#             self.basin_shp = dpc_base.basin_shp.iloc[iindex_basin_shp: iindex_basin_shp + 1, :]
#             self.basin_shp_original = dpc_base.basin_shp_original
#             self.grid_shp = dpc_base.grid_shp
#             self.boundary_shp = dpc_base.boundary_shp
#             self.boundary_x_y = dpc_base.boundary_x_y
#             self.intersects_grids = dpc_base.intersects_grids
#         else:
#             self.read_basin_grid()
            
#         self.intersectsGridsWithBasins()
#         self.readDataIntoGrids()
#         self.combineIntersectsGridsWithBasins()
#         self.aggregate_grid_to_basins()

#     def readDataIntoGrids(self):
#         # read GLEAME_Daily_Ep
#         self.intersects_grids = readGLEAMEDailyIntoGrids(
#             self.intersects_grids, period=self._date_period, var_name="Ep")

#     def aggregate_grid_to_basins(self):
#         # aggregate GLEAME_Daily
#         self.basin_shp = aggregate_GLEAMEpDaily(self.basin_shp)

# # ------------------------ demo ------------------------


# def reviewCAMELSBasinData(home):
#     subdir = "review_CAMELS"
#     date_period = ["19980101", "20101231"]
#     dpc = dataProcess_CAMELS_read_basin_grid(home, subdir, date_period)
#     dpc()
#     return dpc
    

# def demoMalan_Basins_with_Zeros(home):
#     subdir = "Malan_Basins_with_Zeros"
#     date_period = ["19800101", "20101231"]
#     dpc = dataProcess_CAMELS_Malan_Basins_with_Zeros(home, subdir, date_period)
#     dpc(read_from_exist_file_bool=False)

#     # export to csv
#     exportToCsv(dpc.basin_shp, fpath_dir="F:/work/malan/20230913CAMELSdata")


if __name__ == "__main__":
    # general set
    root, home = setHomePath(root="E:")

    # review
    dpc_review = dataProcess_CAMELS_review(HCDN_home=home)
    dpc_review()
    
