import os

__version__ = "1.0.0"
__author__ = "Xudong Zheng"
__email__ = "zhengxd@sehemodel.club"
__all__ = ["Evb_dir"]

from .tools.utilities import check_and_mkdir, remove_and_mkdir
from . import build_dpc, build_GlobalParam, build_hydroanalysis, build_RVIC_Param, bulid_Domain, bulid_Param, calibrate, warmup
from . import tools

try:
    import nco
    HAS_NCO = True
except ImportError:
    HAS_NCO = False
    
if HAS_NCO:
    from . import build_MeteForcing_nco as build_MeteForcing
    print("Using MeteForcing with nco")
else:
    from . import build_mete_forcing
    print("Using MeteForcing without nco")

class Evb_dir:
    # easy_vic_build dir
    __package_dir__ = "./easy_vic_build" # os.path.abspath(os.path.dirname(__file__))
    __data_dir__ = os.path.join(os.path.dirname(__package_dir__), "data")
    
    def __init__(self, cases_home=None):
        self._cases_dir = cases_home if cases_home is not None else os.path.join(Evb_dir.__package_dir__, "cases")
        self._MeteForcing_src_dir = ""
        self._MeteForcing_src_suffix = ".nc"
        self._forcing_prefix = "forcings"
        self._linux_share_temp_dir = ""
        self._arcpy_python_path = ""
        self._vic_exe_path = ""
        
        self._dpc_VIC_level0_path = ""
        self._dpc_VIC_level1_path = ""
        self._dpc_VIC_level2_path = ""
        self._dpc_VIC_plot_grid_basin_path = ""
        self._dpc_VIC_plot_columns_path = ""
        
        self._domainFile_path = ""
        
        self._veg_param_json_path = ""
        self._params_dataset_level0_path = ""
        self._params_dataset_level1_path = ""
        
        self._flow_direction_file_path = ""
        self._pourpoint_file_path = ""
        self._uhbox_file_path = ""
        self._rvic_param_cfg_file_path = ""
        self._rvic_param_cfg_file_reference_path = ""
        self._rvic_conv_cfg_file_path = ""
        self._rvic_conv_cfg_file_reference_path = ""
        
        self._globalParam_path = ""
        self._globalParam_reference_path = ""
        
        self._rout_param_dir = ""
        
        self._calibrate_cp_path = ""
         
    def builddir(self, case_name):
        # case_name
        self._case_name = case_name
        
        # set dir
        self._case_dir = os.path.join(self._cases_dir, case_name)
        check_and_mkdir(self._case_dir)
        
        self.BasinMap_dir = os.path.join(self._case_dir, "BasinMap")
        check_and_mkdir(self.BasinMap_dir)
        
        self.dpcFile_dir = os.path.join(self._case_dir, "dpcFile")
        check_and_mkdir(self.dpcFile_dir)
        
        self.DomainFile_dir = os.path.join(self._case_dir, "DomainFile")
        check_and_mkdir(self.DomainFile_dir)
        
        self.GlobalParam_dir = os.path.join(self._case_dir, "GlobalParam")
        check_and_mkdir(self.GlobalParam_dir)
        
        self.MeteForcing_dir = os.path.join(self._case_dir, "MeteForcing")
        check_and_mkdir(self.MeteForcing_dir)
        
        self.ParamFile_dir = os.path.join(self._case_dir, "ParamFile")
        check_and_mkdir(self.ParamFile_dir)
        
        self.Hydroanalysis_dir = os.path.join(self._case_dir, "Hydroanalysis")
        check_and_mkdir(self.Hydroanalysis_dir)
        
        self.RVIC_dir = os.path.join(self._case_dir, "RVIC")
        check_and_mkdir(self.RVIC_dir)
        
        self.RVICParam_dir = os.path.join(self.RVIC_dir, "RVICParam")
        check_and_mkdir(self.RVICParam_dir)
        
        self.RVICTemp_dir = os.path.join(self.RVICParam_dir, "temp")
        check_and_mkdir(self.RVICTemp_dir)
        
        self.RVICConv_dir = os.path.join(self.RVIC_dir, "Convolution")
        check_and_mkdir(self.RVICConv_dir)
        
        self.VICLog_dir = os.path.join(self._case_dir, "VICLog")
        check_and_mkdir(self.VICLog_dir)
        
        self.VICResults_dir = os.path.join(self._case_dir, "VICResults")
        check_and_mkdir(self.VICResults_dir)
        
        self.VICResults_fig_dir = os.path.join(self.VICResults_dir, "Figs")
        remove_and_mkdir(self.VICResults_fig_dir)
        
        self.VICStates_dir = os.path.join(self._case_dir, "VICStates")
        check_and_mkdir(self.VICStates_dir)
        
        self.CalibrateVIC_dir = os.path.join(self._case_dir, "CalibrateVIC")
        check_and_mkdir(self.CalibrateVIC_dir)
        
        # set path
        self._dpc_VIC_level0_path = os.path.join(self.dpcFile_dir, "dpc_VIC_level0.pkl")
        self._dpc_VIC_level1_path = os.path.join(self.dpcFile_dir, "dpc_VIC_level1.pkl")
        self._dpc_VIC_level2_path = os.path.join(self.dpcFile_dir, "dpc_VIC_level2.pkl")
        self._dpc_VIC_plot_grid_basin_path = os.path.join(self.dpcFile_dir, "dpc_VIC_plot_grid_basin.tiff")
        self._dpc_VIC_plot_columns_path = os.path.join(self.dpcFile_dir, "dpc_VIC_plot_columns.tiff")
        
        self._domainFile_path = os.path.join(self.DomainFile_dir, "domain.nc")
        
        self._veg_param_json_path = os.path.join(self.__data_dir__, "veg_type_attributes_umd_updated.json")
        self._params_dataset_level0_path = os.path.join(self.ParamFile_dir, "params_level0.nc")
        self._params_dataset_level1_path = os.path.join(self.ParamFile_dir, "params_level1.nc")
        
        self._flow_direction_file_path = os.path.join(self.RVICParam_dir, "flow_direction_file.nc")
        self._pourpoint_file_path = os.path.join(self.RVICParam_dir, "pour_points.csv")
        self._uhbox_file_path = os.path.join(self.RVICParam_dir, "UHBOX.csv")
        self._rvic_param_cfg_file_path = os.path.join(self.RVICParam_dir, "rvic.parameters.cfg")
        self._rvic_param_cfg_file_reference_path = os.path.join(self.__data_dir__, "rvic.parameters.reference.cfg")
        self._rvic_conv_cfg_file_path = os.path.join(self.RVICConv_dir, "rvic.convolution.cfg")
        self._rvic_conv_cfg_file_reference_path = os.path.join(self.__data_dir__, "rvic.convolution.reference.cfg")
        self._rout_param_dir = os.path.join(self.RVICParam_dir, "params")
        
        self._globalParam_path = os.path.join(self.GlobalParam_dir, "global_param.txt")
        self._globalParam_reference_path = os.path.join(self.__data_dir__, "global_param_reference.txt")
        
        self._calibrate_cp_path = os.path.join(self.CalibrateVIC_dir, "calibrate_cp.pkl")
    
    @property
    def MeteForcing_src_dir(self):
        return self._MeteForcing_src_dir

    @MeteForcing_src_dir.setter
    def MeteForcing_src_dir(self, MeteForcing_src_dir):
        self._MeteForcing_src_dir = MeteForcing_src_dir
    
    @property
    def MeteForcing_src_suffix(self):
        return self._MeteForcing_src_suffix

    @MeteForcing_src_suffix.setter
    def MeteForcing_src_suffix(self, MeteForcing_src_suffix):
        self._MeteForcing_src_suffix = MeteForcing_src_suffix
    
    @property
    def forcing_prefix(self):
        return self._forcing_prefix

    @forcing_prefix.setter
    def forcing_prefix(self, forcing_prefix):
        self._forcing_prefix = forcing_prefix
    
    @property
    def linux_share_temp_dir(self):
        return self._linux_share_temp_dir

    @linux_share_temp_dir.setter
    def linux_share_temp_dir(self, linux_share_temp_dir):
        self._linux_share_temp_dir = linux_share_temp_dir
    
    @property
    def arcpy_python_path(self):
        return self._arcpy_python_path
    
    @arcpy_python_path.setter
    def arcpy_python_path(self, arcpy_python_path):
        self._arcpy_python_path = arcpy_python_path
    
    @property
    def vic_exe_path(self):
        return self._vic_exe_path
    
    @vic_exe_path.setter
    def vic_exe_path(self, vic_exe_path):
        self._vic_exe_path = vic_exe_path
    
    # ------------ general path set ------------
    
    @property
    def dpc_VIC_level0_path(self):
        return self._dpc_VIC_level0_path

    @dpc_VIC_level0_path.setter
    def dpc_VIC_level0_path(self, dpc_VIC_level0_path):
        self._dpc_VIC_level0_path = dpc_VIC_level0_path
    
    @property
    def dpc_VIC_level1_path(self):
        return self._dpc_VIC_level1_path

    @dpc_VIC_level1_path.setter
    def dpc_VIC_level1_path(self, dpc_VIC_level1_path):
        self._dpc_VIC_level1_path = dpc_VIC_level1_path
        
    @property
    def dpc_VIC_level2_path(self):
        return self._dpc_VIC_level2_path

    @dpc_VIC_level2_path.setter
    def dpc_VIC_level2_path(self, dpc_VIC_level2_path):
        self._dpc_VIC_level2_path = dpc_VIC_level2_path
        
    @property
    def dpc_VIC_plot_grid_basin_path(self):
        return self._dpc_VIC_plot_grid_basin_path

    @dpc_VIC_plot_grid_basin_path.setter
    def dpc_VIC_plot_grid_basin_path(self, dpc_VIC_plot_grid_basin_path):
        self._dpc_VIC_plot_grid_basin_path = dpc_VIC_plot_grid_basin_path
    
    @property
    def dpc_VIC_plot_columns_path(self):
        return self._dpc_VIC_plot_columns_path

    @dpc_VIC_plot_columns_path.setter
    def dpc_VIC_plot_columns_path(self, dpc_VIC_plot_columns_path):
        self._dpc_VIC_plot_columns_path = dpc_VIC_plot_columns_path
        
    @property
    def domainFile_path(self):
        return self._domainFile_path

    @domainFile_path.setter
    def domainFile_path(self, domainFile_path):
        self._domainFile_path = domainFile_path
    
    @property
    def veg_param_json_path(self):
        return self._veg_param_json_path

    @veg_param_json_path.setter
    def veg_param_json_path(self, veg_param_json_path):
        self._veg_param_json_path = veg_param_json_path
    
    @property
    def params_dataset_level0_path(self):
        return self._params_dataset_level0_path

    @params_dataset_level0_path.setter
    def params_dataset_level0_path(self, params_dataset_level0_path):
        self._params_dataset_level0_path = params_dataset_level0_path
        
    @property
    def params_dataset_level1_path(self):
        return self._params_dataset_level1_path

    @params_dataset_level1_path.setter
    def params_dataset_level1_path(self, params_dataset_level1_path):
        self._params_dataset_level1_path = params_dataset_level1_path
        
    @property
    def flow_direction_file_path(self):
        return self._flow_direction_file_path

    @flow_direction_file_path.setter
    def flow_direction_file_path(self, flow_direction_file_path):
        self._flow_direction_file_path = flow_direction_file_path
        
    @property
    def pourpoint_file_path(self):
        return self._pourpoint_file_path

    @pourpoint_file_path.setter
    def pourpoint_file_path(self, pourpoint_file_path):
        self._pourpoint_file_path = pourpoint_file_path
    
    @property
    def uhbox_file_path(self):
        return self._uhbox_file_path

    @uhbox_file_path.setter
    def uhbox_file_path(self, uhbox_file_path):
        self._uhbox_file_path = uhbox_file_path
        
    @property
    def rvic_param_cfg_file_path(self):
        return self._rvic_param_cfg_file_path

    @rvic_param_cfg_file_path.setter
    def rvic_param_cfg_file_path(self, rvic_param_cfg_file_path):
        self._rvic_param_cfg_file_path = rvic_param_cfg_file_path
    
    @property
    def rvic_param_cfg_file_reference_path(self):
        return self._rvic_param_cfg_file_reference_path

    @rvic_param_cfg_file_reference_path.setter
    def rvic_param_cfg_file_reference_path(self, rvic_param_cfg_file_reference_path):
        self._rvic_param_cfg_file_reference_path = rvic_param_cfg_file_reference_path
    
    @property
    def rvic_conv_cfg_file_path(self):
        return self._rvic_conv_cfg_file_path

    @rvic_conv_cfg_file_path.setter
    def rvic_conv_cfg_file_path(self, rvic_conv_cfg_file_path):
        self._rvic_conv_cfg_file_path = rvic_conv_cfg_file_path
        
    @property
    def rvic_conv_cfg_file_reference_path(self):
        return self._rvic_conv_cfg_file_reference_path

    @rvic_conv_cfg_file_reference_path.setter
    def rvic_conv_cfg_file_reference_path(self, rvic_conv_cfg_file_reference_path):
        self._rvic_conv_cfg_file_reference_path = rvic_conv_cfg_file_reference_path
        
    @property
    def rout_param_dir(self):
        return self._rout_param_dir

    @rout_param_dir.setter
    def rout_param_dir(self, rout_param_dir):
        self._rout_param_dir = rout_param_dir
    
    @property
    def globalParam_path(self):
        return self._globalParam_path

    @globalParam_path.setter
    def globalParam_path(self, globalParam_path):
        self._globalParam_path = globalParam_path
        
    @property
    def globalParam_reference_path(self):
        return self._globalParam_reference_path

    @globalParam_reference_path.setter
    def globalParam_reference_path(self, globalParam_reference_path):
        self._globalParam_reference_path = globalParam_reference_path
        
    @property
    def calibrate_cp_path(self):
        return self._calibrate_cp_path

    @calibrate_cp_path.setter
    def calibrate_cp_path(self, calibrate_cp_path):
        self._calibrate_cp_path = calibrate_cp_path
        