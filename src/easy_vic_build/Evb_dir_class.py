# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: Evb_dir_class

This module provides a class for managing directory structures related to different case scenarios in the VIC 
model. The `Evb_dir` class simplifies the creation, organization, and retrieval of directories required for 
storing input, output, and parameter files used in VIC simulations. It ensures that necessary directories are 
properly structured and accessible.

Class:
------
    - Evb_dir: A class to handle the creation and management of directory structures for different case 
               scenarios in the VIC model. The `Evb_dir` class defines methods and properties for managing 
               directories and paths related to the VIC model's operations. 
               It helps automate the creation of necessary directories for model configuration, calibration, and execution.

Dependencies:
-------------
    - os: Provides functions for interacting with the operating system, such as creating directories.
    - shutil: Used for high-level file and directory operations, including copying and moving files.

Author:
-------
    Xudong Zheng
    Email: zhengxd@sehemodel.club
"""

from .tools.utilities import check_and_mkdir, remove_and_mkdir
from . import logger
import os

# Class to manage directories and paths for the easy_vic_build package
class Evb_dir:
    """
    A class to handle the creation and management of directory structures for different case scenarios in the VIC model.
    """
    
    # Initialize the base directory paths
    # __package_dir__ = "./easy_vic_build" # os.path.abspath(os.path.dirname(__file__))
    # __data_dir__ = os.path.join(os.path.dirname(__package_dir__), "data")
    
    def __init__(self, cases_home=None):
        """
        Initialize the directory paths for a given case name or set the default to current working directory.
        """
        logger.debug("Initializing Evb_dir class")
        
        # self._cases_dir = cases_home if cases_home is not None else os.path.join(Evb_dir.__package_dir__, "cases")
        self._cases_dir = cases_home if cases_home is not None else os.path.join(os.getcwd(), "cases")
        logger.debug(f"Case directory set to: {self._cases_dir}")
        
        # Initialize all paths for meteorological forcing, domain, parameters, etc.
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
        logger.debug("Evb_dir class initialized successfully")
         
    def builddir(self, case_name):
        """
        Create directories and set paths for a specific case name.
        """
        logger.info(f"Starting to create directories for case: {case_name}")
        
        self._case_name = case_name
        
        # Create base directories for the case
        check_and_mkdir(self._cases_dir)
        logger.debug(f"Base case directory created at: {self._cases_dir}")
        
        self._case_dir = os.path.join(self._cases_dir, case_name)
        check_and_mkdir(self._case_dir)
        logger.debug(f"Case directory created at: {self._case_dir}")
        
        
        # Create subdirectories for different components
        self.BasinMap_dir = os.path.join(self._case_dir, "BasinMap")
        check_and_mkdir(self.BasinMap_dir)
        logger.debug(f"BasinMap directory created at: {self.BasinMap_dir}")
        
        
        self.dpcFile_dir = os.path.join(self._case_dir, "dpcFile")
        check_and_mkdir(self.dpcFile_dir)
        logger.debug(f"dpcFile directory created at: {self.dpcFile_dir}")
        
        
        self.DomainFile_dir = os.path.join(self._case_dir, "DomainFile")
        check_and_mkdir(self.DomainFile_dir)
        logger.debug(f"DomainFile directory created at: {self.DomainFile_dir}")
        
        
        self.GlobalParam_dir = os.path.join(self._case_dir, "GlobalParam")
        check_and_mkdir(self.GlobalParam_dir)
        logger.debug(f"GlobalParam directory created at: {self.GlobalParam_dir}")
        
        
        self.MeteForcing_dir = os.path.join(self._case_dir, "MeteForcing")
        check_and_mkdir(self.MeteForcing_dir)
        logger.debug(f"MeteForcing directory created at: {self.MeteForcing_dir}")
        
        
        self.ParamFile_dir = os.path.join(self._case_dir, "ParamFile")
        check_and_mkdir(self.ParamFile_dir)
        logger.debug(f"ParamFile directory created at: {self.ParamFile_dir}")
        
        
        self.Hydroanalysis_dir = os.path.join(self._case_dir, "Hydroanalysis")
        check_and_mkdir(self.Hydroanalysis_dir)
        logger.debug(f"Hydroanalysis directory created at: {self.Hydroanalysis_dir}")
        
        
        self.RVIC_dir = os.path.join(self._case_dir, "RVIC")
        check_and_mkdir(self.RVIC_dir)
        logger.debug(f"RVIC directory created at: {self.RVIC_dir}")
        
        
        # Create directories for RVIC parameters and temporary files
        self.RVICParam_dir = os.path.join(self.RVIC_dir, "RVICParam")
        check_and_mkdir(self.RVICParam_dir)
        logger.debug(f"RVICParam directory created at: {self.RVICParam_dir}")
        
        
        self.RVICTemp_dir = os.path.join(self.RVICParam_dir, "temp")
        check_and_mkdir(self.RVICTemp_dir)
        logger.debug(f"RVICTemp directory created at: {self.RVICTemp_dir}")
        
        
        self.RVICConv_dir = os.path.join(self.RVIC_dir, "Convolution")
        check_and_mkdir(self.RVICConv_dir)
        logger.debug(f"RVICConv directory created at: {self.RVICConv_dir}")
        
        
        # Directories for logs, results, and states
        self.VICLog_dir = os.path.join(self._case_dir, "VICLog")
        check_and_mkdir(self.VICLog_dir)
        logger.debug(f"VICLog directory created at: {self.VICLog_dir}")
        
        
        self.VICResults_dir = os.path.join(self._case_dir, "VICResults")
        check_and_mkdir(self.VICResults_dir)
        logger.debug(f"VICResults directory created at: {self.VICResults_dir}")
        
        
        self.VICResults_fig_dir = os.path.join(self.VICResults_dir, "Figs")
        remove_and_mkdir(self.VICResults_fig_dir)
        logger.debug(f"VICResults figs directory created at: {self.VICResults_fig_dir}")
        
        
        self.VICStates_dir = os.path.join(self._case_dir, "VICStates")
        check_and_mkdir(self.VICStates_dir)
        logger.debug(f"VICStates directory created at: {self.VICStates_dir}")
        
        
        self.CalibrateVIC_dir = os.path.join(self._case_dir, "CalibrateVIC")
        check_and_mkdir(self.CalibrateVIC_dir)
        logger.debug(f"CalibrateVIC directory created at: {self.CalibrateVIC_dir}")
        
        
        # Set paths for specific files
        self._dpc_VIC_level0_path = os.path.join(self.dpcFile_dir, "dpc_VIC_level0.pkl")
        self._dpc_VIC_level1_path = os.path.join(self.dpcFile_dir, "dpc_VIC_level1.pkl")
        self._dpc_VIC_level2_path = os.path.join(self.dpcFile_dir, "dpc_VIC_level2.pkl")
        self._dpc_VIC_plot_grid_basin_path = os.path.join(self.dpcFile_dir, "dpc_VIC_plot_grid_basin.tiff")
        self._dpc_VIC_plot_columns_path = os.path.join(self.dpcFile_dir, "dpc_VIC_plot_columns.tiff")
        
        self._domainFile_path = os.path.join(self.DomainFile_dir, "domain.nc")
        
        # self._veg_param_json_path = os.path.join(self.__data_dir__, "veg_type_attributes_umd_updated.json")
        self._params_dataset_level0_path = os.path.join(self.ParamFile_dir, "params_level0.nc")
        self._params_dataset_level1_path = os.path.join(self.ParamFile_dir, "params_level1.nc")
        
        self._flow_direction_file_path = os.path.join(self.RVICParam_dir, "flow_direction_file.nc")
        self._pourpoint_file_path = os.path.join(self.RVICParam_dir, "pour_points.csv")
        self._uhbox_file_path = os.path.join(self.RVICParam_dir, "UHBOX.csv")
        self._rvic_param_cfg_file_path = os.path.join(self.RVICParam_dir, "rvic.parameters.cfg")
        # self._rvic_param_cfg_file_reference_path = os.path.join(self.__data_dir__, "rvic.parameters.reference.cfg")
        self._rvic_conv_cfg_file_path = os.path.join(self.RVICConv_dir, "rvic.convolution.cfg")
        # self._rvic_conv_cfg_file_reference_path = os.path.join(self.__data_dir__, "rvic.convolution.reference.cfg")
        self._rout_param_dir = os.path.join(self.RVICParam_dir, "params")
        
        self._globalParam_path = os.path.join(self.GlobalParam_dir, "global_param.txt")
        # self._globalParam_reference_path = os.path.join(self.__data_dir__, "global_param_reference.txt")
        
        self._calibrate_cp_path = os.path.join(self.CalibrateVIC_dir, "calibrate_cp.pkl")
        
        logger.info(f"Directories for case '{case_name}' created successfully")
    
    # define property and setter
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