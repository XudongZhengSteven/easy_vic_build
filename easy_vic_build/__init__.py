import os

__version__ = "1.0.0"
__author__ = "Xudong Zheng"
__email__ = "zhengxd@sehemodel.club"

from . import tools
from . import dataPreprocess
from .tools.utilities import check_and_mkdir


class Evb_dir:
    __package_dir__ = "../easy_vic_build" # os.path.abspath(os.path.dirname(__file__))
    __data_dir__ = os.path.join(__package_dir__, "data")
    
    def __init__(self, cases_home=None):
        cases_home = cases_home if cases_home is not None else Evb_dir.__package_dir__
        self._cases_dir = os.path.join(cases_home, "cases")
        self._MeteForcing_src_dir = ""
        self._MeteForcing_src_suffix = ".nc4"
        self._linux_share_temp_dir = ""
        self._arcpy_python_path = ""
        
    def builddir(self, case_name):
        # case_name
        self._case_name = case_name
        
        # set dir
        self._case_dir = os.path.join(self._cases_dir, case_name)
        check_and_mkdir(self._case_dir)
        
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
        
        self.VICLog_dir = os.path.join(self._case_dir, "VICLog")
        check_and_mkdir(self.VICLog_dir)
        
        self.VICResults_dir = os.path.join(self._case_dir, "VICResults")
        check_and_mkdir(self.VICResults_dir)
        
        self.VICStates_dir = os.path.join(self._case_dir, "VICStates")
        check_and_mkdir(self.VICStates_dir)
        
        self.RVICParam_dir = os.path.join(self._case_dir, "RVICParam")
        check_and_mkdir(self.RVICParam_dir)
    
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
        