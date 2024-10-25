# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import sys
sys.path.append("../easy_vic_build")
from easy_vic_build.tools.utilities import readdpc
from easy_vic_build import Evb_dir
from easy_vic_build.build_MeteForcing import buildMeteForcing

if __name__ == "__main__":
    basin_index = 106
    date_period = ["19980101", "20101231"]
    case_name = "106_3km"
    
    # build dir
    evb_dir = Evb_dir()
    evb_dir.builddir(case_name)
    
    # read dpc
    dpc_VIC_level0, dpc_VIC_level1, dpc_level2 = readdpc(evb_dir)

    # set MeteForcing_src_dir and MeteForcing_src_suffix
    evb_dir.MeteForcing_src_dir = "E:\\data\\hydrometeorology\\NLDAS\\NLDAS2_Primary_Forcing_Data_subset_0.125\\data"
    evb_dir.MeteForcing_src_suffix = ".nc4"
    
    # build MeteForcing
    buildMeteForcing(dpc_VIC_level1, evb_dir, date_period,
                     reverse_lat=True, check_search=False,
                     time_re_exp=r"\d{8}.\d{4}")