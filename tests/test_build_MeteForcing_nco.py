# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import sys
sys.path.append("../easy_vic_build")
from easy_vic_build.tools.utilities import readdpc
from easy_vic_build import Evb_dir
from easy_vic_build.build_MeteForcing_nco import buildMeteForcingnco

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
    
    # set linux_share_temp_dir
    evb_dir.linux_share_temp_dir = "F:\\Linux\\C_VirtualBox_Share\\temp"
    
    # build MeteForcingnco: step=1
    # buildMeteForcingnco(dpc_VIC_level1, evb_dir, date_period,
    #                     step=1, reverse_lat=True, check_search=False,
    #                     year_re_exp=r"\d{4}.nc4")
    
    # go to linux, run combineYearly.py
    
    # build MeteForcingnco: step=2
    # buildMeteForcingnco(dpc_VIC_level1, evb_dir, date_period,
    #                     step=2, reverse_lat=True, check_search=False,
    #                     year_re_exp=r"\d{4}.nc4")
    
    # build MeteForcingnco: step=3
    buildMeteForcingnco(dpc_VIC_level1, evb_dir, date_period,
                        step=3, reverse_lat=True, check_search=False,
                        year_re_exp=r"\d{4}.nc4")
    
    