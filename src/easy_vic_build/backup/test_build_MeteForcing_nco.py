# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from easy_vic_build.tools.utilities import readdpc
from easy_vic_build.Evb_dir_class import Evb_dir
from easy_vic_build.build_MeteForcing_nco import buildMeteForcingnco
import shutil, os

"""
general information:

basin set
106(10_100_km_humid); 240(10_100_km_semi_humid); 648(10_100_km_semi_arid); 
213(100_1000_km_humid); 38(100_1000_km_semi_humid); 670(10_100_km_semi_arid);
397(1000_larger_km_humid); 636(1000_larger_km_semi_humid); 580(1000_larger_km_semi_arid) 

grid_res_level0=1km(0.00833)
grid_res_level1=3km(0.025), 6km(0.055), 8km(0.072), 12km(0.11)

""" 

scalemap = {"3km": 0.025, "6km": 0.055, "8km": 0.072, "12km": 0.11}

def test():
    # general set
    basin_index = 213
    model_scale = "6km"
    date_period = ["19980101", "19981231"]
    case_name = f"{basin_index}_{model_scale}"
    
    # build dir
    evb_dir = Evb_dir("./examples")
    evb_dir.builddir(case_name)
    
    # read dpc
    dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level2 = readdpc(evb_dir)

    # set MeteForcing_src_dir and MeteForcing_src_suffix
    evb_dir.MeteForcing_src_dir = "E:\\data\\hydrometeorology\\NLDAS\\NLDAS2_Primary_Forcing_Data_subset_0.125\\data"
    evb_dir.MeteForcing_src_suffix = ".nc4"
    
    # set linux_share_temp_dir
    evb_dir.linux_share_temp_dir = "F:\\Linux\\C_VirtualBox_Share\\temp"
    
    # cp combineYearly.py to share_temp_home
    combineYearly_py_path = "F:\\research\\Research\\easy_vic_build\\easy_vic_build\\scripts\\linux_scripts\\combineYearly.py"
    if not os.path.exists(os.path.join(evb_dir.linux_share_temp_dir, "combineYearly.py")):
        shutil.copy(combineYearly_py_path, os.path.join(evb_dir.linux_share_temp_dir, "combineYearly.py"))
    
    # build MeteForcingnco: step=1
    # buildMeteForcingnco(evb_dir, dpc_VIC_level1, date_period,
    #                     step=1, reverse_lat=True, check_search=False,
    #                     year_re_exp=r"\d{4}.nc4")
    
    # go to linux, run combineYearly.py
    
    # build MeteForcingnco: step=2
    # buildMeteForcingnco(evb_dir, dpc_VIC_level1, date_period,
    #                     step=2, reverse_lat=True, check_search=False,
    #                     year_re_exp=r"\d{4}.nc4")
    
    # build MeteForcingnco: step=3
    buildMeteForcingnco(evb_dir, dpc_VIC_level1, date_period,
                        step=3, reverse_lat=True, check_search=False,
                        year_re_exp=r"\d{4}.nc4")
    
    # build MeteForcingnco: step=4, resample
    # buildMeteForcingnco(evb_dir, dpc_VIC_level1, date_period,
    #                     step=4, reverse_lat=True, check_search=False,
    #                     year_re_exp=r"\d{4}.nc4",
    #                     dst_time_hours=24)

if __name__ == "__main__":
    test()