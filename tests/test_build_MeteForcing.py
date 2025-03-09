# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from easy_vic_build.tools.utilities import readdpc
from easy_vic_build import Evb_dir
from easy_vic_build.build_MeteForcing import buildMeteForcing

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
    dpc_VIC_level0, dpc_VIC_level1, dpc_VIC_level1 = readdpc(evb_dir)
    
    # set MeteForcing_src_dir and MeteForcing_src_suffix
    evb_dir.MeteForcing_src_dir = "E:\\data\\hydrometeorology\\NLDAS\\NLDAS2_Primary_Forcing_Data_subset_0.125\\data"
    evb_dir.MeteForcing_src_suffix = ".nc4"
    
    # build MeteForcing
    buildMeteForcing(evb_dir, dpc_VIC_level1, date_period,
                     reverse_lat=True, check_search=False,
                     time_re_exp=r"\d{8}.\d{4}")


if __name__ == "__main__":
    test()