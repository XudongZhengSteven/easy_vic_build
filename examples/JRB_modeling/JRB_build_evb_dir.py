# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
from easy_vic_build.Evb_dir_class import Evb_dir

def build_modeling_dir(subname="12km"):
    case_name = f"JRB_{subname}"
    evb_dir = Evb_dir(cases_home="../")  # F:\\research\\Research\\JRB_scaling\\modeling
    evb_dir.builddir(case_name)
    return evb_dir