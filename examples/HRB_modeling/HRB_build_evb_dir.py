# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from easy_vic_build.Evb_dir_class import Evb_dir
from easy_vic_build import logger
from pathlib import Path
import os

def build_modeling_dir(subname="12km"):
    case_name = f"HRB_{subname}"
    cases_home = Path(__file__).resolve().parent.parent / "modeling"
    evb_dir = Evb_dir(cases_home=str(cases_home))
    evb_dir.builddir(case_name)
    
    return evb_dir