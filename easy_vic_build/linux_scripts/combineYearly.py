import os
import re
from tqdm import *
import sys
#TODO use argparse
#* run it on Linux

def combineYearly(linux_share_temp_dir, suffix=".clip.nc4", year_re_exp=r"\d{8}"):
    # ====================== set dir and path ======================
    # set path
    linux_share_temp_clip_dir = os.path.join(linux_share_temp_dir, "clip")
    linux_share_temp_combineYearly_dir = os.path.join(linux_share_temp_dir, "combineYearly")
    if not os.path.isdir(linux_share_temp_combineYearly_dir):
        os.mkdir(linux_share_temp_combineYearly_dir)
    
    src_fnames = [n for n in os.listdir(linux_share_temp_clip_dir) if n.endswith(suffix)]
    prefix = src_fnames[0][:re.search(year_re_exp, src_fnames[0]).start()]
    
    # groupy based year
    year_list = [re.search(year_re_exp, fn)[0][:4] for fn in src_fnames]
    year_set = list(set(year_list))

    # loop for combine files into yearly
    for i in tqdm(range(len(year_set)), desc="loop for combine files into yearly", colour="green"):
        year_ = year_set[i]
        
        os.system(f'ncrcat {linux_share_temp_clip_dir}/{prefix}{year_}*.nc4 {linux_share_temp_combineYearly_dir}/{prefix}{year_}.nc4')


if __name__ == "__main__":
    # default args
    suffix = ".clip.nc4"
    year_re_exp = r"\d{8}"
    
    # get args from command
    linux_share_temp_dir = sys.argv[1]
    if len(sys.argv) > 2:
        suffix = sys.argv[2]
        if len(sys.argv) > 3:
            year_re_exp = sys.argv[3]
    
    combineYearly(linux_share_temp_dir, suffix, year_re_exp)