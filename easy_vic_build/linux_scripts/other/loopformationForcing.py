import os
import re
from tqdm import *

# general
cwd_path = os.getcwd()
suffix = ".nc"

subdir = "yearNLDASforcings"
src_fnames = [n for n in os.listdir(os.path.join(cwd_path, subdir)) if n.endswith(suffix)]

# loop for combine files
for i in tqdm(range(len(src_fnames))):
    src_fname = src_fnames[i]
    src_path = os.path.join(cwd_path, subdir, src_fname)
    dst_path = src_path[:src_path.find(".nc")-5] + ".formation_temp" + src_path[src_path.find(".nc")-5:]
    os.system(f'ncap2 -S ./formationForcing.nco {src_path} {dst_path}')

for i in tqdm(range(len(src_fnames))): 
    src_fname = src_fnames[i]
    src_path = os.path.join(cwd_path, subdir, src_fname)
    dst_path = src_path[:src_path.find(".nc")] + ".formation_temp.nc*"
    dst_path_final = src_path[:src_path.find(".nc")-5] + ".formation" + src_path[src_path.find(".nc")-5:]
    os.system(f'ncrcat -v SPFH,TMP -x {dst_path} {dst_path_final}')
    os.system(f'rm {dst_path}')


