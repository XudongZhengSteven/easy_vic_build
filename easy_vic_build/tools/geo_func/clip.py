# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import numpy as np

# clip: extract before to improve speed, avoid to creating too large search array, as below
# src_lon_mesh, src_lat_mesh = np.meshgrid(src_lon, src_lat)  # 2D array
def clip(dst_lat, dst_lon, dst_res, src_lat, src_lon, src_data, reverse_lat=True):
    xindex_start = np.where(src_lon <= min(dst_lon) - dst_res/2)[0][-1]
    xindex_end = np.where(src_lon >= max(dst_lon) + dst_res/2)[0][0]
    
    # if reverse_lat (src_lat, large -> small), else (src_lat, small -> large)
    if reverse_lat:
        yindex_start = np.where(src_lat >= max(dst_lat) + dst_res/2)[0][-1]
        yindex_end = np.where(src_lat <= min(dst_lat) - dst_res/2)[0][0]
    else:
        yindex_start = np.where(src_lat <= min(dst_lat) - dst_res/2)[0][-1]
        yindex_end = np.where(src_lat >= max(dst_lat) + dst_res/2)[0][0]
    
    src_data_clip = src_data[yindex_start: yindex_end+1, xindex_start: xindex_end+1]
    src_lon_clip = src_lon[xindex_start: xindex_end+1]
    src_lat_clip = src_lat[yindex_start: yindex_end+1]
    
    ## old version
    # xindex = np.where((src_lon >= min(dst_lon) - dst_res/2) & (src_lon <= max(dst_lon) + dst_res/2))[0]
    # yindex = np.where((src_lat >= min(dst_lat) - dst_res/2) & (src_lat <= max(dst_lat) + dst_res/2))[0]
    
    # src_data_clip = src_data[min(yindex): max(yindex), min(xindex): max(xindex)]
    # src_lon_clip = src_lon[min(xindex): max(xindex)]
    # src_lat_clip = src_lat[min(yindex): max(yindex)]
    
    ## then search grids
    # searched_grids_index = search_grids.search_grids_radius_rectangle(dst_lat=grids_lat, dst_lon=grids_lon,
    #                                                                     src_lat=umd_lat_clip, src_lon=umd_lon_clip,
    #                                                                     lat_radius=grid_shp_res/2, lon_radius=grid_shp_res/2)
    
    return src_data_clip, src_lon_clip, src_lat_clip