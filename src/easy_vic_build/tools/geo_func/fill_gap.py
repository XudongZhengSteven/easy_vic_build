# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
from scipy.ndimage import distance_transform_edt
import numpy as np

def nearest_neighbor_fill(data):
    mask = data.mask if isinstance(data, np.ma.MaskedArray) else (data == np.nan)
    indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return data[tuple(indices)]
