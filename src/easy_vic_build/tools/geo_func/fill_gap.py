# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
"""Module ``easy_vic_build.tools.geo_func.fill_gap``."""

from scipy.ndimage import distance_transform_edt
import numpy as np

def nearest_neighbor_fill(data):
    """
    Fill missing values using nearest-neighbor propagation.

    Parameters
    ----------
    data : numpy.ndarray or numpy.ma.MaskedArray
        Input array with missing values represented by a mask.

    Returns
    -------
    numpy.ndarray
        Array where masked values are replaced by nearest valid neighbors.

    Notes
    -----
    For masked arrays, the mask is used directly. For plain arrays, this
    function follows the existing implementation behavior.
    """
    mask = data.mask if isinstance(data, np.ma.MaskedArray) else (data == np.nan)
    indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return data[tuple(indices)]
