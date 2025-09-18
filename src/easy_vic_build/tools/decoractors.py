# code: utf-8
# author: "Xudong Zheng"
# email: Z786909151@163.com

"""
decorators - A Python module providing utility decorators.

This module provides utility decorators for enhancing function behavior.

Decorators:
-----------
    - `clock_decorator`: A decorator for measuring the execution time of a function and logging the result.
    - `apply_along_axis_decorator`: A decorator that applies a function along a specific axis of a numpy ndarray.

Usage:
------
    1. `clock_decorator`: Wrap your function with this decorator to measure and log its execution time.
       Example:
           @clock_decorator(print_arg_ret=True)
           def your_function(args):
               # Function implementation

    2. `apply_along_axis_decorator`: Wrap your function with this decorator to apply it along a specified axis
       of a numpy ndarray.
       Example:
           @apply_along_axis_decorator(axis=1)
           def your_function(arr):
               # Function implementation

Example:
--------
    @clock_decorator(print_arg_ret=True)
    def example_function(x, y):
        # Perform some task
        return x + y

    @apply_along_axis_decorator(axis=0)
    def example_function(arr):
        # Apply a function along axis 0 of the array
        return np.sum(arr, axis=0)

Dependencies:
-------------
    - `functools`: For wrapping functions and ensuring they maintain their original signature.
    - `numpy`: For applying functions along specific axes of ndarrays.

"""

import functools
import time
from typing import Callable, Dict, List, Optional, Any, Union
import numpy as np

from .. import logger


def clock_decorator(print_arg_ret=True):
    """
    A decorator to measure and log the execution time of a function.

    Parameters
    ----------
    print_arg_ret : bool, optional
        If True, print the function's arguments, execution time, and return value.
        If False, only print the function's name and execution time.
        Default is True.

    Returns
    -------
    function
        A wrapped function that logs the execution time and optionally the arguments and return value.

    Notes
    -----
    This decorator prints the time taken for a function to execute, as well as its arguments and return value.
    It uses the `logger` module for logging if necessary.
    """

    def clock_decorator_real(func):
        """
        The actual decorator that wraps the target function.

        Parameters
        ----------
        func : function
            The function to be wrapped by the decorator.

        Returns
        -------
        function
            A wrapped version of `func` that logs the execution time.
        """

        @functools.wraps(func)
        def clocked(*args, **kwargs):
            """
            The function that records the execution time and logs the result.

            Parameters
            ----------
            *args : tuple
                Positional arguments passed to the wrapped function.
                
            **kwargs : dict
                Keyword arguments passed to the wrapped function.

            Returns
            -------
            result : any
                The return value of the wrapped function.
            """
            t0 = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - t0
            name = func.__name__

            # Prepare argument list for logging
            arg_lst = []
            if args:
                arg_lst.append(",".join(repr(arg) for arg in args))
            if kwargs:
                pairs = ["%s=%r" % (k, w) for k, w in sorted(kwargs.items())]
                arg_lst.append(",".join(pairs))
            arg_str = ",".join(arg_lst)

            # Log the result based on the print_arg_ret flag
            if print_arg_ret:
                log_msg = "[%0.8fs] %s(%s) -> %r" % (elapsed, name, arg_str, result)
            else:
                log_msg = "[%0.8fs] %s" % (elapsed, name)

            # Use logger for logging
            logger.info(log_msg)

            return result

        return clocked

    return clock_decorator_real


def apply_along_axis_decorator(axis=0):
    """
    A decorator to apply a function along a specific axis of a numpy ndarray.

    Parameters
    ----------
    axis : int, optional
        The axis along which to apply the function. Default is 0, which means
        the function will be applied along the rows (axis 0) of the ndarray.

    Returns
    -------
    function
        A wrapped function that applies the given function along the specified axis
        of the input ndarray.

    Notes
    -----
    This decorator uses `np.apply_along_axis` to apply a function to slices of
    an ndarray along the specified axis. It is useful when you need to perform
    element-wise operations along a particular axis of a multidimensional array.
    """

    def decorator(func):
        """
        The actual decorator that wraps the target function.

        Parameters
        ----------
        func : function
            The function to be wrapped by the decorator. It is applied along
            the specified axis of the ndarray.

        Returns
        -------
        function
            A wrapped version of `func` that applies the function along the given axis
            of the ndarray.
        """

        @functools.wraps(func)
        def apply_along_axis_func(*args, **kwargs):
            """
            Apply the wrapped function along the specified axis of the input ndarray.

            Parameters
            ----------
            *args : tuple
                Positional arguments passed to the wrapped function.
                
            **kwargs : dict
                Keyword arguments passed to the wrapped function.

            Returns
            -------
            result : ndarray
                The result of applying the wrapped function along the specified axis.
            """

            result = np.apply_along_axis(func, axis, *args, **kwargs)
            return result

        return apply_along_axis_func

    return decorator


def resample_time_series_wrapper(resample_func):
    """
    Decorator to extend a spatial resampling function to handle time series data.

    If `searched_grids_data` is 1D (single time point), the function is called directly.
    If it is 2D (time x space), the function is applied to each time slice independently,
    and an array of results is returned.

    Parameters
    ----------
    resample_func : callable
        A function that performs resampling on a single time point (1D spatial data).
    
    Returns
    -------
    callable
        Wrapped function that can handle 1D or 2D time series inputs.
    """
    @functools.wraps(resample_func)
    def wrapper(searched_grids_data, searched_grids_lat, searched_grids_lon,
                dst_lat=None, dst_lon=None, missing_value=None, *args, **kwargs):
        
        data_array = np.array(searched_grids_data)
        lat_array  = np.array(searched_grids_lat, dtype=float)
        lon_array  = np.array(searched_grids_lon, dtype=float)
        
        # 1D, searched_grids
        if data_array.ndim == 1:
            return resample_func(data_array, lat_array, lon_array,
                                 dst_lat, dst_lon, missing_value, *args, **kwargs)
        
        # 2D, time * space (searched_grids)
        elif data_array.ndim == 2:
            results = []
            for t in range(data_array.shape[0]):
                result = resample_func(data_array[t], lat_array, lon_array,
                                       dst_lat, dst_lon, missing_value, *args, **kwargs)
                results.append(result)
                
            return np.array(results)
        
        else:
            raise ValueError("searched_grids_data must be 1D or 2D (time, space).")
    
    return wrapper


def resample_missing_wrapper(resample_func):
    """
    Decorator to automatically handle missing values in the input grids
    for a resampling function. Supports `missing_value=None` (np.nan) or
    a specified missing value.
    """
    @functools.wraps(resample_func)
    def wrapper(searched_grids_data, searched_grids_lat, searched_grids_lon,
                dst_lat=None, dst_lon=None, missing_value=None, *args, **kwargs):
        
        data_array = np.asarray(searched_grids_data, dtype=float)
        lat_array  = np.asarray(searched_grids_lat, dtype=float)
        lon_array  = np.asarray(searched_grids_lon, dtype=float)

        if missing_value is None:
            miss_bool = np.isnan(data_array)
        else:
            miss_bool = (data_array == missing_value) | np.isnan(data_array)

        data_clean = data_array[~miss_bool]
        lat_clean  = lat_array[~miss_bool]
        lon_clean  = lon_array[~miss_bool]

        return resample_func(data_clean, lat_clean, lon_clean,
                             dst_lat=dst_lat, dst_lon=dst_lon, missing_value=missing_value, *args, **kwargs)

    return wrapper



@clock_decorator
def test_func_clock_decorator():
    for i in range(5):
        print(i)


def test_clock_decorator():
    test_func_clock_decorator()


@apply_along_axis_decorator(axis=1)
def test_func_apply_along_axis_decorator(data_array):
    data_array = np.array(data_array)
    print("original data_array", data_array)

    # remove data <= 0.001 and np.nan
    data_array = data_array[~((data_array <= 0.001) | (np.isnan(data_array)))]
    print("data_array", data_array)

    # mean
    aggregate_value = np.mean(data_array)

    return aggregate_value


def test_apply_along_axis_decorator():
    x = np.array([np.nan, 0.001, 1, 3, 2, -1])
    print("x:", x)
    y = np.array(
        [
            [np.nan, 0.001, 1, 3, 2, -1],
            [np.nan, 0.001, 1, 1, 1, -1],
            [np.nan, 0.001, 2, 2, 2, -1],
        ]
    )
    print("y:", y)

    aggregate_value = test_func_apply_along_axis_decorator(x)
    print("aggregate_value", aggregate_value)

    aggregate_value = test_func_apply_along_axis_decorator(y)
    print("aggregate_value", aggregate_value)


def processing_step(
    step_name: str,
    save_names: Union[str, List[str]],
    data_level: str,
    deps: Optional[List[str]] = None
):
    def decorator(func: Callable):
        func._step_name = step_name
        func._step_deps = deps or []
        func._save_names = save_names
        func._data_level = data_level
        return func
    return decorator


if __name__ == "__main__":
    test_func_clock_decorator()
    test_apply_along_axis_decorator()
