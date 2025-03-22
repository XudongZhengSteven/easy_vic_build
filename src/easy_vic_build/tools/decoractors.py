# code: utf-8
# author: "Xudong Zheng"
# email: Z786909151@163.com

"""
Module: decorators

This module provides utility decorators for enhancing function behavior.
It includes two main decorators: `clock_decorator` for measuring execution time and logging it,
and `apply_along_axis_decorator` for applying a function along a specified axis of a numpy ndarray.

Decorators:
-----------
    - clock_decorator: A decorator for measuring the execution time of a function and logging the result.
    - apply_along_axis_decorator: A decorator that applies a function along a specific axis of a numpy ndarray.

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

    # Logs the time taken to execute and prints the result

    @apply_along_axis_decorator(axis=0)
    def example_function(arr):
        # Apply a function along axis 0 of the array
        return np.sum(arr, axis=0)

Dependencies:
-------------
    - functools: For wrapping functions and ensuring they maintain their original signature.
    - numpy: For applying functions along specific axes of ndarrays.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com
"""

import functools
import time

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


if __name__ == "__main__":
    test_func_clock_decorator()
    test_apply_along_axis_decorator()
