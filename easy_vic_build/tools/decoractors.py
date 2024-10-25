# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import time
import functools
import numpy as np


# ---------------------------------------- clock_decorator
def clock_decorator(func):
    """ decorator, print the time elapsed (and results) for func running """
    @functools.wraps(func)
    def clocked(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        name = func.__name__
        arg_lst = []
        if args:
            arg_lst.append(','.join(repr(arg) for arg in args))
        if kwargs:
            pairs = ['%s=%r' % (k, w) for k, w in sorted(kwargs.items())]
            arg_lst.append(','.join(pairs))
        arg_str = ','.join(arg_lst)
        print('[%0.8fs] %s(%s) -> %r' % (elapsed, name, arg_str, result))
        return result
    return clocked


@clock_decorator
def test1_func():
    for i in range(5):
        print(i)

def test1():
    test1_func()

# ---------------------------------------- apply_along_axis_decorator
def apply_along_axis_decorator(axis=0):
    def decorator(func):
        """ decorator, apply func along axis in np.ndarray """
        @functools.wraps(func)
        def apply_along_axis_func(*args, **kwargs):
            result = np.apply_along_axis(func, axis, *args, **kwargs)
            return result
        
        return apply_along_axis_func
    return decorator

@apply_along_axis_decorator(axis=1)
def test2_func(data_array):
    data_array = np.array(data_array)
    print("original data_array", data_array)
    
    # remove data <= 0.001 and np.nan
    data_array = data_array[~((data_array <= 0.001) | (np.isnan(data_array)))]
    print("data_array", data_array)
    
    # mean
    aggregate_value = np.mean(data_array)
    
    return aggregate_value
    

def test2():
    x = np.array([np.nan, 0.001, 1, 3, 2, -1])
    print("x:", x)
    y = np.array([[np.nan, 0.001, 1, 3, 2, -1],
                  [np.nan, 0.001, 1, 1, 1, -1],
                  [np.nan, 0.001, 2, 2, 2, -1]])
    print("y:", y)
    
    aggregate_value = test2_func(x)
    print("aggregate_value", aggregate_value)
    
    aggregate_value = test2_func(y)
    print("aggregate_value", aggregate_value)
    

if __name__ == '__main__':
    test1()
    test2()