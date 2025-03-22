# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: Scaling_operator

This module provides various statistical scaling operators for data manipulation. The operators 
defined within this module include methods for calculating common statistical values such as 
the harmonic mean, arithmetic mean, geometric mean, and maximum difference. Additionally, the 
module includes a method to compute the majority value from a dataset, useful for identifying 
the most frequent value.

Class:
------
    - Scaling_operator: A class that contains static methods for different statistical scaling 
      operations, such as mean calculations and majority determination.

Functions:
----------
    - multiply: Multiplies two values together (helper function used in geometric mean calculation).

Class Methods:
--------------
    - Harmonic_mean: Computes the harmonic mean of a given dataset.
    - Arithmetic_mean: Computes the arithmetic mean of a given dataset.
    - Geometric_mean: Computes the geometric mean of a given dataset.
    - Maximum_difference: Computes the difference between the maximum and minimum values of a dataset.
    - Majority: Determines the majority (most frequent) value in a dataset.

Dependencies:
-------------
    - numpy: Used for array manipulation and mathematical operations.
    - collections.Counter: Used to count occurrences of each element in the dataset.
    - functools.reduce: Used for performing the reduction operation in the geometric mean calculation.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com
"""


import numpy as np
from collections import Counter
from functools import reduce


def multiply(x, y):
    """
    Multiplies two values together.

    Parameters:
    -----------
    x : float
        The first value to multiply.
    y : float
        The second value to multiply.

    Returns:
    --------
    float
        The product of the two values.
    """
    return x * y


class Scaling_operator:
    """
    A class that provides various statistical scaling operations for a dataset.
    
    Methods:
    --------
    Harmonic_mean(data):
        Computes the harmonic mean of the dataset.
    
    Arithmetic_mean(data):
        Computes the arithmetic mean of the dataset.
    
    Geometric_mean(data):
        Computes the geometric mean of the dataset.
    
    Maximum_difference(data):
        Computes the difference between the maximum and minimum values in the dataset.
    
    Majority(data):
        Computes the most frequent value (majority) in the dataset.
    """
    #TODO nonlinear scaling operator

    @staticmethod
    def Harmonic_mean(data):
        """
        Computes the harmonic mean of a dataset.

        The harmonic mean is defined as the reciprocal of the arithmetic mean of the reciprocals 
        of the data points.

        Parameters:
        -----------
        data : array-like
            A dataset (list or array) of numerical values.

        Returns:
        --------
        float
            The harmonic mean of the dataset.
        """
        data = np.array(data)
        return len(data) / np.nansum(1/data)
    
    @staticmethod
    def Arithmetic_mean(data):
        """
        Computes the arithmetic mean of a dataset.

        The arithmetic mean is defined as the sum of all data points divided by the number of 
        data points.

        Parameters:
        -----------
        data : array-like
            A dataset (list or array) of numerical values.

        Returns:
        --------
        float
            The arithmetic mean of the dataset.
        """
        data = np.array(data)
        return np.nanmean(data)

    @staticmethod
    def Geometric_mean(data):
        """
        Computes the geometric mean of a dataset.

        The geometric mean is the nth root of the product of all data points, where n is the 
        number of data points.

        Parameters:
        -----------
        data : array-like
            A dataset (list or array) of numerical values.

        Returns:
        --------
        float
            The geometric mean of the dataset.
        """
        data = np.array(data)
        return pow(reduce(multiply, data), 1/len(data))
    
    @staticmethod
    def Maximum_difference(data):
        """
        Computes the difference between the maximum and minimum values in the dataset.

        Parameters:
        -----------
        data : array-like
            A dataset (list or array) of numerical values.

        Returns:
        --------
        float
            The difference between the maximum and minimum values in the dataset.
        """
        data = np.array(data)
        return np.nanmax(data) - np.nanmin(data)
    
    @staticmethod
    def Majority(data):
        """
        Computes the most frequent (majority) value in the dataset.

        Parameters:
        -----------
        data : array-like
            A dataset (list or array) of values, where the most frequent value is sought.

        Returns:
        --------
        The most frequent value in the dataset.
        """
        data = np.array(data)
        counter = Counter(data)
        return max(counter.keys(), key=counter.get)
    

if __name__ == "__main__":
    x = np.array([2, 3])
    so = Scaling_operator()
    so.Arithmetic_mean(x)
    so.Geometric_mean(x)
    so.Harmonic_mean(x)