# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: sampling

This module provides a collection of sampling methods for generating random samples 
from different distributions and techniques. It includes various sampling methods 
such as uniform sampling, Gaussian sampling, Latin Hypercube Sampling (LHS), Sobol, 
Halton, and discrete sampling. These methods allow for the generation of samples 
based on different strategies, such as uniform distribution, Gaussian distribution, 
and constrained discrete values.

Functions:
----------
    - sampling_uniform: Generates random samples from a uniform distribution.
    - sampling_uniform_int: Generates random integer samples from a uniform distribution.
    - sampling_gaussian: Generates random samples from a Gaussian (normal) distribution.
    - sampling_gaussian_clip: Generates random samples from a Gaussian distribution and clips them to a specified range.
    - sampling_LHS_1: Generates samples using the Latin Hypercube Sampling (LHS) method, method 1.
    - sampling_LHS_2: Generates samples using the Latin Hypercube Sampling (LHS) method, method 2.
    - sampling_Sobol: Generates samples using the Sobol sequence method.
    - sampling_Halton: Generates samples using the Halton sequence method.
    - sampling_discrete: Generates random samples from a set of discrete values.
    - sampling_discrete_constrained: Generates random samples from discrete values with a constraint on the sum.
    - mixed_sampling: Placeholder function for a mixed sampling strategy (not implemented).
    - sampling_CONUS_depth_num: Generates samples for depth numbers based on specified layer ranges, with constraints.

Usage:
------
    1. Choose the desired sampling method based on the type of distribution or technique required.
    2. Call the corresponding function with the necessary parameters to generate the samples:
        - `sampling_uniform(n_samples, bounds)` for uniform sampling.
        - `sampling_LHS_1(n_samples, n_dimensions, bounds)` for Latin Hypercube Sampling (LHS) method 1.
        - `sampling_Sobol(n_samples, n_dimensions, bounds)` for Sobol sampling.
        - `sampling_discrete(discrete_values, n_samples)` for discrete value sampling.
    3. The generated samples can be used for further analysis or as input for simulations.

Example:
--------
    uniform_samples = sampling_uniform(100, (0, 10))
    gaussian_samples = sampling_gaussian(100, 0, 1)

Dependencies:
-------------
    - numpy: For generating random numbers and array manipulation.
    - scipy: For specialized sampling techniques such as Sobol and Halton sequences.
    
Author:
-------
    Xudong Zheng
    Email: z786909151@163.com
"""

import numpy as np
import random
from scipy.stats import qmc
from ..params_func.params_set import *

def sampling_uniform(n_samples, bounds):
    """
    Generate random samples from a uniform distribution within the specified bounds.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    bounds : tuple of float
        The lower and upper bounds for the uniform distribution.

    Returns
    -------
    samples : list of float
        The generated random samples.
    """
    samples = [random.uniform(bounds[0], bounds[1]) for _ in range(n_samples)]
    return samples


def sampling_uniform_int(n_samples, bounds):
    """
    Generate random integer samples from a uniform distribution within the specified bounds.

    Parameters
    ----------
    n_samples : int
        The number of integer samples to generate.
    bounds : tuple of int
        The lower and upper bounds for the uniform distribution.

    Returns
    -------
    samples : list of int
        The generated random integer samples.
    """
    samples = [random.randint(bounds[0], bounds[1]) for _ in range(n_samples)]
    return samples


def sampling_gaussian(n_samples, mean, std):
    """
    Generate random samples from a Gaussian (normal) distribution with specified mean and standard deviation.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    mean : float
        The mean of the Gaussian distribution.
    std : float
        The standard deviation of the Gaussian distribution.

    Returns
    -------
    samples : list of float
        The generated random samples.
    """
    samples = [random.gauss(mean, std) for _ in range(n_samples)]
    return samples


def sampling_gaussian_clip(n_samples, mean, std, low=None, up=None):
    """
    Generate random samples from a Gaussian distribution with specified mean and standard deviation,
    and clip the samples to the specified range [low, up].

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    mean : float
        The mean of the Gaussian distribution.
    std : float
        The standard deviation of the Gaussian distribution.
    low : float, optional
        The lower bound for clipping the samples.
    up : float, optional
        The upper bound for clipping the samples.

    Returns
    -------
    samples : numpy.ndarray
        The generated random samples, clipped to the specified range.
    """
    samples = np.random.normal(loc=mean, scale=std, size=n_samples)
    
    if low is not None or up is not None:
        samples = np.clip(samples, low, up)
    
    return samples


def sampling_LHS_1(n_samples, n_dimensions, bounds):
    """
    Generate random samples using Latin Hypercube Sampling (LHS) method, variant 1,
    within the specified bounds.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    n_dimensions : int
        The number of dimensions for each sample.
    bounds : list of tuple
        A list of tuples specifying the lower and upper bounds for each dimension.

    Returns
    -------
    samples : numpy.ndarray
        The generated Latin Hypercube samples.
    """
    # i.e., bounds = [(0, 10), (-5, 5), (100, 200)]
    samples = np.zeros((n_samples, n_dimensions))
    
    for i in range(n_dimensions):
        # generate data between 0~1
        intervals = np.linspace(0, 1, n_samples + 1)
        points = np.random.uniform(intervals[:-1], intervals[1:])
        np.random.shuffle(points)
        
        # remapping to bounds
        min_val, max_val = bounds[i]
        samples[:, i] = points * (max_val - min_val) + min_val
    
    return samples


def sampling_LHS_2(n_samples, n_dimensions, bounds):
    """
    Generate random samples using Latin Hypercube Sampling (LHS) method, variant 2,
    within the specified bounds.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    n_dimensions : int
        The number of dimensions for each sample.
    bounds : list of tuple
        A list of tuples specifying the lower and upper bounds for each dimension.

    Returns
    -------
    population : numpy.ndarray
        The generated Latin Hypercube samples, scaled to the specified bounds.
    """
    # i.e., bounds = [(0, 1), (5, 10), (-5, 5)]
    sampler = qmc.LatinHypercube(d=n_dimensions)
    sample = sampler.random(n=n_samples)
    
    # remapping
    lower_bounds, upper_bounds = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
    population = qmc.scale(sample, lower_bounds, upper_bounds)
    
    return population


def sampling_Sobol(n_samples, n_dimensions, bounds):
    """
    Generate random samples using the Sobol sequence method within the specified bounds.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    n_dimensions : int
        The number of dimensions for each sample.
    bounds : list of tuple
        A list of tuples specifying the lower and upper bounds for each dimension.

    Returns
    -------
    scaled_samples : numpy.ndarray
        The generated Sobol samples, scaled to the specified bounds.
    """
    sobol_sampler = qmc.Sobol(d=n_dimensions, scramble=True)
    samples = sobol_sampler.random(n=n_samples)
    
    # get bounds
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    
    # remapping
    scaled_samples = qmc.scale(samples, lower_bounds, upper_bounds)

    return scaled_samples
    
    
def sampling_Halton(n_samples, n_dimensions, bounds):
    """
    Generate random samples using the Halton sequence method within the specified bounds.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    n_dimensions : int
        The number of dimensions for each sample.
    bounds : list of tuple
        A list of tuples specifying the lower and upper bounds for each dimension.

    Returns
    -------
    scaled_samples : numpy.ndarray
        The generated Halton samples, scaled to the specified bounds.
    """
    halton_sampler = qmc.Halton(d=n_dimensions, scramble=True)
    samples = halton_sampler.random(n=n_samples)
    
    # get bounds
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    
    # remapping
    scaled_samples = qmc.scale(samples, lower_bounds, upper_bounds)
    
    return scaled_samples


def sampling_discrete(discrete_values, n_samples, weights=None):
    """
    Generate random samples from a set of discrete values, optionally with weights.

    Parameters
    ----------
    discrete_values : array-like
        A list or array of discrete values to sample from.
    n_samples : int
        The number of samples to generate.
    weights : array-like, optional
        The weights associated with the discrete values. If None, the values are assumed to be equally likely.

    Returns
    -------
    samples : numpy.ndarray
        The generated discrete samples.
    """
    if weights is None:
        samples = np.random.choice(discrete_values, size=n_samples)
    else:
        samples = np.random.choice(discrete_values, size=n_samples, p=weights)

    return samples
    

def sampling_discrete_constrained(discrete_values, target_sum, n_samples):
    """
    Generate random samples from discrete values, with the constraint that the sum of the samples equals target_sum.

    Parameters
    ----------
    discrete_values : array-like
        A list or array of discrete values to sample from.
    target_sum : int
        The target sum for the generated samples.
    n_samples : int
        The number of samples to generate.

    Returns
    -------
    samples : numpy.ndarray
        The generated discrete samples with the constraint on their sum.
    """
    # i.e., discrete_values = np.array([0, 1, 2]), target_sum = 2, n_samples = 10
    samples = []
    for _ in range(n_samples):
        sample = np.random.multinomial(target_sum, [1/len(discrete_values)] * len(discrete_values))
        samples.append(sample)

    return np.array(samples)


def mixed_sampling(n_samples):
    """
    Placeholder for a mixed sampling method (currently unimplemented).

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.

    Returns
    -------
    None
    """
    pass


def sampling_CONUS_depth_num(n_samples, layer_ranges):
    """
    Generate random samples for depth numbers based on the specified layer ranges,
    with a constraint on layer depths.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    layer_ranges : list of tuple
        A list of tuples specifying the range of values for each layer.

    Returns
    -------
    samples : list of tuple
        The generated samples for layer numbers.
    """
    # layer_ranges = [(1, 3), (3, 8)], this is num, start from 1 (1~11)
    samples = []
    for _ in range(n_samples):
        # first layer: sample for num1
        num1 = np.random.choice(range(layer_ranges[0][0], layer_ranges[0][1] + 1))
        
        # second layer: sample for num2
        num2 = np.random.choice(range(num1 + 1, layer_ranges[1][1] + 1))
        
        # constraint: depth_layer2 > depth_layer1
        depth_layer1, depth_layer2 = CONUS_depth_num_to_depth_layer(num1, num2)
        while True:
            if depth_layer1 < depth_layer2:
                break
            else:
                num2 = np.random.choice(range(num1 + 1, layer_ranges[1][1] + 1))
                depth_layer1, depth_layer2 = CONUS_depth_num_to_depth_layer(num1, num2)
        
        samples.append((num1, num2))
    
    return samples