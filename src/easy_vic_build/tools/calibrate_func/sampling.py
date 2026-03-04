# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Sampling utilities for calibration and parameter exploration.

This module provides commonly used random and quasi-random sampling routines:

- uniform and Gaussian sampling,
- Latin Hypercube Sampling (LHS),
- Sobol and Halton low-discrepancy sequences,
- discrete sampling with optional constraints.

"""

import random

import numpy as np
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


def sampling_LHS_2(n_samples, bounds, seed=None):
    """
    Generate random samples using Latin Hypercube Sampling (LHS) method, variant 2,
    within the specified bounds.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    bounds : list of tuple
        A list of tuples specifying the lower and upper bounds for each dimension.
    seed : int, optional
        Seed used by ``scipy.stats.qmc.LatinHypercube``.

    Returns
    -------
    numpy.ndarray
        The generated Latin Hypercube samples, scaled to the specified bounds.

    Raises
    ------
    ValueError
        If any bound does not satisfy ``min < max``.
    """
    # i.e., bounds = [(0, 1), (5, 10), (-5, 5)]
    n_dimensions = len(bounds)
    
    # check bounds
    if any(b[0] >= b[1] for b in bounds):
        raise ValueError("Each bound must satisfy min < max.")
    
    # sample
    sampler = qmc.LatinHypercube(d=n_dimensions, seed=seed)
    sample = sampler.random(n=n_samples)

    # remapping
    lower_bounds, upper_bounds = np.array([b[0] for b in bounds]), np.array(
        [b[1] for b in bounds]
    )
    scaled_samples = qmc.scale(sample, lower_bounds, upper_bounds)

    # clip boundary
    scaled_samples = np.clip(scaled_samples, lower_bounds, upper_bounds)
    
    return scaled_samples


def sampling_Sobol(n_samples, bounds):
    """
    Generate random samples using the Sobol sequence method within the specified bounds.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    bounds : list of tuple
        A list of tuples specifying the lower and upper bounds for each dimension.

    Returns
    -------
    numpy.ndarray
        The generated Sobol samples, scaled to the specified bounds.
    """
    n_dimensions = len(bounds)
    sobol_sampler = qmc.Sobol(d=n_dimensions, scramble=True)
    samples = sobol_sampler.random(n=n_samples)

    # get bounds
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])

    # remapping
    scaled_samples = qmc.scale(samples, lower_bounds, upper_bounds)

    return scaled_samples


def sampling_Halton(n_samples, bounds):
    """
    Generate random samples using the Halton sequence method within the specified bounds.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    bounds : list of tuple
        A list of tuples specifying the lower and upper bounds for each dimension.

    Returns
    -------
    numpy.ndarray
        The generated Halton samples, scaled to the specified bounds.
    """
    n_dimensions = len(bounds)
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
        sample = np.random.multinomial(
            target_sum, [1 / len(discrete_values)] * len(discrete_values)
        )
        samples.append(sample)

    return np.array(samples)


def mixed_sampling(n_samples):
    """
    Reserved interface for mixed sampling strategies.

    Parameters
    ----------
    n_samples : int
        Requested number of samples.

    Returns
    -------
    None
        This function currently has no implementation.

    Notes
    -----
    This function is intentionally left as a placeholder for future extension.
    """
    pass


