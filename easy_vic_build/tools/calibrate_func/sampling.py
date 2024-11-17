# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import numpy as np
import random
import scipy
from scipy.stats import qmc
from ..params_func.params_set import *

def sampling_uniform(n_samples, bounds):
    samples = [random.uniform(bounds[0], bounds[1]) for _ in range(n_samples)]
    return samples


def sampling_uniform_int(n_samples, bounds):
    samples = [random.randint(bounds[0], bounds[1]) for _ in range(n_samples)]
    return samples


def sampling_gaussian(n_samples, mean, std):
    samples = [random.gauss(mean, std) for _ in range(n_samples)]
    return samples


def sampling_gaussian_clip(n_samples, mean, std, low=None, up=None):
    samples = np.random.normal(loc=mean, scale=std, size=n_samples)
    
    if low is not None or up is not None:
        samples = np.clip(samples, low, up)
    
    return samples


def sampling_LHS_1(n_samples, n_dimensions, bounds):
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
    # i.e., bounds = [(0, 1), (5, 10), (-5, 5)]
    sampler = qmc.LatinHypercube(d=n_dimensions)
    sample = sampler.random(n=n_samples)
    
    # remapping
    lower_bounds, upper_bounds = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
    population = qmc.scale(sample, lower_bounds, upper_bounds)
    
    return population


def sampling_Sobol(n_samples, n_dimensions, bounds):
    sobol_sampler = qmc.Sobol(d=n_dimensions, scramble=True)
    samples = sobol_sampler.random(n=n_samples)
    
    # get bounds
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    
    # remapping
    scaled_samples = qmc.scale(samples, lower_bounds, upper_bounds)

    return scaled_samples
    
    
def sampling_Halton(n_samples, n_dimensions, bounds):
    halton_sampler = qmc.Halton(d=n_dimensions, scramble=True)
    samples = halton_sampler.random(n=n_samples)
    
    # get bounds
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    
    # remapping
    scaled_samples = qmc.scale(samples, lower_bounds, upper_bounds)
    
    return scaled_samples


def sampling_discrete(discrete_values, n_samples, weights=None):
    if weights is None:
        samples = np.random.choice(discrete_values, size=n_samples)
    else:
        samples = np.random.choice(discrete_values, size=n_samples, p=weights)

    return samples
    

def sampling_discrete_constrained(discrete_values, target_sum, n_samples):
    # i.e., discrete_values = np.array([0, 1, 2]), target_sum = 2, n_samples = 10
    samples = []
    for _ in range(n_samples):
        sample = np.random.multinomial(target_sum, [1/len(discrete_values)] * len(discrete_values))
        samples.append(sample)

    return np.array(samples)


def mixed_sampling(n_samples):
    pass


def sampling_CONUS_depth_num(n_samples, layer_ranges):
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