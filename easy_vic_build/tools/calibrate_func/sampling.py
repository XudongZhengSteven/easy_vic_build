# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import numpy as np
import random
import scipy
from scipy.stats import qmc

def sampling_uniform(n_samples, bounds):
    samples = [random.uniform(bounds[0], bounds[1]) for _ in range(n_samples)]
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


def sampling_COUNU_Soil(n_samples, soil_values=None, layer_ranges=None):
    # set soil values
    default_soil_values = np.array([0.05, 0.05, 0.10, 0.10, 0.10, 0.20, 0.20, 0.20, 0.50, 0.50, 0.50])
    soil_values = soil_values if soil_values is not None else default_soil_values
    total_depth = sum(soil_values)
    
    # set layer ranges
    default_layer_ranges = [(1, 3), (3, 8), (8, 11)]  # TODO input depths rather than layers
    layer_ranges = layer_ranges if layer_ranges is not None else default_layer_ranges
    
    samples = []
    for _ in range(n_samples):
        # first layer
        first_layer_indices = np.random.choice(range(layer_ranges[0][0], layer_ranges[0][1]), 
                                               size=np.random.randint(1, layer_ranges[0][1] - layer_ranges[0][0] + 1), 
                                               replace=False)
        
        # second layer
        if first_layer_indices.size > 0:
            second_layer_start = first_layer_indices[-1] + 1
            second_layer_end = layer_ranges[1][1]
            second_layer_indices = np.random.choice(range(second_layer_start, second_layer_end), 
                                                    size=np.random.randint(1, second_layer_end - second_layer_start + 1), 
                                                    replace=False)
        else:
            second_layer_indices = np.array([])
        
        # third layer
        remaining_indices = set(range(len(soil_values))) - set(first_layer_indices) - set(second_layer_indices)
        
        # combine
        layer_sample = {
            'Layer1 percentile': sum(soil_values[first_layer_indices]) / total_depth,
            'Layer2 percentile': sum(soil_values[second_layer_indices]) / total_depth,
            'Layer3 percentile': sum(soil_values[list(remaining_indices)])/ total_depth,
        }
        
        samples.append(layer_sample)
    
    return samples