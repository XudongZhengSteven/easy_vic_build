# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import numpy as np


## ========================= param g =========================
""" 
g_list: global parameters
    [0]             total_depth (g)
    [1, 2]          depth (g1, g2)
    [3, 4]          b_infilt (g1, g2)
    [5, 6, 7]       ksat (g1, g2, g3)
    [8, 9, 10]      phi_s (g1, g2, g3)
    [11, 12, 13]    psis (g1, g2, g3)
    [14, 15, 16]    b_retcurve (g1, g2, g3)
    [17, 18]        expt (g1, g2)
    [19]            fc (g)
    [20]            D4 (g), it can be set as 2
    [21]            D1 (g)
    [22]            D2 (g)
    [23]            D3 (g)
    [24]            dp (g)
    [25, 26]        bubble (g1, g2)
    [27]            quartz (g)
    [28]            bulk_density (g)
    [29, 30, 31]    soil_density (g, g, g), the three g can be set same
    [32]            Wcr_FRACT (g)
    [33]            wp (g)
    [34]            Wpwp_FRACT (g)
    [35]            rough (g), it can be set as 1
    [36]            snow rough (g), it can be set as 1
"""

# g
default_g_list = [1.0,
                    2, 8,  # num1, num2
                    0.0, 1.0,
                    -0.6, 0.0126, -0.0064,
                    50.05, -0.142, -0.037,
                    1.54, -0.0095, 0.0063,
                    3.1, 0.157, -0.003,
                    3.0, 2.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    1.0,
                    1.0,
                    0.32, 4.3,
                    0.8,
                    1.0,
                    1.0, 1.0, 1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0
                    ]

g_boundary = [[0.1, 4.0],
                [1, 3], [3, 8],  # special samples for depths, here is the boundary of num1 and num2
                [-2.0, 1.0], [0.8, 1.2],
                [-0.66, -0.54], [0.0113, 0.0139], [-0.0070, -0.0058],
                [45.5, 55.5], [-0.3, -0.01], [-0.1, -0.01],
                [1.0, 2.0], [-0.01, -0.009], [0.006, 0.0066],
                [2.5, 3.6], [0.1, 0.2], [-0.005, -0.001],
                [2.8, 3.2], [1.5, 2.5],
                [0.8, 1.2],
                [1.2, 2.5],
                [1.75, 3.5],
                [1.75, 3.5],
                [0.001, 2.0],
                [0.9, 1.1],
                [0.1, 0.8], [0.0, 10.0],
                [0.7, 0.9],
                [0.9, 1.1],
                [0.9, 1.1], [0.9, 1.1], [0.9, 1.1],
                [0.8, 1.2],
                [0.8, 1.2],
                [0.8, 1.2],
                [0.9, 1.1],
                [0.9, 1.1],
                ]

g_types = [float,
          int, int,  # num1, num2
          float, float,
          float, float, float,
          float, float, float,
          float, float, float,
          float, float, float,
          float, float,
          float,
          float,
          float,
          float,
          float,
          float,
          float, float,
          float,
          float,
          float, float, float,
          float,
          float,
          float,
          float,
          float]
## ========================= param g for depths transformation =========================
# *special samples for depths
depths_index = [1, 2]  # index for depths

# *constraint: depth_layer2 > depth_layer1

# CONUS layers
CONUS_layers_depths = np.array([0.05, 0.05, 0.10, 0.10, 0.10, 0.20, 0.20, 0.20, 0.50, 0.50, 0.50])
CONUS_layers_num = len(CONUS_layers_depths)  # 11 layer
CONUS_layers_total_depth = sum(CONUS_layers_depths)  # 2.50 m
CONUS_layers_depths_percentile = CONUS_layers_depths / CONUS_layers_total_depth
CONUS_layers_depths_cumsum = np.cumsum(CONUS_layers_depths)

#* note: g for depths, g1, g2 is num1, num2, resepectively (g1, the first layer; g2, the second layer)
""" 
TF
    @staticmethod
    def depth(total_depth, g1, g2):
        # total_depth, m
        # depth, m
        # g1, g2: num1, num2, int
        # set num1 as the num of end CONUS layer num of the first layer
        # set num2 as the num of end CONUS layer num of the second layer
        # Arithmetic mean
        
        # transfer g1, g2 into percentile
        percentile_layer1, percentile_layer2 = CONUS_depth_num_to_percentile(g1, g2)
        ret = [total_depth * percentile_layer1, total_depth * percentile_layer2, total_depth * (1.0 - percentile_layer1 - percentile_layer2)]
        return ret
"""

# transfermation idea
#!index is start from 0 (0-10), num is start from 1 (1-11), num = index + 1
#!how to index: array[index0:index1+1], array[num0-1:num1]
#! i.e., you want to get layer1->layer3, use index: array[0, 2+1], use num: array[1-1:3]
#! i.e., [0, 5], [5, 8], [8, 11] represent layer1->layer5, layer5->layer8, layer9->layer10

#* three steps: CONUS depth num -> percentile -> real depths
#* step1: num->depths (this depths is constrained by CONUS_layers)
# set num1 as the num of end CONUS layer num of the first layer
# set num2 as the num of end CONUS layer num of the second layer
# transfer into depths
# first layer: layer num (0<->num1), depth_layer1 = sum(CONUS_layers_depths[:num1])
# second layer: layer num (num1+1<->num2), depth_layer2 = sum(CONUS_layers_depths[num1:num2])
#* step2: depths->percentile
# divide by the total depths
# first layer: percentile_layer1 = depth_layer1 / CONUS_layers_total_depth
# second layer: percentile_layer2 = depth_layer2 / CONUS_layers_total_depth
# third layer: percentile_layer3 = 1 - percentile_layer1 - percentile_layer2
#* step3: use TF to get real depths (this depths could be a modified value, see TF for total_depth)
# real_depth = percentile * real_total_depth (modified value)
def CONUS_depth_num_to_depth_layer(num1, num2):
    # num start from 1
    depth_layer1 = sum(CONUS_layers_depths[:num1])
    depth_layer2 = sum(CONUS_layers_depths[num1:num2])
    return depth_layer1, depth_layer2

def depth_layer_to_percentile(depth_layer1, depth_layer2):
    percentile_layer1 = depth_layer1 / CONUS_layers_total_depth
    percentile_layer2 = depth_layer2 / CONUS_layers_total_depth
    
    return percentile_layer1, percentile_layer2

def percentile_to_real_depth(real_total_depth, percentile_layer1, percentile_layer2):
    real_depth_layer1 = real_total_depth * percentile_layer1
    real_dapth_layer2 = real_total_depth * percentile_layer2
    
    return real_depth_layer1, real_dapth_layer2

def CONUS_depth_num_to_percentile(num1, num2):
    depth_layer1 = sum(CONUS_layers_depths[:num1])
    depth_layer2 = sum(CONUS_layers_depths[num1:num2])
    
    percentile_layer1 = depth_layer1 / CONUS_layers_total_depth
    percentile_layer2 = depth_layer2 / CONUS_layers_total_depth
    
    return percentile_layer1, percentile_layer2

#* reverse: real depths -> percentile -> CONUS depth num
#* step1: real depths -> percentile
# first layer: percentile_layer1 = real_depth_layer1 / total_depth (it can be a modified value, see TF for total_depth)
# second layer: percentile_layer2 = real_depth_layer2 / total_depth
#* step2: percentile -> depths
# first layer: depth_layer1 = percentile_layer1 * CONUS_layers_total_depth
# second layer: depth_layer2 = percentile_layer2 * CONUS_layers_total_depth
#* step3: depths -> CONUS depth num
# first layer: 0<->num1 = np.argmin(np.abs(CONUS_layers_depths_cumsum - depth_layer1))
# second layer: num1+1<->num2 = np.argmin(np.abs(CONUS_layers_depths_cumsum - depth_layer2))

def real_depth_to_percentile(real_total_depth, real_depth_layer1, real_dapth_layer2):
    percentile_layer1 = real_depth_layer1 / real_total_depth
    percentile_layer2 = real_dapth_layer2 / real_total_depth
    return percentile_layer1, percentile_layer2

def percentile_to_depth_layer(percentile_layer1, percentile_layer2):
    depth_layer1 = percentile_layer1 * CONUS_layers_total_depth
    depth_layer2 = percentile_layer2 * CONUS_layers_total_depth
    return depth_layer1, depth_layer2

def depth_layer_to_CONUS_depth_num(depth_layer1, depth_layer2):
    num1 = np.argmin(np.abs(CONUS_layers_depths_cumsum - depth_layer1)) + 1
    num2 = np.argmin(np.abs(CONUS_layers_depths_cumsum - depth_layer2)) + 1
    return num1, num2

def percentile_to_CONUS_depth_num(percentile_layer1, percentile_layer2):
    depth_layer1 = percentile_layer1 * CONUS_layers_total_depth
    depth_layer2 = percentile_layer2 * CONUS_layers_total_depth
    
    num1 = np.argmin(np.abs(CONUS_layers_depths_cumsum - depth_layer1)) + 1
    num2 = np.argmin(np.abs(CONUS_layers_depths_cumsum - depth_layer2)) + 1
    return num1, num2

## ========================= RVIC params =========================
# uh_params={"tp": 1.4, "mu": 5.0, "m": 3.0}
default_uh_params = [1.4, 5.0, 3.0]
uh_params_boundary = [(0, 2.5), (3.0, 7.0), (1.0, 5.0)]
uh_params_types = [float, float, float]

# cfg_params={"VELOCITY": 1.5, "DIFFUSION": 800.0}
# Lohmann, D., Nolte-Holube, R., and Raschke, E.: A large-scale horizontal routing model to be coupled to land surface parametrization schemes, Tellus A, 48, 10.3402/tellusa.v48i5.12200, 1996.
default_routing_params = [1.5, 800.0]
routing_params_boundary = [(1.0, 3.0), (200, 4000)]
routing_params_types = [float, int]