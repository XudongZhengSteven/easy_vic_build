# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

## ========================= param g =========================
""" 
g_list: global parameters
    [0]             total_depth (g)
    [1, 2]          depth (g1, g2, 1-g1-g2)
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
# *special samples for depths
# CONUS_layers_depths = np.array([0.05, 0.05, 0.10, 0.10, 0.10, 0.20, 0.20, 0.20, 0.50, 0.50, 0.50])  # 11 layers, m
# CONUS_layers_total_depth = sum(CONUS_layers_depths)  # 2.50 m
# CONUS_layers_depths_percentile = CONUS_layers_depths / CONUS_layers_total_depth
# it is percentile of total_depth
# def depth(total_depth, g1, g2):
#     # total_depth, m
#     # depth, m
#     # g1, g2, g3: depend on the percentile of total depths
#     # Arithmetic mean
#     ret = [total_depth * g1, total_depth * g2, total_depth * (1.0 - g1 - g2)]
#     return ret
# g
default_g_list = [1.0,
                    0.1/2.5, 0.9/2.5,  # percentile of total depths
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
                    0.32, 4.2,
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
                [0, 3], [3, 8],  # special samples for depths, here is original layer number
                [-2.0, 1.0], [0.8, 1.2],
                [-0.66, -0.54], [0.0113, 0.0139], [-0.0058, -0.0070],
                [45.5, 55.5], [-0.8, -0.4], [-0.8, -0.4],
                [0.8, 0.4], [-0.8, -0.4], [0.8, 0.4],
                [-0.8, -0.4], [-0.8, -0.4], [-0.8, -0.4],
                [0.8, 1.2], [0.8, 1.2],
                [0.8, 1.2],
                [1.2, 2.5],
                [1.75, 3.5],
                [1.75, 3.5],
                [0.001, 2.0],
                [0.9, 1.1],
                [0.8, 1.2], [0.8, 1.2],
                [0.7, 0.9],
                [0.9, 1.1],
                [0.9, 1.1], [0.9, 1.1], [0.9, 1.1],
                [0.8, 1.2],
                [0.8, 1.2],
                [0.8, 1.2],
                [0.9, 1.1],
                [0.9, 1.1],
                ]

## ========================= RVIC params =========================
# uh_params={"tp": 1.4, "mu": 5.0, "m": 3.0}
default_uh_params = [1.4, 5.0, 3.0]
uh_params_boundary = [(0, 2.5), (3, 7), (1, 5)]

# cfg_params={"VELOCITY": 1.5, "DIFFUSION": 800.0}
default_routing_params = [1.5, 800.0]
routing_params_boundary = [(1.0, 3.0), (500, 1000)]
