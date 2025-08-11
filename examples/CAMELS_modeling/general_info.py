# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
"""
general information:

basin set
106(10_100_km_humid); 240(10_100_km_semi_humid); 648(10_100_km_semi_arid); 
213(100_1000_km_humid); 38(100_1000_km_semi_humid); 670(10_100_km_semi_arid);
397(1000_larger_km_humid); 636(1000_larger_km_semi_humid); 580(1000_larger_km_semi_arid) 

grid_res_level0=1km(0.00833)
grid_res_level1=3km(0.025), 6km(0.055), 8km(0.072), 12km(0.11)

"""

basin_index = 397

grid_res_level0=0.00833

model_scale = "12km"
scalemap = {"3km": 0.025, "6km": 0.055, "8km": 0.072, "12km": 0.11}
grid_res_level1 = scalemap[model_scale]

grid_res_level2 = grid_res_level1

reverse_lat=True
date_period = ["19980101 00:00:00", "20101231 23:00:00"]
timestep = "H"
