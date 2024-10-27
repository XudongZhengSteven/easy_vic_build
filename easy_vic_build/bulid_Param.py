# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import os
import numpy as np
import json
from .tools.utilities import createArray_from_gridshp, grids_array_coord_map
from .tools.utilities import cal_ssc_percentile_grid_array, cal_bd_grid_array
from .tools.params_func.createParametersDataset import createParametersDataset
from .tools.params_func.TansferFunction import TF_VIC
from .tools.params_func.Scaling_operator import Scaling_operator
from .bulid_Domain import cal_mask_frac_area_length
from tqdm import *
from .tools.geo_func import resample, search_grids


def get_default_g_list():
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
    
    default_g_list = [1.0,
                      0.04, 0.36,
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
                  [0, 0], [0, 0],  # special samples for depths
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
    return default_g_list, g_boundary


def buildParam_level0(g_list, dpc_VIC_level0, evb_dir, reverse_lat=True):
    """ 
    # calibrate: MPR: PTF + Scaling (calibrate for scaling coefficient)
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
        
    # TODO Q1: different layer have different global params? Ksat: 3 or 9?
    """
    print("building Param_level0... ...")
    ## ====================== set dir and path ======================
    # set path
    params_dataset_level0_path = os.path.join(evb_dir.ParamFile_dir, "params_dataset_level0.nc")
    
    ## ====================== get grid_shp and basin_shp ======================
    grid_shp_level0 = dpc_VIC_level0.grid_shp
    basin_shp = dpc_VIC_level0.basin_shp
    
    # grids_map_array
    lon_list_level0, lat_list_level0, lon_map_index_level0, lat_map_index_level0 = grids_array_coord_map(grid_shp_level0, reverse_lat=reverse_lat)  #* all lat set as reverse if True
    
    ## ====================== create parameter ======================
    params_dataset_level0 = createParametersDataset(params_dataset_level0_path, lat_list_level0, lon_list_level0)
    tf_VIC = TF_VIC()
    
    ## ===================== level0: assign values for general variables  ======================
    # dimension variables: lat, lon, nlayer, root_zone, veg_class, month
    params_dataset_level0.variables["lat"][:] = np.array(lat_list_level0)  # 1D array
    params_dataset_level0.variables["lon"][:] = np.array(lon_list_level0)  # 1D array
    params_dataset_level0.variables["nlayer"][:] = [1, 2, 3]
    root_zone_list = [1, 2, 3]
    params_dataset_level0.variables["root_zone"][:] = root_zone_list
    veg_class_list = list(range(14))
    params_dataset_level0.variables["veg_class"][:] = veg_class_list
    month_list = list(range(1, 13))
    params_dataset_level0.variables["month"][:] = month_list
    
    # lons, lats, 2D array
    grid_array_lons, grid_array_lats = np.meshgrid(params_dataset_level0.variables["lon"][:], params_dataset_level0.variables["lat"][:])  # 2D array
    params_dataset_level0.variables["lons"][:, :] = grid_array_lons
    params_dataset_level0.variables["lats"][:, :] = grid_array_lats
    
    ## ======================= level0: Transfer function =======================
    # only set the params which should be scaling (aggregation), other params such as run_cell, grid_cell, off_gmt..., will not be set here
    # depth, m
    CONUS_layers_depths = np.array([0.05, 0.05, 0.10, 0.10, 0.10, 0.20, 0.20, 0.20, 0.50, 0.50, 0.50])  # 11 layers, m
    CONUS_layers_total_depth = sum(CONUS_layers_depths)  # 2.50 m
    CONUS_layers_depths_percentile = CONUS_layers_depths / CONUS_layers_total_depth
    
    total_depth = tf_VIC.total_depth(CONUS_layers_total_depth, g_list[0])
    CONUS_layers_depths = CONUS_layers_depths * g_list[0]
    
    depths = tf_VIC.depth(total_depth, g_list[1], g_list[2])
    grid_shp_level0["depth_layer1"] = np.full((len(grid_shp_level0.index), ), fill_value=depths[0])
    grid_shp_level0["depth_layer2"] = np.full((len(grid_shp_level0.index), ), fill_value=depths[1])
    grid_shp_level0["depth_layer3"] = np.full((len(grid_shp_level0.index), ), fill_value=depths[2])
    
    grid_array_depth_layer1, _, _ = createArray_from_gridshp(grid_shp_level0, value_column="depth_layer1", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
    grid_array_depth_layer2, _, _ = createArray_from_gridshp(grid_shp_level0, value_column="depth_layer2", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
    grid_array_depth_layer3, _, _ = createArray_from_gridshp(grid_shp_level0, value_column="depth_layer3", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
    
    params_dataset_level0.variables["depth"][0, :, :] = grid_array_depth_layer1
    params_dataset_level0.variables["depth"][1, :, :] = grid_array_depth_layer2
    params_dataset_level0.variables["depth"][2, :, :] = grid_array_depth_layer3
    
    # VIC_depth(1, 2, 3) -> CONUS_layers_depth(0, ..., 10), index, #*vertical aggregation for three soil layers
    CONUS_layers_depths_cumsum = np.cumsum(CONUS_layers_depths)
    
    depth_layer1_start = 0
    depth_layer1_end = np.where(abs(CONUS_layers_depths_cumsum - depths[0]) <= 0.001)[0][0]
    
    depth_layer2_start = depth_layer1_end + 1
    CONUS_layers_depths_cumsum -= CONUS_layers_depths_cumsum[depth_layer1_end]
    depth_layer2_end = np.where(abs(CONUS_layers_depths_cumsum - depths[1]) <= 0.001)[0][0]

    depth_layer3_start = depth_layer2_end + 1
    CONUS_layers_depths_cumsum -= CONUS_layers_depths_cumsum[depth_layer2_end]
    depth_layer3_end = np.where(abs(CONUS_layers_depths_cumsum - depths[2]) <= 0.001)[0][0]
    
    # ele_std, m (same as StrmDem)
    grid_array_SrtmDEM_std_Value, _, _ = createArray_from_gridshp(grid_shp_level0, value_column="SrtmDEM_std_Value", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
    grid_array_SrtmDEM_std_Value, _, _ = createArray_from_gridshp(grid_shp_level0, value_column="SrtmDEM_std_Value", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
    
    # b_infilt, N/A
    params_dataset_level0.variables["infilt"][:, :] = tf_VIC.b_infilt(grid_array_SrtmDEM_std_Value, g_list[3], g_list[4])
    
    # sand, clay, silt, %
    grid_array_sand_layer1, grid_array_silt_layer1, grid_array_clay_layer1 = cal_ssc_percentile_grid_array(grid_shp_level0, depth_layer1_start, depth_layer1_end)
    grid_array_sand_layer2, grid_array_silt_layer2, grid_array_clay_layer2 = cal_ssc_percentile_grid_array(grid_shp_level0, depth_layer2_start, depth_layer2_end)
    grid_array_sand_layer3, grid_array_silt_layer3, grid_array_clay_layer3 = cal_ssc_percentile_grid_array(grid_shp_level0, depth_layer3_start, depth_layer3_end)
    
    # ksat, mm/s -> mm/day (VIC requirement)
    grid_array_ksat_layer1 = tf_VIC.ksat(grid_array_sand_layer1, grid_array_clay_layer1, g_list[5], g_list[6], g_list[7])
    grid_array_ksat_layer2 = tf_VIC.ksat(grid_array_sand_layer2, grid_array_clay_layer2, g_list[5], g_list[6], g_list[7])
    grid_array_ksat_layer3 = tf_VIC.ksat(grid_array_sand_layer3, grid_array_clay_layer3, g_list[5], g_list[6], g_list[7])
    
    unit_factor_ksat = 60 * 60 * 24
    
    params_dataset_level0.variables["Ksat"][0, :, :] = grid_array_ksat_layer1 * unit_factor_ksat
    params_dataset_level0.variables["Ksat"][1, :, :] = grid_array_ksat_layer2 * unit_factor_ksat
    params_dataset_level0.variables["Ksat"][2, :, :] = grid_array_ksat_layer3 * unit_factor_ksat
    
    # mean slope, % (m/m)
    grid_array_mean_slope, _, _ = createArray_from_gridshp(grid_shp_level0, value_column="SrtmDEM_mean_slope_Value%", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
    
    # phi_s, m3/m3 or mm/mm
    grid_array_phi_s_layer1 = tf_VIC.phi_s(grid_array_sand_layer1, grid_array_clay_layer1, g_list[8], g_list[9], g_list[10])
    grid_array_phi_s_layer2 = tf_VIC.phi_s(grid_array_sand_layer2, grid_array_clay_layer2, g_list[8], g_list[9], g_list[10])
    grid_array_phi_s_layer3 = tf_VIC.phi_s(grid_array_sand_layer3, grid_array_clay_layer3, g_list[8], g_list[9], g_list[10])
    
    params_dataset_level0.variables["phi_s"][0, :, :] = grid_array_phi_s_layer1
    params_dataset_level0.variables["phi_s"][1, :, :] = grid_array_phi_s_layer2
    params_dataset_level0.variables["phi_s"][2, :, :] = grid_array_phi_s_layer3
    
    # psis, kPa/cm-H2O
    grid_array_psis_layer1 = tf_VIC.psis(grid_array_sand_layer1, grid_array_silt_layer1, g_list[11], g_list[12], g_list[13])
    grid_array_psis_layer2 = tf_VIC.psis(grid_array_sand_layer2, grid_array_silt_layer2, g_list[11], g_list[12], g_list[13])
    grid_array_psis_layer3 = tf_VIC.psis(grid_array_sand_layer3, grid_array_silt_layer3, g_list[11], g_list[12], g_list[13])
    
    params_dataset_level0.variables["psis"][0, :, :] = grid_array_psis_layer1
    params_dataset_level0.variables["psis"][1, :, :] = grid_array_psis_layer2
    params_dataset_level0.variables["psis"][2, :, :] = grid_array_psis_layer3
    
    # b_retcurve, N/A
    grid_array_b_retcurve_layer1 = tf_VIC.b_retcurve(grid_array_sand_layer1, grid_array_clay_layer1, g_list[14], g_list[15], g_list[16])
    grid_array_b_retcurve_layer2 = tf_VIC.b_retcurve(grid_array_sand_layer2, grid_array_clay_layer2, g_list[14], g_list[15], g_list[16])
    grid_array_b_retcurve_layer3 = tf_VIC.b_retcurve(grid_array_sand_layer3, grid_array_clay_layer3, g_list[14], g_list[15], g_list[16])
    
    params_dataset_level0.variables["b_retcurve"][0, :, :] = grid_array_b_retcurve_layer1
    params_dataset_level0.variables["b_retcurve"][1, :, :] = grid_array_b_retcurve_layer2
    params_dataset_level0.variables["b_retcurve"][2, :, :] = grid_array_b_retcurve_layer3
    
    # expt, N/A
    grid_array_expt_layer1 = tf_VIC.expt(grid_array_b_retcurve_layer1, g_list[17], g_list[18])
    grid_array_expt_layer2 = tf_VIC.expt(grid_array_b_retcurve_layer2, g_list[17], g_list[18])
    grid_array_expt_layer3 = tf_VIC.expt(grid_array_b_retcurve_layer3, g_list[17], g_list[18])
    
    params_dataset_level0.variables["expt"][0, :, :] = grid_array_expt_layer1
    params_dataset_level0.variables["expt"][1, :, :] = grid_array_expt_layer2
    params_dataset_level0.variables["expt"][2, :, :] = grid_array_expt_layer3
    
    # fc, % or m3/m3
    grid_array_fc_layer1 = tf_VIC.fc(grid_array_phi_s_layer1, grid_array_b_retcurve_layer1, grid_array_psis_layer1, grid_array_sand_layer1, g_list[19])
    grid_array_fc_layer2 = tf_VIC.fc(grid_array_phi_s_layer2, grid_array_b_retcurve_layer2, grid_array_psis_layer2, grid_array_sand_layer2, g_list[19])
    grid_array_fc_layer3 = tf_VIC.fc(grid_array_phi_s_layer3, grid_array_b_retcurve_layer3, grid_array_psis_layer3, grid_array_sand_layer3, g_list[19])
    
    params_dataset_level0.variables["fc"][0, :, :] = grid_array_fc_layer1
    params_dataset_level0.variables["fc"][1, :, :] = grid_array_fc_layer2
    params_dataset_level0.variables["fc"][2, :, :] = grid_array_fc_layer3
    
    # D4, N/A, same as c, typically is 2
    grid_shp_level0["D4"] = np.full((len(grid_shp_level0.index), ), fill_value=tf_VIC.D4(g_list[20]))
    grid_array_D4, _, _ = createArray_from_gridshp(grid_shp_level0, value_column="D4", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
    params_dataset_level0.variables["D4"][:, :] = grid_array_D4
    
    # cexpt
    grid_array_cexpt = grid_array_D4
    params_dataset_level0.variables["c"][:, :] = grid_array_cexpt
    
    # D1 ([day^-1]), D2 ([day^-D4]), D3 ([mm])
    grid_array_D1 = tf_VIC.D1(grid_array_ksat_layer3, grid_array_mean_slope, g_list[21])
    grid_array_D2 = tf_VIC.D2(grid_array_ksat_layer3, grid_array_mean_slope, grid_array_D4, g_list[22])
    grid_array_D3 = tf_VIC.D3(grid_array_fc_layer3, grid_array_depth_layer3, g_list[23])
    params_dataset_level0.variables["D1"][:, :] = grid_array_D1
    params_dataset_level0.variables["D2"][:, :] = grid_array_D2
    params_dataset_level0.variables["D3"][:, :] = grid_array_D3
    
    # Dsmax, mm or mm/day
    grid_array_Dsmax = tf_VIC.Dsmax(grid_array_D1, grid_array_D2, grid_array_D3, grid_array_cexpt, grid_array_phi_s_layer3, grid_array_depth_layer3)
    params_dataset_level0.variables["Dsmax"][:, :] = grid_array_Dsmax
    
    # Ds, [day^-D4] or fraction
    grid_array_Ds = tf_VIC.Ds(grid_array_D1, grid_array_D3, grid_array_Dsmax)
    params_dataset_level0.variables["Ds"][:, :] = grid_array_Ds
    
    # Ws, fraction
    grid_array_Ws = tf_VIC.Ws(grid_array_D3, grid_array_phi_s_layer3, grid_array_depth_layer3)
    params_dataset_level0.variables["Ws"][:, :] = grid_array_Ws
    
    # init_moist, mm
    grid_array_init_moist_layer1 = tf_VIC.init_moist(grid_array_phi_s_layer1, grid_array_depth_layer1)
    grid_array_init_moist_layer2 = tf_VIC.init_moist(grid_array_phi_s_layer2, grid_array_depth_layer2)
    grid_array_init_moist_layer3 = tf_VIC.init_moist(grid_array_phi_s_layer3, grid_array_depth_layer3)
    
    params_dataset_level0.variables["init_moist"][0, :, :] = grid_array_init_moist_layer1
    params_dataset_level0.variables["init_moist"][1, :, :] = grid_array_init_moist_layer2
    params_dataset_level0.variables["init_moist"][2, :, :] = grid_array_init_moist_layer3
    
    # elev, m, Arithmetic mean
    grid_array_SrtmDEM_mean_Value, _, _ = createArray_from_gridshp(grid_shp_level0, value_column="SrtmDEM_mean_Value", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
    params_dataset_level0.variables["elev"][:, :] = grid_array_SrtmDEM_mean_Value
    
    # dp, m, typically is 4m
    grid_shp_level0["dp"] = np.full((len(grid_shp_level0.index), ), fill_value=tf_VIC.dp(g_list[24]))
    grid_array_dp, _, _ = createArray_from_gridshp(grid_shp_level0, value_column="dp", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
    params_dataset_level0.variables["dp"][:, :] = grid_array_dp
    
    # bubble, cm
    grid_array_bubble_layer1 = tf_VIC.bubble(grid_array_expt_layer1, g_list[25], g_list[26])
    grid_array_bubble_layer2 = tf_VIC.bubble(grid_array_expt_layer2, g_list[25], g_list[26])
    grid_array_bubble_layer3 = tf_VIC.bubble(grid_array_expt_layer3, g_list[25], g_list[26])
    
    params_dataset_level0.variables["bubble"][0, :, :] = grid_array_bubble_layer1
    params_dataset_level0.variables["bubble"][1, :, :] = grid_array_bubble_layer2
    params_dataset_level0.variables["bubble"][2, :, :] = grid_array_bubble_layer3
    
    # quartz, N/A, fraction
    grid_array_quartz_layer1 = tf_VIC.quartz(grid_array_sand_layer1, g_list[27])
    grid_array_quartz_layer2 = tf_VIC.quartz(grid_array_sand_layer2, g_list[27])
    grid_array_quartz_layer3 = tf_VIC.quartz(grid_array_sand_layer3, g_list[27])
    
    params_dataset_level0.variables["quartz"][0, :, :] = grid_array_quartz_layer1
    params_dataset_level0.variables["quartz"][1, :, :] = grid_array_quartz_layer2
    params_dataset_level0.variables["quartz"][2, :, :] = grid_array_quartz_layer3
    
    # bulk_density, kg/m3 or mm
    grid_array_bd_layer1 = cal_bd_grid_array(grid_shp_level0, depth_layer1_start, depth_layer1_end)
    grid_array_bd_layer2 = cal_bd_grid_array(grid_shp_level0, depth_layer2_start, depth_layer2_end)
    grid_array_bd_layer3 = cal_bd_grid_array(grid_shp_level0, depth_layer3_start, depth_layer3_end)
    
    grid_array_bd_layer1 = tf_VIC.bulk_density(grid_array_bd_layer1, g_list[28])
    grid_array_bd_layer2 = tf_VIC.bulk_density(grid_array_bd_layer2, g_list[28])
    grid_array_bd_layer3 = tf_VIC.bulk_density(grid_array_bd_layer3, g_list[28])
    
    params_dataset_level0.variables["bulk_density"][0, :, :] = grid_array_bd_layer1
    params_dataset_level0.variables["bulk_density"][1, :, :] = grid_array_bd_layer2
    params_dataset_level0.variables["bulk_density"][2, :, :] = grid_array_bd_layer3
    
    # soil_density, kg/m3
    soil_density_layer1 = tf_VIC.soil_density(g_list[29])
    soil_density_layer2 = tf_VIC.soil_density(g_list[30])
    soil_density_layer3 = tf_VIC.soil_density(g_list[31])
    
    grid_shp_level0["soil_density_layer1"] = np.full((len(grid_shp_level0.index), ), fill_value=soil_density_layer1)
    grid_shp_level0["soil_density_layer2"] = np.full((len(grid_shp_level0.index), ), fill_value=soil_density_layer2)
    grid_shp_level0["soil_density_layer3"] = np.full((len(grid_shp_level0.index), ), fill_value=soil_density_layer3)
    
    grid_array_soil_density_layer1, _, _ = createArray_from_gridshp(grid_shp_level0, value_column="soil_density_layer1", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
    grid_array_soil_density_layer2, _, _ = createArray_from_gridshp(grid_shp_level0, value_column="soil_density_layer2", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
    grid_array_soil_density_layer3, _, _ = createArray_from_gridshp(grid_shp_level0, value_column="soil_density_layer3", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
    
    params_dataset_level0.variables["soil_density"][0, :, :] = grid_array_soil_density_layer1
    params_dataset_level0.variables["soil_density"][1, :, :] = grid_array_soil_density_layer2
    params_dataset_level0.variables["soil_density"][2, :, :] = grid_array_soil_density_layer3
    
    # Wcr_FRACT, fraction
    grid_array_Wcr_FRACT_layer1 = tf_VIC.Wcr_FRACT(grid_array_fc_layer1, grid_array_phi_s_layer1, g_list[32])
    grid_array_Wcr_FRACT_layer2 = tf_VIC.Wcr_FRACT(grid_array_fc_layer2, grid_array_phi_s_layer2, g_list[32])
    grid_array_Wcr_FRACT_layer3 = tf_VIC.Wcr_FRACT(grid_array_fc_layer3, grid_array_phi_s_layer3, g_list[32])
    
    params_dataset_level0.variables["Wcr_FRACT"][0, :, :] = grid_array_Wcr_FRACT_layer1
    params_dataset_level0.variables["Wcr_FRACT"][1, :, :] = grid_array_Wcr_FRACT_layer2
    params_dataset_level0.variables["Wcr_FRACT"][2, :, :] = grid_array_Wcr_FRACT_layer3
    
    # wp, computed field capacity [frac]
    grid_array_wp_layer1 = tf_VIC.wp(grid_array_phi_s_layer1, grid_array_b_retcurve_layer1, grid_array_psis_layer1, g_list[33])
    grid_array_wp_layer2 = tf_VIC.wp(grid_array_phi_s_layer2, grid_array_b_retcurve_layer2, grid_array_psis_layer2, g_list[33])
    grid_array_wp_layer3 = tf_VIC.wp(grid_array_phi_s_layer3, grid_array_b_retcurve_layer3, grid_array_psis_layer3, g_list[33])
    
    params_dataset_level0.variables["wp"][0, :, :] = grid_array_wp_layer1
    params_dataset_level0.variables["wp"][1, :, :] = grid_array_wp_layer2
    params_dataset_level0.variables["wp"][2, :, :] = grid_array_wp_layer3
    
    # Wpwp_FRACT, fraction
    grid_array_Wpwp_FRACT_layer1 = tf_VIC.Wpwp_FRACT(grid_array_wp_layer1, grid_array_phi_s_layer1, g_list[34])
    grid_array_Wpwp_FRACT_layer2 = tf_VIC.Wpwp_FRACT(grid_array_wp_layer2, grid_array_phi_s_layer2, g_list[34])
    grid_array_Wpwp_FRACT_layer3 = tf_VIC.Wpwp_FRACT(grid_array_wp_layer3, grid_array_phi_s_layer3, g_list[34])
    
    params_dataset_level0.variables["Wpwp_FRACT"][0, :, :] = grid_array_Wpwp_FRACT_layer1
    params_dataset_level0.variables["Wpwp_FRACT"][1, :, :] = grid_array_Wpwp_FRACT_layer2
    params_dataset_level0.variables["Wpwp_FRACT"][2, :, :] = grid_array_Wpwp_FRACT_layer3
    
    # rough, m, Surface roughness of bare soil
    rough = tf_VIC.rough(g_list[35])
    grid_shp_level0["rough"] = np.full((len(grid_shp_level0.index), ), fill_value=rough)
    grid_array_rough, _, _ = createArray_from_gridshp(grid_shp_level0, value_column="rough", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
    params_dataset_level0.variables["rough"][:, :] = grid_array_rough
    
    # snow rough, m
    snow_rough = tf_VIC.snow_rough(g_list[36])
    grid_shp_level0["snow_rough"] = np.full((len(grid_shp_level0.index), ), fill_value=snow_rough)
    grid_array_snow_rough, _, _ = createArray_from_gridshp(grid_shp_level0, value_column="snow_rough", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
    params_dataset_level0.variables["snow_rough"][:, :] = grid_array_snow_rough

    return params_dataset_level0


def buildParam_level1(dpc_VIC_level1, evb_dir, reverse_lat=True, domain_dataset=None):
    print("building Param_level1... ...")
    ## ====================== set dir and path ======================
    # set path
    params_dataset_level1_path = os.path.join(evb_dir.ParamFile_dir, "params_dataset_level1.nc")
    veg_param_json_path = os.path.join(evb_dir.__data_dir__, "veg_type_attributes_umd_updated.json")
    
    ## ====================== get grid_shp and basin_shp ======================
    grid_shp_level1 = dpc_VIC_level1.grid_shp
    basin_shp = dpc_VIC_level1.basin_shp
    
    # grids_map_array
    lon_list_level1, lat_list_level1, lon_map_index_level1, lat_map_index_level1 = grids_array_coord_map(grid_shp_level1, reverse_lat=reverse_lat)  #* all lat set as reverse

    ## ====================== create parameter ======================
    params_dataset_level1 = createParametersDataset(params_dataset_level1_path, lat_list_level1, lon_list_level1)
    tf_VIC = TF_VIC()
    
    ## ===================== level1: assign values for general variables  ======================
    # dimension variables: lat, lon, nlayer, root_zone, veg_class, month
    params_dataset_level1.variables["lat"][:] = np.array(lat_list_level1)  # 1D array  #* all lat set as reverse
    params_dataset_level1.variables["lon"][:] = np.array(lon_list_level1)  # 1D array
    params_dataset_level1.variables["nlayer"][:] = [1, 2, 3]
    root_zone_list = [1, 2, 3]
    params_dataset_level1.variables["root_zone"][:] = root_zone_list
    veg_class_list = list(range(14))
    params_dataset_level1.variables["veg_class"][:] = veg_class_list
    month_list = list(range(1, 13))
    params_dataset_level1.variables["month"][:] = month_list
    
    # lons, lats, 2D array
    grid_array_lons, grid_array_lats = np.meshgrid(params_dataset_level1.variables["lon"][:], params_dataset_level1.variables["lat"][:])  # 2D array
    params_dataset_level1.variables["lons"][:, :] = grid_array_lons
    params_dataset_level1.variables["lats"][:, :] = grid_array_lats
    
    # run_cell, bool, same as mask in DomainFile
    if domain_dataset is None:
        mask, frac, area, x_length, y_length = cal_mask_frac_area_length(dpc_VIC_level1, reverse_lat=reverse_lat, plot=False)  #* all lat set as reverse
    else:
        mask = domain_dataset.variables["mask"][:, :]  #* note the reverse_lat should be same
    params_dataset_level1.variables["run_cell"][:, :] = mask
    
    # grid_cell
    grid_shp_level1["grid_cell"] = np.arange(1, len(grid_shp_level1.index) + 1)
    grid_array_grid_cell, _, _ = createArray_from_gridshp(grid_shp_level1, value_column="grid_cell", grid_res=None, dtype=int, missing_value=0, plot=False)
    params_dataset_level1.variables["grid_cell"][:, :] = grid_array_grid_cell

    # off_gmt, hours
    grid_array_off_gmt = tf_VIC.off_gmt(grid_array_lons)
    params_dataset_level1.variables["off_gmt"][:, :] = grid_array_off_gmt
    
    # avg_T, C
    grid_array_avg_T, _, _ = createArray_from_gridshp(grid_shp_level1, value_column="stl_all_layers_mean_Value", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
    params_dataset_level1.variables["avg_T"][:, :] = grid_array_avg_T

    # annual_prec, mm
    grid_array_annual_P, _, _ = createArray_from_gridshp(grid_shp_level1, value_column="annual_P_in_src_grid_Value", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
    params_dataset_level1.variables["annual_prec"][:, :] = grid_array_annual_P
    
    # resid_moist, fraction, set as 0
    grid_shp_level1["resid_moist"] = np.full((len(grid_shp_level1.index), ), fill_value=0)
    grid_array_resid_moist, _, _ = createArray_from_gridshp(grid_shp_level1, value_column="resid_moist", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
    params_dataset_level1.variables["resid_moist"][0, :, :] = grid_array_resid_moist
    params_dataset_level1.variables["resid_moist"][1, :, :] = grid_array_resid_moist
    params_dataset_level1.variables["resid_moist"][2, :, :] = grid_array_resid_moist
    
    # fs_active, bool, whether the frozen soil algorithm is activated
    grid_shp_level1["fs_active"] = np.full((len(grid_shp_level1.index), ), fill_value=0)
    grid_array_fs_active, _, _ = createArray_from_gridshp(grid_shp_level1, value_column="fs_active", grid_res=None, dtype=int, missing_value=np.NAN, plot=False)
    params_dataset_level1.variables["fs_active"][:, :] = grid_array_fs_active
    
    # Nveg, int
    grid_shp_level1["Nveg"] = grid_shp_level1["umd_lc_original_Value"].apply(lambda row: len(list(set(row))))
    grid_array_Nveg, _, _ = createArray_from_gridshp(grid_shp_level1, value_column="Nveg", grid_res=None, dtype=int, missing_value=np.NAN, plot=False)
    params_dataset_level1.variables["Nveg"][:, :] = grid_array_Nveg
    
    # Cv, fraction
    for i in veg_class_list:
        grid_shp_level1[f"umd_lc_{i}_veg_index"] = grid_shp_level1.loc[:, "umd_lc_original_Value"].apply(lambda row: np.where(np.array(row)==i)[0])
        grid_shp_level1[f"umd_lc_{i}_veg_Cv"] = grid_shp_level1.apply(lambda row: sum(np.array(row["umd_lc_original_Cv"])[row[f"umd_lc_{i}_veg_index"]]), axis=1)
        grid_array_i_veg_Cv, _, _ = createArray_from_gridshp(grid_shp_level1, value_column=f"umd_lc_{i}_veg_Cv", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
        params_dataset_level1.variables["Cv"][i, :, :] = grid_array_i_veg_Cv
    
    # read veg params, veg_params_json is a lookup_table
    with open(veg_param_json_path, 'r') as f:
        veg_params_json = json.load(f)
        # veg_params_json = veg_params_json["classAttributes"]
        # veg_keys = [v["class"] for v in veg_params_json]
        # veg_params = [v["properties"] for v in veg_params_json]
        # veg_params_json = dict(zip(veg_keys, veg_params))
        
    # root_depth, m; root_fract, fraction
    for i in veg_class_list:
        for j in root_zone_list:
            grid_shp_level1[f"umd_{i}_veg_{j}_zone_root_depth"] = np.full((len(grid_shp_level1.index), ), fill_value=float(veg_params_json[f"{i}"][f"rootd{j}"]))
            grid_array_i_veg_j_zone_root_depth, _, _ = createArray_from_gridshp(grid_shp_level1, value_column=f"umd_{i}_veg_{j}_zone_root_depth", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
            params_dataset_level1.variables["root_depth"][i, j-1, :, :] = grid_array_i_veg_j_zone_root_depth  # j-1: root_zone_list start from 1
            
            grid_shp_level1[f"umd_{i}_veg_{j}_zone_root_fract"] = np.full((len(grid_shp_level1.index), ), fill_value=float(veg_params_json[f"{i}"][f"rootfr{j}"]))
            grid_array_i_veg_j_zone_root_fract, _, _ = createArray_from_gridshp(grid_shp_level1, value_column=f"umd_{i}_veg_{j}_zone_root_fract", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
            params_dataset_level1.variables["root_fract"][i, j-1, :, :] = grid_array_i_veg_j_zone_root_fract

    # rarc, s/m; rmin, s/m
    for i in veg_class_list:
        grid_shp_level1[f"umd_{i}_veg_rarc"] = np.full((len(grid_shp_level1.index), ), fill_value=float(veg_params_json[f"{i}"][f"rarc"]))
        grid_array_i_veg_rarc, _, _ = createArray_from_gridshp(grid_shp_level1, value_column=f"umd_{i}_veg_rarc", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
        params_dataset_level1.variables["rarc"][i, :, :] = grid_array_i_veg_rarc
        
        grid_shp_level1[f"umd_{i}_veg_rmin"] = np.full((len(grid_shp_level1.index), ), fill_value=float(veg_params_json[f"{i}"][f"rmin"]))
        grid_array_i_veg_rmin, _, _ = createArray_from_gridshp(grid_shp_level1, value_column=f"umd_{i}_veg_rmin", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
        params_dataset_level1.variables["rmin"][i, :, :] = grid_array_i_veg_rmin
    
    # overstory, N/A, bool
    # wind_h, m, adjust wind height value if overstory is true (overstory == 1, wind_h=vegHeight+10, else wind_h=vegHeight+2)
    for i in veg_class_list:
        grid_shp_level1[f"umd_{i}_veg_height"] = np.full((len(grid_shp_level1.index), ), fill_value=float(veg_params_json[f"{i}"][f"h"]))
        grid_array_i_veg_height, _, _ = createArray_from_gridshp(grid_shp_level1, value_column=f"umd_{i}_veg_height", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
        
        grid_shp_level1[f"umd_{i}_veg_overstory"] = np.full((len(grid_shp_level1.index), ), fill_value=int(veg_params_json[f"{i}"][f"overstory"]))
        grid_array_i_veg_overstory, _, _ = createArray_from_gridshp(grid_shp_level1, value_column=f"umd_{i}_veg_overstory", grid_res=None, dtype=int, missing_value=np.NAN, plot=False)

        grid_array_wind_h_add_factor = np.full_like(grid_array_i_veg_overstory, fill_value=10)
        grid_array_wind_h_add_factor[grid_array_i_veg_overstory == 0] = 2
        
        grid_array_wind_h = grid_array_i_veg_height + grid_array_wind_h_add_factor
        
        params_dataset_level1.variables["overstory"][i, :, :] = grid_array_i_veg_overstory
        params_dataset_level1.variables["wind_h"][i, :, :] = grid_array_wind_h
        
        # for j in month_list:
        #     params_dataset_level1.variables["displacement"][i, :, :, :] = grid_array_i_veg_height * 0.67
        #     params_dataset_level1.variables["veg_rough"][i, :, :, :] = grid_array_i_veg_height * 0.123
    
    # displacement, m, Vegetation displacement height (typically 0.67 * vegetation height), or read from veg_param_json_updated
    # veg_rough, m, Vegetation roughness length (typically 0.123 * vegetation height), or read from veg_param_json_updated
    for i in veg_class_list:
        for j in month_list:
            grid_shp_level1[f"umd_{i}_veg_{j}_month_displacement"] = np.full((len(grid_shp_level1.index), ), fill_value=float(veg_params_json[f"{i}"][f"veg_displacement_month_{j}"]))
            grid_array_i_veg_j_month_displacement, _, _ = createArray_from_gridshp(grid_shp_level1, value_column=f"umd_{i}_veg_{j}_month_displacement", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
            
            grid_shp_level1[f"umd_{i}_veg_{j}_month_veg_rough"] = np.full((len(grid_shp_level1.index), ), fill_value=float(veg_params_json[f"{i}"][f"veg_rough_month_{j}"]))
            grid_array_i_veg_j_month_veg_rough, _, _ = createArray_from_gridshp(grid_shp_level1, value_column=f"umd_{i}_veg_{j}_month_veg_rough", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
            
            params_dataset_level1.variables["displacement"][i, j-1, :, :] = grid_array_i_veg_j_month_displacement   # j-1: month_list start from 1
            params_dataset_level1.variables["veg_rough"][i, j-1, :, :] = grid_array_i_veg_j_month_veg_rough
    
    # RGL, W/m2; rad_atten, fract; wind_atten, fract; trunk_ratio, fract
    for i in veg_class_list:
        grid_shp_level1[f"umd_{i}_veg_RGL"] = np.full((len(grid_shp_level1.index), ), fill_value=float(veg_params_json[f"{i}"][f"rgl"]))
        grid_array_i_veg_RGL, _, _ = createArray_from_gridshp(grid_shp_level1, value_column=f"umd_{i}_veg_RGL", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
        params_dataset_level1.variables["RGL"][i, :, :] = grid_array_i_veg_RGL

        grid_shp_level1[f"umd_{i}_veg_rad_atten"] = np.full((len(grid_shp_level1.index), ), fill_value=float(veg_params_json[f"{i}"][f"rad_atn"]))
        grid_array_i_veg_rad_atten, _, _ = createArray_from_gridshp(grid_shp_level1, value_column=f"umd_{i}_veg_rad_atten", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
        params_dataset_level1.variables["rad_atten"][i, :, :] = grid_array_i_veg_rad_atten
    
        grid_shp_level1[f"umd_{i}_veg_wind_atten"] = np.full((len(grid_shp_level1.index), ), fill_value=float(veg_params_json[f"{i}"][f"wnd_atn"]))
        grid_array_i_veg_wind_atten, _, _ = createArray_from_gridshp(grid_shp_level1, value_column=f"umd_{i}_veg_wind_atten", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
        params_dataset_level1.variables["wind_atten"][i, :, :] = grid_array_i_veg_wind_atten

        grid_shp_level1[f"umd_{i}_veg_trunk_ratio"] = np.full((len(grid_shp_level1.index), ), fill_value=float(veg_params_json[f"{i}"][f"trnk_r"]))
        grid_array_i_veg_trunk_ratio, _, _ = createArray_from_gridshp(grid_shp_level1, value_column=f"umd_{i}_veg_trunk_ratio", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
        params_dataset_level1.variables["trunk_ratio"][i, :, :] = grid_array_i_veg_trunk_ratio
    
    # LAI, fraction or m2/m2; albedo, fraction; fcanopy, fraction
    for i in veg_class_list:
        for j in month_list:
            # LAI
            grid_shp_level1[f"MODIS_{i}_veg_{j}_month_LAI"] = grid_shp_level1.apply(lambda row: np.array(row[f"MODIS_LAI_original_Value_month{j}"])[np.where(np.array(row.umd_lc_original_Value)==i)[0]], axis=1)
            grid_shp_level1[f"MODIS_{i}_veg_{j}_month_LAI"] = grid_shp_level1.loc[:, f"MODIS_{i}_veg_{j}_month_LAI"].apply(lambda row: np.mean(row) if len(row) != 0 else 0)
            
            grid_array_i_veg_j_month_LAI, _, _ = createArray_from_gridshp(grid_shp_level1, value_column=f"MODIS_{i}_veg_{j}_month_LAI", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
            params_dataset_level1.variables["LAI"][i, j-1, :, :] = grid_array_i_veg_j_month_LAI   # j-1: month_list start from 1
            
            # BSA, albedo
            grid_shp_level1[f"MODIS_{i}_veg_{j}_month_BSA"] = grid_shp_level1.apply(lambda row: np.array(row[f"MODIS_BSA_original_Value_month{j}"])[np.where(np.array(row.umd_lc_original_Value)==i)[0]], axis=1)
            grid_shp_level1[f"MODIS_{i}_veg_{j}_month_BSA"] = grid_shp_level1.loc[:, f"MODIS_{i}_veg_{j}_month_BSA"].apply(lambda row: np.mean(row) if len(row) != 0 else 0)
            
            grid_array_i_veg_j_month_BSA, _, _ = createArray_from_gridshp(grid_shp_level1, value_column=f"MODIS_{i}_veg_{j}_month_BSA", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
            params_dataset_level1.variables["albedo"][i, j-1, :, :] = grid_array_i_veg_j_month_BSA   # j-1: month_list start from 1
            
            # fcanopy, ((NDVI-NDVI_min)/(NDVI_max-NDVI_min))**2
            grid_shp_level1[f"MODIS_{i}_veg_{j}_month_NDVI"] = grid_shp_level1.apply(lambda row: np.array(row[f"MODIS_NDVI_original_Value_month{j}"])[np.where(np.array(row.umd_lc_original_Value)==i)[0]], axis=1)
            grid_shp_level1[f"MODIS_{i}_veg_{j}_month_NDVI"] = grid_shp_level1.loc[:, f"MODIS_{i}_veg_{j}_month_NDVI"].apply(lambda row: np.mean(row) if len(row) != 0 else 0)
            grid_shp_level1[f"MODIS_{i}_veg_{j}_month_NDVI"] = grid_shp_level1[f"MODIS_{i}_veg_{j}_month_NDVI"] * 0.0001
            NDVI = grid_shp_level1[f"MODIS_{i}_veg_{j}_month_NDVI"]
            
            grid_shp_level1[f"MODIS_{i}_veg_{j}_month_NDVI_max"] = grid_shp_level1.apply(lambda row: np.array(row[f"MODIS_NDVI_max_original_Value_month{j}"])[np.where(np.array(row.umd_lc_original_Value)==i)[0]], axis=1)
            grid_shp_level1[f"MODIS_{i}_veg_{j}_month_NDVI_max"] = grid_shp_level1.loc[:, f"MODIS_{i}_veg_{j}_month_NDVI_max"].apply(lambda row: np.mean(row) if len(row) != 0 else 0)
            grid_shp_level1[f"MODIS_{i}_veg_{j}_month_NDVI_max"] = grid_shp_level1[f"MODIS_{i}_veg_{j}_month_NDVI_max"] * 0.0001
            NDVI_max = grid_shp_level1[f"MODIS_{i}_veg_{j}_month_NDVI_max"]
            
            grid_shp_level1[f"MODIS_{i}_veg_{j}_month_NDVI_min"] = grid_shp_level1.apply(lambda row: np.array(row[f"MODIS_NDVI_min_original_Value_month{j}"])[np.where(np.array(row.umd_lc_original_Value)==i)[0]], axis=1)
            grid_shp_level1[f"MODIS_{i}_veg_{j}_month_NDVI_min"] = grid_shp_level1.loc[:, f"MODIS_{i}_veg_{j}_month_NDVI_min"].apply(lambda row: np.mean(row) if len(row) != 0 else 0)
            grid_shp_level1[f"MODIS_{i}_veg_{j}_month_NDVI_min"] = grid_shp_level1[f"MODIS_{i}_veg_{j}_month_NDVI_min"] * 0.0001
            NDVI_min = grid_shp_level1[f"MODIS_{i}_veg_{j}_month_NDVI_min"]
            
            fcanopy = ((NDVI-NDVI_min)/(NDVI_max-NDVI_min)) ** 2
            fcanopy[np.isnan(fcanopy)] = 0
            grid_shp_level1[f"MODIS_{i}_veg_{j}_month_fcanopy"] = fcanopy
            grid_array_i_veg_j_month_fcanopy, _, _ = createArray_from_gridshp(grid_shp_level1, value_column=f"MODIS_{i}_veg_{j}_month_fcanopy", grid_res=None, dtype=float, missing_value=np.NAN, plot=False)
            
            params_dataset_level1.variables["fcanopy"][i, j-1, :, :] = grid_array_i_veg_j_month_fcanopy   # j-1: month_list start from 1
            
    return params_dataset_level1


def scaling_level0_to_level1_search_grids(params_dataset_level0, params_dataset_level1):
    # read lon, lat from params, cal res
    lon_list_level0, lat_list_level0 = params_dataset_level0.variables["lon"][:], params_dataset_level0.variables["lat"][:]
    lon_list_level1, lat_list_level1 = params_dataset_level1.variables["lon"][:], params_dataset_level1.variables["lat"][:]
    lon_list_level0 = np.ma.filled(lon_list_level0, fill_value=np.NAN)
    lat_list_level0 = np.ma.filled(lat_list_level0, fill_value=np.NAN)
    lon_list_level1 = np.ma.filled(lon_list_level1, fill_value=np.NAN)
    lat_list_level1 = np.ma.filled(lat_list_level1, fill_value=np.NAN)
    
    res_lon_level0 = (max(lon_list_level0) - min(lon_list_level0)) / (len(lon_list_level0) - 1)
    res_lat_level0 = (max(lat_list_level0) - min(lat_list_level0)) / (len(lat_list_level0) - 1)
    res_lon_level1 = (max(lon_list_level1) - min(lon_list_level1)) / (len(lon_list_level1) - 1)
    res_lat_level1 = (max(lat_list_level1) - min(lat_list_level1)) / (len(lat_list_level1) - 1)
    
    # meshgrid and flatten
    lon_list_level1_2D, lat_list_level1_2D = np.meshgrid(lon_list_level1, lat_list_level1)
    lon_list_level1_2D_flatten = lon_list_level1_2D.flatten()
    lat_list_level1_2D_flatten = lat_list_level1_2D.flatten()
    
    # search grids
    searched_grids_index = search_grids.search_grids_radius_rectangle(dst_lat=lat_list_level1_2D_flatten, dst_lon=lon_list_level1_2D_flatten,
                                                                        src_lat=lat_list_level0, src_lon=lon_list_level0,
                                                                        lat_radius=res_lat_level1, lon_radius=res_lon_level1)
    
    return searched_grids_index


def scaling_level0_to_level1(params_dataset_level0, params_dataset_level1, searched_grids_index=None):
    print("scaling Param_level0 to Param_level1... ...")
    # ======================= get grids match (search grids) ======================= 
    # read lon, lat from params, cal res
    lon_list_level0, lat_list_level0 = params_dataset_level0.variables["lon"][:], params_dataset_level0.variables["lat"][:]
    lon_list_level1, lat_list_level1 = params_dataset_level1.variables["lon"][:], params_dataset_level1.variables["lat"][:]
    lon_list_level0 = np.ma.filled(lon_list_level0, fill_value=np.NAN)
    lat_list_level0 = np.ma.filled(lat_list_level0, fill_value=np.NAN)
    lon_list_level1 = np.ma.filled(lon_list_level1, fill_value=np.NAN)
    lat_list_level1 = np.ma.filled(lat_list_level1, fill_value=np.NAN)
    
    res_lon_level0 = (max(lon_list_level0) - min(lon_list_level0)) / (len(lon_list_level0) - 1)
    res_lat_level0 = (max(lat_list_level0) - min(lat_list_level0)) / (len(lat_list_level0) - 1)
    res_lon_level1 = (max(lon_list_level1) - min(lon_list_level1)) / (len(lon_list_level1) - 1)
    res_lat_level1 = (max(lat_list_level1) - min(lat_list_level1)) / (len(lat_list_level1) - 1)
    
    # meshgrid and flatten
    lon_list_level1_2D, lat_list_level1_2D = np.meshgrid(lon_list_level1, lat_list_level1)
    lon_list_level1_2D_flatten = lon_list_level1_2D.flatten()
    lat_list_level1_2D_flatten = lat_list_level1_2D.flatten()
    
    # search grids
    if searched_grid_index is None:
        searched_grid_index = scaling_level0_to_level1_search_grids(params_dataset_level0, params_dataset_level1)
    
    # ======================= scaling (resample) =======================
    scaling_operator = Scaling_operator()
    for i in tqdm(range(len(lat_list_level1_2D_flatten))):
        # lon/lat
        searched_grid_index = searched_grids_index[i]
        searched_grid_lat = [lat_list_level0[searched_grid_index[0][j]] for j in range(len(searched_grid_index[0]))]
        searched_grid_lon = [lon_list_level0[searched_grid_index[1][j]] for j in range(len(searched_grid_index[0]))]
        
        # search and resampled data func
        searched_grid_data_func = lambda src_data: [src_data[searched_grid_index[0][j], searched_grid_index[1][j]] for j in range(len(searched_grid_index[0]))]                                            
        resampled_grid_data_func = lambda searched_data, general_function: resample.resampleMethod_GeneralFunction(searched_data, searched_grid_lat, searched_grid_lon, general_function=general_function, missing_value=None)
        combined_grid_data_func = lambda src_data, general_function: resampled_grid_data_func(searched_grid_data_func(src_data), general_function)
        
        # depth, m
        resampled_depth_layer1 = combined_grid_data_func(params_dataset_level0.variables["depth"][0, :, :], scaling_operator.Arithmetic_mean)
        resampled_depth_layer2 = combined_grid_data_func(params_dataset_level0.variables["depth"][1, :, :], scaling_operator.Arithmetic_mean)
        resampled_depth_layer3 = combined_grid_data_func(params_dataset_level0.variables["depth"][2, :, :], scaling_operator.Arithmetic_mean)
        
        # b_infilt, /NA
        resampled_b_infilt = combined_grid_data_func(params_dataset_level0.variables["infilt"][:, :], scaling_operator.Arithmetic_mean)
        
        # ksat, mm/s -> mm/day (VIC requirement)
        resampled_ksat_layer1 = combined_grid_data_func(params_dataset_level0.variables["Ksat"][0, :, :], scaling_operator.Harmonic_mean)
        resampled_ksat_layer2 = combined_grid_data_func(params_dataset_level0.variables["Ksat"][1, :, :], scaling_operator.Harmonic_mean)
        resampled_ksat_layer3 = combined_grid_data_func(params_dataset_level0.variables["Ksat"][2, :, :], scaling_operator.Harmonic_mean)
        
        # phi_s, m3/m3 or mm/mm
        resampled_phi_s_layer1 = combined_grid_data_func(params_dataset_level0.variables["phi_s"][0, :, :], scaling_operator.Arithmetic_mean)
        resampled_phi_s_layer2 = combined_grid_data_func(params_dataset_level0.variables["phi_s"][1, :, :], scaling_operator.Arithmetic_mean)
        resampled_phi_s_layer3 = combined_grid_data_func(params_dataset_level0.variables["phi_s"][2, :, :], scaling_operator.Arithmetic_mean)
        
        # psis, kPa/cm-H2O
        resampled_psis_layer1 = combined_grid_data_func(params_dataset_level0.variables["psis"][0, :, :], scaling_operator.Arithmetic_mean)
        resampled_psis_layer2 = combined_grid_data_func(params_dataset_level0.variables["psis"][1, :, :], scaling_operator.Arithmetic_mean)
        resampled_psis_layer3 = combined_grid_data_func(params_dataset_level0.variables["psis"][2, :, :], scaling_operator.Arithmetic_mean)
        
        # b_retcurve, /NA
        resampled_b_retcurve_layer1 = combined_grid_data_func(params_dataset_level0.variables["b_retcurve"][0, :, :], scaling_operator.Arithmetic_mean)
        resampled_b_retcurve_layer2 = combined_grid_data_func(params_dataset_level0.variables["b_retcurve"][1, :, :], scaling_operator.Arithmetic_mean)
        resampled_b_retcurve_layer3 = combined_grid_data_func(params_dataset_level0.variables["b_retcurve"][2, :, :], scaling_operator.Arithmetic_mean)
        
        # expt, /NA
        resampled_expt_layer1 = combined_grid_data_func(params_dataset_level0.variables["expt"][0, :, :], scaling_operator.Arithmetic_mean)
        resampled_expt_layer2 = combined_grid_data_func(params_dataset_level0.variables["expt"][1, :, :], scaling_operator.Arithmetic_mean)
        resampled_expt_layer3 = combined_grid_data_func(params_dataset_level0.variables["expt"][2, :, :], scaling_operator.Arithmetic_mean)
        
        # fc, % or m3/m3
        resampled_fc_layer1 = combined_grid_data_func(params_dataset_level0.variables["fc"][0, :, :], scaling_operator.Arithmetic_mean)
        resampled_fc_layer2 = combined_grid_data_func(params_dataset_level0.variables["fc"][1, :, :], scaling_operator.Arithmetic_mean)
        resampled_fc_layer3 = combined_grid_data_func(params_dataset_level0.variables["fc"][2, :, :], scaling_operator.Arithmetic_mean)
        
        # D4, /NA, same as c, typically is 2
        resampled_D4 = combined_grid_data_func(params_dataset_level0.variables["D4"][:, :], scaling_operator.Arithmetic_mean)

        # cexpt
        resampled_c = combined_grid_data_func(params_dataset_level0.variables["c"][:, :], scaling_operator.Arithmetic_mean)

        # D1 ([day^-1]), D2 ([day^-D4])
        resampled_D1 = combined_grid_data_func(params_dataset_level0.variables["D1"][:, :], scaling_operator.Harmonic_mean)
        resampled_D2 = combined_grid_data_func(params_dataset_level0.variables["D2"][:, :], scaling_operator.Harmonic_mean)
        
        # D3 ([mm])
        resampled_D3 = combined_grid_data_func(params_dataset_level0.variables["D3"][:, :], scaling_operator.Arithmetic_mean)

        # Dsmax, mm or mm/day
        resampled_Dsmax = combined_grid_data_func(params_dataset_level0.variables["Dsmax"][:, :], scaling_operator.Harmonic_mean)

        # Ds, [day^-D4] or fraction
        resampled_Ds = combined_grid_data_func(params_dataset_level0.variables["Ds"][:, :], scaling_operator.Harmonic_mean)

        # Ws, fraction
        resampled_Ws = combined_grid_data_func(params_dataset_level0.variables["Ws"][:, :], scaling_operator.Arithmetic_mean)
        
        # init_moist, mm
        resampled_init_moist_layer1 = combined_grid_data_func(params_dataset_level0.variables["init_moist"][0, :, :], scaling_operator.Arithmetic_mean)
        resampled_init_moist_layer2 = combined_grid_data_func(params_dataset_level0.variables["init_moist"][1, :, :], scaling_operator.Arithmetic_mean)
        resampled_init_moist_layer3 = combined_grid_data_func(params_dataset_level0.variables["init_moist"][2, :, :], scaling_operator.Arithmetic_mean)
        
        # elev, m
        resampled_elev = combined_grid_data_func(params_dataset_level0.variables["elev"][:, :], scaling_operator.Arithmetic_mean)
        
        # dp, m, typically is 4m
        resampled_dp = combined_grid_data_func(params_dataset_level0.variables["dp"][:, :], scaling_operator.Arithmetic_mean)
        
        # bubble, cm
        resampled_bubble_layer1 = combined_grid_data_func(params_dataset_level0.variables["bubble"][0, :, :], scaling_operator.Arithmetic_mean)
        resampled_bubble_layer2 = combined_grid_data_func(params_dataset_level0.variables["bubble"][1, :, :], scaling_operator.Arithmetic_mean)
        resampled_bubble_layer3 = combined_grid_data_func(params_dataset_level0.variables["bubble"][2, :, :], scaling_operator.Arithmetic_mean)
        
        # quartz, N/A
        resampled_quartz_layer1 = combined_grid_data_func(params_dataset_level0.variables["quartz"][0, :, :], scaling_operator.Arithmetic_mean)
        resampled_quartz_layer2 = combined_grid_data_func(params_dataset_level0.variables["quartz"][1, :, :], scaling_operator.Arithmetic_mean)
        resampled_quartz_layer3 = combined_grid_data_func(params_dataset_level0.variables["quartz"][2, :, :], scaling_operator.Arithmetic_mean)
        
        # bulk_density, kg/m3 or mm
        resampled_bulk_density_layer1 = combined_grid_data_func(params_dataset_level0.variables["bulk_density"][0, :, :], scaling_operator.Arithmetic_mean)
        resampled_bulk_density_layer2 = combined_grid_data_func(params_dataset_level0.variables["bulk_density"][1, :, :], scaling_operator.Arithmetic_mean)
        resampled_bulk_density_layer3 = combined_grid_data_func(params_dataset_level0.variables["bulk_density"][2, :, :], scaling_operator.Arithmetic_mean)
        
        # soil_density, kg/m3
        resampled_soil_density_layer1 = combined_grid_data_func(params_dataset_level0.variables["soil_density"][0, :, :], scaling_operator.Arithmetic_mean)
        resampled_soil_density_layer2 = combined_grid_data_func(params_dataset_level0.variables["soil_density"][1, :, :], scaling_operator.Arithmetic_mean)
        resampled_soil_density_layer3 = combined_grid_data_func(params_dataset_level0.variables["soil_density"][2, :, :], scaling_operator.Arithmetic_mean)
        
        # Wcr_FRACT, fraction
        resampled_Wcr_FRACT_layer1 = combined_grid_data_func(params_dataset_level0.variables["Wcr_FRACT"][0, :, :], scaling_operator.Arithmetic_mean)
        resampled_Wcr_FRACT_layer2 = combined_grid_data_func(params_dataset_level0.variables["Wcr_FRACT"][1, :, :], scaling_operator.Arithmetic_mean)
        resampled_Wcr_FRACT_layer3 = combined_grid_data_func(params_dataset_level0.variables["Wcr_FRACT"][2, :, :], scaling_operator.Arithmetic_mean)
        
        # wp, computed field capacity [frac]
        resampled_wp_layer1 = combined_grid_data_func(params_dataset_level0.variables["wp"][0, :, :], scaling_operator.Arithmetic_mean)
        resampled_wp_layer2 = combined_grid_data_func(params_dataset_level0.variables["wp"][1, :, :], scaling_operator.Arithmetic_mean)
        resampled_wp_layer3 = combined_grid_data_func(params_dataset_level0.variables["wp"][2, :, :], scaling_operator.Arithmetic_mean)
        
        # Wpwp_FRACT, fraction
        resampled_Wpwp_FRACT_layer1 = combined_grid_data_func(params_dataset_level0.variables["Wpwp_FRACT"][0, :, :], scaling_operator.Arithmetic_mean)
        resampled_Wpwp_FRACT_layer2 = combined_grid_data_func(params_dataset_level0.variables["Wpwp_FRACT"][1, :, :], scaling_operator.Arithmetic_mean)
        resampled_Wpwp_FRACT_layer3 = combined_grid_data_func(params_dataset_level0.variables["Wpwp_FRACT"][2, :, :], scaling_operator.Arithmetic_mean)
        
        # rough, m, Surface roughness of bare soil
        resampled_rough = combined_grid_data_func(params_dataset_level0.variables["rough"][:, :], scaling_operator.Arithmetic_mean)
        
        # snow rough, m
        resampled_snow_rough = combined_grid_data_func(params_dataset_level0.variables["snow_rough"][:, :], scaling_operator.Arithmetic_mean)
        
        # set empty list
        if i == 0:
            depth_layer1_resampled_grids = []
            depth_layer2_resampled_grids = []
            depth_layer3_resampled_grids = []
            
            b_infilt_resampled_grids = []
            
            ksat_layer1_resampled_grids = []
            ksat_layer2_resampled_grids = []
            ksat_layer3_resampled_grids = []
            
            phi_s_layer1_resampled_grids = []
            phi_s_layer2_resampled_grids = []
            phi_s_layer3_resampled_grids = []
        
            psis_layer1_resampled_grids = []
            psis_layer2_resampled_grids = []
            psis_layer3_resampled_grids = []
            
            b_retcurve_layer1_resampled_grids = []
            b_retcurve_layer2_resampled_grids = []
            b_retcurve_layer3_resampled_grids = []
        
            expt_layer1_resampled_grids = []
            expt_layer2_resampled_grids = []
            expt_layer3_resampled_grids = []
            
            fc_layer1_resampled_grids = []
            fc_layer2_resampled_grids = []
            fc_layer3_resampled_grids = []
            
            D4_resampled_grids = []
            
            c_resampled_grids = []
            
            D1_resampled_grids = []
            D2_resampled_grids = []
            
            D3_resampled_grids = []
            
            Dsmax_resampled_grids = []
            
            Ds_resampled_grids = []
            
            Ws_resampled_grids = []
            
            init_moist_layer1_resampled_grids = []
            init_moist_layer2_resampled_grids = []
            init_moist_layer3_resampled_grids = []
            
            elev_resampled_grids = []
            
            dp_resampled_grids = []
            
            bubble_layer1_resampled_grids = []
            bubble_layer2_resampled_grids = []
            bubble_layer3_resampled_grids = []
            
            quartz_layer1_resampled_grids = []
            quartz_layer2_resampled_grids = []
            quartz_layer3_resampled_grids = []
            
            bulk_density_layer1_resampled_grids = []
            bulk_density_layer2_resampled_grids = []
            bulk_density_layer3_resampled_grids = []
            
            soil_density_layer1_resampled_grids = []
            soil_density_layer2_resampled_grids = []
            soil_density_layer3_resampled_grids = []
            
            Wcr_FRACT_layer1_resampled_grids = []
            Wcr_FRACT_layer2_resampled_grids = []
            Wcr_FRACT_layer3_resampled_grids = []
            
            wp_fract_layer1_resampled_grids = []
            wp_fract_layer2_resampled_grids = []
            wp_fract_layer3_resampled_grids = []
            
            Wpwp_FRACT_layer1_resampled_grids = []
            Wpwp_FRACT_layer2_resampled_grids = []
            Wpwp_FRACT_layer3_resampled_grids = []
            
            rough_resampled_grids = []
            
            snow_rough_resampled_grids = []
            
        # append values
        depth_layer1_resampled_grids.append(resampled_depth_layer1)
        depth_layer2_resampled_grids.append(resampled_depth_layer2)
        depth_layer3_resampled_grids.append(resampled_depth_layer3)
        
        b_infilt_resampled_grids.append(resampled_b_infilt)
        
        ksat_layer1_resampled_grids.append(resampled_ksat_layer1)
        ksat_layer2_resampled_grids.append(resampled_ksat_layer2)
        ksat_layer3_resampled_grids.append(resampled_ksat_layer3)
        
        phi_s_layer1_resampled_grids.append(resampled_phi_s_layer1)
        phi_s_layer2_resampled_grids.append(resampled_phi_s_layer2)
        phi_s_layer3_resampled_grids.append(resampled_phi_s_layer3)
        
        psis_layer1_resampled_grids.append(resampled_psis_layer1)
        psis_layer2_resampled_grids.append(resampled_psis_layer2)
        psis_layer3_resampled_grids.append(resampled_psis_layer3)
        
        b_retcurve_layer1_resampled_grids.append(resampled_b_retcurve_layer1)
        b_retcurve_layer2_resampled_grids.append(resampled_b_retcurve_layer2)
        b_retcurve_layer3_resampled_grids.append(resampled_b_retcurve_layer3)

        expt_layer1_resampled_grids.append(resampled_expt_layer1)
        expt_layer2_resampled_grids.append(resampled_expt_layer2)
        expt_layer3_resampled_grids.append(resampled_expt_layer3)
        
        fc_layer1_resampled_grids.append(resampled_fc_layer1)
        fc_layer2_resampled_grids.append(resampled_fc_layer2)
        fc_layer3_resampled_grids.append(resampled_fc_layer3)
        
        D4_resampled_grids.append(resampled_D4)
        
        c_resampled_grids.append(resampled_c)
        
        D1_resampled_grids.append(resampled_D1)
        D2_resampled_grids.append(resampled_D2)
        
        D3_resampled_grids.append(resampled_D3)
        
        Dsmax_resampled_grids.append(resampled_Dsmax)
        
        Ds_resampled_grids.append(resampled_Ds)
        
        Ws_resampled_grids.append(resampled_Ws)
        
        init_moist_layer1_resampled_grids.append(resampled_init_moist_layer1)
        init_moist_layer2_resampled_grids.append(resampled_init_moist_layer2)
        init_moist_layer3_resampled_grids.append(resampled_init_moist_layer3)
    
        elev_resampled_grids.append(resampled_elev)
        
        dp_resampled_grids.append(resampled_dp)
        
        bubble_layer1_resampled_grids.append(resampled_bubble_layer1)
        bubble_layer2_resampled_grids.append(resampled_bubble_layer2)
        bubble_layer3_resampled_grids.append(resampled_bubble_layer3)
        
        quartz_layer1_resampled_grids.append(resampled_quartz_layer1)
        quartz_layer2_resampled_grids.append(resampled_quartz_layer2)
        quartz_layer3_resampled_grids.append(resampled_quartz_layer3)
        
        bulk_density_layer1_resampled_grids.append(resampled_bulk_density_layer1)
        bulk_density_layer2_resampled_grids.append(resampled_bulk_density_layer2)
        bulk_density_layer3_resampled_grids.append(resampled_bulk_density_layer3)
        
        soil_density_layer1_resampled_grids.append(resampled_soil_density_layer1)
        soil_density_layer2_resampled_grids.append(resampled_soil_density_layer2)
        soil_density_layer3_resampled_grids.append(resampled_soil_density_layer3)
        
        Wcr_FRACT_layer1_resampled_grids.append(resampled_Wcr_FRACT_layer1)
        Wcr_FRACT_layer2_resampled_grids.append(resampled_Wcr_FRACT_layer2)
        Wcr_FRACT_layer3_resampled_grids.append(resampled_Wcr_FRACT_layer3)
        
        wp_fract_layer1_resampled_grids.append(resampled_wp_layer1)
        wp_fract_layer2_resampled_grids.append(resampled_wp_layer2)
        wp_fract_layer3_resampled_grids.append(resampled_wp_layer3)
        
        Wpwp_FRACT_layer1_resampled_grids.append(resampled_Wpwp_FRACT_layer1)
        Wpwp_FRACT_layer2_resampled_grids.append(resampled_Wpwp_FRACT_layer2)
        Wpwp_FRACT_layer3_resampled_grids.append(resampled_Wpwp_FRACT_layer3)
        
        rough_resampled_grids.append(resampled_rough)
        
        snow_rough_resampled_grids.append(resampled_snow_rough)
    
    # ======================= reshape =======================
    reshape_func = lambda list_data: np.reshape(list_data, lon_list_level1_2D.shape)
    
    depth_layer1_resampled_grids_2D = reshape_func(depth_layer1_resampled_grids)
    depth_layer2_resampled_grids_2D = reshape_func(depth_layer2_resampled_grids)
    depth_layer3_resampled_grids_2D = reshape_func(depth_layer3_resampled_grids)
    b_infilt_resampled_grids_2D = reshape_func(b_infilt_resampled_grids)
    ksat_layer1_resampled_grids_2D = reshape_func(ksat_layer1_resampled_grids)
    ksat_layer2_resampled_grids_2D = reshape_func(ksat_layer2_resampled_grids)
    ksat_layer3_resampled_grids_2D = reshape_func(ksat_layer3_resampled_grids)
    phi_s_layer1_resampled_grids_2D = reshape_func(phi_s_layer1_resampled_grids)
    phi_s_layer2_resampled_grids_2D = reshape_func(phi_s_layer2_resampled_grids)
    phi_s_layer3_resampled_grids_2D = reshape_func(phi_s_layer3_resampled_grids)
    psis_layer1_resampled_grids_2D = reshape_func(psis_layer1_resampled_grids)
    psis_layer2_resampled_grids_2D = reshape_func(psis_layer2_resampled_grids)
    psis_layer3_resampled_grids_2D = reshape_func(psis_layer3_resampled_grids)
    b_retcurve_layer1_resampled_grids_2D = reshape_func(b_retcurve_layer1_resampled_grids)
    b_retcurve_layer2_resampled_grids_2D = reshape_func(b_retcurve_layer2_resampled_grids)
    b_retcurve_layer3_resampled_grids_2D = reshape_func(b_retcurve_layer3_resampled_grids)
    expt_layer1_resampled_grids_2D = reshape_func(expt_layer1_resampled_grids)
    expt_layer2_resampled_grids_2D = reshape_func(expt_layer2_resampled_grids)
    expt_layer3_resampled_grids_2D = reshape_func(expt_layer3_resampled_grids)
    fc_layer1_resampled_grids_2D = reshape_func(fc_layer1_resampled_grids)
    fc_layer2_resampled_grids_2D = reshape_func(fc_layer2_resampled_grids)
    fc_layer3_resampled_grids_2D = reshape_func(fc_layer3_resampled_grids)
    D4_resampled_grids_2D = reshape_func(D4_resampled_grids)
    c_resampled_grids_2D = reshape_func(c_resampled_grids)
    D1_resampled_grids_2D = reshape_func(D1_resampled_grids)
    D2_resampled_grids_2D = reshape_func(D2_resampled_grids)
    D3_resampled_grids_2D = reshape_func(D3_resampled_grids)
    Dsmax_resampled_grids_2D = reshape_func(Dsmax_resampled_grids)
    Ds_resampled_grids_2D = reshape_func(Ds_resampled_grids)
    Ws_resampled_grids_2D = reshape_func(Ws_resampled_grids)
    init_moist_layer1_resampled_grids_2D = reshape_func(init_moist_layer1_resampled_grids)
    init_moist_layer2_resampled_grids_2D = reshape_func(init_moist_layer2_resampled_grids)
    init_moist_layer3_resampled_grids_2D = reshape_func(init_moist_layer3_resampled_grids)
    elev_resampled_grids_2D = reshape_func(elev_resampled_grids)
    dp_resampled_grids_2D = reshape_func(dp_resampled_grids)
    bubble_layer1_resampled_grids_2D = reshape_func(bubble_layer1_resampled_grids)
    bubble_layer2_resampled_grids_2D = reshape_func(bubble_layer2_resampled_grids)
    bubble_layer3_resampled_grids_2D = reshape_func(bubble_layer3_resampled_grids)
    quartz_layer1_resampled_grids_2D = reshape_func(quartz_layer1_resampled_grids)
    quartz_layer2_resampled_grids_2D = reshape_func(quartz_layer2_resampled_grids)
    quartz_layer3_resampled_grids_2D = reshape_func(quartz_layer3_resampled_grids)
    bulk_density_layer1_resampled_grids_2D = reshape_func(bulk_density_layer1_resampled_grids)
    bulk_density_layer2_resampled_grids_2D = reshape_func(bulk_density_layer2_resampled_grids)
    bulk_density_layer3_resampled_grids_2D = reshape_func(bulk_density_layer3_resampled_grids)
    soil_density_layer1_resampled_grids_2D = reshape_func(soil_density_layer1_resampled_grids)
    soil_density_layer2_resampled_grids_2D = reshape_func(soil_density_layer2_resampled_grids)
    soil_density_layer3_resampled_grids_2D = reshape_func(soil_density_layer3_resampled_grids)
    Wcr_FRACT_layer1_resampled_grids_2D = reshape_func(Wcr_FRACT_layer1_resampled_grids)
    Wcr_FRACT_layer2_resampled_grids_2D = reshape_func(Wcr_FRACT_layer2_resampled_grids)
    Wcr_FRACT_layer3_resampled_grids_2D = reshape_func(Wcr_FRACT_layer3_resampled_grids)
    wp_fract_layer1_resampled_grids_2D = reshape_func(wp_fract_layer1_resampled_grids)
    wp_fract_layer2_resampled_grids_2D = reshape_func(wp_fract_layer2_resampled_grids)
    wp_fract_layer3_resampled_grids_2D = reshape_func(wp_fract_layer3_resampled_grids)
    Wpwp_FRACT_layer1_resampled_grids_2D = reshape_func(Wpwp_FRACT_layer1_resampled_grids)
    Wpwp_FRACT_layer2_resampled_grids_2D = reshape_func(Wpwp_FRACT_layer2_resampled_grids)
    Wpwp_FRACT_layer3_resampled_grids_2D = reshape_func(Wpwp_FRACT_layer3_resampled_grids)
    rough_resampled_grids_2D = reshape_func(rough_resampled_grids)
    snow_rough_resampled_grids_2D = reshape_func(snow_rough_resampled_grids)
    
    # ======================= add data into level1 =======================
    params_dataset_level1.variables["depth"][0, :, :] = depth_layer1_resampled_grids_2D
    params_dataset_level1.variables["depth"][1, :, :] = depth_layer2_resampled_grids_2D
    params_dataset_level1.variables["depth"][2, :, :] = depth_layer3_resampled_grids_2D
    params_dataset_level1.variables["infilt"][:, :] = b_infilt_resampled_grids_2D
    params_dataset_level1.variables["Ksat"][0, :, :] = ksat_layer1_resampled_grids_2D
    params_dataset_level1.variables["Ksat"][1, :, :] = ksat_layer2_resampled_grids_2D
    params_dataset_level1.variables["Ksat"][2, :, :] = ksat_layer3_resampled_grids_2D
    params_dataset_level1.variables["phi_s"][0, :, :] = phi_s_layer1_resampled_grids_2D
    params_dataset_level1.variables["phi_s"][1, :, :] = phi_s_layer2_resampled_grids_2D
    params_dataset_level1.variables["phi_s"][2, :, :] = phi_s_layer3_resampled_grids_2D
    params_dataset_level1.variables["psis"][0, :, :] = psis_layer1_resampled_grids_2D
    params_dataset_level1.variables["psis"][1, :, :] = psis_layer2_resampled_grids_2D
    params_dataset_level1.variables["psis"][2, :, :] = psis_layer3_resampled_grids_2D
    params_dataset_level1.variables["b_retcurve"][0, :, :] = b_retcurve_layer1_resampled_grids_2D
    params_dataset_level1.variables["b_retcurve"][1, :, :] = b_retcurve_layer2_resampled_grids_2D
    params_dataset_level1.variables["b_retcurve"][2, :, :] = b_retcurve_layer3_resampled_grids_2D
    params_dataset_level1.variables["expt"][0, :, :] = expt_layer1_resampled_grids_2D
    params_dataset_level1.variables["expt"][1, :, :] = expt_layer2_resampled_grids_2D
    params_dataset_level1.variables["expt"][2, :, :] = expt_layer3_resampled_grids_2D
    params_dataset_level1.variables["fc"][0, :, :] = fc_layer1_resampled_grids_2D
    params_dataset_level1.variables["fc"][1, :, :] = fc_layer2_resampled_grids_2D
    params_dataset_level1.variables["fc"][2, :, :] = fc_layer3_resampled_grids_2D
    params_dataset_level1.variables["D4"][:, :] = D4_resampled_grids_2D
    params_dataset_level1.variables["c"][:, :] = c_resampled_grids_2D
    params_dataset_level1.variables["D1"][:, :] = D1_resampled_grids_2D
    params_dataset_level1.variables["D2"][:, :] = D2_resampled_grids_2D
    params_dataset_level1.variables["D3"][:, :] = D3_resampled_grids_2D
    params_dataset_level1.variables["Dsmax"][:, :] = Dsmax_resampled_grids_2D
    params_dataset_level1.variables["Ds"][:, :] = Ds_resampled_grids_2D
    params_dataset_level1.variables["Ws"][:, :] = Ws_resampled_grids_2D
    params_dataset_level1.variables["init_moist"][0, :, :] = init_moist_layer1_resampled_grids_2D
    params_dataset_level1.variables["init_moist"][1, :, :] = init_moist_layer2_resampled_grids_2D
    params_dataset_level1.variables["init_moist"][2, :, :] = init_moist_layer3_resampled_grids_2D
    params_dataset_level1.variables["elev"][:, :] = elev_resampled_grids_2D
    params_dataset_level1.variables["dp"][:, :] = dp_resampled_grids_2D
    params_dataset_level1.variables["bubble"][0, :, :] = bubble_layer1_resampled_grids_2D
    params_dataset_level1.variables["bubble"][1, :, :] = bubble_layer2_resampled_grids_2D
    params_dataset_level1.variables["bubble"][2, :, :] = bubble_layer3_resampled_grids_2D
    params_dataset_level1.variables["quartz"][0, :, :] = quartz_layer1_resampled_grids_2D
    params_dataset_level1.variables["quartz"][1, :, :] = quartz_layer2_resampled_grids_2D
    params_dataset_level1.variables["quartz"][2, :, :] = quartz_layer3_resampled_grids_2D
    params_dataset_level1.variables["bulk_density"][0, :, :] = bulk_density_layer1_resampled_grids_2D
    params_dataset_level1.variables["bulk_density"][1, :, :] = bulk_density_layer2_resampled_grids_2D
    params_dataset_level1.variables["bulk_density"][2, :, :] = bulk_density_layer3_resampled_grids_2D
    params_dataset_level1.variables["soil_density"][0,  :, :] = soil_density_layer1_resampled_grids_2D
    params_dataset_level1.variables["soil_density"][1,  :, :] = soil_density_layer2_resampled_grids_2D
    params_dataset_level1.variables["soil_density"][2,  :, :] = soil_density_layer3_resampled_grids_2D
    params_dataset_level1.variables["Wcr_FRACT"][0, :, :] = Wcr_FRACT_layer1_resampled_grids_2D
    params_dataset_level1.variables["Wcr_FRACT"][1, :, :] = Wcr_FRACT_layer2_resampled_grids_2D
    params_dataset_level1.variables["Wcr_FRACT"][2, :, :] = Wcr_FRACT_layer3_resampled_grids_2D
    params_dataset_level1.variables["wp"][0, :, :] = wp_fract_layer1_resampled_grids_2D
    params_dataset_level1.variables["wp"][1, :, :] = wp_fract_layer2_resampled_grids_2D
    params_dataset_level1.variables["wp"][2, :, :] = wp_fract_layer3_resampled_grids_2D
    params_dataset_level1.variables["Wpwp_FRACT"][0, :, :] = Wpwp_FRACT_layer1_resampled_grids_2D
    params_dataset_level1.variables["Wpwp_FRACT"][1, :, :] = Wpwp_FRACT_layer2_resampled_grids_2D
    params_dataset_level1.variables["Wpwp_FRACT"][2, :, :] = Wpwp_FRACT_layer3_resampled_grids_2D
    params_dataset_level1.variables["rough"][:, :] = rough_resampled_grids_2D
    params_dataset_level1.variables["snow_rough"][:, :] = snow_rough_resampled_grids_2D

    return params_dataset_level1, searched_grids_index


