# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import os
import numpy as np
import json
from .tools.utilities import *
from .tools.params_func.createParametersDataset import createParametersDataset
from .tools.params_func.TansferFunction import TF_VIC
from .tools.params_func.Scaling_operator import Scaling_operator
from .tools.params_func.params_set import *
from .tools.dpc_func.basin_grid_func import *
from .bulid_Domain import cal_mask_frac_area_length
from tqdm import *
from .tools.geo_func import resample, search_grids
from .tools.decoractors import clock_decorator
from copy import deepcopy

@clock_decorator(print_arg_ret=False)
def buildParam_level0(evb_dir, g_list, dpc_VIC_level0, reverse_lat=True,
                      stand_grids_lat=None, stand_grids_lon=None,
                      rows_index=None, cols_index=None):
    print("building Param_level0... ...")
    ## ======================= buildParam_level0_basic =======================
    params_dataset_level0 = buildParam_level0_basic(evb_dir, dpc_VIC_level0, reverse_lat)
    
    ## ======================= buildParam_level0_by_g =======================
    params_dataset_level0, stand_grids_lat, stand_grids_lon, rows_index, cols_index = buildParam_level0_by_g(params_dataset_level0, g_list, dpc_VIC_level0, reverse_lat,
                                                                                                             stand_grids_lat, stand_grids_lon,
                                                                                                             rows_index, cols_index)

    return params_dataset_level0, stand_grids_lat, stand_grids_lon, rows_index, cols_index


@clock_decorator(print_arg_ret=False)
def buildParam_level0_basic(evb_dir, dpc_VIC_level0, reverse_lat=True):
    # grids_map_array
    lon_list_level0, lat_list_level0, lon_map_index_level0, lat_map_index_level0 = grids_array_coord_map(dpc_VIC_level0.grid_shp, reverse_lat=reverse_lat)  #* all lat set as reverse if True
    
    ## ====================== create parameter ======================
    params_dataset_level0 = createParametersDataset(evb_dir.params_dataset_level0_path, lat_list_level0, lon_list_level0)
    
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
    
    return params_dataset_level0


@clock_decorator(print_arg_ret=False)
def buildParam_level0_by_g(params_dataset_level0, g_list, dpc_VIC_level0, reverse_lat=True, stand_grids_lat=None, stand_grids_lon=None, rows_index=None, cols_index=None):
    """
    # calibrate: MPR: PTF + Scaling (calibrate for scaling coefficient)
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
        
    # TODO Q1: different layer have different global params? Ksat: 3 or 9?
    """
    ## ====================== get grid_shp and basin_shp ======================
    grid_shp_level0 = deepcopy(dpc_VIC_level0.grid_shp)
    grids_num = len(grid_shp_level0.index)
    
    # these variables can be input outside
    if stand_grids_lat is None:
        stand_grids_lat, stand_grids_lon = createStand_grids_lat_lon_from_gridshp(grid_shp_level0, grid_res=None, reverse_lat=reverse_lat)
    
    if rows_index is None:
        rows_index, cols_index = gridshp_index_to_grid_array_index(grid_shp_level0, stand_grids_lat, stand_grids_lon)
    
    ## ======================= level0: Transfer function =======================
    # TF
    tf_VIC = TF_VIC()
    
    # only set the params which should be scaling (aggregation), other params such as run_cell, grid_cell, off_gmt..., will not be set here
    # depth, m
    total_depth = tf_VIC.total_depth(CONUS_layers_total_depth, g_list[0])
    depths = tf_VIC.depth(total_depth, g_list[1], g_list[2])
    
    grid_array_depth_layer1 = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
    grid_array_depth_layer2 = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
    grid_array_depth_layer3 = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
    
    grid_array_depth_layer1 =  assignValue_for_grid_array(grid_array_depth_layer1, np.full((grids_num, ), fill_value=depths[0]), rows_index, cols_index)
    grid_array_depth_layer2 =  assignValue_for_grid_array(grid_array_depth_layer2, np.full((grids_num, ), fill_value=depths[1]), rows_index, cols_index)
    grid_array_depth_layer3 =  assignValue_for_grid_array(grid_array_depth_layer3, np.full((grids_num, ), fill_value=depths[2]), rows_index, cols_index)

    params_dataset_level0.variables["depth"][0, :, :] = grid_array_depth_layer1
    params_dataset_level0.variables["depth"][1, :, :] = grid_array_depth_layer2
    params_dataset_level0.variables["depth"][2, :, :] = grid_array_depth_layer3
    
    #*vertical aggregation for three soil layers
    num1, num2 = g_list[1], g_list[2] # g1, g2 is the num1, num2
    
    depth_layer1_start = 0
    depth_layer1_end = num1
    depth_layer2_start = num1
    depth_layer2_end = num2
    depth_layer3_start = num2
    depth_layer3_end = CONUS_layers_num
    
    # ele_std, m (same as StrmDem)
    grid_array_SrtmDEM_std_Value = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
    grid_array_SrtmDEM_std_Value = assignValue_for_grid_array(grid_array_SrtmDEM_std_Value, grid_shp_level0.loc[:, "SrtmDEM_std_Value"], rows_index, cols_index)
    
    # b_infilt, N/A
    params_dataset_level0.variables["infilt"][:, :] = tf_VIC.b_infilt(grid_array_SrtmDEM_std_Value, g_list[3], g_list[4])
    
    # sand, clay, silt, %
    grid_array_sand_layer1, grid_array_silt_layer1, grid_array_clay_layer1 = cal_ssc_percentile_grid_array(grid_shp_level0, depth_layer1_start, depth_layer1_end, stand_grids_lat, stand_grids_lon, rows_index, cols_index)
    grid_array_sand_layer2, grid_array_silt_layer2, grid_array_clay_layer2 = cal_ssc_percentile_grid_array(grid_shp_level0, depth_layer2_start, depth_layer2_end, stand_grids_lat, stand_grids_lon, rows_index, cols_index)
    grid_array_sand_layer3, grid_array_silt_layer3, grid_array_clay_layer3 = cal_ssc_percentile_grid_array(grid_shp_level0, depth_layer3_start, depth_layer3_end, stand_grids_lat, stand_grids_lon, rows_index, cols_index)
    
    # ksat, mm/s -> mm/day (VIC requirement)
    grid_array_ksat_layer1 = tf_VIC.ksat(grid_array_sand_layer1, grid_array_clay_layer1, g_list[5], g_list[6], g_list[7])
    grid_array_ksat_layer2 = tf_VIC.ksat(grid_array_sand_layer2, grid_array_clay_layer2, g_list[5], g_list[6], g_list[7])
    grid_array_ksat_layer3 = tf_VIC.ksat(grid_array_sand_layer3, grid_array_clay_layer3, g_list[5], g_list[6], g_list[7])
    
    unit_factor_ksat = 60 * 60 * 24
    
    params_dataset_level0.variables["Ksat"][0, :, :] = grid_array_ksat_layer1 * unit_factor_ksat
    params_dataset_level0.variables["Ksat"][1, :, :] = grid_array_ksat_layer2 * unit_factor_ksat
    params_dataset_level0.variables["Ksat"][2, :, :] = grid_array_ksat_layer3 * unit_factor_ksat
    
    # mean slope, % (m/m)
    grid_array_mean_slope = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
    grid_array_mean_slope = assignValue_for_grid_array(grid_array_mean_slope, grid_shp_level0.loc[:, "SrtmDEM_mean_slope_Value%"], rows_index, cols_index)
    
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
    grid_array_D4 = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
    grid_array_D4 =  assignValue_for_grid_array(grid_array_D4, np.full((grids_num, ), fill_value=tf_VIC.D4(g_list[20])), rows_index, cols_index)
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
    grid_array_SrtmDEM_mean_Value = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
    grid_array_SrtmDEM_mean_Value =  assignValue_for_grid_array(grid_array_SrtmDEM_mean_Value, grid_shp_level0.loc[:, "SrtmDEM_mean_Value"], rows_index, cols_index)
    
    params_dataset_level0.variables["elev"][:, :] = grid_array_SrtmDEM_mean_Value
    
    # dp, m, typically is 4m
    grid_array_dp = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
    grid_array_dp = assignValue_for_grid_array(grid_array_dp, np.full((grids_num, ), fill_value=tf_VIC.dp(g_list[24])), rows_index, cols_index)
    
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
    grid_array_bd_layer1 = cal_bd_grid_array(grid_shp_level0, depth_layer1_start, depth_layer1_end, stand_grids_lat, stand_grids_lon, rows_index, cols_index)
    grid_array_bd_layer2 = cal_bd_grid_array(grid_shp_level0, depth_layer2_start, depth_layer2_end, stand_grids_lat, stand_grids_lon, rows_index, cols_index)
    grid_array_bd_layer3 = cal_bd_grid_array(grid_shp_level0, depth_layer3_start, depth_layer3_end, stand_grids_lat, stand_grids_lon, rows_index, cols_index)
    
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
    
    grid_array_soil_density_layer1 = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
    grid_array_soil_density_layer2 = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
    grid_array_soil_density_layer3 = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
    
    grid_array_soil_density_layer1 =  assignValue_for_grid_array(grid_array_soil_density_layer1, np.full((grids_num, ), fill_value=soil_density_layer1), rows_index, cols_index)
    grid_array_soil_density_layer2 =  assignValue_for_grid_array(grid_array_soil_density_layer2, np.full((grids_num, ), fill_value=soil_density_layer2), rows_index, cols_index)
    grid_array_soil_density_layer3 =  assignValue_for_grid_array(grid_array_soil_density_layer3, np.full((grids_num, ), fill_value=soil_density_layer3), rows_index, cols_index)
    
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
    grid_array_rough = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
    grid_array_rough = assignValue_for_grid_array(grid_array_rough, np.full((grids_num, ), fill_value=tf_VIC.rough(g_list[35])), rows_index, cols_index)
    
    params_dataset_level0.variables["rough"][:, :] = grid_array_rough
    
    # snow rough, m
    grid_array_snow_rough = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
    grid_array_snow_rough = assignValue_for_grid_array(grid_array_snow_rough, np.full((grids_num, ), fill_value=tf_VIC.snow_rough(g_list[36])), rows_index, cols_index)
    
    params_dataset_level0.variables["snow_rough"][:, :] = grid_array_snow_rough
    
    return params_dataset_level0, stand_grids_lat, stand_grids_lon, rows_index, cols_index


@clock_decorator(print_arg_ret=False)
def buildParam_level1(evb_dir, dpc_VIC_level1, reverse_lat=True, domain_dataset=None,
                      stand_grids_lat=None, stand_grids_lon=None,
                      rows_index=None, cols_index=None):
    
    print("building Param_level1... ...")
    ## ====================== get grid_shp and basin_shp ======================
    grid_shp_level1 = deepcopy(dpc_VIC_level1.grid_shp)
    grids_num = len(grid_shp_level1.index)
    
    # grids_map_array, lon/lat_list is from dpc.grid_shp, corresponding to these data
    lon_list_level1, lat_list_level1, lon_map_index_level1, lat_map_index_level1 = grids_array_coord_map(grid_shp_level1, reverse_lat=reverse_lat)  #* all lat set as reverse

    # these variables can be input outside
    # stand grids must be complete rectengle, so, it may be the superset of the lon/lat_list (potentially contains grids with no data)
    if stand_grids_lat is None:
        stand_grids_lat, stand_grids_lon = createStand_grids_lat_lon_from_gridshp(grid_shp_level1, grid_res=None, reverse_lat=reverse_lat)
    
    if rows_index is None:
        rows_index, cols_index = gridshp_index_to_grid_array_index(grid_shp_level1, stand_grids_lat, stand_grids_lon)
    
    ## ====================== create parameter ======================
    params_dataset_level1 = createParametersDataset(evb_dir.params_dataset_level1_path, lat_list_level1, lon_list_level1)
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
    
    mask = mask.astype(int)
    params_dataset_level1.variables["run_cell"][:, :] = mask
    
    # grid_cell
    grid_array_grid_cell = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=int, missing_value=-9999)
    grid_array_grid_cell =  assignValue_for_grid_array(grid_array_grid_cell, np.arange(1, len(grid_shp_level1.index) + 1), rows_index, cols_index)
    params_dataset_level1.variables["grid_cell"][:, :] = grid_array_grid_cell

    # off_gmt, hours
    grid_array_off_gmt = tf_VIC.off_gmt(grid_array_lons)
    params_dataset_level1.variables["off_gmt"][:, :] = grid_array_off_gmt
    
    # avg_T, C
    grid_array_avg_T = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
    grid_array_avg_T =  assignValue_for_grid_array(grid_array_avg_T, grid_shp_level1.loc[:, "stl_all_layers_mean_Value"], rows_index, cols_index)
    params_dataset_level1.variables["avg_T"][:, :] = grid_array_avg_T

    # annual_prec, mm
    grid_array_annual_P = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
    grid_array_annual_P =  assignValue_for_grid_array(grid_array_annual_P, grid_shp_level1.loc[:, "annual_P_in_src_grid_Value"], rows_index, cols_index)
    params_dataset_level1.variables["annual_prec"][:, :] = grid_array_annual_P
    
    # resid_moist, fraction, set as 0
    grid_array_resid_moist = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
    grid_array_resid_moist =  assignValue_for_grid_array(grid_array_resid_moist, np.full((grids_num, ), fill_value=0), rows_index, cols_index)
    params_dataset_level1.variables["resid_moist"][0, :, :] = grid_array_resid_moist
    params_dataset_level1.variables["resid_moist"][1, :, :] = grid_array_resid_moist
    params_dataset_level1.variables["resid_moist"][2, :, :] = grid_array_resid_moist
    
    # fs_active, bool, whether the frozen soil algorithm is activated
    grid_array_fs_active = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=int, missing_value=-9999)
    grid_array_fs_active =  assignValue_for_grid_array(grid_array_fs_active, np.full((grids_num, ), fill_value=0), rows_index, cols_index)
    params_dataset_level1.variables["fs_active"][:, :] = grid_array_fs_active
    
    # Nveg, int
    grid_array_Nveg = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=int, missing_value=-9999)
    grid_array_Nveg =  assignValue_for_grid_array(grid_array_Nveg, grid_shp_level1["umd_lc_original_Value"].apply(lambda row: len(list(set(row)))), rows_index, cols_index)
    params_dataset_level1.variables["Nveg"][:, :] = grid_array_Nveg
    
    # Cv, fraction
    for i in veg_class_list:
        grid_shp_level1_ = deepcopy(grid_shp_level1)
        grid_shp_level1_[f"umd_lc_{i}_veg_index"] = grid_shp_level1_.loc[:, "umd_lc_original_Value"].apply(lambda row: np.where(np.array(row)==i)[0])
        
        grid_array_i_veg_Cv = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
        grid_array_i_veg_Cv =  assignValue_for_grid_array(grid_array_i_veg_Cv, grid_shp_level1_.apply(lambda row: sum(np.array(row["umd_lc_original_Cv"])[row[f"umd_lc_{i}_veg_index"]]), axis=1), rows_index, cols_index)
        
        params_dataset_level1.variables["Cv"][i, :, :] = grid_array_i_veg_Cv
    
    # read veg params, veg_params_json is a lookup_table
    veg_params_json = read_veg_param_json()
    # with open(evb_dir.veg_param_json_path, 'r') as f:
    #     veg_params_json = json.load(f)
        # veg_params_json = veg_params_json["classAttributes"]
        # veg_keys = [v["class"] for v in veg_params_json]
        # veg_params = [v["properties"] for v in veg_params_json]
        # veg_params_json = dict(zip(veg_keys, veg_params))
        
    # root_depth, m; root_fract, fraction
    for i in veg_class_list:
        for j in root_zone_list:
            grid_array_i_veg_j_zone_root_depth = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
            grid_array_i_veg_j_zone_root_depth =  assignValue_for_grid_array(grid_array_i_veg_j_zone_root_depth, np.full((grids_num, ), fill_value=float(veg_params_json[f"{i}"][f"rootd{j}"])), rows_index, cols_index)
            params_dataset_level1.variables["root_depth"][i, j-1, :, :] = grid_array_i_veg_j_zone_root_depth  # j-1: root_zone_list start from 1
            
            grid_array_i_veg_j_zone_root_fract = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
            grid_array_i_veg_j_zone_root_fract =  assignValue_for_grid_array(grid_array_i_veg_j_zone_root_fract, np.full((grids_num, ), fill_value=float(veg_params_json[f"{i}"][f"rootfr{j}"])), rows_index, cols_index)
            params_dataset_level1.variables["root_fract"][i, j-1, :, :] = grid_array_i_veg_j_zone_root_fract

    # rarc, s/m; rmin, s/m
    for i in veg_class_list:
        grid_array_i_veg_rarc = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
        grid_array_i_veg_rarc =  assignValue_for_grid_array(grid_array_i_veg_rarc, np.full((grids_num, ), fill_value=float(veg_params_json[f"{i}"][f"rarc"])), rows_index, cols_index)
        params_dataset_level1.variables["rarc"][i, :, :] = grid_array_i_veg_rarc
        
        grid_array_i_veg_rmin = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
        grid_array_i_veg_rmin =  assignValue_for_grid_array(grid_array_i_veg_rmin, np.full((grids_num, ), fill_value=float(veg_params_json[f"{i}"][f"rmin"])), rows_index, cols_index)
        params_dataset_level1.variables["rmin"][i, :, :] = grid_array_i_veg_rmin
    
    # overstory, N/A, bool
    # wind_h, m, adjust wind height value if overstory is true (overstory == 1, wind_h=vegHeight+10, else wind_h=vegHeight+2)
    for i in veg_class_list:
        grid_array_i_veg_height = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
        grid_array_i_veg_height =  assignValue_for_grid_array(grid_array_i_veg_height, np.full((grids_num, ), fill_value=float(veg_params_json[f"{i}"][f"h"])), rows_index, cols_index)
        
        grid_array_i_veg_overstory = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=int, missing_value=-9999)
        grid_array_i_veg_overstory =  assignValue_for_grid_array(grid_array_i_veg_overstory, np.full((grids_num, ), fill_value=int(veg_params_json[f"{i}"][f"overstory"])), rows_index, cols_index)
        
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
            grid_array_i_veg_j_month_displacement = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
            grid_array_i_veg_j_month_displacement =  assignValue_for_grid_array(grid_array_i_veg_j_month_displacement, np.full((grids_num, ), fill_value=float(veg_params_json[f"{i}"][f"veg_displacement_month_{j}"])), rows_index, cols_index)
            params_dataset_level1.variables["displacement"][i, j-1, :, :] = grid_array_i_veg_j_month_displacement   # j-1: month_list start from 1
            
            grid_array_i_veg_j_month_veg_rough = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
            grid_array_i_veg_j_month_veg_rough =  assignValue_for_grid_array(grid_array_i_veg_j_month_veg_rough, np.full((grids_num, ), fill_value=float(veg_params_json[f"{i}"][f"veg_rough_month_{j}"])), rows_index, cols_index)
            params_dataset_level1.variables["veg_rough"][i, j-1, :, :] = grid_array_i_veg_j_month_veg_rough
    
    # RGL, W/m2; rad_atten, fract; wind_atten, fract; trunk_ratio, fract
    for i in veg_class_list:
        grid_array_i_veg_RGL = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
        grid_array_i_veg_RGL =  assignValue_for_grid_array(grid_array_i_veg_RGL, np.full((grids_num, ), fill_value=float(veg_params_json[f"{i}"][f"rgl"])), rows_index, cols_index)
        params_dataset_level1.variables["RGL"][i, :, :] = grid_array_i_veg_RGL

        grid_array_i_veg_rad_atten = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
        grid_array_i_veg_rad_atten =  assignValue_for_grid_array(grid_array_i_veg_rad_atten, np.full((grids_num, ), fill_value=float(veg_params_json[f"{i}"][f"rad_atn"])), rows_index, cols_index)
        params_dataset_level1.variables["rad_atten"][i, :, :] = grid_array_i_veg_rad_atten
    
        grid_array_i_veg_wind_atten = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
        grid_array_i_veg_wind_atten =  assignValue_for_grid_array(grid_array_i_veg_wind_atten, np.full((grids_num, ), fill_value=float(veg_params_json[f"{i}"][f"wnd_atn"])), rows_index, cols_index)
        params_dataset_level1.variables["wind_atten"][i, :, :] = grid_array_i_veg_wind_atten

        grid_array_i_veg_trunk_ratio = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
        grid_array_i_veg_trunk_ratio =  assignValue_for_grid_array(grid_array_i_veg_trunk_ratio, np.full((grids_num, ), fill_value=float(veg_params_json[f"{i}"][f"trnk_r"])), rows_index, cols_index)
        params_dataset_level1.variables["trunk_ratio"][i, :, :] = grid_array_i_veg_trunk_ratio
    
    # LAI, fraction or m2/m2; albedo, fraction; fcanopy, fraction
    for i in veg_class_list:
        for j in month_list:
            grid_shp_level1_ = deepcopy(grid_shp_level1)
            
            # LAI
            grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_LAI"] = grid_shp_level1_.apply(lambda row: np.array(row[f"MODIS_LAI_original_Value_month{j}"])[np.where(np.array(row.umd_lc_original_Value)==i)[0]], axis=1)
            grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_LAI"] = grid_shp_level1_.loc[:, f"MODIS_{i}_veg_{j}_month_LAI"].apply(lambda row: np.mean(row) if len(row) != 0 else 0)
            
            grid_array_i_veg_j_month_LAI = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
            grid_array_i_veg_j_month_LAI =  assignValue_for_grid_array(grid_array_i_veg_j_month_LAI, grid_shp_level1_.loc[:, f"MODIS_{i}_veg_{j}_month_LAI"], rows_index, cols_index)
            params_dataset_level1.variables["LAI"][i, j-1, :, :] = grid_array_i_veg_j_month_LAI   # j-1: month_list start from 1
            
            # BSA, albedo
            grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_BSA"] = grid_shp_level1_.apply(lambda row: np.array(row[f"MODIS_BSA_original_Value_month{j}"])[np.where(np.array(row.umd_lc_original_Value)==i)[0]], axis=1)
            grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_BSA"] = grid_shp_level1_.loc[:, f"MODIS_{i}_veg_{j}_month_BSA"].apply(lambda row: np.mean(row) if len(row) != 0 else 0)
            
            grid_array_i_veg_j_month_BSA = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
            grid_array_i_veg_j_month_BSA =  assignValue_for_grid_array(grid_array_i_veg_j_month_BSA, grid_shp_level1_.loc[:, f"MODIS_{i}_veg_{j}_month_BSA"], rows_index, cols_index)
            params_dataset_level1.variables["albedo"][i, j-1, :, :] = grid_array_i_veg_j_month_BSA   # j-1: month_list start from 1
            
            # fcanopy, ((NDVI-NDVI_min)/(NDVI_max-NDVI_min))**2
            grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_NDVI"] = grid_shp_level1_.apply(lambda row: np.array(row[f"MODIS_NDVI_original_Value_month{j}"])[np.where(np.array(row.umd_lc_original_Value)==i)[0]], axis=1)
            grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_NDVI"] = grid_shp_level1_.loc[:, f"MODIS_{i}_veg_{j}_month_NDVI"].apply(lambda row: np.mean(row) if len(row) != 0 else 0)
            grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_NDVI"] = grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_NDVI"] * 0.0001
            NDVI = grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_NDVI"]
            
            grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_NDVI_max"] = grid_shp_level1_.apply(lambda row: np.array(row[f"MODIS_NDVI_max_original_Value_month{j}"])[np.where(np.array(row.umd_lc_original_Value)==i)[0]], axis=1)
            grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_NDVI_max"] = grid_shp_level1_.loc[:, f"MODIS_{i}_veg_{j}_month_NDVI_max"].apply(lambda row: np.mean(row) if len(row) != 0 else 0)
            grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_NDVI_max"] = grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_NDVI_max"] * 0.0001
            NDVI_max = grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_NDVI_max"]
            
            grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_NDVI_min"] = grid_shp_level1_.apply(lambda row: np.array(row[f"MODIS_NDVI_min_original_Value_month{j}"])[np.where(np.array(row.umd_lc_original_Value)==i)[0]], axis=1)
            grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_NDVI_min"] = grid_shp_level1_.loc[:, f"MODIS_{i}_veg_{j}_month_NDVI_min"].apply(lambda row: np.mean(row) if len(row) != 0 else 0)
            grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_NDVI_min"] = grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_NDVI_min"] * 0.0001
            NDVI_min = grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_NDVI_min"]
            
            fcanopy = ((NDVI-NDVI_min)/(NDVI_max-NDVI_min)) ** 2
            fcanopy[np.isnan(fcanopy)] = 0
            
            grid_array_i_veg_j_month_fcanopy = createEmptyArray_from_gridshp(stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan)
            grid_array_i_veg_j_month_fcanopy =  assignValue_for_grid_array(grid_array_i_veg_j_month_fcanopy, fcanopy, rows_index, cols_index)
            params_dataset_level1.variables["fcanopy"][i, j-1, :, :] = grid_array_i_veg_j_month_fcanopy   # j-1: month_list start from 1
            
    return params_dataset_level1, stand_grids_lat, stand_grids_lon, rows_index, cols_index
    

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
    
    # transfer into bool index
    searched_grids_bool_index = searched_grids_index_to_bool_index(searched_grids_index, lat_list_level0, lon_list_level0)
    
    return searched_grids_index, searched_grids_bool_index


@clock_decorator(print_arg_ret=False)
def scaling_level0_to_level1(params_dataset_level0, params_dataset_level1, searched_grids_bool_index=None):
    print("scaling Param_level0 to Param_level1... ...")
    
    lon_list_level1, lat_list_level1 = params_dataset_level1.variables["lon"][:], params_dataset_level1.variables["lat"][:]
    lon_list_level1 = np.ma.filled(lon_list_level1, fill_value=np.NAN)
    lat_list_level1 = np.ma.filled(lat_list_level1, fill_value=np.NAN)
    
    # search grids
    if searched_grids_bool_index is None:
        searched_grids_index, searched_grids_bool_index = scaling_level0_to_level1_search_grids(params_dataset_level0, params_dataset_level1)
    
    # ======================= scaling (resample) =======================
    scaling_operator = Scaling_operator()
    
    # resample func
    search_and_resample_func_2d = lambda scaling_func, varibale_name: np.array([scaling_func(params_dataset_level0.variables[varibale_name][searched_grid_bool_index[0], searched_grid_bool_index[1]].flatten()) for searched_grid_bool_index in searched_grids_bool_index]).reshape((len(lat_list_level1), len(lon_list_level1)))
    search_and_resample_func_3d = lambda scaling_func, varibale_name, first_dim: np.array([scaling_func(params_dataset_level0.variables[varibale_name][first_dim, searched_grid_bool_index[0], searched_grid_bool_index[1]].flatten()) for searched_grid_bool_index in searched_grids_bool_index]).reshape((len(lat_list_level1), len(lon_list_level1)))
    
    # depth, m
    params_dataset_level1.variables["depth"][0, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "depth", 0)
    params_dataset_level1.variables["depth"][1, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "depth", 1)
    params_dataset_level1.variables["depth"][2, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "depth", 2)
    
    # b_infilt, /NA
    params_dataset_level1.variables["infilt"][:, :] = search_and_resample_func_2d(scaling_operator.Arithmetic_mean, "infilt")
    
    # ksat, mm/s -> mm/day (VIC requirement)
    params_dataset_level1.variables["Ksat"][0, :, :] = search_and_resample_func_3d(scaling_operator.Harmonic_mean, "Ksat", 0)
    params_dataset_level1.variables["Ksat"][1, :, :] = search_and_resample_func_3d(scaling_operator.Harmonic_mean, "Ksat", 1)
    params_dataset_level1.variables["Ksat"][2, :, :] = search_and_resample_func_3d(scaling_operator.Harmonic_mean, "Ksat", 2)
    
    # phi_s, m3/m3 or mm/mm
    params_dataset_level1.variables["phi_s"][0, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "phi_s", 0)
    params_dataset_level1.variables["phi_s"][1, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "phi_s", 1)
    params_dataset_level1.variables["phi_s"][2, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "phi_s", 2)
    
    # psis, kPa/cm-H2O
    params_dataset_level1.variables["psis"][0, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "psis", 0)
    params_dataset_level1.variables["psis"][1, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "psis", 1)
    params_dataset_level1.variables["psis"][2, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "psis", 2)
    
    # b_retcurve, /NA
    params_dataset_level1.variables["b_retcurve"][0, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "b_retcurve", 0)
    params_dataset_level1.variables["b_retcurve"][1, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "b_retcurve", 1)
    params_dataset_level1.variables["b_retcurve"][2, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "b_retcurve", 2)
    
    # expt, /NA
    params_dataset_level1.variables["expt"][0, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "expt", 0)
    params_dataset_level1.variables["expt"][1, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "expt", 1)
    params_dataset_level1.variables["expt"][2, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "expt", 2)
    
    # fc, % or m3/m3
    params_dataset_level1.variables["fc"][0, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "fc", 0)
    params_dataset_level1.variables["fc"][1, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "fc", 1)
    params_dataset_level1.variables["fc"][2, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "fc", 2)
    
    # D4, /NA, same as c, typically is 2
    params_dataset_level1.variables["D4"][:, :] = search_and_resample_func_2d(scaling_operator.Arithmetic_mean, "D4")
    
    # cexpt
    params_dataset_level1.variables["c"][:, :] = search_and_resample_func_2d(scaling_operator.Arithmetic_mean, "c")
    
    # D1 ([day^-1]), D2 ([day^-D4])
    params_dataset_level1.variables["D1"][:, :] = search_and_resample_func_2d(scaling_operator.Harmonic_mean, "D1")
    params_dataset_level1.variables["D2"][:, :] = search_and_resample_func_2d(scaling_operator.Harmonic_mean, "D2")
    
    # D3 ([mm])
    params_dataset_level1.variables["D3"][:, :] = search_and_resample_func_2d(scaling_operator.Arithmetic_mean, "D3")
    
    # Dsmax, mm or mm/day
    params_dataset_level1.variables["Dsmax"][:, :] = search_and_resample_func_2d(scaling_operator.Harmonic_mean, "Dsmax")
    
    # Ds, [day^-D4] or fraction
    params_dataset_level1.variables["Ds"][:, :] = search_and_resample_func_2d(scaling_operator.Harmonic_mean, "Ds")
    
    # Ws, fraction
    params_dataset_level1.variables["Ws"][:, :] = search_and_resample_func_2d(scaling_operator.Arithmetic_mean, "Ws")
    
    # init_moist, mm
    params_dataset_level1.variables["init_moist"][0, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "init_moist", 0)
    params_dataset_level1.variables["init_moist"][1 :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "init_moist", 1)
    params_dataset_level1.variables["init_moist"][2, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "init_moist", 2)
    
    # elev, m
    params_dataset_level1.variables["elev"][:, :] = search_and_resample_func_2d(scaling_operator.Arithmetic_mean, "elev")
    
    # dp, m, typically is 4m
    params_dataset_level1.variables["dp"][:, :] = search_and_resample_func_2d(scaling_operator.Arithmetic_mean, "dp")
    
    # bubble, cm
    params_dataset_level1.variables["bubble"][0, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "bubble", 0)
    params_dataset_level1.variables["bubble"][1, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "bubble", 1)
    params_dataset_level1.variables["bubble"][2, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "bubble", 2)
    
    # quartz, N/A
    params_dataset_level1.variables["quartz"][0, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "quartz", 0)
    params_dataset_level1.variables["quartz"][1, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "quartz", 1)
    params_dataset_level1.variables["quartz"][2, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "quartz", 2)
    
    # bulk_density, kg/m3 or mm
    params_dataset_level1.variables["bulk_density"][0, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "bulk_density", 0)
    params_dataset_level1.variables["bulk_density"][1, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "bulk_density", 1)
    params_dataset_level1.variables["bulk_density"][2, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "bulk_density", 2)
    
    # soil_density, kg/m3
    params_dataset_level1.variables["soil_density"][0, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "soil_density", 0)
    params_dataset_level1.variables["soil_density"][1, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "soil_density", 1)
    params_dataset_level1.variables["soil_density"][2, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "soil_density", 2)
    
    # Wcr_FRACT, fraction
    params_dataset_level1.variables["Wcr_FRACT"][0, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "Wcr_FRACT", 0)
    params_dataset_level1.variables["Wcr_FRACT"][1, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "Wcr_FRACT", 1)
    params_dataset_level1.variables["Wcr_FRACT"][2, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "Wcr_FRACT", 2)
    
    # wp, computed field capacity [frac]
    params_dataset_level1.variables["wp"][0, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "wp", 0)
    params_dataset_level1.variables["wp"][1, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "wp", 1)
    params_dataset_level1.variables["wp"][2, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "wp", 2)
    
    # Wpwp_FRACT, fraction
    params_dataset_level1.variables["Wpwp_FRACT"][0, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "Wpwp_FRACT", 0)
    params_dataset_level1.variables["Wpwp_FRACT"][1, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "Wpwp_FRACT", 1)
    params_dataset_level1.variables["Wpwp_FRACT"][2, :, :] = search_and_resample_func_3d(scaling_operator.Arithmetic_mean, "Wpwp_FRACT", 2)
    
    # rough, m, Surface roughness of bare soil
    params_dataset_level1.variables["rough"][:, :] = search_and_resample_func_2d(scaling_operator.Arithmetic_mean, "rough")
    
    # snow rough, m
    params_dataset_level1.variables["snow_rough"][:, :] = search_and_resample_func_2d(scaling_operator.Arithmetic_mean, "snow_rough")

    return params_dataset_level1, searched_grids_bool_index


