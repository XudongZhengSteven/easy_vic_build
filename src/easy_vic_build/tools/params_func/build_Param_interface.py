# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import numpy as np
from copy import deepcopy
from ..dpc_func.basin_grid_func import createEmptyArray_from_gridshp, assignValue_for_grid_array
from ..dpc_func.basin_grid_func import createStand_grids_lat_lon_from_gridshp, gridshp_index_to_grid_array_index, createEmptyArray_and_assignValue_from_gridshp
from ..params_func.createParametersDataset import createParametersDataset
from ..params_func.params_set import *
from ..utilities import read_veg_param_json
from ..decoractors import clock_decorator
from ..params_func.TransferFunction import TF_VIC, SoilLayerResampler


class buildParam_level0_interface:
    
    def __init__(
        self,
        evb_dir,
        logger,
        dpc_VIC_level0,
        g_params,
        soillayerresampler: SoilLayerResampler,
        TF_VIC: TF_VIC,
        reverse_lat=True,
        stand_grids_lat_level0=None,
        stand_grids_lon_level0=None,
        rows_index_level0=None,
        cols_index_level0=None,
    ):
        self.evb_dir = evb_dir
        self.logger = logger
        self.dpc_VIC_level0 = dpc_VIC_level0
        self.reverse_lat = reverse_lat
        self.g_params = g_params
        self.tf_VIC = TF_VIC()
        self.soillayerresampler = soillayerresampler
        
        if self.dpc_VIC_level0.get_data_from_cache("merged_grid_shp")[0] is None:
            self.dpc_VIC_level0.merge_grid_data()
            
        self.grid_shp_level0 = deepcopy(self.dpc_VIC_level0.get_data_from_cache("merged_grid_shp")[0])
        self.grids_num_level0 = len(self.grid_shp_level0.index)
        
        self.stand_grids_lat_level0 = stand_grids_lat_level0
        self.stand_grids_lon_level0 = stand_grids_lon_level0
        self.rows_index_level0 = rows_index_level0
        self.cols_index_level0 = cols_index_level0
    
    @clock_decorator(print_arg_ret=False)
    def buildParam_level0_basic(self):
        execution_order = [
            "set_coord_map",
            "create_parameter_dataset",
            "set_dims",
        ]
        
        for fun_name in execution_order:
            fun = getattr(self, fun_name)
            fun()
            
        self.logger.debug(f"execution order: {execution_order}")
        
        return execution_order
    
    @clock_decorator(print_arg_ret=False)
    def buildParam_level0_by_g_tf(self):
        execution_order = [
            "set_depths_vertical_aggregation",
            "set_ele_std",
            "set_b_infilt",
            "set_soil_texture",
            "set_ksat",
            "set_mean_slope",
            "set_phi_s",
            "set_psis",
            "set_b_retcurve",
            "set_expt",
            "set_fc",
            "set_d4",
            "set_cexpt",
            "set_arno_baseflow_layer_num",
            "set_d1",
            "set_d2",
            "set_d3",
            "set_Dsmax",
            "set_Ds",
            "set_Ws",
            "set_init_moist",
            "set_elev",
            "set_dp",
            "set_bubble",
            "set_quartz",
            "set_bulk_density",
            "set_soil_density",
            "set_Wcr_FRACT",
            "set_wp",
            "set_Wpwp_FRACT",
            "set_rough",
            "set_snow_rough",
        ]
        
        for fun_name in execution_order:
            fun = getattr(self, fun_name)
            fun()
        
        self.logger.debug(f"execution order: {execution_order}")
        
        return execution_order
        
    def set_coord_map(self):
        self.logger.debug("setting coord_map... ...")
        
        if self.stand_grids_lat_level0 is None:
            self.stand_grids_lat_level0, self.stand_grids_lon_level0 = createStand_grids_lat_lon_from_gridshp(
                self.grid_shp_level0, grid_res=None, reverse_lat=self.reverse_lat
            )

        if self.rows_index_level0 is None:
            self.rows_index_level0, self.cols_index_level0 = gridshp_index_to_grid_array_index(
                self.grid_shp_level0, self.stand_grids_lat_level0, self.stand_grids_lon_level0
            )
        
    def create_parameter_dataset(self):
        self.logger.debug("creating parameter dataset... ...")
        
        self.params_dataset_level0 = createParametersDataset(
            self.evb_dir.params_dataset_level0_path,
            self.stand_grids_lat_level0,
            self.stand_grids_lon_level0
        )
    
    def set_dims(self):
        # Dimension variables: lat, lon, nlayer, root_zone, veg_class, month
        self.logger.debug("setting dims... ...")
        
        self.params_dataset_level0.variables["lat"][:] = np.array(self.stand_grids_lat_level0)  # 1D array
        self.params_dataset_level0.variables["lon"][:] = np.array(self.stand_grids_lon_level0)  # 1D array
        
        self.nlayer_list = [1, 2, 3]
        self.params_dataset_level0.variables["nlayer"][:] = self.nlayer_list
        
        self.root_zone_list = [1, 2, 3]
        self.params_dataset_level0.variables["root_zone"][:] = self.root_zone_list
        
        self.veg_class_list = list(range(14))
        self.params_dataset_level0.variables["veg_class"][:] = self.veg_class_list
        
        self.month_list = list(range(1, 13))
        self.params_dataset_level0.variables["month"][:] = self.month_list
        
        # lons, lats, 2D array
        self.grid_array_lons_level0, self.grid_array_lats_level0 = np.meshgrid(
            self.params_dataset_level0.variables["lon"][:],
            self.params_dataset_level0.variables["lat"][:],
        )  # 2D array
        
        self.params_dataset_level0.variables["lons"][:, :] = self.grid_array_lons_level0
        self.params_dataset_level0.variables["lats"][:, :] = self.grid_array_lats_level0
    
    def set_depths_vertical_aggregation(self):
        # depth, m
        self.logger.debug("setting depths and vertical aggregation... ...")
        
        # total_dpth
        total_depth = self.tf_VIC.total_depth(self.soillayerresampler.orig_total, *self.g_params["total_depths"]["optimal"])
        
        # resample based on g_params["soil_layers_breakpoints"], get self.grouping attribute (inplace = True)
        depths = self.tf_VIC.depth(total_depth, self.soillayerresampler, self.g_params["soil_layers_breakpoints"]["optimal"])  # do not uppack, cuz the third params is a list
        
        self.grid_array_depth_layers = []
        for i in range(len(self.nlayer_list)):
            grid_array_depth_layer_i = createEmptyArray_from_gridshp(
                self.stand_grids_lat_level0, self.stand_grids_lon_level0, dtype=float, missing_value=np.nan
                )
            
            grid_array_depth_layer_i = assignValue_for_grid_array(
                grid_array_depth_layer_i,
                np.full((self.grids_num_level0,), fill_value=depths[i]),
                self.rows_index_level0,
                self.cols_index_level0,
            )
            
            self.params_dataset_level0.variables["depth"][i, :, :] = grid_array_depth_layer_i
            
            self.grid_array_depth_layers.append(grid_array_depth_layer_i)
        
    def set_ele_std(self):
        # ele_std, m (same as StrmDem)
        self.logger.debug("setting ele_std... ...")
        
        self.grid_array_ele_std = createEmptyArray_from_gridshp(
            self.stand_grids_lat_level0, self.stand_grids_lon_level0, dtype=float, missing_value=np.nan
        )
        
        self.grid_array_ele_std = assignValue_for_grid_array(
            self.grid_array_ele_std,
            self.grid_shp_level0.loc[:, "SrtmDEM_std_Value"],
            self.rows_index_level0,
            self.cols_index_level0,
        )
    
    def set_b_infilt(self):
        # b_infilt, N/A
        self.logger.debug("setting b_infilt... ...")
        
        self.params_dataset_level0.variables["infilt"][:, :] = self.tf_VIC.b_infilt(
            self.grid_array_ele_std, *self.g_params["b_infilt"]["optimal"]
        )
    
    @staticmethod
    def cal_ssc_percentile_grid_array(
        grid_shp_level0,
        orig_depths,
        depth_layer_start,
        depth_layer_end,
        stand_grids_lat,
        stand_grids_lon,
        rows_index,
        cols_index,
    ):
        """
        Computes the weighted mean of sand, silt, and clay percentages over multiple soil depth layers.

        Parameters
        ----------
        grid_shp_level0 : GeoDataFrame
            Geospatial dataframe containing soil property values.
        depth_layer_start : int
            The starting index of the soil depth layers.
        depth_layer_end : int
            The ending index of the soil depth layers.
        stand_grids_lat : list
            Standardized latitude grid array.
        stand_grids_lon : list
            Standardized longitude grid array.
        rows_index : dict
            Mapping of latitude values to row indices.
        cols_index : dict
            Mapping of longitude values to column indices.

        Returns
        -------
        tuple
            - grid_array_sand : ndarray
                Weighted mean sand percentage array.
            - grid_array_silt : ndarray
                Weighted mean silt percentage array.
            - grid_array_clay : ndarray
                Weighted mean clay percentage array.
        """
        # vertical aggregation for sand, silt, clay percentile
        grid_array_sand = [
            createEmptyArray_and_assignValue_from_gridshp(
                stand_grids_lat,
                stand_grids_lon,
                grid_shp_level0.loc[:, f"soil_l{i+1}_sand_nearest_Value"],
                rows_index,
                cols_index,
                dtype=float,
                missing_value=np.nan,
            )
            for i in range(depth_layer_start, depth_layer_end + 1)
        ]
        grid_array_silt = [
            createEmptyArray_and_assignValue_from_gridshp(
                stand_grids_lat,
                stand_grids_lon,
                grid_shp_level0.loc[:, f"soil_l{i+1}_silt_nearest_Value"],
                rows_index,
                cols_index,
                dtype=float,
                missing_value=np.nan,
            )
            for i in range(depth_layer_start, depth_layer_end + 1)
        ]
        grid_array_clay = [
            createEmptyArray_and_assignValue_from_gridshp(
                stand_grids_lat,
                stand_grids_lon,
                grid_shp_level0.loc[:, f"soil_l{i+1}_clay_nearest_Value"],
                rows_index,
                cols_index,
                dtype=float,
                missing_value=np.nan,
            )
            for i in range(depth_layer_start, depth_layer_end + 1)
        ]

        # weight mean
        weights = [
            orig_depths[i] for i in range(depth_layer_start, depth_layer_end + 1)
        ]
        weights /= sum(weights)

        grid_array_sand = np.average(grid_array_sand, axis=0, weights=weights)
        grid_array_silt = np.average(grid_array_silt, axis=0, weights=weights)
        grid_array_clay = np.average(grid_array_clay, axis=0, weights=weights)

        # keep sum = 100
        grid_array_sum = grid_array_sand + grid_array_silt + grid_array_clay
        adjustment = 100 - grid_array_sum

        grid_array_sand += (grid_array_sand / grid_array_sum) * adjustment
        grid_array_silt += (grid_array_silt / grid_array_sum) * adjustment
        grid_array_clay += (grid_array_clay / grid_array_sum) * adjustment

        return grid_array_sand, grid_array_silt, grid_array_clay

    @staticmethod
    def cal_bd_grid_array(
        grid_shp_level0,
        orig_depths,
        depth_layer_start,
        depth_layer_end,
        stand_grids_lat,
        stand_grids_lon,
        rows_index,
        cols_index,
    ):
        """
        Computes the weighted mean bulk density over multiple soil depth layers.

        Parameters
        ----------
        grid_shp_level0 : GeoDataFrame
            Geospatial dataframe containing soil property values.
        depth_layer_start : int
            The starting index of the soil depth layers.
        depth_layer_end : int
            The ending index of the soil depth layers.
        stand_grids_lat : list
            Standardized latitude grid array.
        stand_grids_lon : list
            Standardized longitude grid array.
        rows_index : dict
            Mapping of latitude values to row indices.
        cols_index : dict
            Mapping of longitude values to column indices.

        Returns
        -------
        ndarray
            Weighted mean bulk density array (converted to kg/m³).
        """
        # vertical aggregation for bulk_density
        grid_array_bd = [
            createEmptyArray_and_assignValue_from_gridshp(
                stand_grids_lat,
                stand_grids_lon,
                grid_shp_level0.loc[:, f"soil_l{i+1}_bulk_density_nearest_Value"],
                rows_index,
                cols_index,
                dtype=float,
                missing_value=np.nan,
            )
            for i in range(depth_layer_start, depth_layer_end + 1)
        ]

        # weight mean
        weights = [
            orig_depths[i] for i in range(depth_layer_start, depth_layer_end + 1)
        ]
        weights /= sum(weights)

        grid_array_bd = np.average(grid_array_bd, axis=0, weights=weights)

        grid_array_bd *= 10  # (cg/cm3 -> kg/m3)

        return grid_array_bd

    def set_soil_texture(self):
        # sand, clay, silt, %
        self.logger.debug("setting soil texture... ...")
        
        self.grid_array_sand_layers = []
        self.grid_array_silt_layers = []
        self.grid_array_clay_layers = []
        
        for i in range(len(self.nlayer_list)):
            grouping_scheme_layer_i = self.soillayerresampler.grouping["grouping_scheme"][i]
            depth_layers_start = grouping_scheme_layer_i[0]
            depth_layers_end = grouping_scheme_layer_i[-1]
            
            grid_array_sand_layer_i, grid_array_silt_layer_i, grid_array_clay_layer_i = (
                self.cal_ssc_percentile_grid_array(
                    self.grid_shp_level0,
                    self.soillayerresampler.orig_depths,
                    depth_layers_start,
                    depth_layers_end,
                    self.stand_grids_lat_level0,
                    self.stand_grids_lon_level0,
                    self.rows_index_level0,
                    self.cols_index_level0,
                )
            )
            
            self.grid_array_sand_layers.append(grid_array_sand_layer_i)
            self.grid_array_silt_layers.append(grid_array_silt_layer_i)
            self.grid_array_clay_layers.append(grid_array_clay_layer_i)
    
    def set_ksat(self):
        # ksat, mm/s -> mm/day (VIC requirement)
        self.logger.debug("setting ksat... ...")
        
        unit_factor_ksat = 60 * 60 * 24
        
        self.grid_array_ksat_layers = []
        for i in range(len(self.nlayer_list)):
            grid_array_ksat_layer_i = self.tf_VIC.ksat(
                    self.grid_array_sand_layers[i],
                    self.grid_array_clay_layers[i],
                    *self.g_params["ksat"]["optimal"],
                )
            
            self.params_dataset_level0.variables["Ksat"][i, :, :] = (
                    grid_array_ksat_layer_i * unit_factor_ksat
            )

            self.grid_array_ksat_layers.append(grid_array_ksat_layer_i)
    
    def set_mean_slope(self):
        # mean slope, % (m/m)
        self.logger.debug("setting mean slope... ...")
        
        self.grid_array_mean_slope = createEmptyArray_from_gridshp(
            self.stand_grids_lat_level0, self.stand_grids_lon_level0,
            dtype=float, missing_value=np.nan
        )
        
        self.grid_array_mean_slope = assignValue_for_grid_array(
            self.grid_array_mean_slope,
            self.grid_shp_level0.loc[:, "SrtmDEM_mean_slope_Value%"],
            self.rows_index_level0,
            self.cols_index_level0,
        )
        
        self.params_dataset_level0.variables["slope"][:, :] = self.grid_array_mean_slope
    
    def set_phi_s(self):
        # phi_s, m3/m3 or mm/mm
        self.logger.debug("setting phi_s... ...")
        
        self.grid_array_phi_s_layers = []
        
        for i in range(len(self.nlayer_list)):
            grid_array_phi_s_layer_i = self.tf_VIC.phi_s(
                self.grid_array_sand_layers[i],
                self.grid_array_clay_layers[i],
                *self.g_params["phi_s"]["optimal"]
            )

            self.params_dataset_level0.variables["phi_s"][i, :, :] = grid_array_phi_s_layer_i
            self.grid_array_phi_s_layers.append(grid_array_phi_s_layer_i)
    
    def set_psis(self):
        # psis, kPa/cm-H2O
        self.logger.debug("setting psis... ...")
        
        self.grid_array_psis_layers = []
        
        for i in range(len(self.nlayer_list)):
            grid_array_psis_layer_i = self.tf_VIC.psis(
                self.grid_array_sand_layers[i],
                self.grid_array_silt_layers[i],
                *self.g_params["psis"]["optimal"]
            )
            
            self.params_dataset_level0.variables["psis"][i, :, :] = grid_array_psis_layer_i
            self.grid_array_psis_layers.append(grid_array_psis_layer_i)
            
    def set_b_retcurve(self):
        # b_retcurve, N/A
        self.logger.debug("setting b_retcurve... ...")
        
        self.grid_array_b_retcurve_layers = []
        
        for i in range(len(self.nlayer_list)):
            grid_array_b_retcurve_layer_i = self.tf_VIC.b_retcurve(
                self.grid_array_sand_layers[i],
                self.grid_array_clay_layers[i],
                *self.g_params["b_retcurve"]["optimal"]
            )
            
            self.params_dataset_level0.variables["b_retcurve"][i, :, :] = grid_array_b_retcurve_layer_i
            self.grid_array_b_retcurve_layers.append(grid_array_b_retcurve_layer_i)
    
    def set_expt(self):
        # expt, N/A
        self.logger.debug("setting expt... ...")
        
        self.grid_array_expt_layers = []
        
        for i in range(len(self.nlayer_list)):
            grid_array_expt_layer_i = self.tf_VIC.expt(
                self.grid_array_b_retcurve_layers[i],
                *self.g_params["expt"]["optimal"]
            )
            
            self.params_dataset_level0.variables["expt"][i, :, :] = grid_array_expt_layer_i
            self.grid_array_expt_layers.append(grid_array_expt_layer_i)
        
    def set_fc(self):
        # fc, % or m3/m3
        self.logger.debug("setting fc... ...")
        
        self.grid_array_fc_layers = []
        
        for i in range(len(self.nlayer_list)):
            grid_array_fc_layer_i = self.tf_VIC.fc(
                self.grid_array_phi_s_layers[i],
                self.grid_array_b_retcurve_layers[i],
                self.grid_array_psis_layers[i],
                self.grid_array_sand_layers[i],
                *self.g_params["fc"]["optimal"]
            )
            
            self.params_dataset_level0.variables["fc"][i, :, :] = grid_array_fc_layer_i
            self.grid_array_fc_layers.append(grid_array_fc_layer_i)
    
    def set_d4(self):
        # d4, N/A, same as c, typically is 2
        self.logger.debug("setting d4... ...")
        
        self.grid_array_d4 = createEmptyArray_from_gridshp(
            self.stand_grids_lat_level0, self.stand_grids_lon_level0,
            dtype=float, missing_value=np.nan
        )
        
        self.grid_array_d4 = assignValue_for_grid_array(
            self.grid_array_d4,
            np.full((self.grids_num_level0,), fill_value=self.tf_VIC.d4(*self.g_params["d4"]["optimal"])),
            self.rows_index_level0,
            self.cols_index_level0,
        )
        
        self.params_dataset_level0.variables["d4"][:, :] = self.grid_array_d4
            
    def set_cexpt(self):
        # cexpt
        self.logger.debug("setting cexpt... ...")
        
        self.grid_array_cexpt = self.grid_array_d4
        self.params_dataset_level0.variables["c"][:, :] = self.grid_array_cexpt
    
    def set_arno_baseflow_layer_num(self):
        self.logger.debug("setting arno_baseflow_layer_num... ...")
        
        self.arno_baseflow_layer_num = 2
    
    def set_d1(self):
        # d1 ([day^-1])
        self.logger.debug("setting d1... ...")
        
        self.grid_array_d1 = self.tf_VIC.d1(
            self.grid_array_ksat_layers[self.arno_baseflow_layer_num],
            self.grid_array_mean_slope,
            self.grid_array_phi_s_layers[self.arno_baseflow_layer_num],
            self.grid_array_depth_layers[self.arno_baseflow_layer_num],
            *self.g_params["d1"]["optimal"]
        )
        self.params_dataset_level0.variables["d1"][:, :] = self.grid_array_d1
        
    def set_d2(self):
        # d2 ([day^-d4])
        self.logger.debug("setting d2... ...")
        
        self.grid_array_d2 = self.tf_VIC.d2(
            self.grid_array_ksat_layers[self.arno_baseflow_layer_num], self.grid_array_mean_slope, self.grid_array_d4, *self.g_params["d2"]["optimal"]
        )
        self.params_dataset_level0.variables["d2"][:, :] = self.grid_array_d2
    
    def set_d3(self):
        # d3 ([mm])
        self.logger.debug("setting d3... ...")
        
        self.grid_array_d3 = self.tf_VIC.d3(
            self.grid_array_fc_layers[self.arno_baseflow_layer_num],
            self.grid_array_phi_s_layers[self.arno_baseflow_layer_num],
            self.grid_array_depth_layers[self.arno_baseflow_layer_num],
            *self.g_params["d3"]["optimal"]
        )
        
        self.params_dataset_level0.variables["d3"][:, :] = self.grid_array_d3
        
    def set_Dsmax(self):
        # Dsmax, mm or mm/day
        self.logger.debug("setting Dsmax... ...")
        
        self.grid_array_Dsmax = self.tf_VIC.Dsmax(
            self.grid_array_d1,
            self.grid_array_d2,
            self.grid_array_d3,
            self.grid_array_cexpt,
            self.grid_array_phi_s_layers[self.arno_baseflow_layer_num],
            self.grid_array_depth_layers[self.arno_baseflow_layer_num],
        )
        
        self.params_dataset_level0.variables["Dsmax"][:, :] = self.grid_array_Dsmax
    
    def set_Ds(self):
        # Ds, [day^-d4] or fraction
        self.logger.debug("setting Ds... ...")
        
        grid_array_Ds = self.tf_VIC.Ds(self.grid_array_d1, self.grid_array_d3, self.grid_array_Dsmax)
        self.params_dataset_level0.variables["Ds"][:, :] = grid_array_Ds
        
    def set_Ws(self):
        # Ws, fraction
        self.logger.debug("setting Ws... ...")
        
        grid_array_Ws = self.tf_VIC.Ws(
            self.grid_array_d3, 
            self.grid_array_phi_s_layers[self.arno_baseflow_layer_num],
            self.grid_array_depth_layers[self.arno_baseflow_layer_num]
        )
        self.params_dataset_level0.variables["Ws"][:, :] = grid_array_Ws
        
    def set_init_moist(self):
        # init_moist, mm
        self.logger.debug("setting init_moist... ...")
        
        for i in range(len(self.nlayer_list)):
            grid_array_init_moist_layer_i = self.tf_VIC.init_moist(
                self.grid_array_phi_s_layers[i],
                self.grid_array_depth_layers[i],
            )
            
            self.params_dataset_level0.variables["init_moist"][i, :, :] = grid_array_init_moist_layer_i
        
    def set_elev(self):
        # elev, m, Arithmetic mean
        self.logger.debug("setting elev... ...")
        
        grid_array_elev = createEmptyArray_from_gridshp(
            self.stand_grids_lat_level0, self.stand_grids_lon_level0,
            dtype=float, missing_value=np.nan
        )
        grid_array_elev = assignValue_for_grid_array(
            grid_array_elev,
            self.grid_shp_level0.loc[:, "SrtmDEM_mean_Value"],
            self.rows_index_level0,
            self.cols_index_level0,
        )

        self.params_dataset_level0.variables["elev"][:, :] = grid_array_elev

    def set_dp(self):
        # dp, m, typically is 4m
        self.logger.debug("setting dp... ...")
        
        grid_array_dp = createEmptyArray_from_gridshp(
            self.stand_grids_lat_level0, self.stand_grids_lon_level0, dtype=float, missing_value=np.nan
        )
        grid_array_dp = assignValue_for_grid_array(
            grid_array_dp,
            np.full((self.grids_num_level0,), fill_value=self.tf_VIC.dp(*self.g_params["d4"]["optimal"])),
            self.rows_index_level0,
            self.cols_index_level0,
        )

        self.params_dataset_level0.variables["dp"][:, :] = grid_array_dp
        
    def set_bubble(self):
        # bubble, cm
        self.logger.debug("setting bubble... ...")
        
        for i in range(len(self.nlayer_list)):
            grid_array_bubble_layer_i = self.tf_VIC.bubble(
                self.grid_array_expt_layers[i], *self.g_params["bubble"]["optimal"]
            )
            
            self.params_dataset_level0.variables["bubble"][i, :, :] = grid_array_bubble_layer_i
    
    def set_quartz(self):
        # quartz, N/A, fraction
        self.logger.debug("setting quartz... ...")
        
        for i in range(len(self.nlayer_list)):
            grid_array_quartz_layer_i = self.tf_VIC.quartz(self.grid_array_sand_layers[i], *self.g_params["quartz"]["optimal"])
            self.params_dataset_level0.variables["quartz"][i, :, :] = grid_array_quartz_layer_i
    
    def set_bulk_density(self):
        # bulk_density, kg/m3 or mm
        self.logger.debug("setting bulk_density... ...")
        
        for i in range(len(self.nlayer_list)):
            grouping_scheme_layer_i = self.soillayerresampler.grouping["grouping_scheme"][i]
            depth_layers_start = grouping_scheme_layer_i[0]
            depth_layers_end = grouping_scheme_layer_i[-1]
            
            grid_array_bd_layer_i = self.cal_bd_grid_array(
                self.grid_shp_level0,
                self.soillayerresampler.orig_depths,
                depth_layers_start,
                depth_layers_end,
                self.stand_grids_lat_level0,
                self.stand_grids_lon_level0,
                self.rows_index_level0,
                self.cols_index_level0,
            )
            
            grid_array_bd_layer_i = self.tf_VIC.bulk_density(grid_array_bd_layer_i, *self.g_params["bulk_density"]["optimal"])
            self.params_dataset_level0.variables["bulk_density"][i, :, :] = grid_array_bd_layer_i
   
    def set_soil_density(self):
        # soil_density, kg/m3
        self.logger.debug("setting soil_density... ...")
        
        for i in range(len(self.nlayer_list)):
            soil_density_layer_i = self.tf_VIC.soil_density(self.g_params["soil_density"]["optimal"][i])
            
            grid_array_soil_density_layer_i = createEmptyArray_from_gridshp(
                self.stand_grids_lat_level0,
                self.stand_grids_lon_level0,
                dtype=float, missing_value=np.nan
            )
            
            grid_array_soil_density_layer_i = assignValue_for_grid_array(
                grid_array_soil_density_layer_i,
                np.full((self.grids_num_level0,), fill_value=soil_density_layer_i),
                self.rows_index_level0,
                self.cols_index_level0,
            )
            
            self.params_dataset_level0.variables["soil_density"][i, :, :] = grid_array_soil_density_layer_i
    
    def set_Wcr_FRACT(self):
        # Wcr_FRACT, fraction
        self.logger.debug("setting Wcr_FRACT... ...")
        
        for i in range(len(self.nlayer_list)):
            grid_array_Wcr_FRACT_layer_i = self.tf_VIC.Wcr_FRACT(
                self.grid_array_fc_layers[i], self.grid_array_phi_s_layers[i], *self.g_params["Wcr_FRACT"]["optimal"]
            )
            
            self.params_dataset_level0.variables["Wcr_FRACT"][i, :, :] = grid_array_Wcr_FRACT_layer_i
        
    def set_wp(self):
        # wp, computed field capacity [frac]
        self.logger.debug("setting wp... ...")
        
        self.grid_array_wp_layers = []
        
        for i in range(len(self.nlayer_list)):
            grid_array_wp_layer_i = self.tf_VIC.wp(
                self.grid_array_phi_s_layers[i],
                self.grid_array_b_retcurve_layers[i],
                self.grid_array_psis_layers[i],
                *self.g_params["wp"]["optimal"],
            )
            
            self.params_dataset_level0.variables["wp"][i, :, :] = grid_array_wp_layer_i
            self.grid_array_wp_layers.append(grid_array_wp_layer_i)
    
    def set_Wpwp_FRACT(self):
        # Wpwp_FRACT, fraction
        self.logger.debug("setting Wpwp_FRACT... ...")
        
        for i in range(len(self.nlayer_list)):
            grid_array_Wpwp_FRACT_layer_i = self.tf_VIC.Wpwp_FRACT(
                self.grid_array_wp_layers[i], self.grid_array_phi_s_layers[i], *self.g_params["Wpwp_FRACT"]["optimal"]
            )
            self.params_dataset_level0.variables["Wpwp_FRACT"][i, :, :] = grid_array_Wpwp_FRACT_layer_i
            
    def set_rough(self):
        # rough, m, Surface roughness of bare soil
        self.logger.debug("setting rough... ...")
        
        grid_array_rough = createEmptyArray_from_gridshp(
            self.stand_grids_lat_level0, self.stand_grids_lon_level0,
            dtype=float, missing_value=np.nan
        )
        
        grid_array_rough = assignValue_for_grid_array(
            grid_array_rough,
            np.full((self.grids_num_level0,), fill_value=self.tf_VIC.rough(*self.g_params["rough"]["optimal"])),
            self.rows_index_level0,
            self.cols_index_level0,
        )
        
        self.params_dataset_level0.variables["rough"][:, :] = grid_array_rough
    
    def set_snow_rough(self):
        # snow rough, m
        self.logger.debug("setting snow_rough... ...")
        
        grid_array_snow_rough = createEmptyArray_from_gridshp(
            self.stand_grids_lat_level0, self.stand_grids_lon_level0,
            dtype=float, missing_value=np.nan
        )
        
        grid_array_snow_rough = assignValue_for_grid_array(
            grid_array_snow_rough,
            np.full((self.grids_num_level0,), fill_value=self.tf_VIC.snow_rough(*self.g_params["snow_rough"]["optimal"])),
            self.rows_index_level0,
            self.cols_index_level0,
        )

        self.params_dataset_level0.variables["snow_rough"][:, :] = grid_array_snow_rough
        

class buildParam_level1_interface:
    
    def __init__(
        self,
        evb_dir,
        logger,
        dpc_VIC_level1,
        TF_VIC: TF_VIC,
        reverse_lat=True,
        domain_dataset=None,
        stand_grids_lat_level1=None,
        stand_grids_lon_level1=None,
        rows_index_level1=None,
        cols_index_level1=None,
    ):

        self.evb_dir = evb_dir
        self.logger = logger
        self.dpc_VIC_level1 = dpc_VIC_level1
        self.reverse_lat = reverse_lat
        self.tf_VIC = TF_VIC()
        self.domain_dataset = domain_dataset
        
        if self.dpc_VIC_level1.get_data_from_cache("merged_grid_shp")[0] is None:
            self.dpc_VIC_level1.merge_grid_data()
            
        self.grid_shp_level1 = deepcopy(self.dpc_VIC_level1.get_data_from_cache("merged_grid_shp")[0])
        self.grids_num_level1 = len(self.grid_shp_level1.index)
        
        self.stand_grids_lat_level1 = stand_grids_lat_level1
        self.stand_grids_lon_level1 = stand_grids_lon_level1
        self.rows_index_level1 = rows_index_level1
        self.cols_index_level1 = cols_index_level1
    
    def buildParam_level1_basic(self):
        execution_order = [
            "set_coord_map",
            "create_parameter_dataset",
            "set_dims",
        ]
        
        for fun_name in execution_order:
            fun = getattr(self, fun_name)
            fun()
            
        self.logger.debug(f"execution order: {execution_order}")
        
        return execution_order
    
    def buildParam_level1_by_tf(self):
        execution_order = [
            "set_run_cell",
            "set_grid_cell",
            "set_off_gmt",
            "set_avg_T",
            "set_annual_prec",
            "set_resid_moist",
            "set_fs_active",
            "set_Nveg",
            "set_Cv",
            "set_veg_params_json",
            "set_root_depth",
            "set_root_fract",
            "set_rarc",
            "set_rmin",
            "set_overstory_wind_h",
            "set_displacement",
            "set_veg_rough",
            "set_RGL",
            "set_rad_atten",
            "set_wind_atten",
            "set_trunk_ratio",
            "set_LAI",
            "set_albedo",
            "set_fcanopy",
        ]
        
        for fun_name in execution_order:
            fun = getattr(self, fun_name)
            fun()
        
        self.logger.debug(f"execution order: {execution_order}")
        
        return execution_order
    
    def set_coord_map(self):
        self.logger.debug("setting coord_map... ...")
        
        if self.stand_grids_lat_level1 is None:
            self.stand_grids_lat_level1, self.stand_grids_lon_level1 = createStand_grids_lat_lon_from_gridshp(
                self.grid_shp_level1, grid_res=None, reverse_lat=self.reverse_lat
            )

        if self.rows_index_level1 is None:
            self.rows_index_level1, self.cols_index_level1 = gridshp_index_to_grid_array_index(
                self.grid_shp_level1, self.stand_grids_lat_level1, self.stand_grids_lon_level1
            )
        
    def create_parameter_dataset(self):
        self.logger.debug("creating parameter dataset... ...")
        
        self.params_dataset_level1 = createParametersDataset(
            self.evb_dir.params_dataset_level1_path,
            self.stand_grids_lat_level1,
            self.stand_grids_lon_level1
        )
    
    def set_dims(self):
        # Dimension variables: lat, lon, nlayer, root_zone, veg_class, month
        self.logger.debug("setting dims... ...")
        
        self.params_dataset_level1.variables["lat"][:] = np.array(self.stand_grids_lat_level1)  # 1D array
        self.params_dataset_level1.variables["lon"][:] = np.array(self.stand_grids_lon_level1)  # 1D array
        
        self.nlayer_list = [1, 2, 3]
        self.params_dataset_level1.variables["nlayer"][:] = self.nlayer_list
        
        self.root_zone_list = [1, 2, 3]
        self.params_dataset_level1.variables["root_zone"][:] = self.root_zone_list
        
        self.veg_class_list = list(range(14))
        self.params_dataset_level1.variables["veg_class"][:] = self.veg_class_list
        
        self.month_list = list(range(1, 13))
        self.params_dataset_level1.variables["month"][:] = self.month_list
        
        # lons, lats, 2D array
        self.grid_array_lons_level1, self.grid_array_lats_level1 = np.meshgrid(
            self.params_dataset_level1.variables["lon"][:],
            self.params_dataset_level1.variables["lat"][:],
        )  # 2D array
        
        self.params_dataset_level1.variables["lons"][:, :] = self.grid_array_lons_level1
        self.params_dataset_level1.variables["lats"][:, :] = self.grid_array_lats_level1
        
    def set_run_cell(self):
        # run_cell, bool, same as mask in DomainFile
        self.logger.debug("setting run_cell... ...")
        
        if self.domain_dataset is None:
            from ...bulid_Domain import cal_mask_frac_area_length
            mask, *_ = cal_mask_frac_area_length(
                self.dpc_VIC_level1, reverse_lat=self.reverse_lat, plot=False
            )  # * all lat set as reverse
        else:
            mask = self.domain_dataset.variables["mask"][:, :]  # * note the reverse_lat should be same

        mask = mask.astype(int)
        self.mask = mask
        self.params_dataset_level1.variables["run_cell"][:, :] = self.mask
        
    def set_grid_cell(self):
        # grid_cell
        self.logger.debug("setting grid_cell... ...")
        
        self.grid_array_grid_cell = createEmptyArray_from_gridshp(
            self.stand_grids_lat_level1, self.stand_grids_lon_level1, dtype=int, missing_value=-9999
        )
        
        self.grid_array_grid_cell = assignValue_for_grid_array(
            self.grid_array_grid_cell,
            np.arange(1, len(self.grid_shp_level1.index) + 1),
            self.rows_index_level1,
            self.cols_index_level1,
        )
        
        self.params_dataset_level1.variables["grid_cell"][:, :] = self.grid_array_grid_cell
        
    def set_off_gmt(self):
        # off_gmt, hours
        self.logger.debug("setting off_gmt... ...")
        
        self.grid_array_off_gmt = self.tf_VIC.off_gmt(self.grid_array_lons_level1)
        self.params_dataset_level1.variables["off_gmt"][:, :] = self.grid_array_off_gmt
            
    def set_avg_T(self):
        # avg_T, C
        self.logger.debug("setting avg_T... ...")
        
        self.grid_array_avg_T = createEmptyArray_from_gridshp(
            self.stand_grids_lat_level1, self.stand_grids_lon_level1, dtype=float, missing_value=np.nan
        )
        
        self.grid_array_avg_T = assignValue_for_grid_array(
            self.grid_array_avg_T,
            self.grid_shp_level1.loc[:, "stl_all_layers_mean_Value"],
            self.rows_index_level1,
            self.cols_index_level1,
        )
        
        self.params_dataset_level1.variables["avg_T"][:, :] = self.grid_array_avg_T
        
    def set_annual_prec(self):
        # annual_prec, mm
        self.logger.debug("setting annual_prec... ...")
        
        self.grid_array_annual_P = createEmptyArray_from_gridshp(
            self.stand_grids_lat_level1, self.stand_grids_lon_level1, dtype=float, missing_value=np.nan
        )
        
        self.grid_array_annual_P = assignValue_for_grid_array(
            self.grid_array_annual_P,
            self.grid_shp_level1.loc[:, "annual_P_in_src_grid_Value"],
            self.rows_index_level1,
            self.cols_index_level1,
        )
        
        self.params_dataset_level1.variables["annual_prec"][:, :] = self.grid_array_annual_P
        
    def set_resid_moist(self):
        # resid_moist, fraction, set as 0
        self.logger.debug("setting resid_moist... ...")
        
        self.grid_array_resid_moist = createEmptyArray_from_gridshp(
            self.stand_grids_lat_level1, self.stand_grids_lon_level1, dtype=float, missing_value=np.nan
        )
        
        self.grid_array_resid_moist = assignValue_for_grid_array(
            self.grid_array_resid_moist,
            np.full((self.grids_num_level1,), fill_value=0),
            self.rows_index_level1,
            self.cols_index_level1,
        )
        
        for i in range(len(self.nlayer_list)):
            self.params_dataset_level1.variables["resid_moist"][i, :, :] = self.grid_array_resid_moist
    
    def set_fs_active(self):
        # fs_active, bool, whether the frozen soil algorithm is activated
        self.logger.debug("setting fs_active... ...")
        
        self.grid_array_fs_active = createEmptyArray_from_gridshp(
            self.stand_grids_lat_level1, self.stand_grids_lon_level1, dtype=int, missing_value=-9999
        )
        
        self.grid_array_fs_active = assignValue_for_grid_array(
            self.grid_array_fs_active,
            np.full((self.grids_num_level1,), fill_value=0),
            self.rows_index_level1,
            self.cols_index_level1,
        )
        
        self.params_dataset_level1.variables["fs_active"][:, :] = self.grid_array_fs_active
        
    def set_Nveg(self):
        # Nveg, int
        self.logger.debug("setting Nveg... ...")
        
        self.grid_array_Nveg = createEmptyArray_from_gridshp(
            self.stand_grids_lat_level1, self.stand_grids_lon_level1, dtype=int, missing_value=-9999
        )
        
        self.grid_array_Nveg = assignValue_for_grid_array(
            self.grid_array_Nveg,
            self.grid_shp_level1["umd_lc_original_Value"].apply(lambda row: len(list(set(row)))),
            self.rows_index_level1,
            self.cols_index_level1,
        )
        
        self.params_dataset_level1.variables["Nveg"][:, :] = self.grid_array_Nveg
        
    def set_Cv(self):
        # Cv, fraction
        self.logger.debug("setting Cv... ...")
        
        self.grid_array_veg_classes_Cv = []
        
        for i in self.veg_class_list:
            grid_shp_level1_ = deepcopy(self.grid_shp_level1)
            grid_shp_level1_[f"umd_lc_{i}_veg_index"] = grid_shp_level1_.loc[
                :, "umd_lc_original_Value"
            ].apply(lambda row: np.where(np.array(row) == i)[0])

            grid_array_i_veg_Cv = createEmptyArray_from_gridshp(
                self.stand_grids_lat_level1, self.stand_grids_lon_level1, dtype=float, missing_value=np.nan
            )
            
            grid_array_i_veg_Cv = assignValue_for_grid_array(
                grid_array_i_veg_Cv,
                grid_shp_level1_.apply(
                    lambda row: sum(
                        np.array(row["umd_lc_original_Cv"])[row[f"umd_lc_{i}_veg_index"]]
                    ),
                    axis=1,
                ),
                self.rows_index_level1,
                self.cols_index_level1,
            )
            
            self.params_dataset_level1.variables["Cv"][i, :, :] = grid_array_i_veg_Cv
            
            self.grid_array_veg_classes_Cv.append(grid_array_i_veg_Cv)
        
    def set_veg_params_json(self):
        # read veg params, veg_params_json is a lookup_table
        self.logger.debug("setting veg_params_json... ...")
        
        self.veg_params_json = read_veg_param_json()
        
    def set_root_depth(self):
        # root_depth, m
        self.logger.debug("setting root_depth... ...")
        
        for i in self.veg_class_list:
            for j in self.root_zone_list:
                grid_array_i_veg_j_zone_root_depth = createEmptyArray_from_gridshp(
                    self.stand_grids_lat_level1, self.stand_grids_lon_level1, dtype=float, missing_value=np.nan
                )
                grid_array_i_veg_j_zone_root_depth = assignValue_for_grid_array(
                    grid_array_i_veg_j_zone_root_depth,
                    np.full(
                        (self.grids_num_level1,), fill_value=float(self.veg_params_json[f"{i}"][f"rootd{j}"])
                    ),
                    self.rows_index_level1,
                    self.cols_index_level1,
                )
                
                self.params_dataset_level1.variables["root_depth"][i, j - 1, :, :] = grid_array_i_veg_j_zone_root_depth  # j-1: root_zone_list start from 1
    
    def set_root_fract(self):
        # root_fract, fraction
        self.logger.debug("setting root_fract... ...")
        
        for i in self.veg_class_list:
            for j in self.root_zone_list:
                grid_array_i_veg_j_zone_root_fract = createEmptyArray_from_gridshp(
                    self.stand_grids_lat_level1, self.stand_grids_lon_level1, dtype=float, missing_value=np.nan
                )
                grid_array_i_veg_j_zone_root_fract = assignValue_for_grid_array(
                    grid_array_i_veg_j_zone_root_fract,
                    np.full(
                        (self.grids_num_level1,),
                        fill_value=float(self.veg_params_json[f"{i}"][f"rootfr{j}"]),
                    ),
                    self.rows_index_level1,
                    self.cols_index_level1,
                )
                
                self.params_dataset_level1.variables["root_fract"][i, j - 1, :, :] = grid_array_i_veg_j_zone_root_fract
    
    def set_rarc(self):
        # rarc, s/m
        self.logger.debug("setting rarc... ...")
        
        for i in self.veg_class_list:
            grid_array_i_veg_rarc = createEmptyArray_from_gridshp(
                self.stand_grids_lat_level1, self.stand_grids_lon_level1, dtype=float, missing_value=np.nan
            )
            grid_array_i_veg_rarc = assignValue_for_grid_array(
                grid_array_i_veg_rarc,
                np.full((self.grids_num_level1,), fill_value=float(self.veg_params_json[f"{i}"][f"rarc"])),
                self.rows_index_level1,
                self.cols_index_level1,
            )
            
            self.params_dataset_level1.variables["rarc"][i, :, :] = grid_array_i_veg_rarc
    
    def set_rmin(self):
        # rmin, s/m
        self.logger.debug("setting rmin... ...")
        
        for i in self.veg_class_list:
            grid_array_i_veg_rmin = createEmptyArray_from_gridshp(
                self.stand_grids_lat_level1, self.stand_grids_lon_level1, dtype=float, missing_value=np.nan
            )
            
            grid_array_i_veg_rmin = assignValue_for_grid_array(
                grid_array_i_veg_rmin,
                np.full((self.grids_num_level1,), fill_value=float(self.veg_params_json[f"{i}"][f"rmin"])),
                self.rows_index_level1,
                self.cols_index_level1,
            )
            
            self.params_dataset_level1.variables["rmin"][i, :, :] = grid_array_i_veg_rmin

    def set_overstory_wind_h(self):
        # overstory, N/A, bool
        # wind_h, m, adjust wind height value if overstory is true (overstory == 1, wind_h=vegHeight+10, else wind_h=vegHeight+2)
        self.logger.debug("setting overstory and wind_h... ...")
        
        for i in self.veg_class_list:
            grid_array_i_veg_height = createEmptyArray_from_gridshp(
                self.stand_grids_lat_level1, self.stand_grids_lon_level1, dtype=float, missing_value=np.nan
            )
            grid_array_i_veg_height = assignValue_for_grid_array(
                grid_array_i_veg_height,
                np.full((self.grids_num_level1,), fill_value=float(self.veg_params_json[f"{i}"][f"h"])),
                self.rows_index_level1,
                self.cols_index_level1,
            )

            grid_array_i_veg_overstory = createEmptyArray_from_gridshp(
                self.stand_grids_lat_level1, self.stand_grids_lon_level1, dtype=int, missing_value=-9999
            )
            grid_array_i_veg_overstory = assignValue_for_grid_array(
                grid_array_i_veg_overstory,
                np.full(
                    (self.grids_num_level1,), fill_value=int(self.veg_params_json[f"{i}"][f"overstory"])
                ),
                self.rows_index_level1,
                self.cols_index_level1,
            )

            grid_array_wind_h_add_factor = np.full_like(
                grid_array_i_veg_overstory, fill_value=10
            )
            grid_array_wind_h_add_factor[grid_array_i_veg_overstory == 0] = 2

            grid_array_wind_h = grid_array_i_veg_height + grid_array_wind_h_add_factor

            self.params_dataset_level1.variables["overstory"][i, :, :] = grid_array_i_veg_overstory
            self.params_dataset_level1.variables["wind_h"][i, :, :] = grid_array_wind_h
            
    def set_displacement(self):
        # displacement, m, Vegetation displacement height (typically 0.67 * vegetation height), or read from veg_param_json_updated
        self.logger.debug("setting displacement... ...")
        
        for i in self.veg_class_list:
            for j in self.month_list:
                grid_array_i_veg_j_month_displacement = createEmptyArray_from_gridshp(
                    self.stand_grids_lat_level1, self.stand_grids_lon_level1, dtype=float, missing_value=np.nan
                )
                grid_array_i_veg_j_month_displacement = assignValue_for_grid_array(
                    grid_array_i_veg_j_month_displacement,
                    np.full(
                        (self.grids_num_level1,),
                        fill_value=float(
                            self.veg_params_json[f"{i}"][f"veg_displacement_month_{j}"]
                        ),
                    ),
                    self.rows_index_level1,
                    self.cols_index_level1,
                )
                self.params_dataset_level1.variables["displacement"][
                    i, j - 1, :, :
                ] = grid_array_i_veg_j_month_displacement  # j-1: month_list start from 1
                
    
    def set_veg_rough(self):
        # veg_rough, m, Vegetation roughness length (typically 0.123 * vegetation height), or read from veg_param_json_updated
        self.logger.debug("setting veg_rough... ...")
        
        for i in self.veg_class_list:
            for j in self.month_list:
                grid_array_i_veg_j_month_veg_rough = createEmptyArray_from_gridshp(
                    self.stand_grids_lat_level1, self.stand_grids_lon_level1, dtype=float, missing_value=np.nan
                )
                grid_array_i_veg_j_month_veg_rough = assignValue_for_grid_array(
                    grid_array_i_veg_j_month_veg_rough,
                    np.full(
                        (self.grids_num_level1,),
                        fill_value=float(self.veg_params_json[f"{i}"][f"veg_rough_month_{j}"]),
                    ),
                    self.rows_index_level1,
                    self.cols_index_level1,
                )
                self.params_dataset_level1.variables["veg_rough"][
                    i, j - 1, :, :
                ] = grid_array_i_veg_j_month_veg_rough

    def set_RGL(self):
        # RGL, W/m2
        self.logger.debug("setting RGL... ...")
        
        for i in self.veg_class_list:
            grid_array_i_veg_RGL = createEmptyArray_from_gridshp(
                self.stand_grids_lat_level1, self.stand_grids_lon_level1, dtype=float, missing_value=np.nan
            )
            grid_array_i_veg_RGL = assignValue_for_grid_array(
                grid_array_i_veg_RGL,
                np.full((self.grids_num_level1,), fill_value=float(self.veg_params_json[f"{i}"][f"rgl"])),
                self.rows_index_level1,
                self.cols_index_level1,
            )
            self.params_dataset_level1.variables["RGL"][i, :, :] = grid_array_i_veg_RGL
    
    def set_rad_atten(self):
        # rad_atten, fract
        self.logger.debug("setting rad_atten... ...")
        
        for i in self.veg_class_list:
            grid_array_i_veg_rad_atten = createEmptyArray_from_gridshp(
                self.stand_grids_lat_level1, self.stand_grids_lon_level1, dtype=float, missing_value=np.nan
            )
            grid_array_i_veg_rad_atten = assignValue_for_grid_array(
                grid_array_i_veg_rad_atten,
                np.full(
                    (self.grids_num_level1,), fill_value=float(self.veg_params_json[f"{i}"][f"rad_atn"])
                ),
                self.rows_index_level1,
                self.cols_index_level1,
            )
            self.params_dataset_level1.variables["rad_atten"][
                i, :, :
            ] = grid_array_i_veg_rad_atten
    
    def set_wind_atten(self):
        # wind_atten, fract
        self.logger.debug("setting wind_atten... ...")
        
        for i in self.veg_class_list:
            grid_array_i_veg_wind_atten = createEmptyArray_from_gridshp(
                self.stand_grids_lat_level1, self.stand_grids_lon_level1, dtype=float, missing_value=np.nan
            )
            grid_array_i_veg_wind_atten = assignValue_for_grid_array(
                grid_array_i_veg_wind_atten,
                np.full(
                    (self.grids_num_level1,), fill_value=float(self.veg_params_json[f"{i}"][f"wnd_atn"])
                ),
                self.rows_index_level1,
                self.cols_index_level1,
            )
            self.params_dataset_level1.variables["wind_atten"][
                i, :, :
            ] = grid_array_i_veg_wind_atten

    def set_trunk_ratio(self):
        # trunk_ratio, fract
        self.logger.debug("setting trunk_ratio... ...")
        
        for i in self.veg_class_list:
            grid_array_i_veg_trunk_ratio = createEmptyArray_from_gridshp(
                self.stand_grids_lat_level1, self.stand_grids_lon_level1, dtype=float, missing_value=np.nan
            )
            grid_array_i_veg_trunk_ratio = assignValue_for_grid_array(
                grid_array_i_veg_trunk_ratio,
                np.full((self.grids_num_level1,), fill_value=float(self.veg_params_json[f"{i}"][f"trnk_r"])),
                self.rows_index_level1,
                self.cols_index_level1,
            )
            self.params_dataset_level1.variables["trunk_ratio"][
                i, :, :
            ] = grid_array_i_veg_trunk_ratio
    
    def set_LAI(self):
        # LAI, fraction or m2/m2
        self.logger.debug("setting LAI... ...")
        
        for i in self.veg_class_list:
            for j in self.month_list:
                grid_shp_level1_ = deepcopy(self.grid_shp_level1)

                # LAI
                grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_LAI"] = grid_shp_level1_.apply(
                    lambda row: np.array(row[f"MODIS_LAI_original_Value_month{j}"])[
                        np.where(np.array(row.umd_lc_original_Value) == i)[0]
                    ],
                    axis=1,
                )
                grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_LAI"] = grid_shp_level1_.loc[
                    :, f"MODIS_{i}_veg_{j}_month_LAI"
                ].apply(lambda row: np.mean(row) if len(row) != 0 else 0)

                grid_array_i_veg_j_month_LAI = createEmptyArray_from_gridshp(
                    self.stand_grids_lat_level1, self.stand_grids_lon_level1, dtype=float, missing_value=np.nan
                )
                grid_array_i_veg_j_month_LAI = assignValue_for_grid_array(
                    grid_array_i_veg_j_month_LAI,
                    grid_shp_level1_.loc[:, f"MODIS_{i}_veg_{j}_month_LAI"],
                    self.rows_index_level1,
                    self.cols_index_level1,
                )
                self.params_dataset_level1.variables["LAI"][
                    i, j - 1, :, :
                ] = grid_array_i_veg_j_month_LAI  # j-1: month_list start from 1
    
    def set_albedo(self):
        # albedo, fraction
        self.logger.debug("setting albedo... ...")
        
        for i in self.veg_class_list:
            for j in self.month_list:
                grid_shp_level1_ = deepcopy(self.grid_shp_level1)
                
                # BSA, albedo
                grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_BSA"] = grid_shp_level1_.apply(
                    lambda row: np.array(row[f"MODIS_BSA_original_Value_month{j}"])[
                        np.where(np.array(row.umd_lc_original_Value) == i)[0]
                    ],
                    axis=1,
                )
                grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_BSA"] = grid_shp_level1_.loc[
                    :, f"MODIS_{i}_veg_{j}_month_BSA"
                ].apply(lambda row: np.mean(row) if len(row) != 0 else 0)

                grid_array_i_veg_j_month_BSA = createEmptyArray_from_gridshp(
                    self.stand_grids_lat_level1, self.stand_grids_lon_level1, dtype=float, missing_value=np.nan
                )
                grid_array_i_veg_j_month_BSA = assignValue_for_grid_array(
                    grid_array_i_veg_j_month_BSA,
                    grid_shp_level1_.loc[:, f"MODIS_{i}_veg_{j}_month_BSA"],
                    self.rows_index_level1,
                    self.cols_index_level1,
                )
                self.params_dataset_level1.variables["albedo"][
                    i, j - 1, :, :
                ] = grid_array_i_veg_j_month_BSA  # j-1: month_list start from 1

    
    def set_fcanopy(self):
        # fcanopy, fraction
        self.logger.debug("setting fcanopy... ...")
        
        for i in self.veg_class_list:
            for j in self.month_list:
                grid_shp_level1_ = deepcopy(self.grid_shp_level1)

                # fcanopy, ((NDVI-NDVI_min)/(NDVI_max-NDVI_min))**2
                grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_NDVI"] = grid_shp_level1_.apply(
                    lambda row: np.array(row[f"MODIS_NDVI_original_Value_month{j}"])[
                        np.where(np.array(row.umd_lc_original_Value) == i)[0]
                    ],
                    axis=1,
                )
                grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_NDVI"] = grid_shp_level1_.loc[
                    :, f"MODIS_{i}_veg_{j}_month_NDVI"
                ].apply(lambda row: np.mean(row) if len(row) != 0 else 0)
                grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_NDVI"] *= 0.0001
                NDVI = grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_NDVI"]

                grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_NDVI_max"] = (
                    grid_shp_level1_.apply(
                        lambda row: np.array(
                            row[f"MODIS_NDVI_max_original_Value_month{j}"]
                        )[np.where(np.array(row.umd_lc_original_Value) == i)[0]],
                        axis=1,
                    )
                )
                grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_NDVI_max"] = (
                    grid_shp_level1_.loc[:, f"MODIS_{i}_veg_{j}_month_NDVI_max"].apply(
                        lambda row: np.mean(row) if len(row) != 0 else 0
                    )
                )
                grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_NDVI_max"] *= 0.0001
                
                NDVI_max = grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_NDVI_max"]

                grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_NDVI_min"] = (
                    grid_shp_level1_.apply(
                        lambda row: np.array(
                            row[f"MODIS_NDVI_min_original_Value_month{j}"]
                        )[np.where(np.array(row.umd_lc_original_Value) == i)[0]],
                        axis=1,
                    )
                )
                grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_NDVI_min"] = (
                    grid_shp_level1_.loc[:, f"MODIS_{i}_veg_{j}_month_NDVI_min"].apply(
                        lambda row: np.mean(row) if len(row) != 0 else 0
                    )
                )
                grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_NDVI_min"] *= 0.0001
                NDVI_min = grid_shp_level1_[f"MODIS_{i}_veg_{j}_month_NDVI_min"]

                fcanopy = ((NDVI - NDVI_min) / (NDVI_max - NDVI_min)) ** 2
                fcanopy[np.isnan(fcanopy)] = 0

                grid_array_i_veg_j_month_fcanopy = createEmptyArray_from_gridshp(
                    self.stand_grids_lat_level1, self.stand_grids_lon_level1, dtype=float, missing_value=np.nan
                )
                grid_array_i_veg_j_month_fcanopy = assignValue_for_grid_array(
                    grid_array_i_veg_j_month_fcanopy, fcanopy, self.rows_index_level1, self.cols_index_level1
                )
                self.params_dataset_level1.variables["fcanopy"][
                    i, j - 1, :, :
                ] = grid_array_i_veg_j_month_fcanopy  # j-1: month_list start from 1
    