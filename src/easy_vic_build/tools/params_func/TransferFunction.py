# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: TransferFunction

This module defines various transfer functions used in hydrological modeling. These functions are 
important for transforming and scaling data related to soil properties, water retention, and surface 
roughness. The functions within this module implement common transfer equations such as the calculation 
of soil density, wilting point, and other key hydrological parameters. Each transfer function is 
encapsulated in a static method within the `TF_VIC` class.

Class:
------
    - TF_VIC: A class that contains static methods for various transfer functions, such as soil density 
      calculations, wilting point computations, and other hydrological parameter transformations.

Class Methods:
--------------
    - soil_density: Computes the soil mineral density, scaled by a factor `g`.
    - Wcr_FRACT: Computes the fraction of water retention at field capacity (Wcr) based on soil properties.
    - wp: Computes the wilting point based on Campbell's (1974) equation.
    - Wpwp_FRACT: Computes the fraction of the wilting point moisture content.
    - rough: Computes the soil surface roughness.
    - snow_rough: Computes the snow surface roughness.
    - off_gmt: Computes the offset from GMT based on longitude.
    - fs_active: Determines the activation status based on a flag.

Dependencies:
-------------
    - numpy: Used for array manipulation and mathematical operations.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com
"""


import numpy as np
from .params_set import *

class TF_VIC:
    """ Class containing methods for soil hydraulic parameter calculations, Transfer Functions. """
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def b_infilt(ele_std, g1, g2):
        """
        Calculate base infiltration rate.

        Parameters:
        ele_std (ndarray): Standard elevation values.
        g1 (float): Coefficient for the elevation standard.
        g2 (float): Coefficient for the elevation standard.

        Returns:
        ndarray: Base infiltration rate values.
        """
        # Dumenil, L. and Todini, E.: A rainfall-runoff scheme for use in the Hamburg climate model, Advances in theoretical hydrology, 129-157, 1992.
        # Hurk, B. and Viterbo, P.: The Torne-Kalix PILPS 2(e) experiment as a test bed for modifications to the ECMWF land surface scheme, Global Planet Change, 38, 165-173, 10.1016/S0921-8181(03)00027-4, 2003.
        # b_infilt, N/A, 0.01~0.5
        # g1, g2: 0.0 (-2.0, 1.0), 1.0 (0.8, 1.2)  # TODO recheck (ele_std - g1) / (ele_std + g2*10)
        # Arithmetic mean
        b_infilt_min = 0.01
        b_infilt_max = 1.0  # 0.50  # check
        ret = (np.log(ele_std) - g1) / (np.log(ele_std) + g2*10)
        
        ret[ret > b_infilt_max] = b_infilt_max
        ret[ret < b_infilt_min] = b_infilt_min
        return ret
    
    @staticmethod
    def total_depth(total_depth_original, g):
        """
        Compute total depth of a soil layer.

        Parameters:
        total_depth_original (ndarray): Original total depth values.
        g (float): Coefficient for adjusting depth.

        Returns:
        ndarray: Adjusted total depth values.
        """
        # total_depth, m
        # g: 1.0 (0.1, 4.0)
        # Arithmetic mean
        ret = total_depth_original * g
        return ret
    
    @staticmethod
    def depth(total_depth, g1, g2):
        """
        Calculate the distribution of soil layer depths.

        Parameters:
        total_depth (ndarray): Original total depth.
        g1 (float): Coefficient for the first layer.
        g2 (float): Coefficient for the second layer.

        Returns:
        list: Depth values for each layer.
        """
        # total_depth, m
        # depth, m
        # g1, g2: num1 (1, 3), num2 (3, 8), int  
        # set num1 as the num of end CONUS layer num of the first layer
        # set num2 as the num of end CONUS layer num of the second layer
        # d1 0~0.15, d2 0.2~0.5, d3 0.7~1.5, d2 > d1, default d1 (0.1), d2 (0.5), d3 (3.0)
        # Arithmetic mean
        
        # transfer g1, g2 into percentile
        percentile_layer1, percentile_layer2 = CONUS_depth_num_to_percentile(g1, g2)
        ret = [total_depth * percentile_layer1, total_depth * percentile_layer2, total_depth * (1.0 - percentile_layer1 - percentile_layer2)]
        return ret
    
    @staticmethod
    def ksat(sand, clay, g1, g2, g3):
        """
        Calculate the saturated hydraulic conductivity (Ks) using the formula from Cosby et al. (1984).
        
        Parameters
        ----------
        sand : float
            The percentage of sand in the soil.
        clay : float
            The percentage of clay in the soil.
        g1 : float
            The constant parameter for the model.
        g2 : float
            The constant parameter for the model.
        g3 : float
            The constant parameter for the model.
        
        Returns
        -------
        float
            The saturated hydraulic conductivity in mm/s.
        """
        # Cosby et al. WRR 1984, log Ks = 0.0126x1 (- 0.0064x3) -0.6
        # sand/clay: %
        # inches/hour -> 25.4 -> mm/hour -> /3600 -> mm/s
        # g1, g2, g3: -0.6 (-0.66, -0.54), 0.0126 (0.0113, 0.0139), -0.0064 (-0.0070, -0.0058)
        # Harmonic mean
        unit_factor1 = 25.4
        unit_factor2 = 1/3600
        ret = (10 ** (g1 + g2 * sand + g3 * clay)) * unit_factor1 * unit_factor2
        return ret

    # def ksat(sand, silt, g1, g2, g3):
    #     # campbell & shiozawa 1994
    #     # factor_unit
    #     clay = 100 - sand - silt
    #     ret = g1 * np.exp(g2 * sand + g3 * clay)
    #     return ret
    
    @staticmethod
    def phi_s(sand, clay, g1, g2, g3):
        """
        Calculate the saturated water content (phi_s) or porosity using the formula from Cosby et al. (1984).
        
        Parameters
        ----------
        sand : float
            The percentage of sand in the soil.
        clay : float
            The percentage of clay in the soil.
        g1 : float
            The constant parameter for the model.
        g2 : float
            The constant parameter for the model.
        g3 : float
            The constant parameter for the model.
        
        Returns
        -------
        float
            The saturated water content in m3/m3 (or mm/mm).
        """
        # Cosby et al. WRR 1984, Qs (φs) = -0.142x1 (- 0.037x3) + 50.5
        # Qs, saturated water content, namely, the porosity, namely the phi_s, m3/m3 or mm/mm
        # g1, g2, g3: 50.05 (45.5, 55.5), -0.142 (-0.3, -0.01), -0.037 (-0.1, -0.01)
        # Arithmetic mean
        ret = (g1 + g2 * sand + g3 * clay) / 100
        
        return ret
    
    # def phi_s(sand, silt, bd_in, g1, g2, g3):
    #     # Zacharias & Wessolek 2007
    #     clay = 100 - sand - silt
    #     if sand < 66.5:
    #         ret = g1 + g2 * clay + g3 * bd_in / 1000
    #     else:
    #         ret = g1 + g2 * clay + g3 * bd_in / 1000
    #     return ret
    
    # def phi_s(phi_s):
    #     # read from file
    #     phi_s_min = 0.0
    #     phi_s_max = 1.0
        
    #     ret = phi_s
    #     ret = ret / 100.0
        
    #     ret = ret if ret < phi_s_max else phi_s_max
    #     ret = ret if ret > phi_s_min else phi_s_min
    #     return ret
    
    @staticmethod
    def psis(sand, silt, g1, g2, g3):
        """
        Calculate the saturation matric potential (psis) using the formula from Cosby et al. (1984).
        
        Parameters
        ----------
        sand : float
            The percentage of sand in the soil.
        silt : float
            The percentage of silt in the soil.
        g1 : float
            The constant parameter for the model.
        g2 : float
            The constant parameter for the model.
        g3 : float
            The constant parameter for the model.
        
        Returns
        -------
        float
            The saturation matric potential in kPa.
        """
        # saturation matric potential, ψs, Cosby et al. WRR 1984, logψs = -0.0095x1 (+ 0.0063x2) + 1.54
        # kPa/cm-H2O
        # g1, g2, g3: 1.54 (1.0, 2.0), -0.0095 (-0.01, -0.009), 0.0063 (0.006, 0.0066)
        # Arithmetic mean
        ret = g1 + g2 * sand + g3 * silt
        unit_factor1 = 0.0980665  # 0.0980665 kPa/cm-H2O. Cosby give psi_sat in cm of water (cm-H2O), 1cm H₂O=0.0980665 kPa
        ret = -1 * (10 ** (ret)) * unit_factor1
        return ret

    @staticmethod
    def b_retcurve(sand, clay, g1, g2, g3):
        """
        Calculate the slope of the retention curve (b) using the formula from Cosby et al. (1984).
        
        Parameters
        ----------
        sand : float
            The percentage of sand in the soil.
        clay : float
            The percentage of clay in the soil.
        g1 : float
            The constant parameter for the model.
        g2 : float
            The constant parameter for the model.
        g3 : float
            The constant parameter for the model.
        
        Returns
        -------
        float
            The slope of the retention curve.
        """
        # b (slope of cambell retention curve in log space), N/A, ψ = ψc(θ/θs)-b
        # Cosby et al. WRR 1984, b = 0.157x3 (- 0.003x1) + 3.10, b~=0~20
        # g1, g2, g3: 3.1 (2.5, 3.6), 0.157 (0.1, 0.2), -0.003 (-0.005, -0.001)
        # Arithmetic mean
        ret = g1 + g2 * sand + g3 * clay
        return ret
    
    @staticmethod
    def expt(b_retcurve, g1, g2):
        """
        Calculate the exponent in Campbell's equation for hydraulic conductivity.
        
        Parameters
        ----------
        b_retcurve : float
            The slope of the retention curve.
        g1 : float
            The constant parameter for the model.
        g2 : float
            The constant parameter for the model.
        
        Returns
        -------
        float
            The exponent in Campbell's equation for hydraulic conductivity (k = ks (θ/θs)^(2b+3)).
        """
        # the exponent in Campbell's equation for hydraulic conductivity, k = ks (θ/θs)2b+3
        # expt = 2b+3 should be > 3
        # g1, g2: 3.0 (2.8, 3.2), 2.0 (1.5, 2.5)
        # Arithmetic mean
        ret = g1 + g2 * b_retcurve
        return ret
    
    @staticmethod
    def fc(phi_s, b_retcurve, psis, sand, g):
        """
        Calculate the field capacity (fc) using Campbell's equation for hydraulic conductivity.
        
        Parameters
        ----------
        phi_s : float
            The saturated water content (porosity).
        b_retcurve : float
            The slope of the retention curve (b).
        psis : float
            The saturation matric potential (ψs).
        sand : float
            The percentage of sand in the soil.
        g : float
            The constant parameter for the model.
        
        Returns
        -------
        float
            The field capacity (θ) in m³/m³ or %.
        """
        # campbell 1974, ψ = ψc(θ/θs)^-b, saturation condition
        # ψ = ψc(θ/θs)^-b -> ψ/ψc = (θ/θs)^-b -> θ/θs = (ψ/ψc)^(-1/b) -> θ = θs * ψ/ψc^(-1/b)
        # m3/m3 or %
        # g: 1.0 (0.8, 1.2)
        # Arithmetic mean
        psi_fc = np.full_like(phi_s, fill_value=-10)  # ψfc kPa/cm-H2O, -30~-10kPa
        psi_fc[sand < 70] = -20
        psi_fc[sand < 50] = -33
        
        # psi_fc = np.full_like(phi_s, fill_value=-10)  # ψfc kPa/cm-H2O, -30~-10kPa or -33kPa？
        # psi_fc[sand <= 69] = -20
        
        ret = g * phi_s * (psi_fc / psis) ** (-1 / b_retcurve)
        return ret
    
    @staticmethod
    def D1(Ks, slope_mean, g):
        """
        Calculate the D1 parameter, with units of day^-1.
        
        Parameters
        ----------
        Ks : float
            Hydraulic conductivity at layer 3 (m/s).
        slope_mean : float
            Mean slope value.
        g : float
            Constant parameter for the model (2.0 (1.75, 3.5)).
        
        Returns
        -------
        float
            The D1 value in day^-1, bounded between a minimum and maximum range.
        """
        # Ks: layer3
        # D1, [day^-1]
        # g: 2.0 (1.75, 3.5)
        # Harmonic mean
        Sf = 1.0
        D1_min = 0.0001
        D1_max = 1.0
        unit_factor1 = 60*60*24
        unit_factor2 = 0.01
        ret = (Ks * unit_factor1) * (slope_mean * unit_factor2) / (10 ** g) / Sf
        
        ret[ret > D1_max] = D1_max
        ret[ret < D1_min] = D1_min
        return ret
        
    @staticmethod
    def D2(Ks, slope_mean, D4, g):
        """
        Calculate the D2 parameter with units of day^-D4.
        
        Parameters
        ----------
        Ks : float
            Hydraulic conductivity at layer 3 (m/s).
        slope_mean : float
            Mean slope value.
        D4 : float
            Exponent for D4.
        g : float
            Constant parameter for the model (2.0 (1.75, 3.5)).
        
        Returns
        -------
        float
            The D2 value, bounded between a minimum and maximum range.
        """
        # Ks: layer3
        # D2, [day^-D4]
        # g: 2.0 (1.75, 3.5)
        # Harmonic mean
        Sf = 1.0
        D2_min = 0.0001
        D2_max = 1.0
        unit_factor1 = 60*60*24
        unit_factor2 = 0.01
        ret = (Ks * unit_factor1) * (slope_mean * unit_factor2) / (10 ** g) / (Sf ** D4)
        
        ret[ret > D2_max] = D2_max
        ret[ret < D2_min] = D2_min
        
        return ret
    
    @staticmethod
    def D3(fc, depth, g):
        """
        Calculate the D3 parameter in mm.
        
        Parameters
        ----------
        fc : float
            Field capacity (m³/m³).
        depth : float
            Depth of the soil layer (m).
        g : float
            Constant parameter for the model (1.0 (0.001, 2.0)).
        
        Returns
        -------
        float
            The D3 value in mm, bounded between a minimum and maximum range.
        """
        # depth: layer3, m
        # D3, [mm]
        # g: 1.0 (0.001, 2.0)
        # Arithmetic mean
        D3_min = 0.0001
        D3_max = 1000.0
        unit_factor1 = 1000
        ret = fc * (depth * unit_factor1) * g
        
        ret[ret > D3_max] = D3_max
        ret[ret < D3_min] = D3_min
        return ret
    
    @staticmethod
    def D4(g=2): # set to 2
        """
        Return the value for D4, typically set to 2.
        
        Parameters
        ----------
        g : float, optional
            Constant parameter for the model (default is 2.0 (1.2, 2.5)).
        
        Returns
        -------
        float
            The value of D4.
        """
        # g: 2.0 (1.2, 2.5)
        # Arithmetic mean
        ret = g
        return ret
    
    @staticmethod
    def cexpt(D4): # set to D4
        """
        Return the value for cexpt, which is equal to D4.
        
        Parameters
        ----------
        D4 : float
            Exponent value for the D4 parameter.
        
        Returns
        -------
        float
            The value of cexpt.
        """
        # cexpt is c
        # Arithmetic mean
        ret = D4
        return ret
    
    @staticmethod
    def Dsmax(D1, D2, D3, cexpt, phi_s, depth):
        """
        Calculate the maximum soil moisture (Dsmax).
        
        Parameters
        ----------
        D1 : float
            The D1 parameter value.
        D2 : float
            The D2 parameter value.
        D3 : float
            The D3 parameter value.
        cexpt : float
            The cexpt parameter value, typically equal to D4.
        phi_s : float
            The saturated soil water content (m³/m³).
        depth : float
            Depth of the soil layer (m).
        
        Returns
        -------
        float
            The maximum soil moisture (Dsmax) in mm or mm/day.
        """
        # ceta_s (maximum soil moisture, mm) = phi_s * (depth * unit_factor1), phi_s (Saturated soil water content, m3/m3)
        # Dsmax, mm or mm/day, 0.1~30.0, 10 is a common value
        # layer3
        # Harmonic mean
        Dsmax_min = 0.1
        Dsmax_max = 30.0
        unit_factor1 = 1000
        ret = D2 + D1*(phi_s * (depth * unit_factor1))
        # ret = D2 * (phi_s * (depth * unit_factor1) - D3) ** cexpt + D1*(phi_s * (depth * unit_factor1))
        
        ret[ret > Dsmax_max] = Dsmax_max
        ret[ret < Dsmax_min] = Dsmax_min
        # ret = ret if ret < Dsmax_max else Dsmax_max
        # ret = ret if ret > Dsmax_min else Dsmax_min
        return ret

    @staticmethod
    def Ds(D1, D3, Dsmax):
        """
        Calculate the Ds parameter, typically used as a fraction.
        
        Parameters
        ----------
        D1 : float
            The D1 parameter value.
        D3 : float
            The D3 parameter value.
        Dsmax : float
            The maximum soil moisture (Dsmax).
        
        Returns
        -------
        float
            The Ds value, bounded between a minimum and maximum range.
        """
        # [day^-D4] or fraction, 0.0001~1, 0.02 is a common value
        # Harmonic mean
        Ds_min = 0.0001
        Ds_max = 1.0
        ret = D1 * D3 / Dsmax
        
        ret[ret > Ds_max] = Ds_max
        ret[ret < Ds_min] = Ds_min
        # ret = ret if ret < Ds_max else Ds_max
        # ret = ret if ret > Ds_min else Ds_min
        return ret

    @staticmethod
    def Ws(D3, phi_s, depth):
        """
        Calculate the fraction of water available in the soil (Ws).
        
        Parameters
        ----------
        D3 : float
            The D3 parameter value (mm).
        phi_s : float
            The saturated soil water content (m³/m³).
        depth : float
            The depth of the soil layer (m).
        
        Returns
        -------
        float
            The fraction of available water in the soil (Ws), bounded between a minimum and maximum range.
        """
        # fraction, 0.0001~1, 0.8 is a common value
        # Arithmetic mean
        Ws_min = 0.0001
        Ws_max = 1.0
        unit_factor1 = 1000
        ret = D3 / (phi_s * depth * unit_factor1)
        
        ret[ret > Ws_max] = Ws_max
        ret[ret < Ws_min] = Ws_min
        # ret = ret if ret < Ws_max else Ws_max
        # ret = ret if ret > Ws_min else Ws_min
        return ret
    
    # def Ds():
    #     pass

    # def Dsmax(Ks, slope, beta):
    #     return Ks * slope / (10 ** beta)
    
    # def Ws(Wf, Wm, beta):
    #     return Wf / Wm * beta
    
    @staticmethod
    def init_moist(phi_s, depth):
        """
        Initialize the soil moisture (init_moist) in mm.
        
        Parameters
        ----------
        phi_s : float
            The saturated soil water content (m³/m³).
        depth : float
            The depth of the soil layer (m).
        
        Returns
        -------
        float
            The initialized soil moisture in mm.
        """
        # init_moist, mm
        # Arithmetic mean
        unit_factor1 = 1000.0
        ret = phi_s * (depth * unit_factor1)
        return ret
    
    @staticmethod
    def dp(g):
        """
        Calculate the dp parameter based on the given constant (g).
        
        Parameters
        ----------
        g : float
            A constant parameter for the model (typically 1.0 (0.9, 1.1)).
        
        Returns
        -------
        float
            The dp value.
        """
        # 1.0 (0.9, 1.1)
        # Arithmetic mean
        ret = 4.0 * g
        return ret
    
    @staticmethod
    def bubble(expt, g1, g2):
        """
        Calculate the bubble parameter based on Schaperow et al. (2021).
        
        Parameters
        ----------
        expt : float
            The exponent in Campbell's equation.
        g1 : float
            Constant parameter for the model (typically 0.32 (0.1, 0.8)).
        g2 : float
            Constant parameter for the model (typically 4.3 (0.0, 10.0)).
        
        Returns
        -------
        float
            The bubble parameter value.
        """
        # Schaperow, J., Li, D., Margulis, S., and Lettenmaier, D.: A near-global, high resolution land surface parameter dataset for the variable infiltration capacity model, Scientific Data, 8, 216, 10.1038/s41597-021-00999-4, 2021. 
        # g1, g2: 0.32 (0.1, 0.8), 4.3 (0.0, 10.0)
        # Arithmetic mean
        ret = g1 * expt + g2
        return ret
    
    @staticmethod
    def quartz(sand, g):
        """
        Calculate the quartz content in the soil based on sand content and g.
        
        Parameters
        ----------
        sand : float
            The sand content in the soil (%).
        g : float
            Constant parameter for the model (typically 0.8 (0.7, 0.9)).
        
        Returns
        -------
        float
            The quartz content, bounded between a minimum and maximum range.
        """
        # g: 0.8 (0.7, 0.9)
        # Arithmetic mean
        quartz_min = 0.0
        quartz_max = 1.0
        unit_factor1 = 100
        
        ret = sand * g / unit_factor1
        
        ret[ret > quartz_max] = quartz_max
        ret[ret < quartz_min] = quartz_min
        # ret = ret if ret < quartz_max else quartz_max
        # ret = ret if ret > quartz_min else quartz_min
        return ret
    
    @staticmethod
    def bulk_density(bulk_density, g):
        """
        Calculate the bulk density of the soil based on the given value and g.
        
        Parameters
        ----------
        bulk_density : float
            The bulk density of the soil (kg/m³), read from file.
        g : float
            Constant parameter for the model (typically 1.0 (0.9, 1.1)).
        
        Returns
        -------
        float
            The bulk density value, bounded between a minimum and maximum range.
        """
        # read from file
        # g: 1.0 (0.9, 1.1)
        # Arithmetic mean
        bd_min = 805.0
        bd_max = 1880.0
        
        ret = bulk_density * g
        
        ret[ret > bd_max] = bd_max
        ret[ret < bd_min] = bd_min
        
        # bd_slope = (bd_temp - bd_min) / (bd_max - bd_min)
        # bd_slope[bd_slope > 1.0] = 1.0
        # bd_slope[bd_slope < 0.0] = 0.0
        # ret = bd_slope * (bd_max - bd_min) + bd_min

        return ret
        
    # def bulk_density(bd_in, g):
    #     bd_min = 805.0
    #     bd_max = 1880.0
    #     bd_temp = g * bd_in
    #     bdslope = (bd_temp - bd_min) / (bd_max - bd_min)
        
    #     ret = bdslope * (bd_max - bd_min) + bd_min
    #     return ret
    
    @staticmethod
    def soil_density(g):
        """
        Calculate the soil mineral density based on a scaling factor.
        
        Parameters
        ----------
        g : float
            Constant parameter for the model (typically 1.0 (0.9, 1.1)).
        
        Returns
        -------
        float
            The soil mineral density (kg/cm³), scaled by the factor `g`.
        """
        # g: 1.0 (0.9, 1.1)
        # Arithmetic mean
        srho = 2685.0  # mineral density kg/cm3
        ret = srho * g
        return ret
    
    @staticmethod
    def Wcr_FRACT(fc, phi_s, g):
        """
        Calculate the fraction of water retention at field capacity (Wcr) based on soil properties.
        
        Parameters
        ----------
        fc : float
            The field capacity of the soil (m³/m³).
        phi_s : float
            The saturated soil water content (m³/m³).
        g : float
            Constant parameter for the model (typically 1.0 (0.8, 1.2)).
        
        Returns
        -------
        float
            The fraction of water retention, bounded between a minimum and maximum range.
        """
        # g: 1.0 (0.8, 1.2), ~=70%*fc
        # Arithmetic mean
        fract_min = 0.0001
        fract_max = 1.0
        
        ret = g * fc / phi_s
        
        ret[ret > fract_max] = fract_max
        ret[ret < fract_min] = fract_min
        
        return ret
    
    @staticmethod
    def wp(phi_s, b_retcurve, psis, g):
        """
        Calculate the wilting point based on Campbell's (1974) equation.
        
        Parameters
        ----------
        phi_s : float
            The saturated soil water content (m³/m³).
        b_retcurve : float
            The exponent of the retention curve.
        psis : float
            The soil water potential at saturation (kPa).
        g : float
            Constant parameter for the model (typically 1.0 (0.8, 1.2)).
        
        Returns
        -------
        float
            The wilting point (θ), computed based on the Campbell equation.
        """
        # campbell 1974, ψ = ψc(θ/θs)^-b, saturation condition
        # ψ = ψc(θ/θs)^-b -> ψ/ψc = (θ/θs)^-b -> θ/θs = (ψ/ψc)^(-1/b) -> θ = θs * ψ/ψc^(-1/b)
        # g: 1.0 (0.8, 1.2)
        # Arithmetic mean
        psi_wp = -1500  # -1500~-2000kPa
        ret = g * phi_s * (psi_wp / psis) ** (-1 / b_retcurve)
        return ret
    
    @staticmethod
    def Wpwp_FRACT(wp, phi_s, g): # wp: wilting point
        """
        Calculate the fraction of the wilting point moisture content (Wpwp) based on soil properties.
        
        Parameters
        ----------
        wp : float
            The wilting point moisture content (m³/m³).
        phi_s : float
            The saturated soil water content (m³/m³).
        g : float
            Constant parameter for the model (typically 1.0 (0.8, 1.2)).
        
        Returns
        -------
        float
            The fraction of the wilting point moisture content, bounded between a minimum and maximum range.
        """
        # g: 1.0 (0.8, 1.2)
        # Arithmetic mean
        fract_min = 0.0001
        fract_max = 1.0
        
        ret = g * wp / phi_s
        
        ret[ret > fract_max] = fract_max
        ret[ret < fract_min] = fract_min
        return ret
    
    @staticmethod
    def rough(g):
        """
        Calculate the soil surface roughness based on a scaling factor.
        
        Parameters
        ----------
        g : float
            Constant parameter for the model (typically 1.0 (0.9, 1.1)).
        
        Returns
        -------
        float
            The surface roughness (m), scaled by the factor `g`.
        """
        # g: 1.0 (0.9, 1.1)
        # Arithmetic mean
        ret = 0.001 * g
        return ret
    
    @staticmethod
    def snow_rough(g):
        """
        Calculate the snow surface roughness based on a scaling factor.
        
        Parameters
        ----------
        g : float
            Constant parameter for the model (typically 1.0 (0.9, 1.1)).
        
        Returns
        -------
        float
            The snow surface roughness (m), scaled by the factor `g`.
        """
        # snow roughness of snowpack, 0.001~0.03
        # g: 1.0 (0.9, 1.1)
        ret = 0.0005 * g
        return ret
    
    # def avg_T():
    #     # read from file
    #     pass
    
    # def annual_prec():
    #     # read from file
    #     pass
    
    @staticmethod
    def off_gmt(lon):
        """
        Calculate the offset from GMT based on longitude.
        
        Parameters
        ----------
        lon : float
            The longitude of the location (degrees).
        
        Returns
        -------
        float
            The offset from GMT in hours.
        """
        ret = lon * 24 / 360
        return ret
    
    @staticmethod
    def fs_active(activate=0):
        """
        Activate or deactivate a flag.
        
        Parameters
        ----------
        activate : int, optional
            A flag to activate or deactivate (default is 0, which means inactive).
        
        Returns
        -------
        int
            The activation flag value.
        """
        ret = activate
        return ret
    
    # def resid_moist(): # set as 0
    #     ret = 0.0
    #     return ret