# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import numpy as np
from .params_set import *

class TF_VIC:
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def b_infilt(ele_std, g1, g2):
        # Dumenil, L. and Todini, E.: A rainfall-runoff scheme for use in the Hamburg climate model, Advances in theoretical hydrology, 129-157, 1992.
        # Hurk, B. and Viterbo, P.: The Torne-Kalix PILPS 2(e) experiment as a test bed for modifications to the ECMWF land surface scheme, Global Planet Change, 38, 165-173, 10.1016/S0921-8181(03)00027-4, 2003.
        # b_infilt, N/A, 0.01~0.5
        # g1, g2: 0.0 (-2.0, 1.0), 1.0 (0.8, 1.2)  # TODO recheck (ele_std - g1) / (ele_std + g2*10)
        # Arithmetic mean
        b_infilt_min = 0.01
        b_infilt_max = 0.50
        ret = (np.log(ele_std) - g1) / (np.log(ele_std) + g2*10)
        
        ret[ret > b_infilt_max] = b_infilt_max
        ret[ret < b_infilt_min] = b_infilt_min
        return ret
    
    @staticmethod
    def total_depth(total_depth_original, g):
        # total_depth, m
        # g: 1.0 (0.1, 4.0)
        # Arithmetic mean
        ret = total_depth_original * g
        return ret
    
    @staticmethod
    def depth(total_depth, g1, g2):
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
        # saturation matric potential, ψs, Cosby et al. WRR 1984, logψs = -0.0095x1 (+ 0.0063x2) + 1.54
        # kPa/cm-H2O
        # g1, g2, g3: 1.54 (1.0, 2.0), -0.0095 (-0.01, -0.009), 0.0063 (0.006, 0.0066)
        # Arithmetic mean
        ret = g1 + g2 * sand + g3 * silt
        unit_factor1 = 0.0980665  # 0.0980665 kPa/cm-H2O. Cosby give psi_sat in cm of water (cm-H2O), 1cm H₂O=0.0980665 kPa
        ret = -1 * (10 ** (ret)) * unit_factor1  # TODO recheck -1
        return ret

    @staticmethod
    def b_retcurve(sand, clay, g1, g2, g3):
        # b (slope of cambell retention curve in log space), N/A, ψ = ψc(θ/θs)-b
        # Cosby et al. WRR 1984, b = 0.157x3 (- 0.003x1) + 3.10, b~=0~20
        # g1, g2, g3: 3.1 (2.5, 3.6), 0.157 (0.1, 0.2), -0.003 (-0.005, -0.001)
        # Arithmetic mean
        ret = g1 + g2 * sand + g3 * clay
        return ret
    
    @staticmethod
    def expt(b_retcurve, g1, g2):
        # the exponent in Campbell’s equation for hydraulic conductivity, k = ks (θ/θs)2b+3
        # expt = 2b+3 should be > 3
        # g1, g2: 3.0 (2.8, 3.2), 2.0 (1.5, 2.5)
        # Arithmetic mean
        ret = g1 + g2 * b_retcurve
        return ret
    
    @staticmethod
    def fc(phi_s, b_retcurve, psis, sand, g):
        # campbell 1974, ψ = ψc(θ/θs)^-b, saturation condition
        # ψ = ψc(θ/θs)^-b -> ψ/ψc = (θ/θs)^-b -> θ/θs = (ψ/ψc)^(-1/b) -> θ = θs * ψ/ψc^(-1/b)
        # m3/m3 or %
        # g: 1.0 (0.8, 1.2)
        # Arithmetic mean
        psi_fc = np.full_like(phi_s, fill_value=-10)  # ψfc kPa/cm-H2O, -30~-10kPa
        psi_fc[sand <= 69] = -20  # TODO recheck -20, sand
        
        ret = g * phi_s * (psi_fc / psis) ** (-1 / b_retcurve)
        return ret
    
    @staticmethod
    def D1(Ks, slope_mean, g):
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
        # g: 2.0 (1.2, 2.5)
        # Arithmetic mean
        ret = g
        return ret
    
    @staticmethod
    def cexpt(D4): # set to D4
        # cexpt is c
        # Arithmetic mean
        ret = D4
        return ret
    
    @staticmethod
    def Dsmax(D1, D2, D3, cexpt, phi_s, depth):
        # ceta_s (maximum soil moisture, mm) = phi_s * (depth * unit_factor1), phi_s (Saturated soil water content, m3/m3)
        # Dsmax, mm or mm/day, 0.1~30.0, 10 is a common value
        # layer3
        # Harmonic mean
        Dsmax_min = 0.1
        Dsmax_max = 30.0
        unit_factor1 = 1000
        ret = D2 * (phi_s * (depth * unit_factor1) - D3) ** cexpt + D1*(phi_s * (depth * unit_factor1))
        
        ret[ret > Dsmax_max] = Dsmax_max
        ret[ret < Dsmax_min] = Dsmax_min
        # ret = ret if ret < Dsmax_max else Dsmax_max
        # ret = ret if ret > Dsmax_min else Dsmax_min
        return ret

    @staticmethod
    def Ds(D1, D3, Dsmax):
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
        # fraction, 0.0001~1, 0.8 is a common value
        # Arithmetic mean
        Ws_min = 0.0001
        Ws_max = 1.0
        unit_factor1 = 1000
        ret = D3 / phi_s / (depth * unit_factor1)
        
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
        # init_moist, mm
        # Arithmetic mean
        unit_factor1 = 1000.0
        ret = phi_s * (depth * unit_factor1)
        return ret
    
    @staticmethod
    def dp(g):
        # 1.0 (0.9, 1.1)
        # Arithmetic mean
        ret = 4.0 * g
        return ret
    
    @staticmethod
    def bubble(expt, g1, g2):
        # Schaperow, J., Li, D., Margulis, S., and Lettenmaier, D.: A near-global, high resolution land surface parameter dataset for the variable infiltration capacity model, Scientific Data, 8, 216, 10.1038/s41597-021-00999-4, 2021. 
        # g1, g2: 0.32 (0.1, 0.8), 4.3 (0.0, 10.0)
        # Arithmetic mean
        ret = g1 * expt + g2
        return ret
    
    @staticmethod
    def quartz(sand, g):
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
        # read from file
        # g: 1.0 (0.9, 1.1)
        # Arithmetic mean
        bd_min = 805.0
        bd_max = 1880.0
        
        bd_temp = bulk_density * g
        bd_slope = (bd_temp - bd_min) / (bd_max - bd_min)
        bd_slope[bd_slope > 1.0] = 1.0
        bd_slope[bd_slope < 0.0] = 0.0
        ret = bd_slope * (bd_max - bd_min) + bd_min

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
        # g: 1.0 (0.9, 1.1)
        # Arithmetic mean
        srho = 2685.0  # mineral density kg/cm3
        ret = srho * g
        return ret
    
    @staticmethod
    def Wcr_FRACT(fc, phi_s, g):
        # g: 1.0 (0.8, 1.2)
        # Arithmetic mean
        ret = g * fc / phi_s
        return ret
    
    @staticmethod
    def wp(phi_s, b_retcurve, psis, g):
        # campbell 1974, ψ = ψc(θ/θs)^-b, saturation condition
        # ψ = ψc(θ/θs)^-b -> ψ/ψc = (θ/θs)^-b -> θ/θs = (ψ/ψc)^(-1/b) -> θ = θs * ψ/ψc^(-1/b)
        # g: 1.0 (0.8, 1.2)
        # Arithmetic mean
        psi_wp = -1500  # -1500~-2000kPa
        ret = g * phi_s * (psi_wp / psis) ** (-1 / b_retcurve)
        return ret
    
    @staticmethod
    def Wpwp_FRACT(wp, phi_s, g): # wp: wilting point
        # g: 1.0 (0.8, 1.2)
        # Arithmetic mean
        ret = g * wp / phi_s
        return ret
    
    @staticmethod
    def rough(g):
        # g: 1.0 (0.9, 1.1)
        # Arithmetic mean
        ret = 0.001 * g
        return ret
    
    @staticmethod
    def snow_rough(g):
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
        ret = lon * 24 / 360
        return ret
    
    @staticmethod
    def fs_active(activate=0):
        ret = activate
        return ret
    
    # def resid_moist(): # set as 0
    #     ret = 0.0
    #     return ret