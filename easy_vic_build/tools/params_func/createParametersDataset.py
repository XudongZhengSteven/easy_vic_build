# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
from netCDF4 import Dataset

# TODO Fill_Value_

def createParametersDataset(dst_path, lat_list, lon_list):
    # create dataset
    params_dataset = Dataset(dst_path, "w", format="NETCDF4")
    
    # ===================== define dimensions ===================== 
    lat = params_dataset.createDimension("lat", len(lat_list))
    lon = params_dataset.createDimension("lon", len(lon_list))
    nlayer = params_dataset.createDimension("nlayer", 3)
    root_zone = params_dataset.createDimension("root_zone", 3)
    veg_class = params_dataset.createDimension("veg_class", 14)
    month = params_dataset.createDimension("month", 12)
    
    # ===================== define variables ======================
    #* variables: dimension variables
    lat_v = params_dataset.createVariable("lat", "f8", ("lat",))  # 1D array
    lon_v = params_dataset.createVariable("lon", "f8", ("lon",))   # 1D array
    nlayer_v = params_dataset.createVariable("nlayer", 'i4', ("nlayer",))
    root_zone_v = params_dataset.createVariable("root_zone", 'i4', ("root_zone",))
    veg_class_v = params_dataset.createVariable("veg_class", 'i4', ("veg_class",))
    month_v = params_dataset.createVariable("month", 'i4', ("month",))
    
    #* variables: Soil parameters
    run_cell = params_dataset.createVariable("run_cell", 'i4', ("lat", "lon",))
    grid_cell = params_dataset.createVariable("grid_cell", 'i4', ("lat", "lon",))
    lats = params_dataset.createVariable("lats", "f8", ("lat", "lon",))  # 2D array
    lons = params_dataset.createVariable("lons", "f8", ("lat", "lon",))  # 2D array
    
    D1 = params_dataset.createVariable("D1", "f8", ("lat", "lon",))  # layer3
    D2 = params_dataset.createVariable("D2", "f8", ("lat", "lon",))  # layer3
    D3 = params_dataset.createVariable("D3", "f8", ("lat", "lon",))  # layer3
    D4 = params_dataset.createVariable("D4", "f8", ("lat", "lon",))  # cexpt
    
    b_infilt = params_dataset.createVariable("infilt", "f8", ("lat", "lon",)) # *(Ele std PTF + scaling(Arithmetic)), b
    Ds = params_dataset.createVariable("Ds", "f8", ("lat", "lon",))  # based on D1/2/3/4
    Dsmax = params_dataset.createVariable("Dsmax", "f8", ("lat", "lon",))  # based on D1/2/3/4
    Ws = params_dataset.createVariable("Ws", "f8", ("lat", "lon",))  # based on D1/2/3/4
    
    cexpt = params_dataset.createVariable("c", "f8", ("lat", "lon",))  # general set to 2
    expt = params_dataset.createVariable("expt", "f8", ("nlayer", "lat", "lon",)) # *(sand/clay percentage PTF, Cosby 1984 + scaling(Majority)，b in Campbell equation)
    Ksat = params_dataset.createVariable("Ksat", "f8", ("nlayer", "lat", "lon",)) # *(sand/clay percentage PTF, Cosby 1984 + scaling(Harmonic))
    phi_s = params_dataset.createVariable("phi_s", "f8", ("nlayer", "lat", "lon",))
    psis = params_dataset.createVariable("psis", "f8", ("nlayer", "lat", "lon",))
    b_retcurve = params_dataset.createVariable("b_retcurve", "f8", ("nlayer", "lat", "lon",))
    fc = params_dataset.createVariable("fc", "f8", ("nlayer", "lat", "lon",))
    
    init_moist = params_dataset.createVariable("init_moist", "f8", ("nlayer", "lat", "lon",))
    elev = params_dataset.createVariable("elev", "f8", ("lat", "lon",))
    depth = params_dataset.createVariable("depth", "f8", ("nlayer", "lat", "lon",))  # *(percentile1/2/3 * Ztot(constant, soil datasets) + scaling(Arithmetic)), D1/2/3
    avg_T = params_dataset.createVariable("avg_T", "f8", ("lat", "lon",))
    dp = params_dataset.createVariable("dp", "f8", ("lat", "lon",))
    bubble = params_dataset.createVariable("bubble", "f8", ("nlayer", "lat", "lon",))
    quartz = params_dataset.createVariable("quartz", "f8", ("nlayer", "lat", "lon",))
    bulk_density = params_dataset.createVariable("bulk_density", "f8", ("nlayer", "lat", "lon",))  # *(soil data + scaling), Arithmetic, BD
    soil_density = params_dataset.createVariable("soil_density", "f8", ("nlayer", "lat", "lon",))
    
    off_gmt = params_dataset.createVariable("off_gmt", "f8", ("lat", "lon",))
    Wcr_FRACT = params_dataset.createVariable("Wcr_FRACT", "f8", ("nlayer", "lat", "lon",))
    wp = params_dataset.createVariable("wp", "f8", ("nlayer", "lat", "lon",))
    Wpwp_FRACT = params_dataset.createVariable("Wpwp_FRACT", "f8", ("nlayer", "lat", "lon",))
    rough = params_dataset.createVariable("rough", "f8", ("lat", "lon",))
    snow_rough = params_dataset.createVariable("snow_rough", "f8", ("lat", "lon",))
    annual_prec = params_dataset.createVariable("annual_prec", "f8", ("lat", "lon",))
    resid_moist = params_dataset.createVariable("resid_moist", "f8", ("nlayer", "lat", "lon",))
    
    # frozen soil algorithm
    fs_active = params_dataset.createVariable("fs_active", int, ("lat", "lon",))
    # frost_slope = dst_dataset.createVariable("frost_slope", "f8", ("lat", "lon",))  # SPATIAL_FROST == FALSE
    # max_snow_distrib_slope = dst_dataset.createVariable("max_snow_distrib_slope", "f8", ("lat", "lon",))  # SPATIAL_SNOW == FALSE
    # July_Tavg = dst_dataset.createVariable("July_Tavg", "f8", ("lat", "lon",))  # COMPUTE_TREELINE == FALSE
    
    # variables: Soil parameters: optional for ORGANIC_FRACT = TRUE
    # organic = dst_dataset.createVariable("organic", "f8", ("nlayer", "lat", "lon",))
    # bulk_dens_org = dst_dataset.createVariable("bulk_dens_org", "f8", ("nlayer", "lat", "lon",))
    # soil_dens_org = dst_dataset.createVariable("soil_dens_org", "f8", ("nlayer", "lat", "lon",))
    
    #* variables: Veg parameters
    Nveg = params_dataset.createVariable("Nveg", 'i4', ("lat", "lon",))
    Cv = params_dataset.createVariable("Cv", "f8", ("veg_class", "lat", "lon",))
    root_depth = params_dataset.createVariable("root_depth", "f8", ("veg_class", "root_zone", "lat", "lon",))
    root_fract = params_dataset.createVariable("root_fract", "f8", ("veg_class", "root_zone", "lat", "lon",))

    # variables: Veg parameters: optional for BLOWING_SNOW = TRUE
    # sigma_slope
    # lag_one
    # fetch
    
    # variables: Veg parameters: optional for VEGPARAM_LAI = TRUE
    LAI = params_dataset.createVariable("LAI", "f8", ("veg_class", "month", "lat", "lon",))

    # variables: Veg parameters: optional for VEGPARAM_FCAN = TRUE
    fcanopy = params_dataset.createVariable("fcanopy", "f8", ("veg_class", "month", "lat", "lon",))

    # variables: Veg parameters: optional for VEGPARAM_ALBEDO = TRUE
    albedo = params_dataset.createVariable("albedo", "f8", ("veg_class", "month", "lat", "lon",))

    # variables: Veg Library parameters
    overstory = params_dataset.createVariable("overstory", 'i4', ("veg_class", "lat", "lon",))
    rarc = params_dataset.createVariable("rarc", "f8", ("veg_class", "lat", "lon",))
    rmin = params_dataset.createVariable("rmin", "f8", ("veg_class", "lat", "lon",))
    veg_rough = params_dataset.createVariable("veg_rough", "f8", ("veg_class", "month", "lat", "lon",))
    displacement = params_dataset.createVariable("displacement", "f8", ("veg_class", "month", "lat", "lon",))
    wind_h = params_dataset.createVariable("wind_h", "f8", ("veg_class", "lat", "lon",))
    RGL = params_dataset.createVariable("RGL", "f8", ("veg_class", "lat", "lon",))
    rad_atten = params_dataset.createVariable("rad_atten", "f8", ("veg_class", "lat", "lon",))
    wind_atten = params_dataset.createVariable("wind_atten", "f8", ("veg_class", "lat", "lon",))
    trunk_ratio = params_dataset.createVariable("trunk_ratio", "f8", ("veg_class", "lat", "lon",))
    
    # variables: Veg Library parameters: optional for VEGLIB_PHOTO = TRUE
    # Ctype
    # MaxCarboxRate
    # MaxETransport/CO2Specificity
    # LightUseEff
    # NscaleFlag
    # Wnpp_inhib
    # NPPfactor_sat

    #* variables: Elevation/Snow Bands parameters: Optional, account for considering grid cell is not flat
    # cellnum = params_dataset.createVariable("cellnum", "f8", ("lat", "lon",))
    # AreaFract = params_dataset.createVariable("AreaFract", "f8", ("snow_band", "lat", "lon",))
    # elevation = params_dataset.createVariable("elevation", "f8", ("snow_band", "lat", "lon",))
    # Pfactor = params_dataset.createVariable("Pfactor", "f8", ("snow_band", "lat", "lon",))
    
    # ===================== define variables attribute ======================
    lat_v.units = "degrees_north"
    lat_v.long_name = "latitude of grid cell center"
    lat_v.standard_name = "latitude"
    lat_v.axis = "Y"
    
    lon_v.units = "degrees_east"
    lon_v.long_name = "longitude of grid cell center"
    lon_v.standard_name = "longitude"
    lon_v.axis = "X"
    
    lats.long_name = "lats 2D"
    lats.description = "Latitude of grid cell 2D"
    lats.units = "degrees"
    
    lons.long_name = "lons 2D"
    lons.description = "longitude of grid cell 2D"
    lons.units = "degrees"
    
    nlayer_v.long_name = "soil layer"
    root_zone_v.long_name = "root zone"
    
    veg_class_v.long_name = "land cover class ID code"
    veg_class_v.standard_name = "land cover class ID code"

    month_v.long_name = "month of climatological year, 1-12"

    run_cell.long_name = "run cell"
    run_cell.description = "1 = Run Grid Cell, 0 = Do Not Run"

    grid_cell.long_name = "grid cell"
    grid_cell.description = "Grid cell number"
    
    D1.long_name = "D1"
    D1.description = "D1 is coefficients of linear reservoirs, it is related to vertical water transmission in saturated soil due to the gravity as described by Darcy's law"
    D1.units = "day^-1"
    
    D2.long_name = "D2"
    D2.description = "D2 is coefficients of nonlinear reservoirs, it is related to vertical water transmission in saturated soil due to the gravity as described by Darcy's law"
    D2.units = "day^-D4"
    
    D3.long_name = "D3"
    D3.description = "The formulation of D3, the soil moisture at which the baseflow transitions from linear to nonlinear, is based on the assumption that soil moisture exceeding field capacity is more freely drained, as suggested by the SAC-SMA transfer function"
    D3.units = "mm"
    
    D4.long_name = "D4"
    D4.description = "D4, same as c, typically is 2"
    D4.units = "N/A"

    b_infilt.long_name = "infilt"
    b_infilt.description = "Variable infiltration curve parameter (binfilt)"
    b_infilt.units = "N/A"
    
    Ds.long_name = "Ds"
    Ds.description = "Fraction of Dsmax where non-linear baseflow begins"
    Ds.units = "fraction"

    Dsmax.long_name = "Dsmax"
    Dsmax.description = "Maximum velocity of baseflow"
    Dsmax.units = "mm/day"

    Ws.long_name = "Ws"
    Ws.description = "Fraction of maximum soil moisture where non-linear baseflow occurs"
    Ws.units = "fraction"

    cexpt.long_name = "c"
    cexpt.description = "Exponent used in baseflow curve, normally set to 2"
    cexpt.units = "N/A"

    expt.long_name = "expt"
    expt.description = "Exponent n (=3+2/lambda) in Campbells eqn for hydraulic conductivity, HBH 5.6 (where lambda = soil pore size distribution parameter).  Values should be > 3.0."
    expt.units = "N/A"

    Ksat.long_name = "Ksat"
    Ksat.description = "Saturated hydrologic conductivity"
    Ksat.units = "mm/day"

    phi_s.long_name = "phi_s"
    phi_s.description = "Soil moisture diffusion parameter"
    phi_s.units = "mm/mm"

    psis.long_name = "psis"
    psis.description = "saturation matric potential"
    psis.units = "kPa/cm-H2O"

    b_retcurve.long_name = "b_retcurve"
    b_retcurve.description = "slope of cambell retention curve in log space (b)"
    b_retcurve.units = "N/A"

    fc.long_name = "fc"
    fc.description = "field capacity"
    fc.units = "m3/m3"
    
    init_moist.long_name = "init_moist"
    init_moist.description = "Initial layer moisture content"
    init_moist.units = "mm"

    elev.long_name = "elev"
    elev.description = "Average elevation of grid cell"
    elev.units = "m"

    depth.long_name = "depth"
    depth.description = "Thickness of each soil layer"
    depth.units = "m"
    
    avg_T.long_name = "avg_T"
    avg_T.description = "Average soil temperature, used as the bottom boundary for soil heat flux solutions"
    avg_T.units = "C"
    
    dp.long_name = "dp"
    dp.description = "Soil thermal damping depth (depth at which soil temperature remains constant through the year, ~4 m)"
    dp.units = "m"
    
    bubble.long_name = "bubble"
    bubble.description = "Bubbling pressure of soil. Values should be > 0.0"
    bubble.units = "cm"

    quartz.long_name = "quartz"
    quartz.description = "Quartz content of soil"
    quartz.units = "fraction"

    bulk_density.long_name = "bulk_density"
    bulk_density.description = "Bulk density of soil layer"
    bulk_density.units = "kg/m3"
    
    soil_density.long_name = "soil_density"
    soil_density.description = "Soil particle density, normally 2685 kg/m3"
    soil_density.units = "kg/m3"

    off_gmt.long_name = "off_gmt"
    off_gmt.description = "Time zone offset from GMT. This parameter determines how VIC interprets sub-daily time steps relative to the model start date and time"
    off_gmt.units = "hours"

    Wcr_FRACT.long_name = "Wcr_FRACT"
    Wcr_FRACT.description = "Fractional soil moisture content at the critical point (~70% of field capacity) (fraction of maximum moisture)"
    Wcr_FRACT.units = "fraction"
    
    wp.long_name = "wp"
    wp.description = "computed field capacity, wilting point"
    wp.units = "fraction"

    Wpwp_FRACT.long_name = "Wpwp_FRACT"
    Wpwp_FRACT.description = "Fractional soil moisture content at the wilting point (fraction of maximum moisture)"
    Wpwp_FRACT.units = "fraction"

    rough.long_name = "rough"
    rough.description = "Surface roughness of bare soil"
    rough.units = "m"
    
    snow_rough.long_name = "snow_rough"
    snow_rough.description = "Surface roughness of snowpack"
    snow_rough.units = "m"

    annual_prec.long_name = "annual_prec"
    annual_prec.description = "Average annual precipitation"
    annual_prec.units = "mm"
    
    resid_moist.long_name = "resid_moist"
    resid_moist.description = "Soil moisture layer residual moisture"
    resid_moist.units = "fraction"

    fs_active.long_name = "fs_active"
    fs_active.description = "If set to 1, then frozen soil algorithm is activated for the grid cell. A 0 indicates that frozen soils are not computed even if soil temperatures fall below 0C"
    fs_active.units = "binary"
    
    Nveg.long_name = "Nveg"
    Nveg.description = "Number of vegetation tiles in the grid cell"
    Nveg.units = "int"
    
    Cv.long_name = "Cv"
    Cv.description = "Fraction of grid cell covered by vegetation tile"
    Cv.units = "fraction"

    root_depth.long_name = "root_depth"
    root_depth.description = "Root zone thickness (sum of depths is total depth of root penetration)"
    root_depth.units = "m"

    root_fract.long_name = "root_fract"
    root_fract.description = "Fraction of root in the current root zone"
    root_fract.units = "fraction"

    LAI.long_name = "LAI"
    LAI.description = "Leaf Area Index, one per month, Climatological Mean"
    LAI.units = "m2/m2"

    fcanopy.long_name = "fcanopy"
    fcanopy.description = "Canopy fraction Climatological Mean"
    fcanopy.units = "fraction"

    albedo.long_name = "albedo"
    albedo.description = "Shortwave albedo for vegetation type, MODIS BSA here"
    albedo.units = "fraction"

    overstory.long_name = "overstory"
    overstory.description = "Flag to indicate whether or not the current vegetation type has an overstory (1 for overstory present [e.g. trees], 0 for overstory not present [e.g. grass])"
    overstory.units = "binary"

    rarc.long_name = "rarc"
    rarc.description = "Architectural resistance of vegetation type (~2 s/m)"
    rarc.units = "s/m"

    rmin.long_name = "rmin"
    rmin.description = "Minimum stomatal resistance of vegetation type (~100 s/m)"
    rmin.units = "s/m"
    
    veg_rough.long_name = "veg_rough"
    veg_rough.description = "Vegetation roughness length (typically 0.123 * vegetation height)"
    veg_rough.units = "m"
    
    displacement.long_name = "displacement"
    displacement.description = "Vegetation displacement height (typically 0.67 * vegetation height)"
    displacement.units = "m"
    
    wind_h.long_name = "wind_h"
    wind_h.description = "Height at which wind speed is measured"
    wind_h.units = "m"
    
    RGL.long_name = "RGL"
    RGL.description = "Minimum incoming shortwave radiation at which there will be transpiration. For trees this is about 30 W/m^2, for crops about 100 W/m^2"
    RGL.units = "W/m2"

    rad_atten.long_name = "rad_atten"
    rad_atten.description = "Radiation attenuation factor. Normally set to 0.5, though may need to be adjusted for high latitudes"
    rad_atten.units = "fraction"

    wind_atten.long_name = "wind_atten"
    wind_atten.description = "Wind speed attenuation through the overstory. The default value has been 0.5"
    wind_atten.units = "fraction"

    trunk_ratio.long_name = "trunk_ratio"
    trunk_ratio.description = "Ratio of total tree height that is trunk (no branches). The default value has been 0.2"
    trunk_ratio.units = "fraction"

    # Global attributes
    params_dataset.title = "VIC5 image params dataset"
    params_dataset.note = "params dataset generated by XudongZheng, zhengxd@sehemodel.club"
    params_dataset.Conventions = "CF-1.6"

    return params_dataset


def createFlowDirectionFile(dst_path, lat_list, lon_list):
    # create dataset
    flow_direction_dataset = Dataset(dst_path, "w", format="NETCDF4")
    
    # ===================== define dimensions ===================== 
    lat = flow_direction_dataset.createDimension("lat", len(lat_list))
    lon = flow_direction_dataset.createDimension("lon", len(lon_list))

    # ===================== define variables ======================
    #* variables: dimension variables
    lat_v = flow_direction_dataset.createVariable("lat", "f8", ("lat",))  # 1D array
    lon_v = flow_direction_dataset.createVariable("lon", "f8", ("lon",))   # 1D array

    #* variables: 
    Basin_ID = flow_direction_dataset.createVariable("Basin_ID", 'i4', ("lat", "lon",), fill_value=-9999)
    Flow_Direction = flow_direction_dataset.createVariable("Flow_Direction", 'i4', ("lat", "lon",), fill_value=-9999)
    Flow_Distance = flow_direction_dataset.createVariable("Flow_Distance", "f8", ("lat", "lon",), fill_value=-9999.0)
    Source_Area = flow_direction_dataset.createVariable("Source_Area", "f8", ("lat", "lon",), fill_value=-9999.0)

    # ===================== define variables attribute ======================
    lat_v.units = "degrees_north"
    lat_v.long_name = "latitude of grid cell center"
    lat_v.standard_name = "latitude"
    lat_v.axis = "Y"
    
    lon_v.units = "degrees_east"
    lon_v.long_name = "longitude of grid cell center"
    lon_v.standard_name = "longitude"
    lon_v.axis = "X"
    
    Basin_ID.long_name = "Basin_ID"
    Basin_ID.description = "grids in basin is labeled as 1, otherwise 0"
    
    Flow_Direction.long_name = "Flow_Direction"
    Flow_Direction.description = "Flow_Direction, generated from filled DEM"
    Flow_Direction.units = "ARCGIS_Directions"
    
    Flow_Distance.long_name = "Flow_Distance"
    Flow_Distance.description = "Flow_Distance, generated from Flow_Direction, x_length, and y_length"
    Flow_Distance.units = "meter"
    
    Source_Area.long_name = "Source_Area"
    Source_Area.description = "Source_Area, generated from Flow_Direction"
    Source_Area.units = "Grid cells"
    
    # Global attributes
    flow_direction_dataset.title = "VIC5 image params dataset, for RVIC"
    flow_direction_dataset.note = "params dataset generated by XudongZheng, zhengxd@sehemodel.club"
    flow_direction_dataset.Conventions = "CF-1.6"

    return flow_direction_dataset