Usage
=====

After installation, you can use the package in the following sequence:

This deployment process adheres to the guidelines outlined in the official VIC documentation (https://vic.readthedocs.io/) and focuses primarily on the image driver version of VIC-5

1. **Build Modeling Directory**
===========================================

Use the `Evb_dir` class to build the modeling directory.

.. code-block:: python

   from easy_vic_build.Evb_dir_class import Evb_dir
   case_name = "example_case"
   evb_dir = Evb_dir(cases_home="../")
   evb_dir.builddir(case_name)

This will create a new modeling directory under `../example_case` and automatically manage file paths.

2. **Build Basins and Grids**
===========================================

Build the required basin extents for the model from a shapefile or a custom shapefile using the `Basins` module.

.. code-block:: python

   from easy_vic_build.tools.dpc_func.basin_grid_class import Basins

   basin_shp = Basins.from_shapefile(basin_shp_path)

Based on the basin extents and custom grid resolutions, use the `Grids` module or `dpc_func` module to build the spatial partitions needed for the model.

.. code-block:: python

   from easy_vic_build.tools.dpc_func.basin_grid_func import build_grid_shp

   grid_shp_level0, grid_shp_level1, grid_shp_level2, grid_shp_level3 = build_grid_shp(
       basin_shp,
       grid_res_level0,
       grid_res_level1,
       grid_res_level2,
       expand_grids_num=1,
       plot=True
   )

3. **Build DPC (Data Processing Class)**
===========================================

Inherit from `dataProcess_base` to create subclasses for levels 0-3, overriding the `processing_step` method to handle different data sources.

- `dpc_level0` handles and stores data such as DEM and soil.
- `dpc_level1` handles and stores soil temperature, annual precipitation, LULC, BSA, NDVI, LAI.
- `dpc_level2` handles and stores forcing data, including average air temperature, total precipitation (rain and snow), atmospheric pressure, incoming shortwave radiation, incoming longwave radiation, vapor pressure, and wind speed.

Refer to `examples/JRB_modeling/JRB_build_dpc.py` for implementation details.  
This process should be combined with functions from the `JRB_extractData_func` module.

4. **Build Domain**
===========================================

Use the `buildDomain` function to construct the domain file.  
The `reverse_lat` parameter indicates whether to reverse latitudes (usually `True` for Northern Hemisphere: large to small).

.. code-block:: python

   from easy_vic_build.bulid_Domain import buildDomain
   buildDomain(evb_dir_modeling, dpc_VIC_level1, reverse_lat)

5. **Build Parameters**
===========================================

Inherit from `buildParam_level0_interface` to define customized subclasses based on data sources.  
Read `dpc` instances and pass them to `buildParam_level0` to build effective parameters at level 0 scale.

.. code-block:: python

   from easy_vic_build.build_Param import buildParam_level0, buildParam_level1, scaling_level0_to_level1

   buildParam_level0_interface_instance = buildParam_level0(
       evb_dir_modeling,
       default_params_JRB["g_params"],
       SoilGrids_soillayerresampler,
       dpc_VIC_level0,
       TF_VIC_class=TF_VIC,
       buildParam_level0_interface_class=buildParam_level0_interface_JRB,
       reverse_lat=reverse_lat,
       stand_grids_lat_level0=None,
       stand_grids_lon_level0=None,
       rows_index_level0=None,
       cols_index_level0=None,
   )

Similarly, pass `dpc_level1` to `buildParam_level1` to build parameters at level 1 scale.  
Note that some parameters are left blank here and need to be scaled up from level 0.

.. code-block:: python

   buildParam_level1_interface_instance = buildParam_level1(
       evb_dir_modeling,
       dpc_VIC_level1,
       TF_VIC_class=TF_VIC,
       buildParam_level1_interface_class=buildParam_level1_interface_JRB,
       reverse_lat=reverse_lat,
       domain_dataset=domain_dataset,
       stand_grids_lat_level1=None,
       stand_grids_lon_level1=None,
       rows_index_level1=None,
       cols_index_level1=None,
   )

The scaling process converts parameters from level 0 scale to level 1.

.. code-block:: python

   params_dataset_level1, searched_grids_bool_index = scaling_level0_to_level1(
       params_dataset_level0, params_dataset_level1,
       searched_grids_bool_index=None,
       nlayer_list=[1, 2, 3],
   )

6. **Perform Hydroanalysis**
===========================================

You may need to perform hydrological analysis at the level 0 scale to provide high-resolution basin maps.  
By passing the level 0 scale DEM data, use the `buildHydroanalysis_level0` function to generate hydrological analysis data.  
This will create data such as flow direction, flow accumulation, basin boundaries, outlet points, and streamlines under the `Hydroanalysis` directory.

.. code-block:: python

   from easy_vic_build.build_hydroanalysis import buildHydroanalysis_level0, buildHydroanalysis_level1

   buildHydroanalysis_level0(
       evb_dir_hydroanalysis_level0,
       dem_level0_path,
       flow_direction_pkg="wbw",
       stream_acc_threshold=None,
       calculate_streamnetwork_threshold_kwargs={
           "method": "drainage_area",
           "drainage_area_km2": 0.01,
       },
       d8_streamnetwork_kwargs={
           "snap_dist": 0.001,
       },
       snap_outlet_to_stream_kwargs={
           "snap_dist": 30.0,
       },
       crs_str="EPSG:4326",
       esri_pointer=True,
       outlets_with_reference_coords=[hydroStation_coord_combined.lon.to_list(), hydroStation_coord_combined.lat.to_list()]
   )

By passing parameter files containing DEM information at level 1 scale along with the domain file,  
use the `buildHydroanalysis_level1` function to generate hydrological analysis data at level 1 scale.  
This will also produce flow direction, flow accumulation, basin boundaries, outlet points, and streamlines under the `Hydroanalysis` directory.  

Note that this step is essential for the routing model, as RVIC parameter preparation requires hydrological analysis data at level 1 scale.

.. code-block:: python

   from easy_vic_build.build_hydroanalysis import buildHydroanalysis_level0, buildHydroanalysis_level1

   evb_dir_modeling = build_modeling_dir(subname=f"{station_name}_{model_scale}")

   params_dataset_level0, params_dataset_level1 = readParam(evb_dir_modeling)

   domain_dataset = readDomain(evb_dir_modeling)

   buildHydroanalysis_level1(
       evb_dir_modeling,
       params_dataset_level1,
       domain_dataset,
       reverse_lat,
       stream_acc_threshold=10,
       flow_direction_pkg="wbw",
       crs_str="EPSG:4326",
       d8_streamnetwork_kwargs={
           "snap_dist": 0.001,
       },
       snap_outlet_to_stream_kwargs={
           "snap_dist": 30.0,
       },
       outlets_with_reference_coords=[hydroStation_coord_combined.lon.to_list(), hydroStation_coord_combined.lat.to_list()]
   )

7. **Build Meteorological Forcing**
===========================================

Read the processed forcing data contained in `dpc_level2`, and merge all data into a single DataFrame using `merge_grid_data`.  
Then pass the merged data to `buildMeteForcing` to prepare meteorological forcing data.  
This will generate annual NetCDF files under the `MeteForcing` directory.

.. code-block:: python

   dpc_VIC_level2_CDMet = dataProcess_VIC_level2_CDMet_JRB(
       evb_dir_modeling._dpc_VIC_level2_path.replace(".pkl", "_CDMet.pkl")
   )
   
   dpc_VIC_level2_CDMet.merge_grid_data()

   buildMeteForcing_interface_instance = build_MeteForcing.buildMeteForcing(
       evb_dir_modeling,
       dpc_VIC_level2_CDMet,
       date_period,
       date_period,
       buildMeteForcing_interface,
       reverse_lat=True,
       stand_grids_lat_level2=None,
       stand_grids_lon_level2=None,
       rows_index_level2=None,
       cols_index_level2=None,
       file_format="NETCDF4",
   )

8. **Build Global Parameters**
===========================================

Create the global parameter file by passing a parameter dictionary to `buildGlobalParam`.  
This will generate the global parameter file `global_param.txt` in the `GlobalParam` directory.

.. code-block:: python

   from easy_vic_build.build_GlobalParam import buildGlobalParam

   GlobalParam_dict = {'Simulation': {'MODEL_STEPS_PER_DAY': '1',
                                      'SNOW_STEPS_PER_DAY': '24',
                                      'RUNOFF_STEPS_PER_DAY': '24',
                                      'STARTYEAR': str(date_period[0][:4]),
                                      'STARTMONTH': str(int(date_period[0][4:6])),
                                      'STARTDAY': str(int(date_period[0][6:8])),
                                      'ENDYEAR': str(date_period[1][:4]),
                                      'ENDMONTH': str(int(date_period[1][4:6])),
                                      'ENDDAY': str(int(date_period[1][6:8])),
                                      'OUT_TIME_UNITS': 'DAYS'},
                        'Output': {'AGGFREQ': 'NDAYS   1'},
                        'OUTVAR1': {'OUTVAR': ['OUT_RUNOFF', 'OUT_BASEFLOW', 'OUT_DISCHARGE']}
                        }

   buildGlobalParam(evb_dir_modeling, GlobalParam_dict)

9. **Build RVIC Parameters**
===========================================

Building the complete RVIC parameters requires calling the `rvic` package.  
Use the `buildRVICParam` function to generate RVIC parameters, and then specify the `ROUT_PARAM` path  
in the global parameter file using the `GlobalParamParser` class.

.. code-block:: python

   from easy_vic_build.build_RVIC_Param import buildRVICParam
   from rvic.parameters import parameters as rvic_parameters
   from easy_vic_build.tools.params_func.GlobalParamParser import GlobalParamParser
            
   # build rvic_params
   buildRVICParam(
       evb_dir,
       domain_dataset,
       ppf_kwargs={
           'names': snaped_outlet_names,
           'lons': snaped_outlet_lons,
           'lats': snaped_outlet_lats,
       },
       
       uh_params={
           'createUH_func': createGUH,
           'uh_dt': rvic_uhbox_dt,
           'tp': guh_params['tp']['optimal'][0],
           'mu': guh_params['mu']['optimal'][0],
           'm': guh_params['m']['optimal'][0],
           'plot_bool': True,
           'max_day': None,
           'max_day_range': (0, 10),
           'max_day_converged_threshold': 0.001
       },
       
       cfg_params={
           'VELOCITY': rvic_params['VELOCITY']['optimal'][0],
           'DIFFUSION': rvic_params['DIFFUSION']['optimal'][0],
           'OUTPUT_INTERVAL': rvic_OUTPUT_INTERVAL,
           'SUBSET_DAYS': rvic_SUBSET_DAYS,
           'CELL_FLOWDAYS': None,
           'BASIN_FLOWDAYS': rvic_BASIN_FLOWDAYS,
       }
   )

   globalParam = GlobalParamParser()
   globalParam.load(evb_dir.globalParam_path)
   rout_param_path = os.path.join(
       evb_dir.rout_param_dir, os.listdir(evb_dir.rout_param_dir)[0]
   )
   globalParam.set('Routing', 'ROUT_PARAM', rout_param_path)

10. **Calibrate the Model**
===========================================

You can inherit from the `calibration_base` class to create a subclass  
and override the corresponding class methods to implement different objective functions,  
optimization algorithms, etc.  
This is implemented based on the `DEAP` package.  
You can refer to the example script at `examples/JRB_modeling/JRB_calibrate.py`.


11. **Plot**
===========================================

You can use functions from the `easy_vic_build.tools.plot_func` module  
to conveniently plot model results, basin maps, and other visualizations.

.. note::

   You can refer to the code examples in the ``examples`` directory for more detailed usage.