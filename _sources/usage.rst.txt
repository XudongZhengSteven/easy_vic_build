Usage
=====

This page summarizes the HRB workflow using runnable scripts in
``examples/HRB_modeling``. The section order follows the current EVB usage
sequence, while code snippets map directly to the HRB example implementation.

Before running, update local data paths in:

- ``examples/HRB_modeling/general_info.py``
- ``examples/HRB_modeling/HRB_extractData_func/*.py``

General Configuration (``general_info.py``)
-------------------------------------------

The file ``examples/HRB_modeling/general_info.py`` is the shared configuration
entry for the HRB workflow. It is intended to be prepared first and then
imported by all build/calibration scripts to provide required modeling context.

In practice, this file centralizes:

- case directories (``evb_dir_hydroanalysis``, ``evb_dir_modeling``)
- basin/station metadata (station names, coordinates, nested upstream map)
- spatial settings (model scale and level-0/1/2 grid resolutions)
- temporal settings (simulation/warm-up/calibration/verification periods)
- common flags (for example ``reverse_lat``)

After editing ``general_info.py`` for your environment, the remaining HRB
scripts can run without repeatedly redefining the same configuration.

How This Usage Page Is Derived
------------------------------

The workflow and notes below are derived from the executable HRB entry scripts
in ``examples/HRB_modeling`` and their inline comments/docstrings, mainly:

- ``general_info.py``
- ``HRB_build_evb_dir.py``
- ``HRB_hydroanalysis.py``
- ``HRB_build_dpc.py``
- ``HRB_build_domain.py``
- ``HRB_build_Param.py``
- ``HRB_build_MeteForcing.py``
- ``HRB_build_GlobalParam.py``
- ``HRB_build_RVIC_Param.py``
- ``HRB_calibrate.py``

Workflow Overview
-----------------

1. Build case directories with ``Evb_dir``.
2. Prepare basin and grid shapefiles.
3. Build data-processing classes (DPC) for each data level.
4. Build domain.
5. Build parameter datasets.
6. Build hydroanalysis outputs (level 1 / river network).
7. Build meteorological forcing.
8. Build global parameter file.
9. Build RVIC parameters (optional).
10. Calibrate, evaluate, and plot diagnostics/maps.

Step-to-Source Links
--------------------

The following links point to the HRB reference scripts used in this page
(GitHub ``main`` branch):

1. Build modeling directories:
   `HRB_build_evb_dir.py <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_build_evb_dir.py>`_
2. Build basin and grids:
   `HRB_hydroanalysis.py <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_hydroanalysis.py>`_,
   `HRB_build_dpc.py <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_build_dpc.py>`_
3. Build DPC objects:
   `HRB_build_dpc.py <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_build_dpc.py>`_
4. Build domain:
   `HRB_build_domain.py <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_build_domain.py>`_
5. Build parameters:
   `HRB_build_Param.py <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_build_Param.py>`_
6. Build hydroanalysis (level-1/river network):
   `HRB_hydroanalysis.py <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_hydroanalysis.py>`_
7. Build meteorological forcing:
   `HRB_build_MeteForcing.py <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_build_MeteForcing.py>`_
8. Build global parameter file:
   `HRB_build_GlobalParam.py <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_build_GlobalParam.py>`_
9. Build RVIC parameters:
   `HRB_build_RVIC_Param.py <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_build_RVIC_Param.py>`_
10. Calibration and evaluation:
   `HRB_calibrate.py <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_calibrate.py>`_,
   `HRB_plot_results.py <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_plot_results.py>`_,
   `HRB_plot_Basinmap.py <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_plot_Basinmap.py>`_

1. Build Modeling Directory
---------------------------

.. code-block:: python

   from HRB_build_evb_dir import build_modeling_dir

   evb_dir_hydroanalysis = build_modeling_dir(subname="hydroanalysis")
   evb_dir_modeling = build_modeling_dir(subname="shiquan_6km")

Imported symbols and source anchors:

- ``build_modeling_dir``:
  `HRB_build_evb_dir.py#L10 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_build_evb_dir.py#L10>`_

Code notes:

- ``build_modeling_dir`` creates an ``Evb_dir`` under
  ``examples/modeling`` and calls ``Evb_dir.builddir`` with case name
  ``HRB_<subname>``.
- In ``general_info.py``, this function is called twice (inside ``try`` blocks)
  to initialize both hydroanalysis and modeling case directories.
- ``general_info.py`` also attaches a file logger to ``evb_dir_modeling`` and
  logs a summary of spatial/temporal configuration.

2. Build Basin and Grids
------------------------

HRB uses level-0 hydroanalysis outputs to derive basin polygons, then creates
multi-level grids.

.. code-block:: python

   from general_info import (
       evb_dir_hydroanalysis, station_name,
       grid_res_level0, grid_res_level1, grid_res_level2
   )
   from HRB_hydroanalysis import hydroanalysis_level0_HRB
   from HRB_build_dpc import build_basin_shp_JRB
   from easy_vic_build.tools.dpc_func.basin_grid_func import build_grid_shp

   hydroanalysis_level0_HRB(evb_dir_hydroanalysis)

   basin_shps = build_basin_shp_JRB(evb_dir_hydroanalysis)
   grid_shp_level0, grid_shp_level1, grid_shp_level2, grid_shp_level3 = build_grid_shp(
       basin_shps[station_name],
       grid_res_level0,
       grid_res_level1,
       grid_res_level2,
       expand_grids_num=1,
       plot=True,
   )

Imported symbols and source anchors:

- ``evb_dir_hydroanalysis``:
  `general_info.py#L97 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/general_info.py#L97>`_
- ``station_name``:
  `general_info.py#L31 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/general_info.py#L31>`_
- ``grid_res_level0``:
  `general_info.py#L90 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/general_info.py#L90>`_
- ``grid_res_level1``:
  `general_info.py#L91 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/general_info.py#L91>`_
- ``grid_res_level2``:
  `general_info.py#L92 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/general_info.py#L92>`_
- ``hydroanalysis_level0_HRB``:
  `HRB_hydroanalysis.py#L33 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_hydroanalysis.py#L33>`_
- ``build_basin_shp_JRB``:
  `HRB_build_dpc.py#L373 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_build_dpc.py#L373>`_
- ``build_grid_shp``:
  `basin_grid_func.py#L537 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/src/easy_vic_build/tools/dpc_func/basin_grid_func.py#L537>`_

Code notes:

- ``hydroanalysis_level0_HRB`` calls ``buildHydroanalysis_level0`` with the
  clipped ASTGTM2 DEM, Whitebox-based flow direction (``flow_direction_pkg="wbw"``),
  and outlet reference coordinates from ``station_coords_df``.
- ``build_basin_shp_JRB`` reads
  ``basin_vector_outlet_with_reference_<id>.shp`` files from the level-0
  hydroanalysis working directory and loads them as ``Basins`` objects.
- ``build_grid_shp`` returns four grid levels (0/1/2/3) for the target basin
  and is used as the shared spatial basis for DPC, parameter, and forcing steps.

3. Build DPC Objects
--------------------

HRB defines customized DPC subclasses in ``HRB_build_dpc.py``:

- ``dataProcess_VIC_level0_HRB``
- ``dataProcess_VIC_level1_HRB``
- ``dataProcess_VIC_level2_CMFD_HRB``
- ``dataProcess_VIC_level3_HRB``

Use the wrapper function below to build and cache DPC data.

.. code-block:: python

   from general_info import evb_dir_hydroanalysis, evb_dir_modeling, date_period
   from HRB_build_dpc import build_dpc_VIC_HRB

   build_dpc_VIC_HRB(evb_dir_hydroanalysis, evb_dir_modeling, date_period)

Imported symbols and source anchors:

- ``evb_dir_hydroanalysis``:
  `general_info.py#L97 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/general_info.py#L97>`_
- ``evb_dir_modeling``:
  `general_info.py#L104 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/general_info.py#L104>`_
- ``date_period``:
  `general_info.py#L72 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/general_info.py#L72>`_
- ``build_dpc_VIC_HRB``:
  `HRB_build_dpc.py#L389 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_build_dpc.py#L389>`_

Code notes:

- The DPC classes use ``@processing_step`` declarations with explicit
  dependencies to build/cache variables by level.
- Implemented loaders are:

  - Level-0: DEM and SoilGrids attributes.
  - Level-1: soil temperature, annual precipitation, UMD land cover, MODIS
    BSA/NDVI/LAI.
  - Level-2: CMFD meteorological forcing.
  - Level-3: streamflow and gauge metadata.

- In ``build_dpc_VIC_HRB``, each build block is controlled by a local switch
  (``build_dpc_VIC_level0`` / ``build_dpc_VIC_level1`` / ...). In the current
  script, the active block is ``build_dpc_VIC_level3_load_level1 = True``.
- With ``load_level1=True`` in Level-3, ``load_gauge_info`` reads snapped
  outlets from both level-0 and level-1 hydroanalysis directories.

4. Build Domain
---------------

.. code-block:: python

   from general_info import evb_dir_modeling, reverse_lat
   from HRB_build_domain import build_domain_HRB

   build_domain_HRB(evb_dir_modeling, reverse_lat)

Imported symbols and source anchors:

- ``evb_dir_modeling``:
  `general_info.py#L104 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/general_info.py#L104>`_
- ``reverse_lat``:
  `general_info.py#L87 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/general_info.py#L87>`_
- ``build_domain_HRB``:
  `HRB_build_domain.py#L12 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_build_domain.py#L12>`_

Code notes:

- ``build_domain_HRB`` loads level-0/1/3 DPC files, retrieves ``basin_shps``,
  loops over ``station_names``, and writes one domain file per station as
  ``domain_<station>.nc``.
- The function updates ``evb_dir_modeling.domainFile_path`` before each
  ``buildDomain`` call so each station domain is written to a separate target.
- ``add_elev_into_domain`` is an optional helper that reads parameter datasets
  and calls ``addElevIntoDomain``.

5. Build Parameters
-------------------

.. code-block:: python

   from general_info import evb_dir_hydroanalysis, evb_dir_modeling, reverse_lat
   from HRB_build_Param import (
       build_params_HRB,
       build_params_nested_HRB_basin_hierarchy,
       build_params_HRB_spatially_uniform,
   )

   # Option A: default HRB parameter build
   build_params_HRB(evb_dir_modeling, reverse_lat)

   # Option B: nested-basin hierarchy parameterization
   build_params_nested_HRB_basin_hierarchy(
       evb_dir_hydroanalysis, evb_dir_modeling, reverse_lat
   )

   # Option C: spatially uniform baseflow scheme
   build_params_HRB_spatially_uniform(
       evb_dir_modeling, reverse_lat, baseflow_scheme="Nijssen"
   )

Imported symbols and source anchors:

- ``evb_dir_hydroanalysis``:
  `general_info.py#L97 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/general_info.py#L97>`_
- ``evb_dir_modeling``:
  `general_info.py#L104 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/general_info.py#L104>`_
- ``reverse_lat``:
  `general_info.py#L87 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/general_info.py#L87>`_
- ``build_params_HRB``:
  `HRB_build_Param.py#L104 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_build_Param.py#L104>`_
- ``build_params_nested_HRB_basin_hierarchy``:
  `HRB_build_Param.py#L180 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_build_Param.py#L180>`_
- ``build_params_HRB_spatially_uniform``:
  `HRB_build_Param.py#L321 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_build_Param.py#L321>`_

Code notes:

- ``build_params_HRB`` performs the standard three-stage flow:
  ``buildParam_level0 -> buildParam_level1 -> scaling_level0_to_level1``.
- In this HRB implementation, custom interfaces extend the defaults to set
  topography-related fields (for example elevation/slope/``ele_std``) and
  ``annual_prec`` from ``annual_P_CMFD_mm``.
- The scaling call uses ``nlayer_list=[1, 2, 3]`` and
  ``elev_scaling="Arithmetic_min"``.
- ``build_params_nested_HRB_basin_hierarchy`` computes unique nested-basin
  masks (level-1), remaps them to level-0, and passes
  ``basin_hierarchy`` into level-0 parameter construction.
- ``build_params_HRB_spatially_uniform`` switches to spatially uniform
  baseflow parameter sets and interface classes (``ARNO`` or ``Nijssen``).

6. Build Hydroanalysis
----------------------

After parameters/domain are available, run level-1 hydroanalysis and (optional)
river-network graph construction.

.. code-block:: python

   from general_info import evb_dir_modeling, reverse_lat
   from HRB_hydroanalysis import hydroanalysis_level1_HRB, buildRivernetwork_level1_HRB

   hydroanalysis_level1_HRB(evb_dir_modeling, reverse_lat)
   buildRivernetwork_level1_HRB(evb_dir_modeling, threshold=2)

Imported symbols and source anchors:

- ``evb_dir_modeling``:
  `general_info.py#L104 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/general_info.py#L104>`_
- ``reverse_lat``:
  `general_info.py#L87 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/general_info.py#L87>`_
- ``hydroanalysis_level1_HRB``:
  `HRB_hydroanalysis.py#L74 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_hydroanalysis.py#L74>`_
- ``buildRivernetwork_level1_HRB``:
  `HRB_hydroanalysis.py#L143 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_hydroanalysis.py#L143>`_

Code notes:

- ``hydroanalysis_level1_HRB`` reads gauge information from Level-3 DPC,
  loads params/domain, and calls ``buildHydroanalysis_level1`` with Whitebox
  flow direction and ``stream_acc_threshold=2``.
- ``buildRivernetwork_level1_HRB`` maps outlet coordinates to domain cell
  nodes, passes them as ``labeled_nodes``, and writes:

  - ``river_network_graph.pkl``
  - ``river_network_graph_full.pkl``
  - ``river_network_graph_connected.pkl``
  - ``fig_river_network*.tiff`` figures

7. Build Meteorological Forcing
-------------------------------

.. code-block:: python

   from general_info import evb_dir_modeling
   from HRB_build_MeteForcing import HRB_build_MeteForcing

   HRB_build_MeteForcing(evb_dir_modeling)

Imported symbols and source anchors:

- ``evb_dir_modeling``:
  `general_info.py#L104 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/general_info.py#L104>`_
- ``HRB_build_MeteForcing``:
  `HRB_build_MeteForcing.py#L11 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_build_MeteForcing.py#L11>`_

Code notes:

- ``HRB_build_MeteForcing`` loads the CMFD Level-2 DPC cache and runs
  ``merge_grid_data()`` before forcing construction.
- It then calls ``build_MeteForcing.buildMeteForcing`` with
  ``buildMeteForcing_interface``, ``file_format="NETCDF4"``, and the
  ``timestep`` defined in ``general_info.py``.

8. Build Global Parameter File
------------------------------

.. code-block:: python

   from general_info import evb_dir_modeling
   from HRB_build_GlobalParam import HRB_build_GlobalParam

   HRB_build_GlobalParam(evb_dir_modeling)

Imported symbols and source anchors:

- ``evb_dir_modeling``:
  `general_info.py#L104 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/general_info.py#L104>`_
- ``HRB_build_GlobalParam``:
  `HRB_build_GlobalParam.py#L7 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_build_GlobalParam.py#L7>`_

Code notes:

- ``HRB_build_GlobalParam`` defines ``GlobalParam_dict`` inline and writes
  ``global_param.txt`` via ``buildGlobalParam``.
- The dictionary sets simulation start/end from ``date_period``, output
  aggregation, output variables (``OUT_RUNOFF``, ``OUT_BASEFLOW``,
  ``OUT_DISCHARGE``), and baseflow scheme (``NIJSSEN2001`` in the script).

9. Build RVIC Parameters (Optional)
-----------------------------------

.. code-block:: python

   from general_info import evb_dir_modeling
   from HRB_build_RVIC_Param import HRB_build_RVIC_Param

   HRB_build_RVIC_Param(evb_dir_modeling)

Imported symbols and source anchors:

- ``evb_dir_modeling``:
  `general_info.py#L104 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/general_info.py#L104>`_
- ``HRB_build_RVIC_Param``:
  `HRB_build_RVIC_Param.py#L12 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_build_RVIC_Param.py#L12>`_

Code notes:

- ``HRB_build_RVIC_Param`` reads domain data and Level-3 ``gauge_info`` to
  get level-1 outlet coordinates.
- It calls ``buildRVICParam_basic`` with:

  - ``ppf_kwargs`` for outlet names/coordinates
  - ``uh_params`` using ``createGUH`` and default GUH parameters
  - ``cfg_params`` (for example ``VELOCITY``, ``DIFFUSION``, and timing options)

- The generated files are the RVIC parameter/configuration inputs used by
  downstream routing runs.

If RVIC is used, update ``ROUT_PARAM`` in ``global_param.txt`` accordingly.

10. Calibration and Evaluation
------------------------------

.. code-block:: python

   from general_info import evb_dir_modeling
   from HRB_calibrate import calibrate_HRB

   calibrate_HRB(evb_dir_modeling)

For diagnostics/figures, see ``HRB_plot_results.py`` and
``HRB_plot_Basinmap.py``.

Imported symbols and source anchors:

- ``evb_dir_modeling``:
  `general_info.py#L104 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/general_info.py#L104>`_
- ``calibrate_HRB``:
  `HRB_calibrate.py#L1494 <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_calibrate.py#L1494>`_
- Related post-processing scripts:
  `HRB_plot_results.py <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_plot_results.py>`_,
  `HRB_plot_Basinmap.py <https://github.com/XudongZhengSteven/easy_vic_build/blob/main/examples/HRB_modeling/HRB_plot_Basinmap.py>`_

Code notes:

- ``calibrate_HRB`` sets the VIC executable path, algorithm hyperparameters,
  and multiple predefined calibration case templates.
- In the current source, Case 3 is active by default: distributed VIC
  parameters, uniform soil depths, and fixed RVIC/GUH parameters; other case
  templates are present but commented out.
- The function can instantiate either ``NSGAII_VIC_HRB`` or ``CMA_ES_VIC_HRB``
  based on ``cali_method`` and uses shared helpers for parameter building,
  VIC/RVIC simulation, evaluation, and checkpoint/result management.
- ``HRB_plot_results.py`` and ``HRB_plot_Basinmap.py`` are the post-processing
  scripts for metrics, figures, and basin maps.

Script Entry Points
-------------------

You can also run the workflow scripts directly (from repository root):

.. code-block:: bash

   python examples/HRB_modeling/HRB_hydroanalysis.py
   python examples/HRB_modeling/HRB_build_dpc.py
   python examples/HRB_modeling/HRB_build_domain.py
   python examples/HRB_modeling/HRB_build_Param.py
   python examples/HRB_modeling/HRB_build_MeteForcing.py
   python examples/HRB_modeling/HRB_build_GlobalParam.py
   python examples/HRB_modeling/HRB_build_RVIC_Param.py
   python examples/HRB_modeling/HRB_calibrate.py
   python examples/HRB_modeling/HRB_plot_results.py

Code notes:

- Running scripts in this order matches data dependencies in source code.
- Most build scripts read cached DPC/domain/parameter files, so you can rerun
  a single stage without rebuilding every previous stage.

See also
--------

- :doc:`installation`
- :doc:`api`
- :doc:`notes`
