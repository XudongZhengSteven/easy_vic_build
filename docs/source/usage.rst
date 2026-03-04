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

1. Build Modeling Directory
---------------------------

.. code-block:: python

   from HRB_build_evb_dir import build_modeling_dir

   evb_dir_hydroanalysis = build_modeling_dir(subname="hydroanalysis")
   evb_dir_modeling = build_modeling_dir(subname="shiquan_6km")

In HRB examples, these directories are also created automatically when importing
``general_info``.

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

4. Build Domain
---------------

.. code-block:: python

   from general_info import evb_dir_modeling, reverse_lat
   from HRB_build_domain import build_domain_HRB

   build_domain_HRB(evb_dir_modeling, reverse_lat)

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

6. Build Hydroanalysis
----------------------

After parameters/domain are available, run level-1 hydroanalysis and (optional)
river-network graph construction.

.. code-block:: python

   from general_info import evb_dir_modeling, reverse_lat
   from HRB_hydroanalysis import hydroanalysis_level1_HRB, buildRivernetwork_level1_HRB

   hydroanalysis_level1_HRB(evb_dir_modeling, reverse_lat)
   buildRivernetwork_level1_HRB(evb_dir_modeling, threshold=2)

7. Build Meteorological Forcing
-------------------------------

.. code-block:: python

   from general_info import evb_dir_modeling
   from HRB_build_MeteForcing import HRB_build_MeteForcing

   HRB_build_MeteForcing(evb_dir_modeling)

8. Build Global Parameter File
------------------------------

.. code-block:: python

   from general_info import evb_dir_modeling
   from HRB_build_GlobalParam import HRB_build_GlobalParam

   HRB_build_GlobalParam(evb_dir_modeling)

9. Build RVIC Parameters (Optional)
-----------------------------------

.. code-block:: python

   from general_info import evb_dir_modeling
   from HRB_build_RVIC_Param import HRB_build_RVIC_Param

   HRB_build_RVIC_Param(evb_dir_modeling)

If RVIC is used, update ``ROUT_PARAM`` in ``global_param.txt`` accordingly.

10. Calibration and Evaluation
------------------------------

.. code-block:: python

   from general_info import evb_dir_modeling
   from HRB_calibrate import calibrate_HRB

   calibrate_HRB(evb_dir_modeling)

For diagnostics/figures, see ``HRB_plot_results.py`` and
``HRB_plot_Basinmap.py``.

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

See also
--------

- :doc:`installation`
- :doc:`api`
- :doc:`notes`
