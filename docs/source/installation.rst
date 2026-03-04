Installation
============

Requirements
------------

EVB relies on common scientific Python packages and optional geospatial/routing
dependencies depending on your workflow.

Core packages include ``numpy``, ``pandas``, ``scipy``, ``matplotlib``,
``netCDF4``, and ``tqdm``.

Optional packages include ``nco`` integration, ``rvic``, and geospatial
libraries such as ``geopandas``, ``rasterio``, and ``whitebox_workflows``.

Install from PyPI
-----------------

.. code-block:: bash

   pip install easy_vic_build

Optional extras:

.. code-block:: bash

   pip install easy_vic_build[nco]
   pip install easy_vic_build[rvic]
   pip install easy_vic_build[nco_rvic]

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/XudongZhengSteven/easy_vic_build
   cd easy_vic_build
   pip install -e .

Verify installation
-------------------

.. code-block:: bash

   python -c "import easy_vic_build as evb; print(evb.__version__)"

Notes
-----

- Some geospatial capabilities require system libraries (GDAL/PROJ).
- RVIC-related workflows require a valid ``rvic`` installation.
