Installation
============

Requirements
------------

.. csv-table:: Table 1. A summary of the EVB framework environmental requirements
   :header: "Category", "Package Name", "Functionality", "Reference", "Source"
   :widths: 20, 15, 35, 15, 40

   "Basic computation and visualization", "NumPy", "Arrays and operations", "Harris et al. (2020)", "https://numpy.org/"
   "", "pandas", "Data frames and operations", "Mckinney (2011)", "https://pandas.pydata.org/"
   "", "Matplotlib", "Plotting", "Hunter (2007)", "https://matplotlib.org/"
   "", "SciPy", "Scientific computing", "Virtanen et al. (2020)", "https://github.com/scipy/scipy"
   "", "tqdm", "Progress bar", "Da Costa-Luis (2019)", "https://github.com/tqdm/tqdm"
   "Geospatial data processing", "GeoPandas", "Geospatial data processing and geometric operations", "Jordahl et al. (2020)", "https://geopandas.org/"
   "", "netCDF4", "Processing NetCDF format data", "Rew and Davis (1990)", "https://github.com/Unidata/netcdf4-python"
   "", "Xarray", "Processing multidimensional arrays and NetCDF format data", "Hoyer and Hamman (2017)", "https://github.com/pydata/xarray"
   "", "Rasterio", "Processing gridded raster datasets", "Gillies (2013)", "https://github.com/rasterio/rasterio"
   "", "Cartopy", "Spatial plotting and spatial reference", "Office (2010-2015)", "https://github.com/SciTools/cartopy"
   "", "Whitebox-Workflows", "Hydrological analysis", "Lindsay and Wu (2021)", "https://www.whiteboxgeo.com/whitebox-workflows-for-python/"
   "Optimization", "DEAP", "Evolutionary computation framework", "Fortin et al. (2012)", "https://github.com/deap/deap"
   "", "ArcPy", "Hydrological analysis", "Esri (2024)", "https://pro.arcgis.com/"


To install the package, run one of the following commands:

Basic installation
-------------------

.. code-block:: bash

   pip install easy_vic_build

For additional dependencies
----------------------------

.. code-block:: bash

   pip install easy_vic_build[nco]
   pip install easy_vic_build[rvic]
   pip install easy_vic_build[nco_rvic]

Alternatively, you can install from a `.whl` file:

.. code-block:: bash

   pip install .whl
   pip install .whl[nco]
   pip install .whl[rvic]
   pip install .whl[nco_rvic]

For development purposes
-------------------------

.. code-block:: bash

   git clone https://github.com/XudongZhengSteven/easy_vic_build
   cd easy_vic_build
   pip install -e .