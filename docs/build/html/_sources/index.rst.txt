.. easy_vic_build documentation master file, created by
   sphinx-quickstart on Sun Mar  9 22:51:55 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Easy VIC Build (EVB) documentation!
==========================================

**Easy VIC Build (EVB)** is a Python-based framework designed to streamline the deployment of the VIC model, while remaining flexible, extensible, and aligned with ongoing VIC development.

Easy VIC Build (EVB)
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api
   contributing
   license
   contact
   notes
   references
   citation

Features
--------

The proposed framework, **Easy VIC Build (EVB)**, is designed with the following features.

Full Python Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~
Leverages the extensive Python ecosystem to enable on-demand scalability.

Loosely Coupled, Object-Oriented Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Preserves the independence of individual components so they can be replaced or modified without affecting the overall architecture.  
This flexibility allows adjustments to:

- **Model structures** (e.g., soil layer depths, number of root zones)
- **Data sources and processing workflows** (e.g., forcing and land surface data)
- **Spatiotemporal scales** — to align with specific research objectives

Alignment with VIC Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Supports the processing and generation of widely used data formats, including NetCDF and TIFF.

Advanced Methods
~~~~~~~~~~~~~~~~
To enhance its applicability, EVB tentatively integrates several advanced methods associated with model deployment, including:

- **Multiscale Parameter Regionalization (MPR)**
- **General Unit Hydrograph (GUH)**

[Samaniego2010]_, [Mizukami2017]_, [Guo2022]_

Applications
~~~~~~~~~~~~
The EVB framework has been applied in multiple real-world basin systems to rigorously evaluate its performance.

Vision
~~~~~~
We anticipate that EVB will serve a broad user community—both current and prospective VIC users—by providing a more streamlined and accessible pathway for VIC model deployment, thereby strengthening its utility in both applications and research.

.. note::

   This project is under active development.