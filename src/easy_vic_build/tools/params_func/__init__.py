"""
Subpackage: params_func

A Subpackage of easy_vic_build.tools

This subpackage contains a collection of modules for handling parameters in the
context of the VIC hydrological model. It includes functionalities for parameter
generation, scaling, transfer functions, and vegetation type attributes.

Modules:
--------
    - createParametersDataset:
        Provides functionality to create a dataset containing parameters for VIC model
        simulation.
    - GlobalParamParser:
        A module that parses global parameter files for the VIC model, making it easier
        to configure the model's input data.
    - params_set:
        Contains functions to set and manage different parameter configurations for the
        VIC model.
    - Scaling_operator:
        Implements scaling operations for parameters, allowing for adjustments based on
        spatial or temporal scales.
    - TransferFunction:
        Defines transfer functions used in the VIC model to relate input parameters to
        outputs in a hydrological context.
    - veg_type_attributes_umd_prepare:
        Prepares vegetation type attribute data from UMD (University of Maryland) datasets
        for integration into the VIC model.


Author:
-------
    Xudong Zheng
    Email: z786909151@163.com

"""

# Importing submodules for ease of access
from . import (GlobalParamParser, Scaling_operator, TransferFunction,
               createParametersDataset, params_set,
               veg_type_attributes_umd_prepare)

# Define the package's public API and version
__all__ = [
    "createParametersDataset",
    "GlobalParamParser",
    "params_set",
    "Scaling_operator",
    "TransferFunction",
    "veg_type_attributes_umd_prepare",
]
