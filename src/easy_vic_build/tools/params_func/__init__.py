"""Parameter-generation and transformation modules.

This subpackage provides parsers, dataset builders, scaling operators, transfer
functions, and parameter presets for VIC-related workflows.
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
