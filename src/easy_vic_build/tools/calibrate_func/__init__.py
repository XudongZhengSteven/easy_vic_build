"""Calibration routines, metrics, and sampling strategies.

Modules
-------
algorithm_CMA_ES
    CMA-ES based calibration routines.
algorithm_NSGAII
    NSGA-II based multi-objective calibration routines.
evaluate_metrics
    Simulation-performance metrics.
sampling
    Parameter sampling utilities.
"""

# Importing submodules for ease of access
from . import algorithm_CMA_ES, algorithm_NSGAII, evaluate_metrics, sampling

# Define the package's public API and version
__all__ = ["algorithm_NSGAII", "algorithm_CMA_ES", "evaluate_metrics", "sampling"]
