"""
Subpackage: calibrate_func

A Subpackage of easy_vic_build.tools

This subpackage contains a collection of modules that provide functions for calibration, evaluate metrics, and sampling.

Modules:
--------
    - algorithm_NSGAII: Provides functions for implementing the NSGA-II algorithm for multi-objective 
      optimization in model calibration. This includes setting up the optimization problem, 
      evolving populations, and selecting solutions based on Pareto dominance.
      
    - evaluate_metrics: Contains functions to evaluate the performance of the VIC model by 
      comparing simulated outputs with observed data. Various metrics like NSE (Nash-Sutcliffe Efficiency), 
      RMSE (Root Mean Squared Error), and others are provided for model evaluation.
      
    - sampling: Provides functions for generating sample sets used in the calibration process. This 
      may include random sampling, Latin Hypercube Sampling (LHS), and other sampling techniques 
      for exploring the parameter space of the VIC model.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com
    
"""

# Importing submodules for ease of access
from . import algorithm_NSGAII
from . import evaluate_metrics
from . import sampling


# Define the package's public API and version
__all__ = ["algorithm_NSGAII", "evaluate_metrics", "sampling"]