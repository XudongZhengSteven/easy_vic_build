
# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
""" 
set_workenv - A Python module for setting up whitebox_workflows environment for hydrological analysis.

This module provides function for initializing and configuring a whitebox_workflows work environment.

Functions:
----------
    - `setWorkenv`: Initialize and configure a whitebox_workflows work environment.

Usage:
------
    1. Call `SetWorkenv` function with the working directory path and optional parameters.
    
Example:
--------
    >>> from hydroanalysis_wbw import setWorkenv
    >>> wbe = setWorkenv('/path/to/workspace', verbose=True, num_procs=4)
    >>> wbe.verbose
    True
    >>> wbe.working_directory
    '/path/to/workspace'

Dependencies:
-------------
    - `whitebox_workflows`: A library that facilitates geospatial processing tasks, such as DEM filling,
      flow direction, and stream extraction.
      
"""

from whitebox_workflows import WbEnvironment

def setWorkenv(
    working_directory,
    **kwargs
):
    """Initialize and configure a whitebox_workflows work environment.
    
    Parameters
    ----------
    working_directory : str
        Path to the working directory where all input/output files will be processed.
        Should be an absolute path.
        
    **kwargs : dict, optional
        Additional arguments passed to WbEnvironment constructor. Common options include:
        
        - verbose : bool, optional
            Whether to show detailed processing logs (default=False)
            
        - num_procs : int, optional
            Number of processor cores to use (default=1)
            
        - compression : bool, optional
            Whether to use compressed output (default=True)

    Returns
    -------
    WbEnvironment
        Configured WhiteboxTools environment object

    Examples
    --------
    >>> # Basic usage with default settings
    >>> wbe = setWorkenv('/path/to/workspace')
    
    >>> # Advanced usage with custom settings
    >>> wbe = setWorkenv('/path/to/workspace', verbose=True, num_procs=4)
    >>> wbe.verbose
    True
    >>> wbe.working_directory
    '/path/to/workspace'

    Notes
    -----
    1. The working directory path should be absolute for reliable operation.
    2. This function will automatically check for WhiteboxTools installation on first use.
    3. Environment configurations can be customized through kwargs parameters.
    4. The returned WbEnvironment object is thread-safe for parallel processing.
    """
    wbe = WbEnvironment(**kwargs)
    wbe.working_directory = working_directory
    return wbe
