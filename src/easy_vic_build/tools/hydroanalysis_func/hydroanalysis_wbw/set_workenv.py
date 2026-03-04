
# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
""" 
Create and configure a WhiteboxTools workflow environment.
"""

from whitebox_workflows import WbEnvironment

def setWorkenv(
    working_directory,
    **kwargs
):
    """Create a ``WbEnvironment`` instance and set its working directory.

    Parameters
    ----------
    working_directory : str
        Working directory used by WhiteboxTools for input and output files.

    **kwargs : dict, optional
        Additional keyword arguments passed to ``WbEnvironment``.

    Returns
    -------
    WbEnvironment
        Initialized WhiteboxTools workflow environment.
    """
    wbe = WbEnvironment(**kwargs)
    wbe.working_directory = working_directory
    return wbe
