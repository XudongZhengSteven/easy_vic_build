# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Logging setup utilities used across ``easy_vic_build``.

Public API
----------
``setup_logger``
    Configure and return the module logger.
``logger``
    Preconfigured logger instance created at import time.
"""


import logging
from logging.handlers import RotatingFileHandler

Default_log_format = "%(asctime)s - %(levelname)s - %(message)s"

def setup_logger(log_level=None, log_format=None, log_to_file=None, log_file=None):
    """
    Configure and return the module logger.

    Parameters
    ----------
    log_level : int, optional
        Logger level override.
    log_format : str, optional
        Formatter pattern. Uses ``Default_log_format`` when omitted.
    log_to_file : bool, optional
        If ``True``, also attach a file handler.
    log_file : str, optional
        Log file path used when ``log_to_file=True``.

    Returns
    -------
    logging.Logger
        The configured logger instance.
    """
    # Get the logger instance
    logger = logging.getLogger(__name__)
    
    # If user provides a new log level, update it
    if log_level is not None:
        logger.setLevel(log_level)

    # If user provides a new format, update it
    if log_format is not None:
        formatter = logging.Formatter(log_format)
    else:
        formatter = logging.Formatter(Default_log_format)
    
    # remove handler to avoid repeating handler
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    # Add a console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # If user provides options to log to file, add a file handler
    if log_to_file is not None and log_to_file:
        if log_file is None:
            log_file = "evb.log"  # Default log file name
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

# Default logger setup
print("Initializing logger... ...")
logger = setup_logger(log_level=logging.INFO)

# test
logger.debug("This is a debug message with default setup (should not appear).")
logger.info("This is an info message with the default setup for logger.")
