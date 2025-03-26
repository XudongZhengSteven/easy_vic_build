# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Logger - A Python module for configuring and managing logging.

This module provides a flexible and user-friendly interface to configure logger
for Python applications. It allows users to dynamically set the logging level,
format, and output destination (console or file). The module is designed to be
easy to use while providing advanced customization options.

Functions
---------
    - `setup_logger`: Configure the logger with user-defined settings.

Usage:
------
    1. Call the `setup_logger` to get logger instance.
    2. Use the `logger` method (logger.info('...'), logger.debug('...')).

Example
-------
To use the default logger configuration:
    >>> from Logger import logger
    >>> logger.info("This is an info message with the default setup.")

To customize the logger configuration:
    >>> from Logger import setup_logger
    >>> setup_logger(
    ...     log_level=logging.DEBUG,
    ...     log_format="%(asctime)s - %(levelname)s - %(message)s",
    ...     log_to_file=True,
    ...     log_file="custom_log.log"
    ... )
    >>> logger.debug("This is a debug message with custom configuration.")

Notes
-----
- The default log format is "%(asctime)s - %(levelname)s - %(message)s".
- If no parameters are passed to `setup_logger`, the default configuration is used.

Dependencies
------------
    - `logging` : Python standard library, Provides the core logging functionality.

"""


import logging

Default_log_format = "%(asctime)s - %(levelname)s - %(message)s"

def setup_logger(log_level=None, log_format=None, log_to_file=None, log_file=None):
    """
    Configure the logger with user-defined settings.

    This function allows users to dynamically modify the logger's configuration,
    including the logging level, format, and output destination (console or file).
    If no parameters are provided, the default configuration is used.

    Parameters:
    -----------
    log_level : int, optional
        The logging level to set. Default is None (no change).

    log_format : str, optional
        The log format to set. Default is None (no change).

    log_to_file : bool, optional
        Whether to log to a file. Default is None (no change).

    log_file : str, optional
        If logging to a file, specify the file path. Default is None (no change).

    Returns:
    --------
    Logging.Logger
        The configured logger instance.

    Example:
    --------
    To customize the logger configuration:
        >>> setup_logger(
        ...     log_level=logging.DEBUG,
        ...     log_format="%(asctime)s - %(levelname)s - %(message)s",
        ...     log_to_file=True,
        ...     log_file="custom_log.log"
        ... )
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
