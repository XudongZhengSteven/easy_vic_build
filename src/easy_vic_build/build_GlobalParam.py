# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Build VIC ``global_param.txt`` from a reference template.

Public function
---------------
``buildGlobalParam``
    Load the packaged global-parameter reference, fill required EVB paths, and
    apply user-provided overrides.
"""

import os
import re

from . import logger
from .tools.utilities import read_globalParam_reference


def buildGlobalParam(evb_dir, GlobalParam_dict):
    """
    Build and write ``global_param.txt`` for one case.

    The function first sets required default paths from ``evb_dir``:
    ``FORCING1``, ``DOMAIN``, ``PARAMETERS``, ``LOG_DIR``, and ``RESULT_DIR``.
    It then applies values from ``GlobalParam_dict``.

    Section handling:
    - ``FORCE_TYPE``, ``DOMAIN_TYPE``, and ``OUTVAR*`` use
      ``set_section_values`` (replace section content).
    - Other sections are updated key by key with ``set``.

    Parameters
    ----------
    evb_dir : Evb_dir
        Directory/path container for the current case.
    GlobalParam_dict : dict
        Nested mapping of section names to configuration values.

    Returns
    -------
    None
    """
    logger.info("Starting to generate global parameter file... ...")
    # Load parser initialized from the package reference file.
    globalParam = read_globalParam_reference()

    # Set required default paths.
    globalParam.set(
        "Forcing",
        "FORCING1",
        os.path.join(evb_dir.MeteForcing_dir, f"{evb_dir.forcing_prefix}."),
    )
    globalParam.set("Domain", "DOMAIN", evb_dir.domainFile_path)
    globalParam.set("Param", "PARAMETERS", evb_dir.params_dataset_level1_path)
    globalParam.set("Output", "LOG_DIR", evb_dir.VICLog_dir + "/")
    globalParam.set("Output", "RESULT_DIR", evb_dir.VICResults_dir)
    # Override defaults with user-provided values.
    for section_name in GlobalParam_dict.keys():
        if re.match(r"^(FORCE_TYPE|DOMAIN_TYPE|OUTVAR\d*)$", section_name):
            # Replace the full section for list-like sections.
            section_dict = GlobalParam_dict[section_name]
            globalParam.set_section_values(section_name, section_dict)

        else:
            section_dict = GlobalParam_dict[section_name]
            for key, value in section_dict.items():
                globalParam.set(section_name, key, value)

    # Write output file.
    with open(evb_dir.globalParam_path, "w") as f:
        globalParam.write(f)

    logger.info(
        f"Building global parameter file successfully, saved to {evb_dir.globalParam_path}"
    )
