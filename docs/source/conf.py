from datetime import datetime
import os
import sys
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# logo
html_logo = "_static/logo.png"
html_theme_options = {
    "logo_only": True,
    "display_version": False,
}

# Add project root (src) to Python path
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'easy_vic_build'
copyright = f'{datetime.now().year}, XudongZheng. Licensed under the MIT License'
author = 'XudongZheng'
release = '0.2.0'
version = '0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'sphinx.ext.autosummary',
    # 'sphinx_multiversion',
]

templates_path = ['_templates']
exclude_patterns = []
autosummary_generate = True
autosummary_generate_overwrite = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
}
autodoc_member_order = "bysource"
autoclass_content = "both"
add_module_names = True

# Optional dependencies commonly unavailable in doc build environments
autodoc_mock_imports = [
    "nco",
    "rvic",
    "osgeo",
    "geopandas",
    "rasterio",
    "cartopy",
    "whitebox_workflows",
    "regionmask",
    "netCDF4",
    "cftime",
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# Source file extensions
source_suffix = '.rst'

# Master document
master_doc = 'index'

# sphinx-multiversion settings
smv_tag_whitelist = r'^v\d+\.\d+\.\d+$'
smv_branch_whitelist = r'^main$'
smv_remote_whitelist = r'^origin$'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'  # alabaster

# GitHub link
html_context = {
    'display_github': True,
    'github_user': 'XudongZhengSteven',
    'github_repo': 'easy_vic_build',
    'github_version': 'main',
    'conf_py_path': '/docs/source/',
}

html_static_path = ['_static']

html_css_files = [
    'custom.css',
]
