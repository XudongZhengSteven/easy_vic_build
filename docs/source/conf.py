from datetime import datetime
import os
import sys
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Add project to Python path
sys.path.insert(0, os.path.abspath('../../src/easy_vic_build/'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'easy_vic_build'
copyright = f'{datetime.now().year}, XudongZheng. Licensed under the MIT Lincense'
author = 'XudongZheng'
release = '0.1.0'
version = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Extensions
extensions = [
    'sphinx.ext.autodoc',       # Auto-generate API docs
    'sphinx.ext.viewcode',      # Add links to source code
    'sphinx.ext.napoleon',      # Support for Google-style docstrings
    'sphinx.ext.githubpages',   # Publish to GitHub Pages
    'sphinx.ext.autosummary',   # Generate summaries
    'sphinx_multiversion',
]

templates_path = ['_templates']
exclude_patterns = []
autosummary_generate = True

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

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Source file extensions
source_suffix = '.rst'

# Master document
master_doc = 'index'

# sphinx-multiversion settings
smv_tag_whitelist = r'^v\d+\.\d+\.\d+$'
smv_branch_whitelist = r'^main$'
smv_remote_whitelist = r'^origin$'
