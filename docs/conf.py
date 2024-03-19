# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'PET Processing Module'
copyright = '2024, Furqan Dar, Bradley Judge, Noah Goldman'
author = 'Furqan Dar, Bradley Judge, Noah Goldman'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [ "sphinx.ext.autodoc",
               "sphinx.ext.autosummary",
               "sphinx.ext.intersphinx",
               "sphinx.ext.napoleon",
               "sphinx.ext.todo",
               "sphinx.ext.mathjax"]

add_module_names = False
autoclass_content = 'both'
todo_include_todos = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = True
napoleon_use_ivar=True
autodoc_member_order = 'bysource'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'English (US)'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'python_docs_theme'
# html_theme = 'sphinx_rtd_theme'
# html_theme = 'alabaster'
html_static_path = ['_static']

autosummary_generate = True
autodoc_docstring_signature = True