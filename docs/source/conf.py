# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
from pathlib import Path
import resistics


# -- Project information -----------------------------------------------------

project = "resistics"
copyright = "2019, Neeraj Shah"
author = "Neeraj Shah"

# The full version, including alpha/beta/rc tags
release = resistics.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx.ext.autosectionlabel",
    "matplotlib.sphinxext.plot_directive",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["setup.rst", "modules.rst"]

# resistics configuration
autosectionlabel_prefix_document = True
autodoc_member_order = "bysource"
autodoc_undoc_members = False
# napoleon extension
napoleon_numpy_docstring = True
napoleon_attr_annotations = False
# other configuration
plot_include_source = True
todo_include_todos = True
# copy button
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: "
copybutton_prompt_is_regexp = True
# pydantic configuration
autodoc_pydantic_model_member_order = "bysource"
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_config_member = False
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_hide_paramlist = True
autodoc_pydantic_field_show_default = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_logo = str(Path("_static", "images", "logo.png"))
html_favicon = str(Path("_static", "images", "favicon.png"))

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
