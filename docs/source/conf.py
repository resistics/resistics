# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from pathlib import Path
import os
import re
import resistics
import plotly.io as pio
from plotly.io._sg_scraper import plotly_sg_scraper
from sphinx_gallery.sorting import FileNameSortKey

project = "resistics"
copyright = "2019, Neeraj Shah"
author = "Neeraj Shah"
release = resistics.__version__


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx.ext.autosectionlabel",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_gallery.gen_gallery",
    "sphinxext.opengraph",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["setup.rst", "modules.rst"]

# resistics configuration
# code styles
pygments_style = "gruvbox-light"
pygments_dark_style = "gruvbox-dark"
# autodoc
autosectionlabel_prefix_document = True
autodoc_member_order = "bysource"
autodoc_undoc_members = False
# napoleon extension
napoleon_numpy_docstring = True
napoleon_attr_annotations = False
# other configuration
plot_include_source = True
todo_include_todos = True
# intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}
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
# sphinx gallery
pio.renderers.default = "sphinx_gallery_png"
image_scrapers = ("matplotlib", plotly_sg_scraper)
sphinx_gallery_conf = {
    "run_stale_examples": False,
    "filename_pattern": f"{re.escape(os.sep)}eg_",
    "remove_config_comments": True,
    "thumbnail_size": (300, 300),
    "examples_dirs": [
        "../../examples/read",
        "../../examples/quick",
        "../../examples/project",
        "../../examples/config",
    ],
    "gallery_dirs": [
        "tutorial-read",
        "tutorial-quick",
        "tutorial-project",
        "tutorial-config",
    ],
    "image_scrapers": image_scrapers,
    "within_subsection_order": FileNameSortKey,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]
html_theme = "furo"
html_logo = str(Path("_static", "images", "logo.png"))
html_favicon = str(Path("_static", "images", "favicon.png"))
html_theme_options = {
    "navigation_with_keys": True,
}
