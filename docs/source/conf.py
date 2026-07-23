# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from pathlib import Path
import sys
import resistics
from matplotlib.sphinxext.plot_directive import PlotDirective
from sphinx.application import Sphinx

sys.path.insert(0, str(Path(__file__).parents[1]))

project = "resistics"
copyright = "2019, Neeraj Shah"
author = "Neeraj Shah"
release = resistics.__version__


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx.ext.autosectionlabel",
    "matplotlib.sphinxext.plot_directive",
    "sphinxext.opengraph",
    "myst_nb",
    "_ext.myst_autodoc",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# resistics configuration
# code styles
pygments_style = "gruvbox-light"
pygments_dark_style = "gruvbox-dark"
# autodoc
autosectionlabel_prefix_document = True
autodoc_member_order = "bysource"
autodoc_undoc_members = False
# Cross-reference warnings are fatal in the maintained documentation command.
nitpicky = True
# Autodoc turns annotations and inherited bases into references. These exact
# patterns cover imported aliases, private implementation types, and third-party
# types for which no usable Sphinx inventory exists. Authored Resistics and
# intersphinx references remain subject to nitpicky checking.
nitpick_ignore_regex = [
    (
        "py:class",
        r"(?:BindingType|ConfigDict|DataSource|DataType|Field|FieldInfo|Ge|"
        r"JobDefinition \| None|JsonValue|ModelT|NoneType|Path|RunGroup|"
        r"StationGroup|SurveyGroup|VisualType|_MTH5Handle|_TuiLogBuffer|"
        r"datetime|time)",
    ),
    ("py:class", r"dict\[(?:float|str)"),
    ("py:obj", r"typing\.Literal\['parameters'"),
    ("py:class", r"annotated_types\.(?:Ge|Gt)"),
    (
        "py:class",
        r"attotime\.objects\.(?:attodatetime\.attodatetime|"
        r"attotimedelta\.attotimedelta)",
    ),
    ("py:class", r"(?:go\.Figure|plotly\.graph_objs\._figure\.Figure)"),
    (
        "py:class",
        r"(?:pydantic\.(?:config\.ConfigDict|main\.BaseModel|types\.JsonValue)|"
        r"pydantic_core\.core_schema\.ValidationInfo)",
    ),
    (
        "py:class",
        r"(?:pd\.DataFrame|xarray\.core\.(?:dataarray\.DataArray|dataset\.Dataset))",
    ),
    (
        "py:class",
        r"textual\.(?:app\.App|screen\.(?:ModalScreen|Screen)|widget\.Widget|"
        r"widgets\.(?:_directory_tree\.DirectoryTree\.(?:DirectorySelected|"
        r"FileSelected)|_static\.Static))",
    ),
    (
        "py:class",
        r"resistics\.(?:project\._MTH5InspectionMixin|"
        r"project_mth5\._MTH5Handle|regression\._FittableRegressor|"
        r"tui\.(?:logging\._Tui(?:DiagnosticCapture|LogBuffer)|"
        r"screens\.project_(?:data\._ProjectDataMixin|jobs\._ProjectJobsMixin|"
        r"logs\._ProjectLogsMixin|resources\._ProjectResourcesMixin)|"
        r"state\.TimePlotSelection))",
    ),
]
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
myst_enable_extensions = ["fieldlist"]
myst_ref_domains = ["std", "py"]
nb_execution_mode = "force"
nb_execution_timeout = 120
nb_execution_raise_on_error = True
linkcheck_timeout = 10
linkcheck_retries = 2
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]
html_theme = "furo"
html_logo = str(Path("_static", "images", "logo.png"))
html_favicon = str(Path("_static", "images", "favicon.png"))
html_theme_options = {
    "navigation_with_keys": True,
}


class _DocstringPlotDirective(PlotDirective):
    """Keep generated autodoc plots relative to the active documentation page."""

    def run(self):
        """Run the plot directive with a Sphinx-source path for Python docstrings."""
        document = self.state_machine.document
        source = document["source"]
        if Path(source).resolve().is_relative_to(Path(__file__).parent.resolve()):
            return super().run()

        env = document.settings.env
        document["source"] = env.doc2path(env.docname, base=True)
        try:
            return super().run()
        finally:
            document["source"] = source


def setup(app: Sphinx) -> dict[str, bool]:
    """Register plot handling that supports plots embedded in API docstrings."""
    app.add_directive("plot", _DocstringPlotDirective, override=True)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
