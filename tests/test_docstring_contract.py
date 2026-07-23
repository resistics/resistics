"""Protect documentation examples and plots that intentionally live in code."""

import ast
import re
from collections.abc import Iterator
from pathlib import Path

import resistics.testing as testing_helpers

PROJECT_ROOT = Path(__file__).parents[1]
MAINTAINED_SOURCE_DIRS = (
    ".github",
    "docs",
    "examples",
    "resistics",
    "scripts",
    "tests",
)
LEGACY_RST_DIRECTIVE = re.compile(r"^\s*\.\.\s+[A-Za-z0-9_-]+::", re.MULTILINE)
MINIMUM_EXAMPLE_DOCSTRINGS = 86
PROTECTED_PLOT_DOCSTRINGS = {
    "decimate.DecimatedData",
    "spectra.FourierTransform",
    "time.BandPass",
    "time.CropTimestamps",
    "time.Decimate",
    "time.HighPass",
    "time.LowPass",
    "time.Notch",
    "time.Resample",
    "time.ShiftTimestamps",
    "time.Subsection",
    "time.Subsamples",
    "window",
    "window.get_win_table",
}
DOCTEST_HELPERS = {
    "decimated_data_linear",
    "decimated_data_periodic",
    "decimated_data_random",
    "record_example1",
    "record_example2",
    "solution_mt",
    "spectra_data_basic",
    "time_data_ones",
    "time_data_periodic",
    "time_data_random",
    "time_data_simple",
    "time_data_with_nans",
    "time_data_with_offset",
    "time_metadata_1chan",
    "time_metadata_2chan",
    "time_metadata_mt",
}
RELOCATED_OR_REMOVED_TEST_HELPERS = {
    "assert_soln_equal",
    "assert_time_data_equal",
    "evaluation_data",
    "generate_evaluation_data",
    "history_example",
    "remove_record_times",
    "solution_general",
    "solution_random_float",
    "solution_random_int",
    "spectra_metadata_multilevel",
    "time_data_linear",
    "transfer_function_random",
}


def _iter_docstrings() -> Iterator[tuple[str, str]]:
    """Yield qualified production object names and their raw docstrings."""
    source_root = PROJECT_ROOT / "resistics"
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module_name = ".".join(path.relative_to(source_root).with_suffix("").parts)
        yield module_name, ast.get_docstring(tree, clean=False) or ""
        yield from _iter_node_docstrings(tree.body, module_name)


def _iter_node_docstrings(
    nodes: list[ast.stmt], prefix: str
) -> Iterator[tuple[str, str]]:
    """Yield docstrings recursively from classes and callables."""
    documentable = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in nodes:
        if not isinstance(node, documentable):
            continue
        name = f"{prefix}.{node.name}"
        yield name, ast.get_docstring(node, clean=False) or ""
        yield from _iter_node_docstrings(node.body, name)


def test_docstring_example_inventory_does_not_shrink() -> None:
    """Keep the established body of API-local examples intact."""
    section_markers = ("Examples\n", "Example\n", "**Examples**\n")
    example_count = sum(
        any(marker in docstring for marker in section_markers)
        for _, docstring in _iter_docstrings()
    )
    assert example_count >= MINIMUM_EXAMPLE_DOCSTRINGS


def test_plot_directives_remain_with_their_documented_objects() -> None:
    """Keep every established plot directive attached to its API object."""
    actual = {
        name for name, docstring in _iter_docstrings() if "```{plot}" in docstring
    }
    assert actual >= PROTECTED_PLOT_DOCSTRINGS


def test_authored_rst_sources_are_absent() -> None:
    """Keep Markdown as the sole authored repository documentation format."""
    rst_paths = list(PROJECT_ROOT.glob("*.rst"))
    for dirname in MAINTAINED_SOURCE_DIRS:
        root = PROJECT_ROOT / dirname
        if root.exists():
            rst_paths.extend(root.rglob("*.rst"))
    assert not rst_paths


def test_legacy_rst_directives_are_absent_from_documentation() -> None:
    """Reject RST directives and compatibility fences in maintained prose."""
    paths = [PROJECT_ROOT / "README.md", PROJECT_ROOT / "CHANGELOG.md"]
    paths.extend((PROJECT_ROOT / "docs" / "source").rglob("*.md"))
    findings = []
    for path in paths:
        content = path.read_text(encoding="utf-8")
        if LEGACY_RST_DIRECTIVE.search(content) or "```{eval-rst}" in content:
            findings.append(path.relative_to(PROJECT_ROOT).as_posix())
    assert not findings


def test_legacy_rst_directives_are_absent_from_docstrings() -> None:
    """Reject legacy directives from production API documentation."""
    findings = [
        name
        for name, docstring in _iter_docstrings()
        if LEGACY_RST_DIRECTIVE.search(docstring) or "```{eval-rst}" in docstring
    ]
    assert not findings


def test_testing_module_retains_doctest_helpers_not_suite_only_factories() -> None:
    """Keep documented builders public and regression-only machinery in tests."""
    assert all(hasattr(testing_helpers, name) for name in DOCTEST_HELPERS)
    assert not any(
        hasattr(testing_helpers, name) for name in RELOCATED_OR_REMOVED_TEST_HELPERS
    )
