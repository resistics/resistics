"""Protect documentation examples and plots that intentionally live in code."""

import ast
from collections.abc import Iterator
from pathlib import Path

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


def _iter_docstrings() -> Iterator[tuple[str, str]]:
    """Yield qualified production object names and their raw docstrings."""
    source_root = Path(__file__).parents[1] / "resistics"
    for path in sorted(source_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module_name = path.stem
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
    example_count = sum(
        "Examples\n" in docstring or "Example\n" in docstring
        for _, docstring in _iter_docstrings()
    )
    assert example_count >= MINIMUM_EXAMPLE_DOCSTRINGS


def test_plot_directives_remain_with_their_documented_objects() -> None:
    """Keep every established plot directive attached to its API object."""
    actual = {
        name for name, docstring in _iter_docstrings() if ".. plot::" in docstring
    }
    assert actual >= PROTECTED_PLOT_DOCSTRINGS
