#!/usr/bin/env python3
"""Create a repeatable code-hardening baseline report.

The report measures source size, docstring coverage, embedded examples and
plots, and fresh-process import time without importing the package in the
reporting process. Generated JSON is written below ``.artifacts/`` by default.

Examples
--------
Run the standard baseline from the repository root::

    .venv/bin/python scripts/hardening_report.py

Skip import timing when only the static inventory is required::

    .venv/bin/python scripts/hardening_report.py --import-samples 0
"""

from __future__ import annotations

import argparse
import ast
import json
import platform
import re
import statistics
import subprocess  # noqa: S404 - the command is fixed and contains no user input
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

DEFAULT_OUTPUT = Path(".artifacts/hardening/baseline/codebase.json")
PRIVATE_OPERATION_MIN_LINES = 20
DOCTEST_DIRECTIVE = re.compile(r"(?m)^\s*\.\.\s+doctest::")
MYST_DOCTEST_DIRECTIVE = re.compile(r"(?m)^\s*(?:```|~~~)?\{doctest\}")
PLOT_DIRECTIVE = re.compile(r"(?m)^\s*\.\.\s+plot::")
MYST_PLOT_DIRECTIVE = re.compile(r"(?m)^\s*(?:```|~~~)?\{plot\}")


def _python_files(root: Path, relative_directory: str) -> list[Path]:
    """Return sorted Python files below a repository-relative directory."""
    return sorted((root / relative_directory).rglob("*.py"))


def _physical_lines(path: Path) -> int:
    """Count physical lines in a UTF-8 Python source file."""
    return len(path.read_text(encoding="utf-8").splitlines())


def _module_name(root: Path, path: Path) -> str:
    """Return a dotted module name for a source path below *root*."""
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _all_assignment_value(node: ast.stmt) -> ast.expr | None:
    """Return the value assigned to ``__all__`` by one simple statement."""
    if isinstance(node, ast.Assign):
        targets = node.targets
        value = node.value
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
        value = node.value
    else:
        return None
    assigns_all = any(
        isinstance(target, ast.Name) and target.id == "__all__" for target in targets
    )
    return value if assigns_all else None


def _string_collection(node: ast.expr) -> set[str] | None:
    """Convert a literal sequence of strings to a set when possible."""
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, (list, tuple)):
        return None
    if not all(isinstance(item, str) for item in value):
        return None
    return set(value)


def _literal_exports(tree: ast.Module) -> set[str] | None:
    """Read a simple literal ``__all__`` assignment when one is present."""
    for node in tree.body:
        value = _all_assignment_value(node)
        if value is not None:
            return _string_collection(value)
    return None


def _definition_span(node: ast.AST) -> int:
    """Return the physical line span of a parsed definition."""
    start = getattr(node, "lineno", 0)
    end = getattr(node, "end_lineno", start)
    return max(0, end - start + 1)


def _definitions(
    body: list[ast.stmt],
) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]:
    """Yield function and class definitions directly contained in *body*."""
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node


@dataclass
class DocumentationInventory:
    """Accumulate public-docstring and embedded-example measurements."""

    public_objects: int = 0
    missing_public: list[str] = field(default_factory=list)
    private_candidates: list[dict[str, Any]] = field(default_factory=list)
    doctest_prompts: int = 0
    doctest_directives: int = 0
    plot_directives: int = 0

    def record_public(self, name: str, node: ast.AST) -> None:
        """Record one public object and whether it has a docstring."""
        self.public_objects += 1
        if ast.get_docstring(node, clean=False) is None:
            self.missing_public.append(name)

    def record_private(self, name: str, node: ast.AST) -> None:
        """Record a substantial undocumented private operation candidate."""
        lines = _definition_span(node)
        if ast.get_docstring(node, clean=False) is not None:
            return
        if lines < PRIVATE_OPERATION_MIN_LINES:
            return
        self.private_candidates.append({"name": name, "lines": lines})

    def record_class_members(
        self, class_name: str, node: ast.ClassDef, exported: bool
    ) -> None:
        """Record public and substantial private methods on one class."""
        for member in _definitions(node.body):
            if isinstance(member, ast.ClassDef):
                continue
            member_name = f"{class_name}.{member.name}"
            if not member.name.startswith("_"):
                if exported:
                    self.record_public(member_name, member)
                continue
            if not member.name.startswith("__"):
                self.record_private(member_name, member)

    def record_material(self, docstring: str) -> None:
        """Count executable examples and plot directives in one docstring."""
        self.doctest_prompts += docstring.count(">>>")
        self.doctest_directives += len(DOCTEST_DIRECTIVE.findall(docstring))
        self.doctest_directives += len(MYST_DOCTEST_DIRECTIVE.findall(docstring))
        self.plot_directives += len(PLOT_DIRECTIVE.findall(docstring))
        self.plot_directives += len(MYST_PLOT_DIRECTIVE.findall(docstring))

    def as_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-compatible summary of the inventory."""
        self.missing_public.sort()
        self.private_candidates.sort(key=lambda item: (-item["lines"], item["name"]))
        documented_public = self.public_objects - len(self.missing_public)
        coverage = (
            100.0
            if self.public_objects == 0
            else 100 * documented_public / self.public_objects
        )
        return {
            "public_objects": self.public_objects,
            "documented_public_objects": documented_public,
            "public_docstring_percent": round(coverage, 1),
            "missing_public_count": len(self.missing_public),
            "missing_public": self.missing_public,
            "undocumented_private_operations_min_lines": (PRIVATE_OPERATION_MIN_LINES),
            "undocumented_private_operations": self.private_candidates,
            "doctest_prompts": self.doctest_prompts,
            "doctest_directives": self.doctest_directives,
            "plot_directives": self.plot_directives,
        }


def _record_module_definitions(
    inventory: DocumentationInventory,
    module: str,
    tree: ast.Module,
) -> None:
    """Add top-level definitions and their methods to an inventory."""
    exports = _literal_exports(tree)
    for node in _definitions(tree.body):
        exported = (
            node.name in exports
            if exports is not None
            else not node.name.startswith("_")
        )
        qualified_name = f"{module}.{node.name}"
        if exported:
            inventory.record_public(qualified_name, node)
        else:
            inventory.record_private(qualified_name, node)
        if isinstance(node, ast.ClassDef):
            inventory.record_class_members(qualified_name, node, exported)


def _record_embedded_material(
    inventory: DocumentationInventory, tree: ast.Module
) -> None:
    """Add example and plot counts from every docstring in a module."""
    documentable = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, documentable):
            continue
        docstring = ast.get_docstring(node, clean=False)
        if docstring is not None:
            inventory.record_material(docstring)


def _docstring_metrics(root: Path, files: list[Path]) -> dict[str, Any]:
    """Inventory public documentation and embedded executable material."""
    inventory = DocumentationInventory()
    for path in files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        module = _module_name(root, path)
        inventory.record_public(module, tree)
        _record_module_definitions(inventory, module, tree)
        _record_embedded_material(inventory, tree)
    return inventory.as_dict()


def _source_metrics(root: Path, files: list[Path]) -> dict[str, Any]:
    """Summarise file counts, physical lines, and the largest modules."""
    modules = [
        {"path": str(path.relative_to(root)), "physical_lines": _physical_lines(path)}
        for path in files
    ]
    modules.sort(key=lambda item: (-item["physical_lines"], item["path"]))
    return {
        "files": len(modules),
        "physical_lines": sum(item["physical_lines"] for item in modules),
        "largest_modules": modules[:10],
    }


def _cold_import_metrics(root: Path, samples: int) -> dict[str, Any]:
    """Measure ``import resistics.tui`` in independent Python processes."""
    durations = []
    command = [sys.executable, "-c", "import resistics.tui"]
    for _ in range(samples):
        started = time.perf_counter()
        result = subprocess.run(  # noqa: S603 - executable and code are fixed
            command,
            cwd=root,
            capture_output=True,
            check=False,
            text=True,
        )
        durations.append(time.perf_counter() - started)
        if result.returncode:
            message = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                f"cold import failed with exit {result.returncode}: {message}"
            )

    rounded = [round(duration, 4) for duration in durations]
    if not rounded:
        return {"command": " ".join(command), "samples_seconds": []}
    return {
        "command": " ".join(command),
        "samples_seconds": rounded,
        "minimum_seconds": min(rounded),
        "median_seconds": round(statistics.median(rounded), 4),
        "maximum_seconds": max(rounded),
    }


def build_report(root: Path, import_samples: int) -> dict[str, Any]:
    """Build the complete hardening report for a repository root."""
    production_files = _python_files(root, "resistics")
    test_files = _python_files(root, "tests")
    return {
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "production": _source_metrics(root, production_files),
        "tests": _source_metrics(root, test_files),
        "docstrings": _docstring_metrics(root, production_files),
        "tui_cold_import": _cold_import_metrics(root, import_samples),
    }


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the baseline reporter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--import-samples",
        type=int,
        default=5,
        help="number of fresh-process TUI imports; use 0 to skip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON output path, relative to the repository root",
    )
    return parser.parse_args()


def main() -> int:
    """Write a JSON baseline report and echo it to standard output."""
    args = _parse_args()
    if args.import_samples < 0:
        raise SystemExit("--import-samples must be zero or greater")
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    report = build_report(root, args.import_samples)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
