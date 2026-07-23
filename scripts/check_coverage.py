#!/usr/bin/env python3
"""Run the maintained branch-coverage gate over tests and Sphinx doctests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

try:
    from scripts.check_documentation import _prepare_output, _sphinx_command
except ModuleNotFoundError:  # Direct execution adds scripts/, not its parent.
    from check_documentation import _prepare_output, _sphinx_command

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCTEST_OUTPUT = PROJECT_ROOT / ".artifacts" / "hardening" / "coverage" / "doctest"


def _coverage_commands(output_dir: Path) -> list[list[str]]:
    """Return the ordered commands that produce the combined coverage data."""
    sphinx = _sphinx_command("doctest", output_dir)
    covered_sphinx = [
        sys.executable,
        "-m",
        "coverage",
        "run",
        "--branch",
        "--append",
        *sphinx[1:],
    ]
    return [
        [sys.executable, "-m", "coverage", "erase"],
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--branch",
            "-m",
            "pytest",
            "-m",
            "not performance",
        ],
        covered_sphinx,
        [sys.executable, "-m", "coverage", "report"],
        [sys.executable, "-m", "coverage", "html"],
        [sys.executable, "-m", "coverage", "xml"],
    ]


def main() -> int:
    """Execute the source tests and authoritative doctests under coverage."""
    _prepare_output(DOCTEST_OUTPUT)
    try:
        for command in _coverage_commands(DOCTEST_OUTPUT):
            print(f"$ {' '.join(command)}", flush=True)
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)  # noqa: S603
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Coverage check failed: {error}", file=sys.stderr)
        return 1
    print("Coverage check passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
