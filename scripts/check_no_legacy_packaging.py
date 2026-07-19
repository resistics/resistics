#!/usr/bin/env python3
"""Reject active references to Poetry-era packaging.

The checker scans tracked files and untracked, non-ignored files so that it can
be run before a change is committed. Its own source and the two hardening plans
are excluded because they must name the retired tools to explain the migration.

Examples
--------
Run the check from anywhere inside the repository::

    python scripts/check_no_legacy_packaging.py
"""

from __future__ import annotations

import subprocess  # noqa: S404 - all arguments are fixed by this repository
from pathlib import Path


FORBIDDEN_TERMS = ("poetry", "poetry-core", "pypoetry")
ALLOWED_PATHS = {
    ".agents/plans/code-hardening-implementation.md",
    ".agents/plans/codebase-hardening.md",
    "scripts/check_no_legacy_packaging.py",
}


def repository_files(root: Path) -> list[Path]:
    """Return tracked and unignored working-tree files below *root*."""
    result = subprocess.run(  # noqa: S603 - executable and arguments are fixed
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
    )
    paths = result.stdout.decode("utf-8", errors="surrogateescape").split("\0")
    return [root / path for path in paths if path and path not in ALLOWED_PATHS]


def legacy_references(root: Path) -> list[str]:
    """Return line-oriented findings for retired packaging references."""
    findings = []
    for path in repository_files(root):
        # Tracked files staged for deletion remain in ``git ls-files`` but are
        # intentionally absent from the working tree.
        if not path.is_file():
            continue
        content = path.read_bytes()
        if b"\0" in content:
            continue
        relative = path.relative_to(root).as_posix()
        for line_number, line in enumerate(
            content.decode("utf-8", errors="replace").splitlines(), start=1
        ):
            if any(term in line.casefold() for term in FORBIDDEN_TERMS):
                findings.append(f"{relative}:{line_number}: {line.strip()}")
    return findings


def main() -> int:
    """Print retired references and return a failing status when found."""
    root = Path(__file__).resolve().parents[1]
    findings = legacy_references(root)
    if not findings:
        print("No legacy packaging references found")
        return 0
    print("Legacy packaging references found:")
    for finding in findings:
        print(f"  {finding}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
