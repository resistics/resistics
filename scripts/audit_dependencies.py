#!/usr/bin/env python3
"""Audit the complete locked environment for every supported Python minor.

The supported minors come from the package classifiers in ``pyproject.toml``.
Each audit includes the project's default dependency groups and queries OSV
through uv. Accepted risks are exceptional, expiring records: uv ignores them
only while no fixed release exists.

Examples
--------
Run the complete dependency-security check from the repository root::

    uv run --locked --no-sync python scripts/audit_dependencies.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class AcceptedRisk:
    """A temporary exception for an advisory without an available fix."""

    advisory: str
    reason: str
    owner: str
    review_by: date


# Keep this empty unless an advisory has no compatible fixed release. Every
# exception must explain the mitigation, identify its owner, and have a near
# review date. The audit rejects expired or incomplete records and passes each
# advisory to uv with --ignore-until-fixed, never an unconditional --ignore.
ACCEPTED_RISKS: tuple[AcceptedRisk, ...] = ()

MAX_REVIEW_INTERVAL_DAYS = 90
PYTHON_CLASSIFIER = re.compile(r"^Programming Language :: Python :: (3\.\d+)$")


def _supported_pythons(pyproject_path: Path) -> tuple[str, ...]:
    """Return supported Python minors declared by project classifiers.

    Parameters
    ----------
    pyproject_path : Path
        Resistics ``pyproject.toml`` path.

    Returns
    -------
    tuple[str, ...]
        Supported Python minors in numeric order.

    Raises
    ------
    ValueError
        If no minor-specific Python classifier is declared.
    """
    with pyproject_path.open("rb") as stream:
        pyproject = tomllib.load(stream)
    classifiers = pyproject["project"].get("classifiers", [])
    versions = {
        match.group(1)
        for classifier in classifiers
        if (match := PYTHON_CLASSIFIER.fullmatch(classifier)) is not None
    }
    if not versions:
        raise ValueError("No supported Python minors found in project classifiers")
    return tuple(sorted(versions, key=lambda value: tuple(map(int, value.split(".")))))


def _accepted_risk_arguments(today: date) -> list[str]:
    """Validate accepted risks and return their narrow uv arguments.

    Parameters
    ----------
    today : date
        Date used to enforce the review deadline.

    Returns
    -------
    list[str]
        Repeated uv ``--ignore-until-fixed`` arguments.

    Raises
    ------
    ValueError
        If an exception is duplicated, incomplete, expired, or reviewed too
        far in the future.
    """
    arguments: list[str] = []
    seen: set[str] = set()
    for risk in ACCEPTED_RISKS:
        if risk.advisory in seen:
            raise ValueError(f"Duplicate accepted-risk advisory: {risk.advisory}")
        if not all((risk.advisory.strip(), risk.reason.strip(), risk.owner.strip())):
            raise ValueError("Accepted-risk records require an ID, reason, and owner")
        if risk.review_by < today:
            raise ValueError(
                f"Accepted risk {risk.advisory} expired on {risk.review_by.isoformat()}"
            )
        if (risk.review_by - today).days > MAX_REVIEW_INTERVAL_DAYS:
            raise ValueError(
                f"Accepted risk {risk.advisory} must be reviewed within "
                f"{MAX_REVIEW_INTERVAL_DAYS} days"
            )
        seen.add(risk.advisory)
        arguments.extend(("--ignore-until-fixed", risk.advisory))
    return arguments


def _run(command: list[str], *, cwd: Path, environment: dict[str, str]) -> None:
    """Run one visible audit command and require success.

    Parameters
    ----------
    command : list[str]
        Command and arguments to execute without a shell.
    cwd : Path
        Working directory for the child process.
    environment : dict[str, str]
        Complete child-process environment.

    Raises
    ------
    subprocess.CalledProcessError
        If uv finds an unaccepted advisory or cannot complete the audit.
    """
    print(f"$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=environment, check=True)  # noqa: S603


def main() -> int:
    """Audit every supported locked Python dependency set."""
    root = Path(__file__).resolve().parents[1]
    if shutil.which("uv") is None:
        print("uv is required to run the dependency audit", file=sys.stderr)
        return 2

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    try:
        pythons = _supported_pythons(root / "pyproject.toml")
        risk_arguments = _accepted_risk_arguments(date.today())
        if ACCEPTED_RISKS:
            print("Accepted risks:")
            for risk in ACCEPTED_RISKS:
                print(
                    f"  {risk.advisory}: owner={risk.owner}; "
                    f"review_by={risk.review_by.isoformat()}; {risk.reason}"
                )
        else:
            print("Accepted risks: none")

        for python in pythons:
            _run(
                [
                    "uv",
                    "audit",
                    "--locked",
                    "--python-version",
                    python,
                    "--preview-features",
                    "audit-command",
                    *risk_arguments,
                ],
                cwd=root,
                environment=environment,
            )
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Dependency audit failed: {error}", file=sys.stderr)
        return 1

    print(f"Dependency audit passed for Python {', '.join(pythons)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
