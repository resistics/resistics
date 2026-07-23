#!/usr/bin/env python3
"""Test Resistics' declared dependency floors from paired local wheels.

The check builds Resistics and its sibling RegressionInC package, resolves the
lowest versions of Resistics' direct runtime and test dependencies, installs
the wheels into a temporary environment, and verifies both imports resolve from
the installed distributions. It never modifies the normal project environment
or ``uv.lock``.

Examples
--------
Run the compatibility check from the repository root::

    uv run --locked --no-sync python scripts/check_minimum_dependencies.py
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

PACKAGE_NAME = re.compile(r"^\s*([A-Za-z0-9_.-]+)(?:\[[^]]+\])?(.*)$")
COMPILED_REQUIREMENT = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)")
LOWER_BOUND = re.compile(r"(?:^|,)\s*>=\s*([^,;\s]+)")
CHECKED_GROUPS = ("shared", "tests")


def _normalise_name(name: str) -> str:
    """Return the canonical comparison spelling for a distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _run(command: list[str], *, cwd: Path, environment: dict[str, str]) -> None:
    """Run one visible compatibility command and require success.

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
        If the child command exits unsuccessfully.
    """
    print(f"$ {' '.join(command)}", flush=True)
    # Every caller supplies an explicit argv assembled from repository or
    # freshly created temporary paths; no shell or downloaded command is used.
    subprocess.run(command, cwd=cwd, env=environment, check=True)  # noqa: S603


def _declared_floors(pyproject_path: Path) -> dict[str, str]:
    """Return runtime and checked-group ``>=`` floors from project metadata.

    Parameters
    ----------
    pyproject_path : Path
        Resistics ``pyproject.toml`` path.

    Returns
    -------
    dict[str, str]
        Canonical distribution names mapped to their declared lower versions.

    Raises
    ------
    ValueError
        If a checked direct requirement has no explicit lower bound.
    """
    with pyproject_path.open("rb") as stream:
        pyproject = tomllib.load(stream)
    requirements = list(pyproject["project"]["dependencies"])
    groups = pyproject.get("dependency-groups", {})
    for group in CHECKED_GROUPS:
        requirements.extend(groups.get(group, []))

    floors = {}
    for requirement in requirements:
        match = PACKAGE_NAME.match(requirement)
        if match is None:
            raise ValueError(f"Unable to parse direct requirement: {requirement}")
        name, specifier = match.groups()
        lower_bound = LOWER_BOUND.search(specifier)
        if lower_bound is None:
            raise ValueError(
                f"Direct requirement has no testable >= floor: {requirement}"
            )
        floors[_normalise_name(name)] = lower_bound.group(1)
    return floors


def _compiled_versions(requirements_path: Path) -> dict[str, str]:
    """Read exact versions from a uv-compiled requirements file.

    Parameters
    ----------
    requirements_path : Path
        Requirements file emitted by ``uv pip compile``.

    Returns
    -------
    dict[str, str]
        Canonical distribution names mapped to resolved versions.
    """
    versions = {}
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        match = COMPILED_REQUIREMENT.match(line)
        if match is None:
            continue
        name, version = match.groups()
        versions[_normalise_name(name)] = version
    return versions


def _verify_resolved_floors(
    pyproject_path: Path, requirements_path: Path
) -> dict[str, str]:
    """Require uv to resolve every checked direct dependency at its floor.

    Parameters
    ----------
    pyproject_path : Path
        Resistics ``pyproject.toml`` path.
    requirements_path : Path
        Lowest-direct resolution emitted by uv.

    Returns
    -------
    dict[str, str]
        Verified direct dependency versions.

    Raises
    ------
    RuntimeError
        If a dependency is missing or resolves above its declared floor.
    """
    floors = _declared_floors(pyproject_path)
    resolved = _compiled_versions(requirements_path)
    mismatches = []
    for name, floor in sorted(floors.items()):
        version = resolved.get(name)
        if version != floor:
            mismatches.append(
                f"{name}: declared {floor}, resolved {version or 'missing'}"
            )
    if mismatches:
        details = "\n  ".join(mismatches)
        raise RuntimeError(
            "Declared dependency floors were not installable on the target "
            f"interpreter:\n  {details}"
        )
    return {name: resolved[name] for name in sorted(floors)}


def _one_wheel(directory: Path, project: str) -> Path:
    """Return the sole wheel in a build directory.

    Parameters
    ----------
    directory : Path
        Directory populated by ``uv build``.
    project : str
        Project label used in failure messages.

    Returns
    -------
    Path
        Built wheel path.

    Raises
    ------
    RuntimeError
        If the build directory does not contain exactly one wheel.
    """
    wheels = list(directory.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one {project} wheel in {directory}, got {wheels}")
    return wheels[0]


def _arguments() -> argparse.Namespace:
    """Parse command-line arguments for the compatibility check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        default="3.12",
        help="Python interpreter or version passed to uv (default: 3.12)",
    )
    parser.add_argument(
        "--regressioninc",
        type=Path,
        help="RegressionInC checkout (default: sibling ../regressioninc)",
    )
    return parser.parse_args()


def main() -> int:
    """Build, install, and exercise the declared dependency-floor environment."""
    arguments = _arguments()
    root = Path(__file__).resolve().parents[1]
    regressioninc = (
        arguments.regressioninc.resolve()
        if arguments.regressioninc is not None
        else root.parent.joinpath("regressioninc").resolve()
    )
    if shutil.which("uv") is None:
        print("uv is required to run the dependency-floor check", file=sys.stderr)
        return 2
    if not regressioninc.joinpath("pyproject.toml").is_file():
        print(f"RegressionInC checkout not found at {regressioninc}", file=sys.stderr)
        return 2

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    try:
        with tempfile.TemporaryDirectory(prefix="resistics-minimum-") as temporary:
            work = Path(temporary)
            resistics_dist = work / "dist" / "resistics"
            regressioninc_dist = work / "dist" / "regressioninc"
            requirements = work / "requirements.txt"
            virtual_environment = work / "venv"
            environment["MPLBACKEND"] = "Agg"
            environment["MPLCONFIGDIR"] = str(work / "matplotlib")

            for source, destination in (
                (root, resistics_dist),
                (regressioninc, regressioninc_dist),
            ):
                _run(
                    [
                        "uv",
                        "build",
                        "--wheel",
                        "--no-sources",
                        "--python",
                        arguments.python,
                        "--out-dir",
                        str(destination),
                        str(source),
                    ],
                    cwd=work,
                    environment=environment,
                )

            _run(
                [
                    "uv",
                    "pip",
                    "compile",
                    str(root / "pyproject.toml"),
                    "--group",
                    "shared",
                    "--group",
                    "tests",
                    "--resolution",
                    "lowest-direct",
                    "--python",
                    arguments.python,
                    "--no-sources",
                    "--find-links",
                    str(regressioninc_dist),
                    "--output-file",
                    str(requirements),
                    "--no-header",
                    "--no-annotate",
                ],
                cwd=root,
                environment=environment,
            )
            versions = _verify_resolved_floors(root / "pyproject.toml", requirements)
            print("Verified declared floors:")
            for name, version in versions.items():
                print(f"  {name}=={version}")

            _run(
                [
                    "uv",
                    "venv",
                    "--python",
                    arguments.python,
                    str(virtual_environment),
                ],
                cwd=work,
                environment=environment,
            )
            environment_python = virtual_environment / "bin" / "python"
            if os.name == "nt":
                environment_python = virtual_environment / "Scripts" / "python.exe"
            _run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(environment_python),
                    "--requirements",
                    str(requirements),
                    "--find-links",
                    str(regressioninc_dist),
                    "--strict",
                ],
                cwd=work,
                environment=environment,
            )
            resistics_wheel = _one_wheel(resistics_dist, "Resistics")
            _one_wheel(regressioninc_dist, "RegressionInC")
            _run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(environment_python),
                    "--no-deps",
                    str(resistics_wheel),
                    "--strict",
                ],
                cwd=work,
                environment=environment,
            )
            _run(
                ["uv", "pip", "check", "--python", str(environment_python)],
                cwd=work,
                environment=environment,
            )
            import_check = (
                "from pathlib import Path; import regressioninc, resistics; "
                f"root=Path({str(root)!r}); sibling=Path({str(regressioninc)!r}); "
                "paths=[Path(resistics.__file__).resolve(), "
                "Path(regressioninc.__file__).resolve()]; "
                "assert all(root not in path.parents and sibling not in path.parents "
                "for path in paths), paths; print(*paths, sep='\\n')"
            )
            _run(
                [str(environment_python), "-c", import_check],
                cwd=work,
                environment=environment,
            )
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Dependency-floor check failed: {error}", file=sys.stderr)
        return 1

    print("Dependency-floor check passed using paired installed wheels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
