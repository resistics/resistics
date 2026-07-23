#!/usr/bin/env python3
"""Run the maintained local Sphinx documentation gates."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
SOURCE_DIR = PROJECT_ROOT / "docs" / "source"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / ".artifacts" / "hardening" / "documentation" / "gate"
)
OUTPUT_MARKER = ".resistics-documentation-gate"
EXPECTED_PLOT_PREFIXES = frozenset(
    {
        "band-pass",
        "crop-timestamps",
        "decimate",
        "decimated-data",
        "fourier-transform",
        "high-pass",
        "low-pass",
        "notch",
        "resample",
        "shift-timestamps",
        "subsamples-default",
        "subsamples-negative",
        "subsamples-positive",
        "subsection",
        "window-overlap",
        "window-table",
    }
)


def _sphinx_command(builder: str, output_dir: Path) -> list[str]:
    """Return the strict Sphinx command for one validation builder."""
    command = [
        sys.executable,
        "-m",
        "sphinx",
        "-W",
        "--keep-going",
        "-b",
        builder,
        "-E",
    ]
    if builder == "html":
        command.insert(3, "-n")
    else:
        # The strict HTML gate already executes every MyST-NB tutorial. The
        # validation-only builders should not repeat that independent work.
        command.extend(("-D", "nb_execution_mode=off"))
    command.extend((str(SOURCE_DIR), str(output_dir)))
    return command


def _prepare_output(output_dir: Path) -> None:
    """Create an empty owned output directory so stale artifacts cannot pass."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)


def _claim_output_root(output_root: Path) -> None:
    """Claim an empty root before allowing its generated subdirectories to reset."""
    output_root.mkdir(parents=True, exist_ok=True)
    marker = output_root / OUTPUT_MARKER
    if marker.exists():
        return
    owned_names = ("html", "doctest", "linkcheck")
    conflicts = [name for name in owned_names if (output_root / name).exists()]
    if conflicts:
        raise RuntimeError(
            f"Refusing to replace unowned documentation output: {conflicts}"
        )
    marker.write_text("Owned by scripts/check_documentation.py\n", encoding="utf-8")


def _run_sphinx(builder: str, output_dir: Path, environment: dict[str, str]) -> None:
    """Run one strict Sphinx validation builder in a clean output directory."""
    _prepare_output(output_dir)
    command = _sphinx_command(builder, output_dir)
    print(f"$ {' '.join(command)}", flush=True)
    subprocess.run(  # noqa: S603
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )


def _verify_plot_artifacts(html_dir: Path) -> None:
    """Require exactly the semantic PNG artifacts owned by fenced plot blocks."""
    plot_dir = html_dir / "plot_directive"
    actual = {
        path.stem
        for path in plot_dir.glob("*.png")
        if not path.name.endswith(".hires.png")
    }
    missing = sorted(EXPECTED_PLOT_PREFIXES - actual)
    unexpected = sorted(actual - EXPECTED_PLOT_PREFIXES)
    if missing or unexpected:
        raise RuntimeError(
            "Unexpected documentation plot artifacts: "
            f"missing={missing}, unexpected={unexpected}"
        )
    print(f"Verified {len(actual)} fenced plot artifacts", flush=True)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the selected local documentation gate and output location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "check",
        choices=("build", "links"),
        default="build",
        nargs="?",
        help="run strict HTML/doctest checks or the networked external-link check",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="directory that owns disposable Sphinx validation output",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run strict HTML and doctest checks, or the explicit external-link check."""
    args = _parse_args(argv)
    output_root = args.output_root.resolve()
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", str(output_root / "matplotlib"))

    try:
        _claim_output_root(output_root)
        if args.check == "links":
            _run_sphinx("linkcheck", output_root / "linkcheck", environment)
        else:
            html_dir = output_root / "html"
            _run_sphinx("html", html_dir, environment)
            _verify_plot_artifacts(html_dir)
            _run_sphinx("doctest", output_root / "doctest", environment)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Documentation check failed: {error}", file=sys.stderr)
        return 1

    print(f"Documentation {args.check} check passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
