"""Command-line entry point for resistics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the resistics command-line parser."""
    parser = argparse.ArgumentParser(prog="resistics")
    subparsers = parser.add_subparsers(dest="command", required=True)
    tui_parser = subparsers.add_parser("tui", help="Open the project terminal UI")
    tui_parser.add_argument(
        "project_path", type=Path, help="Resistics project directory"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the requested resistics command."""
    args = build_parser().parse_args(argv)
    if args.command == "tui":
        from resistics.tui import run_tui

        try:
            run_tui(args.project_path)
        except (OSError, ValueError) as exc:
            print(f"Unable to open project: {exc}", file=sys.stderr)
            return 2
        return 0
    return 2
