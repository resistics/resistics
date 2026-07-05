"""
Create a project from an MTH5 file.

This example uses the MTH5-oriented project API. It creates a resistics project
directory, records the MTH5 file path in project metadata, reloads the project,
and prints a small summary from the MTH5 channel table.

Run from the repository root, replacing the paths with local paths:

    python examples/project/eg_09_create_mth5_project.py \
        --project-path data/project/my_project \
        --mth5-path data/mth5/my_data.h5 \
        --ref-time "2020-01-01 00:00:00" \
        --overwrite
"""
from argparse import ArgumentParser
from pathlib import Path

import resistics.resp as resp


def create_project(
    project_path: Path,
    mth5_path: Path,
    ref_time: str,
    overwrite: bool = False,
) -> None:
    """Create and load a resistics project backed by an MTH5 file."""
    resp.init(project_path, mth5_path, ref_time=ref_time, force=overwrite)
    project = resp.load(project_path)

    print(f"Project path: {project.dir_path}")
    print(f"MTH5 path: {project.mth5_path}")
    print(f"Reference time: {project.ref_time}")
    print(f"Surveys: {list(project.surveys)}")
    print(f"Stations: {list(project.stations)}")
    print(f"Runs: {list(project.runs)}")
    print(f"Sample rates: {project.fs()}")
    print(f"Recording interval: {project.start()} to {project.end()}")

    project.close_mth5()


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description="Create a resistics project from MTH5 data.")
    parser.add_argument(
        "--project-path",
        type=Path,
        required=True,
        help="Directory where the resistics project will be created.",
    )
    parser.add_argument(
        "--mth5-path",
        type=Path,
        required=True,
        help="Path to an existing MTH5 file.",
    )
    parser.add_argument(
        "--ref-time",
        default="2020-01-01 00:00:00",
        help="Project reference time.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reinitialise the project directory if it already exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_project(
        project_path=args.project_path,
        mth5_path=args.mth5_path,
        ref_time=args.ref_time,
        overwrite=args.overwrite,
    )
