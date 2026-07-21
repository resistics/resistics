"""
Create a project from an MTH5 file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example uses the MTH5-oriented project API. It creates a resistics project
directory, records the MTH5 file path in project metadata, reloads the project,
and prints a small summary from the MTH5 channel table.

Call ``create_project`` with local project and MTH5 paths. The context manager
closes the read-only MTH5 handle even if summary output fails.
"""
from pathlib import Path

from resistics.project import init, load


def create_project(
    project_path: Path,
    mth5_path: Path,
    ref_time: str,
    overwrite: bool = False,
) -> None:
    """Create and load a resistics project backed by an MTH5 file."""
    init(project_path, mth5_path, ref_time=ref_time, overwrite=overwrite)
    with load(project_path) as project:
        print(f"Project path: {project.project_path}")
        print(f"MTH5 path: {project.mth5_path}")
        print(f"Reference time: {project.ref_time}")
        print(f"Surveys: {project.surveys}")
        print(f"Stations: {project.stations}")
        print(f"Runs: {project.runs}")
        print(f"Sample rates: {project.fs()}")
        print(f"Recording interval: {project.start()} to {project.end()}")


# Replace these paths before calling the example:
# create_project(
#     Path("data/project/my_project"),
#     Path("data/mth5/my_data.h5"),
#     "2020-01-01 00:00:00",
# )
