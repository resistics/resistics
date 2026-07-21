"""
Navigating a project
^^^^^^^^^^^^^^^^^^^^

An MTH5-backed project exposes cached survey, station, run, and channel
summaries. Live groups and samples remain available until the project's owned
MTH5 handle is closed.

The data in this example has been provided for use by the SAMTEX consortium.
For more information, please refer to [Jones2009]_. Additional details about the
data can be found at https://www.mtnet.info/data/kap03/kap03.html.
"""
from pathlib import Path

from resistics.project import load


def describe_project(project_path: Path) -> None:
    """Print the canonical MTH5 hierarchy and inspect its first run."""
    with load(project_path) as project:
        print(project.to_dataframe())
        print(f"Surveys: {project.surveys}")
        print(f"Stations: {project.stations}")
        print(f"Runs: {project.runs}")

        if not project.runs:
            return
        survey, station, run = project.runs[0].split("/")
        print(project.list_channels(survey, station, run))
        time_data = project.read_run(survey, station, run)
        time_data.summary()

# %%
# Point this at a project created by ``eg_09_create_mth5_project.py``. The
# existence guard keeps the gallery runnable when the optional example data is
# not installed.
project_path = Path("..", "..", "data", "project", "kap03")
if (project_path / "resistics.json").is_file():
    describe_project(project_path)
else:
    print(f"Create the optional example project at {project_path} to inspect it")
