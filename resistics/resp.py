import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import pandas as pd
import plotly.graph_objects as go
from attotime import attodatetime, attotimedelta
from loguru import logger
from mth5.groups.run import RunGroup
from mth5.groups.station import StationGroup
from mth5.groups.survey import SurveyGroup
from mth5.mth5 import MTH5
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    TypeAdapter,
    WithJsonSchema,
)
from pydantic_core import core_schema
from typing_extensions import Annotated

# from resistics.common import ResisticsModel, WriteableMetadata
# from resistics.sampling import HighResDateTime, to_timestamp
# from resistics.time import TimeMetadata, TimeReader
from resistics.plot import plot_timeline
from resistics.sampling import datetime_to_string, to_datetime, to_timestamp

DateTimeLike = Union[str, pd.Timestamp, datetime]
TimeDeltaLike = Union[float, timedelta, pd.Timedelta]
PROJ_FILE = "resistics.json"
PROJ_DIRS = [
    "configs",
    "configs/flows",
    "configs/parameters",
    "configs/runs",
    "data",
    "logs",
    "plugins",
]


def datetime_validate(val: Union[attodatetime, DateTimeLike]):
    """Validator to be used by pydantic"""
    if isinstance(val, attodatetime):
        return val
    if isinstance(val, (str, pd.Timestamp, datetime)):
        return to_datetime(val)
    raise TypeError(f"Type {type(val)} not recognised for RSDateTime")


RSDateTime = Annotated[
    attodatetime,
    BeforeValidator(lambda x: datetime_validate(x)),
    PlainSerializer(lambda x: datetime_to_string(x), return_type=str),
    WithJsonSchema({"type": "string"}, mode="serialization"),
]


def get_station_soln_name(
    fs: float, tf_name: str, tf_var: str, postfix: Optional[str] = None
) -> str:
    """Get the name of a solution file"""
    from resistics.common import fs_to_string

    solution_name = f"{fs_to_string(fs)}_{tf_name.lower()}"
    if tf_var != "":
        tf_var = tf_var.replace(" ", "_")
        solution_name = solution_name + f"_{tf_var}"
    if postfix is None:
        return solution_name + ".json"
    return solution_name + "_" + postfix + ".json"


def get_station_dir(
    proj_dir: Path, survey_name: str, station_name: str, config_name: None | str = None
) -> Path:
    """Get path to station data directory"""
    return proj_dir / "data" / survey_name / station_name


def get_results_dir(
    proj_dir: Path, survey_name: str, station_name: str, config_name: str
) -> Path:
    """Get the path to a station results directory with optional config subdir"""
    return proj_dir / "data" / survey_name / station_name / "results" / config_name


def get_run_dir(
    proj_dir: Path,
    survey_name: str,
    station_name: str,
    run_name: str,
    config_name: None | str = None,
) -> Path:
    """Get path to run data directory with optional config subdir"""
    if config_name is None:
        return proj_dir / "data" / survey_name / station_name / run_name
    return proj_dir / "data" / survey_name / station_name / run_name / config_name


# def get_mask_path(proj_dir: Path, site_name: str, config_name: str) -> Path:
#     """Get path to mask data"""
#     return proj_dir / PROJ_DIRS["masks"] / site_name / config_name


# def get_mask_name(fs: float, mask_name: str) -> str:
#     """Get the name of a mask file"""
#     from resistics.common import fs_to_string


#     return f"{fs_to_string(fs)}_{mask_name}.dat"


class Project(BaseModel):
    """
    Class to describe a resistics project

    The resistics Project Class connects all resistics data. It is an essential
    part of processing data with resistics.

    Resistics projects are in directory with several sub-directories. Project
    metadata is saved in the resistics.json file at the top level directory.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dir_path: Path = Field(repr=True)
    mth5_path: Path = Field(repr=True)
    mth5_data: MTH5 = Field(repr=False, exclude=True)
    ref_time: RSDateTime = Field(repr=True)
    experiments: list = Field(repr=True, default=[])
    surveys: list = Field(repr=False, default=[])
    stations: list = Field(repr=False, default=[])
    runs: list = Field(repr=False, default=[])
    plugin_paths: List[Path] = Field(repr=True, default=[])
    table: pd.DataFrame | None = Field(repr=False, exclude=True, default=None)

    def __getitem__(self, obj_path: str) -> SurveyGroup | StationGroup | RunGroup:
        """Get a Site instance given the obj_path of the Site"""
        separator = "/"
        split = obj_path.split(separator)
        logger.info(f"{split=}")
        n_split = len(split)
        logger.info(f"{n_split=}")
        if n_split == 1:
            return self.get_survey(obj_path)
        elif n_split == 2:
            return self.get_station(split[0], split[1])
        elif n_split == 3:
            return self.get_run(split[0], split[1], split[2])
        else:
            raise ValueError(f"Unknown item {obj_path=}")

    def n_surveys(self) -> int:
        """Get the number of surveys in the MTH5 file"""
        return len(self.surveys)

    def n_stations(self, survey: str | None = None) -> int:
        """The number of stations with optional survey filter"""
        return len(self.stations)

    def fs(self) -> list[float]:
        """Get sampling frequencies in the Project"""
        return sorted(list(self.table["sample_rate"].unique()))

    def start(self) -> pd.Timestamp:
        """Get the time of the first sample of the project"""
        return self.table["start"].min()

    def end(self) -> pd.Timestamp:
        """Get the time of the last sample of the project"""
        return self.table["end"].max()

    def get_survey(self, survey: str) -> SurveyGroup:
        """Get a survey from the MTH5 file"""
        if survey not in self.surveys:
            raise ValueError(f"{survey=} not found in MTH5 data")
        return self.mth5_data.surveys_group.get_survey(survey)

    def get_station(self, survey: str, station: str) -> StationGroup:
        """Get a station from the MTH5 file"""
        if f"{survey}/{station}" not in self.stations:
            raise ValueError(f"{station=} not found in {survey=}")
        survey = self.get_survey(survey)
        return survey.stations_group.get_station(station)

    def get_stations(
        self, survey: str | None = None, fs: float | None = None
    ) -> dict[str, StationGroup]:
        """Get stations"""
        table = self.table.copy()
        if survey is not None:
            table = table[table["survey"] == survey]
        if fs is not None:
            table = table[table["sample_rate"] == fs]
        table = table.drop_duplicates(subset=["station_path"], keep="first")
        return {
            x["station_path"]: self.get_station(x["survey"], x["station"])
            for _idx, x in table.iterrows()
        }

    def get_run(self, survey: str, station: str, run: str) -> RunGroup:
        """Get a run from the MTH5 file"""
        if f"{survey}/{station}/{run}" not in self.runs:
            raise ValueError(f"{run=} not found in {station=}, {survey=}")
        station = self.get_station(survey, station)
        return station.get_run(run)

    def get_runs(
        self,
        survey: str | None = None,
        station: str | None = None,
        fs: float | None = None,
    ) -> dict[str, RunGroup]:
        """Filter the runs with various parameters"""
        table = self.table.copy()
        if survey is not None:
            table = table[table["survey"] == survey]
        if station is not None:
            table = table[table["station"] == station]
        if fs is not None:
            table = table[table["sample_rate"] == fs]
        table = table.drop_duplicates(subset=["run_path"], keep="first")
        return {
            x["run_path"]: self.get_run(x["survey"], x["station"], x["run"])
            for _idx, x in table.iterrows()
        }

    def get_concurrent(self, station_path: str, fs: float | None = None) -> List[str]:
        """
        Find sites that recorded concurrently to a specified site

        Parameters
        ----------
        site_name : str
            Search for sites recording concurrently to this site

        Returns
        -------
        List[Site]
            List of Site instances which were recording concurrently
        """
        station_table = self.table[self.table["station_path"] == station_path]
        station_start = station_table["start"].min()
        station_end = station_table["end"].max()

        other_stations = self.table[self.table["station_path"] != station_path]
        if fs is not None:
            other_stations = other_stations[other_stations["sample_rate"] == fs]
        other_stations = other_stations[other_stations["end"] >= station_start]
        other_stations = other_stations[other_stations["start"] <= station_end]
        return other_stations["station_path"].unique().tolist()

    def get_mth5_tree(self):
        pass

    def get_runs_tree(self):
        pass

    def get_proj_tree(self):
        pass

    def to_dataframe(self) -> pd.DataFrame:
        """Provide MTH5 recordings in a dataframe"""
        return self.table.copy()

    def plot(self) -> go.Figure:
        """Plot a timeline of the project"""
        if len(self.table.index) == 0:
            raise ValueError("No measurements found to plot")
        runs_table = self.table.drop_duplicates(subset=["run_path"]).copy()
        runs_table["sample_rate"] = runs_table["sample_rate"].astype(str)
        ref_time = to_timestamp(self.ref_time).tz_localize(tz="UTC")
        return plot_timeline(runs_table, y_col="station_path", ref_time=ref_time)

    def close_mth5(self) -> None:
        """Close the MTH5 file"""
        self.mth5_data.close_mth5()


def init(proj_dir: Path, mth5_path: Path, ref_time, force: bool = False):
    if not mth5_path.exists():
        raise ValueError("MTH5 data file not found")

    if proj_dir.exists() and not force:
        raise ValueError(f"Project already exists in {proj_dir}")
    elif proj_dir.exists():
        logger.warning(f"Forcing reinitialisation of project in {proj_dir}")
    else:
        logger.info(f"Creating new project in {proj_dir}")

    proj_dir.mkdir(exist_ok=True)
    for subdir in PROJ_DIRS:
        subdir_path = proj_dir / subdir
        logger.info(f"Making subdirectory: {subdir_path}")
        subdir_path.mkdir(exist_ok=True)

    mth5_data = MTH5(mth5_path)
    proj = Project(
        dir_path=proj_dir, mth5_path=mth5_path, ref_time=ref_time, mth5_data=mth5_data
    )
    metadata_path = proj_dir / PROJ_FILE
    with metadata_path.open("w") as f:
        f.write(proj.model_dump_json(include={"mth5_path", "ref_time", "plugin_paths"}))

    logger.info(f"Project created in {proj_dir}")


def load(proj_dir: Path):
    metadata_path = proj_dir / PROJ_FILE
    if not metadata_path.exists():
        raise ValueError(proj_dir, f"Resistics project file {metadata_path} not found")

    with metadata_path.open("r") as f:
        metadata = json.load(f)
    mth5_path = Path(metadata.pop("mth5_path"))
    check_project(proj_dir, mth5_path)

    mth5_data = MTH5(mth5_path)
    mth5_data.open_mth5()
    table = mth5_data.channel_summary.to_dataframe()
    table["station_path"] = table[["survey", "station"]].agg("/".join, axis=1)
    table["run_path"] = table[["survey", "station", "run"]].agg("/".join, axis=1)
    return Project(
        dir_path=proj_dir,
        mth5_path=mth5_path,
        mth5_data=mth5_data,
        surveys=table["survey"].unique(),
        stations=table["station_path"].unique(),
        runs=table["run_path"].unique(),
        table=table,
        **metadata,
    )


def check_project(proj_dir, mth5_path) -> bool:
    """Returns True if all require project subdirectories exist otherwise False"""
    # from resistics.common import assert_dir

    if not mth5_path.exists():
        raise ValueError("MTH5 data file not found")
    for subdir in PROJ_DIRS:
        subdir_path = proj_dir / subdir
        # assert_dir(subdir_path)
        return False
    return True
