"""
MTH5-backed resistics project model and path helpers.

The public project API is MTH5-only. Legacy directory readers may still exist as
internal conversion helpers, but project discovery and letsgo workflows should
use surveys, stations, and runs from an MTH5 file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger
from mth5.groups.run import RunGroup
from mth5.groups.station import StationGroup
from mth5.groups.survey import SurveyGroup
from mth5.mth5 import MTH5
from pydantic import Field

from resistics.common import ResisticsModel
from resistics.plot import plot_timeline
from resistics.sampling import DateTimeLike, HighResDateTime, to_datetime, to_timestamp
from resistics.time import ChanMetadata, TimeData, TimeMetadata


PROJ_FILE = "resistics.json"
CANONICAL_PROJ_DIRS = (
    "processing",
    "processing/flows",
    "processing/parameters",
    "processing/jobs",
    "data",
    "logs",
    "plugins",
)

PROJ_DIRS = {
    "processing": "processing",
    "flows": "processing/flows",
    "parameters": "processing/parameters",
    "jobs": "processing/jobs",
    "data": "data",
    "logs": "logs",
    "plugins": "plugins",
    # Legacy derived-artifact keys retained for internal conversion code/tests.
    "time": "time",
    "calibration": "calibrate",
    "spectra": "spectra",
    "evals": "evals",
    "features": "features",
    "masks": "masks",
    "results": "results",
    "images": "images",
}


def _as_path(value: Union[Path, str]) -> Path:
    """Return a path from a string or path-like value."""
    return value if isinstance(value, Path) else Path(value)


def get_flow_path(project_path: Path, flow_name: str) -> Path:
    """Get path to a flow definition."""
    return project_path / "processing" / "flows" / flow_name


def get_parameters_path(project_path: Path, parameters_name: str) -> Path:
    """Get path to a processing parameters definition."""
    return project_path / "processing" / "parameters" / parameters_name


def get_job_path(project_path: Path, job_name: str) -> Path:
    """Get path to a processing job definition."""
    return project_path / "processing" / "jobs" / job_name


def get_run_data_path(
    project_path: Path, survey: str, station: str, run: str
) -> Path:
    """Get path to derived artifacts for an MTH5 run."""
    return project_path / "data" / survey / station / run


def get_results_path(
    project_path: Path,
    survey: str,
    station: str,
    output_label: Optional[str] = None,
) -> Path:
    """Get path to final outputs for a processing job."""
    if output_label is None:
        return project_path / "results" / survey / station
    return project_path / "data" / survey / station / "results" / output_label


def get_calibration_path(project_path: Path) -> Path:
    """Get path to calibration data used by existing calibration processors."""
    return project_path / "calibrate"


def get_meas_time_path(project_path: Path, site_name: str, meas_name: str) -> Path:
    """Legacy path helper retained for internal conversion code."""
    return project_path / "time" / site_name / meas_name


def get_meas_spectra_path(
    project_path: Path, site_name: str, meas_name: str, config_name: str
) -> Path:
    """Legacy path helper retained for internal conversion code."""
    return project_path / "spectra" / site_name / config_name / meas_name


def get_meas_evals_path(
    project_path: Path, site_name: str, meas_name: str, config_name: str
) -> Path:
    """Legacy path helper retained for internal conversion code."""
    return project_path / "evals" / site_name / config_name / meas_name


def get_meas_features_path(
    project_path: Path, site_name: str, meas_name: str, config_name: str
) -> Path:
    """Legacy path helper retained for internal conversion code."""
    return project_path / "features" / site_name / config_name / meas_name


def get_mask_path(project_path: Path, site_name: str, config_name: str) -> Path:
    """Legacy path helper retained for internal conversion code."""
    return project_path / "masks" / site_name / config_name


def get_mask_name(fs: float, mask_name: str) -> str:
    """Get a mask file name."""
    from resistics.common import fs_to_string

    return f"{fs_to_string(fs)}_{mask_name}.dat"


def get_log_path(project_path: Path, job_name: str) -> Path:
    """Get path to a processing-job log file."""
    return project_path / "logs" / f"{job_name}.log"


def get_solution_name(
    fs: float, tf_name: str, tf_var: str, postfix: Optional[str] = None
) -> str:
    """Get the name of a solution file."""
    from resistics.common import fs_to_string

    solution_name = f"{fs_to_string(fs)}_{tf_name.lower()}"
    if tf_var != "":
        solution_name = solution_name + f"_{tf_var.replace(' ', '_')}"
    if postfix is not None:
        solution_name = solution_name + f"_{postfix}"
    return solution_name + ".json"


class ProjectMetadata(ResisticsModel):
    """Serializable MTH5 project metadata stored in ``resistics.json``."""

    mth5_path: Path
    ref_time: HighResDateTime
    plugin_paths: List[Path] = Field(default_factory=list)


class Project(ResisticsModel):
    """An MTH5-backed resistics project."""

    project_path: Path
    mth5_path: Path
    ref_time: HighResDateTime
    plugin_paths: List[Path] = Field(default_factory=list)
    mth5_data: MTH5 = Field(repr=False, exclude=True)
    table: pd.DataFrame = Field(repr=False, exclude=True)
    surveys: List[str] = Field(default_factory=list)
    stations: List[str] = Field(default_factory=list)
    runs: List[str] = Field(default_factory=list)

    @property
    def dir_path(self) -> Path:
        """Backward-compatible alias for existing processing helpers."""
        return self.project_path

    @property
    def metadata(self) -> ProjectMetadata:
        """Backward-compatible metadata wrapper."""
        return ProjectMetadata(
            mth5_path=self.mth5_path,
            ref_time=self.ref_time,
            plugin_paths=self.plugin_paths,
        )

    def __getitem__(self, obj_path: str) -> SurveyGroup | StationGroup | RunGroup:
        """Get an MTH5 survey, station, or run by slash-separated path."""
        parts = obj_path.split("/")
        if len(parts) == 1:
            return self.get_survey(parts[0])
        if len(parts) == 2:
            return self.get_station(parts[0], parts[1])
        if len(parts) == 3:
            return self.get_run(parts[0], parts[1], parts[2])
        raise ValueError(f"Unknown MTH5 object path {obj_path!r}")

    def n_surveys(self) -> int:
        """Get the number of surveys."""
        return len(self.surveys)

    def n_stations(self, survey: Optional[str] = None) -> int:
        """Get the number of stations, optionally filtered by survey."""
        if survey is None:
            return len(self.stations)
        return len(self.get_stations(survey=survey))

    def fs(self) -> List[float]:
        """Get project sample rates."""
        if self.table.empty:
            return []
        return sorted([float(x) for x in self.table["sample_rate"].dropna().unique()])

    def start(self) -> pd.Timestamp:
        """Get the first project timestamp."""
        return self.table["start"].min()

    def end(self) -> pd.Timestamp:
        """Get the last project timestamp."""
        return self.table["end"].max()

    def get_survey(self, survey: str) -> SurveyGroup:
        """Get an MTH5 survey group."""
        if survey not in self.surveys:
            raise ValueError(f"Survey {survey!r} not found in MTH5 data")
        return self.mth5_data.surveys_group.get_survey(survey)

    def get_station(self, survey: str, station: str) -> StationGroup:
        """Get an MTH5 station group."""
        station_path = f"{survey}/{station}"
        if station_path not in self.stations:
            raise ValueError(f"Station {station_path!r} not found in MTH5 data")
        return self.get_survey(survey).stations_group.get_station(station)

    def get_stations(
        self, survey: Optional[str] = None, fs: Optional[float] = None
    ) -> Dict[str, StationGroup]:
        """Get station groups keyed by ``survey/station``."""
        table = self._filter_table(survey=survey, fs=fs)
        table = table.drop_duplicates(subset=["station_path"], keep="first")
        return {
            row["station_path"]: self.get_station(row["survey"], row["station"])
            for _, row in table.iterrows()
        }

    def get_run(self, survey: str, station: str, run: str) -> RunGroup:
        """Get an MTH5 run group."""
        run_path = f"{survey}/{station}/{run}"
        if run_path not in self.runs:
            raise ValueError(f"Run {run_path!r} not found in MTH5 data")
        return self.get_station(survey, station).get_run(run)

    def get_runs(
        self,
        survey: Optional[str] = None,
        station: Optional[str] = None,
        fs: Optional[float] = None,
    ) -> Dict[str, RunGroup]:
        """Get run groups keyed by ``survey/station/run``."""
        table = self._filter_table(survey=survey, station=station, fs=fs)
        table = table.drop_duplicates(subset=["run_path"], keep="first")
        return {
            row["run_path"]: self.get_run(row["survey"], row["station"], row["run"])
            for _, row in table.iterrows()
        }

    def get_concurrent(
        self, station_path: str, fs: Optional[float] = None
    ) -> List[str]:
        """Find station paths that overlap in time with ``station_path``."""
        station_table = self.table[self.table["station_path"] == station_path]
        if station_table.empty:
            raise ValueError(f"Station {station_path!r} not found in MTH5 data")
        station_start = station_table["start"].min()
        station_end = station_table["end"].max()

        other_stations = self.table[self.table["station_path"] != station_path]
        if fs is not None:
            other_stations = other_stations[other_stations["sample_rate"] == fs]
        other_stations = other_stations[other_stations["end"] >= station_start]
        other_stations = other_stations[other_stations["start"] <= station_end]
        return sorted(other_stations["station_path"].unique().tolist())

    def read_run(
        self,
        survey: str,
        station: str,
        run: str,
        chans: Optional[Iterable[str]] = None,
        from_time: Optional[DateTimeLike] = None,
        to_time: Optional[DateTimeLike] = None,
        from_sample: Optional[int] = None,
        to_sample: Optional[int] = None,
    ) -> TimeData:
        """Read an MTH5 run into existing resistics ``TimeData`` containers."""
        run_group = self.get_run(survey, station, run)
        run_ts = _run_group_to_run_ts(run_group)
        time_data = _run_ts_to_time_data(run_ts, chans=chans)
        if from_sample is not None or to_sample is not None:
            data = time_data.data[:, from_sample:to_sample]
            first_sample = 0 if from_sample is None else from_sample
            metadata = time_data.metadata.model_copy(deep=True)
            first_time = metadata.first_time
            if first_sample != 0:
                from resistics.sampling import to_timedelta

                first_time = first_time + to_timedelta(first_sample / metadata.fs)
            metadata.n_samples = data.shape[1]
            metadata.first_time = first_time
            from resistics.time import adjust_time_metadata

            metadata = adjust_time_metadata(metadata, metadata.fs, first_time, data.shape[1])
            time_data = TimeData(metadata, data)
        if from_time is not None or to_time is not None:
            from_time = from_time or time_data.metadata.first_time
            to_time = to_time or time_data.metadata.last_time
            time_data = time_data.subsection(from_time, to_time)
        return time_data

    def to_dataframe(self) -> pd.DataFrame:
        """Return the project MTH5 channel summary table."""
        return self.table.copy()

    def plot(self) -> go.Figure:
        """Plot project run timelines."""
        if self.table.empty:
            raise ValueError("No runs found to plot")
        runs_table = self.table.drop_duplicates(subset=["run_path"]).copy()
        runs_table["sample_rate"] = runs_table["sample_rate"].astype(str)
        ref_time = to_timestamp(self.ref_time)
        return plot_timeline(runs_table, y_col="station_path", ref_time=ref_time)

    def close_mth5(self) -> None:
        """Close the underlying MTH5 file."""
        self.mth5_data.close_mth5()

    def _filter_table(
        self,
        survey: Optional[str] = None,
        station: Optional[str] = None,
        fs: Optional[float] = None,
    ) -> pd.DataFrame:
        """Filter the cached MTH5 summary table."""
        table = self.table.copy()
        if survey is not None:
            table = table[table["survey"] == survey]
        if station is not None:
            table = table[table["station"] == station]
        if fs is not None:
            table = table[table["sample_rate"] == fs]
        return table


def init(
    project_path: Union[Path, str],
    mth5_path: Union[Path, str],
    ref_time: DateTimeLike,
    overwrite: bool = False,
    force: Optional[bool] = None,
    plugin_paths: Optional[List[Union[Path, str]]] = None,
) -> bool:
    """Initialise an MTH5-backed resistics project."""
    if force is not None:
        overwrite = force
    project_path = _as_path(project_path)
    mth5_path = _as_path(mth5_path)
    if not mth5_path.exists():
        raise ValueError(f"MTH5 data file not found: {mth5_path}")
    metadata_path = project_path / PROJ_FILE
    if metadata_path.exists() and not overwrite:
        raise ValueError(f"Project already exists in {project_path}")

    project_path.mkdir(parents=True, exist_ok=True)
    for subdir in CANONICAL_PROJ_DIRS:
        (project_path / subdir).mkdir(parents=True, exist_ok=True)

    metadata = ProjectMetadata(
        mth5_path=mth5_path,
        ref_time=to_datetime(ref_time),
        plugin_paths=[_as_path(path) for path in plugin_paths or []],
    )
    metadata_path.write_text(metadata.model_dump_json())
    logger.info(f"Project created in {project_path}")
    return True


def load(project_path: Union[Path, str]) -> Project:
    """Load an MTH5-backed resistics project."""
    project_path = _as_path(project_path)
    metadata_path = project_path / PROJ_FILE
    if not metadata_path.exists():
        raise ValueError(f"Resistics project file {metadata_path} not found")
    metadata = ProjectMetadata.model_validate_json(metadata_path.read_bytes())
    check_project(project_path, metadata.mth5_path)

    mth5_data = MTH5(metadata.mth5_path)
    mth5_data.open_mth5()
    table = _prepare_channel_summary(mth5_data.channel_summary.to_dataframe())
    return Project(
        project_path=project_path,
        mth5_path=metadata.mth5_path,
        ref_time=metadata.ref_time,
        plugin_paths=metadata.plugin_paths,
        mth5_data=mth5_data,
        table=table,
        surveys=sorted(table["survey"].dropna().unique().tolist()),
        stations=sorted(table["station_path"].dropna().unique().tolist()),
        runs=sorted(table["run_path"].dropna().unique().tolist()),
    )


def check_project(project_path: Union[Path, str], mth5_path: Union[Path, str]) -> bool:
    """Validate an MTH5-backed resistics project directory."""
    project_path = _as_path(project_path)
    mth5_path = _as_path(mth5_path)
    if not mth5_path.exists():
        raise ValueError(f"MTH5 data file not found: {mth5_path}")
    for subdir in CANONICAL_PROJ_DIRS:
        subdir_path = project_path / subdir
        if not subdir_path.exists() or not subdir_path.is_dir():
            raise ValueError(f"Required project directory not found: {subdir_path}")
    return True


def _prepare_channel_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Add canonical path columns to an MTH5 channel summary table."""
    table = table.copy()
    if table.empty:
        for col in ["survey", "station", "run", "station_path", "run_path"]:
            table[col] = []
        return table
    for column in ["survey", "station", "run"]:
        if column not in table.columns:
            raise ValueError(f"MTH5 channel summary missing {column!r} column")
    table["station_path"] = table[["survey", "station"]].astype(str).agg("/".join, axis=1)
    table["run_path"] = table[["survey", "station", "run"]].astype(str).agg("/".join, axis=1)
    if "start" in table.columns:
        table["start"] = pd.to_datetime(table["start"])
    if "end" in table.columns:
        table["end"] = pd.to_datetime(table["end"])
    return table


def _run_group_to_run_ts(run_group: RunGroup) -> Any:
    """Convert a RunGroup to the MTH5 run time-series object."""
    if hasattr(run_group, "to_runts"):
        return run_group.to_runts()
    if hasattr(run_group, "to_run_ts"):
        return run_group.to_run_ts()
    raise NotImplementedError(
        "Unable to read MTH5 run: expected RunGroup.to_runts() or to_run_ts()."
    )


def _run_ts_to_time_data(
    run_ts: Any, chans: Optional[Iterable[str]] = None
) -> TimeData:
    """Convert a run time-series object to ``TimeData``."""
    dataset = _extract_run_ts_dataset(run_ts)
    if chans is not None:
        chans = list(chans)
        dataset = dataset[chans]
    else:
        chans = list(getattr(dataset, "data_vars", []))
    if not chans:
        raise ValueError("No channels found in MTH5 run")

    arrays = [np.asarray(dataset[chan].data) for chan in chans]
    data = np.vstack(arrays)
    first_chan = dataset[chans[0]]
    fs = float(
        getattr(first_chan, "sample_rate", None)
        or first_chan.attrs.get("sample_rate")
        or first_chan.attrs.get("sampling_rate")
    )
    n_samples = data.shape[1]
    first_time = _get_first_time(first_chan)
    chans_metadata = {
        chan: ChanMetadata(name=chan, data_files=None) for chan in chans
    }
    metadata = TimeMetadata(
        fs=fs,
        chans=chans,
        n_chans=len(chans),
        n_samples=n_samples,
        first_time=first_time,
        last_time=first_time,
        chans_metadata=chans_metadata,
    )
    from resistics.time import adjust_time_metadata

    metadata = adjust_time_metadata(metadata, fs, metadata.first_time, n_samples)
    return TimeData(metadata, data)


def _extract_run_ts_dataset(run_ts: Any) -> Any:
    """Return the xarray-like dataset from an MTH5 run time-series object."""
    if hasattr(run_ts, "dataset"):
        return run_ts.dataset
    if hasattr(run_ts, "to_xarray"):
        return run_ts.to_xarray()
    if hasattr(run_ts, "to_dataset"):
        return run_ts.to_dataset()
    raise NotImplementedError(
        "Unable to convert MTH5 RunTS to data: no dataset/to_xarray/to_dataset API."
    )


def _get_first_time(channel_data: Any) -> HighResDateTime:
    """Extract a first timestamp from an xarray-like channel."""
    for attr in ("start", "start_time"):
        value = getattr(channel_data, attr, None) or channel_data.attrs.get(attr)
        if value is not None:
            return to_datetime(value)
    if "time" in getattr(channel_data, "coords", {}):
        return to_datetime(pd.to_datetime(channel_data.coords["time"].values[0]))
    raise ValueError("Unable to determine MTH5 channel start time")


# Legacy names kept only so imports fail less abruptly during migration.
Measurement = None
Site = None
