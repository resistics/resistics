"""
MTH5-backed resistics project model and path helpers.

The public project API is MTH5-only. Legacy directory readers may still exist as
internal conversion helpers, but project discovery and letsgo workflows should
use surveys, stations, and runs from an MTH5 file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger
from mth5.groups.run import RunGroup
from mth5.groups.station import StationGroup
from mth5.groups.survey import SurveyGroup
from mth5.mth5 import MTH5
from pydantic import Field, JsonValue

from resistics.common import ResisticsModel
from resistics.plot import plot_timeline
from resistics.sampling import DateTimeLike, HighResDateTime, to_datetime, to_timestamp
from resistics.templates import install_builtin_processing_templates
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


def get_run_data_path(project_path: Path, survey: str, station: str, run: str) -> Path:
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
    """Get the project calibration-data directory."""
    return project_path / "calibrate"


def get_meas_time_path(project_path: Path, site_name: str, meas_name: str) -> Path:
    """Get the legacy time-data directory for a measurement."""
    return project_path / "time" / site_name / meas_name


def get_meas_spectra_path(
    project_path: Path, site_name: str, meas_name: str, config_name: str
) -> Path:
    """Get the legacy spectra-data directory for a measurement."""
    return project_path / "spectra" / site_name / config_name / meas_name


def get_meas_evals_path(
    project_path: Path, site_name: str, meas_name: str, config_name: str
) -> Path:
    """Get the legacy evaluation-spectra directory for a measurement."""
    return project_path / "evals" / site_name / config_name / meas_name


def get_meas_features_path(
    project_path: Path, site_name: str, meas_name: str, config_name: str
) -> Path:
    """Get the legacy feature-data directory for a measurement."""
    return project_path / "features" / site_name / config_name / meas_name


def get_mask_path(project_path: Path, site_name: str, config_name: str) -> Path:
    """Get the legacy mask-data directory for a site configuration."""
    return project_path / "masks" / site_name / config_name


def get_mask_name(fs: float, mask_name: str) -> str:
    """Get a sampling-rate-specific mask file name."""
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


class MTH5FileSummary(ResisticsModel):
    """Cheap, serializable summary of an MTH5 file."""

    mth5_path: Path
    file_version: str
    n_surveys: int
    n_stations: int
    n_runs: int
    n_channels: int
    sample_rates: List[float] = Field(default_factory=list)
    start_time: Optional[str] = None
    end_time: Optional[str] = None


class SurveySummary(ResisticsModel):
    survey: str
    n_stations: int
    n_runs: int


class StationSummary(ResisticsModel):
    survey: str
    station: str
    station_path: str
    n_runs: int
    sample_rates: List[float] = Field(default_factory=list)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    elevation: Optional[float] = None


class RunSummary(ResisticsModel):
    survey: str
    station: str
    run: str
    run_path: str
    sample_rate: float
    n_samples: int
    channels: List[str] = Field(default_factory=list)
    start_time: str
    end_time: str
    has_data: bool = True


class ChannelSummary(ResisticsModel):
    survey: str
    station: str
    run: str
    component: str
    sample_rate: float
    n_samples: int
    start_time: str
    end_time: str
    measurement_type: str = ""
    units: str = ""
    has_data: bool = True


class MetadataDetail(ResisticsModel):
    object_type: Literal["survey", "station", "run", "channel"]
    object_path: str
    values: Dict[str, JsonValue] = Field(default_factory=dict)


class _MTH5InspectionMixin:
    """Shared app-safe inspection behavior for files and projects."""

    mth5_path: Path
    mth5_data: MTH5
    table: pd.DataFrame

    def file_summary(self) -> MTH5FileSummary:
        table = self.table
        return MTH5FileSummary(
            mth5_path=self.mth5_path,
            file_version=str(self.mth5_data.file_version),
            n_surveys=int(table["survey"].nunique()) if not table.empty else 0,
            n_stations=int(table["station_path"].nunique()) if not table.empty else 0,
            n_runs=int(table["run_path"].nunique()) if not table.empty else 0,
            n_channels=len(table.index),
            sample_rates=self.fs(),
            start_time=_iso_min(table, "start"),
            end_time=_iso_max(table, "end"),
        )

    def list_surveys(self) -> List[SurveySummary]:
        ans = []
        for survey, table in self.table.groupby("survey"):
            ans.append(
                SurveySummary(
                    survey=str(survey),
                    n_stations=int(table["station"].nunique()),
                    n_runs=int(table["run_path"].nunique()),
                )
            )
        return ans

    def list_stations(self, survey: Optional[str] = None) -> List[StationSummary]:
        table = self._filter_table(survey=survey)
        ans = []
        for (survey_name, station), rows in table.groupby(["survey", "station"]):
            ans.append(
                StationSummary(
                    survey=str(survey_name),
                    station=str(station),
                    station_path=f"{survey_name}/{station}",
                    n_runs=int(rows["run"].nunique()),
                    sample_rates=sorted(float(x) for x in rows["sample_rate"].unique()),
                    start_time=_iso_min(rows, "start"),
                    end_time=_iso_max(rows, "end"),
                    latitude=_optional_float(rows, "latitude"),
                    longitude=_optional_float(rows, "longitude"),
                    elevation=_optional_float(rows, "elevation"),
                )
            )
        return ans

    def list_runs(
        self, survey: Optional[str] = None, station: Optional[str] = None
    ) -> List[RunSummary]:
        table = self._filter_table(survey=survey, station=station)
        ans = []
        for (survey_name, station_name, run), rows in table.groupby(
            ["survey", "station", "run"]
        ):
            ans.append(
                RunSummary(
                    survey=str(survey_name),
                    station=str(station_name),
                    run=str(run),
                    run_path=f"{survey_name}/{station_name}/{run}",
                    sample_rate=float(rows["sample_rate"].iloc[0]),
                    n_samples=int(rows["n_samples"].max()),
                    channels=[str(x) for x in rows["component"].tolist()],
                    start_time=str(rows["start"].min().isoformat()),
                    end_time=str(rows["end"].max().isoformat()),
                    has_data=(
                        bool(rows["has_data"].all()) if "has_data" in rows else True
                    ),
                )
            )
        return ans

    def list_channels(
        self, survey: str, station: str, run: str
    ) -> List[ChannelSummary]:
        rows = self._filter_table(survey=survey, station=station)
        rows = rows[rows["run"] == run]
        return [
            ChannelSummary(
                survey=survey,
                station=station,
                run=run,
                component=str(row["component"]),
                sample_rate=float(row["sample_rate"]),
                n_samples=int(row["n_samples"]),
                start_time=str(row["start"].isoformat()),
                end_time=str(row["end"].isoformat()),
                measurement_type=str(row.get("measurement_type", "")),
                units=str(row.get("units", "")),
                has_data=bool(row.get("has_data", True)),
            )
            for _, row in rows.iterrows()
        ]

    def get_metadata(self, object_path: str) -> MetadataDetail:
        parts = object_path.split("/")
        if len(parts) == 1:
            obj, kind = self.get_survey(parts[0]), "survey"
        elif len(parts) == 2:
            obj, kind = self.get_station(parts[0], parts[1]), "station"
        elif len(parts) == 3:
            obj, kind = self.get_run(parts[0], parts[1], parts[2]), "run"
        elif len(parts) == 4:
            obj = self.mth5_data.get_channel(
                parts[1], parts[2], parts[3], survey=parts[0]
            )
            kind = "channel"
        else:
            raise ValueError(f"Unknown MTH5 object path {object_path!r}")
        values = json.loads(obj.metadata.to_json())
        return MetadataDetail(object_type=kind, object_path=object_path, values=values)


class MTH5File(_MTH5InspectionMixin, ResisticsModel):
    """Explicitly opened, read-only MTH5 inspection source."""

    mth5_path: Path
    mth5_data: MTH5 = Field(repr=False, exclude=True)
    table: pd.DataFrame = Field(repr=False, exclude=True)

    def fs(self) -> List[float]:
        return sorted(float(x) for x in self.table["sample_rate"].dropna().unique())

    def get_survey(self, survey: str) -> SurveyGroup:
        return self.mth5_data.get_survey(survey)

    def get_station(self, survey: str, station: str) -> StationGroup:
        return self.mth5_data.get_station(station, survey=survey)

    def get_run(self, survey: str, station: str, run: str) -> RunGroup:
        return self.mth5_data.get_run(station, run, survey=survey)

    def read_run(self, survey: str, station: str, run: str, **kwargs: Any) -> TimeData:
        return _read_run(self, survey, station, run, **kwargs)

    def close_mth5(self) -> None:
        self.mth5_data.close_mth5()

    def _filter_table(self, survey=None, station=None, fs=None) -> pd.DataFrame:
        return _filter_table(self.table, survey, station, fs)


class Project(_MTH5InspectionMixin, ResisticsModel):
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
        """Alias for the project root used by the gathering implementation."""
        return self.project_path

    @property
    def metadata(self) -> ProjectMetadata:
        """Project metadata view used by existing processing helpers."""
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
        return self.mth5_data.get_survey(survey)

    def get_station(self, survey: str, station: str) -> StationGroup:
        """Get an MTH5 station group."""
        station_path = f"{survey}/{station}"
        if station_path not in self.stations:
            raise ValueError(f"Station {station_path!r} not found in MTH5 data")
        return self.mth5_data.get_station(station, survey=survey)

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
        return self.mth5_data.get_run(station, run, survey=survey)

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
        return _read_run(
            self,
            survey,
            station,
            run,
            chans=chans,
            from_time=from_time,
            to_time=to_time,
            from_sample=from_sample,
            to_sample=to_sample,
        )

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
    install_builtin_processing_templates(project_path)

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
    mth5_data.open_mth5(mode="r")
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


def open_mth5(mth5_path: Union[Path, str]) -> MTH5File:
    """Open an existing MTH5 file as a read-only inspection source."""
    mth5_path = _as_path(mth5_path)
    if not mth5_path.exists():
        raise ValueError(f"MTH5 data file not found: {mth5_path}")
    mth5_data = MTH5(mth5_path)
    mth5_data.open_mth5(mode="r")
    try:
        table = _prepare_channel_summary(mth5_data.channel_summary.to_dataframe())
        return MTH5File(mth5_path=mth5_path, mth5_data=mth5_data, table=table)
    except Exception:
        mth5_data.close_mth5()
        raise


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
    table["station_path"] = (
        table[["survey", "station"]].astype(str).agg("/".join, axis=1)
    )
    table["run_path"] = (
        table[["survey", "station", "run"]].astype(str).agg("/".join, axis=1)
    )
    if "start" in table.columns:
        table["start"] = pd.to_datetime(table["start"])
    if "end" in table.columns:
        table["end"] = pd.to_datetime(table["end"])
    return table


def _filter_table(
    table: pd.DataFrame,
    survey: Optional[str] = None,
    station: Optional[str] = None,
    fs: Optional[float] = None,
) -> pd.DataFrame:
    table = table.copy()
    if survey is not None:
        table = table[table["survey"] == survey]
    if station is not None:
        table = table[table["station"] == station]
    if fs is not None:
        table = table[table["sample_rate"] == fs]
    return table


def _optional_float(table: pd.DataFrame, column: str) -> Optional[float]:
    if column not in table or table[column].dropna().empty:
        return None
    return float(table[column].dropna().iloc[0])


def _iso_min(table: pd.DataFrame, column: str) -> Optional[str]:
    if table.empty or column not in table:
        return None
    value = table[column].min()
    return None if pd.isna(value) else str(value.isoformat())


def _iso_max(table: pd.DataFrame, column: str) -> Optional[str]:
    if table.empty or column not in table:
        return None
    value = table[column].max()
    return None if pd.isna(value) else str(value.isoformat())


def _read_run(
    source: _MTH5InspectionMixin,
    survey: str,
    station: str,
    run: str,
    chans: Optional[Iterable[str]] = None,
    from_time: Optional[DateTimeLike] = None,
    to_time: Optional[DateTimeLike] = None,
    from_sample: Optional[int] = None,
    to_sample: Optional[int] = None,
) -> TimeData:
    """Read a bounded MTH5 run using MTH5's RunTS slicing API."""
    run_group = source.get_run(survey, station, run)
    start = None if from_time is None else str(to_timestamp(from_time).isoformat())
    end = None if to_time is None else str(to_timestamp(to_time).isoformat())
    n_samples = None
    if from_sample is not None or to_sample is not None:
        summary = source.list_runs(survey=survey, station=station)
        selected = next(item for item in summary if item.run == run)
        first = 0 if from_sample is None else from_sample
        start_time = pd.Timestamp(selected.start_time) + pd.to_timedelta(
            first / selected.sample_rate, unit="s"
        )
        start = start_time.isoformat()
        if to_sample is not None:
            n_samples = max(0, to_sample - first + 1)
    run_ts = run_group.to_runts(start=start, end=end, n_samples=n_samples)
    return _run_ts_to_time_data(run_ts, chans=chans)


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
        chan: ChanMetadata(
            name=chan,
            data_files=None,
            chan_type=(
                "electric"
                if chan.lower().startswith("e")
                else "magnetic" if chan.lower().startswith(("h", "b")) else "unknown"
            ),
        )
        for chan in chans
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


# The full gathering module still uses these type names while its MTH5
# remote-reference adapter is being built out.
Measurement = None
Site = None
