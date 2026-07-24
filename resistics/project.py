"""MTH5-backed resistics project model and canonical artifact-path helpers.

The project API uses MTH5 surveys, stations, and runs as its sole source-data
hierarchy. Derived processing artifacts remain in the project's ``data`` tree.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from shutil import rmtree
from typing import TYPE_CHECKING, Any, Literal, cast

import h5py
import pandas as pd
import plotly.graph_objects as go
from loguru import logger
from pydantic import Field, JsonValue

from resistics.common import ResisticsModel, validate_output_label
from resistics.plot import plot_timeline
from resistics.project_mth5 import (
    _close_failed_mth5,
    _MTH5Handle,
    _MTH5HandleOwner,
    _open_read_only_mth5,
)
from resistics.sampling import DateTimeLike, HighResDateTime, to_datetime, to_timestamp
from resistics.templates import install_builtin_processing_templates
from resistics.time import MTH5TimeReader, TimeData

if TYPE_CHECKING:
    from mth5.groups.run import RunGroup
    from mth5.groups.station import StationGroup
    from mth5.groups.survey import SurveyGroup

PROJ_FILE = "resistics.json"
CANONICAL_PROJ_DIRS = (
    "processing",
    "processing/flows",
    "processing/parameters",
    "processing/criteria",
    "processing/jobs",
    "data",
    "logs",
    "plugins",
)


def _as_path(value: Path | str) -> Path:
    """Return a path from a string or path-like value.

    :param value: Value to validate or normalize.
    :return: A path from a string or path-like value.
    """
    return value if isinstance(value, Path) else Path(value)


def get_flow_path(project_path: Path, flow_name: str) -> Path:
    """Get path to a flow definition.

    :param project_path: Project root used to locate configuration and artifacts.
    :param flow_name: Flow name used by this operation.
    :return: Get path to a flow definition.
    """
    return project_path / "processing" / "flows" / flow_name


def get_parameters_path(project_path: Path, parameters_name: str) -> Path:
    """Get path to a processing parameters definition.

    :param project_path: Project root used to locate configuration and artifacts.
    :param parameters_name: Parameters name used by this operation.
    :return: Get path to a processing parameters definition.
    """
    return project_path / "processing" / "parameters" / parameters_name


def get_job_path(project_path: Path, job_name: str) -> Path:
    """Get path to a processing job definition.

    :param project_path: Project root used to locate configuration and artifacts.
    :param job_name: Job name used by this operation.
    :return: Get path to a processing job definition.
    """
    return project_path / "processing" / "jobs" / job_name


def get_run_data_path(project_path: Path, survey: str, station: str, run: str) -> Path:
    """Get path to derived artifacts for an MTH5 run.

    :param project_path: Project root used to locate configuration and artifacts.
    :param survey: Survey identifier or optional survey filter.
    :param station: Station identifier or optional station filter.
    :param run: Run identifier.
    :return: Get path to derived artifacts for an MTH5 run.
    """
    return project_path / "data" / survey / station / run


def get_results_path(
    project_path: Path, survey: str, station: str, output_label: str
) -> Path:
    """Get the canonical output-label path for a processing job.

    :param project_path: Resistics project root.
    :param survey: MTH5 survey identifier.
    :param station: MTH5 station identifier.
    :param output_label: Processing output namespace.

    :return: Station result directory beneath the canonical project data tree.
    """
    return project_path / "data" / survey / station / "results" / output_label


def get_log_path(project_path: Path, job_name: str) -> Path:
    """Get path to a processing-job log file.

    :param project_path: Project root used to locate configuration and artifacts.
    :param job_name: Job name used by this operation.
    :return: Get path to a processing-job log file.
    """
    return project_path / "logs" / f"{job_name}.log"


class ProjectMetadata(ResisticsModel):
    """Serializable MTH5 project metadata stored in ``resistics.json``."""

    mth5_path: Path
    ref_time: HighResDateTime


class MTH5FileSummary(ResisticsModel):
    """Cheap, serializable summary of an MTH5 file."""

    mth5_path: Path
    file_version: str
    n_surveys: int
    n_stations: int
    n_runs: int
    n_channels: int
    sample_rates: list[float] = Field(default_factory=list)
    start_time: str | None = None
    end_time: str | None = None


class SurveySummary(ResisticsModel):
    """Counts describing one survey in an MTH5 file."""

    survey: str
    n_stations: int
    n_runs: int


class StationSummary(ResisticsModel):
    """Metadata-only summary of one station and its available data."""

    survey: str
    station: str
    station_path: str
    n_runs: int
    sample_rates: list[float] = Field(default_factory=list)
    start_time: str | None = None
    end_time: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    elevation: float | None = None


class RunSummary(ResisticsModel):
    """Metadata-only summary of one recording run."""

    survey: str
    station: str
    run: str
    run_path: str
    sample_rate: float
    n_samples: int
    channels: list[str] = Field(default_factory=list)
    start_time: str
    end_time: str
    has_data: bool = True


class ChannelSummary(ResisticsModel):
    """Metadata-only summary of one recorded channel."""

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
    """Serialized metadata for one selected MTH5 object."""

    object_type: Literal["survey", "station", "run", "channel"]
    object_path: str
    values: dict[str, JsonValue] = Field(default_factory=dict)


DataSource = Literal["project", "mth5"]
DataType = Literal["time", "spectra", "mask", "transfer_function", "other"]


class ProjectDataItem(ResisticsModel):
    """One metadata-only entry in a project's Data browser."""

    source: DataSource
    path: str
    parent_path: str | None = None
    name: str
    kind: Literal["directory", "file", "group", "dataset"]
    data_type: DataType
    is_dataset: bool = False
    """Whether this item is one logical saved dataset rather than tree chrome."""


class ProjectDataMetadata(ResisticsModel):
    """Metadata displayed for one item in a project's Data browser."""

    source: DataSource
    path: str
    kind: Literal["directory", "file", "group", "dataset"]
    data_type: DataType
    values: dict[str, JsonValue] = Field(default_factory=dict)


class ProjectDataDeletion(ResisticsModel):
    """Preview or summary of a derived-project-data deletion."""

    output_label: str | None = None
    paths: list[str] = Field(default_factory=list)

    @property
    def count(self) -> int:
        """Return the number of top-level artifact paths affected."""
        return len(self.paths)


def _metadata_value(value: Any) -> JsonValue:
    """Convert HDF5 and filesystem metadata into a JSON-safe value.

    :param value: Value to validate or normalize.
    :return: The value produced when this operation completes.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): _metadata_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _metadata_value(value.tolist())
    return str(value)


def _mth5_data_type(path: str) -> DataType:
    """Classify known MTH5 dataset locations, preserving unknown entries.

    :param path: Path or routed coordinates to process.
    :return: Classify known MTH5 dataset locations, preserving unknown entries.
    """
    value = path.lower()
    if "transfer" in value or "tf_summary" in value:
        return "transfer_function"
    if "fourier" in value or "fc_summary" in value or "eval" in value:
        return "spectra"
    parts = [part for part in value.split("/") if part]
    if "stations" in parts:
        after_station = parts[parts.index("stations") + 2 :]
        if after_station and after_station[0] == "runs":
            after_station = after_station[1:]
        if after_station and after_station[0] not in {
            "features",
            "fourier_coefficients",
            "transfer_functions",
        }:
            return "time"
    return "other"


def _project_data_type(path: Path) -> DataType:
    """Classify project artifacts using their stable storage conventions.

    :param path: Path or routed coordinates to process.
    :return: Classify project artifacts using their stable storage conventions.
    """
    parts = {part.lower() for part in path.parts}
    if "masks" in parts:
        return "mask"
    if "evals" in parts:
        return "spectra"
    if path.name == "solution.json" or (
        path.is_dir() and (path / "solution.json").is_file()
    ):
        return "transfer_function"
    return "other"


class _MTH5InspectionMixin(_MTH5HandleOwner):
    """Shared inspection and owned-handle lifecycle for files and projects."""

    mth5_path: Path
    mth5_data: _MTH5Handle
    table: pd.DataFrame

    def fs(self) -> list[float]:
        """Return the distinct sampling frequencies in the inspection source.

        :return: Sampling frequencies represented by the source.

        :raises NotImplementedError: If a concrete inspection source does not implement the contract.
        """
        raise NotImplementedError

    def get_survey(self, survey: str) -> SurveyGroup:
        """Return a survey group by identifier.

        :param survey: Survey identifier.

        :return: Matching MTH5 survey group.

        :raises NotImplementedError: If a concrete inspection source does not implement the contract.
        """
        raise NotImplementedError

    def get_station(self, survey: str, station: str) -> StationGroup:
        """Return a station group within a survey.

        :param survey: Survey identifier.
        :param station: Station identifier.

        :return: Matching MTH5 station group.

        :raises NotImplementedError: If a concrete inspection source does not implement the contract.
        """
        raise NotImplementedError

    def get_run(self, survey: str, station: str, run: str) -> RunGroup:
        """Return a run group within a survey and station.

        :param survey: Survey identifier.
        :param station: Station identifier.
        :param run: Run identifier.

        :return: Matching MTH5 run group.

        :raises NotImplementedError: If a concrete inspection source does not implement the contract.
        """
        raise NotImplementedError

    def _filter_table(
        self,
        survey: str | None = None,
        station: str | None = None,
        fs: float | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def file_summary(self) -> MTH5FileSummary:
        self._require_open()
        table = self.table
        return MTH5FileSummary(
            mth5_path=self.mth5_path,
            file_version=str(self.mth5_data.file_version),
            n_surveys=table["survey"].nunique() if not table.empty else 0,
            n_stations=table["station_path"].nunique() if not table.empty else 0,
            n_runs=table["run_path"].nunique() if not table.empty else 0,
            n_channels=len(table.index),
            sample_rates=self.fs(),
            start_time=_iso_min(table, "start"),
            end_time=_iso_max(table, "end"),
        )

    def list_surveys(self) -> list[SurveySummary]:
        ans = []
        for survey, table in self.table.groupby("survey"):
            ans.append(
                SurveySummary(
                    survey=str(survey),
                    n_stations=table["station"].nunique(),
                    n_runs=table["run_path"].nunique(),
                )
            )
        return ans

    def list_stations(self, survey: str | None = None) -> list[StationSummary]:
        table = self._filter_table(survey=survey)
        ans = []
        for (survey_name, station), rows in table.groupby(["survey", "station"]):
            ans.append(
                StationSummary(
                    survey=str(survey_name),
                    station=str(station),
                    station_path=f"{survey_name}/{station}",
                    n_runs=rows["run"].nunique(),
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
        self, survey: str | None = None, station: str | None = None
    ) -> list[RunSummary]:
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
    ) -> list[ChannelSummary]:
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
        self._require_open()
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
    """Owned, read-only MTH5 inspection source supporting ``with`` and close."""

    mth5_path: Path
    mth5_data: _MTH5Handle = Field(repr=False, exclude=True)
    table: pd.DataFrame = Field(repr=False, exclude=True)

    def fs(self) -> list[float]:
        """Return the distinct sampling frequencies present in the file.

        :return: The distinct sampling frequencies present in the file.
        """
        return sorted(float(x) for x in self.table["sample_rate"].dropna().unique())

    def get_survey(self, survey: str) -> SurveyGroup:
        """Return a survey group by identifier.

        :param survey: Survey identifier or optional survey filter.
        :return: A survey group by identifier.
        """
        self._require_open()
        return self.mth5_data.get_survey(survey)

    def get_station(self, survey: str, station: str) -> StationGroup:
        """Return a station group within a survey.

        :param survey: Survey identifier or optional survey filter.
        :param station: Station identifier or optional station filter.
        :return: A station group within a survey.
        """
        self._require_open()
        return self.mth5_data.get_station(station, survey=survey)

    def get_run(self, survey: str, station: str, run: str) -> RunGroup:
        """Return a run group within a survey and station.

        :param survey: Survey identifier or optional survey filter.
        :param station: Station identifier or optional station filter.
        :param run: Run identifier.
        :return: A run group within a survey and station.
        """
        self._require_open()
        return self.mth5_data.get_run(station, run, survey=survey)

    def read_run(self, survey: str, station: str, run: str, **kwargs: Any) -> TimeData:
        """Read one run as resistics time data.

        :param survey: Survey identifier or optional survey filter.
        :param station: Station identifier or optional station filter.
        :param run: Run identifier.
        :param **kwargs: Kwargs used by this operation.
        :return: The value produced when this operation completes.
        """
        return _read_run(self, survey, station, run, **kwargs)

    def _filter_table(
        self,
        survey: str | None = None,
        station: str | None = None,
        fs: float | None = None,
    ) -> pd.DataFrame:
        return _filter_table(self.table, survey, station, fs)


class Project(_MTH5InspectionMixin, ResisticsModel):
    """An MTH5-backed project owning one read-only handle until closed."""

    project_path: Path
    mth5_path: Path
    ref_time: HighResDateTime
    mth5_data: _MTH5Handle = Field(repr=False, exclude=True)
    table: pd.DataFrame = Field(repr=False, exclude=True)
    surveys: list[str] = Field(default_factory=list)
    stations: list[str] = Field(default_factory=list)
    runs: list[str] = Field(default_factory=list)

    def list_mth5_data_items(self) -> list[ProjectDataItem]:
        """List the MTH5 group and dataset hierarchy without reading data values.

        :return: List the MTH5 group and dataset hierarchy without reading data values.
        """
        items = []
        with h5py.File(self.mth5_path, "r") as mth5_file:

            def add_item(name: str, value: h5py.Group | h5py.Dataset) -> None:
                path = f"/{name}"
                parent = "/" if "/" not in name else f"/{name.rsplit('/', 1)[0]}"
                items.append(
                    ProjectDataItem(
                        source="mth5",
                        path=path,
                        parent_path=parent,
                        name=name.rsplit("/", 1)[-1],
                        kind="group" if isinstance(value, h5py.Group) else "dataset",
                        data_type=_mth5_data_type(path),
                        is_dataset=isinstance(value, h5py.Dataset),
                    )
                )

            mth5_file.visititems(add_item)
        return sorted(items, key=lambda item: item.path)

    def get_mth5_data_metadata(self, path: str) -> ProjectDataMetadata:
        """Return serializable metadata for one MTH5 group or dataset.

        :param path: Path or routed coordinates to process.
        :return: Serializable metadata for one MTH5 group or dataset.
        """
        with h5py.File(self.mth5_path, "r") as mth5_file:
            value = mth5_file[path]
            kind = "group" if isinstance(value, h5py.Group) else "dataset"
            values: dict[str, JsonValue] = {
                "attributes": {
                    str(key): _metadata_value(item) for key, item in value.attrs.items()
                }
            }
            if isinstance(value, h5py.Dataset):
                values.update(
                    {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "size_bytes": int(value.nbytes),
                        "chunks": None if value.chunks is None else list(value.chunks),
                        "compression": value.compression,
                    }
                )
            return ProjectDataMetadata(
                source="mth5",
                path=path,
                kind=kind,
                data_type=_mth5_data_type(path),
                values=values,
            )

    @property
    def _data_path(self) -> Path:
        """Return the project data root used by generated artifacts."""
        return self.project_path / "data"

    def list_project_data_items(self) -> list[ProjectDataItem]:
        """List saved project artifacts without opening their data payloads.

        :return: List saved project artifacts without opening their data payloads.
        """
        data_path = self._data_path
        if not data_path.is_dir():
            return []
        artifact_roots = self._project_artifact_roots()
        items = []
        for item_path in sorted(data_path.rglob("*")):
            relative = item_path.relative_to(data_path)
            path = relative.as_posix()
            parent = (
                None if relative.parent == Path(".") else relative.parent.as_posix()
            )
            items.append(
                ProjectDataItem(
                    source="project",
                    path=path,
                    parent_path=parent,
                    name=item_path.name,
                    kind="directory" if item_path.is_dir() else "file",
                    data_type=_project_data_type(item_path),
                    is_dataset=self._is_project_dataset_item(item_path, artifact_roots),
                )
            )
        return items

    def _project_artifact_roots(self) -> dict[Path, DataType]:
        """Return the directories that each represent one saved artifact.

        :return: The directories that each represent one saved artifact.
        """
        data_path = self._data_path
        if not data_path.is_dir():
            return {}
        roots: dict[Path, DataType] = {}
        for item_path in data_path.rglob("*"):
            if not item_path.is_dir():
                continue
            data_type = _project_data_type(item_path)
            if (
                (data_type == "spectra" and item_path.parent.name == "evals")
                or (
                    data_type == "mask"
                    and (item_path / "metadata.json").is_file()
                    and (item_path / "data.npz").is_file()
                )
                or (
                    data_type == "transfer_function"
                    and (item_path / "solution.json").is_file()
                )
            ):
                roots[item_path] = data_type
        return roots

    @staticmethod
    def _is_project_dataset_item(
        item_path: Path, artifact_roots: dict[Path, DataType]
    ) -> bool:
        """Identify artifact roots and standalone unrecognised data files.

        :param item_path: Item path used by this operation.
        :param artifact_roots: Artifact roots used by this operation.
        :return: Identify artifact roots and standalone unrecognised data files.
        """
        if item_path in artifact_roots:
            return True
        if not item_path.is_file():
            return False
        return not any(root in item_path.parents for root in artifact_roots)

    def get_project_data_metadata(self, path: str) -> ProjectDataMetadata:
        """Return generic and recognized metadata for one saved project artifact.

        :param path: Path or routed coordinates to process.
        :return: Generic and recognized metadata for one saved project artifact.
        """
        item_path = self._project_data_item_path(path)
        kind = "directory" if item_path.is_dir() else "file"
        values: dict[str, JsonValue] = {
            "size_bytes": None if item_path.is_dir() else item_path.stat().st_size,
        }
        values.update(self._project_data_descriptors(item_path))
        return ProjectDataMetadata(
            source="project",
            path=path,
            kind=kind,
            data_type=_project_data_type(item_path),
            values=values,
        )

    def get_project_data_json(self, path: str) -> JsonValue:
        """Read one JSON artifact selected through the project-data browser.

        :param path: Path or routed coordinates to process.
        :return: The value produced when this operation completes.
        :raises ValueError: If the requested operation cannot satisfy its contract.
        """
        item_path = self._project_data_item_path(path)
        if not item_path.is_file() or item_path.suffix.lower() != ".json":
            raise ValueError("Project data item is not a JSON file")
        try:
            return _metadata_value(json.loads(item_path.read_text(encoding="utf-8")))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read JSON file {path}: {exc}") from exc

    def list_project_output_labels(self) -> list[str]:
        """List output labels represented by recognised derived artifacts.

        :return: List output labels represented by recognised derived artifacts.
        """
        labels = set()
        for path, data_type in self._project_artifact_roots().items():
            if data_type == "spectra" and path.parent.name == "evals":
                labels.add(path.name)
            elif (data_type == "mask" and path.parent.parent.name == "masks") or (
                data_type == "transfer_function"
                and path.parent.parent.name == "results"
            ):
                labels.add(path.parent.name)
        return sorted(labels)

    def preview_project_data_deletion(
        self, output_label: str | None = None
    ) -> ProjectDataDeletion:
        """Return the derived paths that a labelled or complete clear removes.

        :param output_label: Artifact namespace containing or receiving the data.
        :return: The derived paths that a labelled or complete clear removes.
        """
        data_path = self._data_path
        if output_label is None:
            paths = (
                []
                if not data_path.is_dir()
                else [
                    path.relative_to(data_path).as_posix()
                    for path in sorted(data_path.iterdir())
                ]
            )
            return ProjectDataDeletion(paths=paths)

        output_label = validate_output_label(output_label)
        targets = set()
        for path, data_type in self._project_artifact_roots().items():
            if data_type == "spectra" and path.parent.name == "evals":
                if path.name == output_label:
                    targets.add(path)
            elif (
                (data_type == "mask" and path.parent.parent.name == "masks")
                or (
                    data_type == "transfer_function"
                    and path.parent.parent.name == "results"
                )
            ) and path.parent.name == output_label:
                targets.add(path.parent)
        return ProjectDataDeletion(
            output_label=output_label,
            paths=sorted(path.relative_to(data_path).as_posix() for path in targets),
        )

    def delete_project_data(
        self, output_label: str | None = None
    ) -> ProjectDataDeletion:
        """Delete one output-label namespace or all derived project data.

                The project's MTH5 file is protected even when it happens to be stored
                beneath ``project/data``.

        :param output_label: Artifact namespace containing or receiving the data.
        :return: Delete one output-label namespace or all derived project data.
        """
        deletion = self.preview_project_data_deletion(output_label)
        data_path = self._data_path
        if not data_path.is_dir():
            return deletion
        if output_label is None:
            protected = self.mth5_path.resolve()
            for path in list(data_path.iterdir()):
                self._delete_data_path(path, protected)
            return deletion

        for relative in deletion.paths:
            path = data_path / relative
            if path.exists():
                self._remove_data_path(path)
                self._prune_empty_data_parents(path.parent)
        return deletion

    @staticmethod
    def _remove_data_path(path: Path) -> None:
        """Remove one known derived artifact without following a symlink.

        :param path: Path or routed coordinates to process.
        """
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            rmtree(path)

    def _delete_data_path(self, path: Path, protected: Path) -> None:
        """Clear data recursively while retaining an in-tree MTH5 source.

        :param path: Path or routed coordinates to process.
        :param protected: Protected used by this operation.
        """
        resolved = path.resolve()
        if resolved == protected:
            return
        if path.is_dir() and not path.is_symlink() and resolved in protected.parents:
            for child in list(path.iterdir()):
                self._delete_data_path(child, protected)
            return
        self._remove_data_path(path)

    def _prune_empty_data_parents(self, path: Path) -> None:
        """Remove empty structural directories left by a labelled deletion.

        :param path: Path or routed coordinates to process.
        """
        data_path = self._data_path
        while path != data_path and path.is_dir():
            try:
                path.rmdir()
            except OSError:
                break
            path = path.parent

    def _project_data_item_path(self, path: str) -> Path:
        """Resolve a browser path and keep it within the project data root.

        :param path: Path or routed coordinates to process.
        :return: The value produced when this operation completes.
        :raises ValueError: If the requested operation cannot satisfy its contract.
        """
        data_path = self._data_path.resolve()
        item_path = (data_path / path).resolve()
        if item_path != data_path and data_path not in item_path.parents:
            raise ValueError("Project data path must be inside project/data")
        if not item_path.exists():
            raise ValueError(f"Project data item not found: {path}")
        return item_path

    @staticmethod
    def _project_data_descriptors(item_path: Path) -> dict[str, JsonValue]:
        """Read recognized, small JSON descriptors associated with an artifact.

        :param item_path: Item path used by this operation.
        :return: The value produced when this operation completes.
        """
        descriptors = (
            [item_path / "metadata.json", item_path / "job_info.json"]
            if item_path.is_dir()
            else (
                [item_path]
                if item_path.name in {"metadata.json", "job_info.json"}
                else []
            )
        )
        values = {}
        for descriptor in (path for path in descriptors if path.is_file()):
            try:
                values[descriptor.name] = _metadata_value(
                    json.loads(descriptor.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                values[f"{descriptor.name}_error"] = str(exc)
        return values

    def __getitem__(self, obj_path: str) -> SurveyGroup | StationGroup | RunGroup:
        """Get an MTH5 survey, station, or run by slash-separated path.

        :param obj_path: Obj path used by this operation.
        :return: Get an MTH5 survey, station, or run by slash-separated path.
        :raises ValueError: If the requested operation cannot satisfy its contract.
        """
        parts = obj_path.split("/")
        if len(parts) == 1:
            return self.get_survey(parts[0])
        if len(parts) == 2:
            return self.get_station(parts[0], parts[1])
        if len(parts) == 3:
            return self.get_run(parts[0], parts[1], parts[2])
        raise ValueError(f"Unknown MTH5 object path {obj_path!r}")

    def n_surveys(self) -> int:
        """Get the number of surveys.

        :return: Get the number of surveys.
        """
        return len(self.surveys)

    def n_stations(self, survey: str | None = None) -> int:
        """Get the number of stations, optionally filtered by survey.

        :param survey: Survey identifier or optional survey filter.
        :return: Get the number of stations, optionally filtered by survey.
        """
        if survey is None:
            return len(self.stations)
        return len(self.get_stations(survey=survey))

    def fs(self) -> list[float]:
        """Get project sample rates.

        :return: Get project sample rates.
        """
        if self.table.empty:
            return []
        return sorted([float(x) for x in self.table["sample_rate"].dropna().unique()])

    def start(self) -> pd.Timestamp:
        """Get the first project timestamp.

        :return: Get the first project timestamp.
        """
        return self.table["start"].min()

    def end(self) -> pd.Timestamp:
        """Get the last project timestamp.

        :return: Get the last project timestamp.
        """
        return self.table["end"].max()

    def get_survey(self, survey: str) -> SurveyGroup:
        """Get an MTH5 survey group.

        :param survey: Survey identifier or optional survey filter.
        :return: Get an MTH5 survey group.
        :raises ValueError: If the requested operation cannot satisfy its contract.
        """
        self._require_open()
        if survey not in self.surveys:
            raise ValueError(f"Survey {survey!r} not found in MTH5 data")
        return self.mth5_data.get_survey(survey)

    def get_station(self, survey: str, station: str) -> StationGroup:
        """Get an MTH5 station group.

        :param survey: Survey identifier or optional survey filter.
        :param station: Station identifier or optional station filter.
        :return: Get an MTH5 station group.
        :raises ValueError: If the requested operation cannot satisfy its contract.
        """
        self._require_open()
        station_path = f"{survey}/{station}"
        if station_path not in self.stations:
            raise ValueError(f"Station {station_path!r} not found in MTH5 data")
        return self.mth5_data.get_station(station, survey=survey)

    def get_stations(
        self, survey: str | None = None, fs: float | None = None
    ) -> dict[str, StationGroup]:
        """Get station groups keyed by ``survey/station``.

        :param survey: Survey identifier or optional survey filter.
        :param fs: Optional sample rate used to filter or describe the data.
        :return: Get station groups keyed by ``survey/station``.
        """
        table = self._filter_table(survey=survey, fs=fs)
        table = table.drop_duplicates(subset=["station_path"], keep="first")
        return {
            row["station_path"]: self.get_station(row["survey"], row["station"])
            for _, row in table.iterrows()
        }

    def get_run(self, survey: str, station: str, run: str) -> RunGroup:
        """Get an MTH5 run group.

        :param survey: Survey identifier or optional survey filter.
        :param station: Station identifier or optional station filter.
        :param run: Run identifier.
        :return: Get an MTH5 run group.
        :raises ValueError: If the requested operation cannot satisfy its contract.
        """
        self._require_open()
        run_path = f"{survey}/{station}/{run}"
        if run_path not in self.runs:
            raise ValueError(f"Run {run_path!r} not found in MTH5 data")
        return self.mth5_data.get_run(station, run, survey=survey)

    def get_runs(
        self,
        survey: str | None = None,
        station: str | None = None,
        fs: float | None = None,
    ) -> dict[str, RunGroup]:
        """Get run groups keyed by ``survey/station/run``.

        :param survey: Survey identifier or optional survey filter.
        :param station: Station identifier or optional station filter.
        :param fs: Optional sample rate used to filter or describe the data.
        :return: Get run groups keyed by ``survey/station/run``.
        """
        table = self._filter_table(survey=survey, station=station, fs=fs)
        table = table.drop_duplicates(subset=["run_path"], keep="first")
        return {
            row["run_path"]: self.get_run(row["survey"], row["station"], row["run"])
            for _, row in table.iterrows()
        }

    def get_concurrent(self, station_path: str, fs: float | None = None) -> list[str]:
        """Find station paths that overlap in time with ``station_path``.

        :param station_path: Canonical survey/station path.
        :param fs: Optional sample rate used to filter or describe the data.
        :return: The value produced when this operation completes.
        :raises ValueError: If the requested operation cannot satisfy its contract.
        """
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
        chans: Iterable[str] | None = None,
        from_time: DateTimeLike | None = None,
        to_time: DateTimeLike | None = None,
        from_sample: int | None = None,
        to_sample: int | None = None,
    ) -> TimeData:
        """Read an MTH5 run into existing resistics ``TimeData`` containers.

        :param survey: Survey identifier or optional survey filter.
        :param station: Station identifier or optional station filter.
        :param run: Run identifier.
        :param chans: Channels to include, or all available channels when omitted.
        :param from_time: Optional inclusive start time.
        :param to_time: Optional inclusive end time.
        :param from_sample: Optional inclusive first sample index.
        :param to_sample: Optional inclusive final sample index.
        :return: The value produced when this operation completes.
        """
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
        """Return the project MTH5 channel summary table.

        :return: The project MTH5 channel summary table.
        """
        return self.table.copy()

    def plot(self) -> go.Figure:
        """Plot project run timelines.

        :return: Plot project run timelines.
        :raises ValueError: If the requested operation cannot satisfy its contract.
        """
        if self.table.empty:
            raise ValueError("No runs found to plot")
        runs_table = self.table.drop_duplicates(subset=["run_path"]).copy()
        runs_table["sample_rate"] = runs_table["sample_rate"].astype(str)
        ref_time = to_timestamp(self.ref_time)
        return plot_timeline(runs_table, y_col="station_path", ref_time=ref_time)

    def _filter_table(
        self,
        survey: str | None = None,
        station: str | None = None,
        fs: float | None = None,
    ) -> pd.DataFrame:
        """Filter the cached MTH5 summary table.

        :param survey: Survey identifier or optional survey filter.
        :param station: Station identifier or optional station filter.
        :param fs: Optional sample rate used to filter or describe the data.
        :return: Filter the cached MTH5 summary table.
        """
        table = cast(pd.DataFrame, self.table.copy())
        if survey is not None:
            table = cast(pd.DataFrame, table.loc[table["survey"] == survey])
        if station is not None:
            table = cast(pd.DataFrame, table.loc[table["station"] == station])
        if fs is not None:
            table = cast(pd.DataFrame, table.loc[table["sample_rate"] == fs])
        return table


def init(
    project_path: Path | str,
    mth5_path: Path | str,
    ref_time: DateTimeLike,
    overwrite: bool = False,
) -> bool:
    """Initialise an MTH5-backed resistics project.

        :param project_path: Directory in which to create the project structure.
        :param mth5_path: Existing MTH5 file used as the project's read-only source data.
        :param ref_time: Reference time used for sample and window calculations.
        :param overwrite: Replace existing project metadata when ``True``.
        :return: ``True`` after the project structure and metadata are created.

        **Examples**

        Initialise a project from an existing MTH5 file, then load it for use.

        ```{doctest}
        >>> from resistics.project import init, load
        >>> init("example-project", "recordings.mth5", "2020-01-01")  # doctest: +SKIP
        True
        >>> project = load("example-project")  # doctest: +SKIP

        ```

    :param project_path: Project root used to locate configuration and artifacts.
    :param mth5_path: Path to the MTH5 file.
    :param ref_time: Ref time used by this operation.
    :param overwrite: Overwrite used by this operation.
    :return: Initialise an MTH5-backed resistics project.
    :raises ValueError: If the requested operation cannot satisfy its contract.
    """
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
    )
    metadata_path.write_text(metadata.model_dump_json())
    logger.info(f"Project created in {project_path}")
    return True


def load(project_path: Path | str) -> Project:
    """Load an MTH5-backed project that owns its handle until ``close()``.

    :param project_path: Existing canonical resistics project directory.

    :return: Open project supporting deterministic ``close()`` and ``with``.

    :raises ValueError: If project metadata, structure, or the configured MTH5 path is absent.
    :raises Exception: If metadata validation, MTH5 opening, channel-summary preparation, or
        project construction fails. An acquired handle is closed first.
    """
    project_path = _as_path(project_path)
    metadata_path = project_path / PROJ_FILE
    if not metadata_path.exists():
        raise ValueError(f"Resistics project file {metadata_path} not found")
    metadata = ProjectMetadata.model_validate_json(metadata_path.read_bytes())
    check_project(project_path, metadata.mth5_path)

    mth5_data = _open_read_only_mth5(metadata.mth5_path)
    try:
        table = _prepare_channel_summary(mth5_data.channel_summary.to_dataframe())
        return Project(
            project_path=project_path,
            mth5_path=metadata.mth5_path,
            ref_time=metadata.ref_time,
            mth5_data=mth5_data,
            table=table,
            surveys=sorted(table["survey"].dropna().unique().tolist()),
            stations=sorted(table["station_path"].dropna().unique().tolist()),
            runs=sorted(table["run_path"].dropna().unique().tolist()),
        )
    # This is the ownership-transfer boundary: until Project construction
    # succeeds, this function remains responsible for releasing the handle.
    except Exception:
        _close_failed_mth5(mth5_data, metadata.mth5_path)
        raise


def open_mth5(mth5_path: Path | str) -> MTH5File:
    """Open a read-only inspection source that owns its handle until closed.

    :param mth5_path: Existing MTH5 file to inspect.

    :return: Open inspection source supporting deterministic ``close()`` and ``with``.

    :raises ValueError: If the MTH5 path does not exist.
    :raises Exception: If MTH5 opening, channel-summary preparation, or source construction
        fails. An acquired handle is closed first.
    """
    mth5_path = _as_path(mth5_path)
    if not mth5_path.exists():
        raise ValueError(f"MTH5 data file not found: {mth5_path}")
    mth5_data = _open_read_only_mth5(mth5_path)
    try:
        table = _prepare_channel_summary(mth5_data.channel_summary.to_dataframe())
        return MTH5File(mth5_path=mth5_path, mth5_data=mth5_data, table=table)
    except Exception:
        _close_failed_mth5(mth5_data, mth5_path)
        raise


def check_project(project_path: Path | str, mth5_path: Path | str) -> bool:
    """Validate an MTH5-backed resistics project directory.

    :param project_path: Project root used to locate configuration and artifacts.
    :param mth5_path: Path to the MTH5 file.
    :return: The value produced when this operation completes.
    :raises ValueError: If the requested operation cannot satisfy its contract.
    """
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
    """Add canonical path columns to an MTH5 channel summary table.

    :param table: Table used by this operation.
    :return: Add canonical path columns to an MTH5 channel summary table.
    :raises ValueError: If the requested operation cannot satisfy its contract.
    """
    table = table.copy()
    if table.empty:
        for col in ["survey", "station", "run", "station_path", "run_path"]:
            table[col] = []
        return table
    for column in ["survey", "station", "run"]:
        if column not in table.columns:
            raise ValueError(f"MTH5 channel summary missing {column!r} column")
    surveys = table["survey"].astype(str).tolist()
    stations = table["station"].astype(str).tolist()
    runs = table["run"].astype(str).tolist()
    table["station_path"] = [
        f"{survey}/{station}" for survey, station in zip(surveys, stations, strict=True)
    ]
    table["run_path"] = [
        f"{survey}/{station}/{run}"
        for survey, station, run in zip(surveys, stations, runs, strict=True)
    ]
    if "start" in table.columns:
        table["start"] = pd.to_datetime(table["start"])
    if "end" in table.columns:
        table["end"] = pd.to_datetime(table["end"])
    return table


def _filter_table(
    table: pd.DataFrame,
    survey: str | None = None,
    station: str | None = None,
    fs: float | None = None,
) -> pd.DataFrame:
    table = cast(pd.DataFrame, table.copy())
    if survey is not None:
        table = cast(pd.DataFrame, table.loc[table["survey"] == survey])
    if station is not None:
        table = cast(pd.DataFrame, table.loc[table["station"] == station])
    if fs is not None:
        table = cast(pd.DataFrame, table.loc[table["sample_rate"] == fs])
    return table


def _optional_float(table: pd.DataFrame, column: str) -> float | None:
    if column not in table or table[column].dropna().empty:
        return None
    return float(table[column].dropna().iloc[0])


def _iso_min(table: pd.DataFrame, column: str) -> str | None:
    if table.empty or column not in table:
        return None
    value = table[column].min()
    return None if pd.isna(value) else str(value.isoformat())


def _iso_max(table: pd.DataFrame, column: str) -> str | None:
    if table.empty or column not in table:
        return None
    value = table[column].max()
    return None if pd.isna(value) else str(value.isoformat())


def _read_run(
    source: _MTH5InspectionMixin,
    survey: str,
    station: str,
    run: str,
    chans: Iterable[str] | None = None,
    from_time: DateTimeLike | None = None,
    to_time: DateTimeLike | None = None,
    from_sample: int | None = None,
    to_sample: int | None = None,
) -> TimeData:
    """Resolve an MTH5 run and delegate its reading to ``MTH5TimeReader``.

    :param source: Open MTH5-backed data source.
    :param survey: Survey identifier or optional survey filter.
    :param station: Station identifier or optional station filter.
    :param run: Run identifier.
    :param chans: Channels to include, or all available channels when omitted.
    :param from_time: Optional inclusive start time.
    :param to_time: Optional inclusive end time.
    :param from_sample: Optional inclusive first sample index.
    :param to_sample: Optional inclusive final sample index.
    :return: The value produced when this operation completes.
    """
    run_group = source.get_run(survey, station, run)
    summary = source.list_runs(survey=survey, station=station)
    selected = next(item for item in summary if item.run == run)
    return MTH5TimeReader().run(
        run_group,
        chans=None if chans is None else list(chans),
        from_time=from_time,
        to_time=to_time,
        from_sample=from_sample,
        to_sample=to_sample,
        sample_rate=selected.sample_rate,
    )
