"""
Module for gathering data that will be combined to calculate transfer functions

There are two supported scenarios. The first is quick processing outside the
project environment, where one evaluation-frequency dataset is gathered in a
single step.

- QuickGather to put together the out_data, in_data and cross_data

Inside an MTH5-backed project, gathering follows:

- ``GatherCriteria`` resolves the target, remote-reference, and mask policy.
- ``Gather`` aligns admitted windows and combines evaluation-frequency data.

.. warning::

    There may be some confusion in the code with many references to spectra data
    and evaluation-frequency data. Evaluation-frequency data, referred to below
    as ``eval_data``, is an instance of spectra data. However, it is named
    differently to highlight the fact that it is not the complete spectra data,
    but is actually spectra data at a reduced set of frequencies corresponding
    to the evaluation frequncies.

    Within a project, there are separate folders for users who want to save both
    the full spectra data with all the frequencies as well as the evaluation
    frequency spectra data with the smaller subset of frequencies. Only the
    evaluation frequency data is required to calculate the transfer function,
    but the full spectral data might be useful for visualisation and analysis
    reasons.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from loguru import logger
from pydantic import ConfigDict, Field, field_validator, model_validator

from resistics.common import (
    History,
    ResisticsData,
    ResisticsModel,
    ResisticsProcess,
    WriteableMetadata,
    validate_output_label,
)
from resistics.decimate import DecimationParameters
from resistics.mask import (
    WindowMask,
    WindowMaskReader,
    get_run_mask_path,
    validate_mask_name,
)
from resistics.project import Project
from resistics.spectra import (
    SpectraData,
    SpectraMetadata,
)
from resistics.transfunc import TransferFunction


def _cross_channels(tf: TransferFunction) -> list[str]:
    return tf.cross_chans


def _validate_station_path(value: str) -> str:
    parts = value.split("/")
    if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
        raise ValueError("Station paths must have the form 'survey/station'")
    return value


class MaskCriteria(ResisticsModel):
    """Named masks and the rule used to combine their keep decisions."""

    model_config = ConfigDict(extra="forbid")
    combine: Literal["and", "or"] = "and"
    names: list[str]

    @field_validator("names")
    @classmethod
    def validate_names(cls, values: list[str]) -> list[str]:
        if not values:
            raise ValueError("Mask criteria must name at least one mask")
        for value in values:
            validate_mask_name(value)
        if len(values) != len(set(values)):
            raise ValueError("Mask criteria cannot contain duplicate names")
        return values


class RateGatherCriteria(ResisticsModel):
    """Gather policy for one station and original sampling frequency."""

    model_config = ConfigDict(extra="forbid")
    remote_references: Literal["auto"] | list[str] | None = None
    masks: MaskCriteria | None = None

    @field_validator("remote_references")
    @classmethod
    def validate_remotes(
        cls, value: Literal["auto"] | list[str] | None
    ) -> Literal["auto"] | list[str] | None:
        if value is None or value == "auto":
            return value
        if not value:
            raise ValueError("remote_references must be 'auto' or a non-empty list")
        for station_path in value:
            _validate_station_path(station_path)
        if len(value) != len(set(value)):
            raise ValueError("remote_references cannot contain duplicates")
        return value


class StationGatherCriteria(ResisticsModel):
    """Sampling-frequency keyed gather policies for one station."""

    model_config = ConfigDict(extra="forbid")
    sampling_frequencies: dict[float, RateGatherCriteria] = Field(default_factory=dict)

    @field_validator("sampling_frequencies")
    @classmethod
    def validate_sample_rates(
        cls, value: dict[float, RateGatherCriteria]
    ) -> dict[float, RateGatherCriteria]:
        if any(sample_rate <= 0 for sample_rate in value):
            raise ValueError("Sampling frequencies must be positive")
        sample_rates = list(value)
        for index, left in enumerate(sample_rates):
            if any(
                np.isclose(left, right, rtol=1e-9, atol=1e-12)
                for right in sample_rates[index + 1 :]
            ):
                raise ValueError("Sampling-frequency keys must be distinct")
        return value


class ResolvedGatherCriteria(ResisticsModel):
    """Resolved rate policy returned uniformly for every level/eval index."""

    remote_references: Literal["auto"] | list[str] | None = None
    masks: MaskCriteria | None = None


class GatherSelection(ResisticsData):
    """Resolved target/rate inputs and admissible global windows for gathering."""

    def __init__(
        self,
        station_rate_batch: dict[str, Any],
        remote_station: str | None,
        criteria: "GatherCriteria",
        automatic_remote: bool = False,
    ) -> None:
        self.station_rate_batch = dict(station_rate_batch)
        self.remote_station = remote_station
        self.criteria = criteria
        self.automatic_remote = automatic_remote


class GatherCriteria(ResisticsProcess):
    """Strict station/rate policy for gathering persisted evaluation data.

    An empty instance is the no-criteria policy: all target windows are kept and
    gathering is single-site.  Policies are currently rate-wide, while
    :meth:`resolve` accepts level and evaluation-frequency indices so callers do
    not need to change when the schema gains more granular overrides.

    Examples
    --------
    Resolve an automatic remote-reference policy for one target and rate.

    >>> criteria = GatherCriteria(
    ...     stations={
    ...         "survey/target": {
    ...             "sampling_frequencies": {
    ...                 128: {"remote_references": "auto"}
    ...             }
    ...         }
    ...     }
    ... )
    >>> criteria.resolve("survey/target", 128).remote_references
    'auto'
    """

    model_config = ConfigDict(extra="forbid")

    output_type: ClassVar[str] = "gather_selection"
    runtime_requirements: ClassVar[list[str]] = ["station_rate_batch"]

    name: str = Field(default="", exclude=True)
    stations: dict[str, StationGatherCriteria] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def reject_flat_schema(cls, value: Any) -> Any:
        if isinstance(value, dict) and {
            "remote_references",
            "absolute_include",
            "absolute_exclude",
            "daily_include",
            "daily_exclude",
            "automatic_remote",
        }.intersection(value):
            raise ValueError(
                "Legacy flat gather criteria are not supported. Use "
                "stations.<survey/station>.sampling_frequencies.<rate> with "
                "remote_references and/or masks; time constraints belong in a "
                "named TimeMask artifact."
            )
        return value

    @field_validator("stations")
    @classmethod
    def validate_stations(
        cls, value: dict[str, StationGatherCriteria]
    ) -> dict[str, StationGatherCriteria]:
        for station_path, station_criteria in value.items():
            _validate_station_path(station_path)
            for rate_criteria in station_criteria.sampling_frequencies.values():
                remotes = rate_criteria.remote_references
                if isinstance(remotes, list) and station_path in remotes:
                    raise ValueError(
                        f"Target station {station_path!r} cannot be its own remote"
                    )
        return value

    def resolve(
        self,
        station_path: str,
        sample_rate: float,
        level: int = 0,
        evaluation_frequency_index: int = 0,
    ) -> ResolvedGatherCriteria:
        """Resolve a policy, returning the no-criteria policy when unlisted."""
        if level < 0 or evaluation_frequency_index < 0:
            raise ValueError(
                "Level and evaluation-frequency index must be non-negative"
            )
        station = self.stations.get(station_path)
        if station is None:
            return ResolvedGatherCriteria()
        for configured_rate, policy in station.sampling_frequencies.items():
            if np.isclose(configured_rate, sample_rate, rtol=1e-9, atol=1e-12):
                return ResolvedGatherCriteria(**policy.model_dump())
        return ResolvedGatherCriteria()

    def remote_station_paths(self, station_path: str, sample_rate: float) -> list[str]:
        """Return explicitly configured remotes (``auto`` resolves at runtime)."""
        remotes = self.resolve(station_path, sample_rate).remote_references
        return [] if remotes is None or remotes == "auto" else list(remotes)

    def remote_reference_count(self) -> int:
        """Return the number of explicit remote assignments for summaries."""
        total = 0
        for station in self.stations.values():
            for policy in station.sampling_frequencies.values():
                if policy.remote_references == "auto":
                    total += 1
                elif isinstance(policy.remote_references, list):
                    total += len(policy.remote_references)
        return total

    def run(self, station_rate_batch: dict[str, Any]) -> GatherSelection:
        """Resolve the remote assignment for one target station/rate batch."""
        station_rate_batch = dict(station_rate_batch)
        target = station_rate_batch.get("station_path")
        if target is None:
            target = f"{station_rate_batch['survey']}/{station_rate_batch['station']}"
            station_rate_batch["station_path"] = target
        sample_rate = station_rate_batch["sample_rate"]
        remotes = self.resolve(target, sample_rate).remote_references
        remote_station = remotes[0] if isinstance(remotes, list) and remotes else None
        return GatherSelection(
            station_rate_batch,
            remote_station,
            self,
            automatic_remote=remotes == "auto",
        )

    def execute(self, inputs: dict[str, Any], context: Any) -> GatherSelection:
        """Resolve criteria from the station-rate batch supplied by the executor."""
        del inputs
        return self.run(context["station_rate_batch"])

    def includes(self, timestamp: datetime) -> bool:
        """Compatibility hook: time rules now live in named mask artifacts."""
        del timestamp
        return True


class EvaluationFrequencyGather(ResisticsProcess):
    """Gather every persisted target-run evaluation artifact for regression.

    The selection is deliberately a separate input: it makes the persisted
    evaluation artifact boundary explicit and leaves window-selection policy in
    :class:`GatherCriteria`, not in a job or parameter file.  Remote-reference
    alignment is intentionally not guessed; an explicit resolved remote is
    rejected until the cross-station aligner is implemented.
    """

    input_types: ClassVar[dict[str, str]] = {"selection": "gather_selection"}
    output_type: ClassVar[str] = "gathered_data"
    runtime_requirements: ClassVar[list[str]] = ["project_path"]

    def execute(self, inputs: dict[str, Any], context: Any) -> "GatheredData":
        """Load and concatenate all selected local-run evaluation artifacts."""
        from resistics.spectra import EvaluationFrequencyReader
        from resistics.transfunc import ImpedanceTensor

        selection = inputs["selection"]
        if not isinstance(selection, GatherSelection):
            raise ValueError("EvaluationFrequencyGather requires GatherSelection")
        if selection.automatic_remote:
            raise NotImplementedError(
                "Automatic remote selection is declared but has not been implemented"
            )
        if selection.remote_station is not None:
            raise NotImplementedError(
                "Remote-reference gathering is not implemented yet; the criteria "
                "mapping was resolved but cannot be applied"
            )
        batch = selection.station_rate_batch
        gathered = []
        for run_path in batch["run_paths"]:
            survey, station, run = run_path.split("/", 2)
            artifact = EvaluationFrequencyReader(
                label=validate_output_label(context["output_label"])
            ).execute(
                {},
                {
                    "project_path": context["project_path"],
                    "run_batch": {"survey": survey, "station": station, "run": run},
                },
            )
            artifact = self._apply_criteria(artifact, selection.criteria)
            gathered.append(
                QuickGather().run(
                    Path(run_path),
                    artifact.decimation_parameters,
                    ImpedanceTensor(),
                    artifact.spectra_data,
                )
            )
        if not gathered:
            raise ValueError("No evaluation-frequency artifacts selected for gather")
        return self._combine(gathered)

    @staticmethod
    def _apply_criteria(artifact: Any, criteria: GatherCriteria) -> Any:
        """Drop persisted evaluation windows excluded by the reusable criteria."""
        data = artifact.spectra_data
        metadata = data.metadata.model_copy(deep=True)
        selected = {}
        for level in range(metadata.n_levels):
            timestamps = data.get_timestamps(level)
            mask = np.array(
                [
                    criteria.includes(timestamp.to_pydatetime())
                    for timestamp in timestamps
                ]
            )
            selected[level] = data.get_level(level)[mask]
            metadata.levels_metadata[level].n_wins = int(mask.sum())
        if not any(level_data.shape[0] for level_data in selected.values()):
            raise ValueError("Gather criteria excludes every evaluation window")
        artifact.spectra_data = SpectraData(metadata, selected)
        return artifact

    @staticmethod
    def _combine(values: list["GatheredData"]) -> "GatheredData":
        def combine(kind: str) -> SiteCombinedData:
            first = getattr(values[0], kind)
            metadata = first.metadata.model_copy(
                update={
                    "measurements": [
                        measurement
                        for value in values
                        for measurement in (
                            getattr(value, kind).metadata.measurements or []
                        )
                    ],
                    "histories": {
                        name: history
                        for value in values
                        for name, history in getattr(
                            value, kind
                        ).metadata.histories.items()
                    },
                }
            )
            keys = first.data
            return SiteCombinedData(
                metadata,
                {
                    key: np.concatenate(
                        [getattr(value, kind).data[key] for value in values]
                    )
                    for key in keys
                },
            )

        return GatheredData(
            combine("out_data"), combine("in_data"), combine("cross_data")
        )


class SiteCombinedMetadata(WriteableMetadata):
    """
    Metadata for combined data

    Combined metadata stores metadata for measurements that are combined from
    a single site.
    """

    site_name: str
    """The name of the site"""
    site_names: list[str] = Field(default_factory=list)
    """Authoritative station paths, including every pooled remote station"""
    fs: float
    """Recording sampling frequency"""
    system: str = ""
    """The system used for recording"""
    serial: str = ""
    """Serial number of the system"""
    wgs84_latitude: float = -999.0
    """Latitude in WGS84"""
    wgs84_longitude: float = -999.0
    """Longitude in WGS84"""
    easting: float = -999.0
    """The easting of the site in local cartersian coordinates"""
    northing: float = -999.0
    """The northing of the site in local cartersian coordinates"""
    elevation: float = -999.0
    """The elevation of the site"""
    measurements: list[str] | None = None
    """List of measurement names that were included in the combined data"""
    chans: list[str]
    """List of channels, these are common amongst all the measurements"""
    n_evals: int
    """The number of evaluation frequencies"""
    eval_freqs: list[float]
    """The evaluation frequencies"""
    histories: dict[str, History]
    """Dictionary mapping measurement name to measurement processing history"""

    @model_validator(mode="after")
    def populate_site_names(self) -> "SiteCombinedMetadata":
        if not self.site_names:
            self.site_names = [self.site_name]
        return self


class SiteCombinedData(ResisticsData):
    """
    Combined data is data that is combined from a single site for the purposes
    of regression.

    All of the data that is combined should have the same sampling frequency,
    same evaluation frequencies and some shared channels.

    Data is stored in the data attribute of the class. This is a dictionary
    mapping evaluation frequency index to data for the evaluation frequency
    from all windows in the site. The shape of data for a single evaluation
    frequency is:

    n_wins x n_chans

    The data is complex valued.
    """

    def __init__(self, metadata: SiteCombinedMetadata, data: dict[int, np.ndarray]):
        """
        Initialise the CombinedData

        Parameters
        ----------
        metadata : CombinedMetadata
            The combined metadata
        data : Dict[int, np.ndarray]
            The data with index the evaluation frequency index and value the
            combined data from a site for the evaluation frequency
        """
        self.metadata = metadata
        self.data = data


class GatheredData(ResisticsData):
    """
    Class to hold data to be used in by Regression preparers

    Gathered data has an out_data, in_data and cross_data. The important thing
    here is that the data is all aligned with regards to windows
    """

    def __init__(
        self,
        out_data: SiteCombinedData,
        in_data: SiteCombinedData,
        cross_data: SiteCombinedData,
    ) -> None:
        self.out_data = out_data
        self.in_data = in_data
        self.cross_data = cross_data


@dataclass(frozen=True)
class _EvaluationLocator:
    """One evaluation row addressed by its station-global window index."""

    run_path: str
    local_index: int
    artifact: Any


class Gather(ResisticsProcess):
    """Gather aligned persisted evaluation data for local or remote regression.

    Target data supplies output and input channels.  With remote references,
    target/remote pairs are pooled and the remote data supplies cross channels.
    The same target window is intentionally repeated when it aligns with more
    than one configured remote.
    """

    input_types: ClassVar[dict[str, str]] = {
        "selection": "gather_selection",
        "tf": "transfer_function",
    }
    output_type: ClassVar[str] = "gathered_data"
    runtime_requirements: ClassVar[list[str]] = ["project", "project_path"]

    def execute(self, inputs: dict[str, Any], context: Any) -> GatheredData:
        """Gather the flow inputs using the project supplied at runtime."""
        selection = inputs["selection"]
        if not isinstance(selection, GatherSelection):
            raise ValueError("Gather requires GatherSelection")
        return self.run(
            context["project"],
            Path(context["project_path"]),
            selection,
            inputs["tf"],
            context["output_label"],
        )

    def run(  # noqa: C901 - gather decomposition is owned by Phase 5.5
        self,
        project: Project,
        project_path: Path,
        selection: GatherSelection,
        tf: TransferFunction,
        output_label: str = "default",
    ) -> GatheredData:
        """Gather aligned evaluation data for one target station and rate.

        Parameters
        ----------
        project : Project
            The open project containing target and remote station data.
        project_path : Path
            Project root containing persisted evaluation artifacts.
        selection : GatherSelection
            Resolved target, rate, masks, and remote-reference policy.
        tf : TransferFunction
            Transfer function defining output, input, and cross channels.
        output_label : str, optional
            Namespace containing the persisted inputs.

        Returns
        -------
        GatheredData
            Window-aligned arrays ready for regression.
        """
        from resistics.spectra import EvaluationFrequencyReader

        cross_channels = _cross_channels(tf)
        batch = selection.station_rate_batch
        target = batch["station_path"]
        sample_rate = float(batch["sample_rate"])
        target_runs = sorted(batch["run_paths"])
        if not target_runs:
            raise ValueError(f"Target {target} at {sample_rate:g} Hz has no runs")

        self._output_label = validate_output_label(output_label)
        self._reader = EvaluationFrequencyReader(label=self._output_label)
        self._project_path = project_path
        self._criteria = selection.criteria
        self._artifact_cache: dict[str, Any] = {}
        self._mask_cache: dict[tuple[str, str], WindowMask] = {}

        target_artifacts = self._load_runs(target_runs, required=True, role="target")
        baseline = next(iter(target_artifacts.values()))
        for run_path, artifact in target_artifacts.items():
            self._validate_compatible(baseline, artifact, f"target run {run_path}")
        level_references = self._level_references(target_artifacts, target)

        resolved = self._criteria.resolve(target, sample_rate)
        remote_setting = resolved.remote_references
        if remote_setting is None:
            candidate_paths: list[str] = []
            automatic = False
        elif remote_setting == "auto":
            candidate_paths = project.get_concurrent(target, sample_rate)
            automatic = True
        else:
            candidate_paths = sorted(remote_setting)
            automatic = False

        remote_artifacts: dict[str, dict[str, Any]] = {}
        candidate_reasons: dict[str, str] = {}
        for station_path in sorted(candidate_paths):
            try:
                run_paths = self._station_runs(project, station_path, sample_rate)
                artifacts = self._load_runs(
                    run_paths,
                    required=not automatic,
                    role=f"remote {station_path}",
                )
                if not artifacts:
                    candidate_reasons[station_path] = "no readable evaluation artifacts"
                    continue
                compatible = {}
                for run_path, artifact in artifacts.items():
                    try:
                        self._validate_compatible(
                            baseline, artifact, f"remote run {run_path}"
                        )
                        for level, reference in level_references.items():
                            if level < artifact.spectra_data.metadata.n_levels:
                                self._validate_level_compatible(
                                    reference,
                                    artifact,
                                    level,
                                    f"remote run {run_path}",
                                )
                    except Exception as exc:
                        if not automatic:
                            raise
                        logger.warning(
                            f"Skipping incompatible automatic remote run "
                            f"{run_path}: {exc}"
                        )
                        continue
                    compatible[run_path] = artifact
                if not compatible:
                    candidate_reasons[station_path] = (
                        "no compatible evaluation artifacts"
                    )
                    continue
                remote_artifacts[station_path] = compatible
            except Exception as exc:
                if not automatic:
                    raise
                candidate_reasons[station_path] = str(exc)
                logger.warning(
                    f"Skipping automatic remote {station_path} for {target}: {exc}"
                )

        out_values: dict[int, np.ndarray] = {}
        in_values: dict[int, np.ndarray] = {}
        cross_values: dict[int, np.ndarray] = {}
        target_used: set[str] = set()
        remote_used: set[str] = set()
        usable_remotes: set[str] = set()
        eval_freqs: list[float] = []

        key = 0
        for level, level_reference in level_references.items():
            level_metadata = level_reference.spectra_data.metadata.levels_metadata[
                level
            ]
            target_catalog = self._catalog(target_artifacts, level, target)
            for evaluation_index, evaluation_frequency in enumerate(
                level_metadata.freqs
            ):
                eval_freqs.append(evaluation_frequency)
                target_valid = self._apply_masks(
                    target,
                    sample_rate,
                    level,
                    evaluation_index,
                    target_catalog,
                )
                pairs: list[tuple[_EvaluationLocator, _EvaluationLocator]] = []
                if remote_setting is None:
                    pairs = [
                        (target_valid[global_index], target_valid[global_index])
                        for global_index in sorted(target_valid)
                    ]
                else:
                    skipped = []
                    for remote_path in sorted(remote_artifacts):
                        remote_catalog = self._catalog(
                            remote_artifacts[remote_path], level, remote_path
                        )
                        remote_valid = self._apply_masks(
                            remote_path,
                            sample_rate,
                            level,
                            evaluation_index,
                            remote_catalog,
                        )
                        shared = sorted(set(target_valid).intersection(remote_valid))
                        if not shared:
                            skipped.append(f"{remote_path}: no shared admitted windows")
                            continue
                        usable_remotes.add(remote_path)
                        pairs.extend(
                            (target_valid[global_index], remote_valid[global_index])
                            for global_index in shared
                        )
                    if skipped:
                        logger.warning(
                            f"Gather {target}, level {level}, evaluation index "
                            f"{evaluation_index}: {'; '.join(skipped)}"
                        )
                if not pairs:
                    reasons = "; ".join(
                        f"{name}: {reason}"
                        for name, reason in sorted(candidate_reasons.items())
                    )
                    suffix = f" Candidate diagnostics: {reasons}" if reasons else ""
                    raise ValueError(
                        f"No admissible gather windows for target {target}, "
                        f"{sample_rate:g} Hz, level {level}, evaluation-frequency "
                        f"index {evaluation_index}.{suffix}"
                    )
                target_locs = [pair[0] for pair in pairs]
                cross_locs = [pair[1] for pair in pairs]
                out_values[key] = self._extract(
                    target_locs, level, evaluation_index, tf.out_chans
                )
                in_values[key] = self._extract(
                    target_locs, level, evaluation_index, tf.in_chans
                )
                cross_values[key] = self._extract(
                    cross_locs, level, evaluation_index, cross_channels
                )
                target_used.update(locator.run_path for locator in target_locs)
                if remote_setting is not None:
                    remote_used.update(locator.run_path for locator in cross_locs)
                key += 1

        if remote_setting is not None and not usable_remotes:
            raise ValueError(
                f"No usable remote reference remains for target {target} at "
                f"{sample_rate:g} Hz"
            )

        out_metadata = self._combined_metadata(
            target,
            [target],
            target_used,
            tf.out_chans,
            eval_freqs,
            target_artifacts,
        )
        in_metadata = self._combined_metadata(
            target,
            [target],
            target_used,
            tf.in_chans,
            eval_freqs,
            target_artifacts,
        )
        cross_stations = [target] if remote_setting is None else sorted(usable_remotes)
        cross_source = (
            target_artifacts
            if remote_setting is None
            else {
                run: artifact
                for station_artifacts in remote_artifacts.values()
                for run, artifact in station_artifacts.items()
            }
        )
        cross_metadata = self._combined_metadata(
            target if remote_setting is None else "+".join(cross_stations),
            cross_stations,
            target_used if remote_setting is None else remote_used,
            cross_channels,
            eval_freqs,
            cross_source,
        )
        logger.info(
            f"Gathered {target} at {sample_rate:g} Hz using cross stations "
            f"{cross_stations}"
        )
        return GatheredData(
            SiteCombinedData(out_metadata, out_values),
            SiteCombinedData(in_metadata, in_values),
            SiteCombinedData(cross_metadata, cross_values),
        )

    def _station_runs(
        self, project: Project, station_path: str, sample_rate: float
    ) -> list[str]:
        _validate_station_path(station_path)
        rows = project.table[
            (project.table["station_path"] == station_path)
            & np.isclose(project.table["sample_rate"].astype(float), sample_rate)
        ]
        run_paths = sorted(rows["run_path"].unique().tolist())
        if not run_paths:
            raise ValueError(
                f"Remote station {station_path!r} has no runs at {sample_rate:g} Hz"
            )
        return run_paths

    def _load_runs(
        self, run_paths: list[str], required: bool, role: str
    ) -> dict[str, Any]:
        values = {}
        failures = []
        for run_path in sorted(run_paths):
            if run_path in self._artifact_cache:
                values[run_path] = self._artifact_cache[run_path]
                continue
            survey, station, run = run_path.split("/", 2)
            try:
                artifact = self._reader.execute(
                    {},
                    {
                        "project_path": self._project_path,
                        "run_batch": {
                            "survey": survey,
                            "station": station,
                            "run": run,
                        },
                    },
                )
            except Exception as exc:
                failures.append(f"{run_path}: {exc}")
                continue
            self._artifact_cache[run_path] = artifact
            values[run_path] = artifact
        if failures and required:
            raise ValueError(
                f"Missing or unreadable evaluation artifact for {role}: "
                + "; ".join(failures)
            )
        if failures:
            logger.warning(f"Skipping unreadable {role} runs: {'; '.join(failures)}")
        return values

    @staticmethod
    def _validate_compatible(reference: Any, candidate: Any, description: str) -> None:
        ref_dec = reference.decimation_parameters
        got_dec = candidate.decimation_parameters
        failures = []
        if not np.isclose(ref_dec.fs, got_dec.fs):
            failures.append("original sample rate")
        ref_data = reference.spectra_data
        got_data = candidate.spectra_data
        if str(ref_data.metadata.ref_time) != str(got_data.metadata.ref_time):
            failures.append("reference time")
        for level in range(min(ref_data.metadata.n_levels, got_data.metadata.n_levels)):
            failures.extend(
                Gather._level_compatibility_failures(reference, candidate, level)
            )
        if failures:
            raise ValueError(
                f"Incompatible {description}: {', '.join(dict.fromkeys(failures))}"
            )

    @staticmethod
    def _level_compatibility_failures(
        reference: Any, candidate: Any, level: int
    ) -> list[str]:
        """Compare one realised evaluation level using spectra metadata only."""
        left = reference.spectra_data.metadata.levels_metadata[level]
        right = candidate.spectra_data.metadata.levels_metadata[level]
        failures = []
        if not np.isclose(left.fs, right.fs):
            failures.append(f"level {level} sample rate")
        if left.win_size != right.win_size or left.olap_size != right.olap_size:
            failures.append(f"level {level} window signature")
        if (
            left.n_freqs != right.n_freqs
            or len(left.freqs) != len(right.freqs)
            or not np.allclose(left.freqs, right.freqs)
        ):
            failures.append(f"level {level} evaluation frequencies")
        return failures

    @staticmethod
    def _validate_level_compatible(
        reference: Any, candidate: Any, level: int, description: str
    ) -> None:
        failures = Gather._level_compatibility_failures(reference, candidate, level)
        if failures:
            raise ValueError(f"Incompatible {description}: {', '.join(failures)}")

    @staticmethod
    def _level_references(
        artifacts: dict[str, Any], station_path: str
    ) -> dict[int, Any]:
        """Choose and validate an authoritative target artifact for each level."""
        n_levels = max(
            artifact.spectra_data.metadata.n_levels for artifact in artifacts.values()
        )
        if n_levels == 0:
            raise ValueError(
                f"Target station {station_path} has no realised evaluation levels"
            )
        references = {}
        for level in range(n_levels):
            available = {
                run_path: artifact
                for run_path, artifact in artifacts.items()
                if level < artifact.spectra_data.metadata.n_levels
            }
            reference_run = sorted(available)[0]
            reference = available[reference_run]
            for run_path, artifact in available.items():
                Gather._validate_level_compatible(
                    reference,
                    artifact,
                    level,
                    f"target run {run_path} for station {station_path}",
                )
            references[level] = reference
        return references

    @staticmethod
    def _catalog(
        artifacts: dict[str, Any], level: int, station_path: str
    ) -> dict[int, _EvaluationLocator]:
        catalog = {}
        for run_path in sorted(artifacts):
            artifact = artifacts[run_path]
            metadata = artifact.spectra_data.metadata
            if level >= metadata.n_levels:
                continue
            level_meta = metadata.levels_metadata[level]
            for local_index in range(level_meta.n_wins):
                global_index = level_meta.index_offset + local_index
                if global_index in catalog:
                    raise ValueError(
                        f"Ambiguous global window {global_index} for station "
                        f"{station_path}, level {level}: {catalog[global_index].run_path} "
                        f"and {run_path}"
                    )
                catalog[global_index] = _EvaluationLocator(
                    run_path, local_index, artifact
                )
        return catalog

    def _apply_masks(
        self,
        station_path: str,
        sample_rate: float,
        level: int,
        evaluation_index: int,
        catalog: dict[int, _EvaluationLocator],
    ) -> dict[int, _EvaluationLocator]:
        policy = self._criteria.resolve(
            station_path, sample_rate, level, evaluation_index
        )
        if policy.masks is None:
            return catalog
        result = {}
        for global_index, locator in catalog.items():
            decisions = []
            for name in policy.masks.names:
                mask = self._load_mask(
                    station_path,
                    locator.run_path,
                    name,
                    locator.artifact,
                    level,
                    evaluation_index,
                )
                decisions.append(
                    bool(mask.get_keep(level, evaluation_index).loc[global_index])
                )
            keep = all(decisions) if policy.masks.combine == "and" else any(decisions)
            if keep:
                result[global_index] = locator
        return result

    def _load_mask(
        self,
        station_path: str,
        run_path: str,
        name: str,
        artifact: Any,
        level: int,
        evaluation_index: int,
    ) -> WindowMask:
        cache_key = (run_path, name)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]
        survey, station, run = run_path.split("/", 2)
        path = get_run_mask_path(
            self._project_path,
            {"survey": survey, "station": station, "run": run},
            name,
            self._output_label,
        )
        try:
            mask = WindowMaskReader().run(path)
        except Exception as exc:
            raise ValueError(
                f"Required mask {name!r} is unavailable for target/remote "
                f"{station_path}, run {run_path}, level {level}, "
                f"evaluation-frequency index {evaluation_index}: {exc}"
            ) from exc
        self._validate_mask(mask, artifact, run_path, name)
        self._mask_cache[cache_key] = mask
        return mask

    @staticmethod
    def _validate_mask(
        mask: WindowMask, artifact: Any, run_path: str, name: str
    ) -> None:
        data = artifact.spectra_data
        dec_params = artifact.decimation_parameters
        failures = []
        if not np.isclose(mask.metadata.sample_rate, dec_params.fs):
            failures.append("original sample rate")
        if str(mask.metadata.ref_time) != str(data.metadata.ref_time):
            failures.append("reference time")
        if len(mask.metadata.levels) != data.metadata.n_levels:
            failures.append("number of levels")
        else:
            for item, source in zip(
                mask.metadata.levels, data.metadata.levels_metadata, strict=False
            ):
                if item.n_evaluation_frequencies != source.n_freqs:
                    failures.append(f"level {item.level} evaluation count")
                expected_freqs = source.freqs
                if len(item.evaluation_frequencies) != len(
                    expected_freqs
                ) or not np.allclose(item.evaluation_frequencies, expected_freqs):
                    failures.append(f"level {item.level} evaluation frequencies")
                if (
                    not np.isclose(item.fs, source.fs)
                    or item.n_wins != source.n_wins
                    or item.win_size != source.win_size
                    or item.olap_size != source.olap_size
                    or item.index_offset != source.index_offset
                ):
                    failures.append(f"level {item.level} window signature")
        if failures:
            raise ValueError(
                f"Mask {name!r} for run {run_path} is incompatible: "
                + ", ".join(dict.fromkeys(failures))
            )

    @staticmethod
    def _extract(
        locators: list[_EvaluationLocator],
        level: int,
        evaluation_index: int,
        channels: list[str],
    ) -> np.ndarray:
        """Extract selected windows in batches for each source artifact.

        Selecting channels from a whole spectra level uses NumPy advanced
        indexing and therefore copies the entire level.  Doing that once per
        window made a large standard-window gather quadratic in its window
        count.  Grouping by run lets us select only the requested rows for
            each artifact keeps the gather linear in its selected window count.
        """
        from resistics.errors import ChannelNotFoundError

        values = np.empty((len(locators), len(channels)), dtype=np.complex128)
        grouped: dict[str, tuple[Any, list[int], list[int]]] = {}
        for output_index, locator in enumerate(locators):
            if locator.run_path not in grouped:
                grouped[locator.run_path] = (locator.artifact, [], [])
            _, output_indices, local_indices = grouped[locator.run_path]
            output_indices.append(output_index)
            local_indices.append(locator.local_index)

        for artifact, output_indices, local_indices in grouped.values():
            spectra = artifact.spectra_data
            channel_indices = []
            for channel in channels:
                if channel not in spectra.metadata.chans:
                    raise ChannelNotFoundError(channel, spectra.metadata.chans)
                channel_indices.append(spectra.metadata.chans.index(channel))
            level_data = spectra.get_level(level)
            values[np.asarray(output_indices)] = level_data[
                np.asarray(local_indices), :, evaluation_index
            ][:, channel_indices]
        return values

    def _combined_metadata(
        self,
        site_name: str,
        site_names: list[str],
        used_runs: set[str],
        channels: list[str],
        eval_freqs: list[float],
        artifacts: dict[str, Any],
    ) -> SiteCombinedMetadata:
        first_run = sorted(used_runs)[0]
        first_artifact = artifacts[first_run]
        first = first_artifact.spectra_data.metadata
        histories = {}
        for run_path in sorted(used_runs):
            history = artifacts[run_path].spectra_data.metadata.history.model_copy(
                deep=True
            )
            history.add_record(
                self._get_record(
                    [f"Gathered evaluation-frequency data from {run_path}"]
                )
            )
            histories[run_path] = history
        return SiteCombinedMetadata(
            site_name=site_name,
            site_names=site_names,
            fs=float(first_artifact.decimation_parameters.fs),
            system=first.system,
            serial=first.serial,
            wgs84_latitude=first.wgs84_latitude,
            wgs84_longitude=first.wgs84_longitude,
            easting=first.easting,
            northing=first.northing,
            elevation=first.elevation,
            measurements=sorted(used_runs),
            chans=channels,
            n_evals=len(eval_freqs),
            eval_freqs=eval_freqs,
            histories=histories,
        )


class QuickGather(ResisticsProcess):
    """
    Processor to gather data outside of a resistics environment

    This is intended for use when quickly calculating out a transfer function
    for a single measurement and only a single spectra data instance is accepted
    as input.

    Remote reference or intersite processing is not possible using QuickGather

    See Also
    --------
    Gather : For persisted, aligned project gathering.
    """

    def run(
        self,
        dir_path: Path,
        dec_params: DecimationParameters,
        tf: TransferFunction,
        eval_data: SpectraData,
    ) -> GatheredData:
        """
        Generate the GatheredData object for input into regression preparation

        The input is a single spectra data instance and is used to populate the
        in_data, out_data and cross_data.

        Parameters
        ----------
        dir_path : Path
            The directory path to the measurement
        dec_params : DecimationParameters
            The decimation parameters
        tf : TransferFunction
            The transfer function
        eval_data : SpectraData
            The spectra data at the evaluation frequencies

        Returns
        -------
        GatheredData
            GatheredData for regression preparer
        """
        metadata = eval_data.metadata
        cross_channels = _cross_channels(tf)
        out_data = {}
        in_data = {}
        cross_data = {}
        eval_freqs = []
        logger.info("Quick gathering data for regression prepartion")
        for ilevel in range(metadata.n_levels):
            level_metadata = metadata.levels_metadata[ilevel]
            eval_freqs = eval_freqs + level_metadata.freqs
            out_level = eval_data.get_chans(ilevel, tf.out_chans)
            in_level = eval_data.get_chans(ilevel, tf.in_chans)
            cross_level = eval_data.get_chans(ilevel, cross_channels)
            for ifreq in range(level_metadata.n_freqs):
                key = dec_params.per_level * ilevel + ifreq
                out_data[key] = out_level[..., ifreq]
                in_data[key] = in_level[..., ifreq]
                cross_data[key] = cross_level[..., ifreq]
        # make combined data
        fs = dec_params.fs
        out_combined = self._get_combined_data(
            dir_path.name, fs, tf.out_chans, eval_freqs, metadata, out_data
        )
        in_combined = self._get_combined_data(
            dir_path.name, fs, tf.in_chans, eval_freqs, metadata, in_data
        )
        cross_combined = self._get_combined_data(
            dir_path.name, fs, cross_channels, eval_freqs, metadata, cross_data
        )
        return GatheredData(
            out_data=out_combined, in_data=in_combined, cross_data=cross_combined
        )

    def _get_combined_data(
        self,
        meas: str,
        fs: float,
        chans: list[str],
        eval_freqs: list[float],
        metadata: SpectraMetadata,
        data: dict[int, np.ndarray],
    ) -> SiteCombinedData:
        """Get the combined metadata"""
        combined_metadata = SiteCombinedMetadata(
            site_name=meas,
            fs=fs,
            system=metadata.system,
            serial=metadata.serial,
            wgs84_latitude=metadata.wgs84_latitude,
            wgs84_longitude=metadata.wgs84_longitude,
            easting=metadata.easting,
            northing=metadata.northing,
            elevation=metadata.elevation,
            measurements=[meas],
            chans=chans,
            n_evals=len(eval_freqs),
            eval_freqs=eval_freqs,
            histories={meas: metadata.history},
        )
        return SiteCombinedData(combined_metadata, data)
