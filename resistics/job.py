"""Discovery, validation, and execution of project processing jobs."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import re
from shutil import rmtree
from threading import Event
from time import monotonic
from typing import Any, Callable, Dict, List, Optional, Union
import warnings

from loguru import logger
import numpy as np
from pydantic import AliasChoices, BaseModel, Field, field_validator

from resistics import __version__
from resistics.flow import (
    FlowCancelled,
    FlowDefinition,
    FlowExecutor,
    FlowStage,
    FlowValidator,
    ParameterSet,
    ProcessingJob,
    model_from_yaml_file,
)
from resistics.gather import GatherCriteria
from resistics.project import Project, get_results_path
from resistics.common import fs_to_string, validate_output_label


class JobDefinition(BaseModel):
    """Human-authored job referencing project flow and parameter files."""

    name: str
    flow: str
    parameters: str
    criteria: Optional[str] = None
    scope: "JobScope" = Field(default_factory=lambda: JobScope())
    output_label: str = "default"
    overwrite: bool = False

    @field_validator("output_label")
    @classmethod
    def validate_output_label_value(cls, value: str) -> str:
        return validate_output_label(value)


class JobScope(BaseModel):
    """Survey, station, rate, and flow-stage restrictions for a submission."""

    surveys: List[str] = Field(default_factory=list)
    stations: List[str] = Field(default_factory=list)
    sampling_frequencies: List[float] = Field(
        default_factory=list,
        validation_alias=AliasChoices("sampling_frequencies", "sample_rates"),
        serialization_alias="sampling_frequencies",
    )
    stages: List[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("stages", "stage_scope"),
        serialization_alias="stages",
    )


class StationRateBatch(BaseModel):
    """One planned target station and recording-rate processing unit."""

    survey: str
    station: str
    sample_rate: float
    run_paths: List[str]

    @property
    def station_path(self) -> str:
        return f"{self.survey}/{self.station}"


class JobState(str, Enum):
    """Lifecycle state for a locally managed processing job."""

    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class JobProgressEvent(BaseModel):
    """Serializable progress update emitted by :class:`JobRunner`."""

    state: JobState
    message: str
    job_name: str
    survey: Optional[str] = None
    station: Optional[str] = None
    run: Optional[str] = None
    sample_rate: Optional[float] = None
    node_id: Optional[str] = None
    step_type: Optional[str] = None
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class JobSummary(BaseModel):
    """Cheap summary of a job file for user interfaces."""

    name: str
    path: Path
    flow: str = ""
    parameters: str = ""
    output_label: str = ""
    is_valid: bool = False
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ResolvedJob(BaseModel):
    """A file-authored job resolved into executable models."""

    path: Path
    definition: JobDefinition
    processing_job: ProcessingJob
    flow_path: Path
    parameters_path: Path
    criteria_path: Optional[Path] = None
    criteria: Optional[GatherCriteria] = None
    stages: List[FlowStage]
    batches: List[StationRateBatch]
    output_path: Path


class JobValidation(BaseModel):
    """Validation result with an optional resolved job."""

    ok: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    resolved_job: Optional[ResolvedJob] = None


ProgressCallback = Callable[[JobProgressEvent], None]
_JOB_TEMPLATE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def validate_job_template_name(name: str) -> str:
    """Return a filename-safe template name or raise a user-facing error."""
    value = name.strip()
    if not _JOB_TEMPLATE_NAME.fullmatch(value):
        raise ValueError(
            "Job name must use letters, numbers, dots, underscores, or hyphens"
        )
    if Path(value).suffix in {".yaml", ".yml"}:
        raise ValueError("Job name must not include a YAML extension")
    return value


class ProjectJobs:
    """Access and create job definitions belonging to one project."""

    def __init__(self, project: Project):
        self.project = project
        self.jobs_path = project.project_path / "processing" / "jobs"

    def list(self) -> List[JobSummary]:
        """List and validate project YAML jobs without executing them."""
        paths = sorted(
            path
            for pattern in ("*.yaml", "*.yml")
            for path in self.jobs_path.glob(pattern)
            if path.is_file()
        )
        summaries = []
        for path in paths:
            validation = self.validate(path)
            if validation.resolved_job is not None:
                definition = validation.resolved_job.definition
                summaries.append(
                    JobSummary(
                        name=definition.name,
                        path=path,
                        flow=definition.flow,
                        parameters=definition.parameters,
                        output_label=definition.output_label,
                        is_valid=validation.ok,
                        errors=validation.errors,
                        warnings=validation.warnings,
                    )
                )
            else:
                summaries.append(
                    JobSummary(
                        name=path.stem,
                        path=path,
                        errors=validation.errors,
                        warnings=validation.warnings,
                    )
                )
        return summaries

    def create_template(self, definition: JobDefinition) -> Path:
        """Create a new editable YAML job template without overwriting a job."""
        name = validate_job_template_name(definition.name)
        definition = definition.model_copy(update={"name": name})
        for suffix in (".yaml", ".yml"):
            path = self.jobs_path / f"{name}{suffix}"
            if path.exists():
                raise ValueError(f"Job file already exists: {path.name}")
        path = self.jobs_path / f"{name}.yaml"
        import yaml

        self.jobs_path.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as job_file:
            yaml.safe_dump(
                definition.model_dump(exclude_none=True), job_file, sort_keys=False
            )
        return path

    def validate(self, job: Union[Path, str]) -> JobValidation:
        """Resolve and validate a job, returning user-facing errors."""
        errors: List[str] = []
        warnings: List[str] = []
        try:
            job_path = self._job_path(job)
            definition = model_from_yaml_file(JobDefinition, job_path)
            flow_path = self._reference_path("flows", definition.flow)
            parameters_path = self._reference_path("parameters", definition.parameters)
            flow = model_from_yaml_file(FlowDefinition, flow_path)
            parameters = model_from_yaml_file(ParameterSet, parameters_path)
            criteria_path = None
            criteria = None
            if definition.criteria:
                criteria_path = self._reference_path("criteria", definition.criteria)
                criteria = model_from_yaml_file(GatherCriteria, criteria_path)
            runtime = {"project_path": str(self.project.project_path)}
            processing_job = ProcessingJob(
                name=definition.name,
                flow=flow,
                parameters=parameters,
                runtime=runtime,
                output_label=definition.output_label,
            )
            stages = self.selected_stages(definition, flow)
            batches = self.plan_batches(definition)
            output_path = self._output_path(processing_job)
            resolved = ResolvedJob(
                path=job_path,
                definition=definition,
                processing_job=processing_job,
                flow_path=flow_path,
                parameters_path=parameters_path,
                criteria_path=criteria_path,
                criteria=criteria,
                stages=stages,
                batches=batches,
                output_path=output_path,
            )
        except Exception as exc:
            return JobValidation(ok=False, errors=[str(exc)])

        flow_validation = FlowValidator(
            {
                "project",
                "reference_time",
                "run_batch",
                "station_rate_batch",
                "staging_output_path",
                "criteria",
                "output_label",
            }
        ).validate(processing_job, stages=resolved.stages)
        errors.extend(flow_validation.errors)
        warnings.extend(flow_validation.warnings)
        if not batches:
            errors.append("Job scope does not select any project station/rate batches")
        if criteria is not None:
            errors.extend(self._criteria_errors(criteria))

        return JobValidation(
            ok=not errors,
            errors=errors,
            warnings=warnings,
            resolved_job=resolved,
        )

    def _criteria_errors(self, criteria: GatherCriteria) -> List[str]:
        """Validate configured station/rate references against project inventory."""
        table = self.project.table.copy()
        if "station_path" not in table:
            table["station_path"] = (
                table["survey"].astype(str) + "/" + table["station"].astype(str)
            )
        errors = []
        known_stations = set(table["station_path"].unique())
        for station_path, station_policy in criteria.stations.items():
            if station_path not in known_stations:
                errors.append(f"Criteria station not found in project: {station_path}")
                continue
            for sample_rate, rate_policy in station_policy.sampling_frequencies.items():
                station_rates = table[table["station_path"] == station_path][
                    "sample_rate"
                ].astype(float)
                if not np.isclose(station_rates, sample_rate).any():
                    errors.append(
                        f"Criteria station {station_path} has no runs at "
                        f"{sample_rate:g} Hz"
                    )
                remotes = rate_policy.remote_references
                if not isinstance(remotes, list):
                    continue
                for remote in remotes:
                    if remote not in known_stations:
                        errors.append(f"Criteria remote not found in project: {remote}")
                        continue
                    remote_rates = table[table["station_path"] == remote][
                        "sample_rate"
                    ].astype(float)
                    if not np.isclose(remote_rates, sample_rate).any():
                        errors.append(
                            f"Criteria remote {remote} has no runs at "
                            f"{sample_rate:g} Hz"
                        )
        return errors

    @staticmethod
    def selected_stages(
        definition: JobDefinition, flow: FlowDefinition
    ) -> List[FlowStage]:
        """Resolve the optional stage scope, preserving flow execution order."""
        available = flow.flow_stages()
        requested = definition.scope.stages
        if not requested:
            return available
        known = {stage.stage_id for stage in available}
        unknown = sorted(set(requested) - known)
        if unknown:
            raise ValueError(f"Unknown flow stage(s): {', '.join(unknown)}")
        if len(set(requested)) != len(requested):
            raise ValueError("Job stage scope contains duplicate stage ids")
        return [stage for stage in available if stage.stage_id in set(requested)]

    def _job_path(self, job: Union[Path, str]) -> Path:
        value = Path(job)
        if value.is_absolute() or value.parent != Path("."):
            path = value.resolve()
            if path.parent != self.jobs_path.resolve():
                raise ValueError("Job must be inside processing/jobs")
            if not path.is_file():
                raise ValueError(f"Job file not found: {path}")
            return path
        candidates = self._candidates(self.jobs_path, value.name)
        return self._one_candidate("job", value.name, candidates)

    def _reference_path(self, kind: str, name: str) -> Path:
        value = Path(name)
        if value.is_absolute() or value.parent != Path("."):
            raise ValueError(f"{kind[:-1].title()} must be a project file name")
        directory = self.project.project_path / "processing" / kind
        candidates = self._candidates(directory, value.name)
        return self._one_candidate(kind[:-1], name, candidates)

    @staticmethod
    def _candidates(directory: Path, name: str) -> List[Path]:
        value = Path(name)
        if value.suffix in {".yaml", ".yml"}:
            candidate = directory / value.name
            return [candidate] if candidate.is_file() else []
        candidates = []
        for suffix in (".yaml", ".yml"):
            candidate = directory / f"{value.name}{suffix}"
            if candidate.is_file():
                candidates.append(candidate)
        return candidates

    @staticmethod
    def _one_candidate(kind: str, name: str, candidates: List[Path]) -> Path:
        if not candidates:
            raise ValueError(f"{kind.title()} file not found: {name}")
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous {kind} file name: {name}")
        return candidates[0]

    def _output_path(self, processing_job: ProcessingJob) -> Path:
        runtime = processing_job.runtime
        survey = str(runtime.get("survey", "unknown"))
        station = str(runtime.get("station", "unknown"))
        return get_results_path(
            self.project.project_path,
            survey,
            station,
            processing_job.output_label,
        )

    def plan_batches(self, definition: JobDefinition) -> List[StationRateBatch]:
        """Expand a job scope into deterministic station/rate batches."""
        table = self.project.table.copy()
        scope = definition.scope
        if scope.surveys:
            table = table[table["survey"].isin(scope.surveys)]
        if scope.stations:
            table = table[table["station"].isin(scope.stations)]
        if scope.sampling_frequencies:
            table = table[table["sample_rate"].isin(scope.sampling_frequencies)]
        batches = []
        for (survey, station, sample_rate), rows in table.groupby(
            ["survey", "station", "sample_rate"], sort=True
        ):
            run_paths = sorted(rows["run_path"].unique().tolist())
            batches.append(
                StationRateBatch(
                    survey=str(survey),
                    station=str(station),
                    sample_rate=float(sample_rate),
                    run_paths=run_paths,
                )
            )
        return batches

    def batch_output_path(self, batch: StationRateBatch, output_label: str) -> Path:
        """Return the station-centric output path for one rate batch."""
        return get_results_path(
            self.project.project_path, batch.survey, batch.station, output_label
        ) / fs_to_string(batch.sample_rate)

class JobRunner:
    """Execute one validated project job and emit structured progress."""

    def __init__(
        self,
        project: Project,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        self.project = project
        self.progress_callback = progress_callback
        self._cancel_event = Event()

    def cancel(self) -> None:
        """Request cancellation before the next processing node."""
        self._cancel_event.set()

    def run(self, resolved_job: ResolvedJob) -> JobState:
        """Run every selected run and station/rate stage synchronously."""
        processing_job = resolved_job.processing_job
        started = monotonic()
        log_path = self.project.project_path / "logs" / f"{processing_job.name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        sink_id = logger.add(log_path, enqueue=True)
        caught_warnings = []
        self._emit(
            JobState.running,
            processing_job.name,
            "Job started",
            started,
        )
        state = JobState.completed
        error = None
        partial_paths: List[Path] = []
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                executor = FlowExecutor(
                    progress_callback=lambda event: self._flow_event(
                        processing_job.name, event, started
                    ),
                    cancellation_callback=self._cancel_event.is_set,
                )
                criteria = resolved_job.criteria or GatherCriteria()
                batches = ProjectJobs(self.project).plan_batches(
                    resolved_job.definition
                )
                self._run_batches(
                    executor, resolved_job, batches, criteria, partial_paths, started
                )
        except FlowCancelled:
            state = JobState.cancelled
        except Exception as exc:
            logger.exception(f"Processing job {processing_job.name!r} failed")
            state = JobState.failed
            error = str(exc)
        finally:
            logger.remove(sink_id)
            if state != JobState.completed:
                for path in partial_paths:
                    if path.exists():
                        rmtree(path)

        self._record_warnings(log_path, caught_warnings, processing_job.name, started)
        if state == JobState.completed:
            message = "Job completed"
        elif state == JobState.cancelled:
            message = "Job cancelled"
        else:
            message = "Job failed"
        self._emit(state, processing_job.name, message, started, error=error)
        return state

    def _run_batches(
        self,
        executor: FlowExecutor,
        resolved_job: ResolvedJob,
        batches: List[StationRateBatch],
        criteria: GatherCriteria,
        partial_paths: List[Path],
        started: float,
    ) -> None:
        """Run durable run stages before their station/rate gather stages."""
        for stage in resolved_job.stages:
            if stage.scope == "run":
                for run_path in self._run_stage_paths(batches, criteria):
                    survey, station, run = run_path.split("/", 2)
                    self._emit(
                        JobState.running,
                        resolved_job.definition.name,
                        f"Started: {stage.stage_id}",
                        started,
                        survey=survey,
                        station=station,
                        run=run,
                    )
                    executor.run_stage(
                        resolved_job.processing_job,
                        stage,
                        {
                            "project": self.project,
                            "project_path": str(self.project.project_path),
                            "reference_time": self.project.ref_time,
                            "run_batch": {
                                "survey": survey,
                                "station": station,
                                "run": run,
                            },
                        },
                    )
                continue
            for batch in batches:
                output_path = ProjectJobs(self.project).batch_output_path(
                    batch, resolved_job.definition.output_label
                )
                staging_path = output_path.with_name(f".{output_path.name}.partial")
                if staging_path.exists():
                    raise ValueError(f"Partial output already exists: {staging_path}")
                if output_path.exists():
                    if not resolved_job.definition.overwrite:
                        raise ValueError(f"Output already exists: {output_path}")
                    rmtree(output_path)
                partial_paths.append(staging_path)
                self._emit(
                    JobState.running,
                    resolved_job.definition.name,
                    f"Started: {stage.stage_id}",
                    started,
                    survey=batch.survey,
                    station=batch.station,
                    sample_rate=batch.sample_rate,
                )
                executor.run_stage(
                    resolved_job.processing_job,
                    stage,
                    {
                        "project": self.project,
                        "project_path": str(self.project.project_path),
                        "reference_time": self.project.ref_time,
                        "station_rate_batch": batch.model_dump(),
                        "criteria": criteria,
                        "staging_output_path": str(staging_path),
                    },
                )
                if staging_path.exists():
                    self._archive_job(resolved_job, staging_path, batch)
                    staging_path.replace(output_path)

    def _run_stage_paths(
        self, batches: List[StationRateBatch], criteria: GatherCriteria
    ) -> List[str]:
        """Expand target run work by one hop to required remote runs."""
        paths = {run_path for batch in batches for run_path in batch.run_paths}
        table = self.project.table.copy()
        if "station_path" not in table:
            table["station_path"] = (
                table["survey"].astype(str) + "/" + table["station"].astype(str)
            )
        for batch in batches:
            policy = criteria.resolve(batch.station_path, batch.sample_rate)
            if policy.remote_references is None:
                continue
            if policy.remote_references == "auto":
                if hasattr(self.project, "get_concurrent"):
                    remote_stations = self.project.get_concurrent(
                        batch.station_path, batch.sample_rate
                    )
                else:
                    remote_stations = sorted(
                        table[
                            (table["station_path"] != batch.station_path)
                            & np.isclose(
                                table["sample_rate"].astype(float), batch.sample_rate
                            )
                        ]["station_path"]
                        .unique()
                        .tolist()
                    )
            else:
                remote_stations = policy.remote_references
            for station_path in remote_stations:
                matching = table[
                    (table["station_path"] == station_path)
                    & np.isclose(table["sample_rate"].astype(float), batch.sample_rate)
                ]
                if matching.empty:
                    raise ValueError(
                        f"Remote station {station_path!r} has no runs at "
                        f"{batch.sample_rate:g} Hz"
                    )
                paths.update(matching["run_path"].unique().tolist())
        return sorted(paths)

    def _archive_job(
        self,
        resolved_job: ResolvedJob,
        output_path: Path,
        batch: StationRateBatch,
    ) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        archive_path = output_path / "job_info.json"
        archive = {
            "resistics_version": __version__,
            "mth5_path": str(self.project.mth5_path),
            "job_path": str(resolved_job.path),
            "flow_path": str(resolved_job.flow_path),
            "parameters_path": str(resolved_job.parameters_path),
            "criteria_path": (
                None
                if resolved_job.criteria_path is None
                else str(resolved_job.criteria_path)
            ),
            "batch": batch.model_dump(),
            "processing_job": resolved_job.processing_job.model_dump(mode="json"),
        }
        archive_path.write_text(json.dumps(archive, indent=2), encoding="utf-8")

    def _record_warnings(
        self,
        log_path: Path,
        caught_warnings: List[warnings.WarningMessage],
        job_name: str,
        started: float,
    ) -> None:
        """Persist captured Python warnings without writing them to the terminal."""
        if not caught_warnings:
            return
        lines = ["\nCaptured Python warnings:\n"]
        for warning in caught_warnings:
            lines.append(
                f"{warning.category.__name__}: {warning.message} "
                f"({warning.filename}:{warning.lineno})\n"
            )
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.writelines(lines)
        self._emit(
            JobState.running,
            job_name,
            f"Captured {len(caught_warnings)} runtime warning(s); see job log",
            started,
        )

    def _flow_event(self, job_name: str, event: Dict[str, Any], started: float) -> None:
        event_name = event["event"]
        state = JobState.failed if event_name == "failed" else JobState.running
        message = f"{event_name.title()}: {event.get('node_id', 'job')}"
        self._emit(
            state,
            job_name,
            message,
            started,
            node_id=event.get("node_id"),
            step_type=event.get("step_type"),
            error=event.get("error"),
        )

    def _emit(
        self,
        state: JobState,
        job_name: str,
        message: str,
        started: float,
        survey: Optional[str] = None,
        station: Optional[str] = None,
        run: Optional[str] = None,
        sample_rate: Optional[float] = None,
        node_id: Optional[str] = None,
        step_type: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        if self.progress_callback is None:
            return
        self.progress_callback(
            JobProgressEvent(
                state=state,
                message=message,
                job_name=job_name,
                survey=survey,
                station=station,
                run=run,
                sample_rate=sample_rate,
                node_id=node_id,
                step_type=step_type,
                error=error,
                elapsed_seconds=monotonic() - started,
            )
        )
