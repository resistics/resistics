"""Discovery, validation, and execution of project processing jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from shutil import rmtree
from threading import Event
from time import monotonic
from typing import Any, Callable, Dict, List, Optional, Union
import warnings

from loguru import logger
from pydantic import BaseModel, Field

from resistics import __version__
from resistics.flow import (
    FlowCancelled,
    FlowDefinition,
    FlowExecutor,
    FlowValidator,
    ParameterSet,
    ProcessingJob,
    builtin_step_registry,
    model_from_yaml_file,
)
from resistics.project import Project, get_results_path


class JobDefinition(BaseModel):
    """Human-authored job referencing project flow and parameter files."""

    name: str
    flow: str
    parameters: str
    runtime: "JobRuntime"
    output_label: str = "result"


class JobRuntime(BaseModel):
    """MTH5 data selection authored for one processing job."""

    survey: str
    station: str
    run: str
    channels: List[str] = Field(default_factory=lambda: ["Ex", "Ey", "Hx", "Hy"])
    from_time: Optional[str] = None
    to_time: Optional[str] = None
    remote_reference: Optional[str] = None


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
    output_path: Path


class JobValidation(BaseModel):
    """Validation result with an optional resolved job."""

    ok: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    resolved_job: Optional[ResolvedJob] = None


ProgressCallback = Callable[[JobProgressEvent], None]


class ProjectJobs:
    """Read-only access to job definitions belonging to one project."""

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
            runtime = definition.runtime.model_dump(exclude_none=True)
            runtime["project_path"] = str(self.project.project_path)
            processing_job = ProcessingJob(
                name=definition.name,
                flow=flow,
                parameters=parameters,
                runtime=runtime,
                output_label=definition.output_label,
            )
            output_path = self._output_path(processing_job)
            resolved = ResolvedJob(
                path=job_path,
                definition=definition,
                processing_job=processing_job,
                flow_path=flow_path,
                parameters_path=parameters_path,
                output_path=output_path,
            )
        except Exception as exc:
            return JobValidation(ok=False, errors=[str(exc)])

        flow_validation = FlowValidator(builtin_step_registry()).validate(
            processing_job
        )
        errors.extend(flow_validation.errors)
        warnings.extend(flow_validation.warnings)
        errors.extend(self._validate_runtime(processing_job.runtime))

        if definition.runtime.remote_reference:
            errors.append(
                "Remote-reference gathering is not available for MTH5 jobs yet"
            )
        if output_path.exists():
            errors.append(f"Output already exists: {output_path}")

        return JobValidation(
            ok=not errors,
            errors=errors,
            warnings=warnings,
            resolved_job=resolved,
        )

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

    def _validate_runtime(self, runtime: Dict[str, Any]) -> List[str]:
        required = ["survey", "station", "run"]
        values = [runtime.get(key) for key in required]
        if any(value in (None, "") for value in values):
            return []
        run_path = "/".join(str(runtime[key]) for key in required)
        if run_path not in self.project.runs:
            return [f"MTH5 run not found in project: {run_path}"]
        return []

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


@dataclass
class _PipelineValue:
    """Internal processing value with shared intermediate context."""

    data: Any
    context: Dict[str, Any] = field(default_factory=dict)


class _BuiltinHandlers:
    """Adapters from flow steps to the existing numerical processing classes."""

    def __init__(
        self, project: Project, resolved_job: ResolvedJob, staging_output_path: Path
    ):
        self.project = project
        self.resolved_job = resolved_job
        self.staging_output_path = staging_output_path

    def as_dict(self) -> Dict[str, Callable]:
        """Return handlers keyed by built-in flow step type."""
        return {
            "mth5_read": self.read,
            "time_processors": self.time_processors,
            "decimate": self.decimate,
            "window": self.window,
            "fft": self.fft,
            "evals": self.evals,
            "calibrate": self.calibrate,
            "gather": self.gather,
            "solve_tf": self.solve,
            "write_results": self.write,
        }

    @staticmethod
    def _first(inputs: Dict[str, _PipelineValue]) -> _PipelineValue:
        return next(iter(inputs.values()))

    def read(self, inputs, parameters, runtime):
        del inputs
        data = self.project.read_run(
            runtime["survey"],
            runtime["station"],
            runtime["run"],
            chans=runtime["channels"],
            from_time=runtime.get("from_time"),
            to_time=runtime.get("to_time"),
        )
        return _PipelineValue(data=data)

    def time_processors(self, inputs, parameters, runtime):
        from resistics.time import InterpolateNans, RemoveMean

        del runtime
        value = self._first(inputs)
        data = value.data
        if parameters["interpolate_nans"]:
            data = InterpolateNans().run(data)
        if parameters["remove_mean"]:
            data = RemoveMean().run(data)
        return _PipelineValue(data=data, context=value.context)

    def decimate(self, inputs, parameters, runtime):
        from resistics.decimate import DecimationSetup, Decimator

        del runtime
        value = self._first(inputs)
        setup = DecimationSetup(
            n_levels=parameters["n_levels"],
            per_level=parameters["per_level"],
            div_factor=parameters["div_factor"],
            min_samples=parameters["min_samples"],
        )
        decimation_parameters = setup.run(value.data.metadata.fs)
        data = Decimator().run(decimation_parameters, value.data)
        context = dict(value.context, decimation_parameters=decimation_parameters)
        return _PipelineValue(data=data, context=context)

    def window(self, inputs, parameters, runtime):
        from resistics.window import WindowSetup, Windower

        del runtime
        value = self._first(inputs)
        setup = WindowSetup(
            min_size=parameters["min_size"],
            min_olap=parameters["min_olap"],
            win_factor=parameters["win_factor"],
            olap_proportion=parameters["overlap"],
            min_n_wins=parameters["min_n_wins"],
        )
        window_parameters = setup.run(
            value.data.metadata.n_levels, value.data.metadata.fs
        )
        data = Windower().run(self.project.ref_time, window_parameters, value.data)
        return _PipelineValue(data=data, context=value.context)

    def fft(self, inputs, parameters, runtime):
        from resistics.spectra import FourierTransform

        del runtime
        value = self._first(inputs)
        data = FourierTransform(win_fnc=parameters["window_type"]).run(value.data)
        return _PipelineValue(data=data, context=value.context)

    def evals(self, inputs, parameters, runtime):
        from resistics.spectra import EvaluationFreqs

        del parameters, runtime
        value = self._first(inputs)
        decimation_parameters = value.context["decimation_parameters"]
        data = EvaluationFreqs().run(decimation_parameters, value.data)
        return _PipelineValue(data=data, context=value.context)

    def calibrate(self, inputs, parameters, runtime):
        value = self._first(inputs)
        if not parameters["enabled"]:
            return value
        del runtime
        raise NotImplementedError(
            "MTH5 response calibration is not implemented yet; disable the "
            "calibrate node for this job"
        )

    def gather(self, inputs, parameters, runtime):
        from resistics.gather import QuickGather
        from resistics.transfunc import ImpedanceTensor

        del parameters
        value = self._first(inputs)
        transfer_function = ImpedanceTensor()
        run_path = Path(runtime["survey"]) / runtime["station"] / runtime["run"]
        data = QuickGather().run(
            run_path,
            value.context["decimation_parameters"],
            transfer_function,
            value.data,
        )
        context = dict(value.context, transfer_function=transfer_function)
        return _PipelineValue(data=data, context=context)

    def solve(self, inputs, parameters, runtime):
        from resistics.regression import RegressionPreparerGathered, SolverOLS

        del parameters, runtime
        value = self._first(inputs)
        transfer_function = value.context["transfer_function"]
        regression_input = RegressionPreparerGathered().run(
            transfer_function, value.data
        )
        solution = SolverOLS().run(regression_input)
        return _PipelineValue(data=solution, context=value.context)

    def write(self, inputs, parameters, runtime):
        del parameters, runtime
        value = self._first(inputs)
        self.staging_output_path.mkdir(parents=True, exist_ok=False)
        value.data.write(self.staging_output_path / "solution.json")
        return _PipelineValue(
            data={"result_path": str(self.resolved_job.output_path)},
            context=value.context,
        )


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
        """Run a resolved job synchronously."""
        processing_job = resolved_job.processing_job
        started = monotonic()
        staging_output_path = resolved_job.output_path.with_name(
            f".{resolved_job.output_path.name}.partial"
        )
        if resolved_job.output_path.exists():
            self._emit(
                JobState.failed,
                processing_job.name,
                f"Output already exists: {resolved_job.output_path}",
                started,
            )
            return JobState.failed
        if staging_output_path.exists():
            self._emit(
                JobState.failed,
                processing_job.name,
                f"Partial output already exists: {staging_output_path}",
                started,
            )
            return JobState.failed
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
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                executor = FlowExecutor(
                    builtin_step_registry(),
                    handlers=self._handlers(resolved_job, staging_output_path),
                    progress_callback=lambda event: self._flow_event(
                        processing_job.name, event, started
                    ),
                    cancellation_callback=self._cancel_event.is_set,
                )
                executor.run(processing_job)
                self._archive_job(resolved_job, staging_output_path)
                staging_output_path.replace(resolved_job.output_path)
        except FlowCancelled:
            state = JobState.cancelled
        except Exception as exc:
            logger.exception(f"Processing job {processing_job.name!r} failed")
            state = JobState.failed
            error = str(exc)
        finally:
            logger.remove(sink_id)
            if state != JobState.completed and staging_output_path.exists():
                rmtree(staging_output_path)

        self._record_warnings(log_path, caught_warnings, processing_job.name, started)
        if state == JobState.completed:
            message = "Job completed"
        elif state == JobState.cancelled:
            message = "Job cancelled"
        else:
            message = "Job failed"
        self._emit(state, processing_job.name, message, started, error=error)
        return state

    def _handlers(
        self, resolved_job: ResolvedJob, staging_output_path: Path
    ) -> Dict[str, Callable]:
        return _BuiltinHandlers(
            self.project, resolved_job, staging_output_path
        ).as_dict()

    def _archive_job(self, resolved_job: ResolvedJob, output_path: Path) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        archive_path = output_path / "job_info.json"
        archive = {
            "resistics_version": __version__,
            "mth5_path": str(self.project.mth5_path),
            "job_path": str(resolved_job.path),
            "flow_path": str(resolved_job.flow_path),
            "parameters_path": str(resolved_job.parameters_path),
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
                node_id=node_id,
                step_type=step_type,
                error=error,
                elapsed_seconds=monotonic() - started,
            )
        )
