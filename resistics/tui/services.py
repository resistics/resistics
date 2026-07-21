"""UI-neutral helpers shared by the terminal application screens."""

from __future__ import annotations

from asyncio import get_running_loop
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import TYPE_CHECKING, Literal, TypeVar

from textual.screen import Screen
from textual.widget import Widget

_WorkerValue = TypeVar("_WorkerValue")

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pydantic import BaseModel

    from resistics.explorer import (
        IndexedJob,
        IndexedResource,
        IndexSection,
        ProjectExplorerIndex,
        ProjectExplorerState,
        ResourceKind,
    )
    from resistics.job import (
        JobDefinition,
        JobProgressEvent,
        JobRunner,
        JobValidation,
        ProjectJobs,
    )
    from resistics.project import (
        Project,
        ProjectDataDeletion,
        ProjectDataItem,
        RunSummary,
    )
    from resistics.tui.app import ResisticsTui
    from resistics.tui.state import PlotTarget, TimePlotSelection


def _feature_error(feature: str, error: Exception) -> str:
    """Return actionable detail for a lazily imported feature failure.

    Parameters
    ----------
    feature : str
        User-facing feature name.
    error : Exception
        Import or runtime failure raised at the feature boundary.

    Returns
    -------
    str
        Error detail that preserves ordinary failures and explains how to
        recover from a missing required dependency.
    """
    if isinstance(error, ModuleNotFoundError):
        dependency = error.name or "unknown"
        return (
            f"{feature} requires the missing dependency {dependency!r}. "
            "Reinstall resistics with its required dependencies."
        )
    return str(error)


async def _run_in_worker_thread(
    operation: Callable[[], _WorkerValue],
) -> _WorkerValue:
    """Run blocking work in a dedicated thread owned by one Textual worker.

    The executor is not shared with the asyncio event loop, so cancelling a
    Textual worker never blocks the UI while a synchronous API finishes.

    Parameters
    ----------
    operation : Callable[[], _WorkerValue]
        Blocking callable that does not mutate Textual widgets.

    Returns
    -------
    _WorkerValue
        Value returned by the blocking callable.
    """
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="resistics-tui")
    try:
        return await get_running_loop().run_in_executor(executor, operation)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _focus_relative(
    controls: Sequence[Widget],
    focused: Widget | None,
    increment: int,
    *,
    move_from_unfocused: bool = True,
) -> None:
    """Move focus within a bounded group of controls.

    Parameters
    ----------
    controls : Sequence[Widget]
        Ordered focusable widgets.
    focused : Widget | None
        Currently focused widget, if any.
    increment : int
        Relative movement through the group.
    move_from_unfocused : bool
        Whether to use the first control as the fallback position.
    """
    if not controls:
        return
    if focused is None or focused not in controls:
        if not move_from_unfocused:
            return
        index = 0
    else:
        index = controls.index(focused)
    controls[(index + increment) % len(controls)].focus()


def _resistics_app(screen: Screen[None]) -> ResisticsTui:
    """Return the application contract required by project screens.

    Parameters
    ----------
    screen : Screen[None]
        Project screen mounted by the application.

    Returns
    -------
    ResisticsTui
        Owning resistics application.

    Raises
    ------
    RuntimeError
        If the screen is mounted by an incompatible Textual application.
    """
    from resistics.tui.app import ResisticsTui

    app = screen.app
    if not isinstance(app, ResisticsTui):
        raise RuntimeError("Project screens require ResisticsTui")
    return app


class ProjectExplorerService:
    """Own project operations required by an interactive explorer.

    The service contains no Textual objects and can be exercised from scripts,
    notebooks, tests, or another presentation layer. Public discovery methods
    return the frozen Pydantic DTOs owned by :mod:`resistics.explorer`; mutable
    runner and lifecycle records remain private implementation state.

    Parameters
    ----------
    project : Project
        Open project used for discovery and project-owned mutations.
    """

    def __init__(self, project: Project):
        from resistics.explorer import ProjectExplorerIndex

        self.project = project
        self.index: ProjectExplorerIndex = ProjectExplorerIndex(project)
        self._lifecycle_lock = Lock()
        self._active_discoveries = 0
        self._close_requested = False
        self._project_closed = False
        self._job_runner: JobRunner | None = None

    @property
    def project_jobs(self) -> ProjectJobs:
        """Return the project job repository owned by the explorer index."""
        return self.index.project_jobs

    @property
    def job_runner(self) -> JobRunner | None:
        """Return the active job runner, when processing is underway."""
        return self._job_runner

    @property
    def active_discoveries(self) -> int:
        """Return the number of synchronous discovery calls still active."""
        with self._lifecycle_lock:
            return self._active_discoveries

    @contextmanager
    def discovery(self) -> Iterator[None]:
        """Protect one discovery operation from premature project closure.

        Raises
        ------
        RuntimeError
            If project closure was already requested.
        """
        with self._lifecycle_lock:
            if self._close_requested:
                raise RuntimeError("Project screen closed")
            self._active_discoveries += 1
        try:
            yield
        finally:
            close_project = False
            with self._lifecycle_lock:
                self._active_discoveries -= 1
                if (
                    self._close_requested
                    and self._active_discoveries == 0
                    and not self._project_closed
                ):
                    self._project_closed = True
                    close_project = True
            if close_project:
                self.project.close_mth5()

    def close(self) -> None:
        """Close the project now or after already-running discovery exits."""
        close_project = False
        with self._lifecycle_lock:
            self._close_requested = True
            if self._active_discoveries == 0 and not self._project_closed:
                self._project_closed = True
                close_project = True
        if close_project:
            self.project.close_mth5()

    def project_state(self) -> ProjectExplorerState:
        """Return cached project discovery as a frozen Pydantic DTO."""
        return self.index.project_state()

    def runs(self) -> tuple[RunSummary, ...]:
        """Return cached MTH5 run summaries used for selections."""
        return self.index.runs()

    def resources(self, kind: ResourceKind) -> tuple[IndexedResource, ...]:
        """Return cached processing resources for one namespace."""
        return self.index.resources(kind)

    def jobs(self) -> tuple[IndexedJob, ...]:
        """Return cached job summaries and validation results."""
        return self.index.jobs()

    def job_validation(self, path: Path) -> JobValidation | None:
        """Return cached validation for one exact job path."""
        return self.index.job_validation(path)

    def resource_for_path(
        self, kind: ResourceKind, path: Path
    ) -> IndexedResource | None:
        """Return one cached resource by exact path."""
        return self.index.resource_for_path(kind, path)

    def invalidate(self, *sections: IndexSection) -> None:
        """Invalidate project-index sections after an owned mutation."""
        self.index.invalidate(*sections)

    def invalidate_all(self) -> None:
        """Invalidate every project-index section after an external change."""
        self.index.invalidate_all()

    def job_resource_options(
        self, resource_type: ResourceKind
    ) -> list[tuple[str, str]]:
        """Return valid resources as form labels and exact filenames."""
        options = []
        for resource in self.resources(resource_type):
            if resource.model is None:
                continue
            display_name = getattr(resource.model, "name", None) or resource.path.stem
            options.append(
                (f"{display_name} ({resource.path.name})", resource.path.name)
            )
        return options

    def job_template_names(self) -> set[str]:
        """Return existing job filename stems for duplicate-name checks."""
        return {resource.path.stem for resource in self.resources("jobs")}

    def create_job_template(self, definition: JobDefinition) -> Path:
        """Create one editable job definition and invalidate job discovery."""
        path = self.project_jobs.create_template(definition)
        self.invalidate("jobs")
        return path

    @staticmethod
    def yaml_source(path: Path) -> str:
        """Return YAML source exactly as authored, including comments."""
        return path.read_text(encoding="utf-8")

    def copy_yaml(self, source: Path, name: str, kind: ResourceKind) -> Path:
        """Copy a YAML resource verbatim without replacing an existing file."""
        from resistics.tui.screens.dialogs import _validate_yaml_file_stem

        name = _validate_yaml_file_stem(name)
        for suffix in (".yaml", ".yml"):
            if (source.parent / f"{name}{suffix}").exists():
                raise ValueError(f"A YAML file named {name!r} already exists")
        destination = source.with_name(f"{name}{source.suffix}")
        with source.open("rb") as input_file, destination.open("xb") as output_file:
            output_file.write(input_file.read())
        self.invalidate(kind)
        return destination

    def delete_yaml(self, source: Path, kind: ResourceKind) -> None:
        """Delete a YAML resource and invalidate its cached namespace."""
        source.unlink()
        self.invalidate(kind)

    def save_yaml(
        self,
        path: Path,
        model_type: type[BaseModel],
        content: str,
        kind: ResourceKind,
    ) -> None:
        """Validate and atomically replace one project YAML resource."""
        self.validate_yaml(model_type, content)
        self.write_yaml(path, content, kind)

    @staticmethod
    def validate_yaml(model_type: type[BaseModel], content: str) -> None:
        """Validate authored YAML against its expected Pydantic model."""
        from resistics.flow import model_from_yaml

        model_from_yaml(model_type, content)

    def write_yaml(self, path: Path, content: str, kind: ResourceKind) -> None:
        """Atomically replace one validated YAML resource and invalidate it."""
        temporary_path = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_file.write(content)
                temporary_path = Path(temporary_file.name)
            temporary_path.replace(path)
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
        self.invalidate(kind)

    def restore_templates(self, kind: ResourceKind) -> list[Path]:
        """Restore missing built-in resources for one supported namespace."""
        from resistics.templates import (
            install_builtin_criteria_templates,
            install_builtin_flow_templates,
            install_builtin_parameter_templates,
        )

        installers = {
            "flows": install_builtin_flow_templates,
            "parameters": install_builtin_parameter_templates,
            "criteria": install_builtin_criteria_templates,
        }
        installer = installers.get(kind)
        if installer is None:
            raise ValueError(f"No built-in templates exist for {kind}")
        installed = installer(self.project.project_path)
        self.invalidate(kind)
        return installed

    def deletion_options(self) -> tuple[list[str], ProjectDataDeletion]:
        """Return derived-data labels and a complete deletion preview."""
        return (
            self.project.list_project_output_labels(),
            self.project.preview_project_data_deletion(),
        )

    def preview_project_data_deletion(
        self, output_label: str | None
    ) -> ProjectDataDeletion:
        """Preview derived project data removal for one output label."""
        return self.project.preview_project_data_deletion(output_label)

    def delete_project_data(self, output_label: str | None) -> ProjectDataDeletion:
        """Delete derived project data and invalidate project discovery."""
        deleted = self.project.delete_project_data(output_label)
        self.invalidate("project")
        return deleted

    @staticmethod
    def mth5_time_run_path(item: ProjectDataItem) -> str | None:
        """Return an MTH5 item's canonical run path, when represented."""
        parts = [part for part in item.path.split("/") if part]
        lower_parts = [part.lower() for part in parts]
        try:
            station_index = lower_parts.index("stations")
            run_index = station_index + 2
            if lower_parts[run_index] == "runs":
                run_index += 1
            if run_index >= len(parts):
                return None
            return "/" + "/".join(parts[: run_index + 1])
        except (ValueError, IndexError):
            return None

    @classmethod
    def data_category_count(
        cls, source_data: object, items: list[ProjectDataItem], data_type: str
    ) -> int:
        """Count displayed data, treating MTH5 time channels as one run."""
        if source_data != ("mth5", "/") or data_type != "time":
            return sum(item.is_dataset for item in items)
        run_paths = {
            run_path
            for item in items
            if (run_path := cls.mth5_time_run_path(item)) is not None
        }
        return len(run_paths) if run_paths else sum(item.is_dataset for item in items)

    @staticmethod
    def visible_data_paths(items: list[ProjectDataItem], data_type: str) -> set[str]:
        """Return matching data entries and their represented ancestors."""
        by_path = {item.path: item for item in items}
        visible = set()
        for item in items:
            if data_type != item.data_type:
                continue
            current: ProjectDataItem | None = item
            while current is not None:
                visible.add(current.path)
                current = (
                    None
                    if current.parent_path is None
                    else by_path.get(current.parent_path)
                )
        return visible

    def _project_data_path(self, path: str) -> Path | None:
        data_root = (self.project.project_path / "data").resolve()
        item_path = (data_root / path).resolve()
        if item_path != data_root and data_root not in item_path.parents:
            return None
        return item_path

    def _find_project_artifact(
        self, item: ProjectDataItem, required_files: tuple[str, ...]
    ) -> Path | None:
        item_path = self._project_data_path(item.path)
        if item_path is None:
            return None
        data_root = (self.project.project_path / "data").resolve()
        current = item_path if item_path.is_dir() else item_path.parent
        while current != data_root.parent:
            if all((current / filename).is_file() for filename in required_files):
                return current
            if current == data_root:
                break
            current = current.parent
        return None

    @staticmethod
    def _load_solution(solution_path: Path):
        try:
            from resistics.regression import Solution

            return Solution.model_validate_json(solution_path.read_bytes())
        except Exception:
            return None

    def _mth5_time_plot_target(
        self, item: ProjectDataItem
    ) -> tuple[Literal["time"], TimePlotSelection] | None:
        from mth5.helpers import validate_name as validate_mth5_name

        parts = [part for part in item.path.split("/") if part]
        lower_parts = [part.lower() for part in parts]
        try:
            survey_index = lower_parts.index("surveys")
            station_index = lower_parts.index("stations")
            survey = parts[survey_index + 1]
            station = parts[station_index + 1]
            run_index = station_index + 2
            if lower_parts[run_index] == "runs":
                run_index += 1
            run = parts[run_index]
        except (ValueError, IndexError):
            return None
        matches = [
            summary
            for summary in self.runs()
            if (
                validate_mth5_name(summary.survey) == survey
                and validate_mth5_name(summary.station) == station
                and validate_mth5_name(summary.run) == run
            )
        ]
        if len(matches) != 1:
            return None
        summary = matches[0]
        channel = None
        if item.kind == "dataset":
            channel_parts = parts[run_index + 1 :]
            if channel_parts and channel_parts[0].lower() == "channels":
                channel_parts = channel_parts[1:]
            if len(channel_parts) != 1:
                return None
            channel = channel_parts[0]
        return ("time", (summary.survey, summary.station, summary.run, channel))

    def data_plot_target(self, item: ProjectDataItem | None) -> PlotTarget | None:
        """Resolve a supported plot target from one project data item."""
        if item is None:
            return None
        if item.source == "mth5" and item.data_type == "time":
            return self._mth5_time_plot_target(item)
        if item.source != "project":
            return None
        if item.data_type == "spectra":
            artifact = self._find_project_artifact(item, ("metadata.json", "data.npz"))
            return None if artifact is None else ("spectra", artifact)
        if item.data_type == "transfer_function":
            artifact = self._find_project_artifact(item, ("solution.json",))
            if artifact is None:
                return None
            solution_path = artifact / "solution.json"
            solution = self._load_solution(solution_path)
            from resistics.transfunc import ImpedanceTensor, Tipper

            if solution is None or not isinstance(
                solution.tf, (ImpedanceTensor, Tipper)
            ):
                return None
            return ("transfer_function", solution_path)
        return None

    @staticmethod
    def build_plot_figure(project: Project, target: PlotTarget):  # noqa: C901
        """Build a selected Plotly figure through existing plotting APIs."""
        from resistics.tui.state import TIME_PLOT_MAX_POINTS

        target_type, payload = target
        if target_type == "flow":
            from resistics.flow import FlowDefinition, model_from_yaml_file
            from resistics.plot import plot_flow

            if not isinstance(payload, Path):
                raise ValueError("A flow plot requires a YAML path")
            flow = model_from_yaml_file(FlowDefinition, payload)
            return plot_flow(flow, project.project_path)
        if target_type == "job":
            from resistics.job import ProjectJobs
            from resistics.plot import plot_job

            if not isinstance(payload, Path):
                raise ValueError("A job plot requires a YAML path")
            validation = ProjectJobs(project).validate(payload)
            if not validation.ok or validation.resolved_job is None:
                raise ValueError("; ".join(validation.errors) or "Job is invalid")
            return plot_job(validation.resolved_job, project.project_path)
        if target_type == "project":
            return project.plot()
        if target_type == "time":
            if not isinstance(payload, tuple) or len(payload) != 4:
                raise ValueError("A time plot requires survey, station, run, channel")
            survey, station, run, channel = payload
            if not all(isinstance(value, str) for value in (survey, station, run)) or (
                channel is not None and not isinstance(channel, str)
            ):
                raise ValueError("A time plot requires string run identifiers")
            time_data = project.read_run(
                survey,
                station,
                run,
                chans=None if channel is None else [channel],
            )
            return time_data.plot(max_pts=TIME_PLOT_MAX_POINTS)
        if target_type == "spectra":
            from resistics.spectra import SpectraDataReader, SpectraMetadata

            if not isinstance(payload, Path):
                raise ValueError("A spectra plot requires a data path")
            spectra_data = SpectraDataReader().run(payload)
            if isinstance(spectra_data, SpectraMetadata):
                raise ValueError("A spectra plot requires array data")
            return spectra_data.plot()
        if target_type == "transfer_function":
            from resistics.regression import Solution
            from resistics.transfunc import ImpedanceTensor, Tipper

            if not isinstance(payload, Path):
                raise ValueError("A transfer-function plot requires a solution path")
            solution = Solution.model_validate_json(payload.read_bytes())
            if isinstance(solution.tf, (ImpedanceTensor, Tipper)):
                return solution.tf.plot(solution.freqs, solution.components)
        raise ValueError("The selected data is not plottable")

    def run_job(
        self,
        validation: JobValidation,
        progress_callback: Callable[[JobProgressEvent], None],
    ) -> None:
        """Execute a validated job with structured progress and safe closure."""
        from resistics.job import JobProgressEvent, JobRunner, JobState
        from resistics.project import load

        if validation.resolved_job is None:
            raise ValueError("A resolved job is required for execution")
        processing_project = None
        try:
            processing_project = load(self.project.project_path)
            self._job_runner = JobRunner(
                processing_project, progress_callback=progress_callback
            )
            self._job_runner.run(validation.resolved_job)
        except Exception as exc:
            progress_callback(
                JobProgressEvent(
                    state=JobState.failed,
                    message="Unable to start job",
                    job_name=validation.resolved_job.definition.name,
                    error=_feature_error("Job processing", exc),
                )
            )
        finally:
            self._job_runner = None
            if processing_project is not None:
                processing_project.close_mth5()

    def cancel_job(self) -> bool:
        """Request cancellation from the active runner, if one exists."""
        runner = self._job_runner
        if runner is None:
            return False
        runner.cancel()
        return True
