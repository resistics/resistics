"""Textual application for creating, inspecting, and processing projects."""

from __future__ import annotations

import json
import sys
import warnings
from asyncio import get_running_loop
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from time import monotonic
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, TypeVar

from loguru import logger
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    DataTable,
    Footer,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
    Tree,
)
from textual.worker import Worker, WorkerState

from resistics.tui.screens.dialogs import (
    ConfirmJobScreen,
    ConfirmProjectDataDeletionScreen,
    CopyYamlFileScreen,
    CreateJobScreen,
    DeleteProjectDataScreen,
    DeleteYamlFileScreen,
    ProjectDataDeletionRequest,
    _validate_yaml_file_stem,
)
from resistics.tui.screens.dialogs import (
    DirectoryPickerScreen as DirectoryPickerScreen,
)
from resistics.tui.screens.launcher import (
    CreateProjectScreen,
    HomeScreen,
    ProjectLoadingScreen,
    TuiHeader,
)
from resistics.tui.services import _feature_error, _resistics_app

if TYPE_CHECKING:
    from resistics.explorer import (
        IndexedJob,
        IndexedResource,
        ProjectExplorerState,
        ResourceKind,
    )
    from resistics.job import (
        JobDefinition,
        JobProgressEvent,
        JobRunner,
        JobState,
        JobSummary,
        JobValidation,
    )
    from resistics.project import (
        Project,
        ProjectDataDeletion,
        ProjectDataItem,
        RunSummary,
    )
    from resistics.regression import Solution

TIME_PLOT_MAX_POINTS = 5_000
_WorkerValue = TypeVar("_WorkerValue")

TimePlotSelection: TypeAlias = tuple[str, str, str, str | None]
PlotTarget: TypeAlias = (
    tuple[Literal["flow", "job", "spectra", "transfer_function"], Path]
    | tuple[Literal["project"], None]
    | tuple[Literal["time"], TimePlotSelection]
)
ExplorerView: TypeAlias = Literal[
    "project", "data", "flows", "parameters", "criteria", "jobs"
]


def _progress_details(event: JobProgressEvent) -> tuple[str, str]:
    """Return the counter suffix and activity status for a job event.

    Parameters
    ----------
    event : JobProgressEvent
        Job event that may contain fine-grained process progress.

    Returns
    -------
    tuple[str, str]
        Counter suffix and complete activity-status text.
    """
    status = f"{event.job_name}: {event.state.value}"
    if event.progress is None:
        return "", status
    if event.progress.total is None:
        return f" [{event.progress.current}]", status
    counter = f"{event.progress.current}/{event.progress.total}"
    return f" [{counter}]", f"{event.job_name}: {event.progress.task} {counter}"


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


@dataclass(frozen=True)
class _ProjectOpenResult:
    """Immutable outcome returned by a project-opening thread worker.

    Attributes
    ----------
    generation : int
        App generation that requested the project.
    project_path : Path
        Project directory requested by the user.
    project : Project | None
        Open project on success.
    startup_warnings : tuple[str, ...]
        Warnings captured while opening the project.
    error : str | None
        User-facing opening error on failure.
    """

    generation: int
    project_path: Path
    project: Project | None = None
    startup_warnings: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class _ExplorerLoadResult:
    """Immutable discovery result passed from a worker to the UI thread.

    Attributes
    ----------
    generation : int
        Screen generation that requested the discovery operation.
    section : ExplorerView
        Explorer view owning the result.
    state : ProjectExplorerState | None
        Project catalogue result for Project and Data views.
    resources : tuple[IndexedResource, ...]
        Parsed YAML resources for a resource view.
    jobs : tuple[IndexedJob, ...]
        Validated jobs for the Jobs view.
    runs : tuple[RunSummary, ...]
        MTH5 run summaries preloaded for Data plot selection.
    error : str | None
        User-facing discovery error on failure.
    """

    generation: int
    section: ExplorerView
    state: ProjectExplorerState | None = None
    resources: tuple[IndexedResource, ...] = ()
    jobs: tuple[IndexedJob, ...] = ()
    runs: tuple[RunSummary, ...] = ()
    error: str | None = None


@dataclass
class _ProjectActionState:
    """Cache action eligibility that would otherwise require project I/O.

    Attributes
    ----------
    plot_targets : dict[str, PlotTarget | None]
        Cached plot targets for the Project and Data tabs.
    valid_flow_paths : set[Path]
        Flow files that passed validation during the latest tree population.
    has_project_data_to_delete : bool
        Whether the latest data catalogue contains removable derived data.
    """

    plot_targets: dict[str, PlotTarget | None] = field(
        default_factory=lambda: {"project": None, "data": None}
    )
    valid_flow_paths: set[Path] = field(default_factory=set)
    has_project_data_to_delete: bool = False


class _DataTreeNode(Protocol):
    """Tree-node operations used while constructing the data catalogue."""

    @property
    def data(self) -> object:
        return None

    def add(self, label: str, data: tuple[str, str]) -> _DataTreeNode: ...

    def add_leaf(self, label: str, data: tuple[str, str]) -> _DataTreeNode: ...


class ProjectExplorerScreen(Screen[None]):
    """Project browser and YAML editor with managed processing-job execution."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("d", "restore_defaults", "Restore defaults"),
        ("n", "create_job", "New job"),
        ("e", "edit_yaml", "Edit YAML"),
        ("y", "copy_yaml", "Copy YAML"),
        ("delete", "delete_yaml", "Delete"),
        ("ctrl+s", "save_yaml", "Save YAML"),
        Binding("escape", "discard_yaml", "Discard YAML", priority=True),
        ("j", "run_selected_job", "Run job"),
        ("p", "plot", "Plot"),
        ("right_square_bracket", "expand_data_node", "Expand"),
        ("left_square_bracket", "collapse_data_node", "Collapse"),
        ("c", "cancel_job", "Cancel job"),
        ("x", "close_project", "Close project"),
        ("q", "quit", "Quit"),
    ]

    DATA_CATEGORIES = [
        ("Time data", "time"),
        ("Spectra/evaluations", "spectra"),
        ("Masks", "mask"),
        ("Transfer functions", "transfer_function"),
        ("Other", "other"),
    ]

    def __init__(self, project: Project, startup_warnings: list[str] | None = None):
        super().__init__()
        from resistics.explorer import ProjectExplorerIndex
        from resistics.job import JobProgressEvent, JobRunner, JobState

        self.project = project
        self.explorer_index = ProjectExplorerIndex(project)
        self.project_jobs = self.explorer_index.project_jobs
        self.flow_paths: dict[str, Path] = {}
        self.parameter_paths: dict[str, Path] = {}
        self.criteria_paths: dict[str, Path] = {}
        self.selected_flow_path: Path | None = None
        self.selected_parameter_path: Path | None = None
        self.selected_criteria_path: Path | None = None
        self.job_summaries: dict[str, JobSummary] = {}
        self.selected_job_path: Path | None = None
        self.selected_validation: JobValidation | None = None
        self._pending_job_path: Path | None = None
        self.job_runner: JobRunner | None = None
        self.job_state: JobState | None = None
        self._job_progress_type = JobProgressEvent
        self._job_runner_type = JobRunner
        self._job_state_type = JobState
        self.data_items: dict[str, ProjectDataItem] = {}
        self.action_state = _ProjectActionState()
        self.editing_yaml = False
        self.editing_path: Path | None = None
        self.editing_model = None
        self.editing_editor_id: str | None = None
        self.startup_warnings = startup_warnings or []
        self._load_generation = 0
        self._loaded_sections: set[ExplorerView] = set()
        self._loading_sections: set[ExplorerView] = set()
        self._load_state_lock = Lock()
        self._active_discoveries = 0
        self._close_requested = False
        self._project_closed = False

    def compose(self) -> ComposeResult:
        yield TuiHeader(id="app-header")
        with TabbedContent(initial="project"):
            with TabPane("Project", id="project"):
                with VerticalScroll(classes="pane"):
                    yield Static(id="project-content")
            with TabPane("Data", id="data"):
                with Horizontal(classes="pane split"):
                    with Vertical(classes="left"):
                        data_tree = Tree("Data", id="data-tree")
                        data_tree.show_root = False
                        yield data_tree
                    with Vertical(classes="right"):
                        yield TextArea.code_editor(
                            '{\n  "message": "Select Project or MTH5 data"\n}',
                            language="json",
                            theme="vscode_dark",
                            read_only=True,
                            id="data-metadata",
                        )
            with TabPane("Flows", id="flows"), Vertical(classes="pane"):
                with Horizontal(classes="split"):
                    with Vertical(classes="left"):
                        yield DataTable(id="flow-table", cursor_type="row")
                    with Vertical(classes="right"):
                        yield TextArea.code_editor(
                            "Select a flow",
                            language="yaml",
                            theme="vscode_dark",
                            read_only=True,
                            id="flow-content",
                        )
            with TabPane("Parameters", id="parameters"):
                with Vertical(classes="pane"):
                    with Horizontal(classes="split"):
                        with Vertical(classes="left"):
                            yield DataTable(id="parameter-table", cursor_type="row")
                        with Vertical(classes="right"):
                            yield TextArea.code_editor(
                                "Select a parameter set",
                                language="yaml",
                                theme="vscode_dark",
                                read_only=True,
                                id="parameter-content",
                            )
            with TabPane("Criteria", id="criteria"):
                with Vertical(classes="pane"):
                    with Horizontal(classes="split"):
                        with Vertical(classes="left"):
                            yield DataTable(id="criteria-table", cursor_type="row")
                        with Vertical(classes="right"):
                            yield TextArea.code_editor(
                                "Select a criteria file",
                                language="yaml",
                                theme="vscode_dark",
                                read_only=True,
                                id="criteria-content",
                            )
            with TabPane("Jobs", id="jobs"), Vertical(classes="pane"):
                with Horizontal(classes="split"):
                    with Vertical(classes="left"):
                        yield DataTable(id="job-table", cursor_type="row")
                    with Vertical(classes="right"):
                        yield TextArea.code_editor(
                            "Select a job",
                            language="yaml",
                            theme="vscode_dark",
                            read_only=True,
                            id="job-content",
                        )
            with TabPane("Activity", id="activity"):
                with Vertical(classes="pane"):
                    yield Static("No active job", id="activity-status")
                    yield RichLog(id="activity-log", markup=True, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#project-content", Static).update(
            "[bold]Loading project overview…[/bold]"
        )
        self._request_explorer_section("project")
        if self.startup_warnings:
            self.query_one("#activity-log", RichLog).write(
                f"[yellow]Suppressed {len(self.startup_warnings)} warning(s) "
                "while opening the project.[/]"
            )

    def on_unmount(self) -> None:
        self._load_generation += 1
        self.workers.cancel_group(self, "explorer-load")
        close_project = False
        with self._load_state_lock:
            self._close_requested = True
            if self._active_discoveries == 0 and not self._project_closed:
                self._project_closed = True
                close_project = True
        if close_project:
            self.project.close_mth5()

    @work(group="explorer-load", exit_on_error=False)
    async def _load_explorer_section(
        self, generation: int, section: ExplorerView
    ) -> _ExplorerLoadResult:
        """Load one explorer section without touching Textual widgets.

        Parameters
        ----------
        generation : int
            Screen generation requesting the load.
        section : ExplorerView
            Explorer section to discover.

        Returns
        -------
        _ExplorerLoadResult
            Immutable result consumed on the Textual UI thread.
        """
        return await _run_in_worker_thread(
            partial(self._discover_explorer_section, generation, section)
        )

    def _discover_explorer_section(
        self, generation: int, section: ExplorerView
    ) -> _ExplorerLoadResult:
        """Perform one synchronous discovery operation in a worker thread.

        Parameters
        ----------
        generation : int
            Screen generation requesting the load.
        section : ExplorerView
            Explorer section to discover.

        Returns
        -------
        _ExplorerLoadResult
            Immutable success or failure result containing no widgets.
        """
        with self._load_state_lock:
            if self._close_requested:
                return _ExplorerLoadResult(
                    generation=generation,
                    section=section,
                    error="Project screen closed",
                )
            self._active_discoveries += 1
        try:
            if section in {"project", "data"}:
                state = self.explorer_index.project_state()
                runs = self.explorer_index.runs() if section == "data" else ()
                return _ExplorerLoadResult(
                    generation=generation,
                    section=section,
                    state=state,
                    runs=runs,
                )
            if section == "jobs":
                return _ExplorerLoadResult(
                    generation=generation,
                    section=section,
                    jobs=self.explorer_index.jobs(),
                )
            return _ExplorerLoadResult(
                generation=generation,
                section=section,
                resources=self.explorer_index.resources(section),
            )
        except Exception as exc:
            return _ExplorerLoadResult(
                generation=generation,
                section=section,
                error=str(exc),
            )
        finally:
            close_project = False
            with self._load_state_lock:
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

    def _request_explorer_section(
        self, section: ExplorerView, *, force: bool = False
    ) -> None:
        """Start one lazy section load unless its current generation is ready.

        Parameters
        ----------
        section : ExplorerView
            Explorer view whose cached data is required.
        force : bool
            Reload a section even when it is already marked ready.
        """
        if not force and (
            section in self._loaded_sections or section in self._loading_sections
        ):
            return
        self._loaded_sections.discard(section)
        self._loading_sections.add(section)
        self._show_section_loading(section)
        self._load_explorer_section(self._load_generation, section)

    def _show_section_loading(self, section: ExplorerView) -> None:
        """Render a lightweight loading placeholder without performing I/O.

        Parameters
        ----------
        section : ExplorerView
            Explorer section whose content is loading.
        """
        if section == "project":
            self.query_one("#project-content", Static).update(
                "[bold]Loading project overview…[/bold]"
            )
        elif section == "data":
            tree = self.query_one("#data-tree", Tree)
            tree.clear()
            tree.root.add_leaf("Loading project data…")
        else:
            editor_ids = {
                "flows": "#flow-content",
                "parameters": "#parameter-content",
                "criteria": "#criteria-content",
                "jobs": "#job-content",
            }
            self.query_one(editor_ids[section], TextArea).text = "Loading…"

    @on(Worker.StateChanged)
    def _apply_explorer_worker_result(self, event: Worker.StateChanged) -> None:
        """Apply successful current-generation discovery on the UI thread.

        Parameters
        ----------
        event : Worker.StateChanged
            Textual lifecycle event for a discovery worker.
        """
        if event.state != WorkerState.SUCCESS or event.worker.group != "explorer-load":
            return
        result = event.worker.result
        if not isinstance(result, _ExplorerLoadResult):
            return
        if result.generation != self._load_generation:
            return
        self._loading_sections.discard(result.section)
        if result.error is not None:
            self._show_section_error(result.section, result.error)
            self.refresh_bindings()
            return
        with self.app.batch_update():
            self._render_explorer_result(result)
        self._loaded_sections.add(result.section)
        self.refresh_bindings()

    def _render_explorer_result(self, result: _ExplorerLoadResult) -> None:
        """Mutate widgets from one current worker result on the UI thread.

        Parameters
        ----------
        result : _ExplorerLoadResult
            Successful current-generation discovery result.
        """
        if result.section == "project" and result.state is not None:
            self._populate_overview(result.state)
        elif result.section == "data" and result.state is not None:
            self._populate_data_tree(result.state)
        elif result.section == "flows":
            self._populate_flows(result.resources)
        elif result.section == "parameters":
            self._populate_parameters(result.resources)
        elif result.section == "criteria":
            self._populate_criteria(result.resources)
        elif result.section == "jobs":
            self._populate_jobs(result.jobs)
            self._restore_pending_job_selection(result.jobs)

    def _restore_pending_job_selection(
        self, indexed_jobs: tuple[IndexedJob, ...]
    ) -> None:
        """Restore a job created or saved before its worker refresh completed.

        Parameters
        ----------
        indexed_jobs : tuple[IndexedJob, ...]
            Current job results returned by the discovery worker.
        """
        path = self._pending_job_path
        if path is None:
            return
        self._pending_job_path = None
        match = next(
            (job for job in indexed_jobs if job.resource.path == path),
            None,
        )
        if match is None:
            return
        self.selected_job_path = path
        self.selected_validation = match.validation
        self._show_yaml("#job-content", path)

    def _show_section_error(self, section: ExplorerView, error: str) -> None:
        """Replace one loading placeholder with a user-facing failure.

        Parameters
        ----------
        section : ExplorerView
            Explorer section whose load failed.
        error : str
            Failure detail returned by the worker.
        """
        message = f"Unable to load {section}: {error}"
        if section == "project":
            self.query_one("#project-content", Static).update(f"[red]{message}[/red]")
        elif section == "data":
            tree = self.query_one("#data-tree", Tree)
            tree.clear()
            tree.root.add_leaf(message)
        else:
            editor_ids = {
                "flows": "#flow-content",
                "parameters": "#parameter-content",
                "criteria": "#criteria-content",
                "jobs": "#job-content",
            }
            self.query_one(editor_ids[section], TextArea).text = message
        self.notify(message, severity="error")

    def _start_new_load_generation(self) -> None:
        """Reject outstanding results before a cache invalidation transition."""
        self._load_generation += 1
        self._loading_sections.clear()
        self.workers.cancel_group(self, "explorer-load")

    def _populate_overview(self, state: ProjectExplorerState) -> None:
        """Render one worker-loaded project summary.

        Parameters
        ----------
        state : ProjectExplorerState
            Handle-free project discovery result.
        """
        summary = state.summary
        self.action_state.plot_targets["project"] = (
            ("project", None) if summary.n_runs > 0 else None
        )
        content = (
            f"[b]{self.project.project_path.name}[/b]\n\n"
            f"Project: {self.project.project_path}\n"
            f"MTH5: {summary.mth5_path}\n"
            f"MTH5 version: {summary.file_version}\n"
            f"Reference time: {self.project.ref_time!s}\n"
            f"Time span: {summary.start_time or '-'} → {summary.end_time or '-'}\n"
            "Sample rates: "
            f"{', '.join(str(value) for value in summary.sample_rates) or '-'}\n\n"
            f"Surveys: {summary.n_surveys}\n"
            f"Stations: {summary.n_stations}\n"
            f"Runs: {summary.n_runs}\n"
            f"Channels: {summary.n_channels}"
        )
        self.query_one("#project-content", Static).update(content)

    def _populate_data_tree(self, state: ProjectExplorerState) -> None:
        """Populate the filtered Project and MTH5 data hierarchy.

        Parameters
        ----------
        state : ProjectExplorerState
            Handle-free project and MTH5 catalogue returned by a worker.
        """
        tree = self.query_one("#data-tree", Tree)
        tree.clear()
        tree.root.label = "Data"
        tree.root.data = None
        project_node = tree.root.add("Project", data=("project", "."))
        mth5_node = tree.root.add("MTH5", data=("mth5", "/"))
        self.data_items.clear()
        self.action_state.plot_targets["data"] = None
        self.action_state.has_project_data_to_delete = state.has_project_data_to_delete
        for issue in state.issues:
            self.notify(
                f"Unable to inspect {issue.section}: {issue.message}",
                severity="warning",
            )
        self._add_data_catalog(project_node, list(state.project_data_items))
        self._add_data_catalog(mth5_node, list(state.mth5_data_items))
        tree.root.expand()
        project_node.expand()
        mth5_node.expand()

    def _add_data_catalog(
        self, root: _DataTreeNode, items: list[ProjectDataItem]
    ) -> None:
        """Add each persistent data-type category below one source root.

        Parameters
        ----------
        root : _DataTreeNode
            Source node receiving the category branches.
        items : list[ProjectDataItem]
            Persistent items belonging to the source.
        """
        for label, data_type in self.DATA_CATEGORIES:
            matching = [item for item in items if item.data_type == data_type]
            category = root.add(
                f"{label} ({self._data_category_count(root.data, matching, data_type)})",
                data=("category", label),
            )
            self._add_data_items(category, items, data_type)

    @staticmethod
    def _data_category_count(
        source_data: object, items: list[ProjectDataItem], data_type: str
    ) -> int:
        """Count displayed data, treating MTH5 time channels as one run."""
        if source_data != ("mth5", "/") or data_type != "time":
            return sum(item.is_dataset for item in items)

        run_paths = {
            run_path
            for item in items
            if (run_path := ProjectExplorerScreen._mth5_time_run_path(item)) is not None
        }
        # Retain a useful count for non-standard MTH5 layouts that do not
        # encode the survey/station/run hierarchy in their item paths.
        return len(run_paths) if run_paths else sum(item.is_dataset for item in items)

    @staticmethod
    def _mth5_time_run_path(item: ProjectDataItem) -> str | None:
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

    def _add_data_items(
        self,
        root: _DataTreeNode,
        items: list[ProjectDataItem],
        data_type: str,
    ) -> None:
        """Add one category's items, retaining their path ancestors.

        Parameters
        ----------
        root : _DataTreeNode
            Category node receiving visible items.
        items : list[ProjectDataItem]
            Persistent items available below the source.
        data_type : str
            Data type selected for this category.
        """
        visible = self._visible_data_paths(items, data_type)
        nodes: dict[str | None, _DataTreeNode] = {None: root}
        for item in sorted(
            (item for item in items if item.path in visible),
            key=lambda value: (value.path.count("/"), value.path),
        ):
            parent = nodes.get(item.parent_path, root)
            key = f"{item.source}:{item.path}"
            self.data_items[key] = item
            if item.kind in {"directory", "group"}:
                nodes[item.path] = parent.add(item.name, data=(item.source, item.path))
            else:
                parent.add_leaf(item.name, data=(item.source, item.path))

    def _visible_data_paths(
        self, items: list[ProjectDataItem], data_type: str
    ) -> set[str]:
        """Return matching entries and the ancestors required to display them."""
        by_path = {item.path: item for item in items}
        visible = set()
        for item in items:
            if data_type == item.data_type:
                self._add_data_ancestors(item, by_path, visible)
        return visible

    @staticmethod
    def _add_data_ancestors(
        item: ProjectDataItem,
        by_path: dict[str, ProjectDataItem],
        visible: set[str],
    ) -> None:
        """Include one item and each represented parent in the filtered tree."""
        current: ProjectDataItem | None = item
        while current is not None:
            visible.add(current.path)
            current = (
                None
                if current.parent_path is None
                else by_path.get(current.parent_path)
            )

    def _project_data_path(self, path: str) -> Path | None:
        """Resolve an internal data-browser path without leaving project/data."""
        data_root = (self.project.project_path / "data").resolve()
        item_path = (data_root / path).resolve()
        if item_path != data_root and data_root not in item_path.parents:
            return None
        return item_path

    def _find_project_artifact(
        self, item: ProjectDataItem, required_files: tuple[str, ...]
    ) -> Path | None:
        """Find the containing saved artifact for a selected project item."""
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
    def _load_solution(solution_path: Path) -> Solution | None:
        """Read a saved solution for a lightweight plot eligibility check."""
        try:
            from resistics.regression import Solution

            return Solution.model_validate_json(solution_path.read_bytes())
        except Exception:
            return None

    def _mth5_time_plot_target(
        self, item: ProjectDataItem
    ) -> tuple[Literal["time"], TimePlotSelection] | None:
        """Turn a canonical MTH5 time path into a resistics run selection."""
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
            for summary in self.explorer_index.runs()
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

    def _data_item_for_node(self, node=None) -> ProjectDataItem | None:
        """Return the browsed data item for a tree node, excluding tree chrome."""
        if node is None:
            node = self.query_one("#data-tree", Tree).cursor_node
        data = None if node is None else node.data
        if not isinstance(data, tuple) or len(data) != 2:
            return None
        source, path = data
        if source not in {"project", "mth5"}:
            return None
        return self.data_items.get(f"{source}:{path}")

    def _data_tree_cursor(self):
        """Return the focused Data-tree node, if there is one."""
        if self.query_one(TabbedContent).active != "data":
            return None
        tree = self.query_one("#data-tree", Tree)
        if self.focused is not tree:
            return None
        return tree.cursor_node

    def _data_plot_target(self, node=None) -> PlotTarget | None:
        """Return a plot target for the highlighted item, if it is supported."""
        item = self._data_item_for_node(node)
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

    def _flow_plot_target(self) -> PlotTarget | None:
        """Return the focused or opened cached-valid flow plot target."""
        highlighted = self._highlighted_yaml_file()
        path: Path | None
        if highlighted is not None and highlighted[1] == "#flow-content":
            path = highlighted[0]
        else:
            path = self.selected_flow_path
        if path is None or path not in self.action_state.valid_flow_paths:
            return None
        return ("flow", path)

    def _job_plot_target(self) -> PlotTarget | None:
        """Return the focused or opened cached-valid job plot target."""
        highlighted_path = self._highlighted_job_path()
        if highlighted_path is not None:
            summary = next(
                (
                    value
                    for value in self.job_summaries.values()
                    if value.path == highlighted_path
                ),
                None,
            )
            return (
                ("job", highlighted_path)
                if summary is not None and summary.is_valid
                else None
            )
        if (
            self.selected_job_path is not None
            and self.selected_validation is not None
            and self.selected_validation.ok
            and self.selected_validation.resolved_job is not None
        ):
            return ("job", self.selected_job_path)
        return None

    def _has_project_timeline(self) -> bool:
        """Return the cached project-timeline eligibility."""
        return self.action_state.plot_targets["project"] is not None

    def _start_project_plot(self) -> None:
        if not self._has_project_timeline():
            return
        self.notify("Opening project timeline")
        self._open_plot(("project", None))

    def _start_selected_data_plot(self) -> None:
        target = self.action_state.plot_targets["data"]
        if target is None:
            return
        self.notify("Opening plot")
        self._open_plot(target)

    def _start_selected_flow_plot(self) -> None:
        target = self._flow_plot_target()
        if target is None:
            return
        self.notify("Opening flow plot")
        self._open_plot(target)

    def _start_selected_job_plot(self) -> None:
        target = self._job_plot_target()
        if target is None:
            return
        self.notify("Opening job plot")
        self._open_plot(target)

    @work(thread=True, exclusive=True, group="plotting")
    def _open_plot(self, target: PlotTarget) -> None:
        """Open a selected Plotly figure in the browser.

        Parameters
        ----------
        target : PlotTarget
            Validated plot kind and its typed payload.
        """
        plot_project = None
        target_type = target[0]
        plot_name = {
            "flow": "flow plot",
            "job": "job plot",
            "project": "project timeline",
        }.get(target_type, "plot")
        started = monotonic()
        try:
            import plotly.io as pio

            from resistics.project import load

            self.app.call_from_thread(self.notify, f"Building {plot_name}")
            plot_project = load(self.project.project_path)
            figure = self._build_plot_figure(plot_project, target)
            build_seconds = monotonic() - started
            self.app.call_from_thread(
                self.notify,
                f"{plot_name.capitalize()} built in {build_seconds:.1f}s; "
                "opening browser",
            )
            pio.show(figure)
            self.app.call_from_thread(self.notify, f"{plot_name.capitalize()} opened")
        except Exception as exc:
            self.app.call_from_thread(
                self.notify,
                f"Unable to open {plot_name}: {_feature_error('Plotting', exc)}",
                severity="error",
            )
        finally:
            if plot_project is not None and plot_project is not self.project:
                plot_project.close_mth5()

    @staticmethod
    def _build_plot_figure(  # noqa: C901 - plotting branches move in Phase 5.4
        project: Project, target: PlotTarget
    ):
        """Build a selected Plotly figure through its existing plot API.

        Parameters
        ----------
        project : Project
            Open project used to resolve and load the requested data.
        target : PlotTarget
            Validated plot kind and its typed payload.
        """
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

    def _populate_jobs(self, indexed_jobs: tuple[IndexedJob, ...]) -> None:
        """Populate the Jobs table from worker-loaded validation results.

        Parameters
        ----------
        indexed_jobs : tuple[IndexedJob, ...]
            Current job summaries and validations.
        """
        table = self.query_one("#job-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Job", "Flow", "Parameters", "Output", "Status")
        self.job_summaries.clear()
        self.selected_job_path = None
        self.selected_validation = None
        for indexed_job in indexed_jobs:
            summary = indexed_job.summary
            key = str(summary.path)
            self.job_summaries[key] = summary
            status = (
                "[green]valid[/green]" if summary.is_valid else "[red]invalid[/red]"
            )
            table.add_row(
                summary.name,
                summary.flow,
                summary.parameters,
                summary.output_label,
                status,
                key=key,
            )
        if self.job_summaries:
            self.query_one("#job-content", TextArea).text = "Select a job"
        else:
            self.query_one(
                "#job-content", TextArea
            ).text = "No YAML jobs found in processing/jobs"

    def _job_resource_options(
        self, resource_type: ResourceKind
    ) -> list[tuple[str, str]]:
        """Return valid resource files as dropdown labels and exact filenames.

        Parameters
        ----------
        resource_type : ResourceKind
            Flow, parameter, or criteria namespace used by the job form.

        Returns
        -------
        list[tuple[str, str]]
            Display labels paired with exact project-local filenames.
        """
        options = []
        for resource in self.explorer_index.resources(resource_type):
            if resource.model is None:
                logger.debug(
                    f"Unable to use {resource_type} resource {resource.path}: "
                    f"{resource.error}"
                )
                continue
            display_name = getattr(resource.model, "name", None) or resource.path.stem
            options.append(
                (
                    f"{display_name} ({resource.path.name})",
                    resource.path.name,
                )
            )
        return options

    def action_create_job(self) -> None:
        """Open the Jobs-tab form for a new editable job template."""
        if self.job_state == self._job_state_type.running:
            self.notify(
                "Job creation is unavailable while a job is running", severity="warning"
            )
            return
        flow_options = self._job_resource_options("flows")
        parameter_options = self._job_resource_options("parameters")
        criteria_options = self._job_resource_options("criteria")
        existing_names = {
            resource.path.stem for resource in self.explorer_index.resources("jobs")
        }
        self.app.push_screen(
            CreateJobScreen(
                flow_options, parameter_options, criteria_options, existing_names
            ),
            self._job_template_created,
        )

    def _job_template_created(self, definition: JobDefinition | None) -> None:
        """Persist a completed creation form and present the generated YAML."""
        if definition is None:
            return
        try:
            path = self.project_jobs.create_template(definition)
        except Exception as exc:
            self.notify(f"Unable to create job: {exc}", severity="error")
            return
        self._start_new_load_generation()
        self.explorer_index.invalidate("jobs")
        self._loaded_sections.discard("jobs")
        self._pending_job_path = path
        self._request_explorer_section("jobs", force=True)
        self.notify(f"Created {path.name}")

    def _populate_flows(self, resources: tuple[IndexedResource, ...]) -> None:
        """Populate the read-only flow browser.

        Parameters
        ----------
        resources : tuple[IndexedResource, ...]
            Current parsed flow resources.
        """
        from resistics.flow import FlowDefinition

        table = self.query_one("#flow-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Flow", "ID", "Version", "Nodes", "Status")
        self.flow_paths.clear()
        self.action_state.valid_flow_paths.clear()
        for resource in resources:
            path = resource.path
            key = str(path)
            self.flow_paths[key] = path
            if isinstance(resource.model, FlowDefinition):
                flow = resource.model
                self.action_state.valid_flow_paths.add(path)
                n_nodes = sum(len(stage.nodes) for stage in flow.flow_stages())
                table.add_row(
                    flow.name,
                    flow.id,
                    flow.version,
                    str(n_nodes),
                    "[green]valid[/green]",
                    key=key,
                )
            else:
                table.add_row(
                    path.stem,
                    "-",
                    "-",
                    "-",
                    "[red]invalid[/red]",
                    key=key,
                )
                logger.debug(f"Unable to read flow {path}: {resource.error}")
        if self.flow_paths and self.selected_flow_path is None:
            self.query_one("#flow-content", TextArea).text = "Select a flow"
        elif not self.flow_paths:
            self.query_one(
                "#flow-content", TextArea
            ).text = "No YAML flows found in processing/flows"

    def _populate_parameters(self, resources: tuple[IndexedResource, ...]) -> None:
        """Populate the read-only parameter-set browser.

        Parameters
        ----------
        resources : tuple[IndexedResource, ...]
            Current parsed parameter-set resources.
        """
        from resistics.flow import ParameterSet

        table = self.query_one("#parameter-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Parameters", "Processes", "Status")
        self.parameter_paths.clear()
        for resource in resources:
            path = resource.path
            key = str(path)
            self.parameter_paths[key] = path
            if isinstance(resource.model, ParameterSet):
                parameters = resource.model
                table.add_row(
                    parameters.name,
                    str(len(parameters.processes)),
                    "[green]valid[/green]",
                    key=key,
                )
            else:
                table.add_row(
                    path.stem,
                    "-",
                    "[red]invalid[/red]",
                    key=key,
                )
                logger.debug(f"Unable to read parameter set {path}: {resource.error}")
        if self.parameter_paths and self.selected_parameter_path is None:
            self.query_one(
                "#parameter-content", TextArea
            ).text = "Select a parameter set"
        elif not self.parameter_paths:
            self.query_one(
                "#parameter-content", TextArea
            ).text = "No YAML parameter sets found in processing/parameters"

    def _populate_criteria(self, resources: tuple[IndexedResource, ...]) -> None:
        """Populate the read-only criteria browser.

        Parameters
        ----------
        resources : tuple[IndexedResource, ...]
            Current parsed gather-criteria resources.
        """
        from resistics.gather import GatherCriteria

        table = self.query_one("#criteria-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Criteria", "Remote references", "Status")
        self.criteria_paths.clear()
        for resource in resources:
            path = resource.path
            key = str(path)
            self.criteria_paths[key] = path
            if isinstance(resource.model, GatherCriteria):
                criteria = resource.model
                table.add_row(
                    path.stem,
                    str(criteria.remote_reference_count()),
                    "[green]valid[/green]",
                    key=key,
                )
            else:
                table.add_row(path.stem, "-", "[red]invalid[/red]", key=key)
                logger.debug(f"Unable to read criteria {path}: {resource.error}")
        if self.criteria_paths and self.selected_criteria_path is None:
            self.query_one(
                "#criteria-content", TextArea
            ).text = "Select a criteria file"
        elif not self.criteria_paths:
            self.query_one(
                "#criteria-content", TextArea
            ).text = "No YAML criteria files found in processing/criteria"

    @on(Tree.NodeSelected, "#data-tree")
    def show_data_metadata(self, event: Tree.NodeSelected) -> None:
        item = event.node.data
        details = self.query_one("#data-metadata", TextArea)
        if item is None:
            details.text = json.dumps(
                {"message": "Select Project or MTH5 data"}, indent=2
            )
            return
        try:
            source, path = item
            if source == "category":
                details.text = json.dumps(
                    {"message": f"Expand {path} to inspect its data."}, indent=2
                )
                return
            if source == "project" and Path(path).suffix.lower() == ".json":
                details.text = json.dumps(
                    self.project.get_project_data_json(path), indent=2
                )
            else:
                metadata = (
                    self.project.get_project_data_metadata(path)
                    if source == "project"
                    else self.project.get_mth5_data_metadata(path)
                )
                details.text = metadata.model_dump_json(indent=2)
        except Exception as exc:
            details.text = json.dumps({"error": str(exc)}, indent=2)

    @on(Tree.NodeHighlighted, "#data-tree")
    def update_data_plot_selection(self, event: Tree.NodeHighlighted) -> None:
        """Cache plot eligibility when the highlighted data item changes."""
        self.action_state.plot_targets["data"] = self._data_plot_target(event.node)
        self.refresh_bindings()

    @on(DataTable.RowSelected, "#job-table")
    def show_job(self, event: DataTable.RowSelected) -> None:
        if self.editing_yaml:
            self.notify(
                "Save or discard the current YAML edits first", severity="warning"
            )
            return
        key = str(event.row_key.value)
        summary = self.job_summaries[key]
        self.selected_job_path = summary.path
        self.selected_validation = self.explorer_index.job_validation(summary.path)
        validation = self.selected_validation
        self._show_yaml("#job-content", summary.path)
        if validation is not None and validation.ok:
            self.notify("Job YAML is valid")
        else:
            errors = [] if validation is None else validation.errors
            logger.debug(f"Invalid job YAML {summary.path}: {'; '.join(errors)}")
            self.notify(
                f"Job YAML is invalid ({len(errors)} error(s)); "
                "source shown for repair or deletion",
                severity="warning",
            )
        self.refresh_bindings()

    @on(DataTable.RowHighlighted)
    def refresh_yaml_highlight_bindings(self, event: DataTable.RowHighlighted) -> None:
        """Refresh YAML actions when a resource-table cursor moves."""
        resource_tabs = {
            "flow-table": "flows",
            "parameter-table": "parameters",
            "criteria-table": "criteria",
            "job-table": "jobs",
        }
        active = self.query_one(TabbedContent).active
        table_id = event.data_table.id
        if (
            table_id is not None
            and event.data_table.has_focus
            and resource_tabs.get(table_id) == active
        ):
            self.refresh_bindings()

    @on(DataTable.RowSelected, "#flow-table")
    def show_flow(self, event: DataTable.RowSelected) -> None:
        """Show the complete selected flow definition."""
        if self.editing_yaml:
            self.notify(
                "Save or discard the current YAML edits first", severity="warning"
            )
            return
        path = self.flow_paths[str(event.row_key.value)]
        self.selected_flow_path = path
        resource = self.explorer_index.resource_for_path("flows", path)
        if resource is None or not resource.is_valid:
            logger.debug(f"Invalid flow YAML {path}: {resource and resource.error}")
            self.notify(
                "Flow YAML is invalid; source shown for repair or deletion",
                severity="warning",
            )
        self._show_yaml("#flow-content", path)
        self.refresh_bindings()

    @on(DataTable.RowSelected, "#parameter-table")
    def show_parameters(self, event: DataTable.RowSelected) -> None:
        """Show the complete selected parameter set."""
        if self.editing_yaml:
            self.notify(
                "Save or discard the current YAML edits first", severity="warning"
            )
            return
        path = self.parameter_paths[str(event.row_key.value)]
        self.selected_parameter_path = path
        resource = self.explorer_index.resource_for_path("parameters", path)
        if resource is None or not resource.is_valid:
            logger.debug(
                f"Invalid parameter YAML {path}: {resource and resource.error}"
            )
            self.notify(
                "Parameter YAML is invalid; source shown for repair or deletion",
                severity="warning",
            )
        self._show_yaml("#parameter-content", path)
        self.refresh_bindings()

    @on(DataTable.RowSelected, "#criteria-table")
    def show_criteria(self, event: DataTable.RowSelected) -> None:
        """Show the complete selected criteria definition."""
        if self.editing_yaml:
            self.notify(
                "Save or discard the current YAML edits first", severity="warning"
            )
            return
        path = self.criteria_paths[str(event.row_key.value)]
        self.selected_criteria_path = path
        resource = self.explorer_index.resource_for_path("criteria", path)
        if resource is None or not resource.is_valid:
            logger.debug(f"Invalid criteria YAML {path}: {resource and resource.error}")
            self.notify(
                "Criteria YAML is invalid; source shown for repair or deletion",
                severity="warning",
            )
        self._show_yaml("#criteria-content", path)
        self.refresh_bindings()

    @staticmethod
    def _yaml_source(path: Path) -> str:
        """Return YAML source exactly as authored, including comments."""
        return path.read_text(encoding="utf-8")

    def _show_yaml(self, editor_id: str, path: Path) -> None:
        """Load YAML source into one read-only, syntax-aware editor."""
        editor = self.query_one(editor_id, TextArea)
        editor.text = self._yaml_source(path)
        editor.read_only = True

    def _yaml_edit_target(self):
        """Return the selected YAML source and model for the active resource tab."""
        from resistics.flow import FlowDefinition, ParameterSet
        from resistics.gather import GatherCriteria
        from resistics.job import JobDefinition

        active = self.query_one(TabbedContent).active
        if active == "flows" and self.selected_flow_path is not None:
            return self.selected_flow_path, FlowDefinition, "#flow-content"
        if active == "parameters" and self.selected_parameter_path is not None:
            return self.selected_parameter_path, ParameterSet, "#parameter-content"
        if active == "criteria" and self.selected_criteria_path is not None:
            return self.selected_criteria_path, GatherCriteria, "#criteria-content"
        if active == "jobs" and self.selected_job_path is not None:
            return self.selected_job_path, JobDefinition, "#job-content"
        return None

    def _selected_yaml_file(self) -> tuple[Path, str] | None:
        """Return the selected YAML source and its editor selector."""
        target = self._yaml_edit_target()
        if target is None:
            return None
        path, _, editor_id = target
        return path, editor_id

    def _highlighted_yaml_file(self) -> tuple[Path, str] | None:
        """Return the row highlighted in the focused active resource table."""
        active = self.query_one(TabbedContent).active
        resources = {
            "flows": ("#flow-table", self.flow_paths, "#flow-content"),
            "parameters": (
                "#parameter-table",
                self.parameter_paths,
                "#parameter-content",
            ),
            "criteria": (
                "#criteria-table",
                self.criteria_paths,
                "#criteria-content",
            ),
            "jobs": (
                "#job-table",
                {key: summary.path for key, summary in self.job_summaries.items()},
                "#job-content",
            ),
        }
        resource = resources.get(active)
        if resource is None:
            return None
        table_id, paths, editor_id = resource
        table = self.query_one(table_id, DataTable)
        if not table.has_focus or not table.is_valid_row_index(table.cursor_row):
            return None
        key = str(table.ordered_rows[table.cursor_row].key.value)
        path = paths.get(key)
        return None if path is None else (path, editor_id)

    def _highlighted_job_path(self) -> Path | None:
        """Return the focused Jobs-table row without requiring it to be opened."""
        highlighted = self._highlighted_yaml_file()
        if highlighted is None or highlighted[1] != "#job-content":
            return None
        return highlighted[0]

    @staticmethod
    def _copy_yaml_file(source: Path, name: str) -> Path:
        """Copy source verbatim to a new, non-overwriting YAML filename."""
        name = _validate_yaml_file_stem(name)
        for suffix in (".yaml", ".yml"):
            if (source.parent / f"{name}{suffix}").exists():
                raise ValueError(f"A YAML file named {name!r} already exists")
        destination = source.with_name(f"{name}{source.suffix}")
        with source.open("rb") as input_file, destination.open("xb") as output_file:
            output_file.write(input_file.read())
        return destination

    def action_copy_yaml(self) -> None:
        """Copy the focused highlighted or currently opened YAML source."""
        selected = self._highlighted_yaml_file() or self._selected_yaml_file()
        if selected is None:
            self.notify("Select a YAML file first", severity="warning")
            return
        source, editor_id = selected
        self.app.push_screen(
            CopyYamlFileScreen(source),
            lambda name: self._yaml_file_copied(source, editor_id, name),
        )

    def _yaml_file_copied(self, source: Path, editor_id: str, name: str | None) -> None:
        if name is None:
            return
        try:
            destination = self._copy_yaml_file(source, name)
        except Exception as exc:
            self.notify(f"Unable to copy YAML: {exc}", severity="error")
            return
        self._refresh_yaml_resource(editor_id)
        self._select_yaml_file(editor_id, destination)
        self.notify(f"Copied {source.name} to {destination.name}")

    def action_delete_yaml(self) -> None:
        """Delete Data-tab artifacts or the highlighted YAML source."""
        if self.query_one(TabbedContent).active == "data":
            self._start_project_data_deletion()
            return
        selected = self._highlighted_yaml_file() or self._selected_yaml_file()
        if selected is None:
            self.notify("Select a YAML file first", severity="warning")
            return
        source, editor_id = selected
        self.app.push_screen(
            DeleteYamlFileScreen(source),
            lambda confirmed: self._yaml_file_deleted(source, editor_id, confirmed),
        )

    def _yaml_file_deleted(self, source: Path, editor_id: str, confirmed: bool) -> None:
        if not confirmed:
            return
        try:
            source.unlink()
        except FileNotFoundError:
            self.notify(f"{source.name} was already deleted", severity="warning")
            return
        except Exception as exc:
            self.notify(f"Unable to delete YAML: {exc}", severity="error")
            return
        self._refresh_yaml_resource(editor_id)
        self._clear_selected_yaml_file(editor_id)
        self.notify(f"Deleted {source.name}")

    def _start_project_data_deletion(self) -> None:
        """Choose the namespace of derived project data to remove."""
        if self.job_state == self._job_state_type.running:
            self.notify("Data deletion is unavailable while a job is running")
            return
        try:
            labels = self.project.list_project_output_labels()
            preview = self.project.preview_project_data_deletion()
        except Exception as exc:
            self.notify(f"Unable to inspect project data: {exc}", severity="error")
            return
        if not labels and not preview.paths:
            self.notify(
                "There is no derived Project data to delete", severity="warning"
            )
            return
        self.app.push_screen(
            DeleteProjectDataScreen(labels), self._project_data_deletion_selected
        )

    def _project_data_deletion_selected(
        self, request: ProjectDataDeletionRequest | None
    ) -> None:
        if request is None:
            return
        try:
            deletion = self.project.preview_project_data_deletion(request.output_label)
        except Exception as exc:
            self.notify(f"Unable to prepare deletion: {exc}", severity="error")
            return
        if not deletion.paths:
            label = request.output_label
            message = (
                "There is no derived Project data to delete"
                if label is None
                else f"No data exists for output label {label!r}"
            )
            self.notify(message, severity="warning")
            return
        self.app.push_screen(
            ConfirmProjectDataDeletionScreen(deletion),
            lambda confirmed: self._project_data_deletion_confirmed(
                deletion, confirmed
            ),
        )

    def _project_data_deletion_confirmed(
        self, deletion: ProjectDataDeletion, confirmed: bool
    ) -> None:
        if not confirmed:
            return
        try:
            deleted = self.project.delete_project_data(deletion.output_label)
        except Exception as exc:
            self.notify(f"Unable to delete Project data: {exc}", severity="error")
            return
        self._start_new_load_generation()
        self.explorer_index.invalidate("project")
        self._loaded_sections.discard("data")
        self.query_one("#data-metadata", TextArea).text = json.dumps(
            {"message": "Select Project or MTH5 data"}, indent=2
        )
        self._request_explorer_section("data", force=True)
        self.notify(f"Deleted {deleted.count} Project data path(s)")

    def _select_yaml_file(self, editor_id: str, path: Path) -> None:
        """Make path the current selection and display its source."""
        if editor_id == "#flow-content":
            self.selected_flow_path = path
        elif editor_id == "#parameter-content":
            self.selected_parameter_path = path
        elif editor_id == "#criteria-content":
            self.selected_criteria_path = path
        elif editor_id == "#job-content":
            self.selected_job_path = path
            self.selected_validation = self.explorer_index.job_validation(path)
        self._show_yaml(editor_id, path)
        self.refresh_bindings()

    def _clear_selected_yaml_file(self, editor_id: str) -> None:
        """Clear the deleted source selection and restore its placeholder."""
        placeholders = {
            "#flow-content": "Select a flow",
            "#parameter-content": "Select a parameter set",
            "#criteria-content": "Select a criteria file",
            "#job-content": "Select a job",
        }
        if editor_id == "#flow-content":
            self.selected_flow_path = None
        elif editor_id == "#parameter-content":
            self.selected_parameter_path = None
        elif editor_id == "#criteria-content":
            self.selected_criteria_path = None
        elif editor_id == "#job-content":
            self.selected_job_path = None
            self.selected_validation = None
        editor = self.query_one(editor_id, TextArea)
        editor.text = placeholders[editor_id]
        editor.read_only = True
        self.refresh_bindings()

    def action_edit_yaml(self) -> None:
        """Make the selected YAML source editable."""
        if self.job_state == self._job_state_type.running:
            self.notify(
                "Editing is unavailable while a job is running", severity="warning"
            )
            return
        target = self._yaml_edit_target()
        if target is None:
            self.notify("Select a YAML file first", severity="warning")
            return
        self.editing_path, self.editing_model, self.editing_editor_id = target
        self.editing_yaml = True
        editor = self.query_one(self.editing_editor_id, TextArea)
        editor.read_only = False
        editor.focus()
        self.notify("Editing YAML — Ctrl+S saves; Esc discards")
        self.refresh_bindings()

    def action_save_yaml(self) -> None:
        """Validate and atomically save the active YAML editor."""
        from resistics.flow import model_from_yaml

        if (
            not self.editing_yaml
            or self.editing_path is None
            or self.editing_model is None
            or self.editing_editor_id is None
        ):
            return
        editor = self.query_one(self.editing_editor_id, TextArea)
        try:
            model_from_yaml(self.editing_model, editor.text)
        except Exception as exc:
            self.notify(f"YAML was not saved: {exc}", severity="error")
            return
        try:
            self._write_yaml(self.editing_path, editor.text)
        except Exception as exc:
            self.notify(f"Unable to save YAML: {exc}", severity="error")
            return
        editor.read_only = True
        saved_path = self.editing_path
        editor_id = self.editing_editor_id
        self._clear_yaml_editing()
        if editor_id == "#job-content":
            self._pending_job_path = saved_path
        self._refresh_yaml_resource(editor_id)
        self._show_yaml(editor_id, saved_path)
        self.refresh_bindings()
        self.notify(f"Saved {saved_path.name}")

    def action_discard_yaml(self) -> None:
        """Discard the active YAML draft and restore its saved source."""
        if (
            not self.editing_yaml
            or self.editing_path is None
            or self.editing_editor_id is None
        ):
            return
        self._show_yaml(self.editing_editor_id, self.editing_path)
        self._clear_yaml_editing()
        self.refresh_bindings()
        self.notify("YAML edits discarded")

    @staticmethod
    def _write_yaml(path: Path, content: str) -> None:
        """Atomically replace a YAML file after validation has succeeded."""
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

    def _clear_yaml_editing(self) -> None:
        """Clear YAML edit state and restore normal footer actions."""
        self.editing_yaml = False
        self.editing_path = None
        self.editing_model = None
        self.editing_editor_id = None

    def _refresh_yaml_resource(self, editor_id: str) -> None:
        """Refresh the table associated with a saved YAML resource."""
        resource_types: dict[str, ResourceKind] = {
            "#flow-content": "flows",
            "#parameter-content": "parameters",
            "#criteria-content": "criteria",
            "#job-content": "jobs",
        }
        resource_type = resource_types.get(editor_id)
        if resource_type is None:
            return
        reload_jobs = resource_type != "jobs" and "jobs" in self._loaded_sections
        self._start_new_load_generation()
        self.explorer_index.invalidate(resource_type)
        self._loaded_sections.discard(resource_type)
        self._loaded_sections.discard("jobs")
        self._request_explorer_section(resource_type, force=True)
        if reload_jobs:
            self._request_explorer_section("jobs", force=True)

    def action_run_selected_job(self) -> None:
        """Confirm and run the opened or focused highlighted job."""
        highlighted_path = self._highlighted_job_path()
        if highlighted_path is not None:
            validation = self.explorer_index.job_validation(highlighted_path)
            self.selected_job_path = highlighted_path
            self.selected_validation = validation
        else:
            validation = self.selected_validation
        if validation is None or not validation.ok:
            self.notify("Select a valid job first", severity="warning")
            return
        self.app.push_screen(ConfirmJobScreen(validation), self._submission_confirmed)

    def _restore_flows(self) -> None:
        """Restore only missing built-in flow templates."""
        from resistics.templates import install_builtin_flow_templates

        installed = install_builtin_flow_templates(self.project.project_path)
        reload_jobs = "jobs" in self._loaded_sections
        self._start_new_load_generation()
        self.explorer_index.invalidate("flows")
        self._loaded_sections.discard("flows")
        self._loaded_sections.discard("jobs")
        self._request_explorer_section("flows", force=True)
        if reload_jobs:
            self._request_explorer_section("jobs", force=True)
        if installed:
            self.notify(f"Restored {len(installed)} flow template(s)")
        else:
            self.notify("All built-in flow templates are already present")

    def _restore_parameters(self) -> None:
        """Restore only missing built-in parameter-set templates."""
        from resistics.templates import install_builtin_parameter_templates

        installed = install_builtin_parameter_templates(self.project.project_path)
        reload_jobs = "jobs" in self._loaded_sections
        self._start_new_load_generation()
        self.explorer_index.invalidate("parameters")
        self._loaded_sections.discard("parameters")
        self._loaded_sections.discard("jobs")
        self._request_explorer_section("parameters", force=True)
        if reload_jobs:
            self._request_explorer_section("jobs", force=True)
        if installed:
            self.notify(f"Restored {len(installed)} parameter-set template(s)")
        else:
            self.notify("All built-in parameter-set templates are already present")

    def _restore_criteria(self) -> None:
        """Restore only missing criteria examples."""
        from resistics.templates import install_builtin_criteria_templates

        installed = install_builtin_criteria_templates(self.project.project_path)
        reload_jobs = "jobs" in self._loaded_sections
        self._start_new_load_generation()
        self.explorer_index.invalidate("criteria")
        self._loaded_sections.discard("criteria")
        self._loaded_sections.discard("jobs")
        self._request_explorer_section("criteria", force=True)
        if reload_jobs:
            self._request_explorer_section("jobs", force=True)
        if installed:
            self.notify(f"Restored {len(installed)} criteria example(s)")
        else:
            self.notify("All built-in criteria examples are already present")

    def action_restore_defaults(self) -> None:
        """Restore defaults for the active Flows or Parameters tab."""
        active = self.query_one(TabbedContent).active
        if active == "flows":
            self._restore_flows()
        elif active == "parameters":
            self._restore_parameters()
        elif active == "criteria":
            self._restore_criteria()
        else:
            return
        self.refresh_bindings()

    def _submission_confirmed(self, confirmed: bool | None) -> None:
        if confirmed:
            self._execute_selected_job()

    @work(thread=True, exclusive=True, group="processing-job")
    def _execute_selected_job(self) -> None:
        validation = self.selected_validation
        if validation is None or validation.resolved_job is None:
            raise ValueError("A resolved job is required for execution")
        self.app.call_from_thread(self._set_running)
        processing_project = None
        try:
            from resistics.project import load

            processing_project = load(self.project.project_path)
            self.job_runner = self._job_runner_type(
                processing_project,
                progress_callback=lambda event: self.app.call_from_thread(
                    self._show_progress, event
                ),
            )
            self.job_runner.run(validation.resolved_job)
        except Exception as exc:
            self.app.call_from_thread(
                self._show_progress,
                self._job_progress_type(
                    state=self._job_state_type.failed,
                    message="Unable to start job",
                    job_name=validation.resolved_job.definition.name,
                    error=_feature_error("Job processing", exc),
                ),
            )
        finally:
            if processing_project is not None:
                processing_project.close_mth5()

    def _set_running(self) -> None:
        self.job_state = self._job_state_type.running
        self.query_one("#activity-status", Static).update("Job running")
        self.query_one("#activity-log", RichLog).clear()
        tabs = self.query_one(TabbedContent)
        if tabs.active == "activity":
            self.refresh_bindings()
        else:
            tabs.active = "activity"

    def _show_progress(self, event: JobProgressEvent) -> None:
        self.job_state = event.state
        context = []
        if event.station:
            station = (
                f"{event.survey}/{event.station}" if event.survey else event.station
            )
            context.append(f"station {station}")
        elif event.survey:
            context.append(f"survey {event.survey}")
        if event.run:
            context.append(f"run {event.run}")
        if event.sample_rate is not None:
            context.append(f"sample rate {event.sample_rate:g} Hz")
        target = f" — {', '.join(context)}" if context else ""
        progress, status = _progress_details(event)
        line = (
            f"[{event.state.value}] {event.message}{progress}{target} "
            f"({event.elapsed_seconds:.1f}s)"
        )
        if event.error:
            line += f"\n[red]{event.error}[/red]"
        self.query_one("#activity-log", RichLog).write(line)
        self.query_one("#activity-status", Static).update(status)
        if event.state in {
            self._job_state_type.completed,
            self._job_state_type.failed,
            self._job_state_type.cancelled,
        }:
            self.job_runner = None
            reload_jobs = "jobs" in self._loaded_sections
            reload_data = "data" in self._loaded_sections
            self._start_new_load_generation()
            self.explorer_index.invalidate("project", "jobs")
            self._loaded_sections.discard("jobs")
            self._loaded_sections.discard("data")
            if reload_jobs:
                self._request_explorer_section("jobs", force=True)
            if reload_data:
                self._request_explorer_section("data", force=True)
        self.refresh_bindings()

    @on(TabbedContent.TabActivated)
    def refresh_tab_bindings(self) -> None:
        """Lazily load the active explorer tab and refresh its Footer."""
        active = self.query_one(TabbedContent).active
        if active in {"project", "data", "flows", "parameters", "criteria", "jobs"}:
            self._request_explorer_section(active)
        self.refresh_bindings()

    def _check_resource_action(self, action: str, active: str) -> bool:
        """Check one YAML-resource action using only in-memory state.

        Parameters
        ----------
        action : str
            Action name to check.
        active : str
            Identifier of the active tab.

        Returns
        -------
        bool
            Whether the action is currently available.
        """
        if active in {"flows", "parameters", "criteria", "jobs"} and (
            active not in self._loaded_sections
        ):
            return False
        if action == "edit_yaml":
            return (
                not self.editing_yaml
                and self.job_state != self._job_state_type.running
                and self._yaml_edit_target() is not None
            )
        if action == "create_job":
            return (
                active == "jobs"
                and not self.editing_yaml
                and self.job_state != self._job_state_type.running
            )
        if action == "delete_yaml" and active == "data":
            return (
                not self.editing_yaml
                and self.job_state != self._job_state_type.running
                and self.action_state.has_project_data_to_delete
            )
        if action in {"copy_yaml", "delete_yaml"}:
            return (
                not self.editing_yaml
                and self.job_state != self._job_state_type.running
                and (
                    self._highlighted_yaml_file() is not None
                    or self._selected_yaml_file() is not None
                )
            )
        if action in {"save_yaml", "discard_yaml"}:
            return self.editing_yaml
        return not self.editing_yaml and active in {
            "flows",
            "parameters",
            "criteria",
        }

    def _check_selected_job_action(self, active: str) -> bool | None:
        """Check job execution eligibility from cached validation summaries.

        Parameters
        ----------
        active : str
            Identifier of the active tab.

        Returns
        -------
        bool | None
            Whether a selected job can run, or ``None`` to disable it.
        """
        if active != "jobs":
            return False
        highlighted_path = self._highlighted_job_path()
        if highlighted_path is not None:
            summary = next(
                (
                    value
                    for value in self.job_summaries.values()
                    if value.path == highlighted_path
                ),
                None,
            )
            return True if summary is not None and summary.is_valid else None
        return (
            True
            if self.selected_validation is not None and self.selected_validation.ok
            else None
        )

    def _check_data_tree_action(self, action: str) -> bool:
        """Check expansion eligibility from the current in-memory tree node.

        Parameters
        ----------
        action : str
            Expansion or collapse action name.

        Returns
        -------
        bool
            Whether the tree action is currently available.
        """
        node = self._data_tree_cursor()
        if node is None or not node.allow_expand:
            return False
        return (
            not node.is_expanded if action == "expand_data_node" else node.is_expanded
        )

    def _check_plot_action(self, active: str) -> bool:
        """Check plot eligibility using cached project and selection state.

        Parameters
        ----------
        active : str
            Identifier of the active tab.

        Returns
        -------
        bool
            Whether the active tab has a valid cached plot target.
        """
        if active in {"project", "data"}:
            return self.action_state.plot_targets[active] is not None
        if active == "flows":
            return not self.editing_yaml and self._flow_plot_target() is not None
        if active == "jobs":
            return not self.editing_yaml and self._job_plot_target() is not None
        return False

    def check_action(self, action: str, parameters: tuple[object, ...]):
        """Expose only Footer actions relevant to cached screen state."""
        tabbed_content = self.query(TabbedContent)
        if not tabbed_content.nodes:
            return False
        active = tabbed_content.first(TabbedContent).active
        resource_actions = {
            "edit_yaml",
            "create_job",
            "copy_yaml",
            "delete_yaml",
            "save_yaml",
            "discard_yaml",
            "restore_defaults",
        }
        if action in resource_actions:
            return self._check_resource_action(action, active)
        if action == "close_project":
            return (
                self.job_state != self._job_state_type.running and not self.editing_yaml
            )
        if action == "run_selected_job":
            return self._check_selected_job_action(active)
        if action == "cancel_job":
            return (
                active == "activity" and self.job_state == self._job_state_type.running
            )
        if action in {"expand_data_node", "collapse_data_node"}:
            return self._check_data_tree_action(action)
        if action == "plot":
            return self._check_plot_action(active)
        return super().check_action(action, parameters)

    def action_plot(self) -> None:
        """Plot the project, highlighted data, selected flow, or selected job."""
        active = self.query_one(TabbedContent).active
        if active == "project":
            self._start_project_plot()
        elif active == "data":
            self._start_selected_data_plot()
        elif active == "flows":
            self._start_selected_flow_plot()
        elif active == "jobs":
            self._start_selected_job_plot()

    def action_expand_data_node(self) -> None:
        """Expand the highlighted Data-tree branch and all of its descendants."""
        node = self._data_tree_cursor()
        if node is not None and node.allow_expand:
            node.expand_all()
            self.refresh_bindings()

    def action_collapse_data_node(self) -> None:
        """Collapse the highlighted Data-tree branch and all of its descendants."""
        node = self._data_tree_cursor()
        if node is not None and node.allow_expand:
            node.collapse_all()
            self.refresh_bindings()

    def action_refresh(self) -> None:
        if self.job_state == self._job_state_type.running:
            self.notify("Refresh is unavailable while a job is running")
            return
        if self.editing_yaml:
            self.notify(
                "Save or discard the current YAML edits first", severity="warning"
            )
            return
        active = self.query_one(TabbedContent).active
        section: ExplorerView = (
            active
            if active in {"project", "data", "flows", "parameters", "criteria", "jobs"}
            else "project"
        )
        self._start_new_load_generation()
        self.explorer_index.invalidate_all()
        self._loaded_sections.clear()
        self._request_explorer_section(section, force=True)
        self.refresh_bindings()
        self.notify("Project refreshed")

    def action_cancel_job(self) -> None:
        if self.job_runner is None or self.job_state != self._job_state_type.running:
            self.notify("No active job")
            return
        self.job_runner.cancel()
        self.notify("Cancellation requested; the current step will finish first")

    def action_close_project(self) -> None:
        if self.job_state == self._job_state_type.running:
            self.notify(
                "Close project is unavailable while a job is running",
                severity="warning",
            )
            return
        if self.editing_yaml:
            self.notify(
                "Save or discard the current YAML edits first", severity="warning"
            )
            return
        _resistics_app(self).show_home()

    def action_quit(self) -> None:
        if self.job_state == self._job_state_type.running:
            self.notify(
                "A job is running. Press C to request cancellation before quitting.",
                severity="warning",
            )
            return
        if self.editing_yaml:
            self.notify(
                "Save or discard the current YAML edits first", severity="warning"
            )
            return
        self.app.exit()


class ResisticsTui(App[None]):
    """The resistics terminal application and project launcher."""

    TITLE = "resistics"
    SUB_TITLE = ""
    NOTIFICATION_TIMEOUT = 4.0
    CSS = """
    Screen { layout: vertical; background: #101010; color: #f7f4f2; }
    #app-header {
        height: 1;
        padding: 0 1;
        background: #0a009f;
        color: #f7f4f2;
    }
    Footer { background: #070066; color: #faa881; }
    Footer > .footer--key { background: #ac3600; color: #f7f4f2; }
    ToastRack {
        dock: bottom;
        align: right bottom;
        width: auto;
        margin: 0 1 1 0;
    }
    Toast {
        width: 48;
        max-width: 45%;
        margin-top: 0;
        padding: 0 1;
        background: #202020;
    }
    .launcher-layout { height: 1fr; align-horizontal: center; }
    #home, #create-project-form {
        width: 72;
        height: auto;
        padding: 2;
        margin-top: 3;
        background: #202020;
    }
    #home Button, #create-project-form Button { margin-top: 1; }
    #home-message, #create-status { margin-top: 1; color: #faa881; }
    #parent-path, #mth5-path { color: #aaa6ad; }
    #create-actions { height: auto; align-horizontal: right; margin-top: 1; }
    #create-actions Button { margin-left: 1; }
    Input { background: #101010; color: #f7f4f2; border: tall #343434; }
    Input:focus { background: #202020; border: tall #0a009f; }
    TabbedContent { height: 1fr; background: #101010; color: #f7f4f2; }
    Tabs { background: #202020; color: #f7f4f2; }
    Tab { color: #f7f4f2; }
    Tab.-active { background: #faa881; color: #101010; text-style: bold; }
    TabPane { background: #101010; color: #f7f4f2; }
    .pane { height: 1fr; padding: 1; background: #101010; color: #f7f4f2; }
    .split { height: 1fr; }
    .left { width: 2fr; border-right: solid #faa881; padding-right: 1; }
    .right { width: 3fr; }
    Tree, DataTable { background: #202020; color: #f7f4f2; }
    DataTable > .datatable--header { background: #0a009f; color: #f7f4f2; }
    Tree > .tree--label, DataTable > .datatable--cursor,
    Tree > .tree--cursor, Tree:focus > .tree--cursor,
    DataTable:focus > .datatable--cursor {
        text-style: none;
    }
    Tree > .tree--cursor, Tree:focus > .tree--cursor,
    DataTable > .datatable--cursor, DataTable:focus > .datatable--cursor {
        background: #faa881;
        color: #101010;
    }
    Tree > .tree--highlight-line, DataTable > .datatable--hover {
        background: #343434;
    }
    Tree > .tree--guides-selected { color: #faa881; }
    #data-tree, #flow-table, #parameter-table, #criteria-table, #job-table {
        height: 1fr;
    }
    #data-metadata, #flow-content, #parameter-content, #criteria-content,
    #job-content {
        height: 1fr;
        background: #202020;
        color: #f7f4f2;
        border: tall #343434;
    }
    #data-metadata:focus, #flow-content:focus, #parameter-content:focus,
    #criteria-content:focus, #job-content:focus { border: tall #343434; }
    Button { background: #faa881; color: #101010; border: none; }
    Button.-success { background: #ac3600; color: #f7f4f2; }
    Button:focus {
        background: #0a009f;
        color: #f7f4f2;
        border: none;
        text-style: none;
    }
    Button:disabled { background: #343434; color: #aaa6ad; }
    Button.dialog-action {
        background: #343434;
        color: #f7f4f2;
        text-style: none;
    }
    Button.dialog-action:focus {
        background: #faa881;
        color: #101010;
        border: none;
        text-style: bold;
    }
    #activity-log {
        height: 1fr;
        border: round #ac3600;
        background: #202020;
        color: #f7f4f2;
    }
    #activity-status { height: auto; margin-bottom: 1; color: #faa881; }
    """

    def __init__(self, project_path: Path | None = None):
        super().__init__()
        self.initial_project_path = project_path
        self._has_started_screen = False
        self._project_open_generation = 0

    def on_mount(self) -> None:
        if self.initial_project_path is None:
            self.show_home()
        else:
            self.open_project_path(self.initial_project_path)

    def on_unmount(self) -> None:
        """Reject any project-open result completed after app shutdown."""
        self._project_open_generation += 1

    def show_home(self, message: str | None = None) -> None:
        self._reject_pending_project_open()
        self._mount_home(message)

    def _mount_home(self, message: str | None = None) -> None:
        """Replace the current screen with the project launcher.

        Parameters
        ----------
        message : str | None
            Optional status or failure detail shown on the launcher.
        """
        self.title = "resistics"
        self.sub_title = "project launcher"
        self._show_screen(HomeScreen(message))

    def show_create_project(self) -> None:
        self._reject_pending_project_open()
        self.title = "resistics"
        self.sub_title = "create project"
        self._show_screen(CreateProjectScreen())

    def open_project_path(self, project_path: Path) -> None:
        self._reject_pending_project_open()
        generation = self._project_open_generation
        self.title = "resistics"
        self.sub_title = str(project_path)
        self._show_screen(ProjectLoadingScreen(project_path))
        self._load_project_path(generation, project_path)

    @work(group="project-open", exit_on_error=False)
    async def _load_project_path(
        self, generation: int, project_path: Path
    ) -> _ProjectOpenResult:
        """Open one project without blocking the Textual event loop.

        Parameters
        ----------
        generation : int
            App generation requesting the project.
        project_path : Path
            Project directory to open.

        Returns
        -------
        _ProjectOpenResult
            Immutable success or failure passed back to the UI thread.
        """
        return await _run_in_worker_thread(
            partial(self._open_project_path, generation, project_path)
        )

    def _open_project_path(
        self, generation: int, project_path: Path
    ) -> _ProjectOpenResult:
        """Perform synchronous project opening inside a worker thread.

        Parameters
        ----------
        generation : int
            App generation requesting the project.
        project_path : Path
            Project directory to open.

        Returns
        -------
        _ProjectOpenResult
            Immutable project-open outcome containing no widgets.
        """
        try:
            from resistics.project import load

            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                project = load(project_path)
        except Exception as exc:
            return _ProjectOpenResult(
                generation=generation,
                project_path=project_path,
                error=_feature_error("Project loading", exc),
            )
        if generation != self._project_open_generation:
            try:
                project.close_mth5()
            except Exception:
                logger.exception(f"Unable to close superseded project {project_path}")
            return _ProjectOpenResult(
                generation=generation,
                project_path=project_path,
            )
        return _ProjectOpenResult(
            generation=generation,
            project_path=project_path,
            project=project,
            startup_warnings=tuple(str(warning.message) for warning in caught_warnings),
        )

    @on(Worker.StateChanged)
    def _apply_project_open_result(self, event: Worker.StateChanged) -> None:
        """Open only the latest project returned to the Textual UI thread.

        Parameters
        ----------
        event : Worker.StateChanged
            Textual lifecycle event for a project-open worker.
        """
        if event.state != WorkerState.SUCCESS or event.worker.group != "project-open":
            return
        result = event.worker.result
        if not isinstance(result, _ProjectOpenResult):
            return
        if result.generation != self._project_open_generation:
            if result.project is not None:
                result.project.close_mth5()
            return
        if result.error is not None or result.project is None:
            self._mount_home(
                f"[red]Unable to open project:[/] {result.error or 'Unknown error'}"
            )
            return
        self._mount_project(result.project, list(result.startup_warnings))

    def _reject_pending_project_open(self) -> None:
        """Logically cancel pending opens so their late results are rejected."""
        self._project_open_generation += 1

    def cancel_project_open(self) -> None:
        """Cancel the visible project-open operation and show the launcher."""
        self.show_home("Project opening cancelled")

    def open_project(
        self, project: Project, startup_warnings: list[str] | None = None
    ) -> None:
        self._reject_pending_project_open()
        self._mount_project(project, startup_warnings)

    def _mount_project(
        self, project: Project, startup_warnings: list[str] | None = None
    ) -> None:
        """Replace the current screen with one already opened project.

        Parameters
        ----------
        project : Project
            Open project returned by a completed worker or direct caller.
        startup_warnings : list[str] | None
            Warnings captured while opening the project.
        """
        self.title = "resistics"
        self.sub_title = str(project.project_path)
        self._show_screen(ProjectExplorerScreen(project, startup_warnings))

    def _show_screen(self, screen: Screen[None]) -> None:
        """Push the initial screen and replace it for later navigation."""
        if self._has_started_screen:
            self.switch_screen(screen)
        else:
            self.push_screen(screen)
            self._has_started_screen = True


def run_tui(project_path: Path | None = None) -> None:
    """Run the resistics terminal application."""
    logger.remove()
    try:
        ResisticsTui(project_path).run()
    finally:
        logger.remove()
        logger.add(sys.stderr, level="INFO")


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the TUI, optionally opening one project path immediately."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) > 1:
        print("Usage: resistics [PROJECT_PATH]", file=sys.stderr)
        return 2
    project_path = Path(arguments[0]) if arguments else None
    run_tui(project_path)
    return 0
