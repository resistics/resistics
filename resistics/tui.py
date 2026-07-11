"""Terminal user interface for inspecting projects and submitting jobs."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Optional
import warnings

from loguru import logger
from rich.pretty import Pretty
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
    Tree,
)

from resistics.job import (
    JobProgressEvent,
    JobRunner,
    JobState,
    JobSummary,
    JobValidation,
    ProjectJobs,
)
from resistics.project import Project, load


class ConfirmJobScreen(ModalScreen[bool]):
    """Confirm submission of an already validated job."""

    CSS = """
    ConfirmJobScreen { align: center middle; }
    #confirm-dialog {
        width: 72;
        height: auto;
        padding: 1 2;
        border: round #faa881;
        background: #202020;
        color: #f7f4f2;
    }
    #confirm-actions { height: auto; align-horizontal: right; margin-top: 1; }
    #confirm-actions Button { margin-left: 1; }
    """

    def __init__(self, validation: JobValidation):
        super().__init__()
        self.validation = validation

    def compose(self) -> ComposeResult:
        resolved = self.validation.resolved_job
        if resolved is None:
            raise ValueError("A resolved job is required for confirmation")
        definition = resolved.definition
        runtime = definition.runtime
        details = (
            f"Submit [bold]{definition.name}[/bold]?\n\n"
            f"Flow: {definition.flow}\n"
            f"Parameters: {definition.parameters}\n"
            f"Run: {runtime.get('survey', '')}/{runtime.get('station', '')}/"
            f"{runtime.get('run', '')}\n"
            f"Output: {resolved.output_path}"
        )
        with Vertical(id="confirm-dialog"):
            yield Static(details)
            with Horizontal(id="confirm-actions"):
                yield Button("Cancel", id="cancel")
                yield Button("Run job", id="confirm", variant="success")

    @on(Button.Pressed, "#cancel")
    def cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#confirm")
    def confirm(self) -> None:
        self.dismiss(True)


class ResisticsTui(App[None]):
    """Read-only project browser with managed processing-job execution."""

    TITLE = "resistics"
    SUB_TITLE = "project explorer"
    CSS = """
    Screen { layout: vertical; background: #101010; color: #f7f4f2; }
    Header { background: #0a009f; color: #f7f4f2; text-style: bold; }
    Footer { background: #070066; color: #faa881; }
    Footer > .footer--key { background: #ac3600; color: #f7f4f2; }
    TabbedContent { height: 1fr; background: #101010; color: #f7f4f2; }
    Tabs { background: #202020; color: #f7f4f2; }
    Tab { color: #f7f4f2; }
    Tab.-active { background: #faa881; color: #101010; text-style: bold; }
    TabPane { background: #101010; color: #f7f4f2; }
    .pane { padding: 1; background: #101010; color: #f7f4f2; }
    .split { height: 1fr; }
    .left { width: 2fr; border-right: solid #faa881; padding-right: 1; }
    .right { width: 3fr; padding-left: 1; }
    Tree, DataTable { background: #202020; color: #f7f4f2; }
    DataTable > .datatable--header { background: #0a009f; color: #f7f4f2; }
    DataTable > .datatable--cursor { background: #faa881; color: #101010; }
    #project-tree, #job-table { height: 1fr; }
    #job-details, #metadata-details { height: 1fr; background: #202020; }
    #job-details:focus, #metadata-details:focus { background: #343434; }
    #job-actions { height: auto; margin-top: 1; }
    Button { background: #faa881; color: #101010; border: none; }
    Button.-success { background: #ac3600; color: #f7f4f2; }
    Button:disabled { background: #343434; color: #aaa6ad; }
    #activity-log {
        height: 1fr;
        border: round #ac3600;
        background: #202020;
        color: #f7f4f2;
    }
    #activity-status { height: auto; margin-bottom: 1; color: #faa881; }
    """
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("c", "cancel_job", "Cancel job"),
    ]

    def __init__(self, project: Project, startup_warnings: Optional[list[str]] = None):
        super().__init__()
        self.project = project
        self.sub_title = str(project.project_path)
        self.project_jobs = ProjectJobs(project)
        self.job_summaries: Dict[str, JobSummary] = {}
        self.selected_job_path: Optional[Path] = None
        self.selected_validation: Optional[JobValidation] = None
        self.job_runner: Optional[JobRunner] = None
        self.job_state: Optional[JobState] = None
        self.startup_warnings = startup_warnings or []

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="overview"):
            with TabPane("Overview", id="overview"):
                yield Static(id="overview-content", classes="pane")
            with TabPane("Project", id="project"):
                with Horizontal(classes="pane split"):
                    with Vertical(classes="left"):
                        yield Tree("Project", id="project-tree")
                    with Vertical(classes="right"):
                        with VerticalScroll(id="metadata-details"):
                            yield Static("Select an item", id="metadata-content")
            with TabPane("Jobs", id="jobs"):
                with Horizontal(classes="pane split"):
                    with Vertical(classes="left"):
                        yield DataTable(id="job-table", cursor_type="row")
                    with Vertical(classes="right"):
                        with VerticalScroll(id="job-details"):
                            yield Static("Select a job", id="job-content")
                        with Horizontal(id="job-actions"):
                            yield Button(
                                "Run selected job",
                                id="run-job",
                                variant="success",
                                disabled=True,
                            )
            with TabPane("Activity", id="activity"):
                with Vertical(classes="pane"):
                    yield Static("No active job", id="activity-status")
                    yield RichLog(id="activity-log", markup=True, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        self._populate_overview()
        self._populate_tree()
        self._populate_jobs()
        if self.startup_warnings:
            self.query_one("#activity-log", RichLog).write(
                f"[yellow]Suppressed {len(self.startup_warnings)} warning(s) "
                "while opening the project.[/]"
            )

    def on_unmount(self) -> None:
        self.project.close_mth5()

    def _populate_overview(self) -> None:
        summary = self.project.file_summary()
        content = (
            f"[b]{self.project.project_path.name}[/b]\n\n"
            f"Project: {self.project.project_path}\n"
            f"MTH5: {summary.mth5_path}\n"
            f"MTH5 version: {summary.file_version}\n"
            f"Reference time: {self.project.ref_time}\n"
            f"Time span: {summary.start_time or '-'} → {summary.end_time or '-'}\n"
            "Sample rates: "
            f"{', '.join(str(value) for value in summary.sample_rates) or '-'}\n\n"
            f"Surveys: {summary.n_surveys}\n"
            f"Stations: {summary.n_stations}\n"
            f"Runs: {summary.n_runs}\n"
            f"Channels: {summary.n_channels}"
        )
        self.query_one("#overview-content", Static).update(content)

    def _populate_tree(self) -> None:
        tree = self.query_one("#project-tree", Tree)
        tree.clear()
        tree.root.label = self.project.project_path.name
        tree.root.data = None
        for survey_summary in self.project.list_surveys():
            survey_path = survey_summary.survey
            survey_node = tree.root.add(
                f"{survey_summary.survey} ({survey_summary.n_stations} stations)",
                data=survey_path,
            )
            for station_summary in self.project.list_stations(survey_summary.survey):
                station_node = survey_node.add(
                    f"{station_summary.station} ({station_summary.n_runs} runs)",
                    data=station_summary.station_path,
                )
                for run_summary in self.project.list_runs(
                    survey_summary.survey, station_summary.station
                ):
                    run_node = station_node.add(
                        f"{run_summary.run} ({run_summary.sample_rate:g} Hz)",
                        data=run_summary.run_path,
                    )
                    for channel in self.project.list_channels(
                        run_summary.survey, run_summary.station, run_summary.run
                    ):
                        run_node.add_leaf(
                            channel.component,
                            data=f"{run_summary.run_path}/{channel.component}",
                        )
        tree.root.expand()

    def _populate_jobs(self) -> None:
        table = self.query_one("#job-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Job", "Flow", "Parameters", "Output", "Status")
        self.job_summaries.clear()
        self.selected_job_path = None
        self.selected_validation = None
        self.query_one("#run-job", Button).disabled = True
        for summary in self.project_jobs.list():
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
        if not self.job_summaries:
            self.query_one("#job-content", Static).update(
                "No YAML jobs found in processing/jobs"
            )

    @on(Tree.NodeSelected, "#project-tree")
    def show_metadata(self, event: Tree.NodeSelected) -> None:
        object_path = event.node.data
        details = self.query_one("#metadata-content", Static)
        if object_path is None:
            details.update("Select a survey, station, run, or channel")
            return
        try:
            metadata = self.project.get_metadata(str(object_path))
            details.update(Pretty(metadata.model_dump(mode="json"), expand_all=True))
        except Exception as exc:
            details.update(f"[red]Unable to read metadata:[/red] {exc}")

    @on(DataTable.RowSelected, "#job-table")
    def show_job(self, event: DataTable.RowSelected) -> None:
        key = str(event.row_key.value)
        summary = self.job_summaries[key]
        self.selected_job_path = summary.path
        self.selected_validation = self.project_jobs.validate(summary.path)
        validation = self.selected_validation
        content = {
            "name": summary.name,
            "path": str(summary.path),
            "flow": summary.flow,
            "parameters": summary.parameters,
            "output_label": summary.output_label,
            "valid": validation.ok,
            "errors": validation.errors,
            "warnings": validation.warnings,
        }
        if validation.resolved_job is not None:
            content["runtime"] = validation.resolved_job.definition.runtime
            content["output_path"] = str(validation.resolved_job.output_path)
        self.query_one("#job-content", Static).update(Pretty(content, expand_all=True))
        self.query_one("#run-job", Button).disabled = (
            not validation.ok or self.job_state == JobState.running
        )

    @on(Button.Pressed, "#run-job")
    def confirm_job(self) -> None:
        validation = self.selected_validation
        if validation is None or not validation.ok:
            self.notify("Select a valid job first", severity="warning")
            return
        self.push_screen(ConfirmJobScreen(validation), self._submission_confirmed)

    def _submission_confirmed(self, confirmed: Optional[bool]) -> None:
        if confirmed:
            self._execute_selected_job()

    @work(thread=True, exclusive=True, group="processing-job")
    def _execute_selected_job(self) -> None:
        validation = self.selected_validation
        if validation is None or validation.resolved_job is None:
            raise ValueError("A resolved job is required for execution")
        self.call_from_thread(self._set_running)
        processing_project = None
        try:
            processing_project = load(self.project.project_path)
            self.job_runner = JobRunner(
                processing_project,
                progress_callback=lambda event: self.call_from_thread(
                    self._show_progress, event
                ),
            )
            self.job_runner.run(validation.resolved_job)
        except Exception as exc:
            self.call_from_thread(
                self._show_progress,
                JobProgressEvent(
                    state=JobState.failed,
                    message="Unable to start job",
                    job_name=validation.resolved_job.definition.name,
                    error=str(exc),
                ),
            )
        finally:
            if processing_project is not None:
                processing_project.close_mth5()

    def _set_running(self) -> None:
        self.job_state = JobState.running
        self.query_one("#run-job", Button).disabled = True
        self.query_one("#activity-status", Static).update("Job running")
        self.query_one("#activity-log", RichLog).clear()
        self.query_one(TabbedContent).active = "activity"

    def _show_progress(self, event: JobProgressEvent) -> None:
        self.job_state = event.state
        line = f"[{event.state.value}] {event.message} ({event.elapsed_seconds:.1f}s)"
        if event.error:
            line += f"\n[red]{event.error}[/red]"
        self.query_one("#activity-log", RichLog).write(line)
        self.query_one("#activity-status", Static).update(
            f"{event.job_name}: {event.state.value}"
        )
        if event.state in {JobState.completed, JobState.failed, JobState.cancelled}:
            self.job_runner = None
            self._populate_jobs()

    def action_refresh(self) -> None:
        if self.job_state == JobState.running:
            self.notify("Refresh is unavailable while a job is running")
            return
        self._populate_overview()
        self._populate_tree()
        self._populate_jobs()
        self.notify("Project refreshed")

    def action_cancel_job(self) -> None:
        if self.job_runner is None or self.job_state != JobState.running:
            self.notify("No active job")
            return
        self.job_runner.cancel()
        self.notify("Cancellation requested; the current step will finish first")

    async def action_quit(self) -> None:
        if self.job_state == JobState.running:
            self.notify(
                "A job is running. Press C to request cancellation before quitting.",
                severity="warning",
            )
            return
        self.exit()


def run_tui(project_path: Path) -> None:
    """Load a project and run the terminal UI."""
    logger.remove()
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            project = load(project_path)
        startup_warnings = [str(warning.message) for warning in caught_warnings]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ResisticsTui(project, startup_warnings=startup_warnings).run()
    finally:
        logger.remove()
        logger.add(sys.stderr, level="INFO")
