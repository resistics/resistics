"""Textual application for creating, inspecting, and processing projects."""

from __future__ import annotations

import sys
import warnings
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from textual import on, work
from textual.app import App
from textual.screen import Screen
from textual.worker import Worker, WorkerState

from resistics.tui.logging import (
    _exception_entry,
    _legacy_warning_entry,
    _TuiDiagnosticCapture,
    _TuiLogBuffer,
    _warning_entry,
)
from resistics.tui.screens.dialogs import (
    ConfirmJobScreen as ConfirmJobScreen,
)
from resistics.tui.screens.dialogs import (
    ConfirmProjectDataDeletionScreen as ConfirmProjectDataDeletionScreen,
)
from resistics.tui.screens.dialogs import (
    CopyYamlFileScreen as CopyYamlFileScreen,
)
from resistics.tui.screens.dialogs import (
    CreateJobScreen as CreateJobScreen,
)
from resistics.tui.screens.dialogs import (
    DeleteProjectDataScreen as DeleteProjectDataScreen,
)
from resistics.tui.screens.dialogs import (
    DeleteYamlFileScreen as DeleteYamlFileScreen,
)
from resistics.tui.screens.dialogs import (
    DirectoryPickerScreen as DirectoryPickerScreen,
)
from resistics.tui.screens.launcher import (
    CreateProjectScreen as CreateProjectScreen,
)
from resistics.tui.screens.launcher import (
    HomeScreen as HomeScreen,
)
from resistics.tui.screens.launcher import (
    ProjectLoadingScreen as ProjectLoadingScreen,
)
from resistics.tui.screens.launcher import (
    TuiHeader as TuiHeader,
)
from resistics.tui.screens.project import (
    ProjectExplorerScreen as ProjectExplorerScreen,
)
from resistics.tui.services import (
    _feature_error,
    _run_in_worker_thread,
)
from resistics.tui.state import (
    TIME_PLOT_MAX_POINTS as TIME_PLOT_MAX_POINTS,
)
from resistics.tui.state import (
    DiagnosticLogEntry as DiagnosticLogEntry,
)
from resistics.tui.state import (
    ExplorerView as ExplorerView,
)
from resistics.tui.state import (
    PlotTarget as PlotTarget,
)
from resistics.tui.state import (
    ProjectDataDeletionRequest as ProjectDataDeletionRequest,
)
from resistics.tui.state import (
    TimePlotSelection as TimePlotSelection,
)
from resistics.tui.state import (
    _ProjectOpenResult,
)

if TYPE_CHECKING:
    from resistics.project import (
        Project,
    )


class ResisticsTui(App[None]):
    """The resistics terminal application and project launcher.

    :param project_path: Optional project opened immediately after mount.
    :param _diagnostic_buffer: Buffer installed by the official TUI launcher.
    :param _diagnostic_capture: Process-global capture installed by the official TUI launcher.
    """

    TITLE = "resistics"
    SUB_TITLE = ""
    NOTIFICATION_TIMEOUT = 4.0
    CSS = """
    Screen { layout: vertical; background: #101010; color: #f7f4f2; }
    Widget {
        scrollbar-color: #003054;
        scrollbar-color-hover: #003c6a;
        scrollbar-color-active: #0178d4;
        scrollbar-background: #101010;
        scrollbar-background-hover: #101010;
        scrollbar-background-active: #101010;
        scrollbar-corner-color: #101010;
    }
    #app-header {
        height: 1;
        padding: 0 1;
        background: #0a009f;
        color: #f7f4f2;
    }
    Footer { background: #070066; color: #faa881; }
    Footer > .footer--key { background: #faa881; color: #101010; }
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
    #home-message { margin-top: 1; color: #faa881; }
    #home-message.status-error { color: #ff6b6b; }
    .form-status { height: auto; margin-top: 1; color: #faa881; }
    .form-status.status-error { color: #ff6b6b; }
    .form-status.status-warning { color: #f0c674; }
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
    .pane {
        height: 1fr;
        padding: 0 0 1 0;
        background: #101010;
        color: #f7f4f2;
    }
    .tab-explainer {
        height: auto;
        min-height: 1;
        padding: 0 1;
        margin-bottom: 1;
        color: #aaa6ad;
    }
    .tab-body { height: 1fr; }
    .split { height: 1fr; }
    .left { width: 2fr; padding-right: 1; }
    .right { width: 3fr; }
    Tree, DataTable {
        padding: 0;
        background: #202020;
        color: #f7f4f2;
        border: heavy #555555;
        outline: none;
    }
    Tree:focus, DataTable:focus {
        background-tint: 0%;
        border: heavy #003054;
        outline: none;
    }
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
        padding: 0;
        background: #202020;
        color: #f7f4f2;
        border: heavy #555555;
        outline: none;
    }
    #data-metadata:focus, #flow-content:focus, #parameter-content:focus,
    #criteria-content:focus, #job-content:focus {
        background-tint: 0%;
        border: heavy #003054;
        outline: none;
    }
    .panel-state {
        height: 1fr;
        padding: 0;
        content-align: center middle;
        text-align: center;
        background: #202020;
        color: #aaa6ad;
        border: heavy #555555;
        outline: none;
    }
    .panel-state.panel-state-error { color: #ff6b6b; }
    Button {
        background: #343434;
        color: #f7f4f2;
        border: none;
        outline: none;
        text-style: none;
    }
    Button:focus {
        background: #003054;
        color: #f7f4f2;
        background-tint: 0%;
        outline: none;
        text-style: bold;
    }
    Button.-active { tint: transparent; }
    Button:disabled { background: #202020; color: #aaa6ad; }
    ModalScreen { align: center middle; background: transparent; }
    .modal-dialog {
        width: 72;
        height: auto;
        padding: 1 2;
        border: round #faa881;
        background: #202020;
        color: #f7f4f2;
    }
    .modal-dialog.modal-danger { border: round #ac3600; }
    .modal-title { height: auto; text-style: bold; margin-bottom: 1; }
    .modal-actions { height: auto; align-horizontal: right; margin-top: 1; }
    .modal-actions Button { margin-left: 1; }
    #activity-log, #logs-log {
        height: 1fr;
        padding: 0;
        border: heavy #555555;
        outline: none;
        background: #202020;
        color: #f7f4f2;
    }
    #activity-log:focus, #logs-log:focus {
        background-tint: 0%;
        border: heavy #003054;
        outline: none;
    }
    #activity-status, #logs-status {
        height: auto;
        padding: 0 1;
        margin-bottom: 1;
        color: #faa881;
    }
    """

    def __init__(
        self,
        project_path: Path | None = None,
        *,
        _diagnostic_buffer: _TuiLogBuffer | None = None,
        _diagnostic_capture: _TuiDiagnosticCapture | None = None,
    ):
        super().__init__()
        self.initial_project_path = project_path
        self.diagnostic_buffer = _diagnostic_buffer or _TuiLogBuffer()
        self._diagnostic_capture = _diagnostic_capture
        self._has_started_screen = False
        self._project_open_generation = 0

    def _reinstall_diagnostic_capture(self) -> None:
        """Restore the TUI sink after a lazy dependency changes Loguru globally."""
        if self._diagnostic_capture is not None:
            self._diagnostic_capture.reinstall()

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

    def _mount_home(self, message: str | None = None, *, error: bool = False) -> None:
        """Replace the current screen with the project launcher.

        :param message: Optional status or failure detail shown on the launcher.
        :param error: Whether the message describes a failed operation.
        """
        self.title = "resistics"
        self.sub_title = "project launcher"
        self._show_screen(HomeScreen(message, message_is_error=error))

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

        :param generation: App generation requesting the project.
        :param project_path: Project directory to open.

        :return: Immutable success or failure passed back to the UI thread.
        """
        return await _run_in_worker_thread(
            partial(self._open_project_path, generation, project_path)
        )

    def _open_project_path(
        self, generation: int, project_path: Path
    ) -> _ProjectOpenResult:
        """Perform synchronous project opening inside a worker thread.

        :param generation: App generation requesting the project.
        :param project_path: Project directory to open.

        :return: Immutable project-open outcome containing no widgets.
        """
        try:
            from resistics.project import load

            self._reinstall_diagnostic_capture()
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                project = load(project_path)
        except Exception as exc:
            error = _feature_error("Project loading", exc)
            return _ProjectOpenResult(
                generation=generation,
                project_path=project_path,
                diagnostics=(_exception_entry("Project loading", error, exc),),
                error=error,
            )
        finally:
            # MTH5 configures the process-global Loguru logger on its first
            # lazy import, replacing the sink installed before project loading.
            self._reinstall_diagnostic_capture()
        if generation != self._project_open_generation:
            try:
                project.close()
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
            diagnostics=tuple(_warning_entry(warning) for warning in caught_warnings),
        )

    @on(Worker.StateChanged)
    def _apply_project_open_result(self, event: Worker.StateChanged) -> None:
        """Open only the latest project returned to the Textual UI thread.

        :param event: Textual lifecycle event for a project-open worker.
        """
        if event.state != WorkerState.SUCCESS or event.worker.group != "project-open":
            return
        result = event.worker.result
        if not isinstance(result, _ProjectOpenResult):
            return
        if result.generation != self._project_open_generation:
            if result.project is not None:
                result.project.close()
            return
        self.diagnostic_buffer.extend(result.diagnostics)
        if result.error is not None or result.project is None:
            self._mount_home(
                f"Unable to open project: {result.error or 'Unknown error'}",
                error=True,
            )
            return
        self._mount_project(result.project)

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

        :param project: Open project returned by a completed worker or direct caller.
        :param startup_warnings: Warnings captured while opening the project.
        """
        if startup_warnings:
            self.diagnostic_buffer.extend(
                tuple(_legacy_warning_entry(message) for message in startup_warnings)
            )
        self.title = "resistics"
        self.sub_title = str(project.project_path)
        self._show_screen(
            ProjectExplorerScreen(project, log_buffer=self.diagnostic_buffer)
        )

    def _show_screen(self, screen: Screen[None]) -> None:
        """Push the initial screen and replace it for later navigation.

        :param screen: Screen used by this operation.
        """
        if self._has_started_screen:
            self.switch_screen(screen)
        else:
            self.push_screen(screen)
            self._has_started_screen = True


def run_tui(project_path: Path | None = None) -> None:
    """Run the resistics terminal application.

    :param project_path: Project root used to locate configuration and artifacts.
    """
    buffer = _TuiLogBuffer()
    with _TuiDiagnosticCapture(buffer) as capture:
        ResisticsTui(
            project_path,
            _diagnostic_buffer=buffer,
            _diagnostic_capture=capture,
        ).run()


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the TUI, optionally opening one project path immediately.

    :param argv: Command-line arguments, or the process arguments when omitted.
    :return: Launch the TUI, optionally opening one project path immediately.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) > 1:
        print("Usage: resistics [PROJECT_PATH]", file=sys.stderr)
        return 2
    project_path = Path(arguments[0]) if arguments else None
    run_tui(project_path)
    return 0
