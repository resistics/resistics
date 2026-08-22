"""Project explorer screen and Textual presentation wiring."""

from __future__ import annotations

from functools import partial

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
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

from resistics.tui.logging import _exception_entry
from resistics.tui.screens.launcher import TuiHeader
from resistics.tui.screens.project_data import _ProjectDataMixin
from resistics.tui.screens.project_help import (
    TAB_HELP,
    ProjectTab,
    ProjectTabHelpScreen,
)
from resistics.tui.screens.project_jobs import _ProjectJobsMixin
from resistics.tui.screens.project_logs import _ProjectLogsMixin
from resistics.tui.screens.project_resources import _ProjectResourcesMixin
from resistics.tui.services import (
    _resistics_app,
    _run_in_worker_thread,
)
from resistics.tui.state import (
    ExplorerView,
    _ExplorerLoadResult,
)


class ProjectExplorerScreen(
    _ProjectDataMixin,
    _ProjectJobsMixin,
    _ProjectLogsMixin,
    _ProjectResourcesMixin,
):
    """Project browser and YAML editor with managed processing-job execution."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("d", "restore_defaults", "Restore defaults"),
        ("n", "create_job", "New job"),
        ("e", "edit_yaml", "Edit YAML"),
        ("y", "copy_yaml", "Copy YAML"),
        Binding("delete", "delete_project_data", "Delete derived data"),
        Binding("delete", "delete_yaml", "Delete YAML"),
        ("ctrl+s", "save_yaml", "Save YAML"),
        Binding("escape", "discard_yaml", "Discard YAML", priority=True),
        ("j", "run_selected_job", "Run job"),
        ("p", "plot", "Plot"),
        ("h", "show_help", "Help"),
        ("right_square_bracket", "expand_data_node", "Expand"),
        ("left_square_bracket", "collapse_data_node", "Collapse"),
        ("c", "cancel_job", "Cancel job"),
        ("x", "close_project", "Close project"),
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield TuiHeader(id="app-header")
        with TabbedContent(initial="project"):
            with TabPane("Project", id="project"):
                with Vertical(classes="pane"):
                    yield self._tab_explainer("project")
                    with VerticalScroll(classes="tab-body"):
                        yield Static(id="project-content")
            with TabPane("Data", id="data"):
                with Vertical(classes="pane"):
                    yield self._tab_explainer("data")
                    with Horizontal(classes="split"):
                        with Vertical(classes="left"):
                            data_tree = Tree("Data", id="data-tree")
                            data_tree.show_root = False
                            yield data_tree
                        with Vertical(classes="right"):
                            yield Static(
                                "Select project or MTH5 data to view its metadata.",
                                id="data-metadata-state",
                                classes="panel-state",
                                markup=False,
                            )
                            editor = TextArea.code_editor(
                                "",
                                language="json",
                                theme="vscode_dark",
                                read_only=True,
                                id="data-metadata",
                            )
                            editor.display = False
                            yield editor
            with TabPane("Flows", id="flows"), Vertical(classes="pane"):
                yield self._tab_explainer("flows")
                with Horizontal(classes="split"):
                    with Vertical(classes="left"):
                        yield DataTable(id="flow-table", cursor_type="row")
                    with Vertical(classes="right"):
                        yield Static(
                            "Select a flow to view its YAML.",
                            id="flow-content-state",
                            classes="panel-state",
                            markup=False,
                        )
                        editor = TextArea.code_editor(
                            "",
                            language="yaml",
                            theme="vscode_dark",
                            read_only=True,
                            id="flow-content",
                        )
                        editor.display = False
                        yield editor
            with TabPane("Parameters", id="parameters"):
                with Vertical(classes="pane"):
                    yield self._tab_explainer("parameters")
                    with Horizontal(classes="split"):
                        with Vertical(classes="left"):
                            yield DataTable(id="parameter-table", cursor_type="row")
                        with Vertical(classes="right"):
                            yield Static(
                                "Select a parameter set to view its YAML.",
                                id="parameter-content-state",
                                classes="panel-state",
                                markup=False,
                            )
                            editor = TextArea.code_editor(
                                "",
                                language="yaml",
                                theme="vscode_dark",
                                read_only=True,
                                id="parameter-content",
                            )
                            editor.display = False
                            yield editor
            with TabPane("Criteria", id="criteria"):
                with Vertical(classes="pane"):
                    yield self._tab_explainer("criteria")
                    with Horizontal(classes="split"):
                        with Vertical(classes="left"):
                            yield DataTable(id="criteria-table", cursor_type="row")
                        with Vertical(classes="right"):
                            yield Static(
                                "Select a criteria file to view its YAML.",
                                id="criteria-content-state",
                                classes="panel-state",
                                markup=False,
                            )
                            editor = TextArea.code_editor(
                                "",
                                language="yaml",
                                theme="vscode_dark",
                                read_only=True,
                                id="criteria-content",
                            )
                            editor.display = False
                            yield editor
            with TabPane("Jobs", id="jobs"), Vertical(classes="pane"):
                yield self._tab_explainer("jobs")
                with Horizontal(classes="split"):
                    with Vertical(classes="left"):
                        yield DataTable(id="job-table", cursor_type="row")
                    with Vertical(classes="right"):
                        yield Static(
                            "Select a job to view its YAML.",
                            id="job-content-state",
                            classes="panel-state",
                            markup=False,
                        )
                        editor = TextArea.code_editor(
                            "",
                            language="yaml",
                            theme="vscode_dark",
                            read_only=True,
                            id="job-content",
                        )
                        editor.display = False
                        yield editor
            with TabPane("Activity", id="activity"):
                with Vertical(classes="pane"):
                    yield self._tab_explainer("activity")
                    yield Static("No active job", id="activity-status")
                    yield RichLog(id="activity-log", markup=False, wrap=True)
            with TabPane("Logs", id="logs"):
                with Vertical(classes="pane"):
                    yield self._tab_explainer("logs")
                    yield Static("Session logs · INFO+", id="logs-status")
                    yield RichLog(
                        id="logs-log",
                        max_lines=4_000,
                        wrap=True,
                        auto_scroll=True,
                    )
        yield Footer()

    @staticmethod
    def _tab_explainer(tab: ProjectTab) -> Static:
        """Return the consistently styled summary for one project tab.

        :param tab: Project tab whose summary should be displayed.
        :return: Static explanatory text for the tab.
        """
        return Static(TAB_HELP[tab].summary, classes="tab-explainer")

    def on_mount(self) -> None:
        self.query_one("#project-content", Static).update(
            "[bold]Loading project overview…[/bold]"
        )
        self._start_log_updates()
        self._request_explorer_section("project")

    def on_unmount(self) -> None:
        self._load_generation += 1
        self.workers.cancel_group(self, "explorer-load")
        self.service.close()

    @work(group="explorer-load", exit_on_error=False)
    async def _load_explorer_section(
        self, generation: int, section: ExplorerView
    ) -> _ExplorerLoadResult:
        """Load one explorer section without touching Textual widgets.

        :param generation: Screen generation requesting the load.
        :param section: Explorer section to discover.

        :return: Immutable result consumed on the Textual UI thread.
        """
        return await _run_in_worker_thread(
            partial(self._discover_explorer_section, generation, section)
        )

    def _discover_explorer_section(
        self, generation: int, section: ExplorerView
    ) -> _ExplorerLoadResult:
        """Perform one synchronous discovery operation in a worker thread.

        :param generation: Screen generation requesting the load.
        :param section: Explorer section to discover.

        :return: Immutable success or failure result containing no widgets.
        """
        try:
            with self.service.discovery():
                if section in {"project", "data"}:
                    state = self.service.project_state()
                    runs = self.service.runs() if section == "data" else ()
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
                        jobs=self.service.jobs(),
                    )
                return _ExplorerLoadResult(
                    generation=generation,
                    section=section,
                    resources=self.service.resources(section),
                )
        except Exception as exc:
            source = f"{section.title()} loading"
            message = str(exc)
            return _ExplorerLoadResult(
                generation=generation,
                section=section,
                error=message,
                diagnostic=_exception_entry(source, message, exc),
            )

    def _request_explorer_section(
        self, section: ExplorerView, *, force: bool = False
    ) -> None:
        """Start one lazy section load unless its current generation is ready.

        :param section: Explorer view whose cached data is required.
        :param force: Reload a section even when it is already marked ready.
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

        :param section: Explorer section whose content is loading.
        """
        if section == "project":
            self.query_one("#project-content", Static).update(
                "[bold]Loading project overview…[/bold]"
            )
        elif section == "data":
            tree = self.query_one("#data-tree", Tree)
            tree.clear()
            tree.root.add_leaf("Loading project data…")
            self._show_detail_state("#data-metadata", "Loading metadata…")
        else:
            editor_ids = {
                "flows": "#flow-content",
                "parameters": "#parameter-content",
                "criteria": "#criteria-content",
                "jobs": "#job-content",
            }
            self._show_detail_state(editor_ids[section], "Loading…")

    @on(Worker.StateChanged)
    def _apply_explorer_worker_result(self, event: Worker.StateChanged) -> None:
        """Apply successful current-generation discovery on the UI thread.

        :param event: Textual lifecycle event for a discovery worker.
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
            if result.diagnostic is not None:
                self.log_buffer.append(result.diagnostic)
            self._show_section_error(result.section, result.error)
            self.refresh_bindings()
            return
        with self.app.batch_update():
            self._render_explorer_result(result)
        self._loaded_sections.add(result.section)
        self.refresh_bindings()

    def _render_explorer_result(self, result: _ExplorerLoadResult) -> None:
        """Mutate widgets from one current worker result on the UI thread.

        :param result: Successful current-generation discovery result.
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

    def _show_section_error(self, section: ExplorerView, error: str) -> None:
        """Replace one loading placeholder with a user-facing failure.

        :param section: Explorer section whose load failed.
        :param error: Failure detail returned by the worker.
        """
        labels = {
            "project": "project overview",
            "data": "data",
            "flows": "flows",
            "parameters": "parameter sets",
            "criteria": "gather criteria",
            "jobs": "jobs",
        }
        message = f"Unable to load {labels[section]}: {error}"
        if section == "project":
            from rich.text import Text

            self.query_one("#project-content", Static).update(
                Text(message, style="red")
            )
        elif section == "data":
            tree = self.query_one("#data-tree", Tree)
            tree.clear()
            tree.root.add_leaf(message)
            self._show_detail_state("#data-metadata", message, error=True)
        else:
            editor_ids = {
                "flows": "#flow-content",
                "parameters": "#parameter-content",
                "criteria": "#criteria-content",
                "jobs": "#job-content",
            }
            self._show_detail_state(editor_ids[section], message, error=True)
        self.notify(
            f"{message}\nSee Logs for the full traceback.",
            severity="error",
            markup=False,
        )

    def _start_new_load_generation(self) -> None:
        """Reject outstanding results before a cache invalidation transition."""
        self._load_generation += 1
        self._loading_sections.clear()
        self.workers.cancel_group(self, "explorer-load")

    @on(TabbedContent.TabActivated)
    def refresh_tab_bindings(self) -> None:
        """Lazily load the active explorer tab and refresh its Footer."""
        active = self.query_one(TabbedContent).active
        if active in {"project", "data", "flows", "parameters", "criteria", "jobs"}:
            self._request_explorer_section(active)
        elif active == "logs":
            self._activate_log_view()
        self.refresh_bindings()

    def _check_resource_action(self, action: str, active: str) -> bool:
        """Check one YAML-resource action using only in-memory state.

        :param action: Action name to check.
        :param active: Identifier of the active tab.

        :return: Whether the action is currently available.
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
        if action == "delete_project_data":
            return (
                active == "data"
                and not self.editing_yaml
                and self.job_state != self._job_state_type.running
                and self.action_state.has_project_data_to_delete
            )
        if action == "delete_yaml":
            return (
                active in {"flows", "parameters", "criteria", "jobs"}
                and not self.editing_yaml
                and self.job_state != self._job_state_type.running
                and (
                    self._highlighted_yaml_file() is not None
                    or self._selected_yaml_file() is not None
                )
            )
        if action == "copy_yaml":
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

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Expose only Footer actions relevant to cached screen state.

        :param action: Action whose current availability is being evaluated.
        :param parameters: Callback or command parameters.
        :return: Whether the action is enabled, or ``None`` to defer to Textual.
        """
        tabbed_content = self.query(TabbedContent)
        if not tabbed_content.nodes:
            return False
        active = tabbed_content.first(TabbedContent).active
        resource_actions = {
            "edit_yaml",
            "create_job",
            "copy_yaml",
            "delete_yaml",
            "delete_project_data",
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
        if action == "show_help":
            return not self.editing_yaml
        return super().check_action(action, parameters)

    def action_show_help(self) -> None:
        """Open contextual help for the active project tab."""
        active = self.query_one(TabbedContent).active
        if active not in TAB_HELP or self.editing_yaml:
            return
        self.app.push_screen(ProjectTabHelpScreen(active))

    def action_refresh(self) -> None:
        if self.job_state == self._job_state_type.running:
            self.notify(
                "Refresh is unavailable while a job is running", severity="warning"
            )
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
        self.service.invalidate_all()
        self._loaded_sections.clear()
        self._request_explorer_section(section, force=True)
        self.refresh_bindings()
        self.notify("Project refreshed")

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
