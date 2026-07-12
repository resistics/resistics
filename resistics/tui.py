"""Terminal user interface for creating, inspecting, and processing projects."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from tempfile import NamedTemporaryFile
from typing import Dict, Optional, Sequence
import warnings

from loguru import logger
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    DataTable,
    DirectoryTree,
    Footer,
    Input,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
    Tree,
)

from resistics.job import (
    JobDefinition,
    JobProgressEvent,
    JobRunner,
    JobState,
    JobSummary,
    JobValidation,
    ProjectJobs,
    validate_job_template_name,
)
from resistics.flow import (
    FlowDefinition,
    ParameterSet,
    model_from_yaml,
    model_from_yaml_file,
)
from resistics.gather import GatherCriteria
from resistics.project import Project, init as init_project, load, open_mth5
from resistics.sampling import to_datetime
from resistics.templates import (
    install_builtin_criteria_templates,
    install_builtin_flow_templates,
    install_builtin_parameter_templates,
)


class TuiHeader(Static):
    """Render the application title without Header's reactive mount timing."""

    def on_mount(self) -> None:
        title = f"[bold]{self.app.title}[/bold]"
        if self.app.sub_title:
            title += f" [dim]— {self.app.sub_title}[/]"
        self.update(title)


class ConfirmJobScreen(ModalScreen[bool]):
    """Confirm submission of an already validated job."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        Binding("left", "previous_action", "Previous action", priority=True),
        Binding("right", "next_action", "Next action", priority=True),
    ]

    CSS = """
    ConfirmJobScreen { align: center middle; background: transparent; }
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
        stages = ", ".join(definition.scope.stages) or "all stages"
        details = (
            f"Submit [bold]{definition.name}[/bold]?\n\n"
            f"Flow: {definition.flow}\n"
            f"Parameters: {definition.parameters}\n"
            f"Criteria: {definition.criteria or '-'}\n"
            f"Stages: {stages}\n"
            f"Output label: {definition.output_label}"
        )
        with Vertical(id="confirm-dialog"):
            yield Static(details)
            with Horizontal(id="confirm-actions"):
                yield Button("Cancel", id="cancel", classes="dialog-action")
                yield Button(
                    "Run job",
                    id="confirm",
                    variant="success",
                    classes="dialog-action",
                )

    def on_mount(self) -> None:
        self.query_one("#cancel", Button).focus()

    @on(Button.Pressed, "#cancel")
    def cancel(self) -> None:
        self.action_cancel()

    def action_cancel(self) -> None:
        self.dismiss(False)

    def action_next_action(self) -> None:
        self._focus_action(1)

    def action_previous_action(self) -> None:
        self._focus_action(-1)

    @on(Button.Pressed, "#confirm")
    def confirm(self) -> None:
        self.dismiss(True)

    def _focus_action(self, increment: int) -> None:
        actions = [
            self.query_one("#cancel", Button),
            self.query_one("#confirm", Button),
        ]
        try:
            index = actions.index(self.focused)
        except ValueError:
            index = 0
        actions[(index + increment) % len(actions)].focus()


_NO_CRITERIA_VALUE = "__no_criteria__"
_YAML_FILE_STEM = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class CreateJobScreen(ModalScreen[Optional[JobDefinition]]):
    """Create a minimal job template from project YAML resources."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        Binding("left", "previous_action", "Previous action", priority=True),
        Binding("right", "next_action", "Next action", priority=True),
    ]

    CSS = """
    CreateJobScreen { align: center middle; }
    #create-job-dialog {
        width: 72;
        height: auto;
        padding: 1 2;
        border: round #faa881;
        background: #202020;
        color: #f7f4f2;
    }
    #create-job-dialog Input, #create-job-dialog Select { margin-bottom: 1; }
    #create-job-status { height: auto; color: #faa881; }
    #create-job-actions { height: auto; align-horizontal: right; margin-top: 1; }
    #create-job-actions Button { margin-left: 1; }
    """

    def __init__(
        self,
        flow_options: Sequence[tuple[str, str]],
        parameter_options: Sequence[tuple[str, str]],
        criteria_options: Sequence[tuple[str, str]],
        existing_names: set[str],
    ):
        super().__init__()
        self.flow_options = list(flow_options)
        self.parameter_options = list(parameter_options)
        self.criteria_options = [("No criteria", _NO_CRITERIA_VALUE)] + list(
            criteria_options
        )
        self.existing_names = existing_names

    def compose(self) -> ComposeResult:
        can_create = bool(self.flow_options and self.parameter_options)
        flow_value = self.flow_options[0][1] if self.flow_options else Select.NULL
        parameter_value = (
            self.parameter_options[0][1] if self.parameter_options else Select.NULL
        )
        with Vertical(id="create-job-dialog"):
            yield Static("[bold]Create job template[/bold]")
            yield Static("Job name")
            yield Input(placeholder="my_job", id="job-name")
            yield Static("Flow")
            yield Select(
                self.flow_options,
                prompt="Select a flow",
                allow_blank=not self.flow_options,
                value=flow_value,
                id="job-flow",
            )
            yield Static("Parameters")
            yield Select(
                self.parameter_options,
                prompt="Select parameters",
                allow_blank=not self.parameter_options,
                value=parameter_value,
                id="job-parameters",
            )
            yield Static("Criteria")
            yield Select(
                self.criteria_options,
                allow_blank=False,
                value=_NO_CRITERIA_VALUE,
                id="job-criteria",
            )
            yield Static("", id="create-job-status")
            with Horizontal(id="create-job-actions"):
                yield Button("Cancel", id="cancel-job-template", classes="dialog-action")
                yield Button(
                    "Create job",
                    id="create-job-template",
                    variant="success",
                    classes="dialog-action",
                    disabled=not can_create,
                )

    def on_mount(self) -> None:
        self.query_one("#job-name", Input).focus()
        if not self.flow_options or not self.parameter_options:
            self._set_status(
                "Add valid flow and parameter YAML files before creating a job"
            )

    @on(Button.Pressed, "#cancel-job-template")
    def cancel(self) -> None:
        self.action_cancel()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_next_action(self) -> None:
        self._focus_action(1)

    def action_previous_action(self) -> None:
        self._focus_action(-1)

    @on(Button.Pressed, "#create-job-template")
    def create(self) -> None:
        name = self.query_one("#job-name", Input).value
        try:
            name = validate_job_template_name(name)
        except ValueError as exc:
            self._set_status(str(exc))
            return
        if name in self.existing_names:
            self._set_status(f"A job named '{name}' already exists")
            return
        flow = self._select_value("#job-flow")
        parameters = self._select_value("#job-parameters")
        criteria = self._select_value("#job-criteria")
        if flow is None or parameters is None:
            self._set_status("Choose a flow and parameters file")
            return
        self.dismiss(
            JobDefinition(
                name=name,
                flow=flow,
                parameters=parameters,
                criteria=None if criteria == _NO_CRITERIA_VALUE else criteria,
            )
        )

    def _select_value(self, selector: str) -> Optional[str]:
        value = self.query_one(selector, Select).value
        return None if value is Select.NULL else str(value)

    def _set_status(self, message: str) -> None:
        self.query_one("#create-job-status", Static).update(message)

    def _focus_action(self, increment: int) -> None:
        actions = [
            self.query_one("#cancel-job-template", Button),
            self.query_one("#create-job-template", Button),
        ]
        if self.focused not in actions:
            return
        actions[(actions.index(self.focused) + increment) % len(actions)].focus()


class CopyYamlFileScreen(ModalScreen[Optional[str]]):
    """Ask for the filename stem of a YAML copy."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        Binding("left", "previous_action", "Previous action", priority=True),
        Binding("right", "next_action", "Next action", priority=True),
    ]

    CSS = """
    CopyYamlFileScreen { align: center middle; background: transparent; }
    #copy-yaml-dialog {
        width: 72;
        height: auto;
        padding: 1 2;
        border: round #faa881;
        background: #202020;
        color: #f7f4f2;
    }
    #copy-yaml-name { margin-top: 1; }
    #copy-yaml-status { height: auto; margin-top: 1; color: #faa881; }
    #copy-yaml-actions { height: auto; align-horizontal: right; margin-top: 1; }
    #copy-yaml-actions Button { margin-left: 1; }
    """

    def __init__(self, source: Path):
        super().__init__()
        self.source = source

    def compose(self) -> ComposeResult:
        with Vertical(id="copy-yaml-dialog"):
            yield Static(f"[bold]Copy {self.source.name}[/bold]")
            yield Static("The YAML content will be copied unchanged.")
            yield Input(value=f"{self.source.stem}_copy", id="copy-yaml-name")
            yield Static("", id="copy-yaml-status")
            with Horizontal(id="copy-yaml-actions"):
                yield Button("Cancel", id="cancel-copy-yaml", classes="dialog-action")
                yield Button(
                    "Copy file",
                    id="confirm-copy-yaml",
                    variant="success",
                    classes="dialog-action",
                )

    def on_mount(self) -> None:
        self.query_one("#copy-yaml-name", Input).focus()

    @on(Button.Pressed, "#cancel-copy-yaml")
    def cancel(self) -> None:
        self.action_cancel()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_next_action(self) -> None:
        self._focus_action(1)

    def action_previous_action(self) -> None:
        self._focus_action(-1)

    @on(Button.Pressed, "#confirm-copy-yaml")
    def copy(self) -> None:
        try:
            self.dismiss(_validate_yaml_file_stem(self.query_one("#copy-yaml-name", Input).value))
        except ValueError as exc:
            self.query_one("#copy-yaml-status", Static).update(str(exc))

    def _focus_action(self, increment: int) -> None:
        actions = [
            self.query_one("#cancel-copy-yaml", Button),
            self.query_one("#confirm-copy-yaml", Button),
        ]
        try:
            index = actions.index(self.focused)
        except ValueError:
            index = 0
        actions[(index + increment) % len(actions)].focus()


class DeleteYamlFileScreen(ModalScreen[bool]):
    """Require explicit confirmation before deleting a project YAML file."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        Binding("left", "previous_action", "Previous action", priority=True),
        Binding("right", "next_action", "Next action", priority=True),
    ]

    CSS = """
    DeleteYamlFileScreen { align: center middle; background: transparent; }
    #delete-yaml-dialog {
        width: 72;
        height: auto;
        padding: 1 2;
        border: round #ac3600;
        background: #202020;
        color: #f7f4f2;
    }
    #delete-yaml-actions { height: auto; align-horizontal: right; margin-top: 1; }
    #delete-yaml-actions Button { margin-left: 1; }
    """

    def __init__(self, source: Path):
        super().__init__()
        self.source = source

    def compose(self) -> ComposeResult:
        with Vertical(id="delete-yaml-dialog"):
            yield Static(
                f"Delete [bold]{self.source.name}[/bold]?\n\nThis cannot be undone."
            )
            with Horizontal(id="delete-yaml-actions"):
                yield Button(
                    "Cancel", id="cancel-delete-yaml", classes="dialog-action"
                )
                yield Button(
                    "Delete file",
                    id="confirm-delete-yaml",
                    variant="error",
                    classes="dialog-action",
                )

    def on_mount(self) -> None:
        self.query_one("#cancel-delete-yaml", Button).focus()

    @on(Button.Pressed, "#cancel-delete-yaml")
    def cancel(self) -> None:
        self.action_cancel()

    def action_cancel(self) -> None:
        self.dismiss(False)

    def action_next_action(self) -> None:
        self._focus_action(1)

    def action_previous_action(self) -> None:
        self._focus_action(-1)

    @on(Button.Pressed, "#confirm-delete-yaml")
    def delete(self) -> None:
        self.dismiss(True)

    def _focus_action(self, increment: int) -> None:
        actions = [
            self.query_one("#cancel-delete-yaml", Button),
            self.query_one("#confirm-delete-yaml", Button),
        ]
        try:
            index = actions.index(self.focused)
        except ValueError:
            index = 0
        actions[(index + increment) % len(actions)].focus()


def _validate_yaml_file_stem(value: str) -> str:
    """Validate a filename stem used when copying a project YAML file."""
    value = value.strip()
    if not _YAML_FILE_STEM.fullmatch(value):
        raise ValueError(
            "New name must use letters, numbers, dots, underscores, or hyphens"
        )
    if Path(value).suffix in {".yaml", ".yml"}:
        raise ValueError("New name must not include a YAML extension")
    return value


class DirectoryPickerScreen(ModalScreen[Optional[Path]]):
    """Select either a directory or file with the terminal file browser."""

    BINDINGS = [("u", "parent_directory", "Up"), ("escape", "cancel", "Cancel")]

    CSS = """
    DirectoryPickerScreen { align: center middle; }
    #path-picker-dialog {
        width: 80%;
        height: 80%;
        padding: 1 2;
        border: round #faa881;
        background: #202020;
    }
    #path-picker { height: 1fr; background: #202020; color: #f7f4f2; }
    #path-picker:focus { background: #343434; }
    #path-picker-help, #path-picker-path {
        height: auto;
        margin-bottom: 1;
        color: #aaa6ad;
    }
    #path-picker-actions { height: auto; align-horizontal: right; margin-top: 1; }
    #path-picker-actions Button { margin-left: 1; }
    """

    def __init__(
        self, title: str, select_files: bool, start_path: Optional[Path] = None
    ):
        super().__init__()
        self.title = title
        self.select_files = select_files
        self.start_path = start_path or Path.home()
        self.navigation_instruction = (
            "Up/Down: move  •  Space: expand/collapse  •  U or Up: parent"
        )
        self.selection_instruction = (
            "Press Enter on the MTH5 file to select it."
            if select_files
            else "Press Enter on the resistics project folder to select it."
        )

    def compose(self) -> ComposeResult:
        with Vertical(id="path-picker-dialog"):
            yield Static(f"[bold]{self.title}[/bold]")
            yield Static(
                f"{self.navigation_instruction}\n{self.selection_instruction}",
                id="path-picker-help",
            )
            yield Static(str(self.start_path), id="path-picker-path")
            yield DirectoryTree(self.start_path, id="path-picker")
            with Horizontal(id="path-picker-actions"):
                yield Button("Up", id="up-picker")
                yield Button("Cancel", id="cancel-picker")

    def on_mount(self) -> None:
        self.query_one("#path-picker", DirectoryTree).focus()

    @on(Button.Pressed, "#up-picker")
    def go_to_parent_directory(self) -> None:
        self.action_parent_directory()

    def action_parent_directory(self) -> None:
        """Move the browser root to its parent directory."""
        tree = self.query_one("#path-picker", DirectoryTree)
        parent_path = tree.path.parent
        if parent_path == tree.path:
            self.notify("Already at the filesystem root")
            return
        tree.path = parent_path
        self.query_one("#path-picker-path", Static).update(str(parent_path))

    @on(Button.Pressed, "#cancel-picker")
    def cancel(self) -> None:
        self.action_cancel()

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(DirectoryTree.DirectorySelected, "#path-picker")
    def select_directory(self, event: DirectoryTree.DirectorySelected) -> None:
        if self.select_files:
            self.notify("Select an MTH5 file, not a directory", severity="warning")
            return
        self.dismiss(event.path)

    @on(DirectoryTree.FileSelected, "#path-picker")
    def select_file(self, event: DirectoryTree.FileSelected) -> None:
        if not self.select_files:
            self.notify("Select a directory, not a file", severity="warning")
            return
        self.dismiss(event.path)


class HomeScreen(Screen[None]):
    """Landing screen shown when no project has been opened."""

    BINDINGS = [
        ("up", "previous_option", "Previous option"),
        ("down", "next_option", "Next option"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, message: Optional[str] = None):
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        yield TuiHeader(id="app-header")
        with Horizontal(classes="launcher-layout"):
            with Vertical(id="home"):
                yield Static(
                    "[bold]Welcome to resistics[/bold]\nOpen or create a project."
                )
                if self.message:
                    yield Static(self.message, id="home-message")
                yield Button("Open project", id="open-project", variant="success")
                yield Button("Create project", id="create-project")
                yield Button("Quit", id="quit")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#open-project", Button).focus()

    @on(Button.Pressed, "#open-project")
    def choose_project(self) -> None:
        self.app.push_screen(
            DirectoryPickerScreen("Select an existing resistics project", False),
            self._open_project,
        )

    def _open_project(self, project_path: Optional[Path]) -> None:
        if project_path is not None:
            self.app.open_project_path(project_path)

    @on(Button.Pressed, "#create-project")
    def create_project(self) -> None:
        self.app.show_create_project()

    @on(Button.Pressed, "#quit")
    def quit(self) -> None:
        self.app.exit()

    def action_quit(self) -> None:
        self.app.exit()

    def action_next_option(self) -> None:
        self._focus_option(1)

    def action_previous_option(self) -> None:
        self._focus_option(-1)

    def _focus_option(self, increment: int) -> None:
        buttons = list(self.query(Button))
        try:
            index = buttons.index(self.focused)
        except ValueError:
            index = 0
        buttons[(index + increment) % len(buttons)].focus()


class CreateProjectScreen(Screen[None]):
    """Create a new MTH5-backed project through the terminal UI."""

    BINDINGS = [
        ("up", "previous_option", "Previous option"),
        ("down", "next_option", "Next option"),
        ("left", "previous_action", "Previous action"),
        ("right", "next_action", "Next action"),
        ("escape", "home", "Back"),
    ]

    def __init__(self):
        super().__init__()
        self.parent_path: Optional[Path] = None
        self.mth5_path: Optional[Path] = None

    def compose(self) -> ComposeResult:
        yield TuiHeader(id="app-header")
        with Horizontal(classes="launcher-layout"):
            with Vertical(id="create-project-form"):
                yield Static("[bold]Create project[/bold]")
                yield Static("Parent directory")
                yield Button("Choose parent directory", id="choose-parent")
                yield Static("Not selected", id="parent-path")
                yield Static("Project folder name")
                yield Input(placeholder="my_project", id="project-name")
                yield Static("MTH5 file")
                yield Button("Choose MTH5 file", id="choose-mth5")
                yield Static("Not selected", id="mth5-path")
                yield Static("Reference time")
                yield Input(placeholder="YYYY-MM-DD HH:MM:SS", id="reference-time")
                yield Static("", id="create-status")
                with Horizontal(id="create-actions"):
                    yield Button("Back", id="back")
                    yield Button("Create and open", id="create", variant="success")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#choose-parent", Button).focus()

    @on(Button.Pressed, "#choose-parent")
    def choose_parent(self) -> None:
        self.app.push_screen(
            DirectoryPickerScreen("Select the parent directory", False),
            self._parent_selected,
        )

    def _parent_selected(self, parent_path: Optional[Path]) -> None:
        if parent_path is None:
            return
        self.parent_path = parent_path
        self.query_one("#parent-path", Static).update(str(parent_path))
        self._set_status("")

    @on(Button.Pressed, "#choose-mth5")
    def choose_mth5(self) -> None:
        self.app.push_screen(
            DirectoryPickerScreen("Select an MTH5 file", True), self._mth5_selected
        )

    def _mth5_selected(self, mth5_path: Optional[Path]) -> None:
        if mth5_path is None:
            return
        try:
            source = open_mth5(mth5_path)
            try:
                summary = source.file_summary()
            finally:
                source.close_mth5()
        except Exception as exc:
            self._set_status(f"[red]Unable to read MTH5 file:[/] {exc}")
            return
        self.mth5_path = mth5_path
        self.query_one("#mth5-path", Static).update(str(mth5_path))
        self.query_one("#reference-time", Input).value = summary.start_time or ""
        if summary.start_time is None:
            self._set_status(
                "[yellow]The MTH5 file has no recording start time; enter one manually.[/]"
            )
        else:
            self._set_status("")

    @on(Button.Pressed, "#create")
    def create(self) -> None:
        project_name = self.query_one("#project-name", Input).value.strip()
        reference_time = self.query_one("#reference-time", Input).value.strip()
        if self.parent_path is None:
            self._set_status("[red]Choose a parent directory.[/]")
            return
        if (
            not project_name
            or Path(project_name).name != project_name
            or project_name
            in {
                ".",
                "..",
            }
        ):
            self._set_status("[red]Enter a single new project folder name.[/]")
            return
        project_path = self.parent_path / project_name
        if project_path.exists():
            if (project_path / "resistics.json").exists():
                self._set_status(
                    "[yellow]This is already a resistics project. Use Open project instead.[/]"
                )
            else:
                self._set_status(
                    "[red]Choose a project folder name that does not exist.[/]"
                )
            return
        if self.mth5_path is None:
            self._set_status("[red]Choose an MTH5 file.[/]")
            return
        if not reference_time:
            self._set_status("[red]Enter a project reference time.[/]")
            return
        try:
            to_datetime(reference_time)
        except Exception as exc:
            self._set_status(f"[red]Invalid reference time:[/] {exc}")
            return
        try:
            init_project(project_path, self.mth5_path, reference_time)
            project = load(project_path)
        except Exception as exc:
            self._set_status(f"[red]Unable to create project:[/] {exc}")
            return
        self.app.open_project(project)

    @on(Button.Pressed, "#back")
    def back(self) -> None:
        self.app.show_home()

    def action_home(self) -> None:
        self.app.show_home()

    def action_next_option(self) -> None:
        self._focus_option(1)

    def action_previous_option(self) -> None:
        self._focus_option(-1)

    def action_next_action(self) -> None:
        self._focus_action(1)

    def action_previous_action(self) -> None:
        self._focus_action(-1)

    def _focus_option(self, increment: int) -> None:
        controls = [
            self.query_one("#choose-parent", Button),
            self.query_one("#project-name", Input),
            self.query_one("#choose-mth5", Button),
            self.query_one("#reference-time", Input),
            self.query_one("#back", Button),
            self.query_one("#create", Button),
        ]
        try:
            index = controls.index(self.focused)
        except ValueError:
            index = 0
        controls[(index + increment) % len(controls)].focus()

    def _focus_action(self, increment: int) -> None:
        actions = [
            self.query_one("#back", Button),
            self.query_one("#create", Button),
        ]
        try:
            index = actions.index(self.focused)
        except ValueError:
            return
        actions[(index + increment) % len(actions)].focus()

    def _set_status(self, message: str) -> None:
        self.query_one("#create-status", Static).update(message)


class ProjectExplorerScreen(Screen[None]):
    """Project browser and YAML editor with managed processing-job execution."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("d", "restore_defaults", "Restore defaults"),
        ("n", "create_job", "New job"),
        ("e", "edit_yaml", "Edit YAML"),
        ("y", "copy_yaml", "Copy YAML"),
        ("delete", "delete_yaml", "Delete YAML"),
        ("ctrl+s", "save_yaml", "Save YAML"),
        Binding("escape", "discard_yaml", "Discard YAML", priority=True),
        ("j", "run_selected_job", "Run job"),
        ("c", "cancel_job", "Cancel job"),
        ("x", "close_project", "Close project"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, project: Project, startup_warnings: Optional[list[str]] = None):
        super().__init__()
        self.project = project
        self.project_jobs = ProjectJobs(project)
        self.flow_paths: Dict[str, Path] = {}
        self.parameter_paths: Dict[str, Path] = {}
        self.criteria_paths: Dict[str, Path] = {}
        self.selected_flow_path: Optional[Path] = None
        self.selected_parameter_path: Optional[Path] = None
        self.selected_criteria_path: Optional[Path] = None
        self.job_summaries: Dict[str, JobSummary] = {}
        self.selected_job_path: Optional[Path] = None
        self.selected_validation: Optional[JobValidation] = None
        self.job_runner: Optional[JobRunner] = None
        self.job_state: Optional[JobState] = None
        self.editing_yaml = False
        self.editing_path: Optional[Path] = None
        self.editing_model = None
        self.editing_editor_id: Optional[str] = None
        self.startup_warnings = startup_warnings or []

    def compose(self) -> ComposeResult:
        yield TuiHeader(id="app-header")
        with TabbedContent(initial="overview"):
            with TabPane("Overview", id="overview"):
                with Vertical(classes="pane"):
                    yield Static(id="overview-content")
            with TabPane("Project", id="project"):
                with Horizontal(classes="pane split"):
                    with Vertical(classes="left"):
                        yield Tree("Project", id="project-tree")
                    with Vertical(classes="right"):
                        yield TextArea.code_editor(
                            '{\n  "message": "Select an item"\n}',
                            language="json",
                            theme="vscode_dark",
                            read_only=True,
                            id="metadata-content",
                        )
            with TabPane("Flows", id="flows"):
                with Vertical(classes="pane"):
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
            with TabPane("Jobs", id="jobs"):
                with Vertical(classes="pane"):
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
        self._populate_overview()
        self._populate_tree()
        self._populate_flows()
        self._populate_parameters()
        self._populate_criteria()
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
        self.refresh_bindings()
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
            self.query_one("#job-content", TextArea).text = (
                "No YAML jobs found in processing/jobs"
            )

    @staticmethod
    def _yaml_paths(directory: Path) -> list[Path]:
        """Return project YAML files in a stable display order."""
        return sorted(
            path
            for pattern in ("*.yaml", "*.yml")
            for path in directory.glob(pattern)
            if path.is_file()
        )

    def _job_resource_options(
        self, resource_type: str, model_type
    ) -> list[tuple[str, str]]:
        """Return valid resource files as dropdown labels and exact filenames."""
        directory = self.project.project_path / "processing" / resource_type
        options = []
        for path in self._yaml_paths(directory):
            try:
                model = model_from_yaml_file(model_type, path)
            except Exception as exc:
                logger.debug(f"Unable to use {resource_type} resource {path}: {exc}")
                continue
            display_name = getattr(model, "name", None) or path.stem
            options.append((f"{display_name} ({path.name})", path.name))
        return options

    def action_create_job(self) -> None:
        """Open the Jobs-tab form for a new editable job template."""
        if self.job_state == JobState.running:
            self.notify(
                "Job creation is unavailable while a job is running", severity="warning"
            )
            return
        flow_options = self._job_resource_options("flows", FlowDefinition)
        parameter_options = self._job_resource_options("parameters", ParameterSet)
        criteria_options = self._job_resource_options("criteria", GatherCriteria)
        existing_names = {
            path.stem for path in self._yaml_paths(self.project_jobs.jobs_path)
        }
        self.app.push_screen(
            CreateJobScreen(
                flow_options, parameter_options, criteria_options, existing_names
            ),
            self._job_template_created,
        )

    def _job_template_created(self, definition: Optional[JobDefinition]) -> None:
        """Persist a completed creation form and present the generated YAML."""
        if definition is None:
            return
        try:
            path = self.project_jobs.create_template(definition)
        except Exception as exc:
            self.notify(f"Unable to create job: {exc}", severity="error")
            return
        self._populate_jobs()
        self.selected_job_path = path
        self.selected_validation = self.project_jobs.validate(path)
        self._show_yaml("#job-content", path)
        self.notify(f"Created {path.name}")

    def _populate_flows(self) -> None:
        """Populate the read-only flow browser."""
        table = self.query_one("#flow-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Flow", "ID", "Version", "Nodes", "Status")
        self.flow_paths.clear()
        for path in self._yaml_paths(
            self.project.project_path / "processing" / "flows"
        ):
            key = str(path)
            self.flow_paths[key] = path
            try:
                flow = model_from_yaml_file(FlowDefinition, path)
                n_nodes = sum(len(stage.nodes) for stage in flow.flow_stages())
                table.add_row(
                    flow.name,
                    flow.id,
                    flow.version,
                    str(n_nodes),
                    "[green]valid[/green]",
                    key=key,
                )
            except Exception as exc:
                table.add_row(
                    path.stem,
                    "-",
                    "-",
                    "-",
                    "[red]invalid[/red]",
                    key=key,
                )
                logger.debug(f"Unable to read flow {path}: {exc}")
        if not self.flow_paths:
            self.query_one("#flow-content", TextArea).text = (
                "No YAML flows found in processing/flows"
            )

    def _populate_parameters(self) -> None:
        """Populate the read-only parameter-set browser."""
        table = self.query_one("#parameter-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Parameters", "Processes", "Status")
        self.parameter_paths.clear()
        directory = self.project.project_path / "processing" / "parameters"
        for path in self._yaml_paths(directory):
            key = str(path)
            self.parameter_paths[key] = path
            try:
                parameters = model_from_yaml_file(ParameterSet, path)
                table.add_row(
                    parameters.name,
                    str(len(parameters.processes)),
                    "[green]valid[/green]",
                    key=key,
                )
            except Exception as exc:
                table.add_row(
                    path.stem,
                    "-",
                    "[red]invalid[/red]",
                    key=key,
                )
                logger.debug(f"Unable to read parameter set {path}: {exc}")
        if not self.parameter_paths:
            self.query_one("#parameter-content", TextArea).text = (
                "No YAML parameter sets found in processing/parameters"
            )

    def _populate_criteria(self) -> None:
        """Populate the read-only criteria browser."""
        table = self.query_one("#criteria-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Criteria", "Remote references", "Status")
        self.criteria_paths.clear()
        directory = self.project.project_path / "processing" / "criteria"
        for path in self._yaml_paths(directory):
            key = str(path)
            self.criteria_paths[key] = path
            try:
                criteria = model_from_yaml_file(GatherCriteria, path)
                table.add_row(
                    path.stem,
                    str(len(criteria.remote_references)),
                    "[green]valid[/green]",
                    key=key,
                )
            except Exception as exc:
                table.add_row(path.stem, "-", "[red]invalid[/red]", key=key)
                logger.debug(f"Unable to read criteria {path}: {exc}")
        if not self.criteria_paths:
            self.query_one("#criteria-content", TextArea).text = (
                "No YAML criteria files found in processing/criteria"
            )

    @on(Tree.NodeSelected, "#project-tree")
    def show_metadata(self, event: Tree.NodeSelected) -> None:
        object_path = event.node.data
        details = self.query_one("#metadata-content", TextArea)
        if object_path is None:
            details.text = json.dumps(
                {"message": "Select a survey, station, run, or channel"}, indent=2
            )
            return
        try:
            metadata = self.project.get_metadata(str(object_path))
            details.text = metadata.model_dump_json(indent=2)
        except Exception as exc:
            details.text = json.dumps({"error": str(exc)}, indent=2)

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
        self.selected_validation = self.project_jobs.validate(summary.path)
        validation = self.selected_validation
        self._show_yaml("#job-content", summary.path)
        if validation.ok:
            self.notify("Job YAML is valid")
        else:
            self.notify("; ".join(validation.errors), severity="warning")
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
        try:
            model_from_yaml_file(FlowDefinition, path)
        except Exception as exc:
            self.notify(f"Invalid flow YAML: {exc}", severity="warning")
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
        try:
            model_from_yaml_file(ParameterSet, path)
        except Exception as exc:
            self.notify(f"Invalid parameter YAML: {exc}", severity="warning")
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
        try:
            model_from_yaml_file(GatherCriteria, path)
        except Exception as exc:
            self.notify(f"Invalid criteria YAML: {exc}", severity="warning")
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

    def _selected_yaml_file(self) -> Optional[tuple[Path, str]]:
        """Return the selected YAML source and its editor selector."""
        target = self._yaml_edit_target()
        if target is None:
            return None
        path, _, editor_id = target
        return path, editor_id

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
        """Prompt for a new filename and copy the selected YAML source verbatim."""
        selected = self._selected_yaml_file()
        if selected is None:
            self.notify("Select a YAML file first", severity="warning")
            return
        source, editor_id = selected
        self.app.push_screen(
            CopyYamlFileScreen(source),
            lambda name: self._yaml_file_copied(source, editor_id, name),
        )

    def _yaml_file_copied(
        self, source: Path, editor_id: str, name: Optional[str]
    ) -> None:
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
        """Confirm deletion of the currently selected YAML source."""
        selected = self._selected_yaml_file()
        if selected is None:
            self.notify("Select a YAML file first", severity="warning")
            return
        source, editor_id = selected
        self.app.push_screen(
            DeleteYamlFileScreen(source),
            lambda confirmed: self._yaml_file_deleted(source, editor_id, confirmed),
        )

    def _yaml_file_deleted(
        self, source: Path, editor_id: str, confirmed: bool
    ) -> None:
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
            self.selected_validation = self.project_jobs.validate(path)
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
        if self.job_state == JobState.running:
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
        self._refresh_yaml_resource(editor_id)
        self._show_yaml(editor_id, saved_path)
        if editor_id == "#job-content":
            self.selected_job_path = saved_path
            self.selected_validation = self.project_jobs.validate(saved_path)
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
        self.refresh_bindings()

    def _refresh_yaml_resource(self, editor_id: str) -> None:
        """Refresh the table associated with a saved YAML resource."""
        if editor_id == "#flow-content":
            self._populate_flows()
        elif editor_id == "#parameter-content":
            self._populate_parameters()
        elif editor_id == "#criteria-content":
            self._populate_criteria()
        elif editor_id == "#job-content":
            self._populate_jobs()

    def action_run_selected_job(self) -> None:
        """Confirm and run the valid job selected in the Jobs tab."""
        validation = self.selected_validation
        if validation is None or not validation.ok:
            self.notify("Select a valid job first", severity="warning")
            return
        self.app.push_screen(ConfirmJobScreen(validation), self._submission_confirmed)

    def _restore_flows(self) -> None:
        """Restore only missing built-in flow templates."""
        installed = install_builtin_flow_templates(self.project.project_path)
        self._populate_flows()
        if installed:
            self.notify(f"Restored {len(installed)} flow template(s)")
        else:
            self.notify("All built-in flow templates are already present")

    def _restore_parameters(self) -> None:
        """Restore only missing built-in parameter-set templates."""
        installed = install_builtin_parameter_templates(self.project.project_path)
        self._populate_parameters()
        if installed:
            self.notify(f"Restored {len(installed)} parameter-set template(s)")
        else:
            self.notify("All built-in parameter-set templates are already present")

    def _restore_criteria(self) -> None:
        """Restore only missing criteria examples."""
        installed = install_builtin_criteria_templates(self.project.project_path)
        self._populate_criteria()
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
        self.refresh_bindings()
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
        self.refresh_bindings()

    @on(TabbedContent.TabActivated)
    def refresh_tab_bindings(self) -> None:
        """Refresh the Footer when the active tab changes."""
        self.refresh_bindings()

    def check_action(self, action: str, parameters: tuple[object, ...]):
        """Expose only Footer actions relevant to the active tab and job state."""
        active = self.query_one(TabbedContent).active
        if action == "edit_yaml":
            return (
                not self.editing_yaml
                and self.job_state != JobState.running
                and self._yaml_edit_target() is not None
            )
        if action == "create_job":
            return (
                active == "jobs"
                and not self.editing_yaml
                and self.job_state != JobState.running
            )
        if action in {"copy_yaml", "delete_yaml"}:
            return (
                not self.editing_yaml
                and self.job_state != JobState.running
                and self._selected_yaml_file() is not None
            )
        if action in {"save_yaml", "discard_yaml"}:
            return self.editing_yaml
        if action == "close_project":
            return self.job_state != JobState.running and not self.editing_yaml
        if action == "restore_defaults":
            return not self.editing_yaml and active in {
                "flows",
                "parameters",
                "criteria",
            }
        if action == "run_selected_job":
            if active != "jobs":
                return False
            return (
                True
                if self.selected_validation is not None and self.selected_validation.ok
                else None
            )
        if action == "cancel_job":
            return active == "activity" and self.job_state == JobState.running
        return super().check_action(action, parameters)

    def action_refresh(self) -> None:
        if self.job_state == JobState.running:
            self.notify("Refresh is unavailable while a job is running")
            return
        if self.editing_yaml:
            self.notify(
                "Save or discard the current YAML edits first", severity="warning"
            )
            return
        self._populate_overview()
        self._populate_tree()
        self._populate_flows()
        self._populate_parameters()
        self._populate_criteria()
        self._populate_jobs()
        self.notify("Project refreshed")

    def action_cancel_job(self) -> None:
        if self.job_runner is None or self.job_state != JobState.running:
            self.notify("No active job")
            return
        self.job_runner.cancel()
        self.notify("Cancellation requested; the current step will finish first")

    def action_close_project(self) -> None:
        if self.job_state == JobState.running:
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
        self.app.show_home()

    def action_quit(self) -> None:
        if self.job_state == JobState.running:
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
    #project-tree, #flow-table, #parameter-table, #criteria-table, #job-table {
        height: 1fr;
    }
    #metadata-content, #flow-content, #parameter-content, #criteria-content,
    #job-content {
        height: 1fr;
        background: #202020;
        color: #f7f4f2;
        border: tall #343434;
    }
    #metadata-content:focus, #flow-content:focus, #parameter-content:focus,
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

    def __init__(self, project_path: Optional[Path] = None):
        super().__init__()
        self.initial_project_path = project_path
        self._has_started_screen = False

    def on_mount(self) -> None:
        if self.initial_project_path is None:
            self.show_home()
        else:
            self.open_project_path(self.initial_project_path)

    def show_home(self, message: Optional[str] = None) -> None:
        self.title = "resistics"
        self.sub_title = "project launcher"
        self._show_screen(HomeScreen(message))

    def show_create_project(self) -> None:
        self.title = "resistics"
        self.sub_title = "create project"
        self._show_screen(CreateProjectScreen())

    def open_project_path(self, project_path: Path) -> None:
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                project = load(project_path)
        except Exception as exc:
            self.show_home(f"[red]Unable to open project:[/] {exc}")
            return
        self.open_project(
            project, [str(warning.message) for warning in caught_warnings]
        )

    def open_project(
        self, project: Project, startup_warnings: Optional[list[str]] = None
    ) -> None:
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


def run_tui(project_path: Optional[Path] = None) -> None:
    """Run the resistics terminal application."""
    logger.remove()
    try:
        ResisticsTui(project_path).run()
    finally:
        logger.remove()
        logger.add(sys.stderr, level="INFO")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Launch the TUI, optionally opening one project path immediately."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) > 1:
        print("Usage: resistics [PROJECT_PATH]", file=sys.stderr)
        return 2
    project_path = Path(arguments[0]) if arguments else None
    run_tui(project_path)
    return 0
