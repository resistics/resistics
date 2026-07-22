"""Modal dialogs used by the resistics terminal application."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Input, Select, Static

from resistics.tui.services import _focus_relative
from resistics.tui.state import ProjectDataDeletionRequest

if TYPE_CHECKING:
    from resistics.job import JobDefinition, JobValidation  # noqa: F401
    from resistics.project import ProjectDataDeletion


class ConfirmJobScreen(ModalScreen[bool]):
    """Confirm submission of an already validated job.

    :param validation: Successful validation containing the resolved job.
    """

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
        _focus_relative(actions, self.focused, increment)


_NO_CRITERIA_VALUE = "__no_criteria__"


_YAML_FILE_STEM = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class CreateJobScreen(ModalScreen["JobDefinition | None"]):
    """Create a minimal job template from project YAML resources.

    :param flow_options: Display names and stored values for available flows.
    :param parameter_options: Display names and stored values for parameter sets.
    :param criteria_options: Display names and stored values for gather criteria.
    :param existing_names: Job names that cannot be reused.
    """

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
        self.criteria_options = [
            ("No criteria", _NO_CRITERIA_VALUE),
            *list(criteria_options),
        ]
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
            yield Static("Output label")
            yield Input(value="default", placeholder="default", id="job-output-label")
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
                yield Button(
                    "Cancel", id="cancel-job-template", classes="dialog-action"
                )
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
        from resistics.common import validate_output_label
        from resistics.job import JobDefinition, validate_job_template_name

        name = self.query_one("#job-name", Input).value
        output_label = self.query_one("#job-output-label", Input).value
        try:
            name = validate_job_template_name(name)
        except ValueError as exc:
            self._set_status(str(exc))
            return
        try:
            output_label = validate_output_label(output_label)
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
                output_label=output_label,
            )
        )

    def _select_value(self, selector: str) -> str | None:
        value = self.query_one(selector, Select).value
        return None if value is Select.NULL else str(value)

    def _set_status(self, message: str) -> None:
        self.query_one("#create-job-status", Static).update(message)

    def _focus_action(self, increment: int) -> None:
        actions = [
            self.query_one("#cancel-job-template", Button),
            self.query_one("#create-job-template", Button),
        ]
        _focus_relative(actions, self.focused, increment, move_from_unfocused=False)


class CopyYamlFileScreen(ModalScreen[str | None]):
    """Ask for the filename stem of a YAML copy.

    :param source: YAML resource being copied.
    """

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
            self.dismiss(
                _validate_yaml_file_stem(self.query_one("#copy-yaml-name", Input).value)
            )
        except ValueError as exc:
            self.query_one("#copy-yaml-status", Static).update(str(exc))

    def _focus_action(self, increment: int) -> None:
        actions = [
            self.query_one("#cancel-copy-yaml", Button),
            self.query_one("#confirm-copy-yaml", Button),
        ]
        _focus_relative(actions, self.focused, increment)


class DeleteYamlFileScreen(ModalScreen[bool]):
    """Require explicit confirmation before deleting a project YAML file.

    :param source: YAML resource proposed for deletion.
    """

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
                yield Button("Cancel", id="cancel-delete-yaml", classes="dialog-action")
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
        _focus_relative(actions, self.focused, increment)


class DeleteProjectDataScreen(ModalScreen[ProjectDataDeletionRequest | None]):
    """Choose a derived-data namespace or all generated project data.

    :param labels: Existing output-label namespaces available for deletion.
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        Binding("left", "previous_action", "Previous action", priority=True),
        Binding("right", "next_action", "Next action", priority=True),
    ]

    CSS = """
    DeleteProjectDataScreen { align: center middle; background: transparent; }
    #delete-data-dialog {
        width: 72;
        height: auto;
        padding: 1 2;
        border: round #ac3600;
        background: #202020;
        color: #f7f4f2;
    }
    #delete-data-dialog Select { margin-top: 1; }
    #delete-data-actions { height: auto; align-horizontal: right; margin-top: 1; }
    #delete-data-actions Button { margin-left: 1; }
    """

    def __init__(self, labels: Sequence[str]):
        super().__init__()
        self.labels = list(labels)

    def compose(self) -> ComposeResult:
        options = [(label, label) for label in self.labels]
        value = self.labels[0] if self.labels else Select.NULL
        with Vertical(id="delete-data-dialog"):
            yield Static("[bold]Delete derived Project data[/bold]")
            yield Static("Output label")
            yield Select(
                options,
                prompt="No labelled data available",
                allow_blank=not self.labels,
                value=value,
                id="delete-data-label",
            )
            yield Static("The MTH5 file and project setup are never deleted.")
            with Horizontal(id="delete-data-actions"):
                yield Button("Cancel", id="cancel-delete-data", classes="dialog-action")
                yield Button(
                    "Delete label",
                    id="delete-data-label-action",
                    variant="error",
                    classes="dialog-action",
                    disabled=not self.labels,
                )
                yield Button(
                    "Delete all Project data",
                    id="delete-all-data-action",
                    variant="error",
                    classes="dialog-action",
                )

    def on_mount(self) -> None:
        self.query_one("#cancel-delete-data", Button).focus()

    @on(Button.Pressed, "#cancel-delete-data")
    def cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#delete-data-label-action")
    def delete_label(self) -> None:
        value = self.query_one("#delete-data-label", Select).value
        if value is not Select.NULL:
            self.dismiss(ProjectDataDeletionRequest(output_label=str(value)))

    @on(Button.Pressed, "#delete-all-data-action")
    def delete_all(self) -> None:
        self.dismiss(ProjectDataDeletionRequest())

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_next_action(self) -> None:
        self._focus_action(1)

    def action_previous_action(self) -> None:
        self._focus_action(-1)

    def _focus_action(self, increment: int) -> None:
        actions = list(self.query(".dialog-action"))
        enabled = [action for action in actions if not action.disabled]
        _focus_relative(enabled, self.focused, increment)


class ConfirmProjectDataDeletionScreen(ModalScreen[bool]):
    """Require explicit confirmation before deleting generated project data.

    :param deletion: Exact deletion preview presented for confirmation.
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        Binding("left", "previous_action", "Previous action", priority=True),
        Binding("right", "next_action", "Next action", priority=True),
    ]

    CSS = """
    ConfirmProjectDataDeletionScreen { align: center middle; background: transparent; }
    #confirm-delete-data-dialog {
        width: 72;
        height: auto;
        padding: 1 2;
        border: round #ac3600;
        background: #202020;
        color: #f7f4f2;
    }
    #confirm-delete-data-actions { height: auto; align-horizontal: right; margin-top: 1; }
    #confirm-delete-data-actions Button { margin-left: 1; }
    """

    def __init__(self, deletion: ProjectDataDeletion):
        super().__init__()
        self.deletion = deletion

    def compose(self) -> ComposeResult:
        if self.deletion.output_label is None:
            scope = "all derived Project data"
        else:
            scope = f"output label [bold]{self.deletion.output_label}[/bold]"
        with Vertical(id="confirm-delete-data-dialog"):
            yield Static(
                f"Delete {scope}?\n\n"
                f"{self.deletion.count} top-level path(s) will be removed. "
                "The MTH5 file and project setup are retained.\n\n"
                "This cannot be undone."
            )
            with Horizontal(id="confirm-delete-data-actions"):
                yield Button(
                    "Cancel", id="cancel-confirm-delete-data", classes="dialog-action"
                )
                yield Button(
                    "Delete data",
                    id="confirm-delete-data",
                    variant="error",
                    classes="dialog-action",
                )

    def on_mount(self) -> None:
        self.query_one("#cancel-confirm-delete-data", Button).focus()

    @on(Button.Pressed, "#cancel-confirm-delete-data")
    def cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#confirm-delete-data")
    def delete(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)

    def action_next_action(self) -> None:
        self._focus_action(1)

    def action_previous_action(self) -> None:
        self._focus_action(-1)

    def _focus_action(self, increment: int) -> None:
        actions = [
            self.query_one("#cancel-confirm-delete-data", Button),
            self.query_one("#confirm-delete-data", Button),
        ]
        _focus_relative(actions, self.focused, increment)


def _validate_yaml_file_stem(value: str) -> str:
    """Validate a filename stem used when copying a project YAML file.

    :param value: Value to validate or normalize.
    :return: The value produced when this operation completes.
    :raises ValueError: If the requested operation cannot satisfy its contract.
    """
    value = value.strip()
    if not _YAML_FILE_STEM.fullmatch(value):
        raise ValueError(
            "New name must use letters, numbers, dots, underscores, or hyphens"
        )
    if Path(value).suffix in {".yaml", ".yml"}:
        raise ValueError("New name must not include a YAML extension")
    return value


class DirectoryPickerScreen(ModalScreen[Path | None]):
    """Select either a directory or file with the terminal file browser.

    :param title: Title displayed above the file browser.
    :param select_files: Whether files, rather than directories, may be selected.
    :param start_path: Initial directory, or the current working directory when omitted.
    """

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

    def __init__(self, title: str, select_files: bool, start_path: Path | None = None):
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
        tree_path = Path(tree.path)
        parent_path = tree_path.parent
        if parent_path == tree_path:
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
