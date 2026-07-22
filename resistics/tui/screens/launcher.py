"""Launcher screens used before a resistics project is opened."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, Static

from resistics.tui.screens.dialogs import DirectoryPickerScreen
from resistics.tui.services import _feature_error, _focus_relative, _resistics_app


class TuiHeader(Static):
    """Render the application title without Header's reactive mount timing."""

    def on_mount(self) -> None:
        title = f"[bold]{self.app.title}[/bold]"
        if self.app.sub_title:
            title += f" [dim]— {self.app.sub_title}[/]"
        self.update(title)


class HomeScreen(Screen[None]):
    """Landing screen shown when no project has been opened.

    :param message: Optional status or failure message displayed to the user.
    """

    BINDINGS = [
        ("up", "previous_option", "Previous option"),
        ("down", "next_option", "Next option"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, message: str | None = None):
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        yield TuiHeader(id="app-header")
        with Horizontal(classes="launcher-layout"), Vertical(id="home"):
            yield Static("[bold]Welcome to resistics[/bold]\nOpen or create a project.")
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

    def _open_project(self, project_path: Path | None) -> None:
        if project_path is not None:
            _resistics_app(self).open_project_path(project_path)

    @on(Button.Pressed, "#create-project")
    def create_project(self) -> None:
        _resistics_app(self).show_create_project()

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
        _focus_relative(buttons, self.focused, increment)


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
        self.parent_path: Path | None = None
        self.mth5_path: Path | None = None

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

    def _parent_selected(self, parent_path: Path | None) -> None:
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

    def _mth5_selected(self, mth5_path: Path | None) -> None:
        if mth5_path is None:
            return
        from resistics.project import open_mth5

        _resistics_app(self)._reinstall_diagnostic_capture()
        try:
            source = open_mth5(mth5_path)
            try:
                summary = source.file_summary()
            finally:
                source.close()
        except Exception as exc:
            self._set_status(
                f"[red]Unable to read MTH5 file:[/] "
                f"{_feature_error('MTH5 inspection', exc)}"
            )
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
        from resistics.project import init as init_project
        from resistics.sampling import to_datetime

        _resistics_app(self)._reinstall_diagnostic_capture()
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
        except Exception as exc:
            self._set_status(
                f"[red]Unable to create project:[/] "
                f"{_feature_error('Project creation', exc)}"
            )
            return
        _resistics_app(self).open_project_path(project_path)

    @on(Button.Pressed, "#back")
    def back(self) -> None:
        _resistics_app(self).show_home()

    def action_home(self) -> None:
        _resistics_app(self).show_home()

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
        _focus_relative(controls, self.focused, increment)

    def _focus_action(self, increment: int) -> None:
        actions = [
            self.query_one("#back", Button),
            self.query_one("#create", Button),
        ]
        _focus_relative(actions, self.focused, increment, move_from_unfocused=False)

    def _set_status(self, message: str) -> None:
        self.query_one("#create-status", Static).update(message)


class ProjectLoadingScreen(Screen[None]):
    """Immediate, cancellable surface shown while a project opens.

    :param project_path: Project directory being opened.

    **Attributes**

    - **BINDINGS** — Keyboard actions available while the project is opening.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        ("escape", "cancel", "Cancel"),
        ("x", "cancel", "Cancel"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, project_path: Path):
        super().__init__()
        self.project_path = project_path

    def compose(self) -> ComposeResult:
        yield TuiHeader(id="app-header")
        with VerticalScroll(classes="pane"):
            yield Static(
                f"[bold]Opening project[/bold]\n\n{self.project_path}\n\n"
                "Loading MTH5 metadata…",
                id="project-loading",
            )
        yield Footer()

    def action_cancel(self) -> None:
        """Reject the pending worker result and return to the launcher."""
        _resistics_app(self).cancel_project_open()

    def action_quit(self) -> None:
        self.app.exit()
