"""Smoke tests for the project terminal UI."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

from textual.containers import VerticalScroll
from textual.widgets import Button, DataTable, Input, Static, TabbedContent, Tree

from resistics.tui import CreateProjectScreen, DirectoryPickerScreen, ResisticsTui


class FakeProject:
    """Minimal project contract used by the interface smoke test."""

    def __init__(self, project_path, create_jobs=True):
        self.project_path = project_path
        self.ref_time = "2020-01-01T00:00:00"
        self.runs = []
        self.closed = False
        if create_jobs:
            (project_path / "processing/jobs").mkdir(parents=True)

    def file_summary(self):
        return SimpleNamespace(
            mth5_path=self.project_path / "data.h5",
            file_version="0.2.0",
            start_time=None,
            end_time=None,
            sample_rates=[],
            n_surveys=0,
            n_stations=0,
            n_runs=0,
            n_channels=0,
        )

    def list_surveys(self):
        return []

    def list_stations(self, survey=None):
        return []

    def list_runs(self, survey=None, station=None):
        return []

    def close_mth5(self):
        self.closed = True


def test_tui_mounts_project_views(monkeypatch, tmp_path):
    project = FakeProject(tmp_path / "project")
    monkeypatch.setattr("resistics.tui.load", lambda project_path: project)
    app = ResisticsTui(project.project_path)

    async def run_test():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            overview = app.screen.query_one("#overview-content", Static)
            tree = app.screen.query_one("#project-tree", Tree)
            table = app.screen.query_one("#job-table", DataTable)
            metadata_details = app.screen.query_one("#metadata-details", VerticalScroll)
            assert "project" in str(overview.render())
            assert str(tree.root.label) == "project"
            assert table.row_count == 0
            assert app.sub_title == str(project.project_path)
            app.screen.query_one(TabbedContent).active = "project"
            await asyncio.sleep(0)
            app.screen.set_focus(tree)
            app.action_focus_next()
            assert app.focused is metadata_details

    asyncio.run(run_test())
    assert project.closed


def test_close_project_returns_to_home(monkeypatch, tmp_path):
    project = FakeProject(tmp_path / "project")
    monkeypatch.setattr("resistics.tui.load", lambda project_path: project)
    app = ResisticsTui(project.project_path)

    async def run_test():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            assert app.screen.query_one("#close-project", Button)
            await pilot.press("x")
            await pilot.pause()
            assert app.sub_title == "project launcher"
            assert app.screen.query_one("#open-project", Button)

    asyncio.run(run_test())
    assert project.closed


def test_tui_starts_on_the_project_home_screen():
    app = ResisticsTui()

    async def run_test():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            assert app.sub_title == "project launcher"
            assert app.screen.query_one("#open-project", Button).label == "Open project"
            assert (
                app.screen.query_one("#create-project", Button).label
                == "Create project"
            )
            await pilot.press("down")
            assert app.focused is app.screen.query_one("#create-project", Button)
            await pilot.press("down")
            assert app.focused is app.screen.query_one("#quit", Button)
            await pilot.press("up")
            assert app.focused is app.screen.query_one("#create-project", Button)

    asyncio.run(run_test())


def test_directory_picker_starts_at_home_with_parent_navigation():
    picker = DirectoryPickerScreen("Select a project", False)
    assert picker.start_path == Path.home()
    assert ("u", "parent_directory", "Up") in picker.BINDINGS
    assert "Space: expand/collapse" in picker.navigation_instruction
    assert "project folder" in picker.selection_instruction
    assert "MTH5 file" in DirectoryPickerScreen(
        "Select an MTH5 file", True
    ).selection_instruction


def test_invalid_startup_project_returns_to_home(monkeypatch, tmp_path):
    def fail(project_path):
        raise ValueError("not a resistics project")

    monkeypatch.setattr("resistics.tui.load", fail)
    app = ResisticsTui(tmp_path / "missing")

    async def run_test():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            assert app.screen.query_one("#open-project", Button)
            message = app.screen.query_one("#home-message", Static)
            assert "Unable to open project" in str(message.render())

    asyncio.run(run_test())


def test_create_project_opens_the_new_project(monkeypatch, tmp_path):
    created = []
    parent_path = tmp_path / "projects"
    parent_path.mkdir()
    mth5_path = tmp_path / "data.h5"
    mth5_path.write_text("")
    project_path = parent_path / "new_project"
    project = FakeProject(project_path, create_jobs=False)

    class FakeMTH5Source:
        closed = False

        def file_summary(self):
            return SimpleNamespace(start_time="2020-01-01T00:00:00")

        def close_mth5(self):
            self.closed = True

    source = FakeMTH5Source()

    def initialise(path, selected_mth5_path, reference_time):
        created.append((path, selected_mth5_path, reference_time))
        (path / "processing/jobs").mkdir(parents=True)

    monkeypatch.setattr("resistics.tui.open_mth5", lambda path: source)
    monkeypatch.setattr("resistics.tui.init_project", initialise)
    monkeypatch.setattr("resistics.tui.load", lambda path: project)
    app = ResisticsTui()

    async def run_test():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app.show_create_project()
            await pilot.pause()
            form = app.screen
            assert isinstance(form, CreateProjectScreen)
            assert app.focused is form.query_one("#choose-parent", Button)
            await pilot.press("down")
            assert app.focused is form.query_one("#project-name", Input)
            await pilot.press("down")
            assert app.focused is form.query_one("#choose-mth5", Button)
            await pilot.press("up")
            assert app.focused is form.query_one("#project-name", Input)
            form.query_one("#back", Button).focus()
            await pilot.press("right")
            assert app.focused is form.query_one("#create", Button)
            await pilot.press("left")
            assert app.focused is form.query_one("#back", Button)
            form._parent_selected(parent_path)
            form.query_one("#project-name", Input).value = "new_project"
            form._mth5_selected(mth5_path)
            assert (
                form.query_one("#reference-time", Input).value == "2020-01-01T00:00:00"
            )
            form.create()
            await pilot.pause()
            assert app.sub_title == str(project_path)
            assert app.screen.query_one("#overview-content", Static)

    asyncio.run(run_test())
    assert source.closed
    assert created == [(project_path, mth5_path, "2020-01-01T00:00:00")]
    assert project.closed


def test_tui_uses_dark_surfaces_with_resistics_accents():
    """Keep the project explorer dark without losing the brand accents."""
    assert "background: #101010" in ResisticsTui.CSS
    assert "background: #202020" in ResisticsTui.CSS
    assert ".launcher-layout { height: 1fr; align-horizontal: center; }" in (
        ResisticsTui.CSS
    )
    assert "#job-details:focus, #metadata-details:focus { background: #343434; }" in (
        ResisticsTui.CSS
    )
    assert "#faa881" in ResisticsTui.CSS
    assert "#ac3600" in ResisticsTui.CSS
