"""Smoke tests for the project terminal UI."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
    Tree,
)

from resistics.flow import default_parameter_set, model_to_yaml, standard_mt_flow
from resistics.gather import GatherCriteria
from resistics.tui import (
    CopyYamlFileScreen,
    CreateJobScreen,
    CreateProjectScreen,
    DeleteYamlFileScreen,
    DirectoryPickerScreen,
    ResisticsTui,
)


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
    flow = standard_mt_flow()
    flow_path = project.project_path / "processing/flows/standard_mt.yaml"
    flow_path.parent.mkdir(parents=True)
    flow_path.write_text(model_to_yaml(flow))
    parameters_path = project.project_path / "processing/parameters/default_mt.yaml"
    parameters_path.parent.mkdir(parents=True)
    parameters_path.write_text(model_to_yaml(default_parameter_set()))
    monkeypatch.setattr("resistics.tui.load", lambda project_path: project)
    app = ResisticsTui(project.project_path)

    async def run_test():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            overview = app.screen.query_one("#overview-content", Static)
            tree = app.screen.query_one("#project-tree", Tree)
            table = app.screen.query_one("#job-table", DataTable)
            flow_table = app.screen.query_one("#flow-table", DataTable)
            parameter_table = app.screen.query_one("#parameter-table", DataTable)
            metadata_details = app.screen.query_one("#metadata-content", TextArea)
            assert "project" in str(overview.render())
            assert str(tree.root.label) == "project"
            assert table.row_count == 0
            assert flow_table.row_count == 1
            assert flow_table.cell_padding == 1
            assert parameter_table.row_count == 1
            assert app.screen.query_one("#flows", TabPane)
            assert app.screen.query_one("#parameters", TabPane)
            assert app.screen.query_one("#criteria", TabPane)
            assert app.screen.query_one("#criteria-table", DataTable)
            flow_editor = app.screen.query_one("#flow-content", TextArea)
            metadata_editor = app.screen.query_one("#metadata-content", TextArea)
            assert metadata_editor.language == "json"
            assert metadata_editor.theme == "vscode_dark"
            assert metadata_editor.read_only
            assert metadata_editor.show_line_numbers
            assert flow_editor.language == "yaml"
            assert flow_editor.theme == "vscode_dark"
            assert flow_editor.read_only
            assert flow_editor.show_line_numbers
            app.screen.query_one(TabbedContent).active = "flows"
            await asyncio.sleep(0)
            assert "restore_defaults" in {
                binding.binding.action
                for binding in app.screen.active_bindings.values()
            }
            app.screen.selected_flow_path = flow_path
            app.screen._show_yaml("#flow-content", flow_path)
            await pilot.press("e")
            assert not flow_editor.read_only
            flow_editor.text = "# edited in the TUI\n" + flow_editor.text
            await pilot.press("ctrl+s")
            assert flow_editor.read_only
            assert flow_path.read_text().startswith("# edited in the TUI\n")
            app.screen.action_edit_yaml()
            flow_editor.text = "id: incomplete\n"
            app.screen.action_save_yaml()
            assert not flow_editor.read_only
            assert flow_path.read_text().startswith("# edited in the TUI\n")
            await pilot.press("escape")
            assert flow_editor.read_only
            assert not app.screen.editing_yaml
            app.screen.query_one(TabbedContent).active = "activity"
            await asyncio.sleep(0)
            assert "cancel_job" not in {
                binding.binding.action
                for binding in app.screen.active_bindings.values()
            }
            assert not app.screen.query("#install-defaults")
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
            assert "close_project" in {
                binding.binding.action
                for binding in app.screen.active_bindings.values()
            }
            await pilot.press("x")
            await pilot.pause()
            assert app.sub_title == "project launcher"
            assert app.screen.query_one("#open-project", Button)

    asyncio.run(run_test())
    assert project.closed


def test_tui_creates_a_job_template_from_dropdowns(monkeypatch, tmp_path):
    project = FakeProject(tmp_path / "project")
    project.table = pd.DataFrame(
        columns=["survey", "station", "sample_rate", "run_path"]
    )
    flow_path = project.project_path / "processing/flows/standard.yaml"
    flow_path.parent.mkdir(parents=True)
    flow_path.write_text(model_to_yaml(standard_mt_flow()))
    parameters_path = project.project_path / "processing/parameters/default.yaml"
    parameters_path.parent.mkdir(parents=True)
    parameters_path.write_text(model_to_yaml(default_parameter_set()))
    criteria_path = project.project_path / "processing/criteria/field.yaml"
    criteria_path.parent.mkdir(parents=True)
    criteria_path.write_text(model_to_yaml(GatherCriteria()))
    monkeypatch.setattr("resistics.tui.load", lambda project_path: project)
    app = ResisticsTui(project.project_path)

    async def run_test():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app.screen.query_one(TabbedContent).active = "jobs"
            await pilot.press("n")
            await pilot.pause()
            form = app.screen
            assert isinstance(form, CreateJobScreen)
            assert form.query_one("#job-flow", Select).value == "standard.yaml"
            assert form.query_one("#job-parameters", Select).value == "default.yaml"
            form.query_one("#cancel-job-template", Button).focus()
            await pilot.press("right")
            assert form.focused is form.query_one("#create-job-template", Button)
            await pilot.press("left")
            assert form.focused is form.query_one("#cancel-job-template", Button)
            form.query_one("#job-name", Input).value = "field_job"
            form.query_one("#job-criteria", Select).value = "field.yaml"
            form.create()
            await pilot.pause()
            assert app.screen.query_one("#job-table", DataTable).row_count == 1
            yaml_text = (
                project.project_path / "processing/jobs/field_job.yaml"
            ).read_text()
            assert "flow: standard.yaml" in yaml_text
            assert "parameters: default.yaml" in yaml_text
            assert "criteria: field.yaml" in yaml_text
            assert "scope:" in yaml_text

    asyncio.run(run_test())
    assert project.closed


def test_tui_copies_and_deletes_selected_yaml_files(monkeypatch, tmp_path):
    project = FakeProject(tmp_path / "project")
    flow_path = project.project_path / "processing/flows/standard.yaml"
    flow_path.parent.mkdir(parents=True)
    flow_path.write_text("# Retain this comment when copied\n" + model_to_yaml(standard_mt_flow()))
    monkeypatch.setattr("resistics.tui.load", lambda project_path: project)
    app = ResisticsTui(project.project_path)

    async def run_test():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app.screen.query_one(TabbedContent).active = "flows"
            app.screen.selected_flow_path = flow_path
            app.screen._show_yaml("#flow-content", flow_path)
            await pilot.press("y")
            await pilot.pause()
            copy_form = app.screen
            assert isinstance(copy_form, CopyYamlFileScreen)
            copy_form.query_one("#copy-yaml-name", Input).value = "standard_copy"
            await pilot.press("right")
            assert copy_form.focused is copy_form.query_one("#confirm-copy-yaml", Button)
            await pilot.press("left")
            assert copy_form.focused is copy_form.query_one("#cancel-copy-yaml", Button)
            copy_form.copy()
            await pilot.pause()
            copied_path = flow_path.with_name("standard_copy.yaml")
            assert copied_path.read_bytes() == flow_path.read_bytes()
            assert app.screen.selected_flow_path == copied_path
            assert app.screen.check_action("copy_yaml", ())
            assert app.screen.check_action("delete_yaml", ())
            await pilot.press("delete")
            await pilot.pause()
            delete_form = app.screen
            assert isinstance(delete_form, DeleteYamlFileScreen)
            cancel_button = delete_form.query_one("#cancel-delete-yaml", Button)
            delete_button = delete_form.query_one("#confirm-delete-yaml", Button)
            assert delete_form.focused is cancel_button
            await pilot.press("right")
            assert delete_form.focused is delete_button
            await pilot.press("left")
            assert delete_form.focused is cancel_button
            delete_form.delete()
            await pilot.pause()
            assert not copied_path.exists()
            assert flow_path.exists()
            assert app.screen.selected_flow_path is None
            assert app.screen.query_one("#flow-content", TextArea).text == "Select a flow"
            assert not app.screen.check_action("copy_yaml", ())
            assert not app.screen.check_action("delete_yaml", ())

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


def test_tui_uses_a_four_second_notification_timeout():
    assert ResisticsTui.NOTIFICATION_TIMEOUT == 4.0


def test_directory_picker_starts_at_home_with_parent_navigation():
    picker = DirectoryPickerScreen("Select a project", False)
    assert picker.start_path == Path.home()
    assert ("u", "parent_directory", "Up") in picker.BINDINGS
    assert ("escape", "cancel", "Cancel") in picker.BINDINGS
    assert "Space: expand/collapse" in picker.navigation_instruction
    assert "project folder" in picker.selection_instruction
    assert (
        "MTH5 file"
        in DirectoryPickerScreen("Select an MTH5 file", True).selection_instruction
    )


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
    assert "ToastRack {" in ResisticsTui.CSS
    assert "align: right bottom;" in ResisticsTui.CSS
    assert "Toast {" in ResisticsTui.CSS
    assert "width: 48;" in ResisticsTui.CSS
    assert ".launcher-layout { height: 1fr; align-horizontal: center; }" in (
        ResisticsTui.CSS
    )
    assert (
        "#metadata-content, #flow-content, #parameter-content, #criteria-content,"
        in (ResisticsTui.CSS)
    )
    assert "Input:focus { background: #202020; border: tall #0a009f; }" in (
        ResisticsTui.CSS
    )
    assert "#faa881" in ResisticsTui.CSS
    assert "#ac3600" in ResisticsTui.CSS
    assert "Button:focus" in ResisticsTui.CSS
    assert "background: #0a009f;" in ResisticsTui.CSS
    assert "Tree:focus > .tree--cursor" in ResisticsTui.CSS
    assert "DataTable:focus > .datatable--cursor" in ResisticsTui.CSS
    assert "text-style: none;" in ResisticsTui.CSS
    assert "background: transparent;" in CopyYamlFileScreen.CSS
    assert "background: transparent;" in DeleteYamlFileScreen.CSS
    assert "Button.dialog-action {" in ResisticsTui.CSS
    assert "background: #343434;" in ResisticsTui.CSS
    assert "Button.dialog-action:focus" in ResisticsTui.CSS
    assert "text-style: bold;" in ResisticsTui.CSS
