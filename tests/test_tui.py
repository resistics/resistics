"""Smoke tests for the project terminal UI."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go
import pytest
from textual.widgets import (
    Button,
    DataTable,
    Input,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
    Tree,
)

import resistics.flow as flow_module
import resistics.tui.app as tui_module
import resistics.tui.services as tui_services
from resistics.common import ProcessingProgressEvent, ProcessingProgressState
from resistics.explorer import ProjectExplorerState
from resistics.flow import default_parameter_set, model_to_yaml, standard_mt_flow
from resistics.gather import GatherCriteria
from resistics.job import JobDefinition, JobProgressEvent, JobScope, JobState
from resistics.project import MTH5FileSummary, ProjectDataDeletion, ProjectDataItem
from resistics.sampling import to_datetime
from resistics.testing import solution_mt
from resistics.tui import (
    ConfirmJobScreen,
    ConfirmProjectDataDeletionScreen,
    CopyYamlFileScreen,
    CreateJobScreen,
    CreateProjectScreen,
    DeleteProjectDataScreen,
    DeleteYamlFileScreen,
    DirectoryPickerScreen,
    ProjectExplorerScreen,
    ResisticsTui,
)


class FakeProject:
    """Minimal project contract used by the interface smoke test."""

    def __init__(self, project_path, create_jobs=True):
        self.project_path = project_path
        self.ref_time = to_datetime("2020-01-01T00:00:00")
        self.runs = []
        self.n_runs = 0
        self.closed = False
        self.project_data_items = []
        self.mth5_data_items = []
        self.output_labels = []
        self.project_data_paths = []
        self.deleted_data_scopes = []
        self.json_data = {}
        if create_jobs:
            (project_path / "processing/jobs").mkdir(parents=True)

    def file_summary(self):
        return MTH5FileSummary(
            mth5_path=self.project_path / "data.h5",
            file_version="0.2.0",
            start_time=None,
            end_time=None,
            sample_rates=[],
            n_surveys=0,
            n_stations=0,
            n_runs=self.n_runs,
            n_channels=0,
        )

    def list_surveys(self):
        return []

    def list_stations(self, survey=None):
        return []

    def list_runs(self, survey=None, station=None):
        return []

    def list_project_data_items(self):
        return self.project_data_items

    def list_mth5_data_items(self):
        return self.mth5_data_items

    def get_project_data_json(self, path):
        return self.json_data[path]

    def list_project_output_labels(self):
        return self.output_labels

    def preview_project_data_deletion(self, output_label=None):
        return ProjectDataDeletion(
            output_label=output_label, paths=list(self.project_data_paths)
        )

    def delete_project_data(self, output_label=None):
        self.deleted_data_scopes.append(output_label)
        return self.preview_project_data_deletion(output_label)

    def close_mth5(self):
        self.closed = True


def test_tui_import_defers_feature_specific_modules():
    deferred_modules = {
        "matplotlib",
        "mth5",
        "mt_io",
        "mt_metadata",
        "plotly",
        "resistics.explorer",
        "resistics.gather",
        "resistics.job",
        "resistics.plot",
        "resistics.project",
        "resistics.regression",
        "resistics.spectra",
        "resistics.transfunc",
        "scipy",
    }
    command = "\n".join(
        (
            "import sys",
            "import resistics.tui",
            f"deferred = {sorted(deferred_modules)!r}",
            "print('\\n'.join(name for name in deferred if name in sys.modules))",
        )
    )

    result = subprocess.run(  # noqa: S603 - executable and code are controlled
        [sys.executable, "-c", command],
        capture_output=True,
        check=True,
        text=True,
    )

    assert result.stdout == "\n"


def test_feature_errors_explain_missing_dependencies():
    missing = ModuleNotFoundError("No module named 'plotly'", name="plotly")

    assert tui_services._feature_error("Plotting", missing) == (
        "Plotting requires the missing dependency 'plotly'. "
        "Reinstall resistics with its required dependencies."
    )
    assert tui_services._feature_error("Plotting", ValueError("invalid plot")) == (
        "invalid plot"
    )


def _record_call(calls, name, operation):
    def wrapped(*args, **kwargs):
        calls.append(name)
        return operation(*args, **kwargs)

    return wrapped


def _measure_action_checks(screen, actions, io_calls) -> float:
    io_calls.clear()
    started = perf_counter()
    for _ in range(100):
        for action in actions:
            screen.check_action(action, ())
    elapsed = perf_counter() - started
    assert io_calls == []
    return elapsed


def _find_tree_node(node, data):
    if node.data == data:
        return node
    for child in node.children:
        found = _find_tree_node(child, data)
        if found is not None:
            return found
    return None


async def _wait_for(condition, timeout=2.0):
    async def poll():
        while not condition():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout=timeout)


def test_project_and_overview_loading_do_not_block_the_ui(monkeypatch, tmp_path):
    project = FakeProject(tmp_path / "project")
    load_started = Event()
    release_load = Event()
    summary_started = Event()
    release_summary = Event()

    def slow_load(project_path):
        load_started.set()
        release_load.wait(timeout=2)
        return project

    file_summary = project.file_summary

    def slow_summary():
        summary_started.set()
        release_summary.wait(timeout=2)
        return file_summary()

    monkeypatch.setattr("resistics.project.load", slow_load)
    monkeypatch.setattr(project, "file_summary", slow_summary)
    app = ResisticsTui(project.project_path)

    async def run_test():
        async with app.run_test(size=(100, 40)) as pilot:
            try:
                await _wait_for(load_started.is_set)
                assert app.screen.query_one("#project-loading", Static)

                release_load.set()
                await _wait_for(summary_started.is_set)
                explorer = app.screen
                assert isinstance(explorer, ProjectExplorerScreen)
                assert "Loading project overview" in str(
                    explorer.query_one("#project-content", Static).render()
                )

                release_summary.set()
                await _wait_for(
                    lambda: (
                        "MTH5 version"
                        in str(explorer.query_one("#project-content", Static).render())
                    )
                )
                await pilot.pause()
            finally:
                release_load.set()
                release_summary.set()

    asyncio.run(run_test())
    assert project.closed


def test_project_open_rejects_and_closes_a_superseded_result(monkeypatch, tmp_path):
    first = FakeProject(tmp_path / "first")
    second = FakeProject(tmp_path / "second")
    first_started = Event()
    release_first = Event()

    def load_project(project_path):
        if project_path == first.project_path:
            first_started.set()
            release_first.wait(timeout=2)
            return first
        return second

    monkeypatch.setattr("resistics.project.load", load_project)
    app = ResisticsTui(first.project_path)

    async def run_test():
        async with app.run_test(size=(100, 40)):
            try:
                await _wait_for(first_started.is_set)
                app.open_project_path(second.project_path)
                await _wait_for(
                    lambda: (
                        isinstance(app.screen, ProjectExplorerScreen)
                        and app.screen.project is second
                    )
                )

                release_first.set()
                await _wait_for(lambda: first.closed)

                assert isinstance(app.screen, ProjectExplorerScreen)
                assert app.screen.project is second
            finally:
                release_first.set()

    asyncio.run(run_test())
    assert first.closed
    assert second.closed


def test_explorer_defers_project_close_until_discovery_finishes(monkeypatch, tmp_path):
    project = FakeProject(tmp_path / "project")
    summary_started = Event()
    release_summary = Event()
    file_summary = project.file_summary

    def slow_summary():
        summary_started.set()
        release_summary.wait(timeout=2)
        return file_summary()

    monkeypatch.setattr("resistics.project.load", lambda project_path: project)
    monkeypatch.setattr(project, "file_summary", slow_summary)
    app = ResisticsTui(project.project_path)

    async def run_test():
        async with app.run_test(size=(100, 40)) as pilot:
            try:
                await _wait_for(summary_started.is_set)
                await pilot.press("x")
                await pilot.pause()

                assert app.screen.query_one("#open-project", Button)

                assert not project.closed

                release_summary.set()
                await _wait_for(lambda: project.closed)
            finally:
                release_summary.set()

    asyncio.run(run_test())
    assert project.closed


def test_explorer_loads_inactive_tabs_lazily(monkeypatch, tmp_path):
    project = FakeProject(tmp_path / "project")
    flow_path = project.project_path / "processing/flows/standard.yaml"
    flow_path.parent.mkdir(parents=True)
    flow_path.write_text(model_to_yaml(standard_mt_flow()))
    parse_calls = []

    from resistics import explorer as explorer_module

    parse = explorer_module.model_from_yaml_file

    def tracked_parse(model_type, path):
        parse_calls.append(path)
        return parse(model_type, path)

    monkeypatch.setattr(explorer_module, "model_from_yaml_file", tracked_parse)
    monkeypatch.setattr("resistics.project.load", lambda project_path: project)
    app = ResisticsTui(project.project_path)

    async def run_test():
        async with app.run_test(size=(100, 40)):
            await _wait_for(lambda: isinstance(app.screen, ProjectExplorerScreen))
            screen = app.screen
            await _wait_for(
                lambda: (
                    "MTH5 version"
                    in str(screen.query_one("#project-content", Static).render())
                )
            )

            assert parse_calls == []
            assert screen.query_one("#flow-table", DataTable).row_count == 0

            screen.query_one(TabbedContent).active = "flows"
            await _wait_for(
                lambda: screen.query_one("#flow-table", DataTable).row_count == 1
            )
            assert parse_calls == [flow_path]

    asyncio.run(run_test())
    assert project.closed


def test_explorer_rejects_stale_worker_results(monkeypatch, tmp_path):
    project = FakeProject(tmp_path / "project")
    monkeypatch.setattr("resistics.project.load", lambda project_path: project)
    app = ResisticsTui(project.project_path)

    async def run_test():
        async with app.run_test(size=(100, 40)):
            await _wait_for(lambda: isinstance(app.screen, ProjectExplorerScreen))
            screen = app.screen
            await _wait_for(
                lambda: (
                    "MTH5 version"
                    in str(screen.query_one("#project-content", Static).render())
                )
            )
            first_started = Event()
            release_first = Event()
            calls = 0

            def state(n_runs):
                summary = project.file_summary().model_copy(update={"n_runs": n_runs})
                return ProjectExplorerState(
                    project_path=project.project_path,
                    mth5_identity=None,
                    summary=summary,
                    project_data_items=(),
                    mth5_data_items=(),
                    has_project_data_to_delete=False,
                )

            def delayed_state():
                nonlocal calls
                calls += 1
                if calls == 1:
                    first_started.set()
                    release_first.wait(timeout=2)
                    return state(1)
                return state(2)

            monkeypatch.setattr(screen.explorer_index, "project_state", delayed_state)
            try:
                screen.action_refresh()
                await _wait_for(first_started.is_set)
                screen.action_refresh()
                await _wait_for(
                    lambda: (
                        "Runs: 2"
                        in str(screen.query_one("#project-content", Static).render())
                    )
                )
                release_first.set()
                await asyncio.sleep(0.05)

                assert "Runs: 2" in str(
                    screen.query_one("#project-content", Static).render()
                )
            finally:
                release_first.set()

    asyncio.run(run_test())
    assert project.closed


def test_tui_mounts_project_views(monkeypatch, tmp_path):
    project = FakeProject(tmp_path / "project")
    flow = standard_mt_flow()
    flow_path = project.project_path / "processing/flows/standard_mt.yaml"
    flow_path.parent.mkdir(parents=True)
    flow_path.write_text(model_to_yaml(flow))
    parameters_path = project.project_path / "processing/parameters/default_mt.yaml"
    parameters_path.parent.mkdir(parents=True)
    parameters_path.write_text(model_to_yaml(default_parameter_set()))
    monkeypatch.setattr("resistics.project.load", lambda project_path: project)
    app = ResisticsTui(project.project_path)

    async def run_test():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            project_overview = app.screen.query_one("#project-content", Static)
            tree = app.screen.query_one("#data-tree", Tree)
            table = app.screen.query_one("#job-table", DataTable)
            flow_table = app.screen.query_one("#flow-table", DataTable)
            parameter_table = app.screen.query_one("#parameter-table", DataTable)
            metadata_details = app.screen.query_one("#data-metadata", TextArea)
            assert "project" in str(project_overview.render())
            assert "Reference time: 2020-01-01 00:00:00" in str(
                project_overview.render()
            )
            assert str(tree.root.label) == "Data"
            assert not tree.show_root
            assert len(tree.root.children) == 0
            assert table.row_count == 0
            assert flow_table.row_count == 0
            assert flow_table.cell_padding == 1
            assert parameter_table.row_count == 0
            assert app.screen.query_one("#flows", TabPane)
            assert app.screen.query_one("#parameters", TabPane)
            assert app.screen.query_one("#criteria", TabPane)
            assert app.screen.query_one("#criteria-table", DataTable)
            flow_editor = app.screen.query_one("#flow-content", TextArea)
            metadata_editor = app.screen.query_one("#data-metadata", TextArea)
            assert metadata_editor.language == "json"
            assert metadata_editor.theme == "vscode_dark"
            assert metadata_editor.read_only
            assert metadata_editor.show_line_numbers
            assert flow_editor.language == "yaml"
            assert flow_editor.theme == "vscode_dark"
            assert flow_editor.read_only
            assert flow_editor.show_line_numbers
            app.screen.query_one(TabbedContent).active = "flows"
            await _wait_for(lambda: flow_table.row_count == 1)
            assert "restore_defaults" in {
                binding.binding.action
                for binding in app.screen.active_bindings.values()
            }
            app.screen.query_one(TabbedContent).active = "parameters"
            await _wait_for(lambda: parameter_table.row_count == 1)
            app.screen.query_one(TabbedContent).active = "flows"
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
            app.screen.query_one(TabbedContent).active = "data"
            await _wait_for(lambda: len(tree.root.children) == 2)
            assert [str(node.label) for node in tree.root.children] == [
                "Project",
                "MTH5",
            ]
            app.screen.set_focus(tree)
            app.action_focus_next()
            assert app.focused is metadata_details

    asyncio.run(run_test())
    assert project.closed


def test_cached_action_checks_are_fast_and_do_not_repeat_io(
    monkeypatch, record_property, tmp_path
):
    """Keep every Footer check fast and independent of project I/O."""
    project = FakeProject(tmp_path / "project")
    project.n_runs = 1
    project.project_data_paths = [project.project_path / "data/derived"]
    project.table = pd.DataFrame(
        [
            {
                "survey": "survey",
                "station": "field",
                "sample_rate": 128.0,
                "run_path": "survey/field/run1",
            }
        ]
    )
    project.mth5_data_items = [
        ProjectDataItem(
            source="mth5",
            path="/Experiment/Surveys/survey/Stations/field/run1",
            name="run1",
            kind="group",
            data_type="time",
        )
    ]
    project.list_runs = lambda survey=None, station=None: [
        SimpleNamespace(survey="survey", station="field", run="run1")
    ]
    flow_path = project.project_path / "processing/flows/standard.yaml"
    flow_path.parent.mkdir(parents=True)
    flow_path.write_text(model_to_yaml(standard_mt_flow()))
    parameters_path = project.project_path / "processing/parameters/default.yaml"
    parameters_path.parent.mkdir(parents=True)
    parameters_path.write_text(model_to_yaml(default_parameter_set()))
    job_path = project.project_path / "processing/jobs/field.yaml"
    job_path.write_text(
        model_to_yaml(
            JobDefinition(
                name="field",
                flow="standard.yaml",
                parameters="default.yaml",
                scope=JobScope(stations=["field"]),
            )
        )
    )
    monkeypatch.setattr("resistics.project.load", lambda project_path: project)
    app = ResisticsTui(project.project_path)

    async def run_test():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            tabs = screen.query_one(TabbedContent)
            data_tree = screen.query_one("#data-tree", Tree)
            tabs.active = "data"
            await _wait_for(
                lambda: (
                    _find_tree_node(
                        data_tree.root,
                        ("mth5", "/Experiment/Surveys/survey/Stations/field/run1"),
                    )
                    is not None
                )
            )
            tabs.active = "flows"
            await _wait_for(
                lambda: screen.query_one("#flow-table", DataTable).row_count == 1
            )
            tabs.active = "jobs"
            await _wait_for(
                lambda: screen.query_one("#job-table", DataTable).row_count == 1
            )
            tabs.active = "project"
            io_calls = []

            monkeypatch.setattr(
                project,
                "file_summary",
                _record_call(io_calls, "project.file_summary", project.file_summary),
            )
            monkeypatch.setattr(
                project,
                "list_runs",
                _record_call(io_calls, "project.list_runs", project.list_runs),
            )
            monkeypatch.setattr(
                project,
                "get_project_data_json",
                _record_call(
                    io_calls,
                    "project.get_project_data_json",
                    project.get_project_data_json,
                ),
            )
            monkeypatch.setattr(
                project,
                "preview_project_data_deletion",
                _record_call(
                    io_calls,
                    "project.preview_project_data_deletion",
                    project.preview_project_data_deletion,
                ),
            )
            for method_name in (
                "exists",
                "glob",
                "is_dir",
                "is_file",
                "iterdir",
                "open",
                "read_bytes",
                "read_text",
                "resolve",
                "rglob",
                "stat",
            ):
                operation = getattr(Path, method_name)
                monkeypatch.setattr(
                    Path,
                    method_name,
                    _record_call(io_calls, f"Path.{method_name}", operation),
                )
            monkeypatch.setattr(
                flow_module,
                "model_from_yaml_file",
                _record_call(
                    io_calls,
                    "model_from_yaml_file",
                    flow_module.model_from_yaml_file,
                ),
            )
            monkeypatch.setattr(
                screen.project_jobs,
                "validate",
                _record_call(
                    io_calls, "project_jobs.validate", screen.project_jobs.validate
                ),
            )

            actions = (
                "edit_yaml",
                "create_job",
                "copy_yaml",
                "delete_yaml",
                "save_yaml",
                "discard_yaml",
                "close_project",
                "restore_defaults",
                "run_selected_job",
                "cancel_job",
                "expand_data_node",
                "collapse_data_node",
                "plot",
            )

            elapsed = _measure_action_checks(screen, actions, io_calls)

            data_node = _find_tree_node(
                data_tree.root,
                (
                    "mth5",
                    "/Experiment/Surveys/survey/Stations/field/run1",
                ),
            )
            assert data_node is not None
            tabs.active = "data"
            data_tree.focus()
            data_tree.move_cursor(data_node)
            await pilot.pause()
            elapsed += _measure_action_checks(screen, actions, io_calls)

            tabs.active = "flows"
            flow_table = screen.query_one("#flow-table", DataTable)
            flow_table.focus()
            flow_table.move_cursor(row=0)
            await pilot.pause()
            elapsed += _measure_action_checks(screen, actions, io_calls)

            tabs.active = "jobs"
            job_table = screen.query_one("#job-table", DataTable)
            job_table.focus()
            job_table.move_cursor(row=0)
            await pilot.pause()
            elapsed += _measure_action_checks(screen, actions, io_calls)

            record_property("cached_action_checks_seconds", f"{elapsed:.6f}")
            assert elapsed < 2.0

    asyncio.run(run_test())


def test_binding_refreshes_follow_owned_state_transitions(
    monkeypatch, record_property, tmp_path
):
    """Refresh the Footer once after state changes and not for content-only work."""
    project = FakeProject(tmp_path / "project")
    monkeypatch.setattr("resistics.project.load", lambda project_path: project)
    app = ResisticsTui(project.project_path)

    async def run_test():
        async with app.run_test(size=(100, 40)):
            screen = app.screen
            refresh_calls = []
            refresh_bindings = screen.refresh_bindings

            def tracked_refresh_bindings():
                refresh_calls.append(perf_counter())
                refresh_bindings()

            monkeypatch.setattr(screen, "refresh_bindings", tracked_refresh_bindings)

            started = perf_counter()
            for _ in range(100):
                screen.refresh_tab_bindings()
            tab_seconds = perf_counter() - started
            tab_calls = len(refresh_calls)

            refresh_calls.clear()
            started = perf_counter()
            screen.action_refresh()
            project_refresh_seconds = perf_counter() - started
            project_refresh_calls = len(refresh_calls)

            refresh_calls.clear()
            started = perf_counter()
            screen._show_progress(
                JobProgressEvent(
                    state=JobState.completed,
                    message="Job complete",
                    job_name="field",
                )
            )
            progress_seconds = perf_counter() - started
            progress_refresh_calls = len(refresh_calls)

            refresh_calls.clear()
            started = perf_counter()
            screen.show_data_metadata(
                SimpleNamespace(node=SimpleNamespace(data=("category", "Time data")))
            )
            metadata_seconds = perf_counter() - started
            metadata_refresh_calls = len(refresh_calls)

            record_property("tab_handler_seconds_100", f"{tab_seconds:.6f}")
            record_property(
                "project_refresh_handler_seconds", f"{project_refresh_seconds:.6f}"
            )
            record_property(
                "terminal_progress_handler_seconds", f"{progress_seconds:.6f}"
            )
            record_property("metadata_handler_seconds", f"{metadata_seconds:.6f}")
            record_property(
                "binding_refresh_calls",
                ",".join(
                    str(value)
                    for value in (
                        tab_calls,
                        project_refresh_calls,
                        progress_refresh_calls,
                        metadata_refresh_calls,
                    )
                ),
            )

            assert tab_calls == 100
            assert project_refresh_calls == 1
            assert progress_refresh_calls == 1
            assert metadata_refresh_calls == 0
            assert (
                max(
                    tab_seconds / 100,
                    project_refresh_seconds,
                    progress_seconds,
                    metadata_seconds,
                )
                < 0.05
            )

    asyncio.run(run_test())
    assert project.closed


def test_tui_invalidates_explorer_index_after_owned_mutations(monkeypatch, tmp_path):
    project = FakeProject(tmp_path / "project")
    flow_path = project.project_path / "processing/flows/standard.yaml"
    flow_path.parent.mkdir(parents=True)
    flow_path.write_text(model_to_yaml(standard_mt_flow()))
    parameters_path = project.project_path / "processing/parameters/default.yaml"
    parameters_path.parent.mkdir(parents=True)
    parameters_path.write_text(model_to_yaml(default_parameter_set()))
    monkeypatch.setattr("resistics.project.load", lambda project_path: project)
    app = ResisticsTui(project.project_path)

    async def run_test():
        async with app.run_test(size=(100, 40)):
            screen = app.screen
            invalidations = []
            invalidate = screen.explorer_index.invalidate
            invalidate_all = screen.explorer_index.invalidate_all

            def tracked_invalidate(*sections):
                invalidations.append(sections)
                invalidate(*sections)

            def tracked_invalidate_all():
                invalidations.append(("all",))
                invalidate_all()

            monkeypatch.setattr(screen.explorer_index, "invalidate", tracked_invalidate)
            monkeypatch.setattr(
                screen.explorer_index, "invalidate_all", tracked_invalidate_all
            )

            screen.action_refresh()

            screen.query_one(TabbedContent).active = "flows"
            screen.selected_flow_path = flow_path
            screen._show_yaml("#flow-content", flow_path)
            screen.action_edit_yaml()
            screen.query_one("#flow-content", TextArea).text += "\n# edited\n"
            screen.action_save_yaml()

            definition = JobDefinition(
                name="field", flow="standard.yaml", parameters="default.yaml"
            )
            screen._job_template_created(definition)
            job_path = project.project_path / "processing/jobs/field.yaml"
            screen._yaml_file_deleted(job_path, "#job-content", True)

            project.project_data_paths = ["data/derived"]
            screen._project_data_deletion_confirmed(
                project.preview_project_data_deletion(), True
            )
            screen._show_progress(
                JobProgressEvent(
                    state=JobState.completed,
                    message="Job complete",
                    job_name="field",
                )
            )

            assert invalidations == [
                ("all",),
                ("flows",),
                ("jobs",),
                ("jobs",),
                ("project",),
                ("project", "jobs"),
            ]
            await _wait_for(
                lambda: all(worker.is_finished for worker in screen.workers)
            )
            await _wait_for(lambda: screen._active_discoveries == 0)

    asyncio.run(run_test())
    assert project.closed


def test_tui_catalogues_project_and_mth5_data_by_type(monkeypatch, tmp_path):
    project = FakeProject(tmp_path / "project")
    project.project_data_items = [
        ProjectDataItem(
            source="project",
            path="run/evals/default",
            parent_path="run/evals",
            name="default",
            kind="directory",
            data_type="spectra",
            is_dataset=True,
        ),
        ProjectDataItem(
            source="project",
            path="run/evals",
            parent_path="run",
            name="evals",
            kind="directory",
            data_type="spectra",
        ),
        ProjectDataItem(
            source="project",
            path="run",
            name="run",
            kind="directory",
            data_type="other",
        ),
    ]
    project.mth5_data_items = [
        ProjectDataItem(
            source="mth5",
            path="/Time/ex",
            parent_path="/Time",
            name="ex",
            kind="dataset",
            data_type="time",
            is_dataset=True,
        ),
        ProjectDataItem(
            source="mth5",
            path="/Experiment/fc_summary",
            parent_path="/Experiment",
            name="fc_summary",
            kind="dataset",
            data_type="spectra",
            is_dataset=True,
        ),
        ProjectDataItem(
            source="mth5",
            path="/Experiment",
            parent_path="/",
            name="Experiment",
            kind="group",
            data_type="other",
        ),
        ProjectDataItem(
            source="mth5",
            path="/Time",
            parent_path="/",
            name="Time",
            kind="group",
            data_type="other",
        ),
    ]
    monkeypatch.setattr("resistics.project.load", lambda project_path: project)
    app = ResisticsTui(project.project_path)

    async def run_test():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            data_tree = screen.query_one("#data-tree", Tree)
            screen.query_one(TabbedContent).active = "data"
            await _wait_for(lambda: len(data_tree.root.children) == 2)
            assert [str(node.label) for node in data_tree.root.children] == [
                "Project",
                "MTH5",
            ]
            project_node, mth5_node = data_tree.root.children
            assert project_node.is_expanded
            assert mth5_node.is_expanded
            assert [str(node.label) for node in project_node.children] == [
                "Time data (0)",
                "Spectra/evaluations (1)",
                "Masks (0)",
                "Transfer functions (0)",
                "Other (0)",
            ]
            assert [str(node.label) for node in mth5_node.children] == [
                "Time data (1)",
                "Spectra/evaluations (1)",
                "Masks (0)",
                "Transfer functions (0)",
                "Other (0)",
            ]
            assert set(screen.data_items) == {
                "project:run",
                "project:run/evals",
                "project:run/evals/default",
                "mth5:/Experiment",
                "mth5:/Experiment/fc_summary",
                "mth5:/Time",
                "mth5:/Time/ex",
            }
            data_tree.focus()
            time_category = mth5_node.children[0]
            data_tree.move_cursor(time_category)
            await pilot.pause()
            assert screen.check_action("expand_data_node", ())
            await pilot.press("right_square_bracket")
            assert time_category.is_expanded
            assert time_category.children[0].is_expanded
            assert screen.check_action("collapse_data_node", ())
            await pilot.press("left_square_bracket")
            assert time_category.is_collapsed
            assert time_category.children[0].is_collapsed

    asyncio.run(run_test())


def test_tui_counts_mth5_time_data_by_run():
    items = [
        ProjectDataItem(
            source="mth5",
            path="/Experiment/Surveys/survey/Stations/station/Runs",
            name="Runs",
            kind="group",
            data_type="time",
        ),
        ProjectDataItem(
            source="mth5",
            path="/Experiment/Surveys/survey/Stations/station/Runs/first/Channels/ex",
            name="ex",
            kind="dataset",
            data_type="time",
            is_dataset=True,
        ),
        ProjectDataItem(
            source="mth5",
            path="/Experiment/Surveys/survey/Stations/station/Runs/first/Channels/hx",
            name="hx",
            kind="dataset",
            data_type="time",
            is_dataset=True,
        ),
        ProjectDataItem(
            source="mth5",
            path="/Experiment/Surveys/survey/Stations/station/Runs/second/Channels/ex",
            name="ex",
            kind="dataset",
            data_type="time",
            is_dataset=True,
        ),
    ]

    assert ProjectExplorerScreen._data_category_count(("mth5", "/"), items, "time") == 2


def test_tui_views_project_json_and_confirms_project_data_deletion(
    monkeypatch, tmp_path
):
    project = FakeProject(tmp_path / "project")
    project.project_data_items = [
        ProjectDataItem(
            source="project",
            path="survey/station/results/default/128/solution.json",
            parent_path="survey/station/results/default/128",
            name="solution.json",
            kind="file",
            data_type="transfer_function",
        )
    ]
    project.json_data["survey/station/results/default/128/solution.json"] = {
        "solution": {"component": "Zxy"}
    }
    project.output_labels = ["default"]
    project.project_data_paths = ["survey"]
    monkeypatch.setattr("resistics.project.load", lambda project_path: project)
    app = ResisticsTui(project.project_path)

    def find_node(node, data):
        if node.data == data:
            return node
        for child in node.children:
            found = find_node(child, data)
            if found is not None:
                return found
        return None

    async def run_test():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            screen.query_one(TabbedContent).active = "data"
            await pilot.pause()
            tree = screen.query_one("#data-tree", Tree)
            node = find_node(
                tree.root,
                ("project", "survey/station/results/default/128/solution.json"),
            )
            assert node is not None
            screen.show_data_metadata(SimpleNamespace(node=node))
            assert screen.query_one("#data-metadata", TextArea).text == json.dumps(
                project.json_data[node.data[1]], indent=2
            )
            assert screen.check_action("delete_yaml", ())
            await pilot.press("delete")
            await pilot.pause()
            picker = app.screen
            assert isinstance(picker, DeleteProjectDataScreen)
            picker.delete_label()
            await pilot.pause()
            confirm = app.screen
            assert isinstance(confirm, ConfirmProjectDataDeletionScreen)
            confirm.delete()
            await pilot.pause()
            assert project.deleted_data_scopes == ["default"]

    asyncio.run(run_test())


def test_tui_plot_controls_follow_supported_data_selection(monkeypatch, tmp_path):
    project = FakeProject(tmp_path / "project")
    project.n_runs = 1
    project.list_runs = lambda survey=None, station=None: [
        SimpleNamespace(survey="CONUS South", station="CAS04", run="a")
    ]
    project.mth5_data_items = [
        ProjectDataItem(
            source="mth5",
            path="/Experiment/Surveys/CONUS_South/Stations/CAS04/a",
            name="a",
            kind="group",
            data_type="time",
        ),
        ProjectDataItem(
            source="mth5",
            path=("/Experiment/Surveys/CONUS_South/Stations/CAS04/a/ex"),
            name="ex",
            kind="dataset",
            data_type="time",
        ),
        ProjectDataItem(
            source="mth5",
            path="/Experiment/fc_summary",
            name="fc_summary",
            kind="dataset",
            data_type="spectra",
        ),
    ]
    monkeypatch.setattr("resistics.project.load", lambda project_path: project)
    app = ResisticsTui(project.project_path)

    def find_node(node, data):
        if node.data == data:
            return node
        for child in node.children:
            found = find_node(child, data)
            if found is not None:
                return found
        return None

    async def run_test():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            data_tree = screen.query_one("#data-tree", Tree)
            assert screen.check_action("plot", ())
            screen.query_one(TabbedContent).active = "data"
            await pilot.pause()
            assert not screen.check_action("plot", ())

            time_node = find_node(
                data_tree.root,
                (
                    "mth5",
                    "/Experiment/Surveys/CONUS_South/Stations/CAS04/a/ex",
                ),
            )
            assert time_node is not None
            time_node.parent.expand()
            await pilot.pause()
            data_tree.focus()
            data_tree.move_cursor(time_node)
            await pilot.pause()
            assert screen.check_action("plot", ())

            run_node = find_node(
                data_tree.root,
                ("mth5", "/Experiment/Surveys/CONUS_South/Stations/CAS04/a"),
            )
            assert run_node is not None
            data_tree.move_cursor(run_node)
            await pilot.pause()
            assert screen.check_action("plot", ())

            raw_spectra_node = find_node(
                data_tree.root, ("mth5", "/Experiment/fc_summary")
            )
            assert raw_spectra_node is not None
            raw_spectra_node.parent.expand()
            await pilot.pause()
            data_tree.move_cursor(raw_spectra_node)
            await pilot.pause()
            assert not screen.check_action("plot", ())

    asyncio.run(run_test())


def test_tui_builds_figures_with_existing_plotters(monkeypatch, tmp_path):
    time_plot_calls = []

    class FakeTimeData:
        def plot(self, max_pts):
            time_plot_calls.append(max_pts)
            return "time figure"

    class FakeProjectForPlot:
        def __init__(self):
            self.read_calls = []

        def plot(self):
            return "timeline figure"

        def read_run(self, survey, station, run, chans=None):
            self.read_calls.append((survey, station, run, chans))
            return FakeTimeData()

    project = FakeProjectForPlot()
    assert ProjectExplorerScreen._build_plot_figure(project, ("project", None)) == (
        "timeline figure"
    )
    assert (
        ProjectExplorerScreen._build_plot_figure(
            project, ("time", ("survey", "station", "run", "ex"))
        )
        == "time figure"
    )
    assert project.read_calls == [("survey", "station", "run", ["ex"])]
    assert time_plot_calls == [5_000]

    spectra_path = tmp_path / "evaluation"
    spectra_path.mkdir()
    spectra_figure = go.Figure()

    class FakeSpectraData:
        def plot(self):
            return spectra_figure

    class FakeSpectraReader:
        def __init__(self):
            self.paths = []

        def run(self, path):
            self.paths.append(path)
            return FakeSpectraData()

    reader = FakeSpectraReader()
    monkeypatch.setattr("resistics.spectra.SpectraDataReader", lambda: reader)
    assert (
        ProjectExplorerScreen._build_plot_figure(project, ("spectra", spectra_path))
        is spectra_figure
    )
    assert reader.paths == [spectra_path]

    solution_path = tmp_path / "solution.json"
    solution_mt().write(solution_path)
    figure = ProjectExplorerScreen._build_plot_figure(
        project, ("transfer_function", solution_path)
    )
    assert isinstance(figure, go.Figure)


@pytest.mark.parametrize(
    "target,message",
    [
        (("time", ("survey", "station")), "survey, station, run, channel"),
        (("spectra", "evaluation"), "data path"),
        (("transfer_function", 3), "solution path"),
    ],
)
def test_tui_rejects_mismatched_plot_payloads(target, message):
    with pytest.raises(ValueError, match=message):
        ProjectExplorerScreen._build_plot_figure(SimpleNamespace(), target)


def test_tui_plots_a_valid_selected_flow(monkeypatch, tmp_path):
    project = FakeProject(tmp_path / "project")
    flow_path = project.project_path / "processing/flows/standard.yaml"
    flow_path.parent.mkdir(parents=True)
    flow_path.write_text(model_to_yaml(standard_mt_flow()))
    invalid_path = project.project_path / "processing/flows/invalid.yaml"
    invalid_path.write_text("not: [valid")
    monkeypatch.setattr("resistics.project.load", lambda project_path: project)
    app = ResisticsTui(project.project_path)

    async def run_test():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            screen.query_one(TabbedContent).active = "flows"
            flow_table = screen.query_one("#flow-table", DataTable)
            await _wait_for(lambda: flow_table.row_count == 2)
            flow_table.focus()
            valid_row = next(
                index
                for index, row in enumerate(flow_table.ordered_rows)
                if row.key.value == str(flow_path)
            )
            invalid_row = next(
                index
                for index, row in enumerate(flow_table.ordered_rows)
                if row.key.value == str(invalid_path)
            )
            flow_table.move_cursor(row=valid_row)
            await pilot.pause()
            assert screen.check_action("plot", ())

            flow_table.move_cursor(row=invalid_row)
            await pilot.pause()
            assert not screen.check_action("plot", ())
            flow_table.move_cursor(row=valid_row)
            await pilot.pause()
            screen.editing_yaml = True
            assert not screen.check_action("plot", ())

    asyncio.run(run_test())
    assert project.closed


def test_tui_builds_flow_figures_without_preview_files(tmp_path):
    """Flow plotting uses the regular Plotly path without a managed preview."""
    project = FakeProject(tmp_path / "project")
    flow_path = project.project_path / "processing/flows/standard.yaml"
    flow_path.parent.mkdir(parents=True)
    flow_path.write_text(model_to_yaml(standard_mt_flow()))

    figure = ProjectExplorerScreen._build_plot_figure(project, ("flow", flow_path))

    assert isinstance(figure, go.Figure)
    assert figure.layout.title.text == "Flow: Single-Site MT (Standard Windowing)"
    assert not hasattr(tui_module, "_flow_preview_url")


def test_tui_builds_and_enables_valid_job_plots(monkeypatch, tmp_path):
    project = FakeProject(tmp_path / "project")
    project.table = pd.DataFrame(
        [
            {
                "survey": "survey",
                "station": "field",
                "sample_rate": 128.0,
                "run_path": "survey/field/run1",
            }
        ]
    )
    flow_path = project.project_path / "processing/flows/standard.yaml"
    flow_path.parent.mkdir(parents=True)
    flow_path.write_text(model_to_yaml(standard_mt_flow()))
    parameters_path = project.project_path / "processing/parameters/default.yaml"
    parameters_path.parent.mkdir(parents=True)
    parameters_path.write_text(model_to_yaml(default_parameter_set()))
    job_path = project.project_path / "processing/jobs/field.yaml"
    job_path.write_text(
        model_to_yaml(
            JobDefinition(
                name="field",
                flow="standard.yaml",
                parameters="default.yaml",
                scope=JobScope(stations=["field"]),
            )
        )
    )
    invalid_path = project.project_path / "processing/jobs/invalid.yaml"
    invalid_path.write_text("not: [valid")
    monkeypatch.setattr("resistics.project.load", lambda project_path: project)
    app = ResisticsTui(project.project_path)

    async def run_test():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            screen.query_one(TabbedContent).active = "jobs"
            table = screen.query_one("#job-table", DataTable)
            await _wait_for(lambda: table.row_count == 2)
            table.focus()
            valid_row = next(
                index
                for index, row in enumerate(table.ordered_rows)
                if row.key.value == str(job_path)
            )
            invalid_row = next(
                index
                for index, row in enumerate(table.ordered_rows)
                if row.key.value == str(invalid_path)
            )
            table.move_cursor(row=valid_row)
            await pilot.pause()
            assert screen.check_action("plot", ())
            figure = screen._build_plot_figure(project, ("job", job_path))
            assert isinstance(figure, go.Figure)
            assert "Job: field" in figure.layout.title.text

            table.move_cursor(row=invalid_row)
            await pilot.pause()
            assert not screen.check_action("plot", ())

    asyncio.run(run_test())
    assert project.closed


def test_close_project_returns_to_home(monkeypatch, tmp_path):
    project = FakeProject(tmp_path / "project")
    monkeypatch.setattr("resistics.project.load", lambda project_path: project)
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
    monkeypatch.setattr("resistics.project.load", lambda project_path: project)
    app = ResisticsTui(project.project_path)

    async def run_test():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app.screen.query_one(TabbedContent).active = "jobs"
            await _wait_for(lambda: "jobs" in app.screen._loaded_sections)
            await pilot.press("n")
            await pilot.pause()
            form = app.screen
            assert isinstance(form, CreateJobScreen)
            assert form.query_one("#job-flow", Select).value == "standard.yaml"
            assert form.query_one("#job-parameters", Select).value == "default.yaml"
            assert form.query_one("#job-output-label", Input).value == "default"
            form.query_one("#cancel-job-template", Button).focus()
            await pilot.press("right")
            assert form.focused is form.query_one("#create-job-template", Button)
            await pilot.press("left")
            assert form.focused is form.query_one("#cancel-job-template", Button)
            form.query_one("#job-name", Input).value = "field_job"
            form.query_one("#job-output-label", Input).value = "field_output"
            form.query_one("#job-criteria", Select).value = "field.yaml"
            form.create()
            await _wait_for(
                lambda: app.screen.query_one("#job-table", DataTable).row_count == 1
            )
            assert app.screen.query_one("#job-table", DataTable).row_count == 1
            yaml_text = (
                project.project_path / "processing/jobs/field_job.yaml"
            ).read_text()
            assert "flow: standard.yaml" in yaml_text
            assert "parameters: default.yaml" in yaml_text
            assert "criteria: field.yaml" in yaml_text
            assert "output_label: field_output" in yaml_text
            assert "scope:" in yaml_text

    asyncio.run(run_test())
    assert project.closed


def test_tui_copies_and_deletes_selected_yaml_files(monkeypatch, tmp_path):
    project = FakeProject(tmp_path / "project")
    flow_path = project.project_path / "processing/flows/standard.yaml"
    flow_path.parent.mkdir(parents=True)
    flow_path.write_text(
        "# Retain this comment when copied\n" + model_to_yaml(standard_mt_flow())
    )
    monkeypatch.setattr("resistics.project.load", lambda project_path: project)
    app = ResisticsTui(project.project_path)

    async def run_test():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app.screen.query_one(TabbedContent).active = "flows"
            await _wait_for(
                lambda: app.screen.query_one("#flow-table", DataTable).row_count == 1
            )
            app.screen.selected_flow_path = flow_path
            app.screen._show_yaml("#flow-content", flow_path)
            await pilot.press("y")
            await pilot.pause()
            copy_form = app.screen
            assert isinstance(copy_form, CopyYamlFileScreen)
            copy_form.query_one("#copy-yaml-name", Input).value = "standard_copy"
            await pilot.press("right")
            assert copy_form.focused is copy_form.query_one(
                "#confirm-copy-yaml", Button
            )
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
            assert (
                app.screen.query_one("#flow-content", TextArea).text == "Select a flow"
            )
            assert not app.screen.check_action("copy_yaml", ())
            assert not app.screen.check_action("delete_yaml", ())

    asyncio.run(run_test())
    assert project.closed


def test_tui_deletes_invalid_yaml_from_highlight_without_opening_it(
    monkeypatch, tmp_path
):
    project = FakeProject(tmp_path / "project")
    resources = [
        (
            "flows",
            "#flow-table",
            "#flow-content",
            project.project_path / "processing/flows/invalid.yaml",
            "bad: [",
            "Select a flow",
            "No YAML flows found in processing/flows",
        ),
        (
            "parameters",
            "#parameter-table",
            "#parameter-content",
            project.project_path / "processing/parameters/invalid.yaml",
            "bad: [",
            "Select a parameter set",
            "No YAML parameter sets found in processing/parameters",
        ),
        (
            "criteria",
            "#criteria-table",
            "#criteria-content",
            project.project_path / "processing/criteria/legacy.yaml",
            "remote_references:\n  survey/a: survey/b\n",
            "Select a criteria file",
            "No YAML criteria files found in processing/criteria",
        ),
        (
            "jobs",
            "#job-table",
            "#job-content",
            project.project_path / "processing/jobs/invalid.yaml",
            "bad: [",
            "Select a job",
            "No YAML jobs found in processing/jobs",
        ),
    ]
    for _, _, _, path, content, _, _ in resources:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    monkeypatch.setattr("resistics.project.load", lambda project_path: project)
    app = ResisticsTui(project.project_path)

    async def run_test():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            for (
                tab,
                table_id,
                editor_id,
                path,
                _,
                placeholder,
                empty_message,
            ) in resources:
                app.screen.query_one(TabbedContent).active = tab
                table = app.screen.query_one(table_id, DataTable)
                await _wait_for(lambda table=table: table.row_count == 1)
                table.focus()
                table.move_cursor(row=0)
                await pilot.pause()
                assert app.screen.query_one(editor_id, TextArea).text == placeholder
                assert app.screen.check_action("delete_yaml", ())
                await pilot.press("delete")
                await pilot.pause()
                delete_form = app.screen
                assert isinstance(delete_form, DeleteYamlFileScreen)
                assert delete_form.source == path
                delete_form.delete()
                await pilot.pause()
                assert not path.exists()
                await _wait_for(lambda tab=tab: tab in app.screen._loaded_sections)
                assert app.screen.query_one(editor_id, TextArea).text == empty_message

    asyncio.run(run_test())
    assert project.closed


def test_tui_copies_and_runs_highlighted_job_without_opening_it(monkeypatch, tmp_path):
    project = FakeProject(tmp_path / "project")
    project.table = pd.DataFrame(
        [
            {
                "survey": "survey",
                "station": "target",
                "sample_rate": 128.0,
                "run_path": "survey/target/run",
            }
        ]
    )
    flow_path = project.project_path / "processing/flows/standard.yaml"
    flow_path.parent.mkdir(parents=True)
    flow_path.write_text(model_to_yaml(standard_mt_flow()))
    parameters_path = project.project_path / "processing/parameters/default.yaml"
    parameters_path.parent.mkdir(parents=True)
    parameters_path.write_text(model_to_yaml(default_parameter_set()))
    job_path = project.project_path / "processing/jobs/field.yaml"
    job_path.write_text(
        model_to_yaml(
            JobDefinition(
                name="field",
                flow="standard.yaml",
                parameters="default.yaml",
                scope=JobScope(stations=["target"]),
            )
        )
    )
    monkeypatch.setattr("resistics.project.load", lambda project_path: project)
    app = ResisticsTui(project.project_path)

    async def run_test():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app.screen.query_one(TabbedContent).active = "jobs"
            table = app.screen.query_one("#job-table", DataTable)
            await _wait_for(lambda: table.row_count == 1)
            table.focus()
            table.move_cursor(row=0)
            await pilot.pause()
            assert app.screen.selected_job_path is None
            assert app.screen.query_one("#job-content", TextArea).text == "Select a job"
            assert app.screen.check_action("run_selected_job", ())
            assert app.screen.check_action("copy_yaml", ())

            await pilot.press("j")
            await pilot.pause()
            confirmation = app.screen
            assert isinstance(confirmation, ConfirmJobScreen)
            assert confirmation.validation.resolved_job.path == job_path
            confirmation.action_cancel()
            await pilot.pause()
            assert app.screen.query_one("#job-content", TextArea).text == "Select a job"

            table = app.screen.query_one("#job-table", DataTable)
            table.focus()
            await pilot.press("y")
            await pilot.pause()
            copy_form = app.screen
            assert isinstance(copy_form, CopyYamlFileScreen)
            copy_form.query_one("#copy-yaml-name", Input).value = "field_copy"
            copy_form.copy()
            await pilot.pause()
            copied_path = job_path.with_name("field_copy.yaml")
            assert copied_path.read_bytes() == job_path.read_bytes()

    asyncio.run(run_test())
    assert project.closed


def test_tui_copies_highlighted_flow_parameters_and_criteria_without_opening(
    monkeypatch, tmp_path
):
    project = FakeProject(tmp_path / "project")
    resources = [
        (
            "flows",
            "#flow-table",
            "#flow-content",
            project.project_path / "processing/flows/standard.yaml",
            model_to_yaml(standard_mt_flow()),
            "Select a flow",
        ),
        (
            "parameters",
            "#parameter-table",
            "#parameter-content",
            project.project_path / "processing/parameters/default.yaml",
            model_to_yaml(default_parameter_set()),
            "Select a parameter set",
        ),
        (
            "criteria",
            "#criteria-table",
            "#criteria-content",
            project.project_path / "processing/criteria/single_site.yaml",
            model_to_yaml(GatherCriteria()),
            "Select a criteria file",
        ),
    ]
    for _, _, _, path, content, _ in resources:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    monkeypatch.setattr("resistics.project.load", lambda project_path: project)
    app = ResisticsTui(project.project_path)

    async def run_test():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            for tab, table_id, editor_id, source, _, placeholder in resources:
                app.screen.query_one(TabbedContent).active = tab
                table = app.screen.query_one(table_id, DataTable)
                await _wait_for(lambda table=table: table.row_count == 1)
                table.focus()
                table.move_cursor(row=0)
                await pilot.pause()
                assert app.screen.query_one(editor_id, TextArea).text == placeholder
                assert app.screen.check_action("copy_yaml", ())

                await pilot.press("y")
                await pilot.pause()
                copy_form = app.screen
                assert isinstance(copy_form, CopyYamlFileScreen)
                assert copy_form.source == source
                copy_form.query_one(
                    "#copy-yaml-name", Input
                ).value = f"{source.stem}_copy"
                copy_form.copy()
                await pilot.pause()
                copied = source.with_name(f"{source.stem}_copy.yaml")
                assert copied.read_bytes() == source.read_bytes()

    asyncio.run(run_test())
    assert project.closed


def test_tui_opens_invalid_criteria_source_without_crashing(monkeypatch, tmp_path):
    project = FakeProject(tmp_path / "project")
    criteria_path = project.project_path / "processing/criteria/legacy.yaml"
    criteria_path.parent.mkdir(parents=True)
    criteria_source = "remote_references:\n  survey/a: survey/b\n"
    criteria_path.write_text(criteria_source)
    monkeypatch.setattr("resistics.project.load", lambda project_path: project)
    app = ResisticsTui(project.project_path)

    async def run_test():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app.screen.query_one(TabbedContent).active = "criteria"
            table = app.screen.query_one("#criteria-table", DataTable)
            await _wait_for(lambda: table.row_count == 1)
            table.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert app.screen.selected_criteria_path == criteria_path
            assert (
                app.screen.query_one("#criteria-content", TextArea).text
                == criteria_source
            )

    asyncio.run(run_test())
    assert project.closed


def test_run_job_confirmation_uses_dialog_navigation():
    validation = SimpleNamespace(
        resolved_job=SimpleNamespace(
            definition=SimpleNamespace(
                name="field_job",
                flow="standard.yaml",
                parameters="default.yaml",
                criteria=None,
                scope=SimpleNamespace(stages=[]),
                output_label="field_job",
            )
        )
    )
    app = ResisticsTui()

    async def run_test():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app.push_screen(ConfirmJobScreen(validation))
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, ConfirmJobScreen)
            cancel_button = dialog.query_one("#cancel", Button)
            run_button = dialog.query_one("#confirm", Button)
            assert dialog.focused is cancel_button
            await pilot.press("right")
            assert dialog.focused is run_button
            await pilot.press("left")
            assert dialog.focused is cancel_button
            dialog.cancel()
            await pilot.pause()

    asyncio.run(run_test())


def test_tui_runs_job_worker_and_reports_threaded_progress(monkeypatch, tmp_path):
    project_path = tmp_path / "project"
    project = FakeProject(project_path)
    processing_project = FakeProject(project_path, create_jobs=False)
    loaded_projects = iter((project, processing_project))

    class FakeJobRunner:
        def __init__(self, project, progress_callback):
            self.project = project
            self.progress_callback = progress_callback

        def run(self, resolved_job):
            self.progress_callback(
                JobProgressEvent(
                    state=JobState.running,
                    message="Started: runs",
                    job_name=resolved_job.definition.name,
                    survey="survey",
                    station="a",
                    run="run1",
                )
            )
            self.progress_callback(
                JobProgressEvent(
                    state=JobState.running,
                    message="Prepared regression frequency 2 of 4",
                    job_name=resolved_job.definition.name,
                    survey="survey",
                    station="a",
                    sample_rate=128.0,
                    progress=ProcessingProgressEvent(
                        state=ProcessingProgressState.advanced,
                        task="prepare_regression",
                        current=2,
                        total=4,
                        message="Prepared regression frequency 2 of 4",
                    ),
                )
            )
            self.progress_callback(
                JobProgressEvent(
                    state=JobState.completed,
                    message="Job complete",
                    job_name=resolved_job.definition.name,
                )
            )

    monkeypatch.setattr("resistics.project.load", lambda path: next(loaded_projects))
    monkeypatch.setattr("resistics.job.JobRunner", FakeJobRunner)
    app = ResisticsTui(project_path)

    async def run_test():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            explorer = app.screen
            explorer.selected_validation = SimpleNamespace(
                resolved_job=SimpleNamespace(
                    definition=SimpleNamespace(name="field_job")
                )
            )
            errors = []

            def run_job():
                try:
                    explorer._execute_selected_job.__wrapped__(explorer)
                except Exception as exc:
                    errors.append(exc)

            thread = Thread(target=run_job)
            thread.start()

            async def wait_for_job_thread():
                while thread.is_alive():
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(wait_for_job_thread(), timeout=2)
            thread.join()
            await pilot.pause()

            assert errors == []
            assert explorer.job_state == JobState.completed
            assert explorer.job_runner is None
            assert (
                str(explorer.query_one("#activity-status", Static).render())
                == "field_job: completed"
            )
            activity_log = explorer.query_one("#activity-log", RichLog)
            activity = "\n".join(line.text for line in activity_log.lines)
            normalized_activity = " ".join(activity.split())
            assert "station survey/a, run run1" in normalized_activity
            assert "station survey/a, sample rate 128 Hz" in normalized_activity
            assert "2/4" in normalized_activity

    asyncio.run(run_test())
    assert processing_project.closed


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

    monkeypatch.setattr("resistics.project.load", fail)
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

    monkeypatch.setattr("resistics.project.open_mth5", lambda path: source)
    monkeypatch.setattr("resistics.project.init", initialise)
    monkeypatch.setattr("resistics.project.load", lambda path: project)
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
            assert app.screen.query_one("#project-content", Static)

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
    assert "#data-metadata, #flow-content, #parameter-content, #criteria-content," in (
        ResisticsTui.CSS
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
    assert "background: transparent;" in ConfirmJobScreen.CSS
    assert "Button.dialog-action {" in ResisticsTui.CSS
    assert "background: #343434;" in ResisticsTui.CSS
    assert "Button.dialog-action:focus" in ResisticsTui.CSS
    assert "text-style: bold;" in ResisticsTui.CSS
