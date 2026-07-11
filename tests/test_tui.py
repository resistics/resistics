"""Smoke tests for the project terminal UI."""

import asyncio
from types import SimpleNamespace

from textual.containers import VerticalScroll
from textual.widgets import DataTable, Static, TabbedContent, Tree

from resistics.tui import ResisticsTui


class FakeProject:
    """Minimal project contract used by the interface smoke test."""

    def __init__(self, project_path):
        self.project_path = project_path
        self.ref_time = "2020-01-01T00:00:00"
        self.runs = []
        self.closed = False
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


def test_tui_mounts_project_views(tmp_path):
    project = FakeProject(tmp_path / "project")
    app = ResisticsTui(project)
    assert app.sub_title == str(project.project_path)

    async def run_test():
        async with app.run_test(size=(100, 40)):
            overview = app.query_one("#overview-content", Static)
            tree = app.query_one("#project-tree", Tree)
            table = app.query_one("#job-table", DataTable)
            metadata_details = app.query_one("#metadata-details", VerticalScroll)
            assert "project" in str(overview.render())
            assert str(tree.root.label) == "project"
            assert table.row_count == 0
            app.query_one(TabbedContent).active = "project"
            await asyncio.sleep(0)
            app.set_focus(tree)
            app.action_focus_next()
            assert app.focused is metadata_details

    asyncio.run(run_test())
    assert project.closed


def test_tui_uses_dark_surfaces_with_resistics_accents():
    """Keep the project explorer dark without losing the brand accents."""
    assert "background: #101010" in ResisticsTui.CSS
    assert "background: #202020" in ResisticsTui.CSS
    assert "#job-details:focus, #metadata-details:focus { background: #343434; }" in (
        ResisticsTui.CSS
    )
    assert "#faa881" in ResisticsTui.CSS
    assert "#ac3600" in ResisticsTui.CSS
