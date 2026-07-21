"""Tests for the cached, UI-neutral project explorer index."""

from collections import Counter
from pathlib import Path
from threading import Event, Thread

import pandas as pd
from pydantic import BaseModel

from resistics.explorer import (
    ExplorerFileIdentity,
    ExplorerIssue,
    IndexedJob,
    IndexedResource,
    ProjectExplorerIndex,
    ProjectExplorerState,
)
from resistics.flow import default_parameter_set, model_to_yaml, standard_mt_flow
from resistics.job import JobDefinition
from resistics.project import (
    MTH5FileSummary,
    ProjectDataDeletion,
    ProjectDataItem,
)


class CountingProject:
    """Minimal project that records explorer-facing reads and handle closure."""

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.mth5_path = project_path / "data.h5"
        self.calls = Counter()
        self.closed = False
        self.table = pd.DataFrame(
            columns=["survey", "station", "sample_rate", "run_path"]
        )

    def _read(self, name: str) -> None:
        if self.closed:
            raise RuntimeError("MTH5 handle is closed")
        self.calls[name] += 1

    def file_summary(self):
        self._read("file_summary")
        return MTH5FileSummary(
            mth5_path=self.mth5_path,
            file_version="0.2.0",
            n_surveys=0,
            n_stations=0,
            n_runs=0,
            n_channels=0,
            sample_rates=[],
        )

    def list_runs(self):
        self._read("list_runs")
        return []

    def list_project_data_items(self):
        self._read("list_project_data_items")
        return [
            ProjectDataItem(
                source="project",
                path="survey/station/results/default/solution.json",
                name="solution.json",
                kind="file",
                data_type="transfer_function",
                is_dataset=True,
            )
        ]

    def list_mth5_data_items(self):
        self._read("list_mth5_data_items")
        return []

    def preview_project_data_deletion(self):
        self._read("preview_project_data_deletion")
        return ProjectDataDeletion(paths=["data/survey/station/results/default"])

    def close_mth5(self):
        self.closed = True


def test_public_explorer_results_are_frozen_serializable_pydantic_models(tmp_path):
    public_models = (
        ExplorerFileIdentity,
        ExplorerIssue,
        IndexedResource,
        IndexedJob,
        ProjectExplorerState,
    )

    assert all(issubclass(model, BaseModel) for model in public_models)
    assert all(model.model_config.get("frozen") for model in public_models)
    assert all(model.model_json_schema()["type"] == "object" for model in public_models)

    project = CountingProject(tmp_path / "project")
    state = ProjectExplorerIndex(project).project_state()
    restored = ProjectExplorerState.model_validate_json(state.model_dump_json())

    assert restored == state


def test_project_index_reuses_cached_project_state_and_survives_closed_handle(
    tmp_path,
):
    project = CountingProject(tmp_path / "project")
    index = ProjectExplorerIndex(project)

    first = index.project_state()
    second = index.project_state()
    runs = index.runs()

    assert second is first
    assert project.calls == {
        "file_summary": 1,
        "list_runs": 1,
        "list_project_data_items": 1,
        "list_mth5_data_items": 1,
        "preview_project_data_deletion": 1,
    }
    project.close_mth5()
    assert index.project_state() is first
    assert index.runs() is runs
    assert first.project_path == project.project_path
    assert first.has_project_data_to_delete


def test_project_index_invalidates_only_requested_sections(tmp_path):
    project = CountingProject(tmp_path / "project")
    flow_path = project.project_path / "processing/flows/standard.yaml"
    flow_path.parent.mkdir(parents=True)
    flow_path.write_text(model_to_yaml(standard_mt_flow()))
    index = ProjectExplorerIndex(project)

    project_state = index.project_state()
    flow_records = index.resources("flows")
    restored_flow = IndexedResource.model_validate_json(
        flow_records[0].model_dump_json()
    )
    index.invalidate("flows")

    assert restored_flow == flow_records[0]
    assert index.project_state() is project_state
    assert index.resources("flows") is not flow_records
    assert index.resources("flows")[0].model is flow_records[0].model
    assert index.resources("flows")[0] is flow_records[0]
    assert project.calls["file_summary"] == 1


def test_project_index_does_not_recapture_a_stale_concurrent_result(tmp_path):
    class BlockingProject(CountingProject):
        def __init__(self, project_path):
            super().__init__(project_path)
            self.first_summary_started = Event()
            self.release_first_summary = Event()

        def file_summary(self):
            self._read("file_summary")
            read_number = self.calls["file_summary"]
            if read_number == 1:
                self.first_summary_started.set()
                self.release_first_summary.wait(timeout=2)
            return MTH5FileSummary(
                mth5_path=self.mth5_path,
                file_version=f"0.{read_number}.0",
                n_surveys=0,
                n_stations=0,
                n_runs=0,
                n_channels=0,
                sample_rates=[],
            )

    project = BlockingProject(tmp_path / "project")
    index = ProjectExplorerIndex(project)
    stale_results = []
    stale_thread = Thread(target=lambda: stale_results.append(index.project_state()))

    stale_thread.start()
    assert project.first_summary_started.wait(timeout=2)
    index.invalidate("project")
    current = index.project_state()
    project.release_first_summary.set()
    stale_thread.join(timeout=2)

    assert not stale_thread.is_alive()
    assert stale_results[0].summary.file_version == "0.1.0"
    assert current.summary.file_version == "0.2.0"
    assert index.project_state() is current


def test_project_index_reparses_stale_files_after_invalidation(tmp_path):
    project = CountingProject(tmp_path / "project")
    flow_path = project.project_path / "processing/flows/standard.yaml"
    flow_path.parent.mkdir(parents=True)
    flow_path.write_text(model_to_yaml(standard_mt_flow()))
    index = ProjectExplorerIndex(project)

    valid = index.resources("flows")[0]
    flow_path.write_text("id: incomplete\nextra: changed-size\n")

    assert index.resources("flows")[0] is valid
    index.invalidate("flows")
    stale = index.resources("flows")[0]
    assert stale.identity != valid.identity
    assert stale.model is None
    assert stale.error is not None


def test_project_index_caches_malformed_resource_errors(tmp_path, monkeypatch):
    project = CountingProject(tmp_path / "project")
    flow_path = project.project_path / "processing/flows/broken.yaml"
    flow_path.parent.mkdir(parents=True)
    flow_path.write_text("id: incomplete\n")
    parse_calls = []

    from resistics import explorer as explorer_module

    parse = explorer_module.model_from_yaml_file

    def tracked_parse(model_type, path):
        parse_calls.append(path)
        return parse(model_type, path)

    monkeypatch.setattr(explorer_module, "model_from_yaml_file", tracked_parse)
    index = ProjectExplorerIndex(project)

    first = index.resources("flows")
    second = index.resources("flows")

    assert second is first
    assert first[0].model is None
    assert first[0].error
    assert parse_calls == [flow_path]


def test_project_index_detects_deleted_resources_on_explicit_refresh(tmp_path):
    project = CountingProject(tmp_path / "project")
    flow_path = project.project_path / "processing/flows/standard.yaml"
    flow_path.parent.mkdir(parents=True)
    flow_path.write_text(model_to_yaml(standard_mt_flow()))
    index = ProjectExplorerIndex(project)

    assert [record.path for record in index.resources("flows")] == [flow_path]
    flow_path.unlink()
    assert index.resources("flows")

    index.invalidate_all()
    assert index.resources("flows") == ()


def test_project_index_lists_jobs_without_reparsing_referenced_resources(
    tmp_path, monkeypatch
):
    project = CountingProject(tmp_path / "project")
    processing_path = project.project_path / "processing"
    resources = {
        processing_path / "flows/standard.yaml": standard_mt_flow(),
        processing_path / "parameters/default.yaml": default_parameter_set(),
        processing_path / "jobs/field.yaml": JobDefinition(
            name="field", flow="standard.yaml", parameters="default.yaml"
        ),
    }
    for path, model in resources.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(model_to_yaml(model))
    parse_calls = []

    from resistics import explorer as explorer_module

    parse = explorer_module.model_from_yaml_file

    def tracked_parse(model_type, path):
        parse_calls.append(path)
        return parse(model_type, path)

    monkeypatch.setattr(explorer_module, "model_from_yaml_file", tracked_parse)
    index = ProjectExplorerIndex(project)

    first = index.jobs()
    second = index.jobs()

    assert second is first
    assert len(first) == 1
    assert first[0].validation.resolved_job is not None
    assert Counter(parse_calls) == Counter(resources.keys())


def test_project_index_invalidates_jobs_when_a_referenced_flow_changes(tmp_path):
    project = CountingProject(tmp_path / "project")
    processing_path = project.project_path / "processing"
    flow_path = processing_path / "flows/standard.yaml"
    resources = {
        flow_path: standard_mt_flow(),
        processing_path / "parameters/default.yaml": default_parameter_set(),
        processing_path / "jobs/field.yaml": JobDefinition(
            name="field", flow="standard.yaml", parameters="default.yaml"
        ),
    }
    for path, model in resources.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(model_to_yaml(model))
    index = ProjectExplorerIndex(project)

    valid = index.jobs()
    flow_path.write_text("id: incomplete\nextra: changed-size\n")
    index.invalidate("flows")
    invalid = index.jobs()

    assert invalid is not valid
    assert valid[0].validation.resolved_job is not None
    assert invalid[0].validation.resolved_job is None
    assert invalid[0].summary.errors
