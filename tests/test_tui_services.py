"""Unit tests for Textual-independent project explorer services."""

from pathlib import Path

import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError

from resistics.flow import model_to_yaml, single_site_mt_flow
from resistics.job import JobDefinition
from resistics.project import MTH5FileSummary, ProjectDataDeletion
from resistics.tui import ProjectDataDeletionRequest
from resistics.tui.services import ProjectExplorerService


class ServiceProject:
    """Minimal project contract for service tests without a Textual app."""

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.mth5_path = project_path / "data.h5"
        self.ref_time = "2020-01-01 00:00:00"
        self.table = pd.DataFrame(
            columns=["survey", "station", "sample_rate", "run_path"]
        )
        self.closed = False
        self.deleted_labels = []
        self.project_data_paths = ["data/survey/station/results/default"]

    def file_summary(self):
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
        return []

    def list_project_data_items(self):
        return []

    def list_mth5_data_items(self):
        return []

    def list_project_output_labels(self):
        return ["default"]

    def preview_project_data_deletion(self, output_label=None):
        paths = self.project_data_paths if output_label in {None, "default"} else []
        return ProjectDataDeletion(output_label=output_label, paths=paths)

    def delete_project_data(self, output_label=None):
        self.deleted_labels.append(output_label)
        return self.preview_project_data_deletion(output_label)

    def close(self):
        self.closed = True


def test_project_explorer_service_returns_pydantic_discovery_without_textual(
    tmp_path,
):
    project = ServiceProject(tmp_path / "project")
    flow_path = project.project_path / "processing/flows/standard.yaml"
    flow_path.parent.mkdir(parents=True)
    flow_path.write_text(model_to_yaml(single_site_mt_flow()), encoding="utf-8")
    service = ProjectExplorerService(project)

    state = service.project_state()
    resources = service.resources("flows")

    assert isinstance(state, BaseModel)
    assert state.model_config.get("frozen")
    assert resources
    assert all(isinstance(resource, BaseModel) for resource in resources)
    assert service.job_resource_options("flows") == [
        ("Single-Site MT (Standard Windowing) (standard.yaml)", "standard.yaml")
    ]
    assert issubclass(ProjectDataDeletionRequest, BaseModel)
    assert ProjectDataDeletionRequest.model_config.get("frozen")
    assert ProjectDataDeletionRequest.model_json_schema()["type"] == "object"
    request = ProjectDataDeletionRequest(output_label="default")
    assert (
        ProjectDataDeletionRequest.model_validate_json(request.model_dump_json())
        == request
    )


def test_project_explorer_service_owns_yaml_mutations_and_invalidation(tmp_path):
    project = ServiceProject(tmp_path / "project")
    flow_path = project.project_path / "processing/flows/standard.yaml"
    flow_path.parent.mkdir(parents=True)
    original = model_to_yaml(single_site_mt_flow())
    flow_path.write_text(original, encoding="utf-8")
    service = ProjectExplorerService(project)
    first_resources = service.resources("flows")

    copy_path = service.copy_yaml(flow_path, "standard_copy", "flows")
    copied_resources = service.resources("flows")

    assert copy_path.read_text(encoding="utf-8") == original
    assert copied_resources is not first_resources
    assert {resource.path for resource in copied_resources} == {flow_path, copy_path}

    with pytest.raises(ValidationError):
        service.save_yaml(
            flow_path, type(single_site_mt_flow()), "id: incomplete\n", "flows"
        )
    assert flow_path.read_text(encoding="utf-8") == original

    updated = "# retained comment\n" + original
    service.save_yaml(flow_path, type(single_site_mt_flow()), updated, "flows")
    assert service.yaml_source(flow_path) == updated

    service.delete_yaml(copy_path, "flows")
    assert not copy_path.exists()
    assert [resource.path for resource in service.resources("flows")] == [flow_path]


def test_project_explorer_service_owns_jobs_deletion_and_project_lifecycle(tmp_path):
    project = ServiceProject(tmp_path / "project")
    service = ProjectExplorerService(project)

    job_path = service.create_job_template(
        JobDefinition(name="field", flow="standard.yaml", parameters="default.yaml")
    )
    labels, preview = service.deletion_options()
    selected = service.preview_project_data_deletion("default")
    deleted = service.delete_project_data("default")

    assert job_path.name == "field.yaml"
    assert service.job_template_names() == {"field"}
    assert labels == ["default"]
    assert preview.count == selected.count == deleted.count == 1
    assert project.deleted_labels == ["default"]

    with service.discovery():
        service.close()
        assert not project.closed
    assert project.closed
    with (
        pytest.raises(RuntimeError, match="Project screen closed"),
        service.discovery(),
    ):
        pass
