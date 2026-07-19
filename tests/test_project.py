"""
Tests for the MTH5-backed project API.
"""

from pathlib import Path

import h5py
import pandas as pd
import pytest
from mth5.mth5 import MTH5

from resistics.flow import FlowDefinition, ParameterSet, model_from_yaml_file
from resistics.project import (
    CANONICAL_PROJ_DIRS,
    PROJ_FILE,
    Project,
    check_project,
    get_flow_path,
    get_job_path,
    get_log_path,
    get_parameters_path,
    get_results_path,
    get_run_data_path,
    init,
    load,
    open_mth5,
)
from resistics.templates import (
    DEFAULT_FLOW_FILENAME,
    DEFAULT_PARAMETERS_FILENAME,
    MASK_CALCULATION_FLOW_FILENAME,
    REMOTE_REFERENCE_CRITERIA_FILENAME,
    REMOTE_REFERENCE_FLOW_FILENAME,
    SINGLE_SITE_CRITERIA_FILENAME,
    SINGLE_SITE_TARGET_FLOW_FILENAME,
    install_builtin_criteria_templates,
    install_builtin_flow_templates,
    install_builtin_parameter_templates,
)


class FakeChannelSummary:
    """Small stand-in for MTH5 channel_summary."""

    def to_dataframe(self):
        """Return an MTH5-like channel summary."""
        return pd.DataFrame(
            [
                {
                    "survey": "survey",
                    "station": "station",
                    "run": "run1",
                    "sample_rate": 128.0,
                    "start": "2020-01-01T00:00:00",
                    "end": "2020-01-01T01:00:00",
                },
                {
                    "survey": "survey",
                    "station": "remote",
                    "run": "run1",
                    "sample_rate": 128.0,
                    "start": "2020-01-01T00:30:00",
                    "end": "2020-01-01T01:30:00",
                },
            ]
        )


class FakeMTH5(MTH5):
    """Small stand-in for mth5.mth5.MTH5."""

    def __init__(self, path):
        self.path = path
        self._channel_summary = FakeChannelSummary()
        self.closed = False
        self._file_version = "0.2.0"

    @property
    def channel_summary(self):
        """Return a fake channel summary."""
        return self._channel_summary

    @property
    def file_version(self):
        return self._file_version

    def open_mth5(self, mode="r"):
        """Open no-op."""
        return None

    def close_mth5(self):
        """Mark closed."""
        self.closed = True


def test_mth5_project_paths():
    """Test canonical project path helpers."""
    project_path = Path("project")
    assert get_flow_path(project_path, "standard.yaml") == (
        project_path / "processing" / "flows" / "standard.yaml"
    )
    assert get_parameters_path(project_path, "default.yaml") == (
        project_path / "processing" / "parameters" / "default.yaml"
    )
    assert get_job_path(project_path, "job.yaml") == (
        project_path / "processing" / "jobs" / "job.yaml"
    )
    assert get_run_data_path(project_path, "survey", "station", "run1") == (
        project_path / "data" / "survey" / "station" / "run1"
    )
    assert get_results_path(project_path, "survey", "station", "proc1") == (
        project_path / "data" / "survey" / "station" / "results" / "proc1"
    )
    assert get_log_path(project_path, "proc1") == project_path / "logs" / "proc1.log"


def test_project_data_browser_lists_mth5_and_project_artifacts(tmp_path):
    project_path = tmp_path / "project"
    data_path = project_path / "data"
    mth5_path = tmp_path / "data.h5"
    with h5py.File(mth5_path, "w") as mth5_file:
        channel = mth5_file.create_dataset(
            "Experiment/Surveys/survey/Stations/station/Runs/run/Channels/ex",
            data=[1.0, 2.0],
        )
        channel.attrs["units"] = "nT"
        mth5_file.create_dataset(
            "Experiment/Surveys/survey/Stations/station/direct_run/ex",
            data=[1.0, 2.0],
        )
        mth5_file.create_dataset("Experiment/fc_summary", data=[1])
        mth5_file.create_dataset("Experiment/tf_summary", data=[1])
    evaluation_path = data_path / "survey" / "station" / "run" / "evals" / "default"
    evaluation_path.mkdir(parents=True)
    (evaluation_path / "metadata.json").write_text('{"name": "evaluation"}')
    (evaluation_path / "data.npz").write_bytes(b"evaluation data")
    mask_path = data_path / "survey" / "station" / "run" / "masks" / "night"
    mask_path.mkdir(parents=True)
    (mask_path / "metadata.json").write_text('{"name": "night"}')
    (mask_path / "data.npz").write_bytes(b"mask data")
    solution_path = data_path / "survey" / "station" / "results" / "mt"
    solution_path.mkdir(parents=True)
    (solution_path / "solution.json").write_text('{"name": "solution"}')
    project = Project.model_construct(
        project_path=project_path,
        mth5_path=mth5_path,
        ref_time="2020-01-01T00:00:00",
        mth5_data=FakeMTH5(mth5_path),
        table=pd.DataFrame(),
        plugin_paths=[],
        surveys=[],
        stations=[],
        runs=[],
    )

    mth5_items = {item.path: item for item in project.list_mth5_data_items()}
    assert (
        mth5_items[
            "/Experiment/Surveys/survey/Stations/station/Runs/run/Channels/ex"
        ].data_type
        == "time"
    )
    assert mth5_items["/Experiment/fc_summary"].data_type == "spectra"
    assert mth5_items["/Experiment/tf_summary"].data_type == "transfer_function"
    assert (
        mth5_items[
            "/Experiment/Surveys/survey/Stations/station/direct_run/ex"
        ].data_type
        == "time"
    )
    mth5_metadata = project.get_mth5_data_metadata(
        "/Experiment/Surveys/survey/Stations/station/Runs/run/Channels/ex"
    )
    assert mth5_metadata.values["shape"] == [2]
    assert mth5_metadata.values["attributes"] == {"units": "nT"}

    project_items = {item.path: item for item in project.list_project_data_items()}
    assert project_items["survey/station/run/evals/default"].data_type == "spectra"
    assert project_items["survey/station/run/evals/default"].is_dataset
    assert project_items["survey/station/run/masks/night"].data_type == "mask"
    assert project_items["survey/station/run/masks/night"].is_dataset
    assert project_items["survey/station/results/mt"].data_type == "transfer_function"
    assert project_items["survey/station/results/mt"].is_dataset
    project_metadata = project.get_project_data_metadata(
        "survey/station/run/evals/default"
    )
    assert project_metadata.values["metadata.json"] == {"name": "evaluation"}
    assert project.get_project_data_json(
        "survey/station/run/evals/default/metadata.json"
    ) == {"name": "evaluation"}
    assert project.get_project_data_json("survey/station/results/mt/solution.json") == {
        "name": "solution"
    }
    with pytest.raises(ValueError, match="inside project/data"):
        project.get_project_data_metadata("../outside")


def test_project_data_deletion_is_labelled_and_preserves_an_in_tree_mth5(tmp_path):
    project_path = tmp_path / "project"
    data_path = project_path / "data"
    mth5_path = data_path / "source" / "input.h5"
    mth5_path.parent.mkdir(parents=True)
    mth5_path.write_bytes(b"mth5")

    def write_artifacts(label: str) -> None:
        eval_path = data_path / "survey" / "station" / "run" / "evals" / label
        eval_path.mkdir(parents=True)
        (eval_path / "metadata.json").write_text("{}")
        (eval_path / "data.npz").write_bytes(b"eval")
        mask_path = data_path / "survey" / "station" / "run" / "masks" / label / "night"
        mask_path.mkdir(parents=True)
        (mask_path / "metadata.json").write_text("{}")
        (mask_path / "data.npz").write_bytes(b"mask")
        result_path = data_path / "survey" / "station" / "results" / label / "128"
        result_path.mkdir(parents=True)
        (result_path / "solution.json").write_text("{}")

    write_artifacts("first")
    write_artifacts("second")
    project = Project.model_construct(
        project_path=project_path,
        mth5_path=mth5_path,
        ref_time="2020-01-01T00:00:00",
        mth5_data=FakeMTH5(mth5_path),
        table=pd.DataFrame(),
        plugin_paths=[],
        surveys=[],
        stations=[],
        runs=[],
    )

    assert project.list_project_output_labels() == ["first", "second"]
    preview = project.preview_project_data_deletion("first")
    assert preview.output_label == "first"
    assert len(preview.paths) == 3
    deleted = project.delete_project_data("first")
    assert deleted.paths == preview.paths
    assert not (data_path / "survey/station/run/evals/first").exists()
    assert not (data_path / "survey/station/run/masks/first").exists()
    assert not (data_path / "survey/station/results/first").exists()
    assert (data_path / "survey/station/run/evals/second").is_dir()

    project.delete_project_data()
    assert mth5_path.is_file()
    assert not (data_path / "survey").exists()
    with pytest.raises(ValueError, match="output_label"):
        project.preview_project_data_deletion("../outside")


def test_init_creates_canonical_project_structure(tmp_path):
    """Test creating a canonical MTH5-backed project."""
    project_path = tmp_path / "project"
    mth5_path = tmp_path / "data.h5"
    mth5_path.write_text("")

    assert init(project_path, mth5_path, "2020-01-01 00:00:00")
    assert (project_path / PROJ_FILE).exists()
    for subdir in CANONICAL_PROJ_DIRS:
        assert (project_path / subdir).is_dir()
    flow_path = project_path / "processing" / "flows" / DEFAULT_FLOW_FILENAME
    default_parameters_path = (
        project_path / "processing" / "parameters" / DEFAULT_PARAMETERS_FILENAME
    )
    model_from_yaml_file(FlowDefinition, flow_path)
    default_parameters = model_from_yaml_file(ParameterSet, default_parameters_path)
    assert "resistics.decimate.DecimationSetup" in default_parameters.processes
    for flow_filename in (
        SINGLE_SITE_TARGET_FLOW_FILENAME,
        REMOTE_REFERENCE_FLOW_FILENAME,
    ):
        installed_flow = model_from_yaml_file(
            FlowDefinition, project_path / "processing" / "flows" / flow_filename
        )
        assert installed_flow.flow_stages()
        assert len(installed_flow.flow_stages()) == 2
    mask_flow = model_from_yaml_file(
        FlowDefinition,
        project_path / "processing" / "flows" / MASK_CALCULATION_FLOW_FILENAME,
    )
    assert mask_flow.id == "mask_calculation"
    assert {node.process for node in mask_flow.stages[0].nodes}.issuperset(
        {"resistics.mask.TimeMask", "resistics.mask.AbsoluteAmplitudeMask"}
    )
    assert set(default_parameters.processes).issuperset(
        {
            "resistics.mask.TimeMask",
            "resistics.mask.AbsoluteAmplitudeMask",
        }
    )
    assert (
        default_parameters.processes["resistics.window.WindowerTarget"]["target"] == 500
    )
    assert not (project_path / "processing" / "runs").exists()
    assert check_project(project_path, mth5_path)


def test_template_restoration_is_scoped_to_its_resource_type(tmp_path):
    project_path = tmp_path / "project"
    (project_path / "processing/flows").mkdir(parents=True)
    (project_path / "processing/parameters").mkdir(parents=True)

    flow_paths = install_builtin_flow_templates(project_path)

    assert {path.name for path in flow_paths} == {
        DEFAULT_FLOW_FILENAME,
        SINGLE_SITE_TARGET_FLOW_FILENAME,
        REMOTE_REFERENCE_FLOW_FILENAME,
        MASK_CALCULATION_FLOW_FILENAME,
    }
    assert not (
        project_path / "processing/parameters" / DEFAULT_PARAMETERS_FILENAME
    ).exists()

    parameter_paths = install_builtin_parameter_templates(project_path)

    assert [path.name for path in parameter_paths] == [DEFAULT_PARAMETERS_FILENAME]

    criteria_paths = install_builtin_criteria_templates(project_path)

    assert {path.name for path in criteria_paths} == {
        SINGLE_SITE_CRITERIA_FILENAME,
        REMOTE_REFERENCE_CRITERIA_FILENAME,
    }


def test_load_builds_mth5_summary(monkeypatch, tmp_path):
    """Test loading project metadata and the MTH5 channel summary."""
    import resistics.project as project_module

    monkeypatch.setattr(project_module, "MTH5", FakeMTH5)
    project_path = tmp_path / "project"
    mth5_path = tmp_path / "data.h5"
    mth5_path.write_text("")
    init(project_path, mth5_path, "2020-01-01 00:00:00")

    project = load(project_path)

    assert isinstance(project, Project)
    assert project.mth5_path == mth5_path
    assert project.surveys == ["survey"]
    assert project.stations == ["survey/remote", "survey/station"]
    assert project.runs == ["survey/remote/run1", "survey/station/run1"]
    assert project.fs() == [128.0]
    assert project.get_concurrent("survey/station") == ["survey/remote"]
    project.close_mth5()
    assert project.mth5_data.closed


def test_missing_mth5_path_fails(tmp_path):
    """Test project creation validates MTH5 path existence."""
    with pytest.raises(ValueError, match="MTH5 data file not found"):
        init(tmp_path / "project", tmp_path / "missing.h5", "2020-01-01")


def test_open_mth5_builds_serializable_read_only_summary(monkeypatch, tmp_path):
    import resistics.project as project_module

    monkeypatch.setattr(project_module, "MTH5", FakeMTH5)
    mth5_path = tmp_path / "data.h5"
    mth5_path.write_text("")

    source = open_mth5(mth5_path)
    summary = source.file_summary()

    assert summary.n_surveys == 1
    assert summary.n_stations == 2
    assert summary.n_runs == 2
    assert summary.model_validate_json(summary.model_dump_json()) == summary
    source.close_mth5()
    assert source.mth5_data.closed
