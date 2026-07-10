"""
Tests for the MTH5-backed project API.
"""
from pathlib import Path

import pandas as pd
import pytest
from mth5.mth5 import MTH5

from resistics.project import (
    PROJ_FILE,
    CANONICAL_PROJ_DIRS,
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

    @property
    def channel_summary(self):
        """Return a fake channel summary."""
        return self._channel_summary

    def open_mth5(self):
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


def test_init_creates_canonical_project_structure(tmp_path):
    """Test creating a canonical MTH5-backed project."""
    project_path = tmp_path / "project"
    mth5_path = tmp_path / "data.h5"
    mth5_path.write_text("")

    assert init(project_path, mth5_path, "2020-01-01 00:00:00")
    assert (project_path / PROJ_FILE).exists()
    for subdir in CANONICAL_PROJ_DIRS:
        assert (project_path / subdir).is_dir()
    assert not (project_path / "processing" / "runs").exists()
    assert check_project(project_path, mth5_path)


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
