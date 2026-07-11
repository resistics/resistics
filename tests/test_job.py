"""Tests for project job discovery, validation, and execution."""

from pathlib import Path
from types import SimpleNamespace
import warnings

from resistics.flow import (
    FlowDefinition,
    FlowNode,
    ParameterSet,
    ProcessingJob,
    model_to_yaml,
)
from resistics.job import (
    JobDefinition,
    JobRunner,
    JobState,
    ProjectJobs,
    ResolvedJob,
)


def make_project(tmp_path):
    project_path = tmp_path / "project"
    for path in (
        "processing/flows",
        "processing/parameters",
        "processing/jobs",
        "logs",
    ):
        (project_path / path).mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        project_path=project_path,
        runs=["survey/station/run"],
    )


def write_valid_job(project):
    flow = FlowDefinition(
        name="read-only",
        nodes=[FlowNode(id="read", type="mth5_read")],
    )
    parameters = ParameterSet(name="defaults")
    definition = JobDefinition(
        name="example",
        flow="standard",
        parameters="defaults.yaml",
        runtime={"survey": "survey", "station": "station", "run": "run"},
        output_label="result",
    )
    (project.project_path / "processing/flows/standard.yaml").write_text(
        model_to_yaml(flow)
    )
    (project.project_path / "processing/parameters/defaults.yaml").write_text(
        model_to_yaml(parameters)
    )
    job_path = project.project_path / "processing/jobs/example.yaml"
    job_path.write_text(model_to_yaml(definition))
    return job_path


def test_project_jobs_resolves_named_files(tmp_path):
    project = make_project(tmp_path)
    job_path = write_valid_job(project)

    validation = ProjectJobs(project).validate(job_path)

    assert validation.ok
    assert validation.resolved_job is not None
    assert validation.resolved_job.flow_path.name == "standard.yaml"
    assert validation.resolved_job.parameters_path.name == "defaults.yaml"
    assert validation.resolved_job.processing_job.runtime["project_path"] == str(
        project.project_path
    )


def test_project_jobs_reports_missing_runtime_run(tmp_path):
    project = make_project(tmp_path)
    job_path = write_valid_job(project)
    project.runs = []

    validation = ProjectJobs(project).validate(job_path)

    assert not validation.ok
    assert "MTH5 run not found" in validation.errors[-1]


def test_project_jobs_refuses_existing_output(tmp_path):
    project = make_project(tmp_path)
    job_path = write_valid_job(project)
    output_path = project.project_path / "data/survey/station/results/result"
    output_path.mkdir(parents=True)

    validation = ProjectJobs(project).validate(job_path)

    assert not validation.ok
    assert f"Output already exists: {output_path}" in validation.errors


def test_project_jobs_lists_invalid_yaml(tmp_path):
    project = make_project(tmp_path)
    job_path = project.project_path / "processing/jobs/broken.yaml"
    job_path.write_text("name: broken\n")

    summaries = ProjectJobs(project).list()

    assert len(summaries) == 1
    assert not summaries[0].is_valid
    assert summaries[0].errors


def test_job_runner_cancels_between_nodes(tmp_path):
    project = make_project(tmp_path)
    project.close_mth5 = lambda: None
    processing_job = ProcessingJob(
        name="cancelled",
        flow=FlowDefinition(name="flow", nodes=[FlowNode(id="read", type="mth5_read")]),
        parameters=ParameterSet(name="parameters"),
        runtime={
            "project_path": str(project.project_path),
            "survey": "survey",
            "station": "station",
            "run": "run",
        },
    )
    resolved = ResolvedJob(
        path=Path("job.yaml"),
        definition=JobDefinition(
            name="cancelled",
            flow="flow.yaml",
            parameters="parameters.yaml",
            runtime={"survey": "survey", "station": "station", "run": "run"},
        ),
        processing_job=processing_job,
        flow_path=Path("flow.yaml"),
        parameters_path=Path("parameters.yaml"),
        output_path=project.project_path / "result",
    )
    events = []
    runner = JobRunner(project, events.append)
    runner.cancel()

    state = runner.run(resolved)

    assert state == JobState.cancelled
    assert events[-1].state == JobState.cancelled


def test_job_runner_archives_completed_job(monkeypatch, tmp_path):
    project = make_project(tmp_path)
    job_path = write_valid_job(project)
    resolved = ProjectJobs(project).validate(job_path).resolved_job
    assert resolved is not None
    events = []
    runner = JobRunner(project, events.append)
    monkeypatch.setattr(
        runner,
        "_handlers",
        lambda job: {"mth5_read": lambda inputs, parameters, runtime: "done"},
    )

    state = runner.run(resolved)

    assert state == JobState.completed
    assert events[-1].state == JobState.completed
    assert (resolved.output_path / "job_info.json").is_file()


def test_job_runner_captures_python_warnings(monkeypatch, tmp_path):
    project = make_project(tmp_path)
    job_path = write_valid_job(project)
    resolved = ProjectJobs(project).validate(job_path).resolved_job
    assert resolved is not None
    events = []
    runner = JobRunner(project, events.append)

    def warn_then_complete(inputs, parameters, runtime):
        warnings.warn("noisy input data", RuntimeWarning)
        return "done"

    monkeypatch.setattr(
        runner, "_handlers", lambda job: {"mth5_read": warn_then_complete}
    )

    assert runner.run(resolved) == JobState.completed
    assert "Captured 1 runtime warning" in events[-2].message
    log_path = project.project_path / "logs/example.log"
    assert "RuntimeWarning: noisy input data" in log_path.read_text()
