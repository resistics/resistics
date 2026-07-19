"""Tests for scoped, batch-oriented project jobs."""

from pathlib import Path

import pandas as pd
import pytest

from resistics.common import ResisticsProcess
from resistics.flow import (
    FlowDefinition,
    FlowNode,
    FlowStage,
    ParameterSet,
    model_from_yaml,
    model_to_yaml,
)
from resistics.gather import (
    GatherCriteria,
    RateGatherCriteria,
    StationGatherCriteria,
)
from resistics.job import JobDefinition, JobRunner, JobScope, JobState, ProjectJobs

RUN_BATCHES = []


class RunMarker(ResisticsProcess):
    output_type = "marker"
    runtime_requirements = ["run_batch"]

    def execute(self, inputs, context):
        RUN_BATCHES.append(context["run_batch"]["run"])
        return "done"


class BatchWriter(ResisticsProcess):
    output_type = "job_result"
    runtime_requirements = ["staging_output_path"]

    def execute(self, inputs, context):
        path = Path(context["staging_output_path"])
        path.mkdir(parents=True)
        (path / "result.txt").write_text("done")
        return {"result_path": str(path)}


def make_project(tmp_path):
    project_path = tmp_path / "project"
    for path in (
        "processing/flows",
        "processing/parameters",
        "processing/criteria",
        "processing/jobs",
        "logs",
    ):
        (project_path / path).mkdir(parents=True, exist_ok=True)
    return type(
        "ProjectStub",
        (),
        {
            "project_path": project_path,
            "mth5_path": project_path / "data.h5",
            "ref_time": "2020-01-01T00:00:00Z",
            "runs": ["survey/a/run1", "survey/a/run2", "survey/b/run3"],
            "table": pd.DataFrame(
                [
                    {
                        "survey": "survey",
                        "station": "a",
                        "sample_rate": 128.0,
                        "run_path": "survey/a/run1",
                    },
                    {
                        "survey": "survey",
                        "station": "a",
                        "sample_rate": 128.0,
                        "run_path": "survey/a/run2",
                    },
                    {
                        "survey": "survey",
                        "station": "b",
                        "sample_rate": 4.0,
                        "run_path": "survey/b/run3",
                    },
                ]
            ),
        },
    )()


def write_job(project, scope=None, criteria=None):
    flow = FlowDefinition(
        id="read_only",
        name="read-only",
        stages=[
            FlowStage(
                stage_id="read",
                scope="run",
                nodes=[FlowNode(id="read", process="resistics.time.MTH5TimeReader")],
            )
        ],
    )
    definition = JobDefinition(
        name="example",
        flow="standard",
        parameters="defaults.yaml",
        criteria=criteria,
        scope=scope or JobScope(),
    )
    (project.project_path / "processing/flows/standard.yaml").write_text(
        model_to_yaml(flow)
    )
    (project.project_path / "processing/parameters/defaults.yaml").write_text(
        model_to_yaml(ParameterSet(name="defaults"))
    )
    path = project.project_path / "processing/jobs/example.yaml"
    path.write_text(model_to_yaml(definition))
    return path


def test_empty_scope_batches_every_station_rate(tmp_path):
    project = make_project(tmp_path)
    path = write_job(project)

    validation = ProjectJobs(project).validate(path)

    assert validation.ok, validation.errors
    assert validation.resolved_job is not None
    batches = validation.resolved_job.batches
    assert [(item.station, item.sample_rate, item.run_paths) for item in batches] == [
        ("a", 128.0, ["survey/a/run1", "survey/a/run2"]),
        ("b", 4.0, ["survey/b/run3"]),
    ]


def test_create_template_writes_a_new_editable_job_without_criteria(tmp_path):
    project = make_project(tmp_path)
    write_job(project)

    path = ProjectJobs(project).create_template(
        JobDefinition(name="new_job", flow="standard.yaml", parameters="defaults.yaml")
    )

    assert path == project.project_path / "processing/jobs/new_job.yaml"
    yaml_text = path.read_text()
    assert "criteria:" not in yaml_text
    definition = model_from_yaml(JobDefinition, yaml_text)
    assert definition.name == "new_job"
    assert definition.flow == "standard.yaml"
    assert definition.parameters == "defaults.yaml"
    assert definition.scope == JobScope()
    assert definition.output_label == "default"


@pytest.mark.parametrize("name", ["", "with space", "job.yaml", "../job"])
def test_create_template_rejects_unsafe_job_names(tmp_path, name):
    project = make_project(tmp_path)
    definition = JobDefinition(name=name, flow="standard", parameters="defaults")

    with pytest.raises(ValueError, match="Job name must"):
        ProjectJobs(project).create_template(definition)


def test_create_template_does_not_overwrite_an_existing_job(tmp_path):
    project = make_project(tmp_path)
    write_job(project)
    definition = JobDefinition(name="example", flow="standard", parameters="defaults")

    with pytest.raises(ValueError, match="already exists"):
        ProjectJobs(project).create_template(definition)


def test_scope_limits_the_station_rate_batches(tmp_path):
    project = make_project(tmp_path)
    path = write_job(project, JobScope(stations=["a"], sample_rates=[128.0]))

    validation = ProjectJobs(project).validate(path)

    assert validation.ok, validation.errors
    assert [
        item.station_path
        for item in ProjectJobs(project).plan_batches(
            validation.resolved_job.definition
        )
    ] == ["survey/a"]


def test_stage_scope_alias_is_accepted_in_job_yaml():
    definition = model_from_yaml(
        JobDefinition,
        """
name: example
flow: standard
parameters: defaults
scope:
  stage_scope: [evals_to_tf]
""",
    )

    assert definition.scope.stages == ["evals_to_tf"]


def test_job_resolves_static_criteria_from_criteria_directory(tmp_path):
    project = make_project(tmp_path)
    project.table = pd.concat(
        [
            project.table,
            pd.DataFrame(
                [
                    {
                        "survey": "survey",
                        "station": "b",
                        "sample_rate": 128.0,
                        "run_path": "survey/b/run4",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    (project.project_path / "processing/criteria/field.yaml").write_text(
        model_to_yaml(
            GatherCriteria(
                stations={
                    "survey/a": StationGatherCriteria(
                        sampling_frequencies={
                            128: RateGatherCriteria(remote_references=["survey/b"])
                        }
                    )
                }
            )
        )
    )
    path = write_job(project, criteria="field")

    validation = ProjectJobs(project).validate(path)

    assert validation.ok, validation.errors
    assert validation.resolved_job.criteria.resolve(
        "survey/a", 128
    ).remote_references == ["survey/b"]


def test_run_stage_expands_one_hop_to_explicit_remote_runs(tmp_path):
    project = make_project(tmp_path)
    project.table = pd.concat(
        [
            project.table,
            pd.DataFrame(
                [
                    {
                        "survey": "survey",
                        "station": "b",
                        "sample_rate": 128.0,
                        "run_path": "survey/b/remote-run",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    criteria = GatherCriteria(
        stations={
            "survey/a": StationGatherCriteria(
                sampling_frequencies={
                    128: RateGatherCriteria(remote_references=["survey/b"])
                }
            )
        }
    )
    batches = ProjectJobs(project).plan_batches(
        JobDefinition(
            name="example",
            flow="standard",
            parameters="default",
            scope=JobScope(stations=["a"]),
        )
    )

    paths = JobRunner(project)._run_stage_paths(batches, criteria)

    assert paths == [
        "survey/a/run1",
        "survey/a/run2",
        "survey/b/remote-run",
    ]


def test_job_rejects_empty_scope_result(tmp_path):
    project = make_project(tmp_path)
    path = write_job(project, JobScope(stations=["missing"]))

    validation = ProjectJobs(project).validate(path)

    assert not validation.ok
    assert "does not select" in validation.errors[-1]


def test_batch_result_path_is_station_and_rate_specific(tmp_path):
    project = make_project(tmp_path)
    batch = ProjectJobs(project).plan_batches(
        JobDefinition(name="x", flow="f", parameters="p")
    )[0]

    assert (
        ProjectJobs(project)
        .batch_output_path(batch, "mt")
        .as_posix()
        .endswith("data/survey/a/results/mt/128_000000")
    )


def test_runner_runs_all_run_batches_before_station_rate_results(tmp_path):
    project = make_project(tmp_path)
    flow = FlowDefinition(
        id="batched",
        name="batched",
        stages=[
            FlowStage(
                stage_id="runs",
                scope="run",
                nodes=[FlowNode(id="mark", process=f"{__name__}.RunMarker")],
            ),
            FlowStage(
                stage_id="results",
                scope="station_rate",
                nodes=[FlowNode(id="write", process=f"{__name__}.BatchWriter")],
            ),
        ],
    )
    (project.project_path / "processing/flows/standard.yaml").write_text(
        model_to_yaml(flow)
    )
    (project.project_path / "processing/parameters/defaults.yaml").write_text(
        model_to_yaml(ParameterSet(name="defaults"))
    )
    path = project.project_path / "processing/jobs/example.yaml"
    path.write_text(
        model_to_yaml(
            JobDefinition(name="example", flow="standard", parameters="defaults")
        )
    )
    resolved = ProjectJobs(project).validate(path).resolved_job
    assert resolved is not None

    progress = []
    RUN_BATCHES.clear()
    assert (
        JobRunner(project, progress_callback=progress.append).run(resolved)
        == JobState.completed
    )

    assert RUN_BATCHES == ["run1", "run2", "run3"]
    assert (
        project.project_path / "data/survey/a/results/default/128_000000/result.txt"
    ).is_file()
    assert (
        project.project_path / "data/survey/b/results/default/4_000000/result.txt"
    ).is_file()
    revalidation = ProjectJobs(project).validate(path)
    assert revalidation.ok, revalidation.errors
    run_events = [event for event in progress if event.message == "Started: runs"]
    assert [
        (event.survey, event.station, event.run, event.sample_rate)
        for event in run_events
    ] == [
        ("survey", "a", "run1", None),
        ("survey", "a", "run2", None),
        ("survey", "b", "run3", None),
    ]
    station_rate_events = [
        event for event in progress if event.message == "Started: results"
    ]
    assert [
        (event.survey, event.station, event.run, event.sample_rate)
        for event in station_rate_events
    ] == [
        ("survey", "a", None, 128.0),
        ("survey", "b", None, 4.0),
    ]
    node_events = [
        event
        for event in progress
        if event.message
        in {"Started: mark", "Completed: mark", "Started: write", "Completed: write"}
    ]
    assert all(
        event.survey is None
        and event.station is None
        and event.run is None
        and event.sample_rate is None
        for event in node_events
    )


def test_stage_scope_runs_only_the_requested_stage(tmp_path):
    project = make_project(tmp_path)
    flow = FlowDefinition(
        id="batched",
        name="batched",
        stages=[
            FlowStage(
                stage_id="runs",
                scope="run",
                nodes=[FlowNode(id="mark", process=f"{__name__}.RunMarker")],
            ),
            FlowStage(
                stage_id="results",
                scope="station_rate",
                nodes=[FlowNode(id="write", process=f"{__name__}.BatchWriter")],
            ),
        ],
    )
    (project.project_path / "processing/flows/standard.yaml").write_text(
        model_to_yaml(flow)
    )
    (project.project_path / "processing/parameters/defaults.yaml").write_text(
        model_to_yaml(ParameterSet(name="defaults"))
    )
    path = project.project_path / "processing/jobs/example.yaml"
    path.write_text(
        model_to_yaml(
            JobDefinition(
                name="example",
                flow="standard",
                parameters="defaults",
                scope=JobScope(stages=["runs"]),
            )
        )
    )

    validation = ProjectJobs(project).validate(path)
    assert validation.ok, validation.errors
    assert [stage.stage_id for stage in validation.resolved_job.stages] == ["runs"]

    RUN_BATCHES.clear()
    assert JobRunner(project).run(validation.resolved_job) == JobState.completed
    assert RUN_BATCHES == ["run1", "run2", "run3"]
    assert not (project.project_path / "data/survey/a/results/result").exists()


def test_stage_scope_rejects_unknown_stage(tmp_path):
    project = make_project(tmp_path)
    path = write_job(project, JobScope(stages=["missing"]))

    validation = ProjectJobs(project).validate(path)

    assert not validation.ok
    assert validation.errors == ["Unknown flow stage(s): missing"]
