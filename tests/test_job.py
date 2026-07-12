"""Tests for scoped, batch-oriented project jobs."""

from datetime import datetime, time, timezone
from pathlib import Path

import pandas as pd

from resistics.common import ResisticsProcess
from resistics.flow import (
    FlowDefinition,
    FlowNode,
    FlowStage,
    ParameterSet,
    model_from_yaml,
    model_to_yaml,
)
from resistics.gather import DailyTimeRange, GatherCriteria
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
    batches = ProjectJobs(project).plan_batches(validation.resolved_job.definition)
    assert [(item.station, item.sample_rate, item.run_paths) for item in batches] == [
        ("a", 128.0, ["survey/a/run1", "survey/a/run2"]),
        ("b", 4.0, ["survey/b/run3"]),
    ]


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
    (project.project_path / "processing/criteria/field.yaml").write_text(
        model_to_yaml(GatherCriteria(remote_references={"survey/a": "survey/b"}))
    )
    path = write_job(project, criteria="field")

    validation = ProjectJobs(project).validate(path)

    assert validation.ok, validation.errors
    assert validation.resolved_job.criteria.remote_references == {
        "survey/a": "survey/b"
    }


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


def test_criteria_evaluates_daily_time_ranges_in_utc():
    criteria = GatherCriteria(
        daily_include=[DailyTimeRange(from_time=time(9), to_time=time(17))]
    )

    assert criteria.includes(datetime(2020, 1, 1, 12, tzinfo=timezone.utc))
    assert not criteria.includes(datetime(2020, 1, 1, 18, tzinfo=timezone.utc))


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

    RUN_BATCHES.clear()
    assert JobRunner(project).run(resolved) == JobState.completed

    assert RUN_BATCHES == ["run1", "run2", "run3"]
    assert (
        project.project_path / "data/survey/a/results/result/128_000000/result.txt"
    ).is_file()
    assert (
        project.project_path / "data/survey/b/results/result/4_000000/result.txt"
    ).is_file()


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
