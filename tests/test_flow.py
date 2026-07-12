"""Tests for direct-process, staged flow definitions."""

import pytest

from resistics.common import ResisticsProcess
from resistics.flow import (
    FlowDefinition,
    FlowExecutor,
    FlowNode,
    FlowStage,
    FlowValidator,
    ParameterSet,
    ProcessCatalog,
    ProcessingJob,
    default_parameter_set,
    evals_to_tf_flow,
    model_from_yaml,
    model_to_yaml,
    remote_reference_mt_flow,
    resolve_process_class,
    single_site_mt_target_flow,
    standard_mt_flow,
    time_to_evals_flow,
    topological_order,
)

RUNTIME = {
    "project_path",
    "project",
    "reference_time",
    "run_batch",
    "station_rate_batch",
    "staging_output_path",
    "criteria",
}


class Source(ResisticsProcess):
    output_type = "number"

    def run(self):
        return 2


class Double(ResisticsProcess):
    input_types = {"value": "number"}
    output_type = "number"

    def run(self, value):
        return value * 2


def get_processing_job(flow=None, params=None):
    return ProcessingJob(
        name="test-job",
        flow=flow or standard_mt_flow(),
        parameters=params or default_parameter_set(),
        runtime={"project_path": "/tmp/project"},
        output_label="test",
    )


@pytest.mark.parametrize(
    "flow",
    [
        standard_mt_flow(),
        single_site_mt_target_flow(),
        remote_reference_mt_flow(),
        time_to_evals_flow(),
        evals_to_tf_flow(),
    ],
)
def test_default_flows_validate_with_shared_defaults(flow):
    result = FlowValidator(RUNTIME).validate(get_processing_job(flow))
    assert result.ok, result.errors


def test_standard_flow_has_durable_run_and_station_rate_stages():
    stages = standard_mt_flow().flow_stages()

    assert [(stage.stage_id, stage.scope) for stage in stages] == [
        ("time_to_evals", "run"),
        ("evals_to_tf", "station_rate"),
    ]
    assert stages[0].nodes[-1].process == "resistics.spectra.EvaluationFrequencyWriter"
    assert stages[1].nodes[0].configuration_source == "criteria"


def test_flow_serialization_has_no_ui_or_process_parameters():
    yaml_text = model_to_yaml(standard_mt_flow())

    assert "position:" not in yaml_text
    assert "parameters:" not in yaml_text
    assert "process: resistics." in yaml_text
    assert model_from_yaml(FlowDefinition, yaml_text) == standard_mt_flow()


def test_flow_requires_explicit_stages():
    with pytest.raises(ValueError):
        model_from_yaml(
            FlowDefinition,
            """
id: old_shape
name: Old shape
nodes: []
""",
        )


def test_parameter_defaults_are_discovered_not_flow_aligned():
    params = default_parameter_set()

    assert "resistics.decimate.DecimationSetup" in params.processes
    assert "resistics.regression.SolverOLS" in params.processes
    assert "resistics.gather.EvaluationFrequencyGather" not in params.processes
    assert params.processes["resistics.window.WindowerTarget"]["target"] == 500


def test_catalog_discovers_a_project_plugin_without_execution_registry(tmp_path):
    plugin_path = tmp_path / "plugins"
    plugin_path.mkdir()
    (plugin_path / "example.py").write_text(
        "from resistics.common import ResisticsProcess\n"
        "class PassThrough(ResisticsProcess):\n"
        "    input_types = {'time_data': 'time_data'}\n"
        "    output_type = 'time_data'\n"
        "    include_in_default_parameters = True\n"
        "    def run(self, time_data): return time_data\n"
    )

    catalog = ProcessCatalog(tmp_path).discover()
    descriptor = next(
        item for item in catalog if item.path == "plugins.example.PassThrough"
    )

    assert descriptor.input_types == {"time_data": "time_data"}
    assert resolve_process_class(descriptor.path).output_type == "time_data"
    assert "plugins.example.PassThrough" in default_parameter_set(tmp_path).processes


def test_unqualified_process_is_rejected():
    flow = FlowDefinition(
        id="bad",
        name="bad",
        stages=[
            FlowStage(
                stage_id="bad",
                scope="run",
                nodes=[FlowNode(id="bad", process="Windower")],
            )
        ],
    )
    result = FlowValidator(RUNTIME).validate(get_processing_job(flow))

    assert not result.ok
    assert "qualified class path" in result.errors[0]


def test_invalid_real_process_parameter_is_rejected():
    params = default_parameter_set()
    params.processes["resistics.decimate.DecimationSetup"]["n_levels"] = "bad"

    result = FlowValidator(RUNTIME).validate(get_processing_job(params=params))

    assert not result.ok
    assert "resistics.decimate.DecimationSetup" in result.errors[0]


def test_executor_runs_direct_process_class():
    flow = FlowDefinition(
        id="direct",
        name="direct",
        stages=[
            FlowStage(
                stage_id="run",
                scope="run",
                nodes=[
                    FlowNode(id="source", process=f"{__name__}.Source"),
                    FlowNode(
                        id="double",
                        process=f"{__name__}.Double",
                        inputs={"value": "source"},
                    ),
                ],
            )
        ],
    )
    events = []
    result = FlowExecutor(progress_callback=events.append).run(
        get_processing_job(flow, ParameterSet(name="empty"))
    )

    assert result["run"]["double"] == 4
    assert events[0]["process"] == f"{__name__}.Source"
    assert [node.id for node in topological_order(flow.flow_stages()[0])] == [
        "source",
        "double",
    ]
