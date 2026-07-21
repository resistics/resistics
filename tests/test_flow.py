"""Tests for direct-process, staged flow definitions."""

import pytest

import resistics.flow as flow_module
from resistics.common import (
    CancellationCallback,
    ProcessingProgressCallback,
    ProcessingProgressEvent,
    ProcessingProgressState,
    ResisticsProcess,
)
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
    mask_calculation_flow,
    mask_calculation_parameter_set,
    model_from_yaml,
    model_to_yaml,
    remote_reference_mt_flow,
    resolve_process_class,
    single_site_mt_flow,
    single_site_mt_target_flow,
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


def test_unused_legacy_flow_builders_are_removed():
    for name in (
        "evals_to_tf_flow",
        "mask_calculation_example_flow",
        "mask_calculation_example_parameter_set",
        "time_to_evals_flow",
    ):
        assert not hasattr(flow_module, name)


class Source(ResisticsProcess):
    output_type = "number"

    def run(self):
        return 2


class Double(ResisticsProcess):
    input_types = {"value": "number"}
    output_type = "number"

    def run(self, value):
        return value * 2


class RuntimeOutputLabel(ResisticsProcess):
    output_type = "label"

    def execute(self, inputs, context):
        del inputs
        return context["output_label"]


class ProgressSource(ResisticsProcess):
    output_type = "number"

    def run(
        self,
        progress_callback: ProcessingProgressCallback | None = None,
        cancellation_callback: CancellationCallback | None = None,
    ):
        assert progress_callback is not None
        assert cancellation_callback is not None
        assert not cancellation_callback()
        progress_callback(
            ProcessingProgressEvent(
                state=ProcessingProgressState.advanced,
                task="read_samples",
                current=1,
                total=1,
                message="Read samples",
            )
        )
        return 2


def get_processing_job(flow=None, params=None):
    return ProcessingJob(
        name="test-job",
        flow=flow or single_site_mt_flow(),
        parameters=params or default_parameter_set(),
        runtime={"project_path": "/tmp/project"},
        output_label="test",
    )


@pytest.mark.parametrize(
    "flow",
    [
        single_site_mt_flow(),
        single_site_mt_target_flow(),
        remote_reference_mt_flow(),
    ],
)
def test_default_flows_validate_with_shared_defaults(flow):
    result = FlowValidator(RUNTIME).validate(get_processing_job(flow))
    assert result.ok, result.errors


def test_executor_makes_the_job_output_label_authoritative():
    flow = FlowDefinition(
        id="label",
        name="label",
        stages=[
            FlowStage(
                stage_id="label",
                scope="run",
                nodes=[FlowNode(id="label", process=f"{__name__}.RuntimeOutputLabel")],
            )
        ],
    )
    job = ProcessingJob(
        name="label",
        flow=flow,
        parameters=ParameterSet(name="label"),
        output_label="field",
    )

    result = FlowExecutor().run_stage(
        job, flow.stages[0], {"output_label": "incorrect"}
    )

    assert result["label"] == "field"


def test_standard_flow_has_durable_run_and_station_rate_stages():
    stages = single_site_mt_flow().flow_stages()

    assert [(stage.stage_id, stage.scope) for stage in stages] == [
        ("time_to_evals", "run"),
        ("evals_to_tf", "station_rate"),
    ]
    assert stages[0].nodes[-1].process == "resistics.spectra.EvaluationFrequencyWriter"
    assert stages[1].nodes[0].configuration_source == "criteria"
    assert [node.process for node in stages[1].nodes[:3]] == [
        "resistics.gather.GatherCriteria",
        "resistics.regression.ImpedanceTensorSetup",
        "resistics.gather.Gather",
    ]


def test_default_mask_flow_is_run_scoped_and_validates_with_its_parameters():
    flow = mask_calculation_flow()
    processes = {node.process for node in flow.stages[0].nodes}

    assert flow.stages[0].scope == "run"
    assert "resistics.mask.TimeMask" in processes
    assert "resistics.mask.AbsoluteAmplitudeMask" in processes
    result = FlowValidator(RUNTIME).validate(
        get_processing_job(flow=flow, params=mask_calculation_parameter_set())
    )
    assert result.ok, result.errors


def test_mask_process_name_is_not_user_configurable():
    flow = mask_calculation_flow()
    params = mask_calculation_parameter_set()
    params.processes["resistics.mask.AbsoluteAmplitudeMask"]["name"] = "custom"

    result = FlowValidator(RUNTIME).validate(
        get_processing_job(flow=flow, params=params)
    )

    assert not result.ok
    assert "name" in result.errors[0]
    assert "Extra inputs are not permitted" in result.errors[0]


def test_flow_serialization_has_no_ui_or_process_parameters():
    yaml_text = model_to_yaml(single_site_mt_flow())

    assert "position:" not in yaml_text
    assert "parameters:" not in yaml_text
    assert "process: resistics." in yaml_text
    assert model_from_yaml(FlowDefinition, yaml_text) == single_site_mt_flow()


def test_flow_requires_explicit_stages():
    with pytest.raises(ValueError, match="Field required"):
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
    assert "resistics.mask.TimeMask" in params.processes
    assert "resistics.mask.AbsoluteAmplitudeMask" in params.processes
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
    assert events[0].process == f"{__name__}.Source"
    assert [node.id for node in topological_order(flow.flow_stages()[0])] == [
        "source",
        "double",
    ]


def test_executor_propagates_and_enriches_process_progress():
    flow = FlowDefinition(
        id="progress",
        name="progress",
        stages=[
            FlowStage(
                stage_id="run",
                scope="run",
                nodes=[FlowNode(id="source", process=f"{__name__}.ProgressSource")],
            )
        ],
    )
    events: list[ProcessingProgressEvent] = []

    FlowExecutor(
        progress_callback=events.append,
        cancellation_callback=lambda: False,
    ).run(get_processing_job(flow, ParameterSet(name="empty")))

    progress = next(event for event in events if event.task == "read_samples")
    assert progress.state == ProcessingProgressState.advanced
    assert (progress.stage_id, progress.node_id) == ("run", "source")
    assert progress.process == f"{__name__}.ProgressSource"
