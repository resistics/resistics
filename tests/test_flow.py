import pytest

from resistics.flow import (
    FlowDefinition,
    FlowExecutor,
    FlowNode,
    FlowValidator,
    ParameterSet,
    RunConfiguration,
    builtin_step_registry,
    default_parameter_set,
    model_from_yaml,
    model_to_yaml,
    standard_mt_flow,
    topological_order,
)


def get_run_config(flow=None, params=None):
    flow = flow or standard_mt_flow()
    return RunConfiguration(
        name="test-run",
        flow=flow,
        parameters=params or default_parameter_set(flow),
        runtime={
            "project_path": "/tmp/project",
            "survey": "survey",
            "station": "station",
            "run": "run",
        },
        output_label="test",
    )


def test_standard_mt_flow_validates():
    registry = builtin_step_registry()
    result = FlowValidator(registry).validate(get_run_config())
    assert result.ok
    assert result.errors == []


def test_topological_order():
    order = [node.id for node in topological_order(standard_mt_flow())]
    assert order == [
        "read",
        "time_processors",
        "decimate",
        "window",
        "fft",
        "evals",
        "calibrate",
        "gather",
        "solve_tf",
        "write_results",
    ]


def test_unknown_step_is_invalid():
    flow = FlowDefinition(
        name="bad",
        nodes=[
            FlowNode(id="read", type="mth5_read"),
            FlowNode(id="bad", type="not_real", inputs=["read"]),
        ],
    )
    result = FlowValidator(builtin_step_registry()).validate(
        get_run_config(flow=flow, params=ParameterSet(name="test"))
    )
    assert not result.ok
    assert "Unknown step type: not_real" in result.errors


def test_cycle_is_invalid():
    flow = FlowDefinition(
        name="cycle",
        nodes=[
            FlowNode(id="a", type="mth5_read", inputs=["b"]),
            FlowNode(id="b", type="time_processors", inputs=["a"]),
        ],
    )
    result = FlowValidator(builtin_step_registry()).validate(get_run_config(flow=flow))
    assert not result.ok
    assert "Flow contains a cycle" in result.errors


def test_missing_runtime_is_invalid():
    run_config = get_run_config()
    run_config.runtime.pop("station")
    result = FlowValidator(builtin_step_registry()).validate(run_config)
    assert not result.ok
    assert "Runtime value 'station' is required by node 'read'" in result.errors


def test_parameter_validation():
    flow = standard_mt_flow()
    params = default_parameter_set(flow)
    params.values["solve_tf"]["solver"] = "not-a-solver"
    result = FlowValidator(builtin_step_registry()).validate(get_run_config(flow, params))
    assert not result.ok
    assert "Node 'solve_tf': Parameter 'solver' must be one of" in result.errors[0]


def test_executor_emits_progress_events():
    events = []
    results = FlowExecutor(
        builtin_step_registry(),
        progress_callback=events.append,
    ).run(get_run_config())

    assert list(results) == [node.id for node in topological_order(standard_mt_flow())]
    assert events[0]["event"] == "started"
    assert events[0]["node_id"] == "read"
    assert events[-1]["event"] == "completed"
    assert events[-1]["node_id"] == "write_results"


def test_executor_rejects_invalid_run_config():
    flow = standard_mt_flow()
    params = ParameterSet(name="bad", values={"decimate": {"n_levels": "nope"}})
    with pytest.raises(ValueError, match="Node 'decimate'"):
        FlowExecutor(builtin_step_registry()).run(get_run_config(flow, params))


def test_flow_yaml_roundtrip():
    flow = standard_mt_flow()
    loaded = model_from_yaml(FlowDefinition, model_to_yaml(flow))
    assert loaded == flow
