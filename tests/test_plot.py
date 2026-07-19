import numpy as np
import plotly.graph_objects as go
import pytest
from pydantic import Field

from resistics.common import ResisticsProcess
from resistics.flow import (
    FlowDefinition,
    FlowNode,
    FlowStage,
    ParameterSet,
    ProcessingJob,
)
from resistics.gather import GatherCriteria
from resistics.job import JobDefinition, ResolvedJob, StationRateBatch


class FlowPlotSource(ResisticsProcess):
    output_type = "number"


class FlowPlotDouble(ResisticsProcess):
    input_types = {"value": "number"}
    output_type = "number"
    factor: float = 1.0
    options: dict[str, int] = Field(default_factory=dict)


class FlowPlotMerge(ResisticsProcess):
    input_types = {"short": "number", "long": "number"}
    output_type = "number"


@pytest.mark.parametrize(
    "y, max_pts, x_expected, y_expected",
    [
        ([0, 1, 3, 4, 2, 3, 4, 3, 4, 5, 5, 5], 5, [0, 3, 4, 9, 11], [0, 4, 2, 5, 5]),
    ],
)
def test_lttb_downsample(
    y: list, max_pts: int, x_expected: list, y_expected: list
) -> None:
    """Test lttb downsampling"""
    from resistics.plot import lttb_downsample

    x = np.arange(len(y))
    y = np.array(y)
    nx, ny = lttb_downsample(x, y, max_pts=max_pts)
    np.testing.assert_array_equal(nx, x_expected)
    np.testing.assert_array_equal(ny, y_expected)


def test_lttb_downsample_preserves_dtype_and_large_indices() -> None:
    """LTTB selection must not lose precision by casting the source arrays."""
    from resistics.plot import lttb_downsample

    x = np.arange(100, dtype=np.int64) + 2**30
    y = np.sin(np.arange(100) / 5)

    nx, ny = lttb_downsample(x, y, max_pts=20)

    assert nx.dtype == x.dtype
    assert ny.dtype == y.dtype
    assert nx.size == 20
    assert nx[0] == x[0]
    assert nx[-1] == x[-1]
    assert np.all(np.diff(nx) > 0)


def test_lttb_downsample_accepts_non_contiguous_arrays() -> None:
    """LTTB selection accepts strided views used by plotting callers."""
    from resistics.plot import lttb_downsample

    x = np.arange(200, dtype=np.int64)[::2]
    y = np.sin(np.arange(200) / 5)[::2]

    nx, ny = lttb_downsample(x, y, max_pts=20)

    assert nx.size == ny.size == 20
    assert nx[0] == x[0]
    assert nx[-1] == x[-1]


def test_lttb_downsample_preserves_nan_gaps() -> None:
    """Downsampling retains a NaN marker so Plotly does not bridge data gaps."""
    from resistics.plot import lttb_downsample

    x = np.arange(100, dtype=np.int64)
    y = np.sin(x / 5)
    y[40:50] = np.nan

    nx, ny = lttb_downsample(x, y, max_pts=20)

    assert nx.size == ny.size == 20
    assert np.isnan(ny).any()
    assert nx[0] == x[0]
    assert nx[-1] == x[-1]


@pytest.mark.parametrize(
    "y, max_pts, x_expected, y_expected",
    [
        (
            [0, 1, 3, 4, 2, 3, 4, 3, 4, 5, 5, 5],
            5,
            [0, 3, 4, 9, 11],
            [0, 4, 2, 5, 5],
        ),
        (
            [0, 1, 3, 4, 2, 3, 4, 3, 4, 5, 5, 5],
            None,
            np.arange(12),
            [0, 1, 3, 4, 2, 3, 4, 3, 4, 5, 5, 5],
        ),
    ],
)
def test_apply_lttb(y: list, max_pts: int, x_expected: list, y_expected: list) -> None:
    """Testing the helper function"""
    from resistics.plot import apply_lttb

    x_new, y_new = apply_lttb(np.array(y), max_pts)
    np.testing.assert_array_equal(x_expected, x_new)
    np.testing.assert_array_equal(y_expected, y_new)


def test_plot_flow_builds_an_interactive_ranked_figure():
    """Flow plots share ranks and use staged card and edge legend entries."""
    from resistics.plot import plot_flow

    source_process = f"{__name__}.FlowPlotSource"
    double_process = f"{__name__}.FlowPlotDouble"
    flow = FlowDefinition(
        id="flow_plot",
        name="Flow Plot",
        stages=[
            FlowStage(
                stage_id="prepare",
                scope="run",
                nodes=[
                    FlowNode(id="read", process=source_process),
                    FlowNode(
                        id="double",
                        process=double_process,
                        inputs={"value": "read"},
                    ),
                ],
            ),
            FlowStage(
                stage_id="write",
                scope="station_rate",
                nodes=[
                    FlowNode(id="source", process=source_process),
                    FlowNode(
                        id="save", process=double_process, inputs={"value": "source"}
                    ),
                ],
            ),
        ],
    )

    figure = plot_flow(flow)

    assert isinstance(figure, go.Figure)
    assert figure.layout.title.text == "Flow: Flow Plot"
    stage_traces = [
        trace for trace in figure.data if trace.meta and trace.meta["kind"] == "nodes"
    ]
    assert len(stage_traces) == 2
    nodes = {
        node_id: (x, y, hover, trace.meta["stage"])
        for trace in stage_traces
        for node_id, x, y, hover in zip(
            trace.ids, trace.x, trace.y, trace.customdata, strict=True
        )
    }
    assert nodes["prepare:read"][1] > nodes["prepare:double"][1]
    assert nodes["prepare:read"][3] != nodes["write:source"][3]
    assert nodes["prepare:read"][1] == nodes["write:source"][1]
    assert source_process in nodes["prepare:read"][2]
    assert "<b>Stage</b><br>prepare · run" in nodes["prepare:read"][2]

    edge = next(
        trace
        for trace in figure.data
        if trace.meta and trace.meta.get("source") == "prepare:read"
    )
    assert edge.meta["target"] == "prepare:double"
    assert "Expected type: number" in edge.text[0]
    assert edge.line.color != "#5f6f7b"
    assert figure.layout.legend.title.text == "Stages"
    legend_traces = [
        trace for trace in figure.data if trace.meta and trace.meta["kind"] == "legend"
    ]
    assert len(legend_traces) == 2
    assert all(trace.mode == "lines+markers" for trace in legend_traces)
    assert edge.line.color == legend_traces[0].line.color
    assert figure.layout.paper_bgcolor == "#FFFFFF"
    assert figure.layout.plot_bgcolor == "#F8FAFC"
    assert figure.layout.legend.orientation == "v"
    assert figure.layout.legend.groupclick == "togglegroup"
    assert figure.layout.legend.x == 0.01
    assert figure.layout.legend.y == 0.96
    assert figure.layout.title.x == 0.5
    assert figure.layout.title.y == 0.99
    for stage_index in range(2):
        stage_traces = [
            trace
            for trace in figure.data
            if trace.meta and trace.meta["stage"] == stage_index
        ]
        assert {trace.meta["kind"] for trace in stage_traces} == {
            "arrow",
            "cards",
            "edge",
            "edge_label",
            "labels",
            "legend",
            "nodes",
        }
        assert {trace.legendgroup for trace in stage_traces} == {f"stage-{stage_index}"}
    edges = [
        trace for trace in figure.data if trace.meta and trace.meta["kind"] == "edge"
    ]
    arrows = [
        trace for trace in figure.data if trace.meta and trace.meta["kind"] == "arrow"
    ]
    assert len(arrows) == len(edges)
    assert all(trace.marker.symbol == "arrow" for trace in arrows)
    edge_labels = [
        trace
        for trace in figure.data
        if trace.meta and trace.meta["kind"] == "edge_label"
    ]
    assert len(edge_labels) == len(edges)
    assert {trace.text[0] for trace in edge_labels} == {"number"}
    assert all(trace.mode == "text" for trace in edge_labels)
    assert all(trace.textfont.size == 13 for trace in edge_labels)
    assert all(
        figure.data.index(trace)
        < min(
            figure.data.index(label)
            for label in figure.data
            if label.meta and label.meta["kind"] == "labels"
        )
        for trace in arrows
    )
    assert all(
        figure.data.index(trace)
        < min(
            figure.data.index(card)
            for card in figure.data
            if card.meta and card.meta["kind"] == "cards"
        )
        for trace in edge_labels
    )
    assert not figure.layout.annotations


def test_plot_flow_routes_long_edges_around_intermediate_nodes():
    """Long dependency edges use solver dummy vertices rather than node centres."""
    from resistics.plot import plot_flow

    source_process = f"{__name__}.FlowPlotSource"
    double_process = f"{__name__}.FlowPlotDouble"
    merge_process = f"{__name__}.FlowPlotMerge"
    flow = FlowDefinition(
        id="routed",
        name="Routed Flow",
        stages=[
            FlowStage(
                stage_id="process",
                scope="run",
                nodes=[
                    FlowNode(id="source", process=source_process),
                    FlowNode(
                        id="middle",
                        process=double_process,
                        inputs={"value": "source"},
                    ),
                    FlowNode(id="isolated", process=source_process),
                    FlowNode(
                        id="target",
                        process=merge_process,
                        inputs={"short": "middle", "long": "source"},
                    ),
                ],
            )
        ],
    )

    figure = plot_flow(flow)
    node_trace = next(
        trace for trace in figure.data if trace.meta and trace.meta["kind"] == "nodes"
    )
    node_positions = {
        node_id: (x, y)
        for node_id, x, y in zip(
            node_trace.ids, node_trace.x, node_trace.y, strict=True
        )
    }
    long_edge = next(
        trace for trace in figure.data if trace.meta and trace.meta["port"] == "long"
    )

    assert len(long_edge.x) > 2
    assert (node_positions["process:middle"]) not in set(
        zip(long_edge.x[1:-1], long_edge.y[1:-1], strict=True)
    )
    assert node_positions["process:isolated"][1] == node_positions["process:source"][1]


def test_flow_arrowheads_stop_outside_card_boundaries():
    """Arrowheads remain visible immediately before vertical and diagonal cards."""
    from resistics.plot import (
        FLOW_ARROW_CLEARANCE,
        FLOW_CARD_HEIGHT,
        FLOW_CARD_WIDTH,
        _flow_arrow_position,
    )

    vertical = _flow_arrow_position([(0.0, 180.0), (0.0, 0.0)])
    diagonal = _flow_arrow_position([(-180.0, 180.0), (0.0, 0.0)])

    assert vertical == (0.0, FLOW_CARD_HEIGHT / 2 + FLOW_ARROW_CLEARANCE)
    assert diagonal[1] > FLOW_CARD_HEIGHT / 2
    assert abs(diagonal[0]) < FLOW_CARD_WIDTH / 2


def test_flow_edge_labels_use_the_midpoint_of_routed_connectors():
    """Type labels stay centered even when a connector has a bend."""
    from resistics.plot import _flow_edge_label_position

    assert _flow_edge_label_position([(0.0, 0.0), (0.0, 3.0), (4.0, 3.0)]) == (
        0.5,
        3.0,
    )


def test_plot_flow_wraps_long_card_labels_without_losing_process_details():
    """Cards bound long labels while hover text retains the full process path."""
    from resistics.plot import plot_flow

    source_process = f"{__name__}.FlowPlotSource"
    long_process = f"{__name__}.FlowPlotDouble"
    flow = FlowDefinition(
        id="long_labels",
        name="Long labels",
        stages=[
            FlowStage(
                stage_id="process",
                scope="run",
                nodes=[
                    FlowNode(id="very_long_source_identifier", process=source_process),
                    FlowNode(
                        id="very_long_processing_identifier",
                        process=long_process,
                        inputs={"value": "very_long_source_identifier"},
                    ),
                ],
            )
        ],
    )

    figure = plot_flow(flow)
    labels = [
        text
        for trace in figure.data
        if trace.meta and trace.meta["kind"] == "labels"
        for text in trace.text
        if "identifier" in text
    ]
    node_trace = next(
        trace for trace in figure.data if trace.meta and trace.meta["kind"] == "nodes"
    )

    assert labels
    assert all(label.count("<br>") >= 2 for label in labels)
    assert long_process in node_trace.customdata[1]


def test_plot_flow_rejects_a_missing_upstream_node():
    """The plotter refuses a graph that would omit a declared dependency."""
    from resistics.plot import plot_flow

    flow = FlowDefinition(
        id="broken",
        name="Broken",
        stages=[
            FlowStage(
                stage_id="broken",
                scope="run",
                nodes=[
                    FlowNode(
                        id="double",
                        process=f"{__name__}.FlowPlotDouble",
                        inputs={"value": "missing"},
                    )
                ],
            )
        ],
    )

    with pytest.raises(ValueError, match="missing input 'missing'"):
        plot_flow(flow)


def test_plot_job_shows_selected_stages_and_parameter_file_values(tmp_path):
    """Job plots make their scoped execution configuration visible on cards."""
    from resistics.plot import plot_job

    source_process = f"{__name__}.FlowPlotSource"
    double_process = f"{__name__}.FlowPlotDouble"
    selected_stage = FlowStage(
        stage_id="prepare",
        scope="run",
        nodes=[
            FlowNode(id="read", process=source_process),
            FlowNode(
                id="configured",
                process=double_process,
                inputs={"value": "read"},
            ),
            FlowNode(
                id="criteria_step",
                process=double_process,
                inputs={"value": "configured"},
                configuration_source="criteria",
            ),
        ],
    )
    excluded_stage = FlowStage(
        stage_id="excluded",
        scope="station_rate",
        nodes=[FlowNode(id="write", process=source_process)],
    )
    flow = FlowDefinition(
        id="job_plot",
        name="Job flow",
        stages=[selected_stage, excluded_stage],
    )
    parameters = ParameterSet(
        name="Field settings",
        processes={double_process: {"factor": 2.5, "options": {"a": 1, "b": 2}}},
    )
    resolved = ResolvedJob(
        path=tmp_path / "jobs/field.yaml",
        definition=JobDefinition(
            name="field",
            flow="job_flow.yaml",
            parameters="field.yaml",
            criteria="criteria.yaml",
            output_label="field_result",
        ),
        processing_job=ProcessingJob(name="field", flow=flow, parameters=parameters),
        flow_path=tmp_path / "flows/job_flow.yaml",
        parameters_path=tmp_path / "parameters/field.yaml",
        criteria_path=tmp_path / "criteria/criteria.yaml",
        criteria=GatherCriteria(),
        stages=[selected_stage],
        batches=[
            StationRateBatch(
                survey="survey",
                station="field",
                sample_rate=128.0,
                run_paths=["survey/field/run1", "survey/field/run2"],
            )
        ],
        output_path=tmp_path / "results",
    )

    figure = plot_job(resolved)

    assert isinstance(figure, go.Figure)
    assert "Job: field" in figure.layout.title.text
    assert "Parameters: Field settings (field.yaml)" in figure.layout.title.text
    assert "1 target batch(es), 2 target run(s)" in figure.layout.title.text
    node_trace = next(
        trace for trace in figure.data if trace.meta and trace.meta["kind"] == "nodes"
    )
    assert set(node_trace.ids) == {
        "prepare:read",
        "prepare:configured",
        "prepare:criteria_step",
    }
    label_trace = next(
        trace for trace in figure.data if trace.meta and trace.meta["kind"] == "labels"
    )
    labels = "\n".join(label_trace.text)
    assert "No parameters" in labels
    assert "factor = 2.5" in labels
    assert "options = {2 keys}" in labels
    assert "Criteria: criteria.yaml" in labels
    configured_hover = node_trace.customdata[
        list(node_trace.ids).index("prepare:configured")
    ]
    criteria_hover = node_trace.customdata[
        list(node_trace.ids).index("prepare:criteria_step")
    ]
    assert '"factor":&nbsp;2.5' in configured_hover
    assert "Criteria configuration (criteria.yaml)" in criteria_hover
    assert figure.layout.legend.groupclick == "togglegroup"
    arrow_traces = [
        trace for trace in figure.data if trace.meta and trace.meta["kind"] == "arrow"
    ]
    assert len(arrow_traces) == 2
    assert all(trace.marker.symbol == "arrow" for trace in arrow_traces)
    assert {trace.legendgroup for trace in arrow_traces} == {"stage-0"}
    edge_labels = [
        trace
        for trace in figure.data
        if trace.meta and trace.meta["kind"] == "edge_label"
    ]
    assert len(edge_labels) == 2
    assert {trace.text[0] for trace in edge_labels} == {"number"}
    assert {trace.legendgroup for trace in edge_labels} == {"stage-0"}
    assert all(
        figure.data.index(trace) < figure.data.index(label_trace)
        for trace in arrow_traces
    )
    assert not figure.layout.annotations
    assert figure.layout.margin.t == 160
